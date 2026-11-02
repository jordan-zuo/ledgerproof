from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class AlreadyDecided(Exception):
    """Raised when a ticket is decided a second time."""

    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"approval ticket already decided: {ticket_id}")
        self.ticket_id = ticket_id


class DuplicateLedgerEntry(Exception):
    """Raised when a bill is posted again, under any entry label."""

    def __init__(self, source_doc: str, entry_id: str) -> None:
        super().__init__(
            f"already posted: {source_doc} is already in the ledger"
            f" (attempted as {entry_id})"
        )
        self.source_doc = source_doc
        self.entry_id = entry_id


class UnknownLedgerEntry(Exception):
    """Raised when a reversal names a row that does not exist."""

    def __init__(self, ledger_id: int) -> None:
        super().__init__(f"unknown ledger entry: {ledger_id}")
        self.ledger_id = ledger_id


class AlreadyReversed(Exception):
    """Raised when a posting already has its one reversal row."""

    def __init__(self, ledger_id: int) -> None:
        super().__init__(f"ledger entry already reversed: {ledger_id}")
        self.ledger_id = ledger_id


class NotReversible(Exception):
    """Raised when a reversal row is itself offered for reversal."""

    def __init__(self, ledger_id: int) -> None:
        super().__init__(f"ledger entry is not a posting: {ledger_id}")
        self.ledger_id = ledger_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    override = os.environ.get("APP_DB", "")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "app.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


_INITIALIZED: set[str] = set()


def init_db(path: Path | None = None) -> Path:
    target = path or db_path()
    key = str(target.resolve())
    if key in _INITIALIZED:
        # Endpoints call this per request; schema work must not take a write
        # lock while a background run is writing.
        return target
    conn = connect(target)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                # Another connection is mid-write; WAL is already on or will
                # be set by the first caller that can take the lock.
                pass
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                request_id TEXT,
                status TEXT NOT NULL,
                step_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                body TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                source_doc TEXT NOT NULL,
                bill_key TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'posting',
                reverses_id INTEGER,
                reason TEXT NOT NULL DEFAULT '',
                approved_by TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approvals (
                ticket_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                display_no INTEGER NOT NULL DEFAULT 0,
                action TEXT NOT NULL,
                detail TEXT NOT NULL,
                meta TEXT NOT NULL,
                evidence TEXT NOT NULL DEFAULT '{}',
                decided INTEGER NOT NULL DEFAULT 0,
                approved INTEGER NOT NULL DEFAULT 0,
                decided_by TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                decided_at TEXT
            );
            """
        )
        _migrate_runs(conn)
        _migrate_approvals(conn)
        _migrate_events(conn)
        _migrate_ledger(conn)
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS ledger_append_only_update
            BEFORE UPDATE ON ledger
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS ledger_append_only_delete
            BEFORE DELETE ON ledger
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_append_only_update
            BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT, 'run events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_append_only_delete
            BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT, 'run events are append-only'); END;
            """
        )
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_run_entry"
                " ON ledger(run_id, entry_id)"
            )
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            # Legacy databases may already hold a duplicate. The insert
            # guard below still enforces one entry per run.
            pass
        try:
            # One reversal per posting, enforced by the database as well as
            # the insert guard.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_one_reversal"
                " ON ledger(reverses_id) WHERE reverses_id IS NOT NULL"
            )
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            pass
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_bill_key"
                " ON ledger(bill_key)"
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_request_id"
                " ON runs(request_id) WHERE request_id IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()
    _INITIALIZED.add(key)
    return target


def _migrate_runs(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    if "request_id" not in existing:
        conn.execute("ALTER TABLE runs ADD COLUMN request_id TEXT")


def _migrate_approvals(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(approvals)").fetchall()
    }
    additions = {
        "display_no": "INTEGER NOT NULL DEFAULT 0",
        "evidence": "TEXT NOT NULL DEFAULT '{}'",
        "decided_by": "TEXT NOT NULL DEFAULT ''",
        "note": "TEXT NOT NULL DEFAULT ''",
    }
    for column, ddl in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE approvals ADD COLUMN {column} {ddl}")

    # Legacy rows get a stable human number in creation order.
    unnumbered = conn.execute(
        "SELECT ticket_id FROM approvals WHERE display_no = 0"
        " ORDER BY created_at, rowid"
    ).fetchall()
    if unnumbered:
        start = int(
            conn.execute(
                "SELECT COALESCE(MAX(display_no), 0) AS m FROM approvals"
            ).fetchone()["m"]
        )
        for offset, row in enumerate(unnumbered, 1):
            conn.execute(
                "UPDATE approvals SET display_no = ? WHERE ticket_id = ?",
                (start + offset, row["ticket_id"]),
            )


def _migrate_events(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(events)").fetchall()
    }
    if "data" not in existing:
        conn.execute("ALTER TABLE events ADD COLUMN data TEXT NOT NULL DEFAULT '{}'")


def _migrate_ledger(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(ledger)").fetchall()
    }
    additions = {
        "bill_key": "TEXT NOT NULL DEFAULT ''",
        "kind": "TEXT NOT NULL DEFAULT 'posting'",
        "reverses_id": "INTEGER",
        "reason": "TEXT NOT NULL DEFAULT ''",
        "approved_by": "TEXT NOT NULL DEFAULT ''",
    }
    for column, ddl in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE ledger ADD COLUMN {column} {ddl}")


def create_run(
    task: str, request_id: str | None = None, path: Path | None = None
) -> str:
    run_id = f"r-{uuid.uuid4().hex[:12]}"
    conn = connect(path)
    try:
        now = _now()
        conn.execute(
            "INSERT INTO runs (run_id, task, request_id, status, step_count,"
            " created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, task, request_id, "running", 0, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def find_run_by_request(request_id: str, path: Path | None = None) -> dict | None:
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT run_id, task, status, step_count, created_at, updated_at"
            " FROM runs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_runs(limit: int = 20, path: Path | None = None) -> list[dict]:
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT run_id, task, status, step_count, created_at, updated_at"
            " FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def cancel_run(run_id: str, path: Path | None = None) -> bool:
    """Mark a live run cancelled. Returns False if it already settled."""
    conn = connect(path)
    try:
        cur = conn.execute(
            "UPDATE runs SET status = 'cancelled', updated_at = ?"
            " WHERE run_id = ? AND status IN ('pending', 'running', 'awaiting_approval')",
            (_now(), run_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def append_event(
    run_id: str,
    kind: str,
    body: str,
    data: dict | None = None,
    path: Path | None = None,
) -> dict:
    """Append one structured event. ``seq`` is per-run and monotonic."""
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) AS m FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        seq = int(row["m"]) + 1
        payload = json.dumps(data or {}, sort_keys=True)
        conn.execute(
            "INSERT INTO events (run_id, seq, kind, body, data, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, seq, kind, body, payload, _now()),
        )
        conn.commit()
        return {"run_id": run_id, "seq": seq, "kind": kind, "body": body, "data": data or {}}
    finally:
        conn.close()


def set_run_status(
    run_id: str, status: str, step_count: int, path: Path | None = None
) -> None:
    conn = connect(path)
    try:
        # Cancellation is sticky: a worker finishing a slow step cannot
        # overwrite a status a human already set.
        conn.execute(
            "UPDATE runs SET status = ?, step_count = ?, updated_at = ?"
            " WHERE run_id = ? AND (status != 'cancelled' OR ? = 'cancelled')",
            (status, step_count, _now(), run_id, status),
        )
        conn.commit()
    finally:
        conn.close()


def get_run(run_id: str, path: Path | None = None) -> dict | None:
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT run_id, task, status, step_count FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_events(
    run_id: str, after: int | None = None, path: Path | None = None
) -> list[dict]:
    """Replay lane. ``after`` returns only events newer than that seq."""
    conn = connect(path)
    try:
        if after is None:
            rows = conn.execute(
                "SELECT seq, kind, body, data FROM events"
                " WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT seq, kind, body, data FROM events"
                " WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, after),
            ).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            try:
                item["data"] = json.loads(item.get("data") or "{}")
            except (TypeError, ValueError):
                item["data"] = {}
            events.append(item)
        return events
    finally:
        conn.close()


def find_ledger_posting(
    source_doc: str,
    amount: float,
    currency: str,
    path: Path | None = None,
    bill_key: str = "",
) -> dict | None:
    """Return an outstanding posting for this bill, if any.

    Identity is the canonical bill key when one could be derived, falling
    back to the source document plus amount. The model-supplied ``entry_id``
    is never used: two runs can invent two labels for one bill. A posting
    that has been reversed no longer blocks a corrected re-post.
    """
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT l.entry_id, l.run_id, l.amount, l.currency, l.source_doc,"
            " l.bill_key, l.created_at"
            " FROM ledger l"
            " WHERE l.kind = 'posting'"
            "   AND NOT EXISTS ("
            "     SELECT 1 FROM ledger r WHERE r.kind = 'reversal'"
            "       AND r.reverses_id = l.id"
            "   )"
            "   AND ("
            "     (? <> '' AND l.bill_key = ?)"
            "     OR (l.source_doc = ? AND l.amount = ? AND l.currency = ?)"
            "   )"
            " ORDER BY l.id LIMIT 1",
            (
                bill_key,
                bill_key,
                source_doc,
                float(amount),
                currency,
            ),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def insert_ledger_entry(
    run_id: str,
    entry_id: str,
    amount: float,
    currency: str,
    source_doc: str,
    path: Path | None = None,
    *,
    bill_key: str = "",
) -> None:
    conn = connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO ledger (entry_id, run_id, amount, currency, source_doc,"
            " bill_key, kind, created_at)"
            " SELECT ?, ?, ?, ?, ?, ?, 'posting', ?"
            " WHERE NOT EXISTS ("
            "   SELECT 1 FROM ledger WHERE run_id = ? AND entry_id = ?"
            " )"
            " AND NOT EXISTS ("
            "   SELECT 1 FROM ledger l"
            "   WHERE l.kind = 'posting'"
            "     AND NOT EXISTS ("
            "       SELECT 1 FROM ledger r WHERE r.kind = 'reversal'"
            "         AND r.reverses_id = l.id"
            "     )"
            "     AND ("
            "       (? <> '' AND l.bill_key = ?)"
            "       OR (l.source_doc = ? AND l.amount = ? AND l.currency = ?)"
            "     )"
            " )",
            (
                entry_id,
                run_id,
                float(amount),
                currency,
                source_doc,
                bill_key,
                _now(),
                run_id,
                entry_id,
                bill_key,
                bill_key,
                source_doc,
                float(amount),
                currency,
            ),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise DuplicateLedgerEntry(source_doc, entry_id)
    finally:
        conn.close()


def reverse_ledger_entry(
    ledger_id: int,
    reason: str,
    approved_by: str,
    path: Path | None = None,
) -> dict:
    """Append a signed negative row. The original posting is never touched.

    The reversal carries the original run, document, currency, and canonical
    bill key, so it is auditable without a join on mutable state. Reversing
    twice is refused by the insert guard and by a unique index.
    """
    cleaned_reason = reason.strip()
    cleaned_by = approved_by.strip()
    if not cleaned_reason or not cleaned_by:
        raise ValueError("a reversal needs a reason and an approver")

    conn = connect(path)
    try:
        try:
            cur = conn.execute(
                "INSERT INTO ledger (entry_id, run_id, amount, currency,"
                " source_doc, bill_key, kind, reverses_id, reason, approved_by,"
                " created_at)"
                " SELECT ?, l.run_id, -l.amount, l.currency, l.source_doc,"
                " l.bill_key, 'reversal', l.id, ?, ?, ?"
                " FROM ledger l"
                " WHERE l.id = ? AND l.kind = 'posting'"
                "   AND NOT EXISTS ("
                "     SELECT 1 FROM ledger r WHERE r.kind = 'reversal'"
                "       AND r.reverses_id = l.id"
                "   )",
                (
                    f"reversal-{ledger_id}",
                    cleaned_reason,
                    cleaned_by,
                    _now(),
                    ledger_id,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise AlreadyReversed(ledger_id) from None
        if cur.rowcount == 0:
            row = conn.execute(
                "SELECT kind FROM ledger WHERE id = ?", (ledger_id,)
            ).fetchone()
            if row is None:
                raise UnknownLedgerEntry(ledger_id)
            if row["kind"] != "posting":
                raise NotReversible(ledger_id)
            raise AlreadyReversed(ledger_id)
        row = conn.execute(
            "SELECT id, entry_id, run_id, amount, currency, source_doc,"
            " bill_key, kind, reverses_id, reason, approved_by, created_at"
            " FROM ledger WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_ledger(path: Path | None = None) -> list[dict]:
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT l.id, l.entry_id, l.run_id, l.amount, l.currency,"
            " l.source_doc, l.bill_key, l.kind, l.reverses_id, l.reason,"
            " l.approved_by, l.created_at,"
            " EXISTS("
            "   SELECT 1 FROM ledger r WHERE r.kind = 'reversal'"
            "     AND r.reverses_id = l.id"
            " ) AS reversed"
            " FROM ledger l ORDER BY l.id"
        ).fetchall()
        entries = []
        for row in rows:
            item = dict(row)
            item["reversed"] = bool(item["reversed"])
            entries.append(item)
        return entries
    finally:
        conn.close()


def table_counts(path: Path | None = None) -> dict[str, int]:
    conn = connect(path)
    try:
        out = {}
        for table in ("runs", "events", "ledger", "approvals"):
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[table] = int(row["n"])
        return out
    finally:
        conn.close()


def archive_and_clear(path: Path | None = None) -> Path:
    """Snapshot the whole database, then start a fresh one.

    Rows are never deleted in place: the append-only triggers are only
    dropped inside this one maintenance transaction, after the archive is
    written. The old ledger survives as a file you can open and compare.
    """
    target = path or db_path()
    archive_dir = target.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_dir / f"{target.stem}-{stamp}.db"
    src = connect(target)
    dst = sqlite3.connect(str(archive))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    conn = connect(target)
    try:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS ledger_append_only_update;
            DROP TRIGGER IF EXISTS ledger_append_only_delete;
            DROP TRIGGER IF EXISTS events_append_only_update;
            DROP TRIGGER IF EXISTS events_append_only_delete;
            DELETE FROM ledger;
            DELETE FROM events;
            DELETE FROM approvals;
            DELETE FROM runs;
            CREATE TRIGGER ledger_append_only_update
            BEFORE UPDATE ON ledger
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            CREATE TRIGGER ledger_append_only_delete
            BEFORE DELETE ON ledger
            BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            CREATE TRIGGER events_append_only_update
            BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT, 'run events are append-only'); END;
            CREATE TRIGGER events_append_only_delete
            BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT, 'run events are append-only'); END;
            """
        )
        conn.commit()
    finally:
        conn.close()
    return archive


def submit_approval(
    ticket_id: str,
    run_id: str,
    action: str,
    detail: str,
    meta: dict,
    evidence: dict | None = None,
    path: Path | None = None,
) -> None:
    conn = connect(path)
    try:
        conn.execute(
            "INSERT INTO approvals (ticket_id, run_id, display_no, action,"
            " detail, meta, evidence, decided, approved, created_at)"
            " VALUES (?, ?, (SELECT COALESCE(MAX(display_no), 0) + 1"
            " FROM approvals), ?, ?, ?, ?, 0, 0, ?)",
            (
                ticket_id,
                run_id,
                action,
                detail,
                json.dumps(meta),
                json.dumps(evidence or {}),
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _row_to_approval(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["meta"] = json.loads(item["meta"])
    item["evidence"] = json.loads(item["evidence"])
    return item


def list_pending_approvals(path: Path | None = None) -> list[dict]:
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT ticket_id, run_id, display_no, action, detail, meta, evidence"
            " FROM approvals WHERE decided = 0 ORDER BY display_no"
        ).fetchall()
        return [_row_to_approval(r) for r in rows]
    finally:
        conn.close()


def pending_approvals_for_run(
    run_id: str, path: Path | None = None
) -> list[dict]:
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT ticket_id, run_id, display_no, action, detail, meta, evidence"
            " FROM approvals WHERE run_id = ? AND decided = 0 ORDER BY display_no",
            (run_id,),
        ).fetchall()
        return [_row_to_approval(r) for r in rows]
    finally:
        conn.close()


def get_approval(ticket_id: str, path: Path | None = None) -> dict | None:
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT ticket_id, run_id, display_no, action, detail, meta,"
            " evidence, decided, approved, decided_by, note"
            " FROM approvals WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
        return _row_to_approval(row) if row else None
    finally:
        conn.close()


def decide_approval(
    ticket_id: str,
    approved: bool,
    decided_by: str = "",
    note: str = "",
    path: Path | None = None,
) -> dict | None:
    """Decide once. A second decision raises instead of writing again."""
    conn = connect(path)
    try:
        cur = conn.execute(
            "UPDATE approvals SET decided = 1, approved = ?, decided_at = ?,"
            " decided_by = ?, note = ?"
            " WHERE ticket_id = ? AND decided = 0",
            (1 if approved else 0, _now(), decided_by, note, ticket_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM approvals WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if exists is None:
                return None
            raise AlreadyDecided(ticket_id)
        row = conn.execute(
            "SELECT ticket_id, run_id, display_no, action, detail, meta,"
            " evidence, decided, approved, decided_by, note"
            " FROM approvals WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
        item = _row_to_approval(row)
        item["approved"] = bool(approved)
        return item
    finally:
        conn.close()
