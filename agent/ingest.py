"""Inbox ingestion, identity at ingest, and duplicate detection.

One ingest run scans the inbox, extracts text, assigns one doc id per file
stem, and merges the result into the corpus. It also records each document's
canonical bill identity so two doc ids for one bill are visible before an
agent ever runs.

Collision rules:

* Two files with the same doc id and the same text are one source. The first
  file name owns the entry and the other is reported as a duplicate source.
* Two files with the same doc id and different text are a collision. The
  corpus keeps one text (the existing text if it came from one of the files,
  otherwise the first file name) and the ignored files are reported. Nothing
  is silently overwritten.
* Two doc ids with the same canonical bill identity are a duplicate identity.
  Both documents stay in the corpus and both are flagged in the catalog.
* Unsupported, empty, and no-text PDF files are reported as skipped.

Corpus writes are atomic. A lock file serializes ingest across processes, so
several uvicorn workers can watch the same inbox.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agent import store
from agent.bill_identity import BillIdentity, identify_bill

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})
DEFAULT_INTERVAL = 5.0


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def inbox_path() -> Path:
    override = os.environ.get("APP_INBOX", "").strip()
    return Path(override) if override else ROOT / "data" / "inbox"


def corpus_path() -> Path:
    override = os.environ.get("APP_CORPUS", "").strip()
    return Path(override) if override else ROOT / "data" / "corpus.json"


def watcher_enabled() -> bool:
    value = os.environ.get("APP_INBOX_WATCH")
    if value is None:
        return True
    return _enabled(value)


def ingest_interval() -> float:
    raw = os.environ.get("APP_INGEST_INTERVAL", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_INTERVAL
    except ValueError:
        value = DEFAULT_INTERVAL
    return max(0.2, value)


def doc_id_for(path: Path) -> str:
    """The file stem, lowercased, with runs of whitespace collapsed to '-'."""
    return "-".join(path.stem.strip().lower().split())


def _read_pdf_text(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "pypdf missing so PDF skipped"
    try:
        reader = PdfReader(str(path))
        parts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        return "", f"PDF read failed with {type(exc).__name__}"
    text = "\n".join(part.strip() for part in parts if part and part.strip())
    if not text.strip():
        return "", "no text layer found, scanned image likely"
    return text.strip(), ""


def extract_text(path: Path) -> tuple[str, str]:
    """Return ``(text, reason)``. A non-empty reason means skip the file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(path)
    try:
        text = path.read_text(encoding="utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return "", "not UTF-8 text"
    except OSError as exc:
        return "", f"read failed with {type(exc).__name__}"
    if not text:
        return "", "empty document"
    return text, ""


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_corpus(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"corpus must be a JSON object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def _write_corpus(path: Path, corpus: dict[str, str]) -> None:
    """Replace the corpus in one step so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(corpus, handle, indent=2)
        for attempt in range(10):
            try:
                os.replace(tmp_name, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@contextmanager
def _ingest_lock(corpus_file: Path) -> Iterator[None]:
    """One ingest at a time, across threads and processes."""
    lock_path = corpus_file.with_name(corpus_file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + 30.0
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _identity_fields(identity: BillIdentity) -> dict[str, str]:
    return {
        "bill_key": identity.key,
        "identity_basis": identity.basis,
        "account": identity.account,
        "period_start": identity.period_start,
        "period_end": identity.period_end,
        "issue_date": identity.issue_date,
    }


def _sync_documents(
    corpus: dict[str, str],
    ingested: dict[str, dict[str, Any]],
    now: str,
) -> None:
    """Keep the documents table aligned with the corpus."""
    existing = {row["doc_id"]: row for row in store.list_documents()}
    for doc_id, text in corpus.items():
        digest = _text_hash(text)
        old = existing.get(doc_id)
        info = ingested.get(doc_id)
        if info is None:
            if old and old["content_hash"] == digest:
                info = {
                    "source_name": old["source_name"],
                    "content_hash": digest,
                    "size_bytes": int(old["size_bytes"]),
                    "modified_ns": int(old["modified_ns"]),
                    "ingested_at": old["ingested_at"],
                }
            else:
                info = {
                    "source_name": "",
                    "content_hash": digest,
                    "size_bytes": 0,
                    "modified_ns": 0,
                    "ingested_at": now,
                }
        identity = identify_bill(doc_id, text)
        store.upsert_document(
            doc_id,
            source_name=str(info["source_name"]),
            content_hash=digest,
            size_bytes=int(info["size_bytes"]),
            modified_ns=int(info["modified_ns"]),
            ingested_at=str(info.get("ingested_at") or now),
            **_identity_fields(identity),
        )
    store.prune_documents(corpus.keys())


def _duplicate_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = str(item["identity"])
        if key.startswith("doc:"):
            continue
        groups.setdefault(key, []).append(item)
    out = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        out.append(
            {
                "bill_key": key,
                "identity_basis": members[0]["identity_basis"],
                "doc_ids": [str(item["doc_id"]) for item in members],
            }
        )
    return sorted(out, key=lambda group: group["bill_key"])


def document_catalog(
    corpus: dict[str, str], rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """One catalog row per corpus document, with duplicate identities marked."""
    stored = {row["doc_id"]: row for row in (rows or [])}
    items: list[dict[str, Any]] = []
    for doc_id in sorted(corpus):
        text = corpus[doc_id]
        digest = _text_hash(text)
        row = stored.get(doc_id) or {}
        if row and row.get("content_hash") == digest:
            identity = BillIdentity(
                key=str(row["bill_key"]),
                basis=str(row["identity_basis"]),
                account=str(row["account"]),
                period_start=str(row["period_start"]),
                period_end=str(row["period_end"]),
                issue_date=str(row["issue_date"]),
            )
        else:
            # Missing or stale metadata: trust the text over the row.
            identity = identify_bill(doc_id, text)
        items.append(
            {
                "doc_id": doc_id,
                "source_name": str(row.get("source_name", "")),
                "identity": identity.key,
                "identity_basis": identity.basis,
                "account": identity.account,
                "period_start": identity.period_start,
                "period_end": identity.period_end,
                "issue_date": identity.issue_date,
                "text_chars": len(text),
                "preview": " ".join(text.split())[:400],
                "ingested_at": str(row.get("ingested_at", "")),
                "duplicate": False,
            }
        )
    duplicates = _duplicate_groups(items)
    duplicate_keys = {group["bill_key"] for group in duplicates}
    for item in items:
        item["duplicate"] = item["identity"] in duplicate_keys
    return {
        "documents": items,
        "duplicates": duplicates,
        "counts": {
            "documents": len(items),
            "duplicate_identities": len(duplicates),
        },
    }


def ingest_once(
    inbox: Path | None = None, corpus_file: Path | None = None
) -> dict[str, Any]:
    """Scan the inbox and merge it into the corpus.

    Existing corpus documents with no inbox file are left alone. A file whose
    text changed updates its doc id in place.
    """
    root = (inbox or inbox_path()).resolve()
    target = (corpus_file or corpus_path()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store.init_db()
    now = datetime.now(timezone.utc).isoformat()

    with _ingest_lock(target):
        corpus = _load_corpus(target)
        before = dict(corpus)
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        skipped: list[dict[str, str]] = []
        collisions: list[dict[str, Any]] = []
        duplicate_sources: list[dict[str, str]] = []
        ingested: dict[str, dict[str, Any]] = {}

        files = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and not path.name.startswith(".")
            ),
            key=lambda path: path.name.lower(),
        )
        groups: dict[str, list[Path]] = {}
        for path in files:
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped.append(
                    {"file": path.name, "reason": "unsupported file type"}
                )
                continue
            doc_id = doc_id_for(path)
            if not doc_id:
                skipped.append(
                    {"file": path.name, "reason": "file name has no usable doc id"}
                )
                continue
            groups.setdefault(doc_id, []).append(path)

        for doc_id, paths in groups.items():
            candidates: list[tuple[Path, str]] = []
            for path in paths:
                text, reason = extract_text(path)
                if not text:
                    skipped.append(
                        {"file": path.name, "reason": reason, "doc_id": doc_id}
                    )
                    continue
                candidates.append((path, text))
            if not candidates:
                continue

            existing = corpus.get(doc_id, "")
            if len({text for _, text in candidates}) == 1:
                kept, text = candidates[0]
                for extra, _ in candidates[1:]:
                    duplicate_sources.append(
                        {
                            "doc_id": doc_id,
                            "kept": kept.name,
                            "file": extra.name,
                            "reason": "same text under one doc id",
                        }
                    )
            else:
                kept, text = candidates[0]
                if existing in {candidate for _, candidate in candidates}:
                    # Keep the text an agent may already be citing.
                    kept, text = next(
                        (pair for pair in candidates if pair[1] == existing)
                    )
                collisions.append(
                    {
                        "doc_id": doc_id,
                        "kept": kept.name,
                        "ignored": [
                            path.name
                            for path, candidate in candidates
                            if candidate != text
                        ],
                        "reason": (
                            "different content maps to one doc id;"
                            " rename the ignored files"
                        ),
                    }
                )

            if doc_id not in corpus:
                added.append(doc_id)
            elif corpus[doc_id] != text:
                updated.append(doc_id)
            else:
                unchanged.append(doc_id)
            corpus[doc_id] = text
            stat = kept.stat()
            ingested[doc_id] = {
                "source_name": kept.name,
                "content_hash": _text_hash(text),
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "ingested_at": now,
            }

        if corpus != before or not target.exists():
            _write_corpus(target, corpus)
        _sync_documents(corpus, ingested, now)

    catalog = document_catalog(corpus, store.list_documents())
    report = {
        "inbox": str(root),
        "corpus_file": str(target),
        "ingested_at": now,
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "skipped": skipped,
        "collisions": collisions,
        "duplicate_sources": duplicate_sources,
        "identity_duplicates": catalog["duplicates"],
        "total_docs": len(corpus),
    }
    if added or updated or collisions or skipped:
        LOGGER.info(
            "inbox ingest: added=%d updated=%d skipped=%d collisions=%d"
            " documents=%d",
            len(added),
            len(updated),
            len(skipped),
            len(collisions),
            len(corpus),
        )
    return report


def inbox_signature(inbox: Path | None = None) -> tuple[tuple[str, int, int], ...]:
    """Cheap change detector for the watcher. File contents are not read."""
    root = inbox or inbox_path()
    try:
        entries = []
        for path in root.iterdir():
            if path.is_file() and not path.name.startswith("."):
                stat = path.stat()
                entries.append((path.name, stat.st_size, stat.st_mtime_ns))
        return tuple(sorted(entries))
    except FileNotFoundError:
        return ()


class InboxWatcher:
    """Poll the inbox and run ``ingest_once`` when its file signature changes."""

    def __init__(
        self,
        interval: float | None = None,
        on_report: Any = None,
    ) -> None:
        # Read the environment at start(), after lifespan has loaded .env.
        self._interval_arg = interval
        self.interval = (
            max(0.2, interval) if interval is not None else DEFAULT_INTERVAL
        )
        self.on_report = on_report
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._signature: tuple[tuple[str, int, int], ...] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.interval = (
            max(0.2, self._interval_arg)
            if self._interval_arg is not None
            else ingest_interval()
        )
        self._signature = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="inbox-watcher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            signature = inbox_signature()
            if signature != self._signature:
                try:
                    report = ingest_once()
                except Exception:
                    LOGGER.exception("inbox ingest failed; will retry")
                else:
                    self._signature = signature
                    if self.on_report is not None:
                        self.on_report(report)
            self._stop.wait(self.interval)
