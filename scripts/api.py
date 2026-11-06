"""HTTP service over the deterministic runner. SQLite backed.

Runs execute in a background thread and return a run id immediately.
Each step is persisted and streamed as it happens (SSE tail), so the UI
never waits for the model to finish before showing progress.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
sys.path.insert(0, str(ROOT))

from agent import store  # noqa: E402
from agent.access import access_state, require_write  # noqa: E402
from agent.accounting import (  # noqa: E402
    GENERIC_COLUMNS,
    XERO_COLUMNS,
    XeroMapping,
    close_snapshot,
    ledger_rows,
    missing_xero_fields,
    periods_summary,
    periods_view,
    valid_period,
    xero_rows,
)
from agent.approvals import ApprovalQueue  # noqa: E402
from agent.bill_identity import identify_bill  # noqa: E402
from agent.bill_math import verify_ledger_math  # noqa: E402
from agent.dispatch import DurableCancel, RunDispatcher  # noqa: E402
from agent.guardrails import assess_due_amount, assess_grounding  # noqa: E402
from agent.ingest import (  # noqa: E402
    InboxWatcher,
    document_catalog,
    inbox_path,
    ingest_once,
    watcher_enabled,
)
from agent.periods import period_compatible  # noqa: E402
from agent.reconciliation import reconcile, summarize  # noqa: E402
from agent.runner import run_with_model  # noqa: E402
from agent.states import AgentState, Evidence  # noqa: E402
from agent.tools import LedgerArgs, LedgerWriteTool, RetrievalTool  # noqa: E402

@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    """Read .env once when a real server starts, not when tests import the app.

    The dispatcher owns the run queue for the life of the process. Every
    process pointed at the same database competes for the same claims.
    """
    from dotenv import load_dotenv

    load_dotenv()
    store.init_db()
    for absent in store.fail_unclaimed_runs():
        store.append_event(
            absent,
            "error",
            "failed: no worker owns this run after a restart",
            {"reason": "restart without a claim"},
        )
    WORKER.start()
    if watcher_enabled():
        WATCHER.start()
    try:
        yield
    finally:
        WATCHER.stop()
        WORKER.stop()


app = FastAPI(title="Ledgerproof", lifespan=lifespan)

TEST_CLIENT: Any = None


class RunRequest(BaseModel):
    task: str = Field(min_length=1)
    max_steps: int = Field(default=6, ge=1, le=20)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class DecideRequest(BaseModel):
    approved: bool
    decided_by: str = Field(default="web-ui", min_length=1)
    note: str = ""


class ResetRequest(BaseModel):
    confirm: bool = False


class ReverseRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
    approved_by: str = Field(min_length=1, max_length=120)


class CloseRequest(BaseModel):
    note: str = Field(min_length=3, max_length=500)
    decided_by: str = Field(default="web-ui", min_length=1, max_length=120)


def _client() -> Any:
    if TEST_CLIENT is not None:
        return TEST_CLIENT
    from agent.driver import build_client

    return build_client()


def _model() -> str:
    return "auto"


def _docs() -> dict[str, str]:
    override = os.environ.get("APP_CORPUS")
    path = Path(override) if override else ROOT / "data" / "corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})

# Same-process stop handles for a fast cancel. The durable signal lives in the
# runs table, so a cancel still reaches a worker in another process.
_CANCELS: dict[str, Any] = {}


def _persist_side_effect(
    run_id: str, kind: str, data: dict[str, Any], queue: ApprovalQueue
) -> None:
    """Make the world match the event before anyone can see the event.

    A ledger row exists when ``ledger_write`` is announced, and a ticket
    exists when ``gated`` is announced, so a UI that reacts to the stream
    always finds both.
    """
    if kind == "ledger_write" and data.get("entry_id"):
        store.insert_ledger_entry(
            run_id,
            str(data["entry_id"]),
            float(data["amount"]),
            str(data["currency"]),
            str(data["source_doc"]),
            bill_key=str(data.get("bill_key") or ""),
        )
    elif kind == "gated":
        wanted = data.get("ticket_id")
        for ticket in queue.pending():
            if ticket.ticket_id == wanted:
                store.submit_approval(
                    ticket.ticket_id,
                    run_id,
                    ticket.action,
                    ticket.detail,
                    ticket.meta,
                    ticket.evidence,
                )
                break


def _deny_pending_tickets(run_id: str) -> list[str]:
    """Close any ticket that exists while the run is being cancelled.

    Called by the cancel endpoint and again by the worker on exit, because a
    gate can be persisted in the small window between the click and the stop.
    """
    denied: list[str] = []
    for ticket in store.pending_approvals_for_run(run_id):
        try:
            store.decide_approval(
                ticket["ticket_id"], False, "system", "cancelled by user"
            )
        except store.AlreadyDecided:
            continue
        store.append_event(
            run_id,
            "denied",
            f"denied: Ticket #{ticket['display_no']} (cancelled by user)",
            {
                "ticket_id": ticket["ticket_id"],
                "action": ticket["action"],
                "decided_by": "system",
                "note": "cancelled by user",
            },
        )
        denied.append(ticket["ticket_id"])
    return denied


def _execute_run(run_id: str, task: str, max_steps: int, cancel: Any) -> None:
    """Worker body. Every step lands in the store the moment it happens."""
    _CANCELS[run_id] = cancel
    try:
        retrieval = RetrievalTool(_docs())
        ledger = LedgerWriteTool(
            posted=lambda args: store.find_ledger_posting(
                args.source_doc,
                args.amount,
                args.currency,
                bill_key=args.bill_key or "",
            )
        )
        queue = ApprovalQueue()
        state = AgentState(task=task, max_steps=max_steps)

        def _status() -> str:
            # Cancellation wins over whatever the step just set, so a slow step
            # finishing after the click can never resurrect "running".
            return "cancelled" if cancel.is_set() else state.status

        def sink(kind: str, body: str, data: dict[str, Any]) -> None:
            try:
                _persist_side_effect(run_id, kind, data, queue)
                store.append_event(run_id, kind, body, data)
                store.set_run_status(run_id, _status(), state.step_count)
            except store.DuplicateLedgerEntry as exc:
                # A duplicate entry is a containment failure, not a silent skip.
                store.append_event(run_id, "error", str(exc), {"reason": str(exc)})
                if not cancel.is_set():
                    state.status = "failed"
                    state.failure_reason = str(exc)
                store.set_run_status(run_id, _status(), state.step_count)
                raise

        state.stream_to(sink)
        try:
            state = run_with_model(
                state,
                _client(),
                _model(),
                retrieval,
                ledger,
                queue,
                cancel=cancel,
            )
        except Exception as exc:
            # A loud failure must still leave an explainable run record.
            message = f"{type(exc).__name__}: {exc}"
            store.append_event(run_id, "error", message, {"reason": message})
            if not cancel.is_set():
                state.status = "failed"
                state.failure_reason = message
            store.set_run_status(run_id, _status(), state.step_count)
            return
        store.set_run_status(run_id, _status(), state.step_count)
    finally:
        if cancel.is_set():
            _deny_pending_tickets(run_id)
        _CANCELS.pop(run_id, None)


def _claimed_run(run: dict[str, Any]) -> None:
    """Adapter from a claimed row to the worker body."""
    run_id = str(run["run_id"])
    attempts = int(run.get("attempts", 1))
    if attempts > 1:
        store.append_event(
            run_id,
            "status",
            f"status: resumed after worker loss (attempt {attempts})",
            {"status": "running", "attempt": attempts, "reason": "worker lost"},
        )
    _execute_run(
        run_id,
        str(run["task"]),
        int(run["max_steps"]),
        DurableCancel(run_id),
    )


WORKER = RunDispatcher(_claimed_run)
WATCHER = InboxWatcher()


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/config")
def config() -> dict[str, object]:
    """Tell the UI whether writes are open, token-gated, or read-only.

    The Xero block only carries mapping defaults. The token itself is never
    part of this response.
    """
    state: dict[str, object] = dict(access_state())
    state["xero"] = {
        "contact": os.environ.get("XERO_CONTACT", ""),
        "account_code": os.environ.get("XERO_ACCOUNT_CODE", ""),
        "tax_type": os.environ.get("XERO_TAX_TYPE", ""),
        "due_days": os.environ.get("XERO_DUE_DAYS", "0"),
    }
    return state


@app.get("/")
def index(request: Request) -> Any:
    index_file = STATIC_DIR / "index.html"
    if "text/html" in request.headers.get("accept", "") and index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-store"})
    return {
        "service": "ledgerproof",
        "ui": "/ui",
        "docs": "/docs",
        "health": "/health",
        "ledger": "/ledger",
        "approvals": "/approvals",
        "corpus": "/corpus",
        "documents": "/documents",
        "ingest": "/ingest",
        "periods": "/periods",
        "export_ledger": "/export/ledger.csv",
        "export_xero": "/export/xero.csv",
    }


@app.get("/ui")
def ui() -> Any:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="UI not installed")


@app.get("/corpus")
def corpus() -> dict[str, str]:
    return _docs()


@app.get("/documents")
def documents() -> dict[str, Any]:
    """The catalog an operator checks before a run: identity and duplicates."""
    store.init_db()
    return document_catalog(_docs(), store.list_documents())


@app.get("/documents/{doc_id}/file")
def document_file(doc_id: str) -> FileResponse:
    """Serve the original inbox file behind a catalog entry."""
    store.init_db()
    row = next(
        (doc for doc in store.list_documents() if doc["doc_id"] == doc_id),
        None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    name = str(row.get("source_name") or "")
    source = Path(name)
    if (
        not name
        or source.name != name
        or source.suffix.lower() not in {".pdf", ".txt", ".md"}
    ):
        raise HTTPException(status_code=404, detail="Source file not available")
    root = inbox_path().resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Source file not available")
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.post("/ingest", dependencies=[Depends(require_write)])
def ingest() -> dict[str, Any]:
    """Scan the inbox now. The watcher does the same on file changes."""
    try:
        return ingest_once()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"ingest failed: {type(exc).__name__}: {exc}"
        ) from None


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/ledger")
def ledger() -> dict[str, Any]:
    store.init_db()
    entries = store.list_ledger()
    # Outstanding postings made before the one-bill-one-row guard can still
    # repeat the same bill. They are immutable, so mark them, not hide them.
    counts: dict[tuple[str, float, str], int] = {}
    for item in entries:
        if item["kind"] != "posting" or item["reversed"]:
            continue
        key = (
            item["bill_key"] or item["source_doc"],
            float(item["amount"]),
            item["currency"],
        )
        counts[key] = counts.get(key, 0) + 1
    for item in entries:
        key = (
            item["bill_key"] or item["source_doc"],
            float(item["amount"]),
            item["currency"],
        )
        item["duplicate"] = item["kind"] == "posting" and counts.get(key, 0) > 1
    return {"entries": entries}


@app.get("/reconciliation")
def reconciliation() -> dict[str, Any]:
    """One row per canonical bill, with its reversal trail.

    Grouping happens over the append-only rows, so a bill corrected under a
    different document id still lands in one place.
    """
    store.init_db()
    bills = reconcile(store.list_ledger())
    return {"bills": bills, "summary": summarize(bills)}


def _documents() -> dict[str, dict]:
    return {doc["doc_id"]: doc for doc in store.list_documents()}


def _period_rows() -> list[dict]:
    return periods_view(
        store.list_ledger(), store.list_period_closures(), _documents()
    )


def _checked_period(period: str) -> str:
    try:
        return valid_period(period)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/periods")
def periods() -> dict[str, Any]:
    """Accounting months, their outstanding totals, and their close state."""
    store.init_db()
    months = _period_rows()
    return {"periods": months, "summary": periods_summary(months)}


@app.post("/periods/{period}/close", dependencies=[Depends(require_write)])
def close_month(period: str, req: CloseRequest) -> dict[str, Any]:
    """Freeze a month. Postings and reversals for its bills are refused."""
    store.init_db()
    _checked_period(period)
    snapshot = close_snapshot(period, store.list_ledger(), _documents())
    try:
        closure = store.close_period(
            period, req.note, req.decided_by, snapshot
        )
    except store.PeriodAlreadyClosed:
        raise HTTPException(
            status_code=409, detail=f"period {period} is already closed"
        ) from None
    return {"period": period, "status": "closed", "closure": closure}


@app.post("/periods/{period}/reopen", dependencies=[Depends(require_write)])
def reopen_month(period: str, req: CloseRequest) -> dict[str, Any]:
    """Reopen a month for a correction. The close action stays in history."""
    store.init_db()
    _checked_period(period)
    try:
        action = store.reopen_period(period, req.note, req.decided_by)
    except store.PeriodNotClosed:
        raise HTTPException(
            status_code=409, detail=f"period {period} is not closed"
        ) from None
    return {"period": period, "status": "open", "closure": action}


def _csv_response(columns: tuple[str, ...], rows: list[dict], name: str) -> Response:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/export/ledger.csv")
def export_ledger_csv(
    period: str | None = Query(default=None),
    scope: str = Query(default="all", pattern="^(all|outstanding)$"),
) -> Response:
    """Every row with its audit fields, or one accounting month."""
    store.init_db()
    if period:
        _checked_period(period)
    rows = ledger_rows(
        store.list_ledger(),
        store.list_period_closures(),
        _documents(),
        period=period,
        scope=scope,
    )
    return _csv_response(
        GENERIC_COLUMNS, rows, f"ledger-{period or 'all'}.csv"
    )


@app.get("/export/xero.csv")
def export_xero_csv(
    period: str | None = Query(default=None),
    contact: str | None = Query(default=None),
    account_code: str | None = Query(default=None),
    tax_type: str | None = Query(default=None),
    due_days: int | None = Query(default=None, ge=0, le=365),
) -> Response:
    """Xero bill import shape from outstanding bills.

    Contact, account code, and tax type are required: the ledger stores
    evidence and amounts, not the chart of accounts. They come from the
    request first, then from XERO_* defaults.
    """
    store.init_db()
    if period:
        _checked_period(period)
    mapping = XeroMapping(
        contact=(contact or os.environ.get("XERO_CONTACT", "")).strip(),
        account_code=(
            account_code or os.environ.get("XERO_ACCOUNT_CODE", "")
        ).strip(),
        tax_type=(tax_type or os.environ.get("XERO_TAX_TYPE", "")).strip(),
        due_days=(
            due_days
            if due_days is not None
            else int(os.environ.get("XERO_DUE_DAYS", "0") or 0)
        ),
    )
    missing = missing_xero_fields(mapping)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Xero export needs "
                + ", ".join(missing)
                + ". The ledger does not invent a supplier, account code, or"
                " tax type. Pass them in the query or set XERO_CONTACT,"
                " XERO_ACCOUNT_CODE, and XERO_TAX_TYPE."
            ),
        )
    rows = xero_rows(
        store.list_ledger(), _documents(), mapping, period=period
    )
    return _csv_response(
        XERO_COLUMNS, rows, f"xero-bills-{period or 'all'}.csv"
    )


@app.post("/ledger/{ledger_id}/reverse", dependencies=[Depends(require_write)])
def reverse_entry(ledger_id: int, req: ReverseRequest) -> dict[str, Any]:
    """Append a signed correction. The human action is the approval.

    Nothing is edited or deleted: the original row stays, and a new negative
    row names it, the reason, and who signed off.
    """
    store.init_db()
    try:
        entry = store.reverse_ledger_entry(ledger_id, req.reason, req.approved_by)
    except store.UnknownLedgerEntry:
        raise HTTPException(
            status_code=404, detail="unknown ledger entry"
        ) from None
    except store.AlreadyReversed:
        raise HTTPException(
            status_code=409, detail="ledger entry is already reversed"
        ) from None
    except store.NotReversible:
        raise HTTPException(
            status_code=409, detail="a reversal entry cannot be reversed"
        ) from None
    except store.PeriodClosed as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"period {exc.period} is closed; reopen it before reversing"
            ),
        ) from None
    store.append_event(
        entry["run_id"],
        "ledger_reversal",
        f"reversal: {entry['source_doc']} {entry['amount']:.2f}"
        f" {entry['currency']} by {req.approved_by} ({req.reason})",
        {
            "entry_id": entry["entry_id"],
            "reverses_id": entry["reverses_id"],
            "amount": entry["amount"],
            "currency": entry["currency"],
            "source_doc": entry["source_doc"],
            "bill_key": entry["bill_key"],
            "reason": req.reason,
            "approved_by": req.approved_by,
        },
    )
    return {"entry": entry}


@app.get("/storage")
def storage() -> dict[str, Any]:
    path = store.init_db()
    counts = store.table_counts()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "tables": counts,
    }


@app.post("/admin/reset", dependencies=[Depends(require_write)])
def reset(req: ResetRequest) -> dict[str, Any]:
    """Archive the current database and start empty.

    Destructive by intent, so it needs an explicit ``confirm`` and it never
    deletes rows without first writing a copy you can open.
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="reset requires confirm=true")
    store.init_db()
    archived = store.archive_and_clear()
    return {
        "archived": str(archived),
        "tables": store.table_counts(),
    }


@app.get("/approvals")
def approvals() -> dict[str, Any]:
    store.init_db()
    return {"pending": store.list_pending_approvals()}


def _run_receipt(run: dict[str, Any], replayed: bool) -> dict[str, Any]:
    run_id = run["run_id"]
    return {
        "run_id": run_id,
        "status": run["status"],
        "stream": f"/runs/{run_id}/stream",
        "replayed": replayed,
    }


@app.post("/run", status_code=202, dependencies=[Depends(require_write)])
def create_run(req: RunRequest) -> dict[str, Any]:
    store.init_db()
    if req.request_id:
        existing = store.find_run_by_request(req.request_id)
        if existing is not None:
            WORKER.wake()
            return _run_receipt(existing, True)
    try:
        run_id = store.create_run(
            req.task,
            request_id=req.request_id,
            max_steps=req.max_steps,
            status="pending",
        )
    except sqlite3.IntegrityError:
        # Lost a race with a concurrent submit carrying the same request id.
        existing = (
            store.find_run_by_request(req.request_id) if req.request_id else None
        )
        if existing is None:
            raise
        return _run_receipt(existing, True)
    WORKER.wake()
    return _run_receipt({"run_id": run_id, "status": "running"}, False)


@app.get("/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    store.init_db()
    return {"runs": store.list_runs(limit)}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    store.init_db()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return {"run": run, "events": store.get_events(run_id)}


@app.post("/runs/{run_id}/cancel", dependencies=[Depends(require_write)])
def cancel_run(run_id: str) -> dict[str, Any]:
    """Stop a live run. Cooperative: checked between steps, never mid-write."""
    store.init_db()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")
    if run["status"] in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409, detail=f"run already {run['status']}"
        )
    was_awaiting = run["status"] == "awaiting_approval"
    handle = _CANCELS.get(run_id)
    if handle is not None:
        handle.set()
    if not store.cancel_run(run_id):
        current = store.get_run(run_id) or run
        raise HTTPException(
            status_code=409, detail=f"run already {current['status']}"
        )
    denied = _deny_pending_tickets(run_id)
    ticket_note = " (cancelled by user)" if was_awaiting else ""
    store.append_event(
        run_id,
        "cancelled",
        f"cancelled: run stopped by user{ticket_note}",
        {"reason": "cancelled by user", "tickets_denied": denied},
    )
    return {"run_id": run_id, "status": "cancelled", "tickets_denied": denied}


def _sse(payload: dict[str, Any], event_id: int | None = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}data: {json.dumps(payload)}\n\n"


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    """Tail the run. Replays from the start, or from Last-Event-ID."""
    store.init_db()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run")

    cursor = request.headers.get("last-event-id") or request.query_params.get("after")
    try:
        last = int(cursor) if cursor not in (None, "") else -1
    except ValueError:
        last = -1

    async def _gen() -> Any:
        nonlocal last
        sent_status: str | None = None
        heartbeat = 0
        while True:
            if await request.is_disconnected():
                return
            events = await anyio.to_thread.run_sync(
                lambda: store.get_events(run_id, after=last)
            )
            for item in events:
                last = int(item["seq"])
                yield _sse(item, event_id=last)
            current = await anyio.to_thread.run_sync(store.get_run, run_id)
            if current is None:
                return
            if current["status"] != sent_status:
                sent_status = current["status"]
                yield _sse(
                    {
                        "seq": last,
                        "kind": "status",
                        "body": f"status: {sent_status}",
                        "data": {
                            "status": sent_status,
                            "step_count": current["step_count"],
                        },
                    }
                )
            if sent_status in TERMINAL_STATUSES:
                return
            if not events:
                heartbeat += 1
                if heartbeat % 75 == 0:  # ~15s of silence
                    yield ": keep-alive\n\n"
            await anyio.sleep(0.2)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _gate(
    ticket_id: str, approved: bool, decided_by: str, note: str
) -> dict[str, Any]:
    try:
        ticket = store.decide_approval(ticket_id, approved, decided_by, note)
    except store.AlreadyDecided:
        raise HTTPException(
            status_code=409, detail=f"ticket already decided: {ticket_id}"
        ) from None
    if ticket is None:
        raise HTTPException(status_code=404, detail="unknown approval ticket")
    return ticket


@app.post("/approvals/{ticket_id}/decide", dependencies=[Depends(require_write)])
def decide(ticket_id: str, req: DecideRequest) -> dict[str, Any]:
    store.init_db()
    ticket = store.get_approval(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="unknown approval ticket")
    if ticket["decided"]:
        raise HTTPException(
            status_code=409, detail=f"ticket already decided: {ticket_id}"
        )

    run_id = ticket["run_id"]
    run = store.get_run(run_id) or {}
    if run.get("status") == "cancelled":
        raise HTTPException(status_code=409, detail="run was cancelled")

    if not req.approved:
        ticket = _gate(ticket_id, False, req.decided_by, req.note)
        body = f"denied: {ticket_id} by {req.decided_by}"
        if req.note:
            body += f" ({req.note})"
        store.append_event(
            run_id,
            "denied",
            body,
            {
                "ticket_id": ticket_id,
                "action": ticket["action"],
                "decided_by": req.decided_by,
                "note": req.note,
            },
        )
        store.set_run_status(run_id, "failed", int(run.get("step_count", 0)))
        return {
            "ticket_id": ticket_id,
            "display_no": ticket["display_no"],
            "run_id": run_id,
            "approved": False,
            "status": "failed",
            "decided_by": req.decided_by,
        }

    if ticket["action"] != "ledger_write":
        ticket = _gate(ticket_id, True, req.decided_by, req.note)
        store.append_event(
            run_id,
            "approval",
            f"approval: {ticket_id} granted by {req.decided_by}",
            {
                "ticket_id": ticket_id,
                "action": ticket["action"],
                "decided_by": req.decided_by,
                "note": req.note,
            },
        )
        store.set_run_status(run_id, "done", int(run.get("step_count", 0)))
        return {
            "ticket_id": ticket_id,
            "display_no": ticket["display_no"],
            "run_id": run_id,
            "approved": True,
            "status": "done",
            "decided_by": req.decided_by,
        }

    meta = ticket.get("meta") or {}
    try:
        args = LedgerArgs(
            entry_id=str(meta.get("entry_id", "")),
            amount=float(meta.get("amount", 0)),
            currency=str(meta.get("currency", "AUD")),
            source_doc=str(meta.get("source_doc", "")),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"stored args invalid: {exc}")

    packet = ticket.get("evidence") or {}
    try:
        evidence = [Evidence(**item) for item in packet.get("citations", [])]
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"stored evidence invalid: {exc}"
        )
    if not evidence:
        raise HTTPException(
            status_code=422,
            detail="ticket has no evidence packet; re-run the task",
        )

    ok, reason = assess_grounding(args, evidence)
    if not ok:
        raise HTTPException(status_code=422, detail=reason)
    math_ok, math_reason = verify_ledger_math(args)
    if not math_ok:
        raise HTTPException(status_code=422, detail=math_reason)
    due_ok, due_reason = assess_due_amount(args, evidence)
    if not due_ok:
        raise HTTPException(status_code=422, detail=due_reason)
    task = str(packet.get("task") or run.get("task") or "")
    cited = "\n".join(
        item.text for item in evidence if item.doc_id == args.source_doc
    )
    period_ok, period_reason = period_compatible(task, cited)
    if not period_ok:
        raise HTTPException(status_code=422, detail=period_reason)

    identity = identify_bill(args.source_doc, cited)
    expected = str(meta.get("bill_key") or "")
    if expected and identity.key != expected:
        raise HTTPException(
            status_code=422,
            detail="bill identity changed since the ticket was raised",
        )
    args.bill_key = identity.key

    closed = store.closed_period_for(args.bill_key, args.source_doc)
    if closed:
        # Refuse before deciding, so the ticket survives for after a reopen.
        raise HTTPException(
            status_code=409,
            detail=(
                f"period {closed} is closed; reopen it before posting"
            ),
        )

    ticket = _gate(ticket_id, True, req.decided_by, req.note)
    receipt = f"ledger_write: {args.source_doc} {args.amount:.2f} {args.currency}"
    try:
        store.insert_ledger_entry(
            run_id,
            args.entry_id,
            args.amount,
            args.currency,
            args.source_doc,
            bill_key=args.bill_key,
        )
    except store.DuplicateLedgerEntry:
        raise HTTPException(
            status_code=409,
            detail=(
                f"already posted: {args.source_doc} is the same bill as an"
                " existing ledger row"
            ),
        )
    except store.PeriodClosed as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"period {exc.period} is closed; reopen it before posting"
            ),
        ) from None
    store.append_event(
        run_id,
        "approval",
        f"approval: {ticket_id} granted by {req.decided_by}",
        {
            "ticket_id": ticket_id,
            "action": ticket["action"],
            "decided_by": req.decided_by,
            "note": req.note,
        },
    )
    store.append_event(
        run_id,
        "ledger_write",
        receipt,
        {
            "entry_id": args.entry_id,
            "amount": args.amount,
            "currency": args.currency,
            "source_doc": args.source_doc,
            "bill_key": args.bill_key,
            "ticket_id": ticket_id,
        },
    )
    store.set_run_status(run_id, "done", int(run.get("step_count", 0)))
    return {
        "ticket_id": ticket_id,
        "display_no": ticket["display_no"],
        "run_id": run_id,
        "approved": True,
        "status": "done",
        "receipt": receipt,
        "decided_by": req.decided_by,
    }
