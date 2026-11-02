"""Regression tests for the Phase 0 trust guarantees.

Each test pins a claim that used to be false: evidence cannot be
manufactured by a rejection, approvals are one-shot, tickets are unique,
failed runs are recorded, and the ledger rejects edits.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from agent.approvals import ApprovalQueue
from agent.ledger_flow import guarded_ledger_write, resume_after_approval
from agent.runner import run_with_model
from agent.states import AgentState, Evidence
from agent.tools import LedgerArgs, LedgerWriteTool, RetrievalTool
from tests.helpers import wait_for_run


class _ScriptedClient:
    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.chat = type("Chat", (), {"completions": self})()
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        payload = self._script.pop(0) if self._script else {
            "action": "finish",
            "args": {},
            "reason": "done",
        }
        content = json.dumps(payload)
        return type(
            "Completion",
            (),
            {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]},
        )()


class _ExplodingClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        raise RuntimeError("model endpoint unavailable")


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


def _args(amount: float = 100.0, source: str = "bill") -> LedgerArgs:
    return LedgerArgs(entry_id="j-1", amount=amount, source_doc=source)


def test_invented_amount_cannot_self_ground_through_rejection() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = AgentState(
        task="t",
        evidence=[Evidence(doc_id="bill", text="Amount due 63.70")],
    )
    invented = _args(777.77)

    first = guarded_ledger_write(state, ledger, invented, queue)
    second = guarded_ledger_write(state, ledger, invented, queue)

    assert first.startswith("rejected:")
    assert second.startswith("rejected:")
    assert ledger.entries == []


def test_request_approval_creates_a_decidable_ticket() -> None:
    queue = ApprovalQueue()
    state = run_with_model(
        AgentState(task="t", max_steps=3),
        _ScriptedClient(
            [{"action": "request_approval", "args": {}, "reason": "need a human"}]
        ),
        "test",
        RetrievalTool({"lease-7": "recovery clause"}),
        LedgerWriteTool(),
        queue,
    )
    pending = queue.pending()
    assert state.status == "awaiting_approval"
    assert len(pending) == 1
    assert pending[0].action == "request_approval"
    assert pending[0].evidence["rule"] == "model invoked the approval channel"


def test_ticket_ids_are_unique_across_queues() -> None:
    first = ApprovalQueue().submit("ledger_write", "gate")
    second = ApprovalQueue().submit("ledger_write", "gate")
    assert first != second


def test_resume_rechecks_evidence_and_fails_closed() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = AgentState(
        task="t",
        evidence=[Evidence(doc_id="bill", text="total 50000.00")],
    )
    args = _args(50_000.0)
    ticket_id = guarded_ledger_write(state, ledger, args, queue)
    assert ticket_id.startswith("t-")

    state.evidence.clear()
    event = resume_after_approval(state, ledger, args, queue, ticket_id)

    assert state.status == "failed"
    assert "post-approval" in event
    assert ledger.entries == []


def test_approval_packet_carries_citations_and_checks() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = AgentState(
        task="t",
        evidence=[Evidence(doc_id="bill", text="Amount due 50000.00")],
    )
    ticket_id = guarded_ledger_write(state, ledger, _args(50_000.0), queue)
    ticket = next(t for t in queue.pending() if t.ticket_id == ticket_id)

    assert ticket.evidence["proposed"]["amount"] == 50_000.0
    assert ticket.evidence["checks"]["grounding"] == "grounded"
    assert ticket.evidence["citations"][0]["doc_id"] == "bill"


def test_tickets_get_sequential_human_numbers(isolated_db) -> None:
    run_id = store.create_run("t")
    store.submit_approval("t-a", run_id, "ledger_write", "first", {})
    store.submit_approval("t-b", run_id, "ledger_write", "second", {})

    pending = store.list_pending_approvals()
    assert [item["display_no"] for item in pending] == [1, 2]
    assert [item["ticket_id"] for item in pending] == ["t-a", "t-b"]


def test_store_decide_is_one_shot(isolated_db) -> None:
    run_id = store.create_run("t")
    store.submit_approval("t-one", run_id, "ledger_write", "gate", {})
    store.decide_approval("t-one", True, "tester")
    with pytest.raises(store.AlreadyDecided):
        store.decide_approval("t-one", True, "tester")


def test_ledger_and_events_are_append_only(isolated_db) -> None:
    run_id = store.create_run("t")
    store.append_event(run_id, "note", "hello")
    store.insert_ledger_entry(run_id, "j-1", 10.0, "AUD", "bill")

    update_conn = store.connect()
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            update_conn.execute("UPDATE ledger SET amount = 1.0")
    finally:
        update_conn.close()

    delete_conn = store.connect()
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            delete_conn.execute("DELETE FROM events")
    finally:
        delete_conn.close()


def test_clear_ledger_archives_first_and_restores_triggers(isolated_db) -> None:
    """Clearing is allowed only as an archive-then-reset, never a bare delete."""
    run_id = store.create_run("t")
    store.append_event(run_id, "note", "hello")
    store.insert_ledger_entry(run_id, "j-1", 10.0, "AUD", "bill")
    store.submit_approval("t-one", run_id, "ledger_write", "post bill", {})

    archive = store.archive_and_clear()

    assert archive.exists(), "the old database must survive as a file"
    archived = sqlite3.connect(str(archive))
    try:
        assert archived.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 1
        assert archived.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert archived.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert archived.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] == 1
    finally:
        archived.close()

    assert store.table_counts() == {
        "runs": 0,
        "events": 0,
        "ledger": 0,
        "approvals": 0,
        "documents": 0,
        "period_closures": 0,
    }

    # A cleared ledger is still append-only, and accepts new postings as if
    # it were brand new.
    fresh = store.create_run("t2")
    store.insert_ledger_entry(fresh, "j-1", 10.0, "AUD", "bill")
    assert len(store.list_ledger()) == 1

    conn = store.connect()
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM ledger")
    finally:
        conn.close()


def test_double_ledger_write_for_same_run_is_rejected(isolated_db) -> None:
    run_id = store.create_run("t")
    store.insert_ledger_entry(run_id, "j-1", 10.0, "AUD", "bill")
    with pytest.raises(store.DuplicateLedgerEntry):
        store.insert_ledger_entry(run_id, "j-1", 10.0, "AUD", "bill")
    assert len(store.list_ledger()) == 1


def test_init_db_never_waits_on_a_busy_writer(isolated_db) -> None:
    """Endpoints call init_db per request; it must not fight a live run."""
    store.init_db()
    conn = store.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO runs (run_id, task, status, step_count, created_at,"
            " updated_at) VALUES ('r-busy', 't', 'running', 0, 'now', 'now')"
        )
        store.init_db()  # would raise "database is locked" if it set WAL again
    finally:
        conn.rollback()
        conn.close()


def test_same_bill_under_a_new_label_is_still_a_duplicate(isolated_db) -> None:
    """entry_id is a label the model invents, so it cannot be the identity."""
    first = store.create_run("t")
    second = store.create_run("t")
    store.insert_ledger_entry(
        first, "alex-bill-001", 13376.47, "AUD", "origin-bill"
    )
    with pytest.raises(store.DuplicateLedgerEntry):
        store.insert_ledger_entry(
            second, "origin-standing-bill-13376.47", 13376.47, "AUD", "origin-bill"
        )
    assert len(store.list_ledger()) == 1


def test_reposting_a_bill_never_reaches_a_human(isolated_db, monkeypatch) -> None:
    monkeypatch.setenv(
        "APP_CORPUS", str(Path(__file__).parent / "fixtures" / "corpus.json")
    )
    seed_run = store.create_run("seed")
    store.insert_ledger_entry(
        seed_run, "alex-bill-001", 4049.18, "AUD", "bill-q3-electricity"
    )
    monkeypatch.setattr(
        api,
        "TEST_CLIENT",
        _ScriptedClient(
            [
                {
                    "action": "retrieve",
                    "args": {"query": "lot 7 Sample Street account 99881"},
                    "reason": "find bill",
                },
                {
                    "action": "ledger_write",
                    "args": {
                        "entry_id": "lease7-q3-electricity",
                        "amount": 4049.18,
                        "source_doc": "bill-q3-electricity",
                    },
                    "reason": "re-post the same bill",
                },
                {"action": "finish", "args": {}, "reason": "done"},
            ]
        ),
    )
    with TestClient(api.app) as client:
        started = client.post("/run", json={"task": "recon", "max_steps": 6})
        run_id = started.json()["run_id"]
        settled = wait_for_run(client, run_id)

        # Nothing new posted, nothing queued for a human to rubber-stamp.
        assert len(store.list_ledger()) == 1
        assert client.get("/approvals").json()["pending"] == []
        rejected = [e for e in settled["events"] if e["kind"] == "rejected"]
        assert rejected, settled["events"]
        assert "already posted" in rejected[0]["data"]["reason"]
        assert settled["run"]["status"] == "done"


def _seed_ticket(
    run_id: str,
    ticket_id: str,
    *,
    action: str = "ledger_write",
    amount: float = 4049.18,
    source_doc: str = "bill-q3-electricity",
    task: str = "recon",
    text: str = "Amount due 4049.18 dollars Bill period 1 July to 30 September",
    with_evidence: bool = True,
) -> None:
    packet = (
        {
            "task": task,
            "proposed": {"amount": amount, "source_doc": source_doc},
            "rule": "test gate",
            "checks": {"grounding": "grounded"},
            "citations": [{"doc_id": source_doc, "text": text, "query": ""}],
        }
        if with_evidence
        else {}
    )
    store.submit_approval(
        ticket_id,
        run_id,
        action,
        "test gate",
        {
            "entry_id": "lease-7-q3",
            "amount": amount,
            "currency": "AUD",
            "source_doc": source_doc,
        },
        packet,
    )


def test_api_approval_is_idempotent(isolated_db) -> None:
    run_id = store.create_run("recon")
    store.set_run_status(run_id, "awaiting_approval", 2)
    _seed_ticket(run_id, "t-idem")

    client = TestClient(api.app)
    first = client.post(
        "/approvals/t-idem/decide",
        json={"approved": True, "decided_by": "alice"},
    )
    second = client.post(
        "/approvals/t-idem/decide",
        json={"approved": True, "decided_by": "alice"},
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 409
    assert len(store.list_ledger()) == 1
    decided = store.get_approval("t-idem")
    assert decided is not None and decided["decided_by"] == "alice"


def test_api_approval_without_evidence_fails_closed(isolated_db) -> None:
    run_id = store.create_run("recon")
    store.set_run_status(run_id, "awaiting_approval", 2)
    _seed_ticket(run_id, "t-noevidence", with_evidence=False)

    res = TestClient(api.app).post(
        "/approvals/t-noevidence/decide", json={"approved": True}
    )

    assert res.status_code == 422
    assert store.list_ledger() == []
    assert store.get_approval("t-noevidence")["decided"] == 0


def test_api_approval_rechecks_period(isolated_db) -> None:
    run_id = store.create_run("agl bill for christmas")
    store.set_run_status(run_id, "awaiting_approval", 2)
    _seed_ticket(
        run_id,
        "t-period",
        amount=63.70,
        source_doc="bill-agl-apr23",
        task="agl bill for christmas",
        text="Amount due 63.70 dollars Due date 23 Apr 2023",
    )

    res = TestClient(api.app).post(
        "/approvals/t-period/decide", json={"approved": True}
    )

    assert res.status_code == 422
    assert store.list_ledger() == []


def test_request_approval_ticket_can_be_approved_via_api(
    isolated_db, monkeypatch
) -> None:
    monkeypatch.setattr(
        api,
        "TEST_CLIENT",
        _ScriptedClient(
            [{"action": "request_approval", "args": {}, "reason": "need a human"}]
        ),
    )
    with TestClient(api.app) as client:
        started = client.post("/run", json={"task": "t", "max_steps": 3})
        assert started.status_code == 202, started.text
        run = wait_for_run(client, started.json()["run_id"])
        assert run["run"]["status"] == "awaiting_approval"

        pending = client.get("/approvals").json()["pending"]
        assert len(pending) == 1
        ticket_id = pending[0]["ticket_id"]

        res = client.post(
            f"/approvals/{ticket_id}/decide",
            json={"approved": True, "decided_by": "alice"},
        )
        assert res.status_code == 200, res.text
        assert res.json()["status"] == "done"
        assert store.get_run(started.json()["run_id"])["status"] == "done"
        assert store.list_ledger() == []


def test_failed_run_is_recorded_not_left_running(isolated_db, monkeypatch) -> None:
    monkeypatch.setattr(api, "TEST_CLIENT", _ExplodingClient())

    with TestClient(api.app, raise_server_exceptions=False) as client:
        res = client.post("/run", json={"task": "explode", "max_steps": 3})

        # Starting a run cannot fail on the model: the work happens after the
        # response, and the failure is recorded on the run instead of the caller.
        assert res.status_code == 202
        run_id = res.json()["run_id"]
        settled = wait_for_run(client, run_id)
        assert settled["run"]["status"] == "failed"
        events = store.get_events(run_id)
        assert any(e["kind"] == "error" for e in events)
