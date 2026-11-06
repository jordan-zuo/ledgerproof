"""Phase 2: run lifecycle. List, replay, reattach, cancel, one-shot decisions."""

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from tests.helpers import wait_for_event, wait_for_run

FIXTURE = Path(__file__).parent / "fixtures" / "corpus.json"
ORIGIN_DOC = "origin-standing-bill-modified-alexcarter"


def _completion(payload: dict):
    content = json.dumps(payload)
    return type(
        "Completion",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {"message": type("Msg", (), {"content": content})()},
                )
            ]
        },
    )()


class _ScriptedClient:
    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        payload = self._script.pop(0) if self._script else {
            "action": "finish",
            "args": {},
            "reason": "done",
        }
        return _completion(payload)


class _BlockingClient:
    """Holds the first model call until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=10)
        return _completion({"action": "finish", "args": {}, "reason": "done"})


def _wait_worker_exit(run_id: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if run_id not in api._CANCELS:
            return
        time.sleep(0.02)
    raise AssertionError(f"worker for {run_id} did not exit")


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


def test_run_list_is_newest_first_and_limited(isolated_db) -> None:
    store.create_run("first")
    second = store.create_run("second")
    third = store.create_run("third")

    client = TestClient(api.app)
    res = client.get("/runs?limit=2")

    assert res.status_code == 200
    runs = res.json()["runs"]
    assert [run["run_id"] for run in runs] == [third, second]
    assert {"run_id", "task", "status", "step_count", "created_at"} <= set(runs[0])


def test_request_id_replay_returns_the_same_run(isolated_db, monkeypatch) -> None:
    monkeypatch.setattr(api, "TEST_CLIENT", _ScriptedClient([]))

    with TestClient(api.app) as client:
        first = client.post(
            "/run", json={"task": "t", "max_steps": 3, "request_id": "req-1"}
        )
        second = client.post(
            "/run", json={"task": "t", "max_steps": 3, "request_id": "req-1"}
        )

        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["replayed"] is False
        assert second.json()["replayed"] is True
        assert first.json()["run_id"] == second.json()["run_id"]
        assert len(client.get("/runs").json()["runs"]) == 1

        # Reusing the request_id with different task returns 409 Conflict
        diff_task = client.post(
            "/run", json={"task": "different", "max_steps": 3, "request_id": "req-1"}
        )
        assert diff_task.status_code == 409
        assert "request_id was already used" in diff_task.json()["detail"]

        # Reusing the request_id with different max_steps returns 409 Conflict
        diff_steps = client.post(
            "/run", json={"task": "t", "max_steps": 5, "request_id": "req-1"}
        )
        assert diff_steps.status_code == 409
        assert "request_id was already used" in diff_steps.json()["detail"]


def test_cancel_running_run_discards_the_inflight_decision(
    isolated_db, monkeypatch
) -> None:
    model = _BlockingClient()
    monkeypatch.setattr(api, "TEST_CLIENT", model)

    with TestClient(api.app) as client:
        started = client.post("/run", json={"task": "post a bill", "max_steps": 5})
        run_id = started.json()["run_id"]
        assert model.started.wait(5)

        cancelled = client.post(f"/runs/{run_id}/cancel")

        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert store.get_run(run_id)["status"] == "cancelled"

        events = wait_for_event(client, run_id, "cancelled")
        model.release.set()
        _wait_worker_exit(run_id)

        assert store.get_run(run_id)["status"] == "cancelled"
        assert not any(event["kind"] == "decision" for event in events)
        assert client.get("/ledger").json()["entries"] == []
        assert client.post(f"/runs/{run_id}/cancel").status_code == 409

        stream = client.get(f"/runs/{run_id}/stream")
        assert '"status": "cancelled"' in stream.text


def test_cancel_reaches_a_worker_with_no_local_signal(
    isolated_db, monkeypatch
) -> None:
    """A cancel can arrive in another process, so the DB flag must stop it."""
    model = _BlockingClient()
    monkeypatch.setattr(api, "TEST_CLIENT", model)

    with TestClient(api.app) as client:
        run_id = client.post(
            "/run", json={"task": "post a bill", "max_steps": 5}
        ).json()["run_id"]
        assert model.started.wait(5)

        # Only the database changes here, as if the click landed elsewhere.
        assert store.cancel_run(run_id) is True
        model.release.set()
        _wait_worker_exit(run_id)

        assert store.get_run(run_id)["status"] == "cancelled"
        events = store.get_events(run_id)
        assert not any(e["kind"] in {"decision", "finish"} for e in events)
        assert client.get("/ledger").json()["entries"] == []


def test_cancel_at_a_gate_denies_the_ticket(isolated_db, monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "TEST_CLIENT",
        _ScriptedClient(
            [{"action": "request_approval", "args": {}, "reason": "need a human"}]
        ),
    )
    with TestClient(api.app) as client:
        run_id = client.post(
            "/run", json={"task": "t", "max_steps": 3}
        ).json()["run_id"]
        settled = wait_for_run(client, run_id)
        assert settled["run"]["status"] == "awaiting_approval"

        ticket = client.get("/approvals").json()["pending"][0]
        cancelled = client.post(f"/runs/{run_id}/cancel")

        assert cancelled.status_code == 200
        assert cancelled.json()["tickets_denied"] == [ticket["ticket_id"]]
        assert store.get_run(run_id)["status"] == "cancelled"
        assert client.get("/approvals").json()["pending"] == []

        decided = store.get_approval(ticket["ticket_id"])
        assert decided["decided"] == 1
        assert decided["approved"] == 0
        assert decided["note"] == "cancelled by user"

        late = client.post(
            f"/approvals/{ticket['ticket_id']}/decide", json={"approved": True}
        )
        assert late.status_code == 409

        events = client.get(f"/runs/{run_id}").json()["events"]
        assert any(event["kind"] == "cancelled" for event in events)
        assert any(event["kind"] == "denied" for event in events)


def test_cancel_reports_a_ticket_the_worker_already_denied(
    isolated_db, monkeypatch
) -> None:
    """The worker denies pending tickets as it exits. If it wins the race,
    the click must still name the ticket it closed instead of an empty list."""
    monkeypatch.setattr(
        api,
        "TEST_CLIENT",
        _ScriptedClient(
            [{"action": "request_approval", "args": {}, "reason": "need a human"}]
        ),
    )
    original_cancel = store.cancel_run

    def cancel_then_let_the_worker_deny(run_id: str) -> bool:
        cancelled = original_cancel(run_id)
        api._deny_pending_tickets(run_id)
        return cancelled

    monkeypatch.setattr(api.store, "cancel_run", cancel_then_let_the_worker_deny)

    with TestClient(api.app) as client:
        run_id = client.post(
            "/run", json={"task": "t", "max_steps": 3}
        ).json()["run_id"]
        assert wait_for_run(client, run_id)["run"]["status"] == "awaiting_approval"
        ticket = client.get("/approvals").json()["pending"][0]

        cancelled = client.post(f"/runs/{run_id}/cancel")

        assert cancelled.status_code == 200
        assert cancelled.json()["tickets_denied"] == [ticket["ticket_id"]]
        denied = [e for e in store.get_events(run_id) if e["kind"] == "denied"]
        assert len(denied) == 1


def test_second_decision_on_a_ledger_ticket_is_refused(isolated_db) -> None:
    """Two tabs approve the same gate: one posts, the other gets 409."""
    run_id = store.create_run("post the Origin standing bill")
    text = json.loads(FIXTURE.read_text(encoding="utf-8"))[ORIGIN_DOC]
    amount = 13376.47
    packet = {
        "task": "post the Origin standing bill",
        "proposed": {
            "entry_id": "origin-phase2",
            "amount": amount,
            "currency": "AUD",
            "source_doc": ORIGIN_DOC,
        },
        "threshold": 10000.0,
        "rule": "amount exceeds approval threshold",
        "checks": {},
        "citations": [{"doc_id": ORIGIN_DOC, "text": text, "query": "origin"}],
    }
    store.submit_approval(
        "t-phase2",
        run_id,
        "ledger_write",
        "amount exceeds approval threshold 10000.00",
        {
            "entry_id": "origin-phase2",
            "amount": amount,
            "currency": "AUD",
            "source_doc": ORIGIN_DOC,
        },
        packet,
    )
    store.set_run_status(run_id, "awaiting_approval", 2)

    client = TestClient(api.app)
    first = client.post(
        "/approvals/t-phase2/decide",
        json={"approved": True, "decided_by": "tab-1"},
    )
    second = client.post(
        "/approvals/t-phase2/decide",
        json={"approved": True, "decided_by": "tab-2"},
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 409
    ledger = client.get("/ledger").json()["entries"]
    assert len([e for e in ledger if e["source_doc"] == ORIGIN_DOC]) == 1
    assert store.get_run(run_id)["status"] == "done"


def test_decide_refuses_a_cancelled_run(isolated_db) -> None:
    run_id = store.create_run("x")
    store.set_run_status(run_id, "cancelled", 1)
    store.submit_approval("t-cancelled", run_id, "request_approval", "x", {})

    res = TestClient(api.app).post(
        "/approvals/t-cancelled/decide", json={"approved": True}
    )

    assert res.status_code == 409
    assert "cancelled" in res.json()["detail"]


def test_decide_refuses_terminal_runs(isolated_db) -> None:
    client = TestClient(api.app)

    # Failed run
    failed_id = store.create_run("f")
    store.set_run_status(failed_id, "failed", 1)
    store.submit_approval("t-failed", failed_id, "request_approval", "x", {})
    res_f = client.post("/approvals/t-failed/decide", json={"approved": True})
    assert res_f.status_code == 409
    assert "failed" in res_f.json()["detail"]

    # Done run
    done_id = store.create_run("d")
    store.set_run_status(done_id, "done", 1)
    store.submit_approval("t-done", done_id, "request_approval", "x", {})
    res_d = client.post("/approvals/t-done/decide", json={"approved": True})
    assert res_d.status_code == 409
    assert "done" in res_d.json()["detail"]
