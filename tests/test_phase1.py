"""Phase 1: the UI must be able to tell the truth while the run is moving.

These pin the contract the dashboard relies on: ``/run`` returns before the
model finishes, steps are persisted as they happen, and the stream is a real
cursor-based tail with structured payloads.
"""

import json
import threading

import pytest
from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from tests.helpers import wait_for_run


class _GatedClient:
    """Blocks the first model call until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=10)
        payload = {"action": "finish", "args": {}, "reason": "done"}
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


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


def test_run_returns_while_model_is_still_thinking(isolated_db, monkeypatch) -> None:
    model = _GatedClient()
    monkeypatch.setattr(api, "TEST_CLIENT", model)

    with TestClient(api.app) as client:
        started = client.post("/run", json={"task": "slow one", "max_steps": 4})

        assert started.status_code == 202
        run_id = started.json()["run_id"]
        # The model has not been allowed to answer yet, so the run is live and
        # the request has already returned. That is the whole point.
        assert model.started.wait(5)
        assert store.get_run(run_id)["status"] == "running"

        model.release.set()
        settled = wait_for_run(client, run_id)
        assert settled["run"]["status"] == "done"


def test_events_are_structured_with_kind_body_and_data(
    isolated_db, monkeypatch
) -> None:
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
                {"action": "finish", "args": {}, "reason": "done"},
            ]
        ),
    )
    with TestClient(api.app) as client:
        run_id = client.post(
            "/run", json={"task": "t", "max_steps": 4}
        ).json()["run_id"]
        settled = wait_for_run(client, run_id)

        events = settled["events"]
        assert all({"seq", "kind", "body", "data"} <= set(e) for e in events)
        assert all(isinstance(e["data"], dict) for e in events)
        retrieve = next(e for e in events if e["kind"] == "retrieve")
        assert retrieve["data"]["doc_ids"]
        decision = next(e for e in events if e["kind"] == "decision")
        assert decision["data"]["action"] == "retrieve"


def test_stream_is_a_cursor_tail_with_status_transitions(
    isolated_db, monkeypatch
) -> None:
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
                {"action": "finish", "args": {}, "reason": "done"},
            ]
        ),
    )
    with TestClient(api.app) as client:
        run_id = client.post(
            "/run", json={"task": "t", "max_steps": 4}
        ).json()["run_id"]
        settled = wait_for_run(client, run_id)

        full = client.get(f"/runs/{run_id}/stream")
        assert full.status_code == 200
        assert "id: 0" in full.text
        assert "keep-alive" not in full.text
        assert '"kind": "status"' in full.text
        assert '"status": "done"' in full.text

        # A reconnecting client resumes after the id it already saw.
        resumed = client.get(
            f"/runs/{run_id}/stream", headers={"Last-Event-ID": "1"}
        )
        assert resumed.status_code == 200
        assert "id: 0\n" not in resumed.text
        assert "id: 2\n" in resumed.text
        assert settled["events"][0]["body"] not in resumed.text


def test_awaiting_approval_keeps_the_stream_open_until_decided(
    isolated_db, monkeypatch
) -> None:
    """The gate is not terminal: a decision lands on the same stream."""
    monkeypatch.setattr(
        api,
        "TEST_CLIENT",
        _ScriptedClient(
            [
                {"action": "request_approval", "args": {}, "reason": "need a human"},
            ]
        ),
    )
    with TestClient(api.app) as client:
        run_id = client.post(
            "/run", json={"task": "t", "max_steps": 3}
        ).json()["run_id"]
        settled = wait_for_run(client, run_id)
        assert settled["run"]["status"] == "awaiting_approval"

        ticket = client.get("/approvals").json()["pending"][0]

        def _decide() -> None:
            client.post(
                f"/approvals/{ticket['ticket_id']}/decide",
                json={"approved": True, "decided_by": "tester"},
            )

        timer = threading.Timer(0.4, _decide)
        timer.start()
        try:
            stream = client.get(f"/runs/{run_id}/stream")
        finally:
            timer.cancel()

        assert '"kind": "approval"' in stream.text
        assert '"status": "done"' in stream.text
