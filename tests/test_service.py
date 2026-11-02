import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from tests.helpers import wait_for_run


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
            {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]},
        )()


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


@pytest.fixture()
def demo_corpus(monkeypatch):
    monkeypatch.setenv(
        "APP_CORPUS", str(Path(__file__).parent / "fixtures" / "corpus.json")
    )


def _quoted_flow() -> list[dict]:
    return [
        {"action": "retrieve", "args": {"query": "lot 7 Sample Street account 99881"}, "reason": "find q3 bill"},
        {"action": "retrieve", "args": {"query": "lease recovery clause"}, "reason": "find lease"},
        {
            "action": "ledger_write",
            "args": {"entry_id": "lease-7-q3", "amount": 4049.18, "source_doc": "bill-q3-electricity"},
            "reason": "post quoted total",
        },
        {"action": "finish", "args": {}, "reason": "done"},
    ]


def test_health(isolated_db) -> None:
    client = TestClient(api.app)
    assert client.get("/health").json() == {"ok": True}


def test_ui_and_corpus_endpoints(isolated_db) -> None:
    client = TestClient(api.app)
    # JSON index
    res_json = client.get("/", headers={"accept": "application/json"})
    assert res_json.status_code == 200
    assert "service" in res_json.json()
    # HTML UI
    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "text/html" in res_ui.headers["content-type"]
    assert "Ledgerproof" in res_ui.text
    # Corpus
    res_corpus = client.get("/corpus")
    assert res_corpus.status_code == 200
    assert set(res_corpus.json()) == {
        "agl-better-bills-sample-bill-necf",
        "aer---better-bills---illustrative-example-bill---paginated",
        "origin-standing-bill-modified-alexcarter",
    }


def test_run_persists_quoted_write_and_replays(
    isolated_db, demo_corpus, monkeypatch
) -> None:
    monkeypatch.setattr(api, "TEST_CLIENT", _ScriptedClient(_quoted_flow()))
    with TestClient(api.app) as client:
        started = client.post("/run", json={"task": "recon", "max_steps": 6})
        assert started.status_code == 202, started.text
        run_id = started.json()["run_id"]
        assert started.json()["status"] == "running"

        settled = wait_for_run(client, run_id)
        assert settled["run"]["status"] == "done"

        replay = client.get(f"/runs/{run_id}")
        assert replay.status_code == 200
        events = replay.json()["events"]
        assert any("4049.18" in e["body"] for e in events)
        assert any(e["kind"] == "decision" for e in events)
        assert all(isinstance(e["data"], dict) for e in events)
        assert any(e["kind"] == "retrieve" and e["data"].get("doc_ids") for e in events)

        stream = client.get(f"/runs/{run_id}/stream")
        assert stream.status_code == 200
        assert "text/event-stream" in stream.headers["content-type"]
        assert "4049.18" in stream.text
        assert "id: 0" in stream.text

        ledger = client.get("/ledger").json()["entries"]
        assert any(e["entry_id"] == "lease-7-q3" for e in ledger)


def test_decide_unknown_ticket_404(isolated_db) -> None:
    client = TestClient(api.app)
    res = client.post("/approvals/t-9999/decide", json={"approved": True})
    assert res.status_code == 404


def test_decide_deny_marks_failed(isolated_db) -> None:
    run_id = store.create_run("deny-case")
    store.append_event(run_id, "retrieve", "retrieve [seed]: q3 | total 100.00")
    store.set_run_status(run_id, "awaiting_approval", 2)
    store.submit_approval("t-0007", run_id, "ledger_write", "gate", {"entry_id": "j-1", "amount": 100.0, "currency": "AUD", "source_doc": "q3"})
    client = TestClient(api.app)
    res = client.post("/approvals/t-0007/decide", json={"approved": False})
    assert res.status_code == 200
    assert res.json()["status"] == "failed"
    assert store.get_run(run_id)["status"] == "failed"

def test_reset_requires_confirm_and_archives(isolated_db) -> None:
    run_id = store.create_run("reset-case")
    store.insert_ledger_entry(run_id, "j-1", 10.0, "AUD", "bill")
    client = TestClient(api.app)

    assert client.post("/admin/reset", json={}).status_code == 400
    assert client.post("/admin/reset", json={"confirm": False}).status_code == 400
    assert len(client.get("/ledger").json()["entries"]) == 1

    res = client.post("/admin/reset", json={"confirm": True})
    assert res.status_code == 200
    body = res.json()
    assert body["tables"]["ledger"] == 0
    assert body["tables"]["runs"] == 0
    assert Path(body["archived"]).exists()
    assert client.get("/ledger").json()["entries"] == []
