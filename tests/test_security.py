"""Abuse controls and output-injection guards for the public demo."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import scripts.api as api
from agent import limits, store
from agent.tools import LedgerArgs, RetrievalArgs


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


def test_rate_limiter_uses_a_sliding_window() -> None:
    limiter = limits.RateLimiter(2, 10.0)
    assert limiter.allow("client", now=0.0)
    assert limiter.allow("client", now=1.0)
    assert not limiter.allow("client", now=2.0)
    assert limiter.allow("client", now=11.0)
    assert limiter.allow("other", now=2.0)


def test_stream_counter_caps_and_releases() -> None:
    counter = limits.ActiveCounter(1)
    assert counter.acquire()
    assert not counter.acquire()
    counter.release()
    assert counter.acquire()
    counter.reset()


def test_run_and_tool_inputs_have_hard_length_ceilings(isolated_db) -> None:
    client = TestClient(api.app)
    assert client.post("/run", json={"task": "x" * 2001}).status_code == 422
    assert (
        client.post(
            "/approvals/t-0000/decide",
            json={"approved": True, "decided_by": "x" * 121},
        ).status_code
        == 422
    )
    with pytest.raises(ValidationError):
        RetrievalArgs(query="x" * 501)
    with pytest.raises(ValidationError):
        LedgerArgs(entry_id="e", amount=1.0, currency="AUD", source_doc="x" * 201)


def test_run_submissions_are_throttled_per_client(isolated_db) -> None:
    client = TestClient(api.app)
    for _ in range(limits.RUNS_PER_WINDOW):
        assert client.post("/run", json={"task": "reconcile"}).status_code == 202
    assert client.post("/run", json={"task": "reconcile"}).status_code == 429
    other = client.post(
        "/run",
        json={"task": "reconcile"},
        headers={"X-Forwarded-For": "203.0.113.9"},
    )
    assert other.status_code == 202


def test_run_queue_is_capped(isolated_db, monkeypatch) -> None:
    monkeypatch.setattr(api, "MAX_ACTIVE_RUNS", 2)
    store.create_run("one", status="pending")
    store.create_run("two", status="pending")
    res = TestClient(api.app).post("/run", json={"task": "three"})
    assert res.status_code == 429


def test_reset_refuses_to_run_over_live_work(isolated_db) -> None:
    run_id = store.create_run("queued", status="pending")
    client = TestClient(api.app)
    assert client.post("/admin/reset", json={"confirm": True}).status_code == 409
    store.set_run_status(run_id, "done", 0)
    assert client.post("/admin/reset", json={"confirm": True}).status_code == 200


def test_reset_keeps_only_the_newest_archives(isolated_db, tmp_path) -> None:
    for _ in range(store.KEEP_ARCHIVES + 2):
        store.archive_and_clear()
    snapshots = sorted((tmp_path / "archive").glob("t-*.db"))
    assert len(snapshots) == store.KEEP_ARCHIVES


def test_csv_export_neutralizes_formula_cells(isolated_db) -> None:
    run_id = store.create_run("seed")
    store.insert_ledger_entry(run_id, "e-1", 10.0, "AUD", "bill")
    store.reverse_ledger_entry(1, "=cmd", "tester")
    text = TestClient(api.app).get("/export/ledger.csv").text
    assert "'=cmd" in text
    # A negative amount is a number, not a formula, and must stay numeric.
    assert ",-10.00," in text


def test_oversized_request_bodies_are_refused(isolated_db) -> None:
    res = TestClient(api.app).post(
        "/run", json={"task": "x" * (api.MAX_REQUEST_BYTES + 1)}
    )
    assert res.status_code == 413


def test_responses_set_browser_hardening_headers(isolated_db) -> None:
    res = TestClient(api.app).get("/health")
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in res.headers["content-security-policy"]


def test_storage_does_not_publish_an_absolute_path(isolated_db) -> None:
    body = TestClient(api.app).get("/storage").json()
    assert body["database"] == "t.db"
    assert "/" not in body["database"] and "\\" not in body["database"]
