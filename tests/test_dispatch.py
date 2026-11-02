"""Durable run ownership: exclusive claims, leases, and restart recovery."""

from __future__ import annotations

import time

import pytest

from agent import store
from agent.dispatch import RunDispatcher

OLD = "2000-01-01T00:00:00+00:00"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


def _age_heartbeat(run_id: str) -> None:
    conn = store.connect()
    try:
        conn.execute(
            "UPDATE runs SET heartbeat_at = ? WHERE run_id = ?", (OLD, run_id)
        )
        conn.commit()
    finally:
        conn.close()


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def test_claim_is_exclusive_and_an_expired_lease_is_takeover(isolated_db) -> None:
    run_id = store.create_run("t", status="pending", max_steps=4)

    first = store.claim_run("worker-a", lease_seconds=30)
    assert first == {
        "run_id": run_id,
        "task": "t",
        "max_steps": 4,
        "attempts": 1,
    }
    # A second worker cannot take a job that is still heartbeating.
    assert store.claim_run("worker-b", lease_seconds=30) is None
    assert store.heartbeat_run(run_id, "worker-a") is True
    assert store.heartbeat_run(run_id, "worker-b") is False

    # The owner is gone and the lease lapsed: another worker may resume it.
    _age_heartbeat(run_id)
    second = store.claim_run("worker-b", lease_seconds=30)
    assert second["run_id"] == run_id
    assert second["attempts"] == 2

    # Rows from before claims existed are never grabbed by accident.
    legacy = store.create_run("legacy")
    assert store.claim_run("worker-c", lease_seconds=30) is None
    assert store.get_run(legacy)["status"] == "running"


def test_a_queued_run_survives_until_a_worker_starts(isolated_db) -> None:
    run_id = store.create_run("t", status="pending", max_steps=3)
    seen: list[int] = []

    def execute(run: dict) -> None:
        seen.append(run["attempts"])
        store.set_run_status(run["run_id"], "done", 1)

    worker = RunDispatcher(execute, workers=1, poll_seconds=0.02)
    worker.start()
    try:
        _wait_for(lambda: bool(seen))
    finally:
        worker.stop()

    assert seen == [1]
    assert store.get_run(run_id)["status"] == "done"


def test_an_expired_claim_is_resumed_not_run_twice(isolated_db) -> None:
    run_id = store.create_run("t", status="pending", max_steps=3)
    assert store.claim_run("dead-worker") is not None
    _age_heartbeat(run_id)

    seen: list[int] = []

    def execute(run: dict) -> None:
        seen.append(run["attempts"])
        store.set_run_status(run["run_id"], "done", 1)

    worker = RunDispatcher(execute, workers=1, poll_seconds=0.02)
    worker.start()
    try:
        _wait_for(lambda: bool(seen))
    finally:
        worker.stop()

    assert seen == [2]
    assert store.get_run(run_id)["status"] == "done"


def test_retry_cap_fails_instead_of_relooping(isolated_db) -> None:
    run_id = store.create_run("t", status="pending", max_steps=3)
    for owner in ("a", "b", "c"):
        assert store.claim_run(owner, max_attempts=3) is not None
        _age_heartbeat(run_id)

    assert store.claim_run("d", max_attempts=3) is None
    assert store.reap_exhausted(max_attempts=3) == [run_id]
    assert store.get_run(run_id)["status"] == "failed"


def test_unclaimed_running_rows_are_failed_not_left_live(isolated_db) -> None:
    legacy = store.create_run("legacy")
    claimed = store.create_run("claimed", status="pending")
    queued = store.create_run("queued", status="pending")
    assert store.claim_run("live-worker") is not None

    assert store.fail_unclaimed_runs() == [legacy]
    assert store.get_run(legacy)["status"] == "failed"
    assert store.get_run(claimed)["status"] == "running"
    assert store.get_run(queued)["status"] == "pending"
