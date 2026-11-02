"""Database-backed run queue.

A worker claims one run at a time by holding a lease. Every process that
points at the same database can serve the queue, so a restart or a second
worker does not strand work. A lost worker's lease expires and another
worker resumes the run from step zero; the ledger's bill identity guard
keeps a resumed run from posting the same bill twice.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from typing import Any, Callable

from agent import store

LEASE_SECONDS = 30.0
HEARTBEAT_SECONDS = 5.0
POLL_SECONDS = 0.5
MAX_ATTEMPTS = 3


class DurableCancel:
    """Cooperative stop signal backed by the database.

    The local event makes a same-process cancel immediate. The database flag
    is what lets a cancel reach a worker in a different process.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._local = threading.Event()

    def set(self) -> None:
        self._local.set()

    def is_set(self) -> bool:
        if self._local.is_set():
            return True
        try:
            if store.cancel_requested(self._run_id):
                self._local.set()
        except Exception:
            # A transient read failure must not look like a cancel.
            return False
        return self._local.is_set()


class RunDispatcher:
    """Poll the runs table, claim one job per worker, heartbeat while it runs."""

    def __init__(
        self,
        execute: Callable[[dict[str, Any]], None],
        workers: int = 4,
        lease_seconds: float = LEASE_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        poll_seconds: float = POLL_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._execute = execute
        self._workers = max(1, workers)
        self._lease = lease_seconds
        self._heartbeat = heartbeat_seconds
        self._poll = poll_seconds
        self._max_attempts = max_attempts
        self._prefix = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._guard = threading.Lock()
        self._wake = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    @property
    def running(self) -> bool:
        return self._started

    def start(self) -> None:
        with self._guard:
            if self._started:
                return
            self._stop.clear()
            self._started = True
            for index in range(self._workers):
                owner = f"{self._prefix}:{index}"
                thread = threading.Thread(
                    target=self._loop,
                    args=(owner,),
                    name=f"run-worker-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def stop(self) -> None:
        with self._guard:
            if not self._started:
                return
            self._started = False
            self._stop.set()
            threads = list(self._threads)
            self._threads.clear()
        self.wake()
        for thread in threads:
            thread.join(timeout=5)

    def wake(self) -> None:
        with self._wake:
            self._wake.notify_all()

    def _pause(self) -> None:
        with self._wake:
            if not self._stop.is_set():
                self._wake.wait(self._poll)

    def _loop(self, owner: str) -> None:
        last_reap = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_reap >= self._lease:
                last_reap = now
                self._reap()
            try:
                run = store.claim_run(
                    owner,
                    lease_seconds=self._lease,
                    max_attempts=self._max_attempts,
                )
            except Exception:
                run = None
            if run is None:
                self._pause()
                continue
            self._run_once(owner, run)

    def _reap(self) -> None:
        try:
            lost = store.reap_exhausted(
                lease_seconds=self._lease,
                max_attempts=self._max_attempts,
            )
        except Exception:
            return
        for run_id in lost:
            try:
                store.append_event(
                    run_id,
                    "error",
                    "failed: the worker was lost and all retries were used",
                    {
                        "reason": "worker lost and retries exhausted",
                        "attempts": self._max_attempts,
                    },
                )
            except Exception:
                pass

    def _run_once(self, owner: str, run: dict[str, Any]) -> None:
        run_id = run["run_id"]
        done = threading.Event()
        beat = threading.Thread(
            target=self._beat,
            args=(run_id, owner, done),
            name=f"run-heartbeat-{run_id}",
            daemon=True,
        )
        beat.start()
        try:
            self._execute(run)
        except Exception as exc:
            # A worker must not die with the run left looking alive.
            message = f"{type(exc).__name__}: {exc}"
            try:
                current = store.get_run(run_id) or {}
                store.append_event(run_id, "error", message, {"reason": message})
                store.set_run_status(
                    run_id, "failed", int(current.get("step_count", 0))
                )
            except Exception:
                pass
        finally:
            done.set()
            beat.join(timeout=1)
            try:
                store.release_run(run_id, owner)
            except Exception:
                pass

    def _beat(self, run_id: str, owner: str, done: threading.Event) -> None:
        while not done.wait(self._heartbeat):
            try:
                if not store.heartbeat_run(run_id, owner):
                    return
            except Exception:
                return
