"""Abuse limits for the public demo.

The open demo has no accounts, so per-client throttles and a global cap are
the only things standing between a script and the model quota. State lives
in this process on purpose: a restart clears it, which is fine for a demo.

A caller can spoof ``X-Forwarded-For``, so per-client limits are a speed
bump, not a wall. The global run cap and queue cap are the hard bound.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import Request

RUNS_PER_WINDOW = 10
RUN_WINDOW_SECONDS = 600.0
GLOBAL_RUNS_PER_WINDOW = 30
RESETS_PER_WINDOW = 3
RESET_WINDOW_SECONDS = 3600.0
INGESTS_PER_WINDOW = 5
INGEST_WINDOW_SECONDS = 600.0

MAX_ACTIVE_RUNS = 12
MAX_EVENT_STREAMS = 16

# Spoofed client keys must not grow the dict without bound.
_MAX_TRACKED_KEYS = 5_000

_LIMITERS: list["RateLimiter"] = []
_COUNTERS: list["ActiveCounter"] = []


class RateLimiter:
    """Sliding-window allowance for one kind of action."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = float(window_seconds)
        self._hits: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        _LIMITERS.append(self)

    def allow(self, key: str, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        cutoff = moment - self.window
        with self._lock:
            self._prune(cutoff)
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(moment)
            return True

    def _prune(self, cutoff: float) -> None:
        if len(self._hits) <= _MAX_TRACKED_KEYS:
            return
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


class ActiveCounter:
    """A cap on things that stay open, such as event streams."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._active = 0
        self._lock = threading.Lock()
        _COUNTERS.append(self)

    def acquire(self) -> bool:
        with self._lock:
            if self._active >= self.capacity:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active:
                self._active -= 1

    def reset(self) -> None:
        with self._lock:
            self._active = 0


def client_key(request: Request) -> str:
    """Identify the caller, preferring the address our proxy saw.

    Managed proxies append the peer address to ``X-Forwarded-For``, so the
    rightmost entry is the one we can trust; anything to its left is caller
    supplied. Without the header, fall back to the socket peer.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        candidate = forwarded.split(",")[-1].strip()
        if candidate:
            return candidate
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def clear_all() -> None:
    """Test hook: start every limit from zero."""
    for limiter in _LIMITERS:
        limiter.clear()
    for counter in _COUNTERS:
        counter.reset()
