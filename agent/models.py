"""Model pool with failover, cooldowns, and per-model limits.

A role (agent or judge) gets an ordered list of specs from config/models.json.
The first healthy model answers every call until it fails; a rate limit or an
unavailable model moves the next call down the list. Every wait is bounded, so
an exhausted pool raises instead of hanging.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOL_PATH = ROOT / "config" / "models.json"
DEFAULT_MAX_WAIT_SECONDS = 90.0
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 1200

PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "reasoning": "effort",
    },
    "llm7": {
        "base_url": "https://api.llm7.io/v1",
        "reasoning": "omit",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "reasoning": "omit",
    },
}

_RETRY_HINT = re.compile(
    r"(?:try again|retry) in\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?",
    re.IGNORECASE,
)


class PoolError(RuntimeError):
    """The pool cannot serve this role."""


class PoolExhausted(PoolError):
    """Every model for the role is cooling, disabled, or out of keys."""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    roles: tuple[str, ...]
    provider: str
    model: str
    base_url: str
    api_key_env: str
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    json_mode: bool = False


def pool_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("MODEL_POOL", "").strip() or DEFAULT_POOL_PATH)


def load_pool(path: str | Path | None = None) -> list[ModelSpec]:
    """Read and validate the tracked pool file. Never contains secrets."""
    source = pool_path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PoolError(f"model pool not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise PoolError(f"{source} is not valid JSON: {exc}") from exc
    entries = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise PoolError(f"{source} needs a non-empty 'models' list")

    specs: list[ModelSpec] = []
    labels: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PoolError(f"{source}: model {index} is not an object")
        label = str(entry.get("label") or "").strip()
        if not label:
            raise PoolError(f"{source}: model {index} has no label")
        if label in labels:
            raise PoolError(f"{source}: duplicate label {label!r}")
        labels.add(label)
        roles = entry.get("roles") or []
        if not isinstance(roles, list) or not roles:
            raise PoolError(f"{source}: {label} has no roles")
        provider = str(entry.get("provider") or "").strip()
        if provider not in PROVIDERS:
            raise PoolError(
                f"{source}: {label} has unknown provider {provider!r}; "
                f"use one of {', '.join(sorted(PROVIDERS))}"
            )
        model = str(entry.get("model") or "").strip()
        if not model:
            raise PoolError(f"{source}: {label} has no model")
        api_key_env = str(entry.get("api_key_env") or "").strip()
        if not api_key_env:
            raise PoolError(f"{source}: {label} has no api_key_env")
        base_url = str(entry.get("base_url") or PROVIDERS[provider]["base_url"]).strip()
        cap = int(entry.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
        if cap <= 0:
            raise PoolError(f"{source}: {label} has an invalid max_output_tokens")
        specs.append(
            ModelSpec(
                label=label,
                roles=tuple(str(role) for role in roles),
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                max_output_tokens=cap,
                json_mode=bool(entry.get("json_mode", False)),
            )
        )
    return specs


def specs_for(role: str, path: str | Path | None = None) -> list[ModelSpec]:
    """Return the specs for a role whose key variables are set."""
    source = pool_path(path)
    pool = [spec for spec in load_pool(source) if role in spec.roles]
    if not pool:
        raise PoolError(f"no model in {source} serves role {role!r}")
    usable = [spec for spec in pool if os.environ.get(spec.api_key_env, "").strip()]
    if usable:
        return usable
    needed = sorted({spec.api_key_env for spec in pool})
    raise PoolError(
        f"no usable model for role {role!r}: set {' or '.join(needed)} in .env"
    )


def describe_pool(role: str | None = None, path: str | Path | None = None) -> list[dict[str, Any]]:
    """Config summary for dry runs. No network, no secrets."""
    rows = []
    for spec in load_pool(path):
        if role and role not in spec.roles:
            continue
        rows.append(
            {
                "label": spec.label,
                "roles": ",".join(spec.roles),
                "provider": spec.provider,
                "model": spec.model,
                "max_output_tokens": spec.max_output_tokens,
                "key_env": spec.api_key_env,
                "key_set": bool(os.environ.get(spec.api_key_env, "").strip()),
            }
        )
    return rows


def reasoning_body(provider: str, mode: str | None) -> dict[str, Any]:
    """Provider shape for a reasoning request. Empty means send nothing."""
    value = (mode or "").strip().lower()
    if not value:
        return {}
    if PROVIDERS.get(provider, {}).get("reasoning") == "omit":
        return {}
    off = value in {"off", "none", "false", "0"}
    # Groq accepts only low, medium, or high, so off means its minimum.
    return {"reasoning_effort": "low" if off else value}


def parse_retry_seconds(text: str) -> float | None:
    """Pull a provider's retry hint out of an error message."""
    match = _RETRY_HINT.search(text or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    if not any((hours, minutes, seconds)):
        return None
    total = 0.0
    if hours:
        total += int(hours) * 3600
    if minutes:
        total += int(minutes) * 60
    if seconds:
        total += float(seconds)
    return total or None


def _default_client(spec: ModelSpec) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ.get(spec.api_key_env, ""),
        base_url=spec.base_url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _has_content(completion: Any) -> bool:
    choices = getattr(completion, "choices", None)
    if not choices:
        return False
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return bool(content and content.strip())


class _Completions:
    def __init__(self, client: "RotatingClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create(**kwargs)


class RotatingClient:
    """OpenAI-shaped facade over an ordered model pool."""

    def __init__(
        self,
        specs: list[ModelSpec],
        client_factory: Callable[[ModelSpec], Any] | None = None,
        max_wait: float | None = None,
    ) -> None:
        if not specs:
            raise PoolError("model pool is empty")
        labels = [spec.label for spec in specs]
        if len(set(labels)) != len(labels):
            raise PoolError("model pool has duplicate labels")
        self._specs = tuple(specs)
        self._by_label = {spec.label: spec for spec in specs}
        self._factory = client_factory or _default_client
        self._clients: dict[str, Any] = {}
        self._cooldown: dict[str, float] = {}
        self._disabled: dict[str, str] = {}
        self._last_error: dict[str, str] = {}
        self._current: ModelSpec | None = None
        self._last_failed: ModelSpec | None = None
        self._lock = threading.RLock()
        raw_wait = os.environ.get("MODEL_POOL_MAX_WAIT", "").strip()
        if raw_wait:
            max_wait = float(raw_wait)
        self.max_wait = DEFAULT_MAX_WAIT_SECONDS if max_wait is None else max_wait
        self.last_label: str | None = None
        self.used: list[str] = []
        self.attempts: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=_Completions(self))

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return self._specs

    def provider_client(self, spec: ModelSpec) -> Any:
        with self._lock:
            if spec.label not in self._clients:
                self._clients[spec.label] = self._factory(spec)
            return self._clients[spec.label]

    def disable(self, label: str, reason: str) -> None:
        with self._lock:
            self._disabled[label] = reason
            self._last_error[label] = reason
            if self._current is not None and self._current.label == label:
                self._current = None

    def disable_provider(self, api_key_env: str, reason: str) -> None:
        with self._lock:
            for spec in self._specs:
                if spec.api_key_env == api_key_env:
                    self._disabled[spec.label] = reason
                    self._last_error[spec.label] = reason
            if self._current is not None and self._current.api_key_env == api_key_env:
                self._current = None

    def available(self) -> list[ModelSpec]:
        now = time.monotonic()
        with self._lock:
            return [
                spec
                for spec in self._specs
                if spec.label not in self._disabled
                and self._cooldown.get(spec.label, 0.0) <= now
            ]

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        rows = []
        with self._lock:
            for spec in self._specs:
                remaining = max(0.0, self._cooldown.get(spec.label, 0.0) - now)
                if spec.label in self._disabled:
                    state = "disabled"
                elif remaining > 0:
                    state = "cooling"
                else:
                    state = "ready"
                rows.append(
                    {
                        "label": spec.label,
                        "provider": spec.provider,
                        "model": spec.model,
                        "key_env": spec.api_key_env,
                        "state": state,
                        "retry_in": round(remaining, 1),
                        "last_error": self._last_error.get(spec.label, ""),
                    }
                )
        return rows

    def _order(self, model: str) -> list[ModelSpec]:
        if model and model != "auto":
            spec = self._by_label.get(model)
            if spec is None:
                raise PoolError(f"unknown model label: {model!r}")
            return [spec]
        available = self.available()
        if self._current is not None and self._current in available:
            return [self._current] + [spec for spec in available if spec is not self._current]
        return available

    def _create(
        self,
        *,
        model: str = "auto",
        messages: list[dict[str, str]],
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        reasoning: str | None = None,
        response_format: dict[str, str] | None = None,
    ) -> Any:
        waited = False
        while True:
            for spec in self._order(model):
                if spec.label in self._disabled:
                    continue
                kwargs: dict[str, Any] = {
                    "model": spec.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": min(int(max_tokens), spec.max_output_tokens),
                }
                body = reasoning_body(spec.provider, reasoning)
                if body:
                    kwargs["extra_body"] = body
                if response_format is not None and spec.json_mode:
                    kwargs["response_format"] = response_format
                try:
                    completion = self.provider_client(spec).chat.completions.create(**kwargs)
                except PoolError:
                    raise
                except Exception as exc:
                    if self._record_failure(spec, exc):
                        with self._lock:
                            self._last_failed = spec
                        raise
                    continue
                if not _has_content(completion):
                    self._cool(spec, 30.0, "empty response")
                    continue
                with self._lock:
                    self._current = spec
                    self._last_failed = None
                    self.last_label = spec.label
                    if not self.used or self.used[-1] != spec.label:
                        self.used.append(spec.label)
                return completion

            remaining = self._min_remaining(model)
            if remaining is not None and 0 < remaining <= self.max_wait and not waited:
                # One bounded wait per call. Never spin on an exhausted pool.
                time.sleep(remaining + 0.05)
                waited = True
                continue
            raise self._exhausted(model, remaining)

    def _record_failure(self, spec: ModelSpec, exc: Exception) -> bool:
        """Classify a failure. True means the caller must see it unchanged."""
        status = getattr(exc, "status_code", None)
        text = str(exc)
        low = text.lower()
        reason = f"{type(exc).__name__}: {text[:160]}"
        if status == 401 or "invalid_api_key" in low or "invalid api key" in low:
            self.disable_provider(spec.api_key_env, f"key rejected (401) at {spec.base_url}")
            return False
        if status in (402, 403, 404) or "model_not_found" in low:
            self.disable(spec.label, f"unavailable ({status or 'error'}): {text[:120]}")
            return False
        if status == 429 or "429" in low or "rate limit" in low:
            self._cool(spec, parse_retry_seconds(text) or 60.0, "rate limited")
            return False
        if status == 400:
            # A rejected request is the caller's to handle; retrying another
            # model would hide a prompt or parameter bug.
            return True
        self._cool(spec, 30.0, reason)
        return False

    def rotate_after_bad_answer(self, reason: str = "bad answer") -> bool:
        """Cool the model that just failed when another one can take the retry."""
        now = time.monotonic()
        with self._lock:
            failing = self._last_failed or self._current
            if failing is None:
                return False
            alternative = any(
                spec is not failing
                and spec.label not in self._disabled
                and self._cooldown.get(spec.label, 0.0) <= now
                for spec in self._specs
            )
        if not alternative:
            return False
        self._cool(failing, 30.0, reason)
        return True

    def _cool(self, spec: ModelSpec, seconds: float, reason: str) -> None:
        with self._lock:
            self._cooldown[spec.label] = time.monotonic() + max(seconds, 1.0)
            self._last_error[spec.label] = reason
            self.attempts.append(
                {
                    "label": spec.label,
                    "reason": reason,
                    "cooldown": round(seconds, 1),
                    "at": round(time.time(), 3),
                }
            )
            if self._current is not None and self._current.label == spec.label:
                self._current = None

    def _min_remaining(self, model: str) -> float | None:
        if model and model != "auto":
            spec = self._by_label[model]
            if spec.label in self._disabled:
                return None
            remaining = self._cooldown.get(spec.label, 0.0) - time.monotonic()
            return remaining if remaining > 0 else None
        with self._lock:
            candidates = [
                spec
                for spec in self._specs
                if spec.label not in self._disabled
            ]
            times = [
                self._cooldown.get(spec.label, 0.0) - time.monotonic()
                for spec in candidates
            ]
        times = [value for value in times if value > 0]
        return min(times) if times else None

    def _exhausted(self, model: str, remaining: float | None) -> PoolExhausted:
        rows = self.snapshot()
        if model and model != "auto":
            rows = [row for row in rows if row["label"] == model]
        details = "; ".join(
            f"{row['label']} via {row['key_env']} ({row['state']}"
            + (f", {row['retry_in']}s" if row["state"] == "cooling" else "")
            + (f", {row['last_error']}" if row["last_error"] else "")
            + ")"
            for row in rows
        )
        tail = f" Earliest retry in {round(remaining, 1)}s." if remaining else ""
        return PoolExhausted(f"no model can answer right now: {details}.{tail}")
