from types import SimpleNamespace

import pytest

from agent.models import (
    ModelSpec,
    PoolExhausted,
    RotatingClient,
    parse_retry_seconds,
    reasoning_body,
)


def _spec(label: str, *, provider: str = "groq", key_env: str = "KEY_A", cap: int = 1200, json_mode: bool = False) -> ModelSpec:
    return ModelSpec(
        label=label,
        roles=("judge",),
        provider=provider,
        model=f"model-{label}",
        base_url="https://example.invalid/v1",
        api_key_env=key_env,
        max_output_tokens=cap,
        json_mode=json_mode,
    )


def _completion(content: str = '{"ok":true}'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class _Provider:
    """One fake provider client: records calls, plays a scripted result."""

    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(item, Exception):
            raise item
        return item


class _RateLimited(Exception):
    status_code = 429

    def __init__(self, hint: str = "try again in 2m") -> None:
        super().__init__(f"429 {hint}")


class _AuthError(Exception):
    status_code = 401

    def __init__(self) -> None:
        super().__init__("invalid api key")


def _client(specs: list[ModelSpec], providers: dict[str, _Provider], **kwargs) -> RotatingClient:
    return RotatingClient(
        specs,
        client_factory=lambda spec: providers[spec.label],
        **kwargs,
    )


def test_pool_rotates_and_cools_a_rate_limited_model() -> None:
    first = _Provider([_RateLimited()])
    second = _Provider([_completion()])
    client = _client([_spec("a"), _spec("b")], {"a": first, "b": second}, max_wait=0)

    assert client.chat.completions.create(model="auto", messages=[]) is not None
    assert client.last_label == "b"
    assert client.used == ["b"]
    assert len(first.calls) == 1

    client.chat.completions.create(model="auto", messages=[])
    assert len(first.calls) == 1  # still cooling, never touched again
    assert len(second.calls) == 2


def test_pool_caps_output_tokens_per_spec() -> None:
    limited = _Provider([_completion()])
    full = _Provider([_completion()])
    client = _client(
        [_spec("small", cap=900), _spec("large", cap=1200)],
        {"small": limited, "large": full},
    )

    client.chat.completions.create(model="small", messages=[], max_tokens=1200)
    client.chat.completions.create(model="large", messages=[], max_tokens=1200)
    assert limited.calls[0]["max_tokens"] == 900
    assert full.calls[0]["max_tokens"] == 1200


def test_pool_disables_a_provider_on_401_and_reports_exhaustion() -> None:
    a1 = _Provider([_AuthError()])
    a2 = _Provider([_completion()])
    b = _Provider([_completion(), _AuthError()])
    client = _client(
        [_spec("a1", key_env="KEY_A"), _spec("a2", key_env="KEY_A"), _spec("b", key_env="KEY_B")],
        {"a1": a1, "a2": a2, "b": b},
        max_wait=0,
    )

    client.chat.completions.create(model="auto", messages=[])
    assert client.last_label == "b"
    assert len(a2.calls) == 0  # same key as a1, disabled with it

    with pytest.raises(PoolExhausted) as exhausted:
        client.chat.completions.create(model="auto", messages=[])
    message = str(exhausted.value)
    assert "a1" in message and "a2" in message and "b" in message
    assert "KEY_A" in message and "KEY_B" in message


def test_pool_keeps_other_keys_on_the_same_provider_alive() -> None:
    rejected = _Provider([_AuthError()])
    spare = _Provider([_completion()])
    client = _client(
        [_spec("a1", key_env="KEY_A"), _spec("a2", key_env="KEY_B")],
        {"a1": rejected, "a2": spare},
        max_wait=0,
    )

    client.chat.completions.create(model="auto", messages=[])
    assert client.last_label == "a2"
    states = [row["state"] for row in client.snapshot()]
    assert states == ["disabled", "ready"]


def test_pool_parses_retry_hints() -> None:
    assert parse_retry_seconds("Please try again in 9m41.02s.") == pytest.approx(581.02)
    assert parse_retry_seconds("try again in 1h2m") == 3720.0
    assert parse_retry_seconds("retry in 45s") == 45.0
    assert parse_retry_seconds("no hint here") is None

    limited = _Provider([_RateLimited("rate limited, no wait given")])
    client = _client([_spec("a")], {"a": limited}, max_wait=0)
    with pytest.raises(PoolExhausted):
        client.chat.completions.create(model="auto", messages=[])
    retry_in = client.snapshot()[0]["retry_in"]
    assert retry_in > 55.0  # no hint falls back to a 60 s cooldown


def test_reasoning_shape_follows_provider() -> None:
    assert reasoning_body("groq", "off") == {"reasoning_effort": "low"}
    assert reasoning_body("groq", "none") == {"reasoning_effort": "low"}
    assert reasoning_body("groq", "low") == {"reasoning_effort": "low"}
    assert reasoning_body("google", "off") == {}
    assert reasoning_body("llm7", "low") == {}
    assert reasoning_body("groq", "") == {}
