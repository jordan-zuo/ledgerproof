import pytest

from agent.driver import ModelDecision, decide
from agent.states import AgentState, Evidence


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs) -> _FakeCompletion:
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] >= 200
        return _FakeCompletion(self._content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


def test_valid_decision_parses() -> None:
    client = _FakeClient('{"action": "retrieve", "args": {"query": "rent"}, "reason": "need clause"}')
    decision = decide(AgentState(task="t"), client, "test-model")
    assert decision == ModelDecision(action="retrieve", args={"query": "rent"}, reason="need clause")


def test_non_json_raises() -> None:
    with pytest.raises(ValueError, match="failed to route"):
        decide(AgentState(task="t"), _FakeClient("just do things"), "test-model")


def test_unknown_action_raises() -> None:
    with pytest.raises(ValueError, match="failed to route"):
        decide(AgentState(task="t"), _FakeClient('{"action": "teleport"}'), "test-model")


class _FlakyClient:
    """Fails once with an invented action, then routes correctly."""

    def __init__(self) -> None:
        self.calls = 0
        self.chat = type(
            "Chat",
            (),
            {"completions": self},
        )()

    def create(self, **kwargs) -> _FakeCompletion:
        self.calls += 1
        if self.calls == 1:
            return _FakeCompletion('{"action": "fetch_lease_clauses"}')
        return _FakeCompletion('{"action": "retrieve", "args": {"query": "rent"}}')


def test_retry_recovers_after_invented_action() -> None:
    client = _FlakyClient()
    decision = decide(AgentState(task="t"), client, "test-model")
    assert decision.action == "retrieve"
    assert client.calls == 2


class _ToolCallRejected(Exception):
    status_code = 400

    def __str__(self) -> str:
        return "tool_choice is none, but model called a tool"


class _ToolCallClient(_FlakyClient):
    """Raises a provider tool-call rejection once, then routes correctly."""

    def __init__(self) -> None:
        super().__init__()
        self.rotations = 0

    def rotate_after_bad_answer(self, reason: str = "") -> bool:
        self.rotations += 1
        return True

    def create(self, **kwargs) -> _FakeCompletion:
        self.calls += 1
        if self.calls == 1:
            raise _ToolCallRejected()
        return _FakeCompletion('{"action": "retrieve", "args": {"query": "rent"}}')


def test_tool_call_rejection_rotates_before_retry() -> None:
    client = _ToolCallClient()
    decision = decide(AgentState(task="t"), client, "test-model")
    assert decision.action == "retrieve"
    assert client.calls == 2
    assert client.rotations == 1


def test_prompt_carries_tool_catalog() -> None:
    from agent.driver import _prompt_for

    prompt = _prompt_for(AgentState(task="t"), context="Documents [lease-7]. Ledger rules: positive.")
    assert "lease-7" in prompt
    assert "never SQL" in prompt
    assert "still call ledger_write" in prompt


def test_prompt_collapses_repeated_retrieves_and_lists_retrieved_docs() -> None:
    from agent.driver import _prompt_for

    state = AgentState(task="Which bill has the highest solar export?")
    state.evidence.append(Evidence(doc_id="doc-a", text="full", query="solar", snippet="snip-a"))
    state.history.extend(
        [
            "retrieve [solar]: doc-a | snip-a",
            "retrieve [solar]: doc-a | snip-a",
            "not finished: this retrieve already ran",
        ]
    )
    prompt = _prompt_for(state)
    assert prompt.count("retrieve [solar]") == 1
    assert "Documents already retrieved: doc-a" in prompt
