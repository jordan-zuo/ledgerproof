import pytest

from agent.approvals import ApprovalQueue
from agent.runner import run_with_model
from agent.states import AgentState, Evidence
from agent.tools import LedgerWriteTool, RetrievalTool


class _ScriptedClient:
    """Replays canned decisions in order, then says finish."""

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        if self._script:
            payload = self._script.pop(0)
        else:
            payload = {"action": "finish", "args": {}, "reason": "nothing left"}
        import json

        content = json.dumps(payload)
        return type(
            "Completion",
            (),
            {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]},
        )()


DOCS = {"lease-7": "recovery clause submetered electricity one hundred percent"}


class _SilentAfterScriptClient(_ScriptedClient):
    """Returns empty model content once the script is exhausted."""

    def create(self, **kwargs):
        if self._script:
            return super().create(**kwargs)
        return type(
            "Completion",
            (),
            {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": ""})()})()]},
        )()


def test_model_run_completes_retrieve_then_finish() -> None:
    client = _ScriptedClient(
        [
            {"action": "retrieve", "args": {"query": "recovery clause"}, "reason": "find clause"},
            {"action": "finish", "args": {}, "reason": "have what is needed"},
        ]
    )
    state = run_with_model(
        AgentState(task="t", max_steps=5),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert any("lease-7" in event for event in state.history)


def test_posting_on_the_last_allowed_step_still_completes() -> None:
    docs = {"bill-1": "amount due 10.00 dollars"}
    client = _ScriptedClient(
        [
            {"action": "retrieve", "args": {"query": "amount due"}, "reason": "find bill"},
            {
                "action": "ledger_write",
                "args": {"entry_id": "j-1", "amount": 10.0, "source_doc": "bill-1"},
                "reason": "post",
            },
        ]
    )
    state = run_with_model(
        AgentState(task="post the bill", max_steps=2),
        client,
        "test",
        RetrievalTool(docs),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert state.posted
    assert any(
        "step ceiling reached after the required posting" in event
        for event in state.history
    )


def test_posting_survives_a_provider_that_stops_answering() -> None:
    """A write that landed must not be reported as a failure."""
    client = _SilentAfterScriptClient(
        [
            {
                "action": "ledger_write",
                "args": {"entry_id": "sam", "amount": 10.0, "source_doc": "d"},
                "reason": "post it",
            }
        ]
    )
    state = run_with_model(
        AgentState(
            task="post the bill",
            max_steps=6,
            evidence=[Evidence(doc_id="d", text="total 10.00")],
        ),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert state.posted
    assert any("posting landed" in event for event in state.history)


def test_model_run_pauses_for_big_ledger_write() -> None:
    client = _ScriptedClient(
        [{"action": "ledger_write", "args": {"entry_id": "j-9", "amount": 99999, "source_doc": "d"}, "reason": "post"}]
    )
    seeded = AgentState(
        task="t",
        max_steps=5,
        evidence=[Evidence(doc_id="d", text="total 99999.00")],
    )
    state = run_with_model(
        seeded,
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "awaiting_approval"


def test_repeated_write_fails_fast_with_reason() -> None:
    client = _ScriptedClient(
        [
            {
                "action": "ledger_write",
                "args": {"entry_id": "j-1", "amount": 10.0, "source_doc": "lease-7"},
                "reason": "loop",
            }
        ]
        * 10
    )
    state = run_with_model(
        AgentState(task="t", max_steps=10),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "failed"
    assert "repeated" in state.failure_reason
    assert state.step_count < 10


def test_second_identical_retrieve_is_nudged_and_the_run_can_still_finish() -> None:
    """One repeated read cannot add evidence; the nudge must not kill the run."""
    client = _ScriptedClient(
        [
            {"action": "retrieve", "args": {"query": "recovery clause"}, "reason": "look"},
            {"action": "retrieve", "args": {"query": "recovery clause"}, "reason": "look again"},
            {"action": "finish", "args": {}, "reason": "have what is needed"},
        ]
    )
    state = run_with_model(
        AgentState(task="t", max_steps=5),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert state.step_count == 3
    assert len(state.evidence) == 1
    assert any(line.startswith("not finished:") for line in state.history)


def test_second_identical_retrieve_is_nudged_then_a_third_fails() -> None:
    """A stuck loop should stop after one correction, not burn the ceiling."""
    client = _ScriptedClient(
        [{"action": "retrieve", "args": {"query": "recovery clause"}, "reason": "look"}] * 4
    )
    state = run_with_model(
        AgentState(task="t", max_steps=10),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "failed"
    assert "repeated ineffective retrieve" in state.failure_reason
    nudge = next(line for line in state.history if line.startswith("not finished:"))
    assert "lease-7" in nudge
    assert state.step_count == 3


def test_changing_top_k_is_not_a_repeated_retrieve() -> None:
    """Widening the same query is an expansion, not a loop."""
    client = _ScriptedClient(
        [
            {"action": "retrieve", "args": {"query": "recovery clause", "top_k": 1}, "reason": "look"},
            {"action": "retrieve", "args": {"query": "recovery clause", "top_k": 4}, "reason": "widen"},
            {"action": "finish", "args": {}, "reason": "have what is needed"},
        ]
    )
    state = run_with_model(
        AgentState(task="t", max_steps=5),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert state.step_count == 3
    assert len(state.evidence) == 2


def test_finish_without_posting_is_not_done() -> None:
    """Saying 'done' after naming an amount, with nothing posted, is a lie."""
    client = _ScriptedClient(
        [
            {"action": "retrieve", "args": {"query": "recovery clause"}, "reason": "look"},
            {"action": "finish", "args": {}, "reason": "Sam owes 63.70"},
        ]
    )
    state = run_with_model(
        AgentState(task="post the bill with the lowest solar export", max_steps=6),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "failed"
    assert "none was made" in state.failure_reason
    assert any(line.startswith("not finished:") for line in state.history)


def test_finish_after_a_posting_is_done() -> None:
    client = _ScriptedClient(
        [
            {
                "action": "ledger_write",
                "args": {"entry_id": "sam", "amount": 10.0, "source_doc": "d"},
                "reason": "post it",
            },
            {"action": "finish", "args": {}, "reason": "posted"},
        ]
    )
    ledger = LedgerWriteTool()
    state = run_with_model(
        AgentState(
            task="post the bill",
            max_steps=6,
            evidence=[Evidence(doc_id="d", text="total 10.00")],
        ),
        client,
        "test",
        RetrievalTool(DOCS),
        ledger,
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert len(ledger.entries) == 1


def test_finish_after_an_already_posted_bill_is_done() -> None:
    """Nothing left to post is a settled outcome, not a failure."""
    client = _ScriptedClient(
        [
            {
                "action": "ledger_write",
                "args": {"entry_id": "sam", "amount": 10.0, "source_doc": "d"},
                "reason": "post it",
            },
            {"action": "finish", "args": {}, "reason": "already posted"},
        ]
    )
    ledger = LedgerWriteTool(posted=lambda args: {"entry_id": "x", "run_id": "r-1"})
    state = run_with_model(
        AgentState(
            task="post the bill",
            max_steps=6,
            evidence=[Evidence(doc_id="d", text="total 10.00")],
        ),
        client,
        "test",
        RetrievalTool(DOCS),
        ledger,
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert ledger.entries == []


def test_duplicate_of_another_bill_does_not_settle_the_task() -> None:
    """A posted row for a different bill is not the posting the task asked for."""
    client = _ScriptedClient(
        [
            {
                "action": "ledger_write",
                "args": {"entry_id": "sam", "amount": 10.0, "source_doc": "d"},
                "reason": "post it",
            },
        ]
    )
    ledger = LedgerWriteTool(posted=lambda args: {"entry_id": "x", "run_id": "r-1"})
    state = run_with_model(
        AgentState(
            task="post the bill",
            max_steps=6,
            evidence=[
                Evidence(doc_id="d", text="total 10.00"),
                Evidence(doc_id="e", text="total 20.00"),
            ],
        ),
        client,
        "test",
        RetrievalTool(DOCS),
        ledger,
        ApprovalQueue(),
    )
    assert state.status == "failed"
    assert state.settled is False
    assert ledger.entries == []


def test_empty_finish_answer_is_rejected_once() -> None:
    """An answer task cannot finish with a blank reason."""
    client = _ScriptedClient(
        [
            {"action": "finish", "args": {}, "reason": ""},
            {"action": "finish", "args": {}, "reason": "Jane Citizen exported 463 kWh"},
        ]
    )
    state = run_with_model(
        AgentState(task="Which customer exported the most solar power?", max_steps=4),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert state.step_count == 2
    assert any(line.startswith("not finished: put the complete answer") for line in state.history)


def test_model_run_bad_args_become_feedback_not_crash() -> None:
    client = _ScriptedClient(
        [
            {"action": "ledger_write", "args": {"entry_id": "", "amount": -1}, "reason": "bad"},
            {"action": "finish", "args": {}, "reason": "giving up cleanly"},
        ]
    )
    state = run_with_model(
        AgentState(task="t", max_steps=5),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "done"
    assert any(event.startswith("rejected:") for event in state.history)


def test_ceiling_marks_failed_never_done() -> None:
    queries = [{"action": "retrieve", "args": {"query": f"topic-{i}"}, "reason": "wander"} for i in range(10)]
    client = _ScriptedClient(queries)
    state = run_with_model(
        AgentState(task="t", max_steps=3),
        client,
        "test",
        RetrievalTool(DOCS),
        LedgerWriteTool(),
        ApprovalQueue(),
    )
    assert state.status == "failed"
    assert "ceiling" in state.failure_reason
    assert state.history[-1].startswith("exhausted:")
