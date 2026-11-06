import pytest

from agent.approvals import ApprovalQueue
from agent.guardrails import assess_ledger, duplicate_is_target
from agent.ledger_flow import guarded_ledger_write, resume_after_approval
from agent.states import AgentState, Evidence
from agent.tools import LedgerArgs, LedgerWriteTool


def _args(amount: float = 100.0) -> LedgerArgs:
    return LedgerArgs(entry_id="j-1", amount=amount, source_doc="q3")


def _seeded_state(amount: float, source: str = "q3") -> AgentState:
    return AgentState(
        task="t",
        evidence=[Evidence(doc_id=source, text=f"total {float(amount):.2f}")],
    )


def test_small_amount_executes_at_once() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = _seeded_state(100.0)
    event = guarded_ledger_write(state, ledger, _args(100.0), queue)
    assert event.startswith("ledger_write:")
    assert state.status == "pending"
    assert queue.pending() == []
    assert any(event.startswith("ledger_write:") for event in state.history) is False


def test_large_amount_gates_behind_ticket() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = _seeded_state(50_000.0)
    ticket_id = guarded_ledger_write(state, ledger, _args(50_000.0), queue)
    assert ticket_id.startswith("t-")
    assert state.status == "awaiting_approval"
    assert len(ledger.entries) == 0
    assert len(queue.pending()) == 1


def test_approval_resumes_and_writes() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = _seeded_state(50_000.0)
    ticket_id = guarded_ledger_write(state, ledger, _args(50_000.0), queue)
    event = resume_after_approval(state, ledger, _args(50_000.0), queue, ticket_id)
    assert event.startswith("ledger_write:")
    assert state.status == "done"
    assert len(ledger.entries) == 1
    assert event in state.history


def test_denial_fails_closed_without_a_row() -> None:
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = _seeded_state(50_000.0)
    ticket_id = guarded_ledger_write(state, ledger, _args(50_000.0), queue)

    event = resume_after_approval(
        state, ledger, _args(50_000.0), queue, ticket_id, approved=False
    )

    assert event == ticket_id
    assert state.status == "failed"
    assert state.failure_reason == f"denied: {ticket_id}"
    assert ledger.entries == []
    assert f"denied: {ticket_id}" in state.history


def test_unquoted_amount_rejected() -> None:
    ledger, queue, state = LedgerWriteTool(), ApprovalQueue(), AgentState(task="t")
    event = guarded_ledger_write(state, ledger, _args(100.0), queue)
    assert event.startswith("rejected:")
    assert len(ledger.entries) == 0
    assert state.status == "pending"


def test_bill_math_mismatch_rejected() -> None:
    from agent.tools import LedgerArgs as _LA

    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    args = _LA(entry_id="j-bill", amount=1250.5, source_doc="bill-q3-electricity")
    state = AgentState(
        task="t",
        evidence=[
            Evidence(
                doc_id="bill-q3-electricity",
                text="Amount due 4049.18 dollars",
            ),
            Evidence(doc_id="bill-q3-electricity", text="total 1250.50"),
        ],
    )
    event = guarded_ledger_write(state, ledger, args, queue)
    assert event.startswith("rejected:")
    assert len(ledger.entries) == 0


def test_named_bill_absent_from_evidence_is_rejected() -> None:
    """A Jemena request must not be settled by posting the AER bill."""
    ledger, queue = LedgerWriteTool(), ApprovalQueue()
    state = AgentState(
        task="Post the amount due from the Jemena gas bill.",
        evidence=[
            Evidence(doc_id="aer-example-bill", text="Amount due 79.87 dollars")
        ],
    )
    args = LedgerArgs(entry_id="j-1", amount=79.87, source_doc="aer-example-bill")
    event = guarded_ledger_write(state, ledger, args, queue)
    assert event.startswith("rejected:")
    assert "jemena" in event
    assert ledger.entries == []


def test_unknown_ticket_raises() -> None:
    queue = ApprovalQueue()
    with pytest.raises(KeyError, match="unknown approval ticket"):
        queue.decide("t-9999", True)


def test_duplicate_settles_only_the_named_bill() -> None:
    """An already-posted AER bill must not settle an AGL request."""
    task = "Post the AGL amount due."
    agl = Evidence(doc_id="agl-better-bills-sample-bill-necf", text="Amount due 63.70")
    aer = Evidence(doc_id="aer-example-bill", text="Amount due 79.87")
    assert duplicate_is_target(
        task, LedgerArgs(entry_id="j-1", amount=63.70, source_doc=agl.doc_id), [agl]
    )
    assert not duplicate_is_target(
        task, LedgerArgs(entry_id="j-2", amount=79.87, source_doc=aer.doc_id), [agl, aer]
    )


def test_unnamed_task_needs_one_candidate_document() -> None:
    """With no named bill, only a single-document run can call it settled."""
    first = Evidence(doc_id="doc-a", text="Amount due 10.00")
    second = Evidence(doc_id="doc-b", text="Amount due 20.00")
    args = LedgerArgs(entry_id="j-1", amount=10.00, source_doc="doc-a")
    assert duplicate_is_target("post the bill", args, [first])
    assert not duplicate_is_target("post the bill", args, [first, second])


def test_threshold_boundary() -> None:
    assert assess_ledger(_args(10_000.0)).needs_approval is False
    assert assess_ledger(_args(10_000.01)).needs_approval is True


def test_is_quoted_matches_amounts_with_trailing_punctuation() -> None:
    from agent.guardrails import is_quoted

    evidence = [
        Evidence(doc_id="doc-1", text="The total amount due is $114.87."),
        Evidence(doc_id="doc-2", text="Payment of $79.87, due on 27 April."),
        Evidence(doc_id="doc-3", text="Discount applied: ($13.00) total."),
    ]
    assert is_quoted(114.87, evidence[:1])
    assert is_quoted(79.87, evidence[1:2])
    assert is_quoted(13.00, evidence[2:3])


def test_is_quoted_rejects_subnumbers_and_extended_decimals() -> None:
    from agent.guardrails import is_quoted

    assert not is_quoted(
        13.00, [Evidence(doc_id="doc", text="Total amount is $113.00.")]
    )
    assert not is_quoted(
        13.00, [Evidence(doc_id="doc", text="Total amount is $1,013.00.")]
    )
    assert not is_quoted(
        13.00, [Evidence(doc_id="doc", text="Total is 13.005.")]
    )
