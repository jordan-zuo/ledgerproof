from agent.approvals import ApprovalQueue
from agent.bill_math import verify_ledger_math
from agent.guardrails import (
    APPROVAL_THRESHOLD,
    assess_due_amount,
    assess_grounding,
    assess_ledger,
    assess_vendor,
    source_evidence,
)
from agent.periods import evidence_months, period_compatible, task_markers
from agent.states import AgentState, Evidence
from agent.tools import LedgerArgs, LedgerWriteTool


def _text(items: list[Evidence]) -> str:
    return "\n".join(item.text for item in items)


def _validate(
    state: AgentState, args: LedgerArgs
) -> tuple[bool, dict[str, str], str]:
    """Run every automatic check. Returns ok, per-check detail, failure reason."""
    cited = source_evidence(args.source_doc, state.evidence)
    ok, grounding_reason = assess_grounding(args, state.evidence)
    if not ok:
        return False, {"grounding": f"failed: {grounding_reason}"}, grounding_reason

    vendor_ok, vendor_reason = assess_vendor(state.task, args, state.evidence)
    if not vendor_ok:
        return (
            False,
            {"grounding": grounding_reason, "vendor": f"failed: {vendor_reason}"},
            vendor_reason,
        )

    math_ok, math_reason = verify_ledger_math(args)
    if not math_ok:
        return (
            False,
            {"grounding": grounding_reason, "math": f"failed: {math_reason}"},
            math_reason,
        )

    due_ok, due_reason = assess_due_amount(args, state.evidence)
    if not due_ok:
        return (
            False,
            {
                "grounding": grounding_reason,
                "math": math_reason,
                "amount_due": f"failed: {due_reason}",
            },
            due_reason,
        )

    period_ok, period_reason = period_compatible(state.task, _text(cited))
    checks = {
        "grounding": grounding_reason,
        "math": math_reason,
        "amount_due": due_reason,
        "period": period_reason,
    }
    if not period_ok:
        checks["period"] = f"failed: {period_reason}"
        return False, checks, period_reason
    return True, checks, "ok"


def _packet(
    state: AgentState, args: LedgerArgs, verdict_reason: str, checks: dict[str, str]
) -> dict:
    cited = source_evidence(args.source_doc, state.evidence)
    return {
        "task": state.task,
        "proposed": args.model_dump(),
        "threshold": APPROVAL_THRESHOLD,
        "rule": verdict_reason,
        "checks": checks,
        "period": {
            "task_markers": sorted(task_markers(state.task)),
            "evidence_months": sorted(evidence_months(_text(cited))),
        },
        "citations": [item.model_dump() for item in cited],
    }


def guarded_ledger_write(
    state: AgentState,
    ledger: LedgerWriteTool,
    args: LedgerArgs,
    queue: ApprovalQueue,
) -> str:
    """Run grounding and math first, then the money gate.

    Unquoted amounts and math mismatches become visible rejections
    the model sees next step. Small amounts execute at once.
    Large amounts pause behind a ticket carrying its evidence packet
    and return the ticket id.
    """
    ok, checks, reason = _validate(state, args)
    if not ok:
        state.log(
            f"rejected: {reason}",
            kind="rejected",
            data={"action": "ledger_write", "reason": reason, "checks": checks},
        )
        return f"rejected: {reason}"

    existing = ledger.already_posted(args)
    if existing is not None:
        # One bill, one posting. The model's entry_id is a label, not an
        # identity, so re-runs used to mint a second row for the same bill.
        reason = (
            f"already posted: {args.source_doc} {args.amount:.2f} {args.currency}"
            f" is already in the ledger as {existing.get('entry_id')}"
            f" (run {existing.get('run_id')})"
        )
        state.log(
            f"rejected: {reason}",
            kind="rejected",
            data={
                "action": "ledger_write",
                "reason": reason,
                "existing": existing,
            },
        )
        return f"rejected: {reason}"

    verdict = assess_ledger(args)
    if not verdict.needs_approval:
        return ledger.execute(state, args)

    packet = _packet(state, args, verdict.reason, checks)
    ticket_id = queue.submit(
        action=ledger.name,
        detail=verdict.reason,
        meta={
            "entry_id": args.entry_id,
            "amount": args.amount,
            "currency": args.currency,
            "source_doc": args.source_doc,
        },
        evidence=packet,
    )
    state.status = "awaiting_approval"
    state.log(
        f"gated: {ticket_id} ({verdict.reason})",
        kind="gated",
        data={
            "ticket_id": ticket_id,
            "action": ledger.name,
            "reason": verdict.reason,
            "amount": args.amount,
            "currency": args.currency,
            "threshold": APPROVAL_THRESHOLD,
            "entry_id": args.entry_id,
            "source_doc": args.source_doc,
        },
    )
    return ticket_id


def resume_after_approval(
    state: AgentState,
    ledger: LedgerWriteTool,
    args: LedgerArgs,
    queue: ApprovalQueue,
    ticket_id: str,
    approved: bool = True,
) -> str:
    """Record the human decision, then re-run every automatic check.

    Approval does not bypass grounding, math, or period rules: a ticket
    approved against stale or tampered evidence still fails closed.
    """
    ticket = queue.decide(ticket_id, approved)
    if not ticket.approved:
        state.status = "failed"
        state.failure_reason = f"denied: {ticket_id}"
        state.log(
            f"denied: {ticket_id}",
            kind="denied",
            data={"ticket_id": ticket_id, "action": ledger.name},
        )
        return ticket_id

    ok, checks, reason = _validate(state, args)
    if not ok:
        state.status = "failed"
        state.failure_reason = f"post-approval check failed: {reason}"
        state.log(
            f"rejected: post-approval {reason}",
            kind="rejected",
            data={
                "action": "ledger_write",
                "reason": reason,
                "phase": "post-approval",
                "checks": checks,
            },
        )
        return state.failure_reason

    event = ledger.execute(state, args)
    state.log(
        event,
        kind="ledger_write",
        data={
            "entry_id": args.entry_id,
            "amount": args.amount,
            "currency": args.currency,
            "source_doc": args.source_doc,
        },
    )
    state.status = "done"
    return event
