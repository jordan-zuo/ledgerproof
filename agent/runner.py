from __future__ import annotations

import json
import re
import threading
from typing import Any

from pydantic import ValidationError

from agent.approvals import ApprovalQueue
from agent.driver import decide
from agent.ledger_flow import guarded_ledger_write, resume_after_approval
from agent.router import route
from agent.states import AgentState
from agent.tools import LedgerArgs, LedgerWriteTool, RetrievalArgs, RetrievalTool

_POST_ASK = re.compile(r"\bpost(?:s|ed|ing)?\b", re.IGNORECASE)


def _asks_for_post(task: str) -> bool:
    return bool(_POST_ASK.search(task))


def _stop_for_cancel(state: AgentState) -> AgentState:
    # The API records the cancellation event; the runner only stops here.
    state.status = "cancelled"
    state.failure_reason = "cancelled by user"
    return state


def run_with_model(
    state: AgentState,
    client: Any,
    model: str,
    retrieval: RetrievalTool,
    ledger: LedgerWriteTool,
    queue: ApprovalQueue,
    auto_approve: bool = False,
    cancel: threading.Event | None = None,
) -> AgentState:
    """Model proposes, graph disposes. Ceiling and gates always enforced."""
    state.status = "running"
    context = f"{retrieval.describe()} Ledger rules: {LedgerWriteTool.RULES}."

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    last_move: str | None = None
    last_query: tuple[str, int] | None = None
    retrieve_repeats = 0
    retrieve_nudged = False
    while True:
        # Cooperative stop: checked between steps, so an in-flight model call
        # finishes but its decision is discarded.
        if _cancelled():
            return _stop_for_cancel(state)
        if state.status == "running" and state.step_count >= state.max_steps:
            if _asks_for_post(state.task) and (state.posted or state.settled):
                # The write that completes the task landed on the final step;
                # the ceiling must not turn a successful posting into a failure.
                state.status = "done"
                state.log(
                    "done: step ceiling reached after the required posting",
                    kind="status",
                    data={
                        "reason": "posting landed at the step ceiling",
                        "max_steps": state.max_steps,
                    },
                )
                return state
            state.status = "failed"
            state.failure_reason = "step ceiling exhausted without completion"
            state.log(
                "exhausted: step ceiling reached",
                kind="error",
                data={"reason": state.failure_reason, "max_steps": state.max_steps},
            )
            return state
        terminal = route(state)
        if terminal.next_node == "terminal":
            if state.status == "running":
                state.status = "done"
            return state
        try:
            decision = decide(state, client, model, context=context)
        except ValueError as exc:
            # A provider that stops returning usable routes must not turn an
            # already-landed posting into a failure.
            if _asks_for_post(state.task) and (state.posted or state.settled):
                state.status = "done"
                state.log(
                    "done: the posting landed before the model stopped routing",
                    kind="status",
                    data={"reason": str(exc)},
                )
                return state
            state.status = "failed"
            state.failure_reason = f"model stopped returning decisions: {exc}"
            state.log(
                "failed: no usable decision from the model",
                kind="error",
                data={"reason": str(exc)},
            )
            return state
        if _cancelled():
            # The answer arrived after the human cancelled; do not execute it.
            return _stop_for_cancel(state)
        record = {
            "step": state.step_count + 1,
            "action": decision.action,
            "args": decision.args,
            "reason": decision.reason,
            "model": getattr(client, "last_label", "") or model,
        }
        state.decisions.append(record)
        state.emit("decision", json.dumps(record, sort_keys=True), record)
        move = decision.action + ":" + json.dumps(decision.args, sort_keys=True)
        # A repeated read is harmless: the same query returns the same
        # documents, and the step ceiling still bounds a model that loops.
        # Finish has its own completion rule below.
        if move == last_move and decision.action not in ("retrieve", "finish"):
            state.status = "failed"
            state.failure_reason = "repeated ineffective action"
            state.log(
                f"stopped: the agent tried the same {decision.action} step twice,"
                " so the run ended.",
                kind="error",
                data={"reason": state.failure_reason, "action": decision.action},
            )
            return state
        last_move = move
        state.step_count += 1
        if decision.action == "finish":
            # "Done" must mean the task happened. A run asked to post cannot
            # end before a posting lands, or before a bill is confirmed as
            # already posted. One nudge, then it fails rather than lie.
            if _asks_for_post(state.task) and not (state.posted or state.settled):
                if state.finish_nudged:
                    state.status = "failed"
                    state.failure_reason = (
                        "the task asked for a posting and none was made"
                    )
                    state.log(
                        "ended: nothing was posted, so the run is marked failed.",
                        kind="error",
                        data={"reason": state.failure_reason, "action": "finish"},
                    )
                    return state
                state.finish_nudged = True
                state.log(
                    "not finished: the task asks for a posting and none has been"
                    " made yet. Post the bill, or say why it cannot be posted.",
                    kind="rejected",
                    data={"action": "finish", "reason": "no ledger write yet"},
                )
                continue
            if not decision.reason.strip() and not _asks_for_post(state.task):
                # An informational task is not done until the answer is
                # actually stated. A blank finish is a non-answer.
                if state.finish_nudged:
                    state.status = "failed"
                    state.failure_reason = "the task asked a question and the answer was empty"
                    state.log(
                        "ended: finish was called with no answer.",
                        kind="error",
                        data={"reason": state.failure_reason, "action": "finish"},
                    )
                    return state
                state.finish_nudged = True
                state.log(
                    "not finished: put the complete answer in the finish reason.",
                    kind="rejected",
                    data={"action": "finish", "reason": "answer was empty"},
                )
                continue
            state.status = "done"
            state.log(
                f"finish: {decision.reason}",
                kind="finish",
                data={"reason": decision.reason},
            )
            return state
        try:
            if decision.action == "retrieve":
                args = RetrievalArgs(**decision.args)
                key = (args.query.strip().lower(), args.top_k)
                if key == last_query:
                    retrieve_repeats += 1
                else:
                    last_query = key
                    retrieve_repeats = 1
                    retrieve_nudged = False
                if retrieve_repeats >= 2:
                    # A second identical read cannot add evidence. Nudge once
                    # with the documents already in hand; fail on the third.
                    if retrieve_nudged:
                        state.status = "failed"
                        state.failure_reason = "repeated ineffective retrieve"
                        state.log(
                            "stopped: the agent kept repeating the same retrieve,"
                            " so the run ended.",
                            kind="error",
                            data={"reason": state.failure_reason, "action": "retrieve"},
                        )
                        return state
                    retrieve_nudged = True
                    seen = sorted({item.doc_id for item in state.evidence})
                    detail = (
                        f"it already returned {', '.join(seen)}"
                        if seen
                        else "it already returned no evidence"
                    )
                    state.log(
                        f"not finished: this retrieve already ran; {detail}."
                        " Use those snippets and answer, or query a different"
                        " document or term.",
                        kind="rejected",
                        data={"action": "retrieve", "reason": "repeated retrieve"},
                    )
                    continue
                seen = len(state.evidence)
                out = retrieval.execute(state, args)
                state.log(
                    out,
                    kind="retrieve",
                    data={
                        "query": args.query,
                        "doc_ids": [item.doc_id for item in state.evidence[seen:]],
                    },
                )
            elif decision.action == "ledger_write":
                args = LedgerArgs(**decision.args)
                out = guarded_ledger_write(state, ledger, args, queue)
                if out.startswith("rejected:"):
                    # A confirmed duplicate is a settled outcome: there is
                    # nothing left to post, so finishing is correct.
                    if "already posted" in out:
                        state.settled = True
                elif state.status == "awaiting_approval":
                    if not auto_approve:
                        return state
                    resume_after_approval(state, ledger, args, queue, out)
                    state.posted = state.status == "done"
                else:
                    state.log(
                        out,
                        kind="ledger_write",
                        data={
                            "entry_id": args.entry_id,
                            "amount": args.amount,
                            "currency": args.currency,
                            "source_doc": args.source_doc,
                        },
                    )
                    state.posted = True
            elif decision.action == "request_approval":
                # A model-requested pause must create a real ticket, or the
                # run is stuck forever with nothing for a human to decide.
                detail = decision.reason or "model requested approval"
                packet = {
                    "task": state.task,
                    "proposed": {
                        "action": "request_approval",
                        "reason": decision.reason,
                    },
                    "rule": "model invoked the approval channel",
                    "citations": [
                        item.model_dump() for item in state.evidence
                    ],
                    "checks": {"gate": "human decision required"},
                    "history": list(state.history[-6:]),
                }
                ticket_id = queue.submit(
                    action="request_approval",
                    detail=detail,
                    meta={"reason": decision.reason},
                    evidence=packet,
                )
                state.status = "awaiting_approval"
                state.log(
                    f"gated: {ticket_id} ({detail})",
                    kind="gated",
                    data={
                        "ticket_id": ticket_id,
                        "action": "request_approval",
                        "reason": detail,
                    },
                )
                return state
        except ValidationError as exc:
            # Bad arguments become history the model sees next iteration,
            # not a crash. The step ceiling bounds repeated failures.
            # Anything else still propagates.
            state.log(
                f"rejected: {exc.errors()[0]['msg']}",
                kind="rejected",
                data={"action": decision.action, "reason": exc.errors()[0]["msg"]},
            )
        if state.status in ("done", "failed", "awaiting_approval"):
            return state
