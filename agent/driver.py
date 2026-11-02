from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelDecision(BaseModel):
    action: Literal["retrieve", "ledger_write", "finish", "request_approval"]
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


KNOWN_ACTIONS = ("retrieve", "ledger_write", "finish", "request_approval")


def build_client() -> Any:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("The 'python-dotenv' package is required.") from exc
    load_dotenv()
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for live calls.") from exc
    from agent.models import RotatingClient, specs_for

    return RotatingClient(specs_for("agent"))


def _prompt_history(history: list[str], limit: int = 16) -> list[str]:
    """Drop duplicate retrieve lines and bound the tail the model re-reads."""
    seen: set[str] = set()
    kept: list[str] = []
    for line in history:
        if line.startswith("retrieve ["):
            key = line.split("]:", 1)[0]
            if key in seen:
                continue
            seen.add(key)
        kept.append(line)
    return kept[-limit:]


def _retrieved_docs(state: Any) -> list[str]:
    seen: list[str] = []
    for item in state.evidence:
        if item.doc_id not in seen:
            seen.append(item.doc_id)
    return seen


def _prompt_for(state: Any, hint: str = "", context: str = "") -> str:
    history = _prompt_history(list(state.history))
    docs = _retrieved_docs(state)
    known = (
        " Documents already retrieved: " + ", ".join(docs) + ". Do not retrieve them again."
        if docs
        else ""
    )
    base = (
        "You route one step of a deterministic workflow. "
        "You MUST pick action from exactly this set: retrieve, ledger_write, "
        "finish, request_approval. Any other action string is invalid. "
        "Advance every step. Never repeat a query that already returned results in history. "
        "For posting tasks: retrieve the bill first, then write. "
        "For informational questions or comparisons: do not write to the ledger; compare the retrieved figures and call finish with the direct answer in reason. "
        "When the task names the Origin standing bill, retrieve "
        "origin-standing-bill-modified-alexcarter and post its Total Amount Due "
        "exactly as shown in the snippet. "
        "Post only the bill matching the task period and account. "
        "If no retrieved bill matches the asked period, do not write. "
        "Finish with the reason instead. "
        "If retrieve says a task-named document is absent, do not substitute "
        "another bill and do not write; finish with that reason. "
        "Use retrieved documents to move toward ledger_write or "
        "finish. "
        "retrieve: keyword search over named documents. Use plain words, "
        "never SQL. Returns id plus text snippet with numbers. Read the snippet. "
        "ledger_write: append-only money record. Amount must already appear "
        "in history with two decimals, copied from a retrieve snippet. "
        "Never invent amounts. source_doc must be a retrieved doc id. "
        "If the amount exceeds the approval threshold, still call ledger_write "
        "with the quoted amount; the orchestrator pauses it automatically. "
        "Do not call request_approval just because an amount is large. "
        "request_approval is only for a blocker that is not a ledger write. "
        "finish: task is fully done; for informational questions, provide the complete answer including all requested details in reason. "
        "For informational answers, quote the exact figures or dates from the snippets in reason; never answer from memory. "
        "If a requested detail (such as customer name) was not in the snippet, retrieve with that term or doc id first before finishing. "
        "For comparative questions (such as who has the highest or lowest amount), make sure to compare all candidate bills in context before concluding. "
        "Once the snippets contain every candidate bill, compare them and finish; do not run the same retrieve again. "
        "One retrieve with top_k up to 10 returns every matching bill; prefer that over fetching documents one by one. "
        "Never repeat a retrieve you already made; the result will not change. "
        "If the documents are already in history, answer from them. "
        "Once a ledger write lands, finish with what you did instead of searching again. "
        "A retrieve result can list several documents separated by '||'; read every one before comparing. "
        "For solar exports, the billing-period total is the feed-in tariff row in kWh; a chart figure such as 'Solar exports: 15 kWh' is a daily average, not the total. "
        "If a bill has no solar export row, that customer has no solar exports. "
        "request_approval: pause for a human decision. "
        "Arg shapes: retrieve takes {query, top_k (optional, 1 to 10)}; ledger_write takes "
        "{entry_id, amount, source_doc}; finish takes {}; request_approval "
        "takes {reason}. Do not emit tool calls or function-call syntax; "
        "this channel accepts no tools. Reply with ONLY this JSON object: "
        '{"action": "<one of the four>", "args": {...}, "reason": "<short>"}. '
        f"Task: {state.task}. Steps used: {state.step_count}/{state.max_steps}. "
        f"History: {history}.{known}"
    )
    if context:
        base += f" Environment: {context}"
    if hint:
        return base + f" Previous answer was rejected: {hint}. Fix it."
    return base


def _parse(content: str) -> ModelDecision:
    text = (content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"model did not return JSON: {content!r}")
    payload = json.loads(text[start : end + 1])
    if payload.get("action") not in KNOWN_ACTIONS:
        raise ValueError(f"unknown action: {payload.get('action')!r}")
    return ModelDecision(
        action=payload["action"],
        args=dict(payload.get("args") or {}),
        reason=str(payload.get("reason") or ""),
    )


def _call_settings() -> dict[str, Any]:
    # Reasoning models spend output budget on hidden tokens before writing the
    # JSON route. Without room and a low effort they return empty content.
    # The pool owns each provider's wire shape for the reasoning request.
    settings: dict[str, Any] = {
        "temperature": 0,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1200")),
    }
    effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
    if effort:
        settings["reasoning"] = effort
    return settings


def _is_tool_call_rejection(exc: Exception) -> bool:
    # Some providers parse tool syntax out of the text before returning it and
    # reject the whole request. Nothing was executed, so it is a bad answer.
    text = str(exc).lower()
    return getattr(exc, "status_code", None) == 400 and "tool" in text


def decide(state: Any, client: Any, model: str, max_attempts: int = 2, context: str = "") -> ModelDecision:
    """Ask the model what to do next. Malformed or unknown answers raise.

    The model gets one retry with the rejection reason fed back. Persistent
    misbehavior raises instead of executing something unintended.
    """
    hint = ""
    for _ in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt_for(state, hint, context)}],
                **_call_settings(),
            )
        except Exception as exc:
            if not _is_tool_call_rejection(exc):
                raise
            hint = "the provider rejected a tool-call shaped answer; reply with the JSON route only"
            continue
        content = completion.choices[0].message.content or ""
        try:
            return _parse(content)
        except ValueError as exc:
            hint = str(exc)
    raise ValueError(f"model failed to route after {max_attempts} attempts: {hint}")
