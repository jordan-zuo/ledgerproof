from __future__ import annotations

import re

from pydantic import BaseModel

from agent.entities import task_entities
from agent.states import Evidence
from agent.tools import LedgerArgs

APPROVAL_THRESHOLD = 10_000.0


class GuardrailVerdict(BaseModel):
    allowed: bool
    needs_approval: bool
    reason: str


def assess_ledger(args: LedgerArgs) -> GuardrailVerdict:
    if args.amount > APPROVAL_THRESHOLD:
        return GuardrailVerdict(
            allowed=True,
            needs_approval=True,
            reason=f"amount {args.amount:.2f} exceeds approval threshold {APPROVAL_THRESHOLD:.2f}",
        )
    return GuardrailVerdict(allowed=True, needs_approval=False, reason="within policy")


def source_evidence(source_doc: str, evidence: list[Evidence]) -> list[Evidence]:
    return [item for item in evidence if item.doc_id == source_doc]


def is_quoted(amount: float, evidence: list[Evidence]) -> bool:
    forms = {f"{float(amount):.2f}", f"{float(amount):,.2f}"}
    text = "\n".join(item.text for item in evidence)
    return any(form in text for form in forms)


def is_source_seen(source_doc: str, evidence: list[Evidence]) -> bool:
    return any(item.doc_id == source_doc for item in evidence)


def assess_grounding(args: LedgerArgs, evidence: list[Evidence]) -> tuple[bool, str]:
    """Amounts must be quoted from the cited document, never from history.

    Only retrieval writes to the evidence lane, so a rejection message that
    echoes an invented amount cannot ground a later retry.
    """
    cited = source_evidence(args.source_doc, evidence)
    if not cited:
        return False, (
            f"source {args.source_doc} absent from evidence; "
            "retrieve it first and quote its total"
        )
    if not is_quoted(args.amount, cited):
        return False, (
            f"amount {args.amount:.2f} absent from source {args.source_doc}; "
            "retrieve the bill first and quote its total instead of inventing"
        )
    return True, "grounded"


def assess_vendor(
    task: str, args: LedgerArgs, evidence: list[Evidence]
) -> tuple[bool, str]:
    """A named provider must appear in the bill being cited.

    Without this, a missing bill is "paid" by posting a different one:
    a Jemena request was answered with the AER example bill.
    """
    terms = task_entities(task)
    cited = source_evidence(args.source_doc, evidence)
    haystack = " ".join([args.source_doc, *(item.text for item in cited)]).lower()
    missing = [term for term in terms if term not in haystack]
    if missing:
        return False, (
            f"task names {', '.join(missing)} but {args.source_doc} does not; "
            "retrieve the named bill or finish without posting"
        )
    return True, "vendor matches the cited document"


# A bill can quote several true figures: charges before credits, the total
# bill, then what is actually payable. Only the last of those is a posting.
_DUE_RE = re.compile(
    r"(?:total\s+amount\s+due|amount\s+due|total\s+due|balance\s+due|amount\s+payable)"
    r"[^0-9$]{0,30}?"
    r"(\$\s*)?([0-9][0-9,]*(?:\.[0-9]{2})?)"
    r"\s*(dollars|dollar|aud)?",
    re.IGNORECASE,
)


def stated_due_amount(text: str) -> float | None:
    """Return the figure the document says is payable, if it names one.

    Matches ``$79.87``, ``79.87 dollars`` and ``Total Amount Due: $13,376.47``.
    A bare integer is ignored so a due date cannot be mistaken for money.
    """
    for match in _DUE_RE.finditer(text):
        sign, number, unit = match.group(1), match.group(2), match.group(3)
        if not (sign or unit or "." in number):
            continue
        try:
            return float(number.replace(",", ""))
        except ValueError:
            continue
    return None


def assess_due_amount(
    args: LedgerArgs, evidence: list[Evidence]
) -> tuple[bool, str]:
    """The posted figure must be what the document says is payable.

    Posting a quoted subtotal is still wrong: a bill that states
    ``Total due 79.87`` is not owed at ``114.87`` even though 114.87 is
    printed on it.
    """
    cited = source_evidence(args.source_doc, evidence)
    due = stated_due_amount("\n".join(item.text for item in cited))
    if due is None:
        return True, "no stated amount due; grounded only"
    if abs(due - float(args.amount)) > 0.005:
        return False, (
            f"amount {args.amount:.2f} is not what is payable; "
            f"{args.source_doc} states {due:.2f} due. Post {due:.2f} instead."
        )
    return True, f"matches the {due:.2f} the document says is due"

