"""Period accounting: which month a bill belongs to, what a close froze,
and the two export shapes.

A bill is booked to the month its billing period ends, not the month the
posting happened. Server time cannot be backdated, so a posting-month close
could never say anything about the bills it was supposed to finalise. When a
bill has no period, its issue month is used, and only then the posting month.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from agent.reconciliation import group_key, reconcile, summarize

PERIOD_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

GENERIC_COLUMNS = (
    "ledger_id",
    "bill_key",
    "source_doc",
    "entry_kind",
    "accounting_period",
    "period_basis",
    "account",
    "period_start",
    "period_end",
    "issue_date",
    "amount",
    "currency",
    "bill_status",
    "outstanding",
    "run_id",
    "entry_id",
    "reverses_id",
    "reason",
    "approved_by",
    "recorded_at",
    "period_closed",
)

XERO_COLUMNS = (
    "ContactName",
    "InvoiceNumber",
    "InvoiceDate",
    "DueDate",
    "Description",
    "Quantity",
    "UnitAmount",
    "AccountCode",
    "TaxType",
    "Currency",
)

REVIEW_STATUSES = ("conflict", "duplicate")


def valid_period(value: str) -> str:
    if not PERIOD_PATTERN.match(value or ""):
        raise ValueError("period must be YYYY-MM")
    return value


def _month(value: str) -> str:
    candidate = (value or "")[:7]
    return candidate if PERIOD_PATTERN.match(candidate) else ""


def period_for(
    bill_key: str,
    source_doc: str,
    recorded_at: str,
    document: dict | None = None,
) -> tuple[str, str]:
    """Return ``(YYYY-MM, basis)`` for one ledger row.

    Identity carries the date that matters when it could be derived at
    ingest. The documents table covers rows whose identity fell back to the
    document id, and the posting month is the last resort.
    """
    for part in (bill_key or "").split("|"):
        if part.startswith("period:"):
            end = part.removeprefix("period:").split(":")[-1]
            if _month(end):
                return _month(end), "bill period end"
    for part in (bill_key or "").split("|"):
        if part.startswith("issued:"):
            issued = _month(part.removeprefix("issued:"))
            if issued:
                return issued, "issue date"
    if document:
        if _month(str(document.get("period_end") or "")):
            return _month(str(document["period_end"])), "document period end"
        if _month(str(document.get("issue_date") or "")):
            return _month(str(document["issue_date"])), "document issue date"
    return _month(recorded_at) or "unknown", "posting month"


def facts(entry: dict, document: dict | None = None) -> dict[str, str]:
    """The bill fields the ledger rows themselves do not carry."""
    account = ""
    period_start = ""
    period_end = ""
    issue_date = ""
    for part in (entry.get("bill_key") or "").split("|"):
        if part.startswith("acct:"):
            account = part.removeprefix("acct:")
        elif part.startswith("period:"):
            start, _, end = part.removeprefix("period:").partition(":")
            period_start, period_end = start, end
        elif part.startswith("issued:"):
            issue_date = part.removeprefix("issued:")
    if document:
        account = account or str(document.get("account") or "")
        period_start = period_start or str(document.get("period_start") or "")
        period_end = period_end or str(document.get("period_end") or "")
        issue_date = issue_date or str(document.get("issue_date") or "")
    return {
        "account": account,
        "period_start": period_start,
        "period_end": period_end,
        "issue_date": issue_date,
    }


def entry_periods(
    entries: list[dict], documents: dict[str, dict]
) -> dict[int, tuple[str, str]]:
    """Resolve every row once, following a reversal back to its posting.

    A correction belongs to the month of the bill it corrects, even though
    its own ``created_at`` is today.
    """
    by_id = {int(entry["id"]): entry for entry in entries}
    cache: dict[int, tuple[str, str]] = {}

    def resolve(entry: dict) -> tuple[str, str]:
        ledger_id = int(entry["id"])
        if ledger_id in cache:
            return cache[ledger_id]
        parent = entry.get("reverses_id")
        if entry.get("kind") == "reversal" and parent in by_id:
            value = resolve(by_id[parent])
        else:
            document = documents.get(str(entry["source_doc"]))
            value = period_for(
                str(entry.get("bill_key") or ""),
                str(entry["source_doc"]),
                str(entry["created_at"]),
                document,
            )
        cache[ledger_id] = value
        return value

    return {int(entry["id"]): resolve(entry) for entry in entries}


def closure_map(closures: list[dict]) -> dict[str, dict]:
    """Latest close or reopen action per period, with its full history."""
    out: dict[str, dict] = {}
    for row in closures:
        period = str(row["period"])
        state = out.setdefault(
            period,
            {
                "period": period,
                "status": "open",
                "actions": [],
                "closed_at": "",
                "closed_by": "",
                "note": "",
                "snapshot": {},
                "reopened_at": "",
                "reopened_by": "",
            },
        )
        state["actions"].append(
            {
                "id": row["id"],
                "action": row["action"],
                "note": row["note"],
                "decided_by": row["decided_by"],
                "at": row["created_at"],
            }
        )
        if row["action"] == "close":
            state.update(
                status="closed",
                closed_at=row["created_at"],
                closed_by=row["decided_by"],
                note=row["note"],
                snapshot=row.get("snapshot") or {},
            )
        else:
            state.update(
                status="open",
                reopened_at=row["created_at"],
                reopened_by=row["decided_by"],
            )
    return out


def _totals(rows: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in rows:
        currency = str(entry["currency"])
        totals[currency] = round(
            totals.get(currency, 0.0) + float(entry["amount"]), 2
        )
    return totals


def close_snapshot(
    period: str, entries: list[dict], documents: dict[str, dict]
) -> dict:
    """What was true when a month was signed off.

    Stored beside the close action so a later report can prove what the
    operator closed, even after a reopen and more activity.
    """
    periods = entry_periods(entries, documents)
    rows = [entry for entry in entries if periods[int(entry["id"])][0] == period]
    bills = reconcile(rows)
    summary = summarize(bills)
    return {
        "period": period,
        "bills": summary["bills"],
        "by_status": summary["by_status"],
        "outstanding": summary["outstanding"],
        "net": _totals(rows),
        "exceptions": [
            bill["bill_key"]
            for bill in bills
            if bill["status"] in REVIEW_STATUSES
        ],
        "ledger_ids": sorted(int(entry["id"]) for entry in rows),
        "row_count": len(rows),
    }


def periods_view(
    entries: list[dict],
    closures: list[dict],
    documents: dict[str, dict],
) -> list[dict]:
    """One row per accounting month, newest first.

    A closed month with no activity still appears: an empty month can be
    closed on purpose, and the operator needs to see that it was.
    """
    periods = entry_periods(entries, documents)
    rows_by_period: dict[str, list[dict]] = {}
    for entry in entries:
        rows_by_period.setdefault(periods[int(entry["id"])][0], []).append(entry)
    states = closure_map(closures)

    out = []
    for period in sorted(set(rows_by_period) | set(states), reverse=True):
        rows = rows_by_period.get(period, [])
        bills = reconcile(rows)
        summary = summarize(bills)
        state = states.get(period) or {
            "period": period,
            "status": "open",
            "actions": [],
            "closed_at": "",
            "closed_by": "",
            "note": "",
            "snapshot": {},
            "reopened_at": "",
            "reopened_by": "",
        }
        out.append(
            {
                "period": period,
                "status": state["status"],
                "bills": summary["bills"],
                "by_status": summary["by_status"],
                "posting_count": sum(
                    1 for entry in rows if entry["kind"] == "posting"
                ),
                "reversal_count": sum(
                    1 for entry in rows if entry["kind"] == "reversal"
                ),
                "outstanding": summary["outstanding"],
                "net": _totals(rows),
                "exceptions": [
                    bill["bill_key"]
                    for bill in bills
                    if bill["status"] in REVIEW_STATUSES
                ],
                "row_count": len(rows),
                "closed_at": state["closed_at"],
                "closed_by": state["closed_by"],
                "note": state["note"],
                "snapshot": state["snapshot"],
                "reopened_at": state["reopened_at"],
                "reopened_by": state["reopened_by"],
                "history": state["actions"],
            }
        )
    return out


def periods_summary(periods: list[dict]) -> dict:
    outstanding: dict[str, float] = {}
    for period in periods:
        if period["status"] != "closed":
            continue
        for currency, amount in period["outstanding"].items():
            outstanding[currency] = round(
                outstanding.get(currency, 0.0) + float(amount), 2
            )
    return {
        "periods": len(periods),
        "open": sum(1 for period in periods if period["status"] == "open"),
        "closed": sum(1 for period in periods if period["status"] == "closed"),
        "exceptions": sum(len(period["exceptions"]) for period in periods),
        "closed_outstanding": outstanding,
    }


def ledger_rows(
    entries: list[dict],
    closures: list[dict],
    documents: dict[str, dict],
    *,
    period: str | None = None,
    scope: str = "all",
) -> list[dict]:
    """Generic audit CSV rows, optionally limited to one month."""
    periods = entry_periods(entries, documents)
    statuses = {bill["bill_key"]: bill["status"] for bill in reconcile(entries)}
    closed = {
        key for key, state in closure_map(closures).items()
        if state["status"] == "closed"
    }
    out = []
    for entry in entries:
        month, basis = periods[int(entry["id"])]
        if period and month != period:
            continue
        outstanding = entry["kind"] == "posting" and not entry["reversed"]
        if scope == "outstanding" and not outstanding:
            continue
        bill = facts(entry, documents.get(str(entry["source_doc"])))
        out.append(
            {
                "ledger_id": entry["id"],
                "bill_key": group_key(entry),
                "source_doc": entry["source_doc"],
                "entry_kind": entry["kind"],
                "accounting_period": month,
                "period_basis": basis,
                "account": bill["account"],
                "period_start": bill["period_start"],
                "period_end": bill["period_end"],
                "issue_date": bill["issue_date"],
                "amount": f"{float(entry['amount']):.2f}",
                "currency": entry["currency"],
                "bill_status": statuses.get(group_key(entry), ""),
                "outstanding": str(outstanding).lower(),
                "run_id": entry["run_id"],
                "entry_id": entry["entry_id"],
                "reverses_id": entry.get("reverses_id") or "",
                "reason": entry.get("reason") or "",
                "approved_by": entry.get("approved_by") or "",
                "recorded_at": entry["created_at"],
                "period_closed": str(month in closed).lower(),
            }
        )
    return out


@dataclass(frozen=True)
class XeroMapping:
    contact: str
    account_code: str
    tax_type: str
    due_days: int = 0


def missing_xero_fields(mapping: XeroMapping) -> list[str]:
    wanted = (
        ("contact", mapping.contact),
        ("account_code", mapping.account_code),
        ("tax_type", mapping.tax_type),
    )
    return [name for name, value in wanted if not str(value).strip()]


def xero_rows(
    entries: list[dict],
    documents: dict[str, dict],
    mapping: XeroMapping,
    *,
    period: str | None = None,
) -> list[dict]:
    """Outstanding bills only: Xero bills are payables, not an audit log.

    A corrected bill exports its live replacement posting; the reversed row
    is deliberately absent. Fields the ledger never stored (supplier,
    account code, tax type) come from the caller, never from a guess.
    """
    rows = ledger_rows(
        entries, [], documents, period=period, scope="outstanding"
    )
    out = []
    for row in rows:
        invoice_date = (
            row["issue_date"] or row["period_end"] or row["recorded_at"][:10]
        )
        try:
            due_date = (
                date.fromisoformat(invoice_date) + timedelta(days=mapping.due_days)
            ).isoformat()
        except ValueError:
            invoice_date = row["recorded_at"][:10]
            due_date = invoice_date
        out.append(
            {
                "ContactName": mapping.contact,
                "InvoiceNumber": row["source_doc"],
                "InvoiceDate": invoice_date,
                "DueDate": due_date,
                "Description": row["source_doc"],
                "Quantity": 1,
                "UnitAmount": row["amount"],
                "AccountCode": mapping.account_code,
                "TaxType": mapping.tax_type,
                "Currency": row["currency"],
            }
        )
    return out
