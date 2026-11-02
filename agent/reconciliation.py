from __future__ import annotations

STATUSES = ("conflict", "duplicate", "corrected", "reversed", "posted")

# Attention first: a conflict is money held twice, a duplicate is a repeated
# posting, a correction is a reversal with a live replacement, a reversed bill
# owes nothing, and a plain posting is settled.
_ORDER = {status: index for index, status in enumerate(STATUSES)}


def group_key(entry: dict) -> str:
    """The canonical bill a row belongs to, with a legacy fallback.

    Rows written before canonical identity existed have no bill_key, so the
    document id is the only identity available for them.
    """
    return entry["bill_key"] or f"doc:{entry['source_doc']}"


def _status(postings: list[dict], reversals: list[dict]) -> str:
    outstanding = [entry for entry in postings if not entry["reversed"]]
    if len(outstanding) > 1:
        amounts = {round(float(entry["amount"]), 2) for entry in outstanding}
        return "conflict" if len(amounts) > 1 else "duplicate"
    if outstanding:
        return "corrected" if reversals else "posted"
    return "reversed"


def _trail(rows: list[dict]) -> list[dict]:
    return [
        {
            "ledger_id": entry["id"],
            "entry_id": entry["entry_id"],
            "kind": entry["kind"],
            "amount": float(entry["amount"]),
            "currency": entry["currency"],
            "run_id": entry["run_id"],
            "reason": entry["reason"],
            "approved_by": entry["approved_by"],
            "reverses_id": entry["reverses_id"],
            "reversed": entry["reversed"],
            "at": entry["created_at"],
        }
        for entry in sorted(rows, key=lambda item: item["id"])
    ]


def _bill(key: str, rows: list[dict]) -> dict:
    postings = [entry for entry in rows if entry["kind"] == "posting"]
    reversals = [entry for entry in rows if entry["kind"] == "reversal"]
    outstanding = [entry for entry in postings if not entry["reversed"]]
    latest = max(rows, key=lambda item: item["id"])
    documents = sorted({entry["source_doc"] for entry in rows})
    status = _status(postings, reversals)
    return {
        "bill_key": key,
        "source_doc": (outstanding or postings or rows)[-1]["source_doc"],
        "documents": documents,
        "currency": latest["currency"],
        "status": status,
        "conflict": status == "conflict",
        "outstanding_amount": round(
            sum(float(entry["amount"]) for entry in outstanding), 2
        ),
        "net_amount": round(sum(float(entry["amount"]) for entry in rows), 2),
        "posting_count": len(postings),
        "reversal_count": len(reversals),
        "last_run": latest["run_id"],
        "last_at": latest["created_at"],
        "last_entry_id": latest["entry_id"],
        "trail": _trail(rows),
    }


def reconcile(entries: list[dict]) -> list[dict]:
    """Collapse ledger rows into one row per bill, newest activity last.

    Reversal rows are negative and linked, so the net of a group is simply
    what remains owed after corrections.
    """
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(group_key(entry), []).append(entry)
    bills = [_bill(key, rows) for key, rows in groups.items()]
    bills.sort(key=lambda bill: (_ORDER[bill["status"]], -bill["trail"][-1]["ledger_id"]))
    return bills


def summarize(bills: list[dict]) -> dict:
    counts = {status: 0 for status in STATUSES}
    outstanding: dict[str, float] = {}
    for bill in bills:
        counts[bill["status"]] += 1
        if bill["outstanding_amount"]:
            currency = bill["currency"]
            outstanding[currency] = round(
                outstanding.get(currency, 0.0) + bill["outstanding_amount"], 2
            )
    return {
        "bills": len(bills),
        "by_status": counts,
        "outstanding": outstanding,
    }
