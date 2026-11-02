from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

# Bills print the same period with different separators and month spellings:
# "5 March 2023 to 4 April 2023", "1 Mar 2022 - 31 Mar 2022",
# and the AER PDF's "1 March 2022 – 31 March 2022" all describe one period.
_DATE_TEXT = r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
_PERIOD_RE = re.compile(
    rf"(?:bill(?:ing)?\s+period|period)?\s*[:\s]*"
    rf"({_DATE_TEXT})\s*(?:to|until|through|[-–—])\s*({_DATE_TEXT})",
    re.IGNORECASE,
)
_ISSUE_RE = re.compile(
    rf"(?:bill\s+)?issue\s+date\s*[:\s]*({_DATE_TEXT})",
    re.IGNORECASE,
)
_ACCOUNT_RE = re.compile(
    r"account\s*(?:number|no\.?|#)?\s*[:\s]*([0-9][0-9 xX-]{4,32})",
    re.IGNORECASE,
)
_NON_ID = re.compile(r"[^0-9A-Za-z]+")

_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%d %B %y", "%d %b %y")


@dataclass(frozen=True)
class BillIdentity:
    """A stable key for one bill, independent of the document id."""

    key: str
    basis: str
    account: str = ""
    period_start: str = ""
    period_end: str = ""
    issue_date: str = ""


def _date(value: str) -> date | None:
    cleaned = " ".join(value.split()).rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _account(text: str) -> str:
    match = _ACCOUNT_RE.search(text)
    if not match:
        return ""
    account = _NON_ID.sub("", match.group(1)).upper()
    return account if len(account) >= 6 else ""


def _period(text: str) -> tuple[date, date] | None:
    for match in _PERIOD_RE.finditer(text):
        start = _date(match.group(1))
        end = _date(match.group(2))
        if start and end:
            return (start, end) if start <= end else (end, start)
    return None


def _issue(text: str) -> date | None:
    match = _ISSUE_RE.search(text)
    return _date(match.group(1)) if match else None


def _doc_key(source_doc: str) -> str:
    cleaned = _NON_ID.sub("", source_doc).lower()
    return f"doc:{cleaned or 'unknown'}"


def identify_bill(source_doc: str, text: str) -> BillIdentity:
    """Prefer account plus period, then account plus issue date.

    Without two stable components the document id is the only safe identity:
    a loose account or date match alone could collapse two different bills.
    """
    account = _account(text)
    period = _period(text)
    issue = _issue(text)
    issue_text = issue.isoformat() if issue else ""
    if account and period:
        start, end = period
        return BillIdentity(
            key=f"acct:{account}|period:{start.isoformat()}:{end.isoformat()}",
            basis="account+period",
            account=account,
            period_start=start.isoformat(),
            period_end=end.isoformat(),
            issue_date=issue_text,
        )
    if account and issue:
        return BillIdentity(
            key=f"acct:{account}|issued:{issue.isoformat()}",
            basis="account+issue",
            account=account,
            issue_date=issue_text,
        )
    return BillIdentity(
        key=_doc_key(source_doc),
        basis="document",
        account=account,
        period_start=period[0].isoformat() if period else "",
        period_end=period[1].isoformat() if period else "",
        issue_date=issue_text,
    )
