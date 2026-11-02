"""Reconciliation: one row per bill, with its full reversal trail."""

from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from agent.bill_identity import identify_bill
from agent.reconciliation import reconcile, summarize

KEY = "acct:1234567891X|period:2023-03-05:2023-04-04"


def _row(
    ledger_id: int,
    amount: float,
    source_doc: str,
    bill_key: str = KEY,
    kind: str = "posting",
    reversed: bool = False,
    reverses_id: int | None = None,
    reason: str = "",
    approved_by: str = "",
) -> dict:
    return {
        "id": ledger_id,
        "entry_id": f"e-{ledger_id}",
        "run_id": f"run-{ledger_id}",
        "amount": amount,
        "currency": "AUD",
        "source_doc": source_doc,
        "bill_key": bill_key,
        "kind": kind,
        "reverses_id": reverses_id,
        "reason": reason,
        "approved_by": approved_by,
        "reversed": reversed,
        "created_at": f"2026-01-0{ledger_id}T00:00:00",
    }


def test_rows_under_two_doc_ids_reconcile_to_one_bill() -> None:
    entries = [
        _row(1, 63.70, "agl-live", reversed=True),
        _row(
            2,
            -63.70,
            "agl-live",
            kind="reversal",
            reverses_id=1,
            reason="wrong period",
            approved_by="alice",
        ),
        _row(3, 63.71, "bill-agl-apr23"),
    ]

    bills = reconcile(entries)

    assert len(bills) == 1
    bill = bills[0]
    assert bill["status"] == "corrected"
    assert bill["outstanding_amount"] == 63.71
    assert bill["net_amount"] == 63.71
    assert bill["documents"] == ["agl-live", "bill-agl-apr23"]
    assert [step["kind"] for step in bill["trail"]] == [
        "posting",
        "reversal",
        "posting",
    ]
    assert bill["trail"][1]["reverses_id"] == 1


def test_a_fully_reversed_bill_owes_nothing() -> None:
    entries = [
        _row(1, 79.87, "aer-live", reversed=True),
        _row(2, -79.87, "aer-live", kind="reversal", reverses_id=1),
    ]

    bill = reconcile(entries)[0]

    assert bill["status"] == "reversed"
    assert bill["outstanding_amount"] == 0.0
    assert bill["net_amount"] == 0.0
    assert bill["conflict"] is False


def test_conflicting_outstanding_amounts_sort_first() -> None:
    entries = [
        _row(1, 63.70, "clean", bill_key="doc:clean"),
        _row(2, 63.70, "conflict"),
        _row(3, 114.87, "conflict"),
    ]

    bills = reconcile(entries)

    assert [bill["status"] for bill in bills] == ["conflict", "posted"]
    conflict = bills[0]
    assert conflict["conflict"] is True
    assert conflict["outstanding_amount"] == 178.57
    summary = summarize(bills)
    assert summary["bills"] == 2
    assert summary["by_status"]["conflict"] == 1
    assert summary["outstanding"] == {"AUD": 242.27}


def test_rows_without_a_bill_key_group_by_document() -> None:
    entries = [
        _row(1, 10.00, "doc-a", bill_key=""),
        _row(2, 20.00, "doc-b", bill_key=""),
    ]

    bills = reconcile(entries)

    assert [bill["bill_key"] for bill in bills] == ["doc:doc-b", "doc:doc-a"]
    assert all(bill["status"] == "posted" for bill in bills)


def test_reconciliation_endpoint_returns_bills_and_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB", str(tmp_path / "recon.db"))
    store.init_db()
    text = (
        "Account number 1234567891X\n"
        "Bill period 5 March 2023 to 4 April 2023\n"
        "Total amount due 63.70\n"
    )
    key = identify_bill("agl-live", text).key
    run_id = store.create_run("agl bill for april 2023")
    store.insert_ledger_entry(
        run_id, "agl", 63.70, "AUD", "agl-live", bill_key=key
    )
    original = store.list_ledger()[0]
    store.reverse_ledger_entry(original["id"], "wrong period", "alice")

    body = TestClient(api.app).get("/reconciliation").json()

    assert len(body["bills"]) == 1
    bill = body["bills"][0]
    assert bill["bill_key"] == key
    assert bill["status"] == "reversed"
    assert bill["outstanding_amount"] == 0.0
    assert bill["trail"][1]["approved_by"] == "alice"
    assert body["summary"] == {
        "bills": 1,
        "by_status": {
            "conflict": 0,
            "duplicate": 0,
            "corrected": 0,
            "reversed": 1,
            "posted": 0,
        },
        "outstanding": {},
    }
