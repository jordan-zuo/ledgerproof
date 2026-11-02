"""Period close and export: a month is final only when its rows are final."""

import csv
import io
import sqlite3

import pytest
from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from agent.accounting import close_snapshot, period_for

MARCH = "2023-03"
KEY_A = "acct:111111|period:2023-03-01:2023-03-31"
KEY_B = "acct:222222|period:2023-03-02:2023-03-30"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "close.db"))
    store.init_db()
    return str(tmp_path / "close.db")


def _post(key: str, doc: str, amount: float = 10.0, entry: str | None = None) -> dict:
    run_id = store.create_run("seed")
    store.insert_ledger_entry(
        run_id, entry or doc, amount, "AUD", doc, bill_key=key
    )
    return store.list_ledger()[-1]


def _close(period: str = MARCH, note: str = "month end") -> dict:
    documents = {doc["doc_id"]: doc for doc in store.list_documents()}
    snapshot = close_snapshot(period, store.list_ledger(), documents)
    return store.close_period(period, note, "alice", snapshot)


def _rows(response) -> list[dict]:
    return list(csv.DictReader(io.StringIO(response.text)))


def test_bill_period_wins_over_issue_and_posting_month() -> None:
    assert period_for(KEY_A, "doc-a", "2026-11-02T00:00:00") == (
        "2023-03",
        "bill period end",
    )
    assert period_for(
        "acct:123456|issued:2024-01-09", "doc-b", "2026-11-02T00:00:00"
    ) == ("2024-01", "issue date")
    assert period_for(
        "doc:c", "doc-c", "2026-11-02T00:00:00", {"period_end": "2022-12-31"}
    ) == ("2022-12", "document period end")
    assert period_for("doc:d", "doc-d", "2026-11-02T00:00:00") == (
        "2026-11",
        "posting month",
    )


def test_closed_month_refuses_postings_and_reversals(isolated_db) -> None:
    first = _post(KEY_A, "doc-a")
    _post(KEY_B, "doc-b", 20.0)
    _close()

    with pytest.raises(store.PeriodClosed):
        _post(
            "acct:333333|period:2023-03-05:2023-03-25", "doc-late", 30.0
        )
    with pytest.raises(store.PeriodClosed):
        store.reverse_ledger_entry(first["id"], "late fix", "alice")

    # A different accounting month is untouched.
    _post("acct:444444|period:2023-04-01:2023-04-30", "doc-april", 40.0)

    conn = store.connect()
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE period_closures SET note = 'rewritten'")
    finally:
        conn.close()


def test_api_close_records_snapshot_and_reopen_allows_correction(
    isolated_db,
) -> None:
    first = _post(KEY_A, "doc-a")
    _post(KEY_B, "doc-b", 20.0)
    client = TestClient(api.app)

    bad = client.post(
        "/periods/2023-13/close", json={"note": "typo month", "decided_by": "alice"}
    )
    assert bad.status_code == 400

    closed = client.post(
        f"/periods/{MARCH}/close",
        json={"note": "month end", "decided_by": "alice"},
    )
    assert closed.status_code == 200, closed.text
    assert closed.json()["closure"]["snapshot"]["bills"] == 2
    assert closed.json()["closure"]["snapshot"]["outstanding"] == {"AUD": 30.0}

    again = client.post(
        f"/periods/{MARCH}/close",
        json={"note": "twice", "decided_by": "alice"},
    )
    assert again.status_code == 409

    blocked = client.post(
        f"/ledger/{first['id']}/reverse",
        json={"reason": "late fix", "approved_by": "bob"},
    )
    assert blocked.status_code == 409
    assert "closed" in blocked.json()["detail"]

    march = next(
        item
        for item in client.get("/periods").json()["periods"]
        if item["period"] == MARCH
    )
    assert march["status"] == "closed"
    assert march["snapshot"]["row_count"] == 2
    assert march["closed_by"] == "alice"

    reopened = client.post(
        f"/periods/{MARCH}/reopen",
        json={"note": "fix a typo", "decided_by": "alice"},
    )
    assert reopened.status_code == 200, reopened.text
    assert client.post(
        f"/periods/{MARCH}/reopen",
        json={"note": "already open", "decided_by": "alice"},
    ).status_code == 409

    fixed = client.post(
        f"/ledger/{first['id']}/reverse",
        json={"reason": "late fix", "approved_by": "bob"},
    )
    assert fixed.status_code == 200, fixed.text


def test_approval_for_a_closed_month_does_not_consume_the_ticket(
    isolated_db,
) -> None:
    _post(KEY_A, "doc-a")
    _close()
    run_id = store.create_run("post the closed bill")
    store.set_run_status(run_id, "awaiting_approval", 2)
    text = (
        "Account number 123456\n"
        "Bill period 1 March 2023 to 31 March 2023\n"
        "Total amount due 10.00"
    )
    store.submit_approval(
        "t-closed",
        run_id,
        "ledger_write",
        "gate",
        {
            "entry_id": "late",
            "amount": 10.0,
            "currency": "AUD",
            "source_doc": "doc-late",
        },
        {
            "task": "post the closed bill",
            "proposed": {"amount": 10.0},
            "citations": [{"doc_id": "doc-late", "text": text, "query": ""}],
        },
    )

    res = TestClient(api.app).post(
        "/approvals/t-closed/decide",
        json={"approved": True, "decided_by": "alice"},
    )

    assert res.status_code == 409
    assert "closed" in res.json()["detail"]
    assert store.get_approval("t-closed")["decided"] == 0
    assert len(store.list_ledger()) == 1


def test_generic_export_filters_by_period_and_scope(isolated_db) -> None:
    first = _post(KEY_A, "doc-a")
    second = _post(KEY_B, "doc-b", 20.0)
    _post("acct:333333|period:2023-04-01:2023-04-30", "doc-april", 30.0)
    store.reverse_ledger_entry(first["id"], "wrong period", "alice")
    client = TestClient(api.app)

    response = client.get("/export/ledger.csv?period=2023-03")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'ledger-2023-03.csv' in response.headers["content-disposition"]

    rows = _rows(response)
    assert [row["entry_kind"] for row in rows] == [
        "posting",
        "posting",
        "reversal",
    ]
    assert {row["accounting_period"] for row in rows} == {MARCH}
    assert rows[0]["period_basis"] == "bill period end"
    assert rows[0]["amount"] == "10.00"
    assert rows[0]["period_closed"] == "false"

    outstanding = _rows(
        client.get("/export/ledger.csv?period=2023-03&scope=outstanding")
    )
    assert [row["ledger_id"] for row in outstanding] == [str(second["id"])]
    assert outstanding[0]["outstanding"] == "true"


def test_xero_export_needs_mapping_and_exports_outstanding_only(
    isolated_db,
) -> None:
    first = _post(KEY_A, "doc-a")
    _post(KEY_B, "doc-b", 20.0)
    store.reverse_ledger_entry(first["id"], "duplicate", "alice")
    client = TestClient(api.app)

    missing = client.get("/export/xero.csv?period=2023-03")
    assert missing.status_code == 400
    detail = missing.json()["detail"]
    assert "contact" in detail
    assert "account_code" in detail
    assert "tax_type" in detail

    response = client.get(
        "/export/xero.csv?period=2023-03&contact=AGL&account_code=400"
        "&tax_type=INPUT&due_days=14"
    )
    assert response.status_code == 200, response.text
    rows = _rows(response)
    assert list(rows[0]) == [
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
    ]
    assert len(rows) == 1
    assert rows[0]["InvoiceNumber"] == "doc-b"
    assert rows[0]["InvoiceDate"] == "2023-03-30"
    assert rows[0]["DueDate"] == "2023-04-13"
    assert rows[0]["UnitAmount"] == "20.00"
    assert rows[0]["ContactName"] == "AGL"


def test_period_writes_are_gated_but_exports_stay_open(
    isolated_db, monkeypatch
) -> None:
    monkeypatch.setenv("READ_ONLY_DEMO", "1")
    monkeypatch.delenv("APP_TOKEN", raising=False)
    client = TestClient(api.app)

    assert client.get("/periods").status_code == 200
    assert client.get("/export/ledger.csv").status_code == 200
    assert client.get("/export/xero.csv").status_code == 400

    assert client.post(
        f"/periods/{MARCH}/close",
        json={"note": "month end", "decided_by": "alice"},
    ).status_code == 403
    assert client.post(
        f"/periods/{MARCH}/reopen",
        json={"note": "reopen", "decided_by": "alice"},
    ).status_code == 403
