"""Phase 3 ledger guarantees: one bill, one identity; corrections are rows.

These tests pin the two remaining ledger risks from the handoff:
a bill posted under a second document id is still the same bill, and a
correction is a signed negative row rather than an edit or delete.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.api as api
from agent import store
from agent.approvals import ApprovalQueue
from agent.bill_identity import identify_bill
from agent.ledger_flow import guarded_ledger_write
from agent.states import AgentState, Evidence
from agent.tools import LedgerArgs, LedgerWriteTool

FIXTURES = Path(__file__).parent / "fixtures" / "corpus.json"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    yield str(tmp_path / "t.db")


def _fixture(doc_id: str) -> str:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))[doc_id]


@pytest.mark.parametrize(
    ("live_doc", "fixture_doc", "amount"),
    [
        ("agl-better-bills-sample-bill-necf", "bill-agl-apr23", 63.70),
        (
            "aer---better-bills---illustrative-example-bill---paginated",
            "bill-retailer-mar22",
            79.87,
        ),
    ],
)
def test_one_bill_under_two_doc_ids_is_one_ledger_identity(
    isolated_db, live_doc: str, fixture_doc: str, amount: float
) -> None:
    key = identify_bill(fixture_doc, _fixture(fixture_doc)).key
    first = store.create_run("t")
    second = store.create_run("t")
    store.insert_ledger_entry(
        first, "live-label", amount, "AUD", live_doc, bill_key=key
    )

    with pytest.raises(store.DuplicateLedgerEntry):
        store.insert_ledger_entry(
            second, "fixture-label", amount, "AUD", fixture_doc, bill_key=key
        )

    assert len(store.list_ledger()) == 1


def test_a_second_doc_id_duplicate_never_reaches_a_human(isolated_db) -> None:
    text = _fixture("bill-agl-apr23")
    key = identify_bill("bill-agl-apr23", text).key
    seed = store.create_run("seed")
    store.insert_ledger_entry(
        seed,
        "agl-live-label",
        63.70,
        "AUD",
        "agl-better-bills-sample-bill-necf",
        bill_key=key,
    )

    state = AgentState(
        task="agl bill for april 2023",
        evidence=[Evidence(doc_id="bill-agl-apr23", text=text)],
    )
    ledger = LedgerWriteTool(
        posted=lambda args: store.find_ledger_posting(
            args.source_doc,
            args.amount,
            args.currency,
            bill_key=args.bill_key or "",
        )
    )
    queue = ApprovalQueue()

    out = guarded_ledger_write(
        state,
        ledger,
        LedgerArgs(
            entry_id="agl-second-label", amount=63.70, source_doc="bill-agl-apr23"
        ),
        queue,
    )

    assert out.startswith("rejected: already posted"), out
    assert ledger.entries == []
    assert len(store.list_ledger()) == 1


def _post(run_id: str, entry_id: str = "agl-q3", amount: float = 63.70) -> dict:
    key = identify_bill("bill-agl-apr23", _fixture("bill-agl-apr23")).key
    store.insert_ledger_entry(
        run_id, entry_id, amount, "AUD", "bill-agl-apr23", bill_key=key
    )
    return store.list_ledger()[-1]


def test_a_legacy_ledger_migrates_to_signed_rows(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy))
    try:
        conn.executescript(
            """
            CREATE TABLE ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                source_doc TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO ledger
                (entry_id, run_id, amount, currency, source_doc, created_at)
            VALUES ('old-label', 'r-old', 63.70, 'AUD', 'bill-agl-apr23', 'now');
            """
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("APP_DB", str(legacy))
    store.init_db()

    rows = store.list_ledger()
    assert rows[0]["kind"] == "posting"
    assert rows[0]["bill_key"] == ""
    assert rows[0]["reversed"] is False
    reversal = store.reverse_ledger_entry(rows[0]["id"], "legacy correction", "alice")
    assert reversal["amount"] == -63.70


def test_reversal_appends_a_signed_row_and_keeps_the_original(isolated_db) -> None:
    original = _post(store.create_run("recon"))

    reversal = store.reverse_ledger_entry(original["id"], "wrong period", "alice")

    assert reversal["kind"] == "reversal"
    assert reversal["amount"] == -63.70
    assert reversal["reverses_id"] == original["id"]
    assert reversal["reason"] == "wrong period"
    assert reversal["approved_by"] == "alice"

    rows = store.list_ledger()
    assert len(rows) == 2
    assert rows[0]["amount"] == 63.70
    assert rows[0]["reversed"] is True
    assert rows[1]["amount"] == -63.70


def test_a_reversed_bill_can_be_posted_again(isolated_db) -> None:
    original = _post(store.create_run("recon"))
    key = original["bill_key"]
    store.reverse_ledger_entry(original["id"], "wrong period", "alice")
    assert store.find_ledger_posting("bill-agl-apr23", 63.70, "AUD", bill_key=key) is None

    second = store.create_run("recon")
    store.insert_ledger_entry(
        second, "agl-corrected", 63.71, "AUD", "bill-agl-apr23", bill_key=key
    )

    assert len(store.list_ledger()) == 3
    found = store.find_ledger_posting("bill-agl-apr23", 63.71, "AUD", bill_key=key)
    assert found is not None and found["entry_id"] == "agl-corrected"


def test_a_posting_can_only_be_reversed_once(isolated_db) -> None:
    original = _post(store.create_run("recon"))
    store.reverse_ledger_entry(original["id"], "wrong period", "alice")

    with pytest.raises(store.AlreadyReversed):
        store.reverse_ledger_entry(original["id"], "second try", "bob")

    assert len(store.list_ledger()) == 2


def test_a_reversal_row_cannot_be_reversed(isolated_db) -> None:
    original = _post(store.create_run("recon"))
    reversal = store.reverse_ledger_entry(original["id"], "wrong period", "alice")

    with pytest.raises(store.NotReversible):
        store.reverse_ledger_entry(reversal["id"], "undo the correction", "bob")

    assert len(store.list_ledger()) == 2


def test_api_reversal_keeps_the_original_and_appends_a_negative_row(
    isolated_db,
) -> None:
    original = _post(store.create_run("recon"))
    client = TestClient(api.app)

    res = client.post(
        f"/ledger/{original['id']}/reverse",
        json={"reason": "wrong period", "approved_by": "alice"},
    )

    assert res.status_code == 200, res.text
    body = res.json()["entry"]
    assert body["kind"] == "reversal"
    assert body["amount"] == -63.70
    assert body["reverses_id"] == original["id"]

    entries = client.get("/ledger").json()["entries"]
    assert len(entries) == 2
    assert entries[0]["reversed"] is True
    assert entries[1]["kind"] == "reversal"
    assert entries[1]["amount"] == -63.70


def test_api_reversal_is_one_shot_and_validates_input(isolated_db) -> None:
    original = _post(store.create_run("recon"))
    client = TestClient(api.app)

    ok = client.post(
        f"/ledger/{original['id']}/reverse",
        json={"reason": "wrong period", "approved_by": "alice"},
    )
    again = client.post(
        f"/ledger/{original['id']}/reverse",
        json={"reason": "try again", "approved_by": "bob"},
    )
    missing = client.post(
        "/ledger/9999/reverse",
        json={"reason": "no such row", "approved_by": "alice"},
    )
    blank = client.post(
        f"/ledger/{original['id']}/reverse",
        json={"reason": "x", "approved_by": "alice"},
    )

    assert ok.status_code == 200
    assert again.status_code == 409
    assert missing.status_code == 404
    assert blank.status_code == 422
    assert len(store.list_ledger()) == 2


def test_api_approval_refuses_a_bill_already_posted_under_another_doc_id(
    isolated_db,
) -> None:
    text = _fixture("bill-agl-apr23")
    key = identify_bill("bill-agl-apr23", text).key
    seed = store.create_run("seed")
    store.insert_ledger_entry(
        seed,
        "agl-live-label",
        63.70,
        "AUD",
        "agl-better-bills-sample-bill-necf",
        bill_key=key,
    )

    run_id = store.create_run("agl bill for april 2023")
    store.set_run_status(run_id, "awaiting_approval", 2)
    packet = {
        "task": "agl bill for april 2023",
        "proposed": {"amount": 63.70, "source_doc": "bill-agl-apr23"},
        "rule": "test gate",
        "checks": {"grounding": "grounded"},
        "citations": [{"doc_id": "bill-agl-apr23", "text": text, "query": ""}],
    }
    store.submit_approval(
        "t-crossdoc",
        run_id,
        "ledger_write",
        "test gate",
        {
            "entry_id": "agl-second-label",
            "amount": 63.70,
            "currency": "AUD",
            "source_doc": "bill-agl-apr23",
            "bill_key": key,
        },
        packet,
    )

    res = TestClient(api.app).post(
        "/approvals/t-crossdoc/decide", json={"approved": True, "decided_by": "alice"}
    )

    assert res.status_code == 409, res.text
    assert len(store.list_ledger()) == 1
