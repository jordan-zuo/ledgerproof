import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


import scripts.api as api
from agent import store
from agent.ingest import InboxWatcher, ingest_once


def _bill(account: str, amount: str) -> str:
    return (
        f"Account number: {account}\n"
        "Billing period: 1 March 2023 to 31 March 2023\n"
        f"Total amount due: {amount}\n"
    )


@pytest.fixture()
def ingestor(tmp_path, monkeypatch) -> tuple[Path, Path]:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    corpus = tmp_path / "corpus.json"
    monkeypatch.setenv("APP_INBOX", str(inbox))
    monkeypatch.setenv("APP_CORPUS", str(corpus))
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    store.init_db()
    return inbox, corpus


def _corpus(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ingest_records_identity_and_is_idempotent(ingestor) -> None:
    inbox, corpus_file = ingestor
    text = _bill("12345678", "10.00")
    (inbox / "Bill One.txt").write_text(text, encoding="utf-8")

    first = ingest_once()

    assert first["added"] == ["bill-one"]
    assert _corpus(corpus_file) == {"bill-one": text.strip()}
    row = store.list_documents()[0]
    assert row["identity_basis"] == "account+period"
    assert row["bill_key"] == "acct:12345678|period:2023-03-01:2023-03-31"
    assert row["source_name"] == "Bill One.txt"

    second = ingest_once()
    assert second["added"] == []
    assert second["unchanged"] == ["bill-one"]

    (inbox / "Bill One.txt").unlink()
    third = ingest_once()
    assert third["total_docs"] == 1
    assert "bill-one" in _corpus(corpus_file)
    assert store.list_documents()[0]["source_name"] == "Bill One.txt"


def test_same_text_under_one_doc_id_is_a_duplicate_source(ingestor) -> None:
    inbox, _ = ingestor
    text = _bill("55555555", "9.00")
    (inbox / "Bill.md").write_text(text, encoding="utf-8")
    (inbox / "bill.txt").write_text(text, encoding="utf-8")

    report = ingest_once()

    assert report["collisions"] == []
    assert report["duplicate_sources"] == [
        {
            "doc_id": "bill",
            "kept": "Bill.md",
            "file": "bill.txt",
            "reason": "same text under one doc id",
        }
    ]


def test_conflicting_files_under_one_doc_id_do_not_overwrite_silently(
    ingestor,
) -> None:
    inbox, corpus_file = ingestor
    (inbox / "Same Bill.md").write_text(
        _bill("11111111", "20.00"), encoding="utf-8"
    )
    (inbox / "same-bill.txt").write_text(
        _bill("11111111", "30.00"), encoding="utf-8"
    )

    report = ingest_once()

    assert report["added"] == ["same-bill"]
    collision = report["collisions"][0]
    assert collision["doc_id"] == "same-bill"
    assert collision["kept"] == "Same Bill.md"
    assert collision["ignored"] == ["same-bill.txt"]
    assert "20.00" in _corpus(corpus_file)["same-bill"]


def test_same_bill_under_two_doc_ids_is_visible_before_a_run(ingestor) -> None:
    inbox, _ = ingestor
    (inbox / "bill-a.txt").write_text(_bill("22222222", "5.00"), encoding="utf-8")
    (inbox / "bill-b.txt").write_text(_bill("22222222", "6.00"), encoding="utf-8")

    report = ingest_once()

    duplicate = report["identity_duplicates"][0]
    assert duplicate["doc_ids"] == ["bill-a", "bill-b"]
    assert duplicate["bill_key"] == "acct:22222222|period:2023-03-01:2023-03-31"

    body = TestClient(api.app).get("/documents").json()
    assert body["counts"] == {"documents": 2, "duplicate_identities": 1}
    assert all(doc["duplicate"] for doc in body["documents"])


def test_unsupported_empty_and_unreadable_files_are_reported(ingestor) -> None:
    inbox, _ = ingestor
    (inbox / "notes.csv").write_text("a,b\n", encoding="utf-8")
    (inbox / "empty.txt").write_text("", encoding="utf-8")
    (inbox / "broken.pdf").write_bytes(b"not a pdf")

    report = ingest_once()

    reasons = {item["file"]: item["reason"] for item in report["skipped"]}
    assert reasons["notes.csv"] == "unsupported file type"
    assert reasons["empty.txt"] == "empty document"
    assert reasons["broken.pdf"].startswith("PDF read failed")
    assert report["total_docs"] == 0


def test_ingest_route_is_write_gated(ingestor, monkeypatch) -> None:
    client = TestClient(api.app)
    monkeypatch.setenv("READ_ONLY_DEMO", "1")
    assert client.post("/ingest").status_code == 403

    monkeypatch.delenv("READ_ONLY_DEMO")
    monkeypatch.setenv("APP_TOKEN", "secret-token")
    assert client.post("/ingest").status_code == 401

    allowed = client.post("/ingest", headers={"X-API-Token": "secret-token"})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["total_docs"] == 0


def test_watcher_ingests_a_new_file_without_a_request(ingestor) -> None:
    inbox, corpus_file = ingestor
    watcher = InboxWatcher(interval=0.2)
    watcher.start()
    try:
        (inbox / "later-bill.txt").write_text(
            _bill("33333333", "7.00"), encoding="utf-8"
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if corpus_file.exists() and "later-bill" in _corpus(corpus_file):
                return
            time.sleep(0.05)
        raise AssertionError("watcher never ingested the new file")
    finally:
        watcher.stop()


def test_ledger_reset_keeps_document_metadata(ingestor) -> None:
    inbox, _ = ingestor
    (inbox / "bill.txt").write_text(_bill("44444444", "8.00"), encoding="utf-8")
    ingest_once()

    store.archive_and_clear()

    assert store.table_counts()["documents"] == 1
    assert store.list_documents()[0]["doc_id"] == "bill"


def test_source_file_endpoint_serves_the_original_file(ingestor) -> None:
    inbox, _ = ingestor
    source = inbox / "Bill One.pdf"
    payload = b"%PDF-1.4\n% provenance check\n"
    source.write_bytes(payload)
    store.upsert_document(
        "bill-one",
        source_name=source.name,
        content_hash="hash",
        size_bytes=len(payload),
        modified_ns=1,
        bill_key="acct:12345678|period:2023-03-01:2023-03-31",
        identity_basis="account+period",
    )

    client = TestClient(api.app)
    res = client.get("/documents/bill-one/file")

    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content == payload
    assert client.get("/documents/missing/file").status_code == 404
