"""Write access: open locally, token-gated, or a read-only public demo."""

from fastapi.testclient import TestClient

import scripts.api as api


def _client(tmp_path, monkeypatch, *, read_only: bool, token: str | None):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    if read_only:
        monkeypatch.setenv("READ_ONLY_DEMO", "1")
    else:
        monkeypatch.delenv("READ_ONLY_DEMO", raising=False)
    if token:
        monkeypatch.setenv("APP_TOKEN", token)
    else:
        monkeypatch.delenv("APP_TOKEN", raising=False)
    return TestClient(api.app)


def test_read_only_demo_blocks_writes_but_serves_reads(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, read_only=True, token=None)

    assert client.get("/config").json()["mode"] == "read_only"
    assert client.get("/ledger").status_code == 200
    assert client.get("/reconciliation").status_code == 200

    writes = [
        client.post("/run", json={"task": "post the bill"}),
        client.post("/admin/reset", json={"confirm": True}),
        client.post(
            "/ledger/1/reverse",
            json={"reason": "wrong period", "approved_by": "alice"},
        ),
        client.post("/runs/missing/cancel"),
        client.post("/approvals/missing/decide", json={"approved": True}),
    ]

    assert [res.status_code for res in writes] == [403] * len(writes)
    assert "read-only demo" in writes[0].json()["detail"]


def test_a_configured_token_is_required_for_writes(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, read_only=False, token="s3cret")

    assert client.get("/config").json() == {
        "mode": "token",
        "read_only": False,
        "auth_required": True,
    }
    missing = client.post("/admin/reset", json={"confirm": True})
    wrong = client.post(
        "/admin/reset", json={"confirm": True}, headers={"X-API-Token": "nope"}
    )
    accepted = client.post(
        "/admin/reset",
        json={"confirm": True},
        headers={"X-API-Token": "s3cret"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200


def test_the_owner_token_unlocks_a_read_only_demo(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch, read_only=True, token="s3cret")

    assert client.get("/config").json()["mode"] == "owner"
    denied = client.post("/admin/reset", json={"confirm": True})
    accepted = client.post(
        "/admin/reset",
        json={"confirm": True},
        headers={"X-API-Token": "s3cret"},
    )

    assert denied.status_code == 403
    assert accepted.status_code == 200
