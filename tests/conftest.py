import pytest


@pytest.fixture(autouse=True)
def _no_inbox_watcher(monkeypatch):
    """Tests must never scan the live inbox or touch the live corpus."""
    monkeypatch.setenv("APP_INBOX_WATCH", "0")
