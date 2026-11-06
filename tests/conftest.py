import pytest

from agent import limits


@pytest.fixture(autouse=True)
def _no_inbox_watcher(monkeypatch):
    """Tests must never scan the live inbox or touch the live corpus."""
    monkeypatch.setenv("APP_INBOX_WATCH", "0")


@pytest.fixture(autouse=True)
def _fresh_abuse_limits():
    """Rate limits are process state; every test starts from zero."""
    limits.clear_all()
    yield
    limits.clear_all()
