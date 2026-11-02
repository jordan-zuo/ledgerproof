from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException

TOKEN_HEADER = "X-API-Token"


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def configured_token() -> str:
    return os.environ.get("APP_TOKEN", "").strip()


def read_only_demo() -> bool:
    return _enabled(os.environ.get("READ_ONLY_DEMO"))


def access_state() -> dict[str, object]:
    """Describe who may write, without ever echoing the token itself.

    open       no token, writes allowed
    token      token configured, writes need the header
    read_only  public demo, every write is refused
    owner      public demo plus a token, so the owner can still drive it
    """
    token = bool(configured_token())
    demo = read_only_demo()
    if demo and token:
        mode = "owner"
    elif demo:
        mode = "read_only"
    elif token:
        mode = "token"
    else:
        mode = "open"
    return {"mode": mode, "read_only": demo, "auth_required": token}


def require_write(
    x_api_token: str | None = Header(default=None, alias=TOKEN_HEADER),
) -> None:
    expected = configured_token()
    supplied = bool(x_api_token) and bool(expected) and secrets.compare_digest(
        str(x_api_token).encode("utf-8"), expected.encode("utf-8")
    )
    if read_only_demo() and not supplied:
        raise HTTPException(
            status_code=403,
            detail="read-only demo: this instance only allows viewing",
        )
    if expected and not supplied:
        raise HTTPException(
            status_code=401, detail="invalid or missing X-API-Token"
        )
