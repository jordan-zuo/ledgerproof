"""One command to start the service, with no container runtime.

    python run.py                 # http://127.0.0.1:8000/ui
    python run.py --check         # validate the install and exit
    python run.py --read-only     # serve a public demo where writes are 403

Hosts set PORT and a start command of ``python run.py --host 0.0.0.0``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the orchestrator or check its install.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000") or "8000"),
        help="port (default $PORT or 8000)",
    )
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    parser.add_argument("--read-only", action="store_true", help="refuse every write")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument("--check", action="store_true", help="print the install report and exit")
    return parser.parse_args(argv)


def _corpus_count() -> int:
    path = os.environ.get("APP_CORPUS", "").strip()
    corpus = Path(path) if path else ROOT / "data" / "corpus.json"
    try:
        data = json.loads(corpus.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return len(data) if isinstance(data, dict) else 0


def _pool_report() -> list[str]:
    from agent.models import PoolError, describe_pool

    lines = []
    try:
        for role in ("agent", "judge"):
            rows = describe_pool(role)
            ready = sum(1 for row in rows if row["key_set"])
            lines.append(f"models  {role}: {len(rows)} spec(s), {ready} with a key")
    except PoolError as exc:
        lines.append(f"models  error: {exc}")
    return lines


def preflight() -> int:
    from agent import store
    from agent.access import access_state

    db = store.db_path()
    corpus = _corpus_count()
    ui = (ROOT / "static" / "index.html").exists()
    state = access_state()

    print(f"python  {sys.version.split()[0]}")
    print(f"root    {ROOT}")
    print(f"db      {db}")
    print(f"corpus  {corpus} document(s)")
    print(f"ui      {'present' if ui else 'MISSING'}")
    print(f"access  {state['mode']} (read_only={state['read_only']}, auth_required={state['auth_required']})")
    for line in _pool_report():
        print(line)

    store.init_db()
    return 0 if corpus and ui else 1


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = _parse_args(argv)
    if args.read_only:
        os.environ["READ_ONLY_DEMO"] = "1"

    if args.check:
        return preflight()

    from agent import store
    from agent.access import access_state

    store.init_db()
    state = access_state()
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{shown_host}:{args.port}/ui"
    print(f"access  {state['mode']} (read_only={state['read_only']}, auth_required={state['auth_required']})")
    print(f"serving {url}")
    print("stop    Ctrl+C")

    local = shown_host in {"127.0.0.1", "localhost", "::1"}
    if local and not args.no_open and sys.stdout.isatty():
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    import uvicorn

    uvicorn.run(
        "scripts.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
