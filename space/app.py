"""Hugging Face Space launcher for the read-only demo snapshot.

The Space uses the Gradio SDK as its Python runtime, but the app at the root
is the same FastAPI service as the repository. Writes stay disabled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("READ_ONLY_DEMO", "1")
os.environ.setdefault("APP_INBOX_WATCH", "0")
os.environ.setdefault("APP_DB", str(ROOT / "demo" / "ledger.db"))

import gradio as gr  # noqa: E402
from scripts.api import app  # noqa: E402

gr.mount_gradio_app(
    app,
    gr.Blocks(title="Ledgerproof"),
    path="/_gradio",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )
