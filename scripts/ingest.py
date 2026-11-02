"""Inbox ingest CLI. The same logic backs ``POST /ingest`` and the watcher.

Scans ``data/inbox`` (or ``APP_INBOX``) and merges .txt, .md, and .pdf files
into ``data/corpus.json`` (or ``APP_CORPUS``). Identity is computed at ingest
and duplicate bills are reported before any agent runs.

Usage (repo root):
    python scripts/ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.ingest import ingest_once  # noqa: E402


def main() -> dict:
    report = ingest_once()
    print(
        f"inbox={report['inbox']} added={len(report['added'])}"
        f" updated={len(report['updated'])}"
        f" total_docs={report['total_docs']}"
    )
    for item in report["skipped"]:
        print(f"skipped {item['file']} with {item['reason']}")
    for item in report["collisions"]:
        ignored = ", ".join(item["ignored"])
        print(
            f"collision {item['doc_id']}: kept {item['kept']},"
            f" ignored {ignored}"
        )
    for item in report["identity_duplicates"]:
        docs = ", ".join(item["doc_ids"])
        print(f"duplicate bill {item['bill_key']}: {docs}")
    return report


if __name__ == "__main__":
    main()
