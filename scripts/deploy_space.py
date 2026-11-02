"""Assemble and upload the Hugging Face Space bundle.

Usage (repo root):
    python scripts/deploy_space.py --build
    python scripts/deploy_space.py --upload jordan6/enterprise-agent-orchestrator

The Space is a filtered copy of this repository plus the read-only launcher in
``space/``. It never contains ``.env``, ``HANDOFF.md``, the live database, tests,
or git metadata.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD = ROOT / ".tmp" / "space-build"

FILES = (
    "space/README.md",
    "space/app.py",
    "space/requirements.txt",
    "scripts/__init__.py",
    "scripts/api.py",
    "agent",
    "static",
    "config",
    "data/corpus.json",
    "demo/ledger.db",
)
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.lock")
FORBIDDEN = (".env", "HANDOFF.md", "app.db", ".git")


def _copy(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, ignore=IGNORE)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build(destination: Path = DEFAULT_BUILD) -> Path:
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for item in FILES:
        source = ROOT / item
        if not source.exists():
            raise SystemExit(f"missing source: {item}")
        name = source.name if item.startswith("space/") else item
        _copy(source, destination / name)

    for path in destination.rglob("*"):
        if path.name in FORBIDDEN or path.suffix == ".env":
            raise SystemExit(f"refusing to package {path}")

    if not (destination / "demo" / "ledger.db").exists():
        raise SystemExit("demo snapshot is missing")
    return destination


def manifest(destination: Path) -> tuple[list[str], int]:
    files = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    )
    size = sum((destination / name).stat().st_size for name in files)
    return files, size


def upload(destination: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(destination),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy the read-only accounting agent demo",
    )
    return str(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or upload the Hugging Face Space.")
    parser.add_argument("--build", action="store_true", help="assemble the bundle")
    parser.add_argument("--upload", metavar="REPO_ID", help="build and upload to a Space")
    parser.add_argument("--dir", default=str(DEFAULT_BUILD), help="bundle directory")
    args = parser.parse_args(argv)

    if not args.build and not args.upload:
        parser.error("choose --build or --upload REPO_ID")

    destination = build(Path(args.dir))
    files, size = manifest(destination)
    print(f"bundle {destination}")
    print(f"files  {len(files)}")
    print(f"bytes  {size}")
    for name in files:
        print(f"  {name}")

    if args.upload:
        url = upload(destination, args.upload)
        print(f"space  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
