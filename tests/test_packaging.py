"""Packaging: the launcher reports a usable install and the host config stays native."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_preflight_reports_a_read_only_install(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "APP_DB": str(tmp_path / "check.db"),
            "APP_CORPUS": str(ROOT / "tests" / "fixtures" / "corpus.json"),
            "READ_ONLY_DEMO": "1",
            "APP_TOKEN": "",
        }
    )

    result = subprocess.run(
        [sys.executable, "run.py", "--check"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "access  read_only" in result.stdout
    assert "corpus  11 document(s)" in result.stdout
    assert "ui      present" in result.stdout
    assert (tmp_path / "check.db").exists()


def test_host_config_uses_a_native_python_runtime() -> None:
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    procfile = (ROOT / "Procfile").read_text(encoding="utf-8")
    version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()

    assert "runtime: python" in render
    assert "healthCheckPath: /health" in render
    assert "python run.py --host 0.0.0.0 --no-open" in render
    assert procfile.strip() == "web: python run.py --host 0.0.0.0 --no-open"
    assert version.startswith("3.12")
    assert not (ROOT / "Dockerfile").exists()
    assert "docker" not in render.lower()
    assert "docker" not in procfile.lower()


def test_space_bundle_carries_the_snapshot_and_no_secrets(tmp_path: Path) -> None:
    from scripts.deploy_space import build, manifest

    target = build(tmp_path / "space")
    files, _ = manifest(target)
    app_py = (target / "app.py").read_text(encoding="utf-8")

    assert "app.py" in files
    assert "demo/ledger.db" in files
    assert "data/corpus.json" in files
    assert "static/index.html" in files
    assert "READ_ONLY_DEMO" in app_py
    assert not any(name.endswith(".env") for name in files)
    assert not any("HANDOFF" in name or name.endswith("app.db") for name in files)
    assert not any(name.startswith(".git/") for name in files)
