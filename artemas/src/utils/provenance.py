from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from src.utils.files import ensure_parent, write_json
from src.utils.runtime import select_device


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ""

    commit = run("rev-parse", "HEAD")
    if not commit:
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "yfinance",
        "pyarrow",
        "matplotlib",
        "plotly",
    ]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def snapshot_configuration(
    config: dict[str, Any],
    paths: dict[str, Any],
    source_config: str | Path,
    source_paths: str | Path,
) -> None:
    destination = Path(paths["artifacts"]["config_snapshot_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, destination / "config.yaml")
    shutil.copy2(source_paths, destination / "paths.yaml")
    with (destination / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)


def start_run_manifest(
    config: dict[str, Any],
    paths: dict[str, Any],
    mode: str,
    stages: list[str],
    command: list[str],
) -> dict[str, Any]:
    root = Path(paths["project_root"])
    now = datetime.now(timezone.utc)
    manifest = {
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "status": "running",
        "started_at": now.isoformat(),
        "completed_at": None,
        "mode": mode,
        "stages": stages,
        "command": command,
        "working_directory": str(Path.cwd()),
        "project_root": str(root),
        "config_hash": config_hash(config),
        "configured_seed": int(config["project"]["seed"]),
        "configured_device": str(config["project"]["device"]),
        "resolved_device": str(select_device(str(config["project"]["device"]))),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "git": _git_info(root),
        "environment": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        },
        "stage_status": {stage: "pending" for stage in stages},
        "error": None,
    }
    write_json(paths["artifacts"]["run_manifest"], manifest)
    return manifest


def update_run_manifest(paths: dict[str, Any], manifest: dict[str, Any]) -> None:
    write_json(paths["artifacts"]["run_manifest"], manifest)


def finish_run_manifest(
    paths: dict[str, Any],
    manifest: dict[str, Any],
    status: str,
    error: str | None = None,
) -> None:
    manifest["status"] = status
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["error"] = error
    update_run_manifest(paths, manifest)
