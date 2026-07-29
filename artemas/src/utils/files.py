from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_directories(paths: dict[str, Any]) -> None:
    for value in _walk_values(paths):
        if isinstance(value, Path):
            if value.suffix:
                value.parent.mkdir(parents=True, exist_ok=True)
            else:
                value.mkdir(parents=True, exist_ok=True)


def bootstrap_workspace(paths: dict[str, Any]) -> list[Path]:
    """Create the generated workspace for a clean checkout.

    The repository intentionally does not need to ship ``data/`` or ``artifacts/``.
    This function creates both roots and every configured parent/output directory before
    logging, manifests, downloads, or preprocessing begin.
    """
    project_root = Path(paths["project_root"]).resolve()
    generated_roots = [project_root / "data", project_root / "artifacts"]
    missing_roots = [path for path in generated_roots if not path.exists()]
    ensure_directories(paths)
    for path in generated_roots:
        path.mkdir(parents=True, exist_ok=True)
    return missing_roots


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def save_npz(path: str | Path, **arrays: Any) -> None:
    destination = ensure_parent(path)
    np.savez_compressed(destination, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def write_frame(frame: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    """Write Parquet when available, otherwise use a pickle fallback at the same path."""
    destination = ensure_parent(path)
    payload = frame if index else frame.reset_index(drop=True)
    try:
        payload.to_parquet(destination, index=index)
    except (ImportError, ModuleNotFoundError):
        payload.to_pickle(destination)


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    try:
        return pd.read_parquet(source)
    except (ImportError, ModuleNotFoundError, ValueError):
        return pd.read_pickle(source)
