from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def seed_everything(seed: int, strict: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if preference not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device preference: {preference}")
    return torch.device("cpu")


def configure_logging(level: str, log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(destination, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def batches(sequence: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]
