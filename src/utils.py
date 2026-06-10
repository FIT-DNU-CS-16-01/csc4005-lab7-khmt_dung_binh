from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists (create if needed)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    """Ensure the parent directory of a file path exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    """Save an object as a JSON file."""
    ensure_parent(path)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path):
    """Load a JSON file and return the parsed object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_size_mb(path: str | Path) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def percent_change(before: float, after: float) -> float:
    """Calculate percentage change from before to after."""
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_size(size_mb: float) -> str:
    """Format model size with appropriate units."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.2f} MB"
