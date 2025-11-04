"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


class DatasetNotFoundError(FileNotFoundError):
    """Raised when required dataset resources are absent."""


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load gene embeddings stored in NumPy-compatible formats."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise DatasetNotFoundError(f"Embeddings file not found: {resolved}")
    return np.load(resolved)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_npz(data: Dict[str, Any], path: str | Path) -> None:
    """Persist arrays in NumPy NPZ format."""
    resolved = Path(path).expanduser().resolve()
    np.savez_compressed(resolved, **data)
