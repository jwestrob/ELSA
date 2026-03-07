"""Shared helpers for tests that rely on real datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

EMBEDDING_CANDIDATES: tuple[tuple[str, Optional[str]], ...] = (
    ("embeddings.npy", None),
    ("gene_embeddings.npy", None),
    ("embeddings.npz", "embeddings"),
    ("gene_embeddings.npz", "embeddings"),
)


def load_real_embeddings(root: Path) -> Optional[np.ndarray]:
    """Load embeddings from a dataset root, if available."""

    for fname, key in EMBEDDING_CANDIDATES:
        candidate = root / fname
        if not candidate.exists():
            continue
        if candidate.suffix == ".npz":
            with np.load(candidate) as archive:  # type: ignore[no-untyped-call]
                target_key = key or archive.files[0]
                if target_key in archive:
                    return np.asarray(archive[target_key], dtype=np.float64)
        else:
            return np.asarray(np.load(candidate), dtype=np.float64)
    return None
