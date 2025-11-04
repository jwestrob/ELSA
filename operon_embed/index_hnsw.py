"""HNSW indexing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import json

import numpy as np


@dataclass(slots=True)
class HNSWIndex:
    """Convenience wrapper around ``hnswlib.Index`` with metadata."""

    index: Any
    dim: int
    space: str

    def query(self, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.knn_query(vectors, k=k)

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(index_path))


def build_hnsw_index(
    vectors: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    space: str = "cosine",
) -> HNSWIndex:
    """Build an HNSW index over the provided vectors."""

    if vectors.ndim != 2:
        raise ValueError("Input vectors must be 2-D")
    num, dim = vectors.shape
    if num == 0:
        raise ValueError("Cannot build index on zero vectors")

    try:
        import hnswlib
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "hnswlib is not installed; install it to build indexes"
        ) from exc

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num, ef_construction=ef_construction, M=m)
    ids = np.arange(num, dtype=np.int32)
    index.add_items(vectors, ids)
    index.set_ef(ef_search)
    return HNSWIndex(index=index, dim=dim, space=space)


def save_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
