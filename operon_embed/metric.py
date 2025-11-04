"""Optional linear metric learning hooks."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


class LinearMetric:
    """Encapsulate a linear transformation learned from shingle pairs."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return vectors @ self.matrix.T


def fit_linear_metric(
    positives: Iterable[Tuple[int, int]],
    negatives: Iterable[Tuple[int, int]],
    embeddings: np.ndarray,
) -> LinearMetric:
    """Placeholder for metric learning implementation."""
    raise NotImplementedError
