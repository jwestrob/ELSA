"""Evaluation helpers for the operon pipeline."""

from __future__ import annotations

from typing import Dict

import numpy as np


def pair_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation metrics for shingle pairs."""
    raise NotImplementedError


def cluster_metrics(
    assignments: np.ndarray, labels: np.ndarray | None = None
) -> Dict[str, float]:
    """Summarize clustering output."""
    raise NotImplementedError
