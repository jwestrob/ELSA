"""Evaluation helpers for the operon pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn import metrics


def pair_metrics(
    similarities: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Summarize pairwise similarities.

    Parameters
    ----------
    similarities:
        Array of pairwise similarity scores (higher values imply a closer match).
    labels:
        Optional binary ground-truth labels aligned with ``similarities`` where
        ``1`` denotes a positive pair. When labels are omitted, the function
        returns descriptive statistics only.
    """

    if similarities.ndim != 1:
        raise ValueError("similarities must be a 1-D array")

    report: Dict[str, float] = {
        "count": float(similarities.size),
        "mean_similarity": float(np.mean(similarities)) if similarities.size else 0.0,
        "std_similarity": float(np.std(similarities)) if similarities.size else 0.0,
        "min_similarity": float(np.min(similarities)) if similarities.size else 0.0,
        "max_similarity": float(np.max(similarities)) if similarities.size else 0.0,
    }

    if labels is None:
        return report

    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array")
    if labels.size != similarities.size:
        raise ValueError("labels and similarities must have the same length")

    positives = int(np.sum(labels > 0))
    negatives = int(labels.size - positives)
    report.update(
        {
            "positives": float(positives),
            "negatives": float(negatives),
        }
    )

    # scikit-learn raises ValueError when all examples belong to the same class.
    try:
        report["average_precision"] = float(
            metrics.average_precision_score(labels, similarities)
        )
    except ValueError:
        report["average_precision"] = float("nan")

    try:
        report["roc_auc"] = float(metrics.roc_auc_score(labels, similarities))
    except ValueError:
        report["roc_auc"] = float("nan")

    precision, recall, _ = metrics.precision_recall_curve(labels, similarities)
    report["precision_at_recall_50"] = float(
        _precision_at_recall_threshold(precision, recall, target=0.5)
    )

    return report


def cluster_metrics(
    assignments: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Summarize clustering assignments and optional supervision."""

    if assignments.ndim != 1:
        raise ValueError("assignments must be a 1-D array")

    num_points = assignments.size
    unique, counts = np.unique(assignments, return_counts=True)
    summary: Dict[str, float] = {
        "num_points": float(num_points),
        "num_clusters": float(unique.size),
        "num_singletons": float(int(np.sum(counts == 1))),
        "max_cluster_size": float(int(counts.max())) if counts.size else 0.0,
        "min_cluster_size": float(int(counts.min())) if counts.size else 0.0,
        "mean_cluster_size": float(np.mean(counts)) if counts.size else 0.0,
    }

    if labels is None:
        return summary

    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array")
    if labels.size != assignments.size:
        raise ValueError("labels and assignments must have the same length")

    try:
        summary["adjusted_rand"] = float(
            metrics.adjusted_rand_score(labels, assignments)
        )
    except ValueError:
        summary["adjusted_rand"] = float("nan")

    try:
        summary["adjusted_mutual_info"] = float(
            metrics.adjusted_mutual_info_score(labels, assignments)
        )
    except ValueError:
        summary["adjusted_mutual_info"] = float("nan")

    return summary


def _precision_at_recall_threshold(
    precision: np.ndarray,
    recall: np.ndarray,
    target: float,
) -> float:
    """Return the maximum precision achieved at or above a recall threshold."""

    if precision.size == 0 or recall.size == 0:
        return float("nan")
    mask = recall >= target
    if not np.any(mask):
        return float("nan")
    return float(np.max(precision[mask]))
