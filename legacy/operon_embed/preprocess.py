"""Preprocessing pipeline: shrinkage whitening, PCA, and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


@dataclass(slots=True)
class PreprocessorArtifacts:
    """Container for serialized preprocessing state."""

    mean: np.ndarray
    whitener: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    output_dim: int


def _validate_matrix(x: ArrayLike) -> np.ndarray:
    """Ensure input is a two-dimensional float64 matrix."""

    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Input embeddings must be a 2-D array")
    if array.shape[0] < 2:
        raise ValueError("Need at least two samples to fit preprocessing")
    return array


def fit_preprocessor(
    x: ArrayLike, dims_out: int, eps: float = 1e-5
) -> PreprocessorArtifacts:
    """Fit whitening + PCA transforms on embedding matrix ``x``."""

    data = _validate_matrix(x)
    input_dim = data.shape[1]

    if dims_out <= 0:
        raise ValueError("dims_out must be positive")
    if dims_out > input_dim:
        raise ValueError("dims_out cannot exceed input dimensionality")

    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean

    cov_estimator = LedoitWolf(assume_centered=False)
    cov_estimator.fit(centered)
    covariance = cov_estimator.covariance_

    evals, evecs = np.linalg.eigh(covariance)
    adjusted = np.maximum(evals, 0.0) + eps
    inv_sqrt = np.diag(1.0 / np.sqrt(adjusted))
    whitener = (evecs @ inv_sqrt) @ evecs.T

    whitened = centered @ whitener

    pca = PCA(n_components=dims_out, svd_solver="full", random_state=0)
    pca.fit(whitened)

    return PreprocessorArtifacts(
        mean=mean.ravel().copy(),
        whitener=whitener.copy(),
        pca_components=pca.components_.copy(),
        pca_mean=pca.mean_.copy(),
        output_dim=dims_out,
    )


def transform_preprocessor(
    x: ArrayLike, artifacts: PreprocessorArtifacts
) -> np.ndarray:
    """Apply fitted preprocessor to embeddings."""
    data = _validate_matrix(x)

    if data.shape[1] != artifacts.mean.shape[0]:
        raise ValueError(
            "Input dimensionality does not match fitted preprocessor",
        )

    centered = data - artifacts.mean
    whitened = centered @ artifacts.whitener
    projected = (whitened - artifacts.pca_mean) @ artifacts.pca_components.T

    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return projected / norms


def save_preprocessor(artifacts: PreprocessorArtifacts) -> Dict[str, Any]:
    """Serialize artifacts for persistence."""
    return {
        "mean": artifacts.mean,
        "whitener": artifacts.whitener,
        "pca_components": artifacts.pca_components,
        "pca_mean": artifacts.pca_mean,
        "output_dim": artifacts.output_dim,
    }


def load_preprocessor(path: str | Path) -> PreprocessorArtifacts:
    """Load artifacts produced by :func:`save_preprocessor`."""

    resolved = Path(path).expanduser().resolve()
    data = joblib.load(resolved)
    required = {"mean", "whitener", "pca_components", "pca_mean", "output_dim"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Preprocessor file missing keys: {', '.join(sorted(missing))}")
    return PreprocessorArtifacts(
        mean=np.asarray(data["mean"], dtype=np.float64),
        whitener=np.asarray(data["whitener"], dtype=np.float64),
        pca_components=np.asarray(data["pca_components"], dtype=np.float64),
        pca_mean=np.asarray(data["pca_mean"], dtype=np.float64),
        output_dim=int(data["output_dim"]),
    )
