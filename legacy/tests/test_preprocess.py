"""Tests for preprocessing utilities (milestone 2)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from operon_embed.preprocess import fit_preprocessor, transform_preprocessor

from tests._helpers import load_real_embeddings


def test_preprocess_whitens_and_normalizes(operon_test_data: Path) -> None:
    embeddings = load_real_embeddings(operon_test_data)
    if embeddings is None:
        pytest.skip(
            "No embeddings file found under OPERON_TEST_DATA; expected one of "
            "embeddings.npy, gene_embeddings.npy, embeddings.npz, gene_embeddings.npz",
        )

    # Reduce memory footprint for enormous datasets
    sample = embeddings[: min(len(embeddings), 512)]
    if sample.shape[0] < 2 or sample.shape[1] < 2:
        pytest.skip("Not enough data to evaluate preprocessing")

    dims_out = min(32, sample.shape[1], sample.shape[0] - 1)
    if dims_out <= 1:
        pytest.skip("Insufficient rank to evaluate PCA output")

    artifacts = fit_preprocessor(sample, dims_out=dims_out)

    transformed = transform_preprocessor(sample, artifacts)
    assert transformed.shape == (sample.shape[0], dims_out)

    norms = np.linalg.norm(transformed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    centered = sample - artifacts.mean
    whitened = centered @ artifacts.whitener
    covariance = np.cov(whitened, rowvar=False)

    diag = np.diag(covariance)
    assert np.allclose(np.mean(diag), 1.0, atol=0.15)

    off_diag = covariance - np.diag(diag)
    assert np.max(np.abs(off_diag)) < 0.25
