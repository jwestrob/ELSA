"""Tests for shingle embeddings (Milestone 3)."""

from __future__ import annotations

import numpy as np
import pytest

from operon_embed.preprocess import fit_preprocessor, transform_preprocessor
from operon_embed.shingle import ShingleResult, build_shingles, vectorize_shingle

from tests._helpers import load_real_embeddings


@pytest.mark.usefixtures("operon_test_data")
def test_shingle_permutation_invariance(operon_test_data):
    embeddings = load_real_embeddings(operon_test_data)
    if embeddings is None or embeddings.shape[0] < 32:
        pytest.skip("Insufficient embeddings to evaluate shingle invariance")

    sample = embeddings[:32]
    dims_out = min(16, sample.shape[1], sample.shape[0] - 1)
    if dims_out < 2:
        pytest.skip("Not enough rank for PCA components")

    artifacts = fit_preprocessor(sample, dims_out=dims_out)
    transformed = transform_preprocessor(sample, artifacts)

    genes = transformed[:10]
    positions = np.arange(10)

    vec1 = vectorize_shingle(genes, positions)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(genes))
    vec2 = vectorize_shingle(genes[perm], positions[perm])

    assert np.allclose(vec1, vec2, atol=1e-6)

    modified = genes.copy()
    modified[0] = transformed[10]
    vec3 = vectorize_shingle(modified, positions)
    cosine = float(np.dot(vec1, vec3))
    assert cosine < 0.999


@pytest.mark.usefixtures("operon_test_data")
def test_build_shingles_norms_and_indices(operon_test_data):
    embeddings = load_real_embeddings(operon_test_data)
    if embeddings is None or embeddings.shape[0] < 20:
        pytest.skip("Insufficient embeddings to build shingles")

    sample = embeddings[:20]
    dims_out = min(8, sample.shape[1], sample.shape[0] - 1)
    if dims_out < 2:
        pytest.skip("Not enough rank for PCA components")

    artifacts = fit_preprocessor(sample, dims_out=dims_out)
    transformed = transform_preprocessor(sample, artifacts)

    result: ShingleResult = build_shingles([transformed], k=4, stride=2)
    expected = ((transformed.shape[0] - 4) // 2) + 1

    assert result.vectors.shape[0] == expected
    if expected > 0:
        norms = np.linalg.norm(result.vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        assert len(result.gene_indices) == expected
        first_start, first_end = result.gene_indices[0]
        assert first_start == 0 and first_end == 3
        last_start, last_end = result.gene_indices[-1]
        assert last_end - last_start == 3
