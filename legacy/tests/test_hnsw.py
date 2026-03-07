"""Tests for HNSW index building (Milestone 4)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from operon_embed.index_hnsw import build_hnsw_index
except ModuleNotFoundError:  # pragma: no cover
    build_hnsw_index = None  # type: ignore[assignment]

from operon_embed.preprocess import fit_preprocessor, transform_preprocessor
from operon_embed.shingle import build_shingles

from tests._helpers import load_real_embeddings


@pytest.mark.usefixtures("operon_test_data")
def test_hnsw_self_neighbor(operon_test_data):
    if build_hnsw_index is None:
        pytest.skip("hnswlib not available in environment")
    embeddings = load_real_embeddings(operon_test_data)
    if embeddings is None or embeddings.shape[0] < 24:
        pytest.skip("Insufficient embeddings for HNSW test")

    sample = embeddings[:24]
    dims_out = min(16, sample.shape[1], sample.shape[0] - 1)
    if dims_out < 2:
        pytest.skip("Not enough rank for PCA components")

    artifacts = fit_preprocessor(sample, dims_out=dims_out)
    transformed = transform_preprocessor(sample, artifacts)

    shingles = build_shingles([transformed], k=4, stride=2)
    vectors = shingles.vectors
    if vectors.size == 0:
        pytest.skip("No shingles generated for HNSW test")

    index = build_hnsw_index(vectors, m=16, ef_construction=100, ef_search=64)
    nbrs, _ = index.query(vectors[:5], k=1)
    assert np.array_equal(nbrs.ravel(), np.arange(5))
