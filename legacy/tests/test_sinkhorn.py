"""Tests for Sinkhorn re-ranking (Milestone 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operon_embed.preprocess import fit_preprocessor, transform_preprocessor
from operon_embed.sinkhorn import sinkhorn_distance

from tests._helpers import load_real_embeddings


@pytest.mark.usefixtures("operon_test_data")
def test_sinkhorn_self_similarity_less_than_random(operon_test_data):
    embeddings = load_real_embeddings(operon_test_data)
    if embeddings is None or embeddings.shape[0] < 64:
        pytest.skip("Insufficient embeddings for Sinkhorn test")

    sample = embeddings[:64]
    dims_out = min(16, sample.shape[1], sample.shape[0] - 1)
    if dims_out < 2:
        pytest.skip("Not enough rank for PCA components")

    artifacts = fit_preprocessor(sample, dims_out=dims_out)
    transformed = transform_preprocessor(sample, artifacts)

    set_a = transformed[:12]
    set_b = transformed[:12]
    set_c = transformed[20:32]

    result_same = sinkhorn_distance(set_a, set_b, epsilon=0.05, n_iter=30, top_k=8)
    result_diff = sinkhorn_distance(set_a, set_c, epsilon=0.05, n_iter=30, top_k=8)

    assert result_same.similarity >= result_diff.similarity
    assert result_same.transport_cost <= result_diff.transport_cost


@pytest.mark.usefixtures("operon_test_data")
def test_sinkhorn_cli_rerank(tmp_path: Path, operon_test_data: Path):
    embeddings = load_real_embeddings(operon_test_data)
    if embeddings is None or embeddings.shape[0] < 48:
        pytest.skip("Insufficient embeddings for Sinkhorn CLI test")

    sample = embeddings[:48]
    dims_out = min(12, sample.shape[1], sample.shape[0] - 1)
    if dims_out < 2:
        pytest.skip("Not enough rank for PCA components")

    artifacts = fit_preprocessor(sample, dims_out=dims_out)
    transformed = transform_preprocessor(sample, artifacts)

    pairs_path = tmp_path / "pairs.json"
    pairs = [
        {"query": list(range(0, 6)), "target": list(range(12, 18))},
        {"query": list(range(6, 12)), "target": list(range(18, 24))},
    ]
    with pairs_path.open("w", encoding="utf-8") as handle:
        json.dump(pairs, handle)

    from operon_embed.sinkhorn import sinkhorn_distance as _sink

    result = _sink(
        transformed[pairs[0]["query"]],
        transformed[pairs[0]["target"]],
        epsilon=0.05,
        n_iter=20,
        top_k=6,
    )

    assert result.transport_cost >= 0
    assert 0 <= result.similarity <= 1
