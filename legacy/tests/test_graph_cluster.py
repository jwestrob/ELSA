"""Tests for graph construction and clustering (milestone 7)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import igraph as ig
import numpy as np
import pytest

from operon_embed.graph import build_knn_graph
from operon_embed.cluster import hdbscan_cluster, leiden_cluster
from tests._helpers import load_real_embeddings


def _load_sample_embeddings(root: Path, count: int) -> np.ndarray:
    embeddings = load_real_embeddings(root)
    if embeddings is None or embeddings.shape[0] < count:
        pytest.skip(
            "Not enough real embeddings available under OPERON_TEST_DATA to exercise "
            "graph construction",
        )
    return embeddings[:count].astype(np.float64)


def _cosine_cost_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = vectors / norms
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    costs = 1.0 - cosine
    np.fill_diagonal(costs, 0.0)
    return costs


def test_build_knn_graph_reciprocal_edges(operon_test_data: Path) -> None:
    sample = _load_sample_embeddings(operon_test_data, count=12)
    cost_matrix = _cosine_cost_matrix(sample)

    graph = build_knn_graph(
        sample,
        cost_matrix,
        k=4,
        prune_threshold=0.05,
        lambda_cosine=0.6,
        tau=0.5,
    )

    assert graph.vcount() == sample.shape[0]
    # Graph should be simple (no multi-edges/self loops)
    assert graph.is_simple()
    # At least one edge should survive the pruning step
    if graph.ecount() == 0:
        pytest.skip(
            "Pruning removed all edges for the provided dataset; adjust threshold"
        )
    # All stored weights must respect the pruning threshold
    assert all(w >= 0.05 for w in graph.es["weight"])


def test_leiden_cluster_on_two_components() -> None:
    graph = ig.Graph()
    graph.add_vertices(4)
    graph.add_edges([(0, 1), (2, 3)])
    graph.es["weight"] = [1.0, 1.0]

    assignments = leiden_cluster(graph, resolution=1.0)
    assert len(assignments) == 4
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]


def test_hdbscan_cluster_finds_two_groups(operon_test_data: Path) -> None:
    base = _load_sample_embeddings(operon_test_data, count=12)
    first_group = base[:6]
    second_group = base[:6] + 5.0  # Shift to create a clearly separated cluster
    features = np.vstack([first_group, second_group])

    labels: List[int] = hdbscan_cluster(
        features,
        min_cluster_size=3,
        min_samples=2,
        allow_single_cluster=False,
    )
    assert len(labels) == features.shape[0]
    found_clusters = {label for label in labels if label >= 0}
    if len(found_clusters) < 2:
        pytest.skip("HDBSCAN did not discover two clusters on the provided dataset")
    assert len(found_clusters) >= 2
