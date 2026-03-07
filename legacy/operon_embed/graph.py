"""Graph construction utilities."""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import igraph as ig
import numpy as np


def _validate_inputs(
    embeddings: np.ndarray,
    sinkhorn_costs: np.ndarray,
    k: int,
    prune_threshold: float,
    lambda_cosine: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, int, float, float, float]:
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")

    costs = np.asarray(sinkhorn_costs, dtype=np.float64)
    if costs.shape != (emb.shape[0], emb.shape[0]):
        raise ValueError("sinkhorn_costs must be a square matrix matching embeddings")

    if k <= 0:
        raise ValueError("k must be positive")
    if prune_threshold < 0:
        raise ValueError("prune_threshold must be non-negative")
    if not (0.0 <= lambda_cosine <= 1.0):
        raise ValueError("lambda_cosine must lie in [0, 1]")
    if tau <= 0:
        raise ValueError("tau must be positive")

    return emb, costs, int(k), float(prune_threshold), float(lambda_cosine), float(tau)


def _top_k_indices(row: np.ndarray, k: int) -> Sequence[int]:
    if k >= row.size:
        k = row.size - 1
    if k <= 0:
        return []
    idx = np.argpartition(row, -k)[-k:]
    return idx[np.argsort(row[idx])[::-1]]


def build_knn_graph(
    embeddings: np.ndarray,
    sinkhorn_costs: np.ndarray,
    k: int,
    prune_threshold: float,
    *,
    lambda_cosine: float = 0.7,
    tau: float = 0.1,
) -> ig.Graph:
    """Construct a reciprocal kNN graph with weighted edges."""

    emb, costs, k, prune_threshold, lambda_cosine, tau = _validate_inputs(
        embeddings, sinkhorn_costs, k, prune_threshold, lambda_cosine, tau
    )

    n_vertices = emb.shape[0]
    if n_vertices == 0:
        return ig.Graph()

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = emb / norms
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)

    costs = np.where(np.isfinite(costs), costs, np.inf)
    sinkhorn_similarity = np.exp(-costs / tau)
    sinkhorn_similarity[costs == np.inf] = 0.0

    combined = lambda_cosine * cosine + (1.0 - lambda_cosine) * sinkhorn_similarity
    np.fill_diagonal(combined, 0.0)

    neighbour_sets: List[Set[int]] = []
    for i in range(n_vertices):
        row = combined[i]
        top_idx = _top_k_indices(row, k)
        neighbours = {
            int(j)
            for j in top_idx
            if j != i and row[int(j)] >= prune_threshold and not np.isnan(row[int(j)])
        }
        neighbour_sets.append(neighbours)

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for i in range(n_vertices):
        for j in neighbour_sets[i]:
            if j <= i:
                continue
            if i in neighbour_sets[j]:
                weight = float((combined[i, j] + combined[j, i]) / 2.0)
                if weight >= prune_threshold:
                    edges.append((i, j))
                    weights.append(weight)

    graph = ig.Graph()
    graph.add_vertices(n_vertices)
    if edges:
        graph.add_edges(edges)
        graph.es["weight"] = weights

    return graph
