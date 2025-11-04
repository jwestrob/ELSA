"""Sinkhorn-based set-to-set similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SinkhornResult:
    """Output of Sinkhorn re-ranking."""

    transport_cost: float
    similarity: float
    transport_plan: np.ndarray


def sinkhorn_distance(
    set_a: np.ndarray,
    set_b: np.ndarray,
    epsilon: float,
    n_iter: int,
    top_k: int,
) -> SinkhornResult:
    """Compute an entropic OT distance between two shingle gene sets."""
    if set_a.ndim != 2 or set_b.ndim != 2:
        raise ValueError("Input sets must be 2-D arrays")
    if set_a.shape[1] != set_b.shape[1]:
        raise ValueError("Gene embeddings must have matching dimensionality")
    if set_a.size == 0 or set_b.size == 0:
        raise ValueError("Gene sets must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    a = set_a.astype(np.float64, copy=False)
    b = set_b.astype(np.float64, copy=False)

    cost = 1.0 - np.clip(a @ b.T, -1.0, 1.0)

    if top_k < cost.shape[1]:
        idx = np.argsort(cost, axis=1)[:, :top_k]
        mask = np.full(cost.shape, np.inf)
        for row, cols in enumerate(idx):
            mask[row, cols] = cost[row, cols]
        cost = mask

    r = np.ones(cost.shape[0]) / cost.shape[0]
    c = np.ones(cost.shape[1]) / cost.shape[1]

    K = np.exp(-cost / epsilon)
    K = np.where(np.isfinite(K), K, 0.0)

    u = np.ones_like(r)
    v = np.ones_like(c)

    for _ in range(n_iter):
        K_v = K @ v
        K_v[K_v == 0.0] = 1.0
        u = r / K_v

        K_t_u = K.T @ u
        K_t_u[K_t_u == 0.0] = 1.0
        v = c / K_t_u

    transport = np.outer(u, v) * K
    transport_cost = float(np.sum(transport * cost))
    similarity = float(np.exp(-transport_cost / max(epsilon, 1e-8)))

    return SinkhornResult(
        transport_cost=transport_cost,
        similarity=similarity,
        transport_plan=transport,
    )
