"""Community detection interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import igraph as ig
import numpy as np


def leiden_cluster(graph: ig.Graph, resolution: float) -> List[int]:
    """Run Leiden community detection with sensible fallbacks."""

    if not isinstance(graph, ig.Graph):
        raise TypeError("graph must be an igraph.Graph")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    n_vertices = graph.vcount()
    if n_vertices == 0:
        return []
    if graph.ecount() == 0:
        return list(range(n_vertices))

    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None

    try:
        partition = graph.community_leiden(
            objective_function="modularity",
            resolution=resolution,
            weights=weights,
        )
        membership = partition.membership
        if membership is None:
            raise ValueError("Leiden returned no membership vector")
        return [int(x) for x in membership]
    except Exception:
        # Fall back to singleton clusters if Leiden fails (e.g. disconnected graph)
        return list(range(n_vertices))


def hdbscan_cluster(
    features: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    allow_single_cluster: bool = True,
) -> List[int]:
    """Fallback clustering using HDBSCAN on feature space."""

    import hdbscan  # Imported lazily to avoid heavy dependency at module import

    data = np.asarray(features, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("features must be a non-empty 2-D array")

    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        allow_single_cluster=allow_single_cluster,
    )
    labels = clusterer.fit_predict(data)
    return labels.tolist()


@dataclass(slots=True)
class ClusterSummary:
    """Simple container used by evaluation utilities."""

    assignments: List[int]
    metadata: Dict[str, Any]
