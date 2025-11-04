"""Order-invariant shingle feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


@dataclass(slots=True)
class ShingleResult:
    """Return type for shingle construction."""

    vectors: np.ndarray
    gene_indices: List[Tuple[int, int]]


def vectorize_shingle(
    genes: Union[Sequence[np.ndarray], np.ndarray],
    positions: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Produce an order-invariant embedding for a single shingle.

    Parameters
    ----------
    genes:
        Sequence of gene embeddings already transformed by the preprocessing
        pipeline. Each entry must have identical dimensionality.
    positions:
        Relative gene positions (e.g., indices along the contig). Order is
        arbitrary; positions are used only through invariant statistics.
    """

    if len(genes) == 0:
        raise ValueError("Cannot build a shingle from zero genes")
    if len(genes) != len(positions):
        raise ValueError("Number of genes must match number of positions")

    gene_matrix = np.asarray(genes, dtype=np.float64)
    if gene_matrix.ndim != 2:
        raise ValueError("Gene embeddings must be a 2-D array")

    pos_array = np.asarray(positions, dtype=np.float64)
    order = np.argsort(pos_array)
    gene_matrix = gene_matrix[order]
    pos_array = pos_array[order]

    pos_rel = pos_array - pos_array[0]
    if pos_rel.size > 1:
        span = pos_rel[-1]
        if span > 0:
            pos_rel = pos_rel / span

    mu = gene_matrix.mean(axis=0)
    var = gene_matrix.var(axis=0) + 1e-8
    log_var = np.log(var)
    max_vec = gene_matrix.max(axis=0)

    p_mean = np.array([float(pos_rel.mean())])
    p_sq = np.array([float(np.mean(pos_rel**2))])

    cross1 = np.array([float(np.mean(pos_rel * gene_matrix[:, 0]))])
    if gene_matrix.shape[1] > 1:
        cross2_val = float(np.mean(pos_rel * gene_matrix[:, 1]))
    else:
        cross2_val = 0.0
    cross2 = np.array([cross2_val])

    feature = np.concatenate([mu, log_var, max_vec, p_mean, p_sq, cross1, cross2])
    norm = np.linalg.norm(feature)
    if norm == 0.0:
        return feature
    return feature / norm


def build_shingles(
    contig_embeddings: Iterable[np.ndarray],
    k: int,
    stride: int,
) -> ShingleResult:
    """Build shingles across contig-ordered embeddings."""

    if k <= 0:
        raise ValueError("k must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    vectors: List[np.ndarray] = []
    indices: List[Tuple[int, int]] = []
    global_offset = 0

    for contig in contig_embeddings:
        matrix = np.asarray(contig, dtype=np.float64)
        num_genes = matrix.shape[0]
        if num_genes < k:
            global_offset += num_genes
            continue

        for start in range(0, num_genes - k + 1, stride):
            end = start + k
            genes = matrix[start:end]
            positions = np.arange(start, end, dtype=np.float64)
            vec = vectorize_shingle(genes, positions)
            vectors.append(vec)
            indices.append((global_offset + start, global_offset + end - 1))

        global_offset += num_genes

    if vectors:
        stacked = np.vstack(vectors)
    else:
        stacked = np.empty((0, 0), dtype=np.float64)

    return ShingleResult(vectors=stacked, gene_indices=indices)
