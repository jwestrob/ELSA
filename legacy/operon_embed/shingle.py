"""Shingle feature construction for HNSW-based operon detection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(slots=True)
class ShingleResult:
    """Return type for shingle construction."""

    vectors: np.ndarray
    gene_indices: List[Tuple[int, int]]


def vectorize_shingle_ordered(
    genes: Union[Sequence[np.ndarray], np.ndarray],
    positions: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Produce an order-SENSITIVE embedding by concatenating genes in positional order.

    This preserves individual gene identity and order, making shingles more
    discriminative for synteny detection. Strand sensitivity is preserved
    from the input embeddings.

    Parameters
    ----------
    genes:
        Sequence of gene embeddings. Each entry must have identical dimensionality.
    positions:
        Gene positions along the contig. Used to determine order.

    Returns
    -------
    np.ndarray:
        Concatenated gene embeddings in positional order, L2-normalized.
        Dimension = k * D where k is number of genes, D is embedding dim.
    """
    if len(genes) == 0:
        raise ValueError("Cannot build a shingle from zero genes")
    if len(genes) != len(positions):
        raise ValueError("Number of genes must match number of positions")

    gene_matrix = np.asarray(genes, dtype=np.float64)
    if gene_matrix.ndim != 2:
        raise ValueError("Gene embeddings must be a 2-D array")

    pos_array = np.asarray(positions, dtype=np.float64)

    # Sort genes by position to get canonical order
    order = np.argsort(pos_array)
    genes_ordered = gene_matrix[order]

    # Concatenate in order: [gene1 | gene2 | gene3 | ...]
    feature = genes_ordered.flatten()

    # L2 normalize
    norm = np.linalg.norm(feature)
    if norm == 0.0:
        return feature
    return feature / norm


def vectorize_shingle_stats(
    genes: Union[Sequence[np.ndarray], np.ndarray],
    positions: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Produce an order-INVARIANT embedding using summary statistics.

    LEGACY: This loses individual gene identity. Kept for backwards compatibility.

    Parameters
    ----------
    genes:
        Sequence of gene embeddings.
    positions:
        Gene positions (used for position statistics).

    Returns
    -------
    np.ndarray:
        Summary statistics vector [mean, log_var, max, position_stats], L2-normalized.
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


# Default to ordered (new behavior)
vectorize_shingle = vectorize_shingle_ordered


def build_shingles(
    contig_embeddings: Iterable[np.ndarray],
    k: int,
    stride: int,
    method: str = "ordered",
    verbose: bool = True,
) -> ShingleResult:
    """Build shingles across contig-ordered embeddings.

    Parameters
    ----------
    contig_embeddings:
        Iterable of 2D arrays, each representing genes on a contig.
    k:
        Shingle window size (number of genes).
    stride:
        Step size between windows.
    method:
        "ordered" (default) - concatenate embeddings, preserves gene identity
        "stats" - use summary statistics (legacy, loses gene identity)
    verbose:
        Print progress information.

    Returns
    -------
    ShingleResult:
        vectors: (n_shingles, dim) array of shingle embeddings
        gene_indices: list of (start, end) global gene indices for each shingle
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    # Select vectorization method
    if method == "ordered":
        vec_fn: Callable = vectorize_shingle_ordered
    elif method == "stats":
        vec_fn = vectorize_shingle_stats
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ordered' or 'stats'.")

    # Convert to list to allow multiple passes if needed
    contigs = list(contig_embeddings)
    n_contigs = len(contigs)

    # Count total genes and estimate shingles
    total_genes = sum(c.shape[0] for c in contigs)
    estimated_shingles = sum(
        max(0, (c.shape[0] - k) // stride + 1) for c in contigs
    )

    if verbose:
        print(f"Building shingles: k={k}, stride={stride}, method={method}")
        print(f"  Contigs: {n_contigs}, Total genes: {total_genes}")
        print(f"  Estimated shingles: {estimated_shingles}")

    vectors: List[np.ndarray] = []
    indices: List[Tuple[int, int]] = []
    global_offset = 0
    shingles_created = 0
    contigs_processed = 0

    for contig in contigs:
        matrix = np.asarray(contig, dtype=np.float64)
        num_genes = matrix.shape[0]

        if num_genes < k:
            global_offset += num_genes
            contigs_processed += 1
            continue

        for start in range(0, num_genes - k + 1, stride):
            end = start + k
            genes = matrix[start:end]
            positions = np.arange(start, end, dtype=np.float64)
            vec = vec_fn(genes, positions)
            vectors.append(vec)
            indices.append((global_offset + start, global_offset + end - 1))
            shingles_created += 1

        global_offset += num_genes
        contigs_processed += 1

        # Progress update every 10 contigs or at the end
        if verbose and (contigs_processed % 10 == 0 or contigs_processed == n_contigs):
            print(f"  Progress: {contigs_processed}/{n_contigs} contigs, "
                  f"{shingles_created} shingles", file=sys.stderr)

    if vectors:
        stacked = np.vstack(vectors)
    else:
        stacked = np.empty((0, 0), dtype=np.float64)

    if verbose:
        dim = stacked.shape[1] if stacked.size > 0 else 0
        print(f"  Complete: {shingles_created} shingles, dimension={dim}")

    return ShingleResult(vectors=stacked, gene_indices=indices)
