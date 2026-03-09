"""
Anchor seeding: find similar gene pairs across genomes.

Provides GeneAnchor dataclass, cross-genome anchor discovery via kNN,
and contig-pair grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any

import numpy as np
import pandas as pd


@dataclass
class GeneAnchor:
    """A similar gene pair across two genomes."""
    query_idx: int           # Gene index in query contig
    target_idx: int          # Gene index in target contig
    query_genome: str
    target_genome: str
    query_contig: str
    target_contig: str
    query_gene_id: str
    target_gene_id: str
    similarity: float
    orientation: int = 0     # Set during chaining: +1 forward, -1 inverted


def find_cross_genome_anchors(
    index_tuple: Tuple[str, Any],
    embeddings: np.ndarray,
    gene_info: pd.DataFrame,
    k: int = 50,
    similarity_threshold: float = 0.85,
) -> List[GeneAnchor]:
    """
    Find similar gene pairs across genomes using kNN search.

    Args:
        index_tuple: (type, index) from build_gene_index
        embeddings: (n_genes, dim) array of gene embeddings
        gene_info: DataFrame with columns [gene_id, sample_id, contig_id, position_index]
        k: Number of neighbors to retrieve per gene
        similarity_threshold: Minimum cosine similarity for anchors

    Returns:
        List of GeneAnchor objects representing cross-genome similar pairs
    """
    index_type, index = index_tuple
    if index is None or len(gene_info) == 0:
        return []

    n = len(gene_info)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)

    k_query = min(k + 10, n)

    if index_type == "hnsw":
        labels, distances = index.knn_query(normalized, k=k_query)
        similarities = 1 - distances
    elif index_type == "faiss":
        import os, faiss as _faiss
        _ncpu = os.cpu_count() or 1
        _faiss.omp_set_num_threads(_ncpu)
        query = normalized.astype(np.float32)
        if n >= 200_000:
            from tqdm import trange
            CHUNK = 100_000
            similarities = np.empty((n, k_query), dtype=np.float32)
            labels = np.empty((n, k_query), dtype=np.int64)
            for start in trange(0, n, CHUNK, desc="FAISS kNN search"):
                end = min(start + CHUNK, n)
                similarities[start:end], labels[start:end] = index.search(query[start:end], k_query)
        else:
            similarities, labels = index.search(query, k_query)
        _faiss.omp_set_num_threads(1)
        labels = labels.astype(np.intp)
    else:  # sklearn
        distances, labels = index.kneighbors(normalized, n_neighbors=k_query)
        similarities = 1 - distances

    # Build lookup arrays
    genome_arr = gene_info['sample_id'].values
    contig_arr = gene_info['contig_id'].values
    pos_arr = gene_info['position_index'].values
    gene_id_arr = gene_info['gene_id'].values
    strand_arr = gene_info['strand'].values if 'strand' in gene_info.columns else None

    # Vectorized anchor discovery — avoids O(n*k) Python loop
    import sys
    print(f"[GeneAnchor] Building anchors vectorized (n={n}, k={k_query})...",
          file=sys.stderr, flush=True)

    # Flatten labels/similarities to 1D
    row_idx = np.repeat(np.arange(n), k_query)       # query gene index
    col_idx = labels.ravel()                          # neighbor gene index
    sims = similarities.ravel()                       # similarity scores

    # Filter: valid labels, above threshold, not self
    valid = (col_idx >= 0) & (sims >= similarity_threshold) & (row_idx != col_idx)
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]
    sims = sims[valid]

    # Filter: cross-genome only
    cross = genome_arr[row_idx] != genome_arr[col_idx]
    row_idx = row_idx[cross]
    col_idx = col_idx[cross]
    sims = sims[cross]

    # Deduplicate: canonical pair (min, max) — keep first occurrence (highest sim from sorted kNN)
    lo = np.minimum(row_idx, col_idx)
    hi = np.maximum(row_idx, col_idx)

    # Use structured array for unique pair detection
    pairs = np.empty(len(lo), dtype=[('lo', np.int64), ('hi', np.int64)])
    pairs['lo'] = lo
    pairs['hi'] = hi
    _, unique_idx = np.unique(pairs, return_index=True)
    unique_idx.sort()  # preserve original order (highest sims first)

    row_idx = row_idx[unique_idx]
    col_idx = col_idx[unique_idx]
    sims = sims[unique_idx]

    print(f"[GeneAnchor] {len(row_idx)} unique cross-genome pairs after filtering",
          file=sys.stderr, flush=True)

    # Compute orientations vectorized
    if strand_arr is not None:
        si = strand_arr[row_idx].copy()
        sj = strand_arr[col_idx].copy()
        si[si == 0] = 1
        sj[sj == 0] = 1
        orientations = np.where(si == sj, 1, -1)
    else:
        orientations = np.zeros(len(row_idx), dtype=np.int64)

    # Build GeneAnchor list
    anchors = []
    q_pos = pos_arr[row_idx]
    t_pos = pos_arr[col_idx]
    q_genome = genome_arr[row_idx]
    t_genome = genome_arr[col_idx]
    q_contig = contig_arr[row_idx]
    t_contig = contig_arr[col_idx]
    q_gene = gene_id_arr[row_idx]
    t_gene = gene_id_arr[col_idx]

    for idx in range(len(row_idx)):
        anchors.append(GeneAnchor(
            query_idx=int(q_pos[idx]),
            target_idx=int(t_pos[idx]),
            query_genome=str(q_genome[idx]),
            target_genome=str(t_genome[idx]),
            query_contig=str(q_contig[idx]),
            target_contig=str(t_contig[idx]),
            query_gene_id=str(q_gene[idx]),
            target_gene_id=str(t_gene[idx]),
            similarity=float(sims[idx]),
            orientation=int(orientations[idx]),
        ))

    return anchors


def group_anchors_by_contig_pair(
    anchors: List[GeneAnchor],
) -> Dict[Tuple[str, str, str, str], List[GeneAnchor]]:
    """
    Group anchors by (query_genome, target_genome, query_contig, target_contig).

    Uses canonical ordering (smaller genome first) to avoid duplicates.
    """
    groups: Dict[Tuple[str, str, str, str], List[GeneAnchor]] = {}

    for anchor in anchors:
        if anchor.query_genome <= anchor.target_genome:
            key = (anchor.query_genome, anchor.target_genome,
                   anchor.query_contig, anchor.target_contig)
            groups.setdefault(key, []).append(anchor)
        else:
            swapped = GeneAnchor(
                query_idx=anchor.target_idx,
                target_idx=anchor.query_idx,
                query_genome=anchor.target_genome,
                target_genome=anchor.query_genome,
                query_contig=anchor.target_contig,
                target_contig=anchor.query_contig,
                query_gene_id=anchor.target_gene_id,
                target_gene_id=anchor.query_gene_id,
                similarity=anchor.similarity,
                orientation=anchor.orientation,
            )
            key = (swapped.query_genome, swapped.target_genome,
                   swapped.query_contig, swapped.target_contig)
            groups.setdefault(key, []).append(swapped)

    return groups
