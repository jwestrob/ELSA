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
        similarities, labels = index.search(normalized.astype(np.float32), k_query)
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

    anchors = []
    seen: Set[Tuple[int, int]] = set()

    for i in range(n):
        genome_i = genome_arr[i]
        for j_pos in range(k_query):
            j = int(labels[i, j_pos])
            if j < 0:  # FAISS returns -1 when no result found
                continue
            sim = float(similarities[i, j_pos])

            if sim < similarity_threshold:
                continue
            if i == j:
                continue

            genome_j = genome_arr[j]
            if genome_i == genome_j:
                continue

            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # Compute relative orientation from strand info
            if strand_arr is not None:
                si = int(strand_arr[i]) if strand_arr[i] != 0 else 1
                sj = int(strand_arr[j]) if strand_arr[j] != 0 else 1
                rel_orient = 1 if si == sj else -1
            else:
                rel_orient = 0  # Unknown

            anchor = GeneAnchor(
                query_idx=int(pos_arr[i]),
                target_idx=int(pos_arr[j]),
                query_genome=str(genome_i),
                target_genome=str(genome_j),
                query_contig=str(contig_arr[i]),
                target_contig=str(contig_arr[j]),
                query_gene_id=str(gene_id_arr[i]),
                target_gene_id=str(gene_id_arr[j]),
                similarity=sim,
                orientation=rel_orient,
            )
            anchors.append(anchor)

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
