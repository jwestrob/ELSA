"""
Overlap-based clustering of syntenic blocks.

Groups blocks that share genomic regions (gene overlap) using
union-find with mutual top-k filtering.
"""

from __future__ import annotations

import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter

import pandas as pd

from .chain import ChainedBlock


def cluster_blocks_by_overlap(
    blocks: List[ChainedBlock],
    jaccard_tau: float = 0.3,
    mutual_k: int = 5,
    min_genome_support: int = 2,
) -> Tuple[Dict[int, int], pd.DataFrame]:
    """
    Cluster blocks based on shared genomic regions (gene overlap).

    Two blocks are connected if they share genes in any genome.
    This allows blocks from different genome pairs to cluster together
    when they represent the same conserved syntenic region.

    Args:
        blocks: List of ChainedBlock objects
        jaccard_tau: Minimum Jaccard similarity for overlap edges
        mutual_k: Mutual top-k parameter for edge filtering
        min_genome_support: Minimum genomes per cluster

    Returns:
        block_to_cluster: mapping from block_id to cluster_id
        clusters_df: DataFrame with cluster metadata
    """
    if not blocks:
        return {}, pd.DataFrame(columns=["cluster_id", "size", "genome_support",
                                          "mean_chain_length", "genes_json"])

    # Build gene -> block mapping
    gene_to_blocks: Dict[Tuple[str, str, int], Set[int]] = defaultdict(set)
    block_genes: Dict[int, Set[Tuple[str, str, int]]] = {}
    block_map = {b.block_id: b for b in blocks}

    for block in blocks:
        genes = set()
        for idx in range(block.query_start, block.query_end + 1):
            key = (block.query_genome, block.query_contig, idx)
            genes.add(key)
            gene_to_blocks[key].add(block.block_id)
        for idx in range(block.target_start, block.target_end + 1):
            key = (block.target_genome, block.target_contig, idx)
            genes.add(key)
            gene_to_blocks[key].add(block.block_id)
        block_genes[block.block_id] = genes

    # Build overlap graph
    edges_by_u: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for bid, genes in block_genes.items():
        candidates: Counter = Counter()
        for gene in genes:
            for other_bid in gene_to_blocks[gene]:
                if other_bid != bid:
                    candidates[other_bid] += 1

        for other_bid, shared_count in candidates.items():
            if other_bid <= bid:
                continue

            other_genes = block_genes[other_bid]
            intersection = len(genes & other_genes)
            union = len(genes | other_genes)

            if union > 0:
                jaccard = intersection / union
                if jaccard >= jaccard_tau:
                    edges_by_u[bid].append((other_bid, jaccard))
                    edges_by_u[other_bid].append((bid, jaccard))

    # Apply mutual top-k filter
    mutual_edges = _mutual_top_k(edges_by_u, mutual_k)

    # Connected components via union-find
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in mutual_edges:
        union(a, b)

    for bid in block_genes.keys():
        find(bid)

    # Group by root
    components: Dict[int, List[int]] = defaultdict(list)
    for bid in block_genes.keys():
        components[find(bid)].append(bid)

    # Assign cluster IDs with genome support filter
    block_to_cluster: Dict[int, int] = {}
    cluster_rows = []
    cid = 1

    for root, members in components.items():
        genomes: Set[str] = set()
        total_genes = 0
        genes_by_genome: Dict[str, List[str]] = defaultdict(list)

        for bid in members:
            block = block_map[bid]
            genomes.add(block.query_genome)
            genomes.add(block.target_genome)

            for idx in range(block.query_start, block.query_end + 1):
                genes_by_genome[block.query_genome].append(f"{block.query_contig}:{idx}")
            for idx in range(block.target_start, block.target_end + 1):
                genes_by_genome[block.target_genome].append(f"{block.target_contig}:{idx}")

            total_genes += block.n_anchors

        if len(genomes) >= min_genome_support:
            for bid in members:
                block_to_cluster[bid] = cid

            mean_chain_len = total_genes / len(members) if members else 0.0
            cluster_rows.append({
                "cluster_id": cid,
                "size": len(members),
                "genome_support": len(genomes),
                "mean_chain_length": round(mean_chain_len, 2),
                "genes_json": json.dumps({g: list(set(ids)) for g, ids in genes_by_genome.items()}),
            })
            cid += 1
        else:
            for bid in members:
                block_to_cluster[bid] = 0

    clusters_df = pd.DataFrame(cluster_rows) if cluster_rows else pd.DataFrame(
        columns=["cluster_id", "size", "genome_support", "mean_chain_length", "genes_json"]
    )

    return block_to_cluster, clusters_df


def _mutual_top_k(edges_by_u: Dict[int, List[Tuple[int, float]]], k: int) -> Set[Tuple[int, int]]:
    """Return set of undirected edges that are mutual top-k by weight."""
    topk: Dict[int, Set[int]] = {}
    for u, neigh in edges_by_u.items():
        neigh_sorted = sorted(neigh, key=lambda x: x[1], reverse=True)[:k]
        topk[u] = {v for v, _ in neigh_sorted}
    keep: Set[Tuple[int, int]] = set()
    for u, neigh in edges_by_u.items():
        for v, _w in neigh:
            if v in topk.get(u, set()) and u in topk.get(v, set()):
                a, b = (u, v) if u < v else (v, u)
                keep.add((a, b))
    return keep
