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


def merge_contained_clusters(
    block_to_cluster: Dict[int, int],
    blocks: List[ChainedBlock],
    containment_threshold: float = 0.8,
) -> Dict[int, int]:
    """Merge clusters whose genomic footprints are largely contained in a larger cluster.

    For each pair of clusters, restrict to genomes they share, then check if
    the smaller cluster's gene positions on those shared genomes are >= threshold
    contained within the larger cluster. This prevents clusters with blocks in
    many extra genomes from avoiding the merge.

    Returns updated block_to_cluster mapping.
    """
    block_map = {b.block_id: b for b in blocks}

    # Build per-cluster gene footprints and per-genome footprints
    cluster_genes: Dict[int, Set[Tuple[str, str, int]]] = defaultdict(set)
    cluster_genomes: Dict[int, Dict[str, Set[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(set))
    for bid, cid in block_to_cluster.items():
        if cid == 0:
            continue
        block = block_map[bid]
        for idx in range(block.query_start, block.query_end + 1):
            cluster_genes[cid].add((block.query_genome, block.query_contig, idx))
            cluster_genomes[cid][block.query_genome].add((block.query_contig, idx))
        for idx in range(block.target_start, block.target_end + 1):
            cluster_genes[cid].add((block.target_genome, block.target_contig, idx))
            cluster_genomes[cid][block.target_genome].add((block.target_contig, idx))

    if not cluster_genes:
        return block_to_cluster

    # Sort by footprint size descending — process largest first
    sorted_cids = sorted(cluster_genes, key=lambda c: len(cluster_genes[c]), reverse=True)

    # Track merges: child -> parent
    merge_map: Dict[int, int] = {}

    def resolve(c: int) -> int:
        while c in merge_map:
            c = merge_map[c]
        return c

    for i, small_cid in enumerate(sorted_cids):
        small_cid_r = resolve(small_cid)
        small_genes = cluster_genes[small_cid]
        if not small_genes:
            continue

        for large_cid in sorted_cids[:i]:
            large_cid_r = resolve(large_cid)
            if large_cid_r == small_cid_r:
                continue

            large_genes = cluster_genes[large_cid_r]

            # Check containment on shared genomes only
            shared_genomes = set(cluster_genomes[small_cid_r]) & set(cluster_genomes[large_cid_r])
            if not shared_genomes:
                continue

            small_shared = set()
            large_shared = set()
            for g in shared_genomes:
                small_shared.update(cluster_genomes[small_cid_r][g])
                large_shared.update(cluster_genomes[large_cid_r][g])

            if not small_shared:
                continue

            overlap = len(small_shared & large_shared)
            containment = overlap / len(small_shared)

            if containment >= containment_threshold:
                merge_map[small_cid_r] = large_cid_r
                cluster_genes[large_cid_r] = large_genes | small_genes
                # Merge per-genome footprints too
                for g, positions in cluster_genomes[small_cid_r].items():
                    cluster_genomes[large_cid_r][g].update(positions)
                break

    if not merge_map:
        return block_to_cluster

    # Apply merges
    new_mapping = {}
    for bid, cid in block_to_cluster.items():
        if cid == 0:
            new_mapping[bid] = 0
        else:
            new_mapping[bid] = resolve(cid)
    return new_mapping


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
