"""
Concordance test: old sparse-matrix clustering vs new interval-sweep clustering.

Runs both implementations on the S. pneumoniae blocks and verifies that
cluster assignments are equivalent (same connected components, possibly
different cluster IDs).
"""

import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import pytest

from elsa.chain import ChainedBlock


# ---- Old sparse-matrix implementation (copied from commit 2d667bb) ----

def _mutual_top_k_old(edges_by_u, k):
    topk = {}
    for u, neigh in edges_by_u.items():
        neigh_sorted = sorted(neigh, key=lambda x: (-x[1], x[0]))[:k]
        topk[u] = {v for v, _ in neigh_sorted}
    keep = set()
    for u, neigh in edges_by_u.items():
        for v, _w in neigh:
            if v in topk.get(u, set()) and u in topk.get(v, set()):
                a, b = (u, v) if u < v else (v, u)
                keep.add((a, b))
    return keep


def cluster_blocks_sparse(blocks, jaccard_tau=0.3, mutual_k=5, min_genome_support=2):
    """Old O(n²) sparse-matrix clustering implementation."""
    if not blocks:
        return {}, pd.DataFrame(columns=["cluster_id", "size", "genome_support",
                                          "mean_chain_length", "genes_json"])

    gene_to_blocks = defaultdict(set)
    block_genes = {}
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

    edges_by_u = defaultdict(list)

    for bid, genes in block_genes.items():
        candidates = Counter()
        for gene in genes:
            for other_bid in gene_to_blocks[gene]:
                if other_bid != bid:
                    candidates[other_bid] += 1

        for other_bid, shared_count in candidates.items():
            if other_bid <= bid:
                continue

            other_genes = block_genes[other_bid]
            intersection = len(genes & other_genes)
            union_count = len(genes | other_genes)

            if union_count > 0:
                jaccard = intersection / union_count
                if jaccard >= jaccard_tau:
                    edges_by_u[bid].append((other_bid, jaccard))
                    edges_by_u[other_bid].append((bid, jaccard))

    mutual_edges = _mutual_top_k_old(edges_by_u, mutual_k)

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in mutual_edges:
        union(a, b)

    for bid in block_genes.keys():
        find(bid)

    components = defaultdict(list)
    for bid in block_genes.keys():
        components[find(bid)].append(bid)

    block_to_cluster = {}
    cluster_rows = []
    cid = 1

    for root, members in components.items():
        genomes = set()
        total_genes = 0
        genes_by_genome = defaultdict(list)

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


# ---- Concordance helpers ----

def _extract_partitions(block_to_cluster):
    """Convert block_to_cluster dict to a set of frozensets (partition)."""
    by_cid = defaultdict(set)
    for bid, cid in block_to_cluster.items():
        by_cid[cid].add(bid)
    # Return non-singleton, non-zero clusters as frozen sets
    return {frozenset(members) for cid, members in by_cid.items() if cid != 0}


def _load_spneumo_blocks():
    """Load S. pneumoniae blocks from CSV if available."""
    import csv
    from pathlib import Path

    blocks_csv = Path("syntenic_analysis/micro_chain/micro_chain_blocks.csv")
    if not blocks_csv.exists():
        return None

    blocks = []
    with open(blocks_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            blocks.append(ChainedBlock(
                block_id=int(row["block_id"]),
                query_genome=row["query_genome"],
                query_contig=row["query_contig"],
                query_start=int(row["query_start"]),
                query_end=int(row["query_end"]),
                target_genome=row["target_genome"],
                target_contig=row["target_contig"],
                target_start=int(row["target_start"]),
                target_end=int(row["target_end"]),
                orientation=int(row["orientation"]),
                n_anchors=int(row["n_anchors"]),
                chain_score=float(row["chain_score"]),
            ))
    return blocks


# ---- Tests ----

class TestClusterConcordance:
    """Verify interval-sweep produces same clusters as sparse-matrix."""

    @pytest.fixture
    def sample_blocks(self):
        """Small synthetic dataset for exact concordance."""
        blocks = []
        # Blocks on genome A/B, contig c1 — overlapping
        blocks.append(ChainedBlock(
            block_id=1, query_genome="A", query_contig="c1",
            query_start=0, query_end=5, target_genome="B", target_contig="c2",
            target_start=10, target_end=15, orientation=1, n_anchors=6, chain_score=6.0))
        blocks.append(ChainedBlock(
            block_id=2, query_genome="A", query_contig="c1",
            query_start=3, query_end=8, target_genome="C", target_contig="c3",
            target_start=20, target_end=25, orientation=1, n_anchors=6, chain_score=6.0))
        # Block on different contig — no overlap with above
        blocks.append(ChainedBlock(
            block_id=3, query_genome="D", query_contig="c4",
            query_start=100, query_end=105, target_genome="E", target_contig="c5",
            target_start=200, target_end=205, orientation=1, n_anchors=6, chain_score=6.0))
        blocks.append(ChainedBlock(
            block_id=4, query_genome="D", query_contig="c4",
            query_start=102, query_end=107, target_genome="F", target_contig="c6",
            target_start=300, target_end=305, orientation=1, n_anchors=6, chain_score=6.0))
        return blocks

    def test_synthetic_concordance(self, sample_blocks):
        """Both implementations produce identical partitions on synthetic data."""
        from elsa.cluster import cluster_blocks_by_overlap

        old_btc, old_df = cluster_blocks_sparse(sample_blocks)
        new_btc, new_df = cluster_blocks_by_overlap(sample_blocks)

        old_parts = _extract_partitions(old_btc)
        new_parts = _extract_partitions(new_btc)

        assert old_parts == new_parts, (
            f"Partitions differ!\n"
            f"Old: {old_parts}\n"
            f"New: {new_parts}"
        )

        # Also check cluster count
        old_nclusters = len([c for c in set(old_btc.values()) if c != 0])
        new_nclusters = len([c for c in set(new_btc.values()) if c != 0])
        assert old_nclusters == new_nclusters

    def test_spneumo_concordance(self):
        """Both implementations produce identical partitions on real S. pneumoniae data."""
        blocks = _load_spneumo_blocks()
        if blocks is None:
            pytest.skip("S. pneumoniae blocks CSV not found (syntenic_analysis/micro_chain/micro_chain_blocks.csv)")

        from elsa.cluster import cluster_blocks_by_overlap

        print(f"\nRunning concordance test on {len(blocks)} S. pneumoniae blocks...",
              file=sys.stderr)

        old_btc, old_df = cluster_blocks_sparse(blocks)
        new_btc, new_df = cluster_blocks_by_overlap(blocks)

        old_parts = _extract_partitions(old_btc)
        new_parts = _extract_partitions(new_btc)

        old_nclusters = len([c for c in set(old_btc.values()) if c != 0])
        new_nclusters = len([c for c in set(new_btc.values()) if c != 0])

        print(f"Old (sparse): {old_nclusters} clusters, {len(old_parts)} partitions",
              file=sys.stderr)
        print(f"New (sweep):  {new_nclusters} clusters, {len(new_parts)} partitions",
              file=sys.stderr)

        # Exact partition match
        assert old_parts == new_parts, (
            f"Partition mismatch!\n"
            f"Old clusters: {old_nclusters}, New clusters: {new_nclusters}\n"
            f"Only in old: {old_parts - new_parts}\n"
            f"Only in new: {new_parts - old_parts}"
        )

        assert old_nclusters == new_nclusters, (
            f"Cluster count mismatch: old={old_nclusters}, new={new_nclusters}"
        )

        print(f"CONCORDANCE VERIFIED: {old_nclusters} clusters, "
              f"{len(old_parts)} partitions match exactly.", file=sys.stderr)

    def test_spneumo_dataframe_concordance(self):
        """DataFrame path produces identical partitions to list-of-objects path."""
        blocks = _load_spneumo_blocks()
        if blocks is None:
            pytest.skip("S. pneumoniae blocks CSV not found")

        from elsa.cluster import cluster_blocks_by_overlap

        # Build DataFrame from blocks (same as checkpoint would provide)
        import json as _json
        rows = []
        for b in blocks:
            rows.append({
                "block_id": b.block_id,
                "query_genome": b.query_genome, "target_genome": b.target_genome,
                "query_contig": b.query_contig, "target_contig": b.target_contig,
                "query_start": b.query_start, "query_end": b.query_end,
                "target_start": b.target_start, "target_end": b.target_end,
                "n_anchors": b.n_anchors, "chain_score": b.chain_score,
                "orientation": b.orientation,
            })
        blocks_df = pd.DataFrame(rows)

        print(f"\nRunning DataFrame concordance on {len(blocks)} blocks...",
              file=sys.stderr)

        list_btc, _ = cluster_blocks_by_overlap(blocks)
        df_btc, _ = cluster_blocks_by_overlap(blocks_df)

        list_parts = _extract_partitions(list_btc)
        df_parts = _extract_partitions(df_btc)

        list_n = len([c for c in set(list_btc.values()) if c != 0])
        df_n = len([c for c in set(df_btc.values()) if c != 0])

        print(f"List path: {list_n} clusters | DataFrame path: {df_n} clusters",
              file=sys.stderr)

        assert list_parts == df_parts, (
            f"DataFrame vs list partition mismatch!\n"
            f"List: {list_n}, DataFrame: {df_n}"
        )
        print(f"DATAFRAME CONCORDANCE VERIFIED: {df_n} clusters match exactly.",
              file=sys.stderr)
