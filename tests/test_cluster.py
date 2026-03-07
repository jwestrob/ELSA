"""Tests for elsa.cluster — overlap-based block clustering."""

from __future__ import annotations

import pytest

from elsa.chain import ChainedBlock
from elsa.seed import GeneAnchor
from elsa.cluster import cluster_blocks_by_overlap


def _block(bid, qg, tg, qs, qe, ts, te, qc="c1", tc="c1"):
    """Helper to create a ChainedBlock."""
    anchors = [
        GeneAnchor(
            query_idx=i, target_idx=ts + (i - qs),
            query_genome=qg, target_genome=tg,
            query_contig=qc, target_contig=tc,
            query_gene_id=f"{qg}_{qc}_g{i}",
            target_gene_id=f"{tg}_{tc}_g{ts + (i - qs)}",
            similarity=0.95,
        )
        for i in range(qs, qe + 1)
    ]
    return ChainedBlock(
        block_id=bid,
        query_genome=qg, target_genome=tg,
        query_contig=qc, target_contig=tc,
        query_start=qs, query_end=qe,
        target_start=ts, target_end=te,
        anchors=anchors,
        chain_score=len(anchors) * 0.95,
    )


class TestClusterBlocksByOverlap:
    def test_empty_input(self):
        mapping, df = cluster_blocks_by_overlap([])
        assert mapping == {}
        assert df.empty

    def test_shared_gene_blocks_cluster_together(self):
        # Block A: g1:c1[0-4] vs g2:c1[0-4]
        # Block B: g1:c1[0-4] vs g3:c1[0-4]
        # They share genes on g1:c1[0-4] → should cluster
        a = _block(0, "g1", "g2", 0, 4, 0, 4)
        b = _block(1, "g1", "g3", 0, 4, 0, 4)

        mapping, df = cluster_blocks_by_overlap([a, b], jaccard_tau=0.1, min_genome_support=2)

        # Both should be in same cluster (cluster > 0)
        assert mapping[0] == mapping[1]
        assert mapping[0] > 0

    def test_no_overlap_separate_clusters(self):
        # Block A: g1:c1[0-2] vs g2:c1[0-2]
        # Block B: g3:c1[10-12] vs g4:c1[10-12]
        # No shared genes → separate clusters (or singletons)
        a = _block(0, "g1", "g2", 0, 2, 0, 2)
        b = _block(1, "g3", "g4", 10, 12, 10, 12)

        mapping, df = cluster_blocks_by_overlap([a, b], jaccard_tau=0.1, min_genome_support=2)

        # They should be in different clusters (either different IDs or both sink)
        assert mapping[0] != mapping[1] or mapping[0] == 0

    def test_genome_support_filtering(self):
        # Single block from 2 genomes → meets min_genome_support=2
        block = _block(0, "g1", "g2", 0, 3, 0, 3)
        mapping, df = cluster_blocks_by_overlap([block], min_genome_support=2)
        # Should be in a real cluster (2 genomes)
        assert mapping[0] > 0

    def test_genome_support_filtering_too_few(self):
        # Single block from 2 genomes, but min_genome_support=3
        block = _block(0, "g1", "g2", 0, 3, 0, 3)
        mapping, df = cluster_blocks_by_overlap([block], min_genome_support=3)
        # Should be in sink (0)
        assert mapping[0] == 0

    def test_cluster_metadata(self):
        a = _block(0, "g1", "g2", 0, 4, 0, 4)
        b = _block(1, "g1", "g3", 0, 4, 0, 4)

        mapping, df = cluster_blocks_by_overlap([a, b], jaccard_tau=0.1, min_genome_support=2)

        assert not df.empty
        row = df.iloc[0]
        assert row["size"] >= 2
        assert row["genome_support"] >= 2
        assert "genes_json" in row

    def test_large_cluster(self):
        # Create 10 blocks all sharing g1:c1[0-5] with different target genomes
        blocks = []
        for i in range(10):
            blocks.append(_block(i, "g1", f"g{i+2}", 0, 5, 0, 5))

        mapping, df = cluster_blocks_by_overlap(
            blocks, jaccard_tau=0.1, mutual_k=10, min_genome_support=2
        )

        # All should be in real clusters (not sink)
        for b in blocks:
            assert mapping[b.block_id] > 0, f"Block {b.block_id} in sink"

        # With high mutual_k, most should cluster together
        cluster_ids = {mapping[b.block_id] for b in blocks}
        real_clusters = {c for c in cluster_ids if c > 0}
        assert len(real_clusters) <= 3  # Allow some splitting due to mutual-k
