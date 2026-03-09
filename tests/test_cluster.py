"""Tests for elsa.cluster — overlap-based block clustering."""

from __future__ import annotations

import pytest

from elsa.chain import ChainedBlock
from elsa.cluster import cluster_blocks_by_overlap


def _block(bid, qg, tg, qs, qe, ts, te, qc="c1", tc="c1"):
    """Helper to create a ChainedBlock."""
    n = qe - qs + 1
    return ChainedBlock(
        block_id=bid,
        query_genome=qg, target_genome=tg,
        query_contig=qc, target_contig=tc,
        query_start=qs, query_end=qe,
        target_start=ts, target_end=te,
        anchor_query_ids=list(range(qs, qe + 1)),
        anchor_target_ids=list(range(ts, ts + n)),
        anchor_query_gene_ids=[f"{qg}_{qc}_g{i}" for i in range(qs, qe + 1)],
        anchor_target_gene_ids=[f"{tg}_{tc}_g{i}" for i in range(ts, ts + n)],
        n_anchors=n,
        chain_score=n * 0.95,
    )


class TestClusterBlocksByOverlap:
    def test_empty_input(self):
        mapping, df = cluster_blocks_by_overlap([])
        assert mapping == {}
        assert df.empty

    def test_shared_gene_blocks_cluster_together(self):
        a = _block(0, "g1", "g2", 0, 4, 0, 4)
        b = _block(1, "g1", "g3", 0, 4, 0, 4)

        mapping, df = cluster_blocks_by_overlap([a, b], jaccard_tau=0.1, min_genome_support=2)

        assert mapping[0] == mapping[1]
        assert mapping[0] > 0

    def test_no_overlap_separate_clusters(self):
        a = _block(0, "g1", "g2", 0, 2, 0, 2)
        b = _block(1, "g3", "g4", 10, 12, 10, 12)

        mapping, df = cluster_blocks_by_overlap([a, b], jaccard_tau=0.1, min_genome_support=2)

        assert mapping[0] != mapping[1] or mapping[0] == 0

    def test_genome_support_filtering(self):
        block = _block(0, "g1", "g2", 0, 3, 0, 3)
        mapping, df = cluster_blocks_by_overlap([block], min_genome_support=2)
        assert mapping[0] > 0

    def test_genome_support_filtering_too_few(self):
        block = _block(0, "g1", "g2", 0, 3, 0, 3)
        mapping, df = cluster_blocks_by_overlap([block], min_genome_support=3)
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
        blocks = []
        for i in range(10):
            blocks.append(_block(i, "g1", f"g{i+2}", 0, 5, 0, 5))

        mapping, df = cluster_blocks_by_overlap(
            blocks, jaccard_tau=0.1, mutual_k=10, min_genome_support=2
        )

        for b in blocks:
            assert mapping[b.block_id] > 0, f"Block {b.block_id} in sink"

        cluster_ids = {mapping[b.block_id] for b in blocks}
        real_clusters = {c for c in cluster_ids if c > 0}
        assert len(real_clusters) <= 3
