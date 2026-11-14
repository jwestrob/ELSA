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
        assert "mean_chain_length" in row

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


class TestConservedLocusRecovery:
    """One locus observed across many genomes must yield one cluster.

    Both regressions guarded here shattered conserved loci into per-genome-pair
    fragments: pooling both block sides into a single Jaccard ratio caps the
    score at 1/3, and ranking tied edges by raw node id makes every block
    prefer the same few low-id neighbours.
    """

    def test_identical_extent_blocks_form_one_cluster(self):
        # One 6-gene locus in g0, matched into 24 other genomes.
        blocks = [
            _block(i, "g0", f"g{i + 1}", 100, 105, 200, 205)
            for i in range(24)
        ]

        mapping, df = cluster_blocks_by_overlap(blocks, min_genome_support=2)

        assigned = {mapping[b.block_id] for b in blocks}
        assert assigned == {1}, f"locus shattered into {len(assigned)} clusters"
        assert int(df.iloc[0]["genome_support"]) == 25
        assert int(df.iloc[0]["size"]) == 24

    def test_shared_side_match_survives_unequal_opposite_sides(self):
        # Identical query interval; target chains run to very different lengths.
        a = _block(0, "g1", "g2", 10, 19, 0, 9)
        b = _block(1, "g1", "g3", 10, 19, 0, 59, tc="c2")

        mapping, _ = cluster_blocks_by_overlap([a, b], min_genome_support=2)

        assert mapping[0] == mapping[1] > 0

    def test_containment_does_not_merge_different_scale_blocks(self):
        # A 6-gene block inside a 300-gene block: different objects, not one locus.
        small = _block(0, "g1", "g2", 100, 105, 100, 105)
        large = _block(1, "g1", "g3", 0, 299, 0, 299)

        mapping, _ = cluster_blocks_by_overlap([small, large], min_genome_support=2)

        assert mapping[0] != mapping[1]

    def test_tiebreak_is_symmetric(self):
        from elsa.cluster import _pair_tiebreak

        assert _pair_tiebreak(3, 91) == _pair_tiebreak(91, 3)
        assert _pair_tiebreak(3, 91) != _pair_tiebreak(3, 92)

    def test_mutual_k_zero_keeps_every_edge(self):
        from elsa.cluster import _mutual_top_k

        edges = {0: [(1, 0.9), (2, 0.9)], 1: [(0, 0.9)], 2: [(0, 0.9)]}
        assert _mutual_top_k(edges, 0) == {(0, 1), (0, 2)}
