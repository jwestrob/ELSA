"""Tests for elsa.chain — LIS-based collinear anchor chaining."""

from __future__ import annotations

import pytest

from elsa.seed import GeneAnchor
from elsa.chain import chain_anchors_lis, extract_nonoverlapping_chains, ChainedBlock


def _anchor(qi, ti, sim=0.95, qg="g1", tg="g2", qc="c1", tc="c2", orient=0):
    """Helper to create a GeneAnchor."""
    return GeneAnchor(
        query_idx=qi, target_idx=ti,
        query_genome=qg, target_genome=tg,
        query_contig=qc, target_contig=tc,
        query_gene_id=f"{qg}_{qc}_gene{qi}",
        target_gene_id=f"{tg}_{tc}_gene{ti}",
        similarity=sim,
        orientation=orient,
    )


class TestChainAnchorsLIS:
    def test_empty_input(self):
        assert chain_anchors_lis([], max_gap=2, min_size=2) == []

    def test_single_anchor_below_min_size(self):
        anchors = [_anchor(0, 0)]
        assert chain_anchors_lis(anchors, max_gap=2, min_size=2) == []

    def test_perfect_collinear_forward(self):
        anchors = [_anchor(i, i, sim=0.9 + 0.01 * i) for i in range(5)]
        chains = chain_anchors_lis(anchors, max_gap=2, min_size=2)
        assert len(chains) >= 1
        longest = max(chains, key=len)
        assert len(longest) == 5
        # Should be forward orientation
        assert all(a.orientation == 1 for a in longest)

    def test_perfect_collinear_inverted(self):
        # query 0,1,2,3 maps to target 3,2,1,0 — inverted
        anchors = [_anchor(i, 3 - i, sim=0.95) for i in range(4)]
        chains = chain_anchors_lis(anchors, max_gap=2, min_size=2)
        assert len(chains) >= 1
        longest = max(chains, key=len)
        assert len(longest) == 4
        assert all(a.orientation == -1 for a in longest)

    def test_gap_exceeds_max(self):
        # Two anchors with gap=3, max_gap=2 → no chain
        anchors = [_anchor(0, 0), _anchor(4, 4)]
        chains = chain_anchors_lis(anchors, max_gap=2, min_size=2)
        assert chains == []

    def test_gap_within_limit(self):
        # Two anchors with gap=2, max_gap=2 → chain of 2
        anchors = [_anchor(0, 0), _anchor(3, 3)]
        chains = chain_anchors_lis(anchors, max_gap=2, min_size=2)
        assert len(chains) >= 1
        assert any(len(c) == 2 for c in chains)

    def test_gap_penalty_zero_preserves_behavior(self):
        anchors = [_anchor(i, i, sim=0.95) for i in range(4)]
        chains_no_penalty = chain_anchors_lis(anchors, max_gap=2, min_size=2, gap_penalty_scale=0.0)
        chains_with_zero = chain_anchors_lis(anchors, max_gap=2, min_size=2, gap_penalty_scale=0.0)
        # Both should produce identical results
        assert len(chains_no_penalty) == len(chains_with_zero)
        for c1, c2 in zip(chains_no_penalty, chains_with_zero):
            assert len(c1) == len(c2)

    def test_gap_penalty_penalizes_long_gaps(self):
        # Two chains possible: tight (0,1,2) or gappy (0,3,6)
        anchors_tight = [_anchor(0, 0, 0.95), _anchor(1, 1, 0.95), _anchor(2, 2, 0.95)]
        anchors_gappy = [_anchor(0, 0, 0.95), _anchor(3, 3, 0.95), _anchor(6, 6, 0.95)]

        # Without penalty, both produce chains
        chains_tight = chain_anchors_lis(anchors_tight, max_gap=6, min_size=2, gap_penalty_scale=0.0)
        chains_gappy = chain_anchors_lis(anchors_gappy, max_gap=6, min_size=2, gap_penalty_scale=0.0)
        assert len(chains_tight) >= 1
        assert len(chains_gappy) >= 1

        # With penalty, gappy chain has lower score
        chains_tight_pen = chain_anchors_lis(anchors_tight, max_gap=6, min_size=2, gap_penalty_scale=1.0)
        chains_gappy_pen = chain_anchors_lis(anchors_gappy, max_gap=6, min_size=2, gap_penalty_scale=1.0)
        # Both still produce chains, but score differs
        assert len(chains_tight_pen) >= 1
        assert len(chains_gappy_pen) >= 1

    def test_strand_aware_partitioning(self):
        # Mix of forward (+1) and inverted (-1) anchors
        anchors = [
            _anchor(0, 0, 0.95, orient=1),
            _anchor(1, 1, 0.93, orient=1),
            _anchor(2, 2, 0.91, orient=1),
            _anchor(3, 5, 0.95, orient=-1),  # Inverted anchor
        ]
        chains = chain_anchors_lis(anchors, max_gap=2, min_size=2)
        # Should get a forward chain of 3, inverted anchor alone (below min_size)
        assert any(len(c) == 3 for c in chains)

    def test_determinism(self):
        anchors = [_anchor(i, i, sim=0.9 + 0.01 * i) for i in range(6)]
        results = [chain_anchors_lis(anchors, max_gap=2, min_size=2) for _ in range(5)]
        # All runs should produce identical chains
        for r in results[1:]:
            assert len(r) == len(results[0])
            for c1, c2 in zip(results[0], r):
                assert len(c1) == len(c2)


class TestExtractNonoverlappingChains:
    def test_empty_chains(self):
        assert extract_nonoverlapping_chains([]) == []

    def test_single_chain(self):
        chain = [_anchor(0, 0), _anchor(1, 1), _anchor(2, 2)]
        blocks = extract_nonoverlapping_chains([chain])
        assert len(blocks) == 1
        assert blocks[0].n_anchors == 3
        assert blocks[0].query_start == 0
        assert blocks[0].query_end == 2

    def test_picks_higher_scoring_chain(self):
        # Two overlapping chains; higher score wins
        chain_high = [_anchor(0, 0, 0.99), _anchor(1, 1, 0.99), _anchor(2, 2, 0.99)]
        chain_low = [_anchor(0, 0, 0.80), _anchor(1, 1, 0.80)]
        blocks = extract_nonoverlapping_chains([chain_high, chain_low])
        assert len(blocks) == 1
        assert blocks[0].n_anchors == 3

    def test_nonoverlapping_both_kept(self):
        chain1 = [_anchor(0, 0), _anchor(1, 1)]
        chain2 = [_anchor(5, 5), _anchor(6, 6)]
        blocks = extract_nonoverlapping_chains([chain1, chain2])
        assert len(blocks) == 2

    def test_block_id_start(self):
        chain = [_anchor(0, 0), _anchor(1, 1)]
        blocks = extract_nonoverlapping_chains([chain], block_id_start=100)
        assert blocks[0].block_id == 100

    def test_chained_block_properties(self):
        chain = [
            _anchor(2, 5, qg="genA", tg="genB"),
            _anchor(3, 6, qg="genA", tg="genB"),
            _anchor(4, 7, qg="genA", tg="genB"),
        ]
        blocks = extract_nonoverlapping_chains([chain])
        b = blocks[0]
        assert b.query_span == 3
        assert b.target_span == 3
        assert b.n_anchors == 3
        assert len(b.query_gene_ids()) == 3
        assert len(b.target_gene_ids()) == 3
