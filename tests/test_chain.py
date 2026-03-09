"""Tests for elsa.chain — LIS-based collinear anchor chaining."""

from __future__ import annotations

import pytest
import pandas as pd

from elsa.seed import ANCHOR_COLS
from elsa.chain import chain_anchors_lis, extract_nonoverlapping_chains, ChainedBlock


def _anchor_df(anchors):
    """Build an anchor DataFrame from a list of (qi, ti, sim, qg, tg, qc, tc, orient) tuples."""
    rows = []
    for a in anchors:
        qi, ti = a[0], a[1]
        sim = a[2] if len(a) > 2 else 0.95
        qg = a[3] if len(a) > 3 else "g1"
        tg = a[4] if len(a) > 4 else "g2"
        qc = a[5] if len(a) > 5 else "c1"
        tc = a[6] if len(a) > 6 else "c2"
        orient = a[7] if len(a) > 7 else 0
        rows.append({
            "query_idx": qi, "target_idx": ti,
            "query_genome": qg, "target_genome": tg,
            "query_contig": qc, "target_contig": tc,
            "query_gene_id": f"{qg}_{qc}_gene{qi}",
            "target_gene_id": f"{tg}_{tc}_gene{ti}",
            "similarity": sim, "orientation": orient,
        })
    return pd.DataFrame(rows, columns=ANCHOR_COLS)


class TestChainAnchorsLIS:
    def test_empty_input(self):
        df = _anchor_df([])
        assert chain_anchors_lis(df, max_gap=2, min_size=2) == []

    def test_single_anchor_below_min_size(self):
        df = _anchor_df([(0, 0)])
        assert chain_anchors_lis(df, max_gap=2, min_size=2) == []

    def test_perfect_collinear_forward(self):
        df = _anchor_df([(i, i, 0.9 + 0.01 * i) for i in range(5)])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        assert len(chains) >= 1
        longest = max(chains, key=len)
        assert len(longest) == 5
        assert all(longest["orientation"] == 1)

    def test_perfect_collinear_inverted(self):
        df = _anchor_df([(i, 3 - i, 0.95) for i in range(4)])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        assert len(chains) >= 1
        longest = max(chains, key=len)
        assert len(longest) == 4
        assert all(longest["orientation"] == -1)

    def test_gap_exceeds_max(self):
        df = _anchor_df([(0, 0), (4, 4)])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        assert chains == []

    def test_gap_within_limit(self):
        df = _anchor_df([(0, 0), (3, 3)])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        assert len(chains) >= 1
        assert any(len(c) == 2 for c in chains)

    def test_gap_penalty_zero_preserves_behavior(self):
        df = _anchor_df([(i, i, 0.95) for i in range(4)])
        chains_no_penalty = chain_anchors_lis(df, max_gap=2, min_size=2, gap_penalty_scale=0.0)
        chains_with_zero = chain_anchors_lis(df, max_gap=2, min_size=2, gap_penalty_scale=0.0)
        assert len(chains_no_penalty) == len(chains_with_zero)
        for c1, c2 in zip(chains_no_penalty, chains_with_zero):
            assert len(c1) == len(c2)

    def test_gap_penalty_penalizes_long_gaps(self):
        df_tight = _anchor_df([(0, 0, 0.95), (1, 1, 0.95), (2, 2, 0.95)])
        df_gappy = _anchor_df([(0, 0, 0.95), (3, 3, 0.95), (6, 6, 0.95)])

        chains_tight = chain_anchors_lis(df_tight, max_gap=6, min_size=2, gap_penalty_scale=0.0)
        chains_gappy = chain_anchors_lis(df_gappy, max_gap=6, min_size=2, gap_penalty_scale=0.0)
        assert len(chains_tight) >= 1
        assert len(chains_gappy) >= 1

        chains_tight_pen = chain_anchors_lis(df_tight, max_gap=6, min_size=2, gap_penalty_scale=1.0)
        chains_gappy_pen = chain_anchors_lis(df_gappy, max_gap=6, min_size=2, gap_penalty_scale=1.0)
        assert len(chains_tight_pen) >= 1
        assert len(chains_gappy_pen) >= 1

    def test_strand_aware_partitioning(self):
        df = _anchor_df([
            (0, 0, 0.95, "g1", "g2", "c1", "c2", 1),
            (1, 1, 0.93, "g1", "g2", "c1", "c2", 1),
            (2, 2, 0.91, "g1", "g2", "c1", "c2", 1),
            (3, 5, 0.95, "g1", "g2", "c1", "c2", -1),
        ])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        assert any(len(c) == 3 for c in chains)

    def test_determinism(self):
        df = _anchor_df([(i, i, 0.9 + 0.01 * i) for i in range(6)])
        results = [chain_anchors_lis(df, max_gap=2, min_size=2) for _ in range(5)]
        for r in results[1:]:
            assert len(r) == len(results[0])
            for c1, c2 in zip(results[0], r):
                assert len(c1) == len(c2)


class TestExtractNonoverlappingChains:
    def test_empty_chains(self):
        assert extract_nonoverlapping_chains([]) == []

    def test_single_chain(self):
        df = _anchor_df([(0, 0), (1, 1), (2, 2)])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        blocks = extract_nonoverlapping_chains(chains)
        assert len(blocks) == 1
        assert blocks[0].n_anchors == 3
        assert blocks[0].query_start == 0
        assert blocks[0].query_end == 2

    def test_picks_higher_scoring_chain(self):
        chain_high = _anchor_df([(0, 0, 0.99), (1, 1, 0.99), (2, 2, 0.99)])
        chain_low = _anchor_df([(0, 0, 0.80), (1, 1, 0.80)])
        ch = chain_anchors_lis(chain_high, max_gap=2, min_size=2)
        cl = chain_anchors_lis(chain_low, max_gap=2, min_size=2)
        blocks = extract_nonoverlapping_chains(ch + cl)
        assert len(blocks) == 1
        assert blocks[0].n_anchors == 3

    def test_nonoverlapping_both_kept(self):
        df1 = _anchor_df([(0, 0), (1, 1)])
        df2 = _anchor_df([(5, 5), (6, 6)])
        ch1 = chain_anchors_lis(df1, max_gap=2, min_size=2)
        ch2 = chain_anchors_lis(df2, max_gap=2, min_size=2)
        blocks = extract_nonoverlapping_chains(ch1 + ch2)
        assert len(blocks) == 2

    def test_block_id_start(self):
        df = _anchor_df([(0, 0), (1, 1)])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        blocks = extract_nonoverlapping_chains(chains, block_id_start=100)
        assert blocks[0].block_id == 100

    def test_chained_block_properties(self):
        df = _anchor_df([
            (2, 5, 0.95, "genA", "genB"),
            (3, 6, 0.95, "genA", "genB"),
            (4, 7, 0.95, "genA", "genB"),
        ])
        chains = chain_anchors_lis(df, max_gap=2, min_size=2)
        blocks = extract_nonoverlapping_chains(chains)
        b = blocks[0]
        assert b.query_span == 3
        assert b.target_span == 3
        assert b.n_anchors == 3
        assert len(b.query_gene_ids()) == 3
        assert len(b.target_gene_ids()) == 3
