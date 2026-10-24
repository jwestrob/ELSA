"""Localization must be hard to report without, and hard to get subtly wrong.

The failure this guards against is not a crash. It is a recall figure published
on its own, from blocks that cover their target by sprawling across the contig.
These tests pin the geometry, and pin that recall and precision come out of one
call over one denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elsa.localization import (
    block_localization,
    format_summary,
    inclusive_overlap,
    localize_pairs,
    summarize,
)


def _locus(start_a=10, end_a=12, start_b=10, end_b=12, operon="op1"):
    return pd.DataFrame([{
        "operon_id": operon,
        "genome_a": "A", "contig_a": "cA", "start_a": start_a, "end_a": end_a,
        "genome_b": "B", "contig_b": "cB", "start_b": start_b, "end_b": end_b,
    }])


def _block(qs, qe, ts, te, block_id=0, qg="A", qc="cA", tg="B", tc="cB"):
    return pd.DataFrame([{
        "block_id": block_id,
        "query_genome": qg, "query_contig": qc, "query_start": qs, "query_end": qe,
        "target_genome": tg, "target_contig": tc, "target_start": ts, "target_end": te,
    }])


def test_overlap_is_inclusive():
    """Gene indices, so [4,4] is one gene and not zero."""
    assert inclusive_overlap(4, 4, 4, 4) == 1
    assert inclusive_overlap(0, 9, 5, 14) == 5
    assert inclusive_overlap(0, 3, 10, 12) == 0


def test_exact_block_scores_perfectly():
    got = localize_pairs(_locus(), _block(10, 12, 10, 12))
    row = got.iloc[0]
    assert row.hit
    assert row.precision_a == pytest.approx(1.0)
    assert row.pair_iou == pytest.approx(1.0)
    assert row.containing_excess == pytest.approx(0.0)


def test_a_block_ten_times_the_locus_is_ten_percent_precise():
    """The whole point: this block has perfect recall and is 90% padding."""
    got = localize_pairs(_locus(start_a=10, end_a=12, start_b=10, end_b=12),
                         _block(10, 39, 10, 39))
    row = got.iloc[0]
    assert row.hit
    assert row.coverage_a == pytest.approx(1.0)
    assert row.precision_a == pytest.approx(3 / 30)
    assert row.containing_excess == pytest.approx(30 + 30 - 3 - 3)


def test_a_block_covering_only_one_side_is_not_a_hit():
    """Strict convention: sprawling across one genome and missing the other is
    not a syntenic observation, and must not inflate recall."""
    got = localize_pairs(_locus(), _block(0, 100, 500, 600))
    assert not got.iloc[0].hit
    assert got.iloc[0].precision_a == 0.0


def test_block_is_undirected():
    """The same block stored with the genomes the other way round is the same
    block; orientation is a storage detail, not a biological one."""
    forward = localize_pairs(_locus(), _block(10, 12, 10, 12))
    flipped = localize_pairs(
        _locus(),
        _block(10, 12, 10, 12, qg="B", qc="cB", tg="A", tc="cA"),
    )
    assert flipped.iloc[0].hit
    assert flipped.iloc[0].pair_iou == pytest.approx(forward.iloc[0].pair_iou)


def test_tightest_block_wins_among_equally_covering_ones():
    blocks = pd.concat([_block(0, 99, 0, 99, block_id=1),
                        _block(10, 12, 10, 12, block_id=2),
                        _block(5, 20, 5, 20, block_id=3)], ignore_index=True)
    row = localize_pairs(_locus(), blocks).iloc[0]
    assert row.block_id == 2
    assert row.precision_a == pytest.approx(1.0)


def test_containing_excess_uses_the_smallest_containing_block():
    blocks = pd.concat([_block(0, 199, 0, 199, block_id=1),
                        _block(8, 17, 8, 17, block_id=2)], ignore_index=True)
    row = localize_pairs(_locus(), blocks).iloc[0]
    assert row.containing_excess == pytest.approx(10 + 10 - 3 - 3)


def test_recall_and_precision_share_one_denominator():
    """A caller must not be able to quote recall over everything while quoting
    precision over the hits alone without noticing."""
    loci = pd.concat([_locus(operon="hit"), _locus(operon="miss")], ignore_index=True)
    loci.loc[1, ["start_a", "end_a", "start_b", "end_b"]] = [900, 902, 900, 902]
    got = localize_pairs(loci, _block(10, 12, 10, 12))
    summary = summarize(got)

    assert summary["n_loci"] == 2
    assert summary["n_hits"] == 1
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["median_boundary_precision"] == pytest.approx(1.0)


def test_summary_text_always_carries_both_numbers():
    got = localize_pairs(_locus(), _block(10, 39, 10, 39))
    text = format_summary(summarize(got), "demo")
    assert "recall" in text and "boundary precision" in text
    assert "locus by extent" in text


def test_bootstrap_resamples_operons_not_rows():
    """Pair rows repeat the same operon across genome pairs, so they are not
    independent — and operons contribute wildly unequal numbers of rows.

    Here one tightly-localized operon supplies most of the rows and two sprawling
    ones supply a handful. Resampling rows almost always returns the dominant
    operon's median and looks precise; resampling operons reflects that the
    estimate rests on three operons, not on a hundred correlated rows.
    """
    loci_rows, block_rows = [], []
    for operon, width, count in (("tight", 0, 100), ("loose1", 60, 5),
                                 ("loose2", 80, 5)):
        for i in range(count):
            partner = f"B_{operon}_{i}"
            row = _locus(operon=operon)
            row["genome_b"] = partner
            loci_rows.append(row)
            block = _block(10, 12 + width, 10, 12 + width,
                           block_id=len(block_rows), tg=partner)
            block_rows.append(block)
    loci = pd.concat(loci_rows, ignore_index=True)
    blocks = pd.concat(block_rows, ignore_index=True)

    got = localize_pairs(loci, blocks)
    grouped = summarize(got, group_column="operon_id")
    ungrouped = summarize(got, group_column=None)

    grouped_width = (grouped["boundary_precision_ci_high"]
                     - grouped["boundary_precision_ci_low"])
    ungrouped_width = (ungrouped["boundary_precision_ci_high"]
                       - ungrouped["boundary_precision_ci_low"])
    assert grouped_width > ungrouped_width


def test_interval_is_unestimable_from_a_single_operon():
    """One group carries no between-group variance. A zero-width interval would
    read as certainty, so this reports unknown instead."""
    loci_rows, block_rows = [], []
    for i in range(20):
        partner = f"B{i}"
        row = _locus(operon="only")
        row["genome_b"] = partner
        loci_rows.append(row)
        block_rows.append(_block(10, 12 + i, 10, 12 + i, block_id=i, tg=partner))

    got = localize_pairs(pd.concat(loci_rows, ignore_index=True),
                         pd.concat(block_rows, ignore_index=True))
    summary = summarize(got, group_column="operon_id")
    assert np.isnan(summary["boundary_precision_ci_low"])
    assert not np.isnan(summary["median_boundary_precision"])


def test_block_localization_needs_no_ground_truth():
    blocks = pd.DataFrame({
        "block_id": [1, 2],
        "query_start": [0, 0], "query_end": [9, 99],
        "target_start": [0, 0], "target_end": [9, 99],
        "n_anchors": [10, 10],
    })
    got = block_localization(blocks)
    assert got.loc[0, "anchor_density"] == pytest.approx(1.0)
    assert got.loc[1, "anchor_density"] == pytest.approx(0.1)
