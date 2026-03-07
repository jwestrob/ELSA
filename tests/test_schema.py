"""Tests for elsa.schema — cluster architecture schema pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elsa.schema import (
    _merge_intervals,
    _monotonic_align,
    build_block_members,
    extract_locus_instances,
    choose_reference_loci,
    assign_slots,
    compute_slot_summaries,
    compute_architecture_summary,
)


def _make_genes(n_genomes=3, genes_per_contig=20):
    """Create a minimal genes DataFrame with random embeddings."""
    rng = np.random.RandomState(42)
    rows = []
    emb_dim = 8
    for g in range(n_genomes):
        genome = f"genome_{g}"
        contig = f"contig_0"
        for i in range(genes_per_contig):
            emb = rng.randn(emb_dim).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            row = {
                "sample_id": genome,
                "contig_id": contig,
                "gene_id": f"{genome}_{contig}_{i}",
                "start": i * 1000,
                "end": i * 1000 + 900,
                "strand": 1 if i % 2 == 0 else -1,
            }
            for d in range(emb_dim):
                row[f"emb_{d:03d}"] = float(emb[d])
            rows.append(row)
    return pd.DataFrame(rows)


def _make_blocks():
    """Create synthetic blocks between 3 genomes with 2 clusters."""
    blocks = [
        # Cluster 1: genes 5-8 across all 3 genomes (3 pairwise blocks)
        {"block_id": 0, "cluster_id": 1, "query_genome": "genome_0", "target_genome": "genome_1",
         "query_contig": "contig_0", "target_contig": "contig_0",
         "query_start": 5, "query_end": 8, "target_start": 5, "target_end": 8,
         "n_anchors": 4, "chain_score": 3.9, "orientation": 1, "n_genes": 4},
        {"block_id": 1, "cluster_id": 1, "query_genome": "genome_0", "target_genome": "genome_2",
         "query_contig": "contig_0", "target_contig": "contig_0",
         "query_start": 5, "query_end": 8, "target_start": 5, "target_end": 8,
         "n_anchors": 4, "chain_score": 3.8, "orientation": 1, "n_genes": 4},
        {"block_id": 2, "cluster_id": 1, "query_genome": "genome_1", "target_genome": "genome_2",
         "query_contig": "contig_0", "target_contig": "contig_0",
         "query_start": 5, "query_end": 8, "target_start": 5, "target_end": 8,
         "n_anchors": 4, "chain_score": 3.7, "orientation": 1, "n_genes": 4},
        # Cluster 2: genes 12-14 across genome_0 and genome_1
        {"block_id": 3, "cluster_id": 2, "query_genome": "genome_0", "target_genome": "genome_1",
         "query_contig": "contig_0", "target_contig": "contig_0",
         "query_start": 12, "query_end": 14, "target_start": 12, "target_end": 14,
         "n_anchors": 3, "chain_score": 2.8, "orientation": 1, "n_genes": 3},
    ]
    return pd.DataFrame(blocks)


class TestMergeIntervals:
    def test_no_overlap(self):
        assert _merge_intervals([(1, 3), (5, 7)]) == [(1, 3), (5, 7)]

    def test_overlap(self):
        assert _merge_intervals([(1, 5), (3, 7)]) == [(1, 7)]

    def test_adjacent(self):
        assert _merge_intervals([(1, 3), (4, 6)]) == [(1, 6)]

    def test_empty(self):
        assert _merge_intervals([]) == []

    def test_single(self):
        assert _merge_intervals([(2, 5)]) == [(2, 5)]


class TestMonotonicAlign:
    def test_identical(self):
        """Perfect match should align all positions."""
        sim = np.eye(4, dtype=np.float32)
        result = _monotonic_align(sim, 4)
        assert result == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_with_gap(self):
        """Member shorter than reference — some slots should be empty."""
        sim = np.array([
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.9],
        ], dtype=np.float32)
        result = _monotonic_align(sim, 4)
        assert (0, 0) in result
        assert (1, 3) in result
        assert len(result) == 2

    def test_below_threshold(self):
        """All similarities below threshold — nothing assigned."""
        sim = np.full((3, 3), 0.1, dtype=np.float32)
        result = _monotonic_align(sim, 3, min_sim=0.5)
        assert result == []


class TestBuildBlockMembers:
    def test_basic(self):
        genes = _make_genes()
        blocks = _make_blocks()
        members = build_block_members(blocks, genes)
        # 4 blocks × 2 sides × varying genes
        assert len(members) > 0
        assert set(members.columns) == {
            "block_id", "cluster_id", "side", "genome_id",
            "contig_id", "gene_idx", "gene_id", "strand", "anchor_order",
        }
        # Block 0 should have 4 genes on each side = 8 total
        b0 = members[members["block_id"] == 0]
        assert len(b0) == 8


class TestExtractLoci:
    def test_deduplication(self):
        genes = _make_genes()
        blocks = _make_blocks()
        members = build_block_members(blocks, genes)
        loci = extract_locus_instances(members)

        # Cluster 1 has 3 genomes, each appearing on both sides of blocks
        c1 = loci[loci["cluster_id"] == 1]
        assert c1["genome_id"].nunique() == 3
        # Each genome should have exactly 1 merged locus (5-8)
        for _, row in c1.iterrows():
            assert row["start_idx"] == 5
            assert row["end_idx"] == 8


class TestReferenceChoice:
    def test_picks_highest_support(self):
        genes = _make_genes()
        blocks = _make_blocks()
        members = build_block_members(blocks, genes)
        loci = extract_locus_instances(members)
        refs = choose_reference_loci(loci)

        # Cluster 1 should pick the locus with highest support
        c1_loci = loci[loci["cluster_id"] == 1]
        ref_lid = refs[1]
        ref_row = c1_loci[c1_loci["locus_id"] == ref_lid].iloc[0]
        max_support = c1_loci["support"].max()
        assert ref_row["support"] == max_support


class TestSlotAssignment:
    def test_produces_assignments(self):
        genes = _make_genes()
        blocks = _make_blocks()
        members = build_block_members(blocks, genes)
        loci = extract_locus_instances(members, cluster_ids=[1])
        refs = choose_reference_loci(loci)
        assignments = assign_slots(loci, genes, refs, blocks)

        assert len(assignments) > 0
        # Should have reference assignments
        ref_rows = assignments[assignments["is_reference"]]
        assert len(ref_rows) == 4  # 4 slots from reference


class TestSlotSummaries:
    def test_produces_stats(self):
        genes = _make_genes()
        blocks = _make_blocks()
        members = build_block_members(blocks, genes)
        loci = extract_locus_instances(members, cluster_ids=[1])
        refs = choose_reference_loci(loci)
        assignments = assign_slots(loci, genes, refs, blocks)
        slots = compute_slot_summaries(assignments, genes)

        assert len(slots) > 0
        assert "occupancy" in slots.columns
        assert "is_core" in slots.columns
        assert "dispersion" in slots.columns


class TestArchitectureSummary:
    def test_produces_metrics(self):
        genes = _make_genes()
        blocks = _make_blocks()
        members = build_block_members(blocks, genes)
        loci = extract_locus_instances(members, cluster_ids=[1])
        refs = choose_reference_loci(loci)
        assignments = assign_slots(loci, genes, refs, blocks)
        slots = compute_slot_summaries(assignments, genes)
        arch = compute_architecture_summary(slots, loci)

        assert len(arch) == 1  # cluster 1 only
        row = arch.iloc[0]
        assert row["n_slots"] == 4
        assert row["n_loci"] == 3
        assert row["arch_label"] in ("coherent", "moderate", "fragmented", "variable (replacement)")
