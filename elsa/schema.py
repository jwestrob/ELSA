"""
Cluster architecture schema: slot-based structural summary of syntenic clusters.

Converts pairwise syntenic blocks into per-cluster locus instances,
assigns genes to reference-based slots, and computes architecture metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._log import tlog as _log


# ---------------------------------------------------------------------------
# Stage 1: Extract block members (gene-level detail per block side)
# ---------------------------------------------------------------------------

def build_block_members(
    blocks_df: pd.DataFrame,
    genes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a table of per-gene block membership.

    For each block, expand both query and target sides into individual gene
    rows with their gene_id, genome, contig, and position.

    Returns DataFrame with columns:
        block_id, cluster_id, side, genome_id, contig_id, gene_idx,
        gene_id, strand, anchor_order
    """
    # Build position_index per genome/contig (matches pipeline ordering)
    genes = genes_df[["sample_id", "contig_id", "gene_id", "start", "end"]].copy()
    if "strand" in genes_df.columns:
        genes["strand"] = genes_df["strand"]
    else:
        genes["strand"] = 0
    genes = genes.sort_values(["sample_id", "contig_id", "start", "end"])
    genes["position_index"] = genes.groupby(["sample_id", "contig_id"]).cumcount()

    # Build lookup: (genome, contig, position_index) -> (gene_id, strand)
    gene_ids = genes["gene_id"].values
    strands = genes["strand"].values
    sample_ids = genes["sample_id"].values
    contig_ids = genes["contig_id"].values
    pos_indices = genes["position_index"].values

    gene_lookup = {}
    for i in range(len(genes)):
        gene_lookup[(sample_ids[i], contig_ids[i], int(pos_indices[i]))] = (
            gene_ids[i], int(strands[i])
        )

    # Fully vectorized expansion: use numpy repeat/arange for both sides
    block_ids = blocks_df["block_id"].values
    cluster_ids = blocks_df["cluster_id"].values
    q_genomes = blocks_df["query_genome"].values
    t_genomes = blocks_df["target_genome"].values
    q_contigs = blocks_df["query_contig"].values
    t_contigs = blocks_df["target_contig"].values
    q_starts = blocks_df["query_start"].values.astype(np.int64)
    q_ends = blocks_df["query_end"].values.astype(np.int64)
    t_starts = blocks_df["target_start"].values.astype(np.int64)
    t_ends = blocks_df["target_end"].values.astype(np.int64)

    # Compute gene counts per block-side
    q_counts = (q_ends - q_starts + 1).astype(np.int64)
    t_counts = (t_ends - t_starts + 1).astype(np.int64)

    # Total rows we'll produce
    total_q = int(q_counts.sum())
    total_t = int(t_counts.sum())

    # --- Query side ---
    q_bid = np.repeat(block_ids, q_counts)
    q_cid = np.repeat(cluster_ids, q_counts)
    q_genome_exp = np.repeat(q_genomes, q_counts)
    q_contig_exp = np.repeat(q_contigs, q_counts)

    # Gene positions: arange per block, concatenated
    q_positions = np.empty(total_q, dtype=np.int64)
    q_orders = np.empty(total_q, dtype=np.int64)
    offset = 0
    for i in range(len(blocks_df)):
        c = int(q_counts[i])
        if c > 0:
            q_positions[offset:offset + c] = np.arange(q_starts[i], q_starts[i] + c)
            q_orders[offset:offset + c] = np.arange(c)
            offset += c

    # --- Target side ---
    t_bid = np.repeat(block_ids, t_counts)
    t_cid = np.repeat(cluster_ids, t_counts)
    t_genome_exp = np.repeat(t_genomes, t_counts)
    t_contig_exp = np.repeat(t_contigs, t_counts)

    t_positions = np.empty(total_t, dtype=np.int64)
    t_orders = np.empty(total_t, dtype=np.int64)
    offset = 0
    for i in range(len(blocks_df)):
        c = int(t_counts[i])
        if c > 0:
            t_positions[offset:offset + c] = np.arange(t_starts[i], t_starts[i] + c)
            t_orders[offset:offset + c] = np.arange(c)
            offset += c

    # Concatenate query + target
    all_bid = np.concatenate([q_bid, t_bid])
    all_cid = np.concatenate([q_cid, t_cid])
    all_side = np.concatenate([
        np.full(total_q, "query", dtype=object),
        np.full(total_t, "target", dtype=object),
    ])
    all_genome = np.concatenate([q_genome_exp, t_genome_exp])
    all_contig = np.concatenate([q_contig_exp, t_contig_exp])
    all_positions = np.concatenate([q_positions, t_positions])
    all_orders = np.concatenate([q_orders, t_orders])

    # Lookup gene_id and strand from the prebuilt dict
    total = len(all_bid)
    all_gene_id = np.empty(total, dtype=object)
    all_strand = np.empty(total, dtype=np.int64)
    for i in range(total):
        info = gene_lookup.get((all_genome[i], all_contig[i], int(all_positions[i])))
        if info is not None:
            all_gene_id[i] = info[0]
            all_strand[i] = info[1]
        else:
            all_gene_id[i] = f"{all_genome[i]}_{all_contig[i]}_{all_positions[i]}"
            all_strand[i] = 0

    return pd.DataFrame({
        "block_id": all_bid,
        "cluster_id": all_cid,
        "side": all_side,
        "genome_id": all_genome,
        "contig_id": all_contig,
        "gene_idx": all_positions,
        "gene_id": all_gene_id,
        "strand": all_strand,
        "anchor_order": all_orders,
    })


# ---------------------------------------------------------------------------
# Stage 2: Deduplicate locus instances per cluster
# ---------------------------------------------------------------------------

@dataclass
class LocusInstance:
    """A unique locus (contiguous gene interval) in a cluster."""
    locus_id: int
    cluster_id: int
    genome_id: str
    contig_id: str
    start_idx: int   # gene position index (inclusive)
    end_idx: int      # gene position index (inclusive)
    gene_ids: List[str] = field(default_factory=list)
    support: int = 0  # number of pairwise blocks that contributed


def extract_locus_instances(
    members_df: pd.DataFrame,
    cluster_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Convert pairwise block membership into unique locus instances per cluster.

    Blocks are pairwise (query/target), so each block contributes two half-loci.
    We merge overlapping intervals on the same genome/contig within a cluster
    to get deduplicated locus instances.

    Returns DataFrame with columns:
        locus_id, cluster_id, genome_id, contig_id, start_idx, end_idx,
        n_genes, support
    """
    if cluster_ids is not None:
        members_df = members_df[members_df["cluster_id"].isin(cluster_ids)]

    # First: compute per-block-side intervals using vectorized groupby
    block_side_intervals = members_df.groupby(
        ["cluster_id", "genome_id", "contig_id", "block_id", "side"]
    )["gene_idx"].agg(["min", "max"]).reset_index()
    block_side_intervals.columns = [
        "cluster_id", "genome_id", "contig_id", "block_id", "side", "start", "end"
    ]

    # Group by cluster + genome + contig and merge intervals
    grouped = block_side_intervals.groupby(["cluster_id", "genome_id", "contig_id"])

    loci = []
    locus_id = 0

    for (cid, genome, contig), grp in grouped:
        starts = grp["start"].values
        ends = grp["end"].values
        intervals = list(zip(starts, ends))
        merged = _merge_intervals(intervals)

        for ms, me in merged:
            support = int(((starts <= me) & (ends >= ms)).sum())
            loci.append({
                "locus_id": locus_id,
                "cluster_id": int(cid),
                "genome_id": genome,
                "contig_id": contig,
                "start_idx": ms,
                "end_idx": me,
                "n_genes": me - ms + 1,
                "support": support,
            })
            locus_id += 1

    return pd.DataFrame(loci)


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent integer intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Stage 3: Choose reference locus per cluster
# ---------------------------------------------------------------------------

def choose_reference_loci(
    loci_df: pd.DataFrame,
) -> Dict[int, int]:
    """Choose a reference locus for each cluster.

    Picks the locus with the highest support (most pairwise blocks contributing).
    Ties broken by n_genes (prefer larger), then locus_id (deterministic).

    Returns: {cluster_id: locus_id}
    """
    # Sort entire DataFrame once, then take first per cluster
    sorted_df = loci_df.sort_values(
        ["cluster_id", "support", "n_genes", "locus_id"],
        ascending=[True, False, False, True],
    )
    best = sorted_df.groupby("cluster_id").first()
    return dict(zip(best.index.astype(int), best["locus_id"].astype(int)))


# ---------------------------------------------------------------------------
# Stage 4: Slot assignment (monotonic order-aware alignment)
# ---------------------------------------------------------------------------

def assign_slots(
    loci_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    ref_loci: Dict[int, int],
    blocks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign genes from each locus instance to reference-based slots.

    For each cluster:
      1. The reference locus defines N slots (one per gene position).
      2. Each other locus is aligned to these slots using embedding similarity
         with a monotonic ordering constraint (no crossing assignments).

    Returns DataFrame with columns:
        cluster_id, locus_id, slot_idx, gene_id, gene_idx, genome_id,
        contig_id, similarity, is_reference, is_insertion
    """
    # Precompute gene embeddings and metadata
    genes = genes_df.copy()
    genes = genes.sort_values(["sample_id", "contig_id", "start", "end"])
    genes["position_index"] = genes.groupby(["sample_id", "contig_id"]).cumcount()
    genes = genes.reset_index(drop=True)

    emb_cols = [c for c in genes.columns if c.startswith("emb_")]
    emb_matrix = genes[emb_cols].values.astype(np.float32)
    gene_id_arr = genes["gene_id"].values
    sample_arr = genes["sample_id"].values
    contig_arr = genes["contig_id"].values
    posidx_arr = genes["position_index"].values

    # Build index: (genome, contig, position_index) -> row number in genes
    pos_to_row = {}
    for i in range(len(genes)):
        key = (sample_arr[i], contig_arr[i], int(posidx_arr[i]))
        pos_to_row[key] = i

    # Determine orientation for each locus relative to reference
    # using the block orientation data
    locus_orientation = _compute_locus_orientations(loci_df, blocks_df)

    # Pre-group loci by cluster_id to avoid repeated DataFrame filtering
    loci_by_cluster = {}
    l_cids = loci_df["cluster_id"].values
    l_lids = loci_df["locus_id"].values
    l_genomes = loci_df["genome_id"].values
    l_contigs = loci_df["contig_id"].values
    l_starts = loci_df["start_idx"].values
    l_ends = loci_df["end_idx"].values
    for i in range(len(loci_df)):
        cid = l_cids[i]
        if cid not in loci_by_cluster:
            loci_by_cluster[cid] = []
        loci_by_cluster[cid].append(i)

    # Helper to gather embeddings for a position range
    n_emb = len(emb_cols)
    zero_emb = np.zeros(n_emb, dtype=np.float32)

    def _gather_embs(genome, contig, positions):
        """Vectorized embedding gather for a list of positions."""
        n = len(positions)
        embs = np.empty((n, n_emb), dtype=np.float32)
        gids = []
        for j in range(n):
            row_idx = pos_to_row.get((genome, contig, positions[j]))
            if row_idx is not None:
                embs[j] = emb_matrix[row_idx]
                gids.append(gene_id_arr[row_idx])
            else:
                embs[j] = zero_emb
                gids.append(f"MISSING_{genome}_{contig}_{positions[j]}")
        return embs, gids

    all_assignments = []

    for cid, ref_lid in ref_loci.items():
        locus_indices = loci_by_cluster.get(cid)
        if not locus_indices:
            continue

        # Find reference locus
        ref_i = None
        for i in locus_indices:
            if int(l_lids[i]) == ref_lid:
                ref_i = i
                break
        if ref_i is None:
            continue

        ref_genome = l_genomes[ref_i]
        ref_contig = l_contigs[ref_i]
        ref_positions = list(range(int(l_starts[ref_i]), int(l_ends[ref_i]) + 1))
        n_slots = len(ref_positions)

        ref_embs, ref_gene_ids = _gather_embs(ref_genome, ref_contig, ref_positions)
        ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
        ref_embs_normed = ref_embs / (ref_norms + 1e-9)

        # Add reference assignments
        for slot_idx in range(n_slots):
            all_assignments.append({
                "cluster_id": int(cid),
                "locus_id": int(ref_lid),
                "slot_idx": slot_idx,
                "gene_id": ref_gene_ids[slot_idx],
                "gene_idx": ref_positions[slot_idx],
                "genome_id": ref_genome,
                "contig_id": ref_contig,
                "similarity": 1.0,
                "is_reference": True,
                "is_insertion": False,
            })

        # Align each other locus to reference slots
        for li in locus_indices:
            lid = int(l_lids[li])
            if lid == ref_lid:
                continue

            loc_genome = l_genomes[li]
            loc_contig = l_contigs[li]
            positions = list(range(int(l_starts[li]), int(l_ends[li]) + 1))

            orient = locus_orientation.get((cid, lid), 1)
            if orient == -1:
                positions = positions[::-1]

            mem_embs, mem_gene_ids = _gather_embs(loc_genome, loc_contig, positions)
            if len(positions) == 0:
                continue

            mem_norms = np.linalg.norm(mem_embs, axis=1, keepdims=True)
            mem_embs_normed = mem_embs / (mem_norms + 1e-9)

            sim_matrix = mem_embs_normed @ ref_embs_normed.T

            assignment = _monotonic_align(sim_matrix, n_slots)

            assigned_mem = set()
            for mem_idx, slot_idx in assignment:
                sim = float(sim_matrix[mem_idx, slot_idx])
                all_assignments.append({
                    "cluster_id": int(cid),
                    "locus_id": lid,
                    "slot_idx": slot_idx,
                    "gene_id": mem_gene_ids[mem_idx],
                    "gene_idx": positions[mem_idx],
                    "genome_id": loc_genome,
                    "contig_id": loc_contig,
                    "similarity": sim,
                    "is_reference": False,
                    "is_insertion": False,
                })
                assigned_mem.add(mem_idx)

            for mem_idx in range(len(positions)):
                if mem_idx not in assigned_mem:
                    nearest_slot = _find_insertion_slot(mem_idx, assignment, n_slots)
                    all_assignments.append({
                        "cluster_id": int(cid),
                        "locus_id": lid,
                        "slot_idx": nearest_slot,
                        "gene_id": mem_gene_ids[mem_idx],
                        "gene_idx": positions[mem_idx],
                        "genome_id": loc_genome,
                        "contig_id": loc_contig,
                        "similarity": 0.0,
                        "is_reference": False,
                        "is_insertion": True,
                    })

    return pd.DataFrame(all_assignments)


def _compute_locus_orientations(
    loci_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
) -> Dict[Tuple[int, int], int]:
    """Determine orientation of each locus relative to the reference.

    Uses the block orientation field. For a locus, if the majority of
    contributing blocks have orientation=-1, the locus is inverted.

    Returns: {(cluster_id, locus_id): orientation}
    """
    if loci_df.empty:
        return {}

    # Pre-extract blocks arrays for vectorized matching
    b_cid = blocks_df["cluster_id"].values
    b_q_genome = blocks_df["query_genome"].values
    b_q_contig = blocks_df["query_contig"].values
    b_q_start = blocks_df["query_start"].values
    b_q_end = blocks_df["query_end"].values
    b_t_genome = blocks_df["target_genome"].values
    b_t_contig = blocks_df["target_contig"].values
    b_t_start = blocks_df["target_start"].values
    b_t_end = blocks_df["target_end"].values
    b_orient = blocks_df["orientation"].values if "orientation" in blocks_df.columns else np.ones(len(blocks_df), dtype=np.int64)

    # Build cluster_id -> block index array for fast filtering
    cid_to_block_idx = {}
    for i in range(len(blocks_df)):
        c = b_cid[i]
        if c not in cid_to_block_idx:
            cid_to_block_idx[c] = []
        cid_to_block_idx[c].append(i)
    # Convert lists to arrays
    for c in cid_to_block_idx:
        cid_to_block_idx[c] = np.array(cid_to_block_idx[c], dtype=np.int64)

    orientations = {}
    l_cid = loci_df["cluster_id"].values
    l_lid = loci_df["locus_id"].values
    l_genome = loci_df["genome_id"].values
    l_contig = loci_df["contig_id"].values
    l_start = loci_df["start_idx"].values
    l_end = loci_df["end_idx"].values

    for i in range(len(loci_df)):
        cid = l_cid[i]
        lid = int(l_lid[i])
        genome = l_genome[i]
        contig = l_contig[i]
        s, e = l_start[i], l_end[i]

        idx = cid_to_block_idx.get(cid)
        if idx is None:
            orientations[(cid, lid)] = 1
            continue

        # Vectorized: check query-side and target-side overlap
        q_match = ((b_q_genome[idx] == genome) & (b_q_contig[idx] == contig)
                   & (b_q_start[idx] <= e) & (b_q_end[idx] >= s))
        t_match = ((b_t_genome[idx] == genome) & (b_t_contig[idx] == contig)
                   & (b_t_start[idx] <= e) & (b_t_end[idx] >= s))

        match_mask = q_match | t_match
        if match_mask.any():
            orient_sum = int(b_orient[idx[match_mask]].sum())
            orientations[(cid, lid)] = 1 if orient_sum >= 0 else -1
        else:
            orientations[(cid, lid)] = 1

    return orientations


def _monotonic_align(
    sim_matrix: np.ndarray,
    n_slots: int,
    min_sim: float = 0.3,
) -> List[Tuple[int, int]]:
    """Monotonic alignment of member genes to reference slots via DP.

    Finds the subset of (member_idx, slot_idx) pairs that maximizes total
    similarity subject to: if (i1, j1) and (i2, j2) are both selected and
    i1 < i2, then j1 < j2 (monotonicity / no crossings).

    Allows gaps (empty slots and unmatched member genes).
    """
    n_mem, n_ref = sim_matrix.shape
    if n_mem == 0 or n_ref == 0:
        return []

    # DP: dp[i][j] = best score using member genes 0..i-1, slots 0..j-1
    # where (i, j) means we assign member i-1 to slot j-1
    INF = -1e9
    dp = np.full((n_mem + 1, n_ref + 1), INF, dtype=np.float64)
    dp[0, :] = 0  # Skip any number of slots at the start

    for i in range(1, n_mem + 1):
        dp[i, 0] = 0  # Skip member gene i
        for j in range(1, n_ref + 1):
            sim = sim_matrix[i - 1, j - 1]

            # Option 1: Skip this slot (don't assign member i to slot j)
            dp[i, j] = dp[i, j - 1]

            # Option 2: Skip this member gene
            dp[i, j] = max(dp[i, j], dp[i - 1, j])

            # Option 3: Assign member i-1 to slot j-1
            if sim >= min_sim:
                # Best score from any (i', j') where i' < i, j' < j
                best_prev = dp[i - 1, j - 1]
                dp[i, j] = max(dp[i, j], best_prev + sim)

    # Traceback
    assignments = []
    i, j = n_mem, n_ref
    while i > 0 and j > 0:
        sim = sim_matrix[i - 1, j - 1]
        if sim >= min_sim and dp[i, j] == dp[i - 1, j - 1] + sim:
            assignments.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j]:
            i -= 1
        else:
            j -= 1

    return list(reversed(assignments))


def _find_insertion_slot(
    mem_idx: int,
    assignment: List[Tuple[int, int]],
    n_slots: int,
) -> int:
    """Find the slot position where an unmatched gene should be recorded.

    Places it between the nearest assigned neighbors, or at the edges.
    Returns a slot index (may equal an existing slot for edge insertions).
    """
    if not assignment:
        return 0

    # Find the nearest assigned gene before and after
    before = [(mi, si) for mi, si in assignment if mi < mem_idx]
    after = [(mi, si) for mi, si in assignment if mi > mem_idx]

    if before and after:
        return before[-1][1]  # Attach to preceding slot
    elif before:
        return min(before[-1][1] + 1, n_slots - 1)
    elif after:
        return max(after[0][1] - 1, 0)
    return 0


# ---------------------------------------------------------------------------
# Stage 5: Slot-level summary statistics
# ---------------------------------------------------------------------------

def compute_slot_summaries(
    assignments_df: pd.DataFrame,
    genes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-slot summary statistics for each cluster.

    Returns DataFrame with columns:
        cluster_id, slot_idx, occupancy, n_occupants, centroid_gene_id,
        dispersion, n_types, is_core, has_alternate
    """
    if assignments_df.empty:
        return pd.DataFrame()

    # Precompute embeddings: map gene_id -> row index for vectorized lookup
    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    emb_matrix = genes_df[emb_cols].values.astype(np.float32)
    gene_ids_arr = genes_df["gene_id"].values
    gene_to_row = {}
    for i in range(len(genes_df)):
        gene_to_row[gene_ids_arr[i]] = i

    # Filter out insertions for slot statistics
    slot_data = assignments_df[~assignments_df["is_insertion"]].copy()

    # Precompute n_loci per cluster (avoid repeated filtering)
    loci_per_cluster = assignments_df.groupby("cluster_id")["locus_id"].nunique()

    # Map gene_ids to embedding row indices for the entire slot_data at once
    slot_gene_ids = slot_data["gene_id"].values
    slot_emb_rows = np.array([gene_to_row.get(g, -1) for g in slot_gene_ids], dtype=np.int64)
    slot_data = slot_data.copy()
    slot_data["_emb_row"] = slot_emb_rows

    summaries = []
    for (cid, slot_idx), grp in slot_data.groupby(["cluster_id", "slot_idx"]):
        n_loci_in_cluster = loci_per_cluster.get(cid, 1)
        n_occupants = grp["locus_id"].nunique()
        occupancy = n_occupants / max(n_loci_in_cluster, 1)

        # Vectorized embedding gather
        emb_rows = grp["_emb_row"].values
        valid = emb_rows >= 0
        valid_rows = emb_rows[valid]

        if len(valid_rows) > 0:
            emb_stack = emb_matrix[valid_rows]
            valid_gene_ids = grp["gene_id"].values[valid]

            # Normalize
            norms = np.linalg.norm(emb_stack, axis=1, keepdims=True)
            emb_normed = emb_stack / (norms + 1e-9)

            # Centroid
            centroid = emb_normed.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid /= centroid_norm

            # Dispersion: 1 - mean cosine similarity to centroid
            cosines = emb_normed @ centroid
            dispersion = float(1.0 - cosines.mean())

            # Centroid gene: closest to centroid
            centroid_idx = int(np.argmax(cosines))
            centroid_gene_id = valid_gene_ids[centroid_idx]

            n_types = _estimate_n_types(emb_normed)
        else:
            dispersion = 0.0
            centroid_gene_id = grp.iloc[0]["gene_id"]
            n_types = 1

        is_core = occupancy >= 0.7 and dispersion < 0.3

        summaries.append({
            "cluster_id": int(cid),
            "slot_idx": int(slot_idx),
            "occupancy": round(occupancy, 3),
            "n_occupants": n_occupants,
            "centroid_gene_id": centroid_gene_id,
            "dispersion": round(dispersion, 4),
            "n_types": n_types,
            "is_core": is_core,
            "has_alternate": n_types > 1,
        })

    return pd.DataFrame(summaries)


def _estimate_n_types(emb_normed: np.ndarray, threshold: float = 0.4) -> int:
    """Estimate whether a slot has 1 or 2+ occupant types.

    Conservative heuristic: requires genuine bimodality — a substantial
    fraction of pairs below threshold AND low mean similarity, indicating
    two distinct protein families rather than normal sequence divergence.
    """
    n = len(emb_normed)
    if n < 3:
        return 1

    sim_matrix = emb_normed @ emb_normed.T
    tril_idx = np.tril_indices(n, k=-1)
    pairwise_sims = sim_matrix[tril_idx]

    if len(pairwise_sims) == 0:
        return 1

    low_sim_frac = (pairwise_sims < threshold).mean()
    mean_sim = float(pairwise_sims.mean())

    # Require both: >40% of pairs below threshold AND mean sim below 0.6
    # This avoids flagging single-divergent-member situations (mean ~0.8)
    if low_sim_frac > 0.4 and mean_sim < 0.6:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Stage 6: Cluster-level architecture summary
# ---------------------------------------------------------------------------

def compute_architecture_summary(
    slots_df: pd.DataFrame,
    loci_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-cluster architecture metrics.

    Returns DataFrame with columns:
        cluster_id, n_loci, n_slots, n_core_slots, n_variable_slots,
        mean_occupancy, coherence, has_replacement_hotspot, arch_label
    """
    if slots_df.empty:
        return pd.DataFrame()

    # Precompute n_loci per cluster
    loci_per_cluster = loci_df.groupby("cluster_id")["locus_id"].nunique()

    summaries = []
    for cid, grp in slots_df.groupby("cluster_id"):
        n_slots = len(grp)
        n_core = int(grp["is_core"].sum())
        n_variable = n_slots - n_core
        mean_occ = float(grp["occupancy"].mean())
        n_loci = int(loci_per_cluster.get(cid, 0))

        coherence = (n_core / max(n_slots, 1)) * mean_occ

        has_hotspot = bool(((grp["has_alternate"]) & (grp["occupancy"] >= 0.5)).any())

        if coherence >= 0.7 and not has_hotspot:
            arch_label = "coherent"
        elif has_hotspot:
            arch_label = "variable (replacement)"
        elif coherence >= 0.4:
            arch_label = "moderate"
        else:
            arch_label = "fragmented"

        summaries.append({
            "cluster_id": int(cid),
            "n_loci": n_loci,
            "n_slots": n_slots,
            "n_core_slots": n_core,
            "n_variable_slots": n_variable,
            "mean_occupancy": round(mean_occ, 3),
            "coherence": round(coherence, 3),
            "has_replacement_hotspot": has_hotspot,
            "arch_label": arch_label,
        })

    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_schema_pipeline(
    blocks_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    output_dir: Path,
    min_cluster_size: int = 3,
    max_slots: int = 50,
) -> Dict[str, Path]:
    """Run the full cluster architecture schema pipeline.

    Args:
        blocks_df: Syntenic blocks with cluster_id
        genes_df: Genes with embeddings
        output_dir: Where to write artifacts
        min_cluster_size: Minimum blocks per cluster to process
        max_slots: Skip clusters with reference loci larger than this

    Returns:
        Dict of artifact name -> path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to non-singleton clusters with sufficient blocks
    cluster_sizes = blocks_df[blocks_df["cluster_id"] > 0].groupby("cluster_id").size()
    eligible = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
    filtered_blocks = blocks_df[blocks_df["cluster_id"].isin(eligible)]

    _log(f"[Schema] {len(eligible)} clusters with >= {min_cluster_size} blocks")

    # Stage 1: Block members
    _log("[Schema] Building block members...")
    members_df = build_block_members(filtered_blocks, genes_df)
    members_path = output_dir / "syntenic_block_members.parquet"
    members_df.to_parquet(members_path, index=False)

    # Stage 2: Locus instances
    _log("[Schema] Extracting locus instances...")
    loci_df = extract_locus_instances(members_df)
    loci_path = output_dir / "cluster_loci.parquet"
    loci_df.to_parquet(loci_path, index=False)
    _log(f"[Schema] {len(loci_df)} locus instances across {loci_df['cluster_id'].nunique()} clusters")

    # Stage 3: Reference loci
    ref_loci = choose_reference_loci(loci_df)

    # Filter out clusters with oversized reference loci
    for cid in list(ref_loci.keys()):
        ref_row = loci_df[loci_df["locus_id"] == ref_loci[cid]].iloc[0]
        if ref_row["n_genes"] > max_slots:
            del ref_loci[cid]

    if not ref_loci:
        _log("[Schema] No eligible clusters after filtering")
        return {"block_members": members_path, "loci": loci_path}

    # Restrict loci to clusters with valid references
    loci_df = loci_df[loci_df["cluster_id"].isin(ref_loci.keys())]

    # Stage 4: Slot assignment
    _log(f"[Schema] Assigning slots for {len(ref_loci)} clusters...")
    assignments_df = assign_slots(loci_df, genes_df, ref_loci, filtered_blocks)
    assignments_path = output_dir / "cluster_slot_assignments.parquet"
    assignments_df.to_parquet(assignments_path, index=False)

    # Stage 5: Slot summaries
    _log("[Schema] Computing slot summaries...")
    slots_df = compute_slot_summaries(assignments_df, genes_df)
    slots_path = output_dir / "cluster_slots.parquet"
    slots_df.to_parquet(slots_path, index=False)

    # Stage 6: Architecture summary
    _log("[Schema] Computing architecture summaries...")
    arch_df = compute_architecture_summary(slots_df, loci_df)
    arch_path = output_dir / "cluster_architecture_summary.parquet"
    arch_df.to_parquet(arch_path, index=False)

    n_coherent = int((arch_df["arch_label"] == "coherent").sum())
    n_variable = int((arch_df["arch_label"].str.contains("variable")).sum())
    _log(f"[Schema] Architecture: {len(arch_df)} clusters, "
         f"{n_coherent} coherent, {n_variable} variable")

    return {
        "block_members": members_path,
        "loci": loci_path,
        "slot_assignments": assignments_path,
        "slots": slots_path,
        "architecture": arch_path,
    }
