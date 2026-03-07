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

    # Vectorized expansion: build arrays for both sides of all blocks
    block_ids = blocks_df["block_id"].values
    cluster_ids = blocks_df["cluster_id"].values
    q_genomes = blocks_df["query_genome"].values
    t_genomes = blocks_df["target_genome"].values
    q_contigs = blocks_df["query_contig"].values
    t_contigs = blocks_df["target_contig"].values
    q_starts = blocks_df["query_start"].values
    q_ends = blocks_df["query_end"].values
    t_starts = blocks_df["target_start"].values
    t_ends = blocks_df["target_end"].values

    out_bid = []
    out_cid = []
    out_side = []
    out_genome = []
    out_contig = []
    out_gidx = []
    out_gene_id = []
    out_strand = []
    out_order = []

    for i in range(len(blocks_df)):
        bid = int(block_ids[i])
        cid = int(cluster_ids[i])

        for side, genome, contig, s, e in [
            ("query", q_genomes[i], q_contigs[i], int(q_starts[i]), int(q_ends[i])),
            ("target", t_genomes[i], t_contigs[i], int(t_starts[i]), int(t_ends[i])),
        ]:
            for order, pos in enumerate(range(s, e + 1)):
                info = gene_lookup.get((genome, contig, pos))
                gid = info[0] if info else f"{genome}_{contig}_{pos}"
                strand = info[1] if info else 0
                out_bid.append(bid)
                out_cid.append(cid)
                out_side.append(side)
                out_genome.append(genome)
                out_contig.append(contig)
                out_gidx.append(pos)
                out_gene_id.append(gid)
                out_strand.append(strand)
                out_order.append(order)

    return pd.DataFrame({
        "block_id": out_bid,
        "cluster_id": out_cid,
        "side": out_side,
        "genome_id": out_genome,
        "contig_id": out_contig,
        "gene_idx": out_gidx,
        "gene_id": out_gene_id,
        "strand": out_strand,
        "anchor_order": out_order,
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

    # Group by cluster + genome + contig and collect intervals
    grouped = members_df.groupby(["cluster_id", "genome_id", "contig_id"])

    loci = []
    locus_id = 0

    for (cid, genome, contig), grp in grouped:
        # Get unique block-side intervals
        block_sides = grp.groupby(["block_id", "side"]).agg(
            start=("gene_idx", "min"),
            end=("gene_idx", "max"),
        ).reset_index()

        # Merge overlapping intervals
        intervals = list(zip(block_sides["start"], block_sides["end"]))
        merged = _merge_intervals(intervals)

        for start, end in merged:
            # Count how many block-sides overlap this merged interval
            support = sum(
                1 for s, e in intervals
                if s <= end and e >= start
            )
            loci.append({
                "locus_id": locus_id,
                "cluster_id": int(cid),
                "genome_id": genome,
                "contig_id": contig,
                "start_idx": start,
                "end_idx": end,
                "n_genes": end - start + 1,
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
    refs = {}
    for cid, grp in loci_df.groupby("cluster_id"):
        best = grp.sort_values(
            ["support", "n_genes", "locus_id"],
            ascending=[False, False, True],
        ).iloc[0]
        refs[int(cid)] = int(best["locus_id"])
    return refs


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

    all_assignments = []

    for cid, ref_lid in ref_loci.items():
        cluster_loci = loci_df[loci_df["cluster_id"] == cid]
        if cluster_loci.empty:
            continue

        ref_row = cluster_loci[cluster_loci["locus_id"] == ref_lid].iloc[0]
        ref_positions = list(range(ref_row["start_idx"], ref_row["end_idx"] + 1))
        n_slots = len(ref_positions)

        # Get reference embeddings
        ref_embs = []
        ref_gene_ids = []
        for pos in ref_positions:
            key = (ref_row["genome_id"], ref_row["contig_id"], pos)
            row_idx = pos_to_row.get(key)
            if row_idx is not None:
                ref_embs.append(emb_matrix[row_idx])
                ref_gene_ids.append(gene_id_arr[row_idx])
            else:
                ref_embs.append(np.zeros(len(emb_cols), dtype=np.float32))
                ref_gene_ids.append(f"MISSING_{key}")

        ref_embs = np.array(ref_embs)
        # Normalize reference embeddings
        ref_norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
        ref_embs_normed = ref_embs / (ref_norms + 1e-9)

        # Add reference assignments
        for slot_idx, pos in enumerate(ref_positions):
            all_assignments.append({
                "cluster_id": int(cid),
                "locus_id": int(ref_lid),
                "slot_idx": slot_idx,
                "gene_id": ref_gene_ids[slot_idx],
                "gene_idx": pos,
                "genome_id": ref_row["genome_id"],
                "contig_id": ref_row["contig_id"],
                "similarity": 1.0,
                "is_reference": True,
                "is_insertion": False,
            })

        # Align each other locus to reference slots
        for _, locus in cluster_loci.iterrows():
            lid = int(locus["locus_id"])
            if lid == ref_lid:
                continue

            positions = list(range(locus["start_idx"], locus["end_idx"] + 1))

            # Check if this locus is inverted relative to reference
            orient = locus_orientation.get((cid, lid), 1)
            if orient == -1:
                positions = positions[::-1]

            # Get member embeddings
            mem_embs = []
            mem_gene_ids = []
            mem_positions = []
            for pos in positions:
                key = (locus["genome_id"], locus["contig_id"], pos)
                row_idx = pos_to_row.get(key)
                if row_idx is not None:
                    mem_embs.append(emb_matrix[row_idx])
                    mem_gene_ids.append(gene_id_arr[row_idx])
                    mem_positions.append(pos)
                else:
                    mem_embs.append(np.zeros(len(emb_cols), dtype=np.float32))
                    mem_gene_ids.append(f"MISSING_{key}")
                    mem_positions.append(pos)

            if not mem_embs:
                continue

            mem_embs = np.array(mem_embs)
            mem_norms = np.linalg.norm(mem_embs, axis=1, keepdims=True)
            mem_embs_normed = mem_embs / (mem_norms + 1e-9)

            # Compute similarity matrix
            sim_matrix = mem_embs_normed @ ref_embs_normed.T  # (n_mem, n_slots)

            # Monotonic alignment via DP
            assignment = _monotonic_align(sim_matrix, n_slots)

            assigned_slots = set()
            for mem_idx, slot_idx in assignment:
                sim = float(sim_matrix[mem_idx, slot_idx])
                all_assignments.append({
                    "cluster_id": int(cid),
                    "locus_id": lid,
                    "slot_idx": slot_idx,
                    "gene_id": mem_gene_ids[mem_idx],
                    "gene_idx": mem_positions[mem_idx],
                    "genome_id": locus["genome_id"],
                    "contig_id": locus["contig_id"],
                    "similarity": sim,
                    "is_reference": False,
                    "is_insertion": False,
                })
                assigned_slots.add(slot_idx)

            # Mark unassigned member genes as insertions
            assigned_mem = {mi for mi, _ in assignment}
            for mem_idx in range(len(mem_positions)):
                if mem_idx not in assigned_mem:
                    # Find the nearest assigned slot for insertion position
                    nearest_slot = _find_insertion_slot(mem_idx, assignment, n_slots)
                    all_assignments.append({
                        "cluster_id": int(cid),
                        "locus_id": lid,
                        "slot_idx": nearest_slot,
                        "gene_id": mem_gene_ids[mem_idx],
                        "gene_idx": mem_positions[mem_idx],
                        "genome_id": locus["genome_id"],
                        "contig_id": locus["contig_id"],
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
    orientations = {}
    for cid, loci_grp in loci_df.groupby("cluster_id"):
        cluster_blocks = blocks_df[blocks_df["cluster_id"] == cid]
        for _, locus in loci_grp.iterrows():
            lid = int(locus["locus_id"])
            genome = locus["genome_id"]
            contig = locus["contig_id"]
            s, e = locus["start_idx"], locus["end_idx"]

            # Find blocks where this locus appears on either side
            orient_votes = []
            for _, blk in cluster_blocks.iterrows():
                if (blk["query_genome"] == genome and blk["query_contig"] == contig
                        and blk["query_start"] <= e and blk["query_end"] >= s):
                    orient_votes.append(int(blk.get("orientation", 1)))
                elif (blk["target_genome"] == genome and blk["target_contig"] == contig
                      and blk["target_start"] <= e and blk["target_end"] >= s):
                    # Target side: orientation is relative to query, so same
                    orient_votes.append(int(blk.get("orientation", 1)))

            if orient_votes:
                orientations[(cid, lid)] = 1 if sum(orient_votes) >= 0 else -1
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

    # Precompute embeddings lookup (vectorized)
    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    emb_matrix = genes_df[emb_cols].values.astype(np.float32)
    gene_ids_arr = genes_df["gene_id"].values
    gene_to_emb = {gene_ids_arr[i]: emb_matrix[i] for i in range(len(genes_df))}

    # Filter out insertions for slot statistics
    slot_data = assignments_df[~assignments_df["is_insertion"]].copy()

    summaries = []
    for (cid, slot_idx), grp in slot_data.groupby(["cluster_id", "slot_idx"]):
        n_loci_in_cluster = assignments_df[
            (assignments_df["cluster_id"] == cid)
        ]["locus_id"].nunique()

        n_occupants = grp["locus_id"].nunique()
        occupancy = n_occupants / max(n_loci_in_cluster, 1)

        # Gather embeddings for this slot
        embs = []
        gene_ids = []
        for _, row in grp.iterrows():
            e = gene_to_emb.get(row["gene_id"])
            if e is not None:
                embs.append(e)
                gene_ids.append(row["gene_id"])

        if embs:
            emb_stack = np.array(embs)
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
            centroid_gene_id = gene_ids[centroid_idx]

            # Detect multiple types: simple heuristic
            # If there are two groups with mean inter-group similarity < 0.5
            n_types = _estimate_n_types(emb_normed)
        else:
            dispersion = 0.0
            centroid_gene_id = grp.iloc[0]["gene_id"]
            n_types = 1

        # Core: high occupancy (>=0.7) and low dispersion (<0.3)
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


def _estimate_n_types(emb_normed: np.ndarray, threshold: float = 0.5) -> int:
    """Estimate whether a slot has 1 or 2+ occupant types.

    Simple heuristic: compute pairwise similarities and check if there's
    a bimodal split (some pairs with very low similarity).
    """
    n = len(emb_normed)
    if n < 3:
        return 1

    # Pairwise similarities
    sim_matrix = emb_normed @ emb_normed.T
    # Get lower triangle values
    tril_idx = np.tril_indices(n, k=-1)
    pairwise_sims = sim_matrix[tril_idx]

    if len(pairwise_sims) == 0:
        return 1

    # If >25% of pairs have similarity below threshold, likely 2+ types
    low_sim_frac = (pairwise_sims < threshold).mean()
    if low_sim_frac > 0.25:
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

    summaries = []
    for cid, grp in slots_df.groupby("cluster_id"):
        n_slots = len(grp)
        n_core = int(grp["is_core"].sum())
        n_variable = n_slots - n_core
        mean_occ = float(grp["occupancy"].mean())
        n_loci = int(loci_df[loci_df["cluster_id"] == cid]["locus_id"].nunique())

        # Coherence: fraction of core slots × mean occupancy
        coherence = (n_core / max(n_slots, 1)) * mean_occ

        # Replacement hotspot: any slot with >1 type and occupancy >= 0.5
        hotspot_slots = grp[(grp["has_alternate"]) & (grp["occupancy"] >= 0.5)]
        has_hotspot = len(hotspot_slots) > 0

        # Architecture label
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
    import sys

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to non-singleton clusters with sufficient blocks
    cluster_sizes = blocks_df[blocks_df["cluster_id"] > 0].groupby("cluster_id").size()
    eligible = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
    filtered_blocks = blocks_df[blocks_df["cluster_id"].isin(eligible)]

    print(f"[Schema] {len(eligible)} clusters with >= {min_cluster_size} blocks",
          file=sys.stderr, flush=True)

    # Stage 1: Block members
    print("[Schema] Building block members...", file=sys.stderr, flush=True)
    members_df = build_block_members(filtered_blocks, genes_df)
    members_path = output_dir / "syntenic_block_members.parquet"
    members_df.to_parquet(members_path, index=False)

    # Stage 2: Locus instances
    print("[Schema] Extracting locus instances...", file=sys.stderr, flush=True)
    loci_df = extract_locus_instances(members_df)
    loci_path = output_dir / "cluster_loci.parquet"
    loci_df.to_parquet(loci_path, index=False)
    print(f"[Schema] {len(loci_df)} locus instances across {loci_df['cluster_id'].nunique()} clusters",
          file=sys.stderr, flush=True)

    # Stage 3: Reference loci
    ref_loci = choose_reference_loci(loci_df)

    # Filter out clusters with oversized reference loci
    for cid in list(ref_loci.keys()):
        ref_row = loci_df[loci_df["locus_id"] == ref_loci[cid]].iloc[0]
        if ref_row["n_genes"] > max_slots:
            del ref_loci[cid]

    if not ref_loci:
        print("[Schema] No eligible clusters after filtering", file=sys.stderr, flush=True)
        return {"block_members": members_path, "loci": loci_path}

    # Restrict loci to clusters with valid references
    loci_df = loci_df[loci_df["cluster_id"].isin(ref_loci.keys())]

    # Stage 4: Slot assignment
    print(f"[Schema] Assigning slots for {len(ref_loci)} clusters...",
          file=sys.stderr, flush=True)
    assignments_df = assign_slots(loci_df, genes_df, ref_loci, filtered_blocks)
    assignments_path = output_dir / "cluster_slot_assignments.parquet"
    assignments_df.to_parquet(assignments_path, index=False)

    # Stage 5: Slot summaries
    print("[Schema] Computing slot summaries...", file=sys.stderr, flush=True)
    slots_df = compute_slot_summaries(assignments_df, genes_df)
    slots_path = output_dir / "cluster_slots.parquet"
    slots_df.to_parquet(slots_path, index=False)

    # Stage 6: Architecture summary
    print("[Schema] Computing architecture summaries...", file=sys.stderr, flush=True)
    arch_df = compute_architecture_summary(slots_df, loci_df)
    arch_path = output_dir / "cluster_architecture_summary.parquet"
    arch_df.to_parquet(arch_path, index=False)

    n_coherent = int((arch_df["arch_label"] == "coherent").sum())
    n_variable = int((arch_df["arch_label"].str.contains("variable")).sum())
    print(f"[Schema] Architecture: {len(arch_df)} clusters, "
          f"{n_coherent} coherent, {n_variable} variable",
          file=sys.stderr, flush=True)

    return {
        "block_members": members_path,
        "loci": loci_path,
        "slot_assignments": assignments_path,
        "slots": slots_path,
        "architecture": arch_path,
    }
