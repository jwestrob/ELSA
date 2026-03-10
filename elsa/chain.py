"""
Collinear anchor chaining via LIS-based dynamic programming.

Pure algorithm module — no I/O, no side effects.
Accepts anchor DataFrames (columnar) for efficiency on large datasets.
Uses Numba JIT for the O(n²) DP inner loop when available.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Numba-accelerated DP kernels (optional — pure-Python fallback below)
# ---------------------------------------------------------------------------
# Force Numba to use its own thread pool ("workqueue") instead of OpenMP.
# FAISS also uses OpenMP via a separate libomp; KMP_DUPLICATE_LIB_OK=TRUE
# prevents the immediate crash but having two OpenMP runtimes causes silent
# thread-pool corruption → SIGKILL on macOS (seen at 7-22 GB RSS, well
# under the 48 GB limit).  workqueue uses POSIX threads directly and
# supports prange parallelism without any OpenMP interaction.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

try:
    from numba import njit, prange

    @njit(cache=True)
    def _dp_fill(sq, st, ss, cmp_vals, max_gap, gap_penalty_scale,
                 dp_len, dp_prev, dp_score):
        """Fill DP arrays for LIS-based collinear chaining (single group)."""
        n = len(sq)
        for i in range(1, n):
            best_len = 1
            best_prev = -1
            best_score = ss[i]

            for j in range(i - 1, -1, -1):
                gap_q = sq[i] - sq[j] - 1
                if max_gap >= 0 and gap_q > max_gap:
                    break

                if cmp_vals[i] <= cmp_vals[j]:
                    continue

                gap_t = abs(st[i] - st[j]) - 1
                if max_gap >= 0 and gap_t > max_gap:
                    continue

                if gap_penalty_scale > 0.0:
                    g = gap_q if gap_q > gap_t else gap_t
                    gap_cost = gap_penalty_scale * math.log2(g + 1)
                else:
                    gap_cost = 0.0

                new_len = dp_len[j] + 1
                new_score = dp_score[j] + ss[i] - gap_cost
                if new_len > best_len or (new_len == best_len and new_score > best_score):
                    best_len = new_len
                    best_prev = j
                    best_score = new_score

            dp_len[i] = best_len
            dp_prev[i] = best_prev
            dp_score[i] = best_score

    @njit(cache=True)
    def _dp_fill_batched(sq, st, ss, cmp_vals, offsets, sizes,
                         max_gap, gap_penalty_scale,
                         dp_len, dp_prev, dp_score):
        """Fill DP arrays for all groups in a single call (serial)."""
        n_groups = len(sizes)
        for g in range(n_groups):
            off = offsets[g]
            n = sizes[g]
            for i in range(1, n):
                ii = off + i
                best_len = np.int32(1)
                best_prev = np.int64(-1)
                best_score = ss[ii]

                for j in range(i - 1, -1, -1):
                    jj = off + j
                    gap_q = sq[ii] - sq[jj] - 1
                    if max_gap >= 0 and gap_q > max_gap:
                        break
                    if cmp_vals[ii] <= cmp_vals[jj]:
                        continue
                    gap_t = abs(st[ii] - st[jj]) - 1
                    if max_gap >= 0 and gap_t > max_gap:
                        continue
                    if gap_penalty_scale > 0.0:
                        gg = gap_q if gap_q > gap_t else gap_t
                        gap_cost = gap_penalty_scale * math.log2(gg + 1)
                    else:
                        gap_cost = 0.0
                    new_len = dp_len[jj] + np.int32(1)
                    new_score = dp_score[jj] + ss[ii] - gap_cost
                    if new_len > best_len or (new_len == best_len and new_score > best_score):
                        best_len = new_len
                        best_prev = jj
                        best_score = new_score

                dp_len[ii] = best_len
                dp_prev[ii] = best_prev
                dp_score[ii] = best_score

    @njit(cache=True, parallel=True)
    def _dp_fill_batched_parallel(sq, st, ss, cmp_vals, offsets, sizes,
                                  max_gap, gap_penalty_scale,
                                  dp_len, dp_prev, dp_score):
        """Fill DP arrays for all groups — parallel over groups via prange."""
        n_groups = len(sizes)
        for g in prange(n_groups):
            off = offsets[g]
            n = sizes[g]
            for i in range(1, n):
                ii = off + i
                best_len = np.int32(1)
                best_prev = np.int64(-1)
                best_score = ss[ii]

                for j in range(i - 1, -1, -1):
                    jj = off + j
                    gap_q = sq[ii] - sq[jj] - 1
                    if max_gap >= 0 and gap_q > max_gap:
                        break
                    if cmp_vals[ii] <= cmp_vals[jj]:
                        continue
                    gap_t = abs(st[ii] - st[jj]) - 1
                    if max_gap >= 0 and gap_t > max_gap:
                        continue
                    if gap_penalty_scale > 0.0:
                        gg = gap_q if gap_q > gap_t else gap_t
                        gap_cost = gap_penalty_scale * math.log2(gg + 1)
                    else:
                        gap_cost = 0.0
                    new_len = dp_len[jj] + np.int32(1)
                    new_score = dp_score[jj] + ss[ii] - gap_cost
                    if new_len > best_len or (new_len == best_len and new_score > best_score):
                        best_len = new_len
                        best_prev = jj
                        best_score = new_score

                dp_len[ii] = best_len
                dp_prev[ii] = best_prev
                dp_score[ii] = best_score

    @njit(cache=True)
    def _backtrack(dp_len, dp_prev, dp_score, min_size):
        """Greedy backtracking: extract chains longest-first (single group)."""
        n = len(dp_len)

        # Sort indices by (-dp_len, -dp_score) — insertion sort is fine for small n
        order = np.empty(n, dtype=np.int32)
        for i in range(n):
            order[i] = i
        for i in range(1, n):
            key = order[i]
            j = i - 1
            while j >= 0 and (dp_len[order[j]] < dp_len[key] or
                              (dp_len[order[j]] == dp_len[key] and
                               dp_score[order[j]] < dp_score[key])):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        used = np.zeros(n, dtype=np.bool_)
        chains = []
        for idx in range(n):
            i = order[idx]
            if used[i] or dp_len[i] < min_size:
                continue

            chain_buf = np.empty(dp_len[i], dtype=np.int32)
            count = 0
            j = i
            while j >= 0:
                if used[j]:
                    break
                chain_buf[count] = j
                used[j] = True
                count += 1
                j = dp_prev[j]

            if count >= min_size:
                chain = chain_buf[:count][::-1]  # reverse to forward order
                chains.append(chain)

        return chains

    @njit(cache=True)
    def _backtrack_batched(dp_len, dp_prev, dp_score, offsets, sizes, min_size,
                           chain_buf, chain_group_buf, chain_len_buf):
        """Greedy backtracking for all groups in a single call.

        Writes local indices into chain_buf, partition index into
        chain_group_buf, and chain length into chain_len_buf.
        Returns (n_chains, n_elements_written).
        """
        n_groups = len(sizes)
        total_chains = np.int64(0)
        write_pos = np.int64(0)

        for g in range(n_groups):
            off = offsets[g]
            n = sizes[g]

            # Sort local indices by (-dp_len, -dp_score) via insertion sort
            order = np.empty(n, dtype=np.int64)
            for i in range(n):
                order[i] = i
            for i in range(1, n):
                key_i = order[i]
                ki = off + key_i
                j = i - 1
                while j >= 0:
                    oj = off + order[j]
                    if dp_len[oj] > dp_len[ki] or (dp_len[oj] == dp_len[ki] and dp_score[oj] >= dp_score[ki]):
                        break
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = key_i

            used = np.zeros(n, dtype=np.bool_)

            for idx in range(n):
                i_local = order[idx]
                i_global = off + i_local
                if used[i_local] or dp_len[i_global] < min_size:
                    continue

                chain_start = write_pos
                count = np.int64(0)
                j_global = i_global
                while j_global >= off:
                    j_local = j_global - off
                    if used[j_local]:
                        break
                    chain_buf[write_pos] = j_local
                    used[j_local] = True
                    count += 1
                    write_pos += 1
                    j_global = dp_prev[j_global]

                if count >= min_size:
                    # Reverse in-place to get forward order
                    left = chain_start
                    right = write_pos - 1
                    while left < right:
                        tmp = chain_buf[left]
                        chain_buf[left] = chain_buf[right]
                        chain_buf[right] = tmp
                        left += 1
                        right -= 1
                    chain_group_buf[total_chains] = g
                    chain_len_buf[total_chains] = count
                    total_chains += 1
                else:
                    write_pos = chain_start  # discard short chain

        return total_chains, write_pos

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


@dataclass
class ChainedBlock:
    """A chained syntenic block from collinear gene anchors."""
    block_id: int
    query_genome: str
    target_genome: str
    query_contig: str
    target_contig: str
    query_start: int         # Gene index (0-based)
    query_end: int           # Gene index (inclusive)
    target_start: int
    target_end: int
    anchor_query_ids: List[int] = field(default_factory=list)
    anchor_target_ids: List[int] = field(default_factory=list)
    anchor_query_gene_ids: List[str] = field(default_factory=list)
    anchor_target_gene_ids: List[str] = field(default_factory=list)
    n_anchors: int = 0
    chain_score: float = 0.0
    orientation: int = 1     # +1 forward, -1 inverted

    @property
    def query_span(self) -> int:
        return self.query_end - self.query_start + 1

    @property
    def target_span(self) -> int:
        return self.target_end - self.target_start + 1

    def query_gene_ids(self) -> List[str]:
        """Get ordered query gene IDs from anchors."""
        return [gid for _, gid in sorted(zip(self.anchor_query_ids, self.anchor_query_gene_ids))]

    def target_gene_ids(self) -> List[str]:
        """Get ordered target gene IDs from anchors."""
        return [gid for _, gid in sorted(zip(self.anchor_target_ids, self.anchor_target_gene_ids))]


def chain_anchors_lis(
    anchors_df: pd.DataFrame,
    max_gap: int = 2,
    min_size: int = 2,
    gap_penalty_scale: float = 0.0,
) -> List[pd.DataFrame]:
    """
    Find collinear chains using LIS-based dynamic programming.

    Accepts a DataFrame of anchors for a single contig pair.
    Returns list of sub-DataFrames, one per chain.

    Args:
        anchors_df: DataFrame with at least query_idx, target_idx,
                    similarity, orientation columns
        max_gap: Maximum gap (in genes) between chain members
        min_size: Minimum anchors required for a valid chain
        gap_penalty_scale: Concave gap penalty multiplier

    Returns:
        List of DataFrames (each is a chain of anchors)
    """
    if len(anchors_df) < min_size:
        return []

    q_idx = anchors_df["query_idx"].values
    t_idx = anchors_df["target_idx"].values
    sims = anchors_df["similarity"].values
    orients = anchors_df["orientation"].values

    # Deduplicate: keep best anchor per (query_idx, target_idx) pair
    pair_keys = q_idx.astype(np.int64) * 1_000_000_000 + t_idx.astype(np.int64)
    order = np.argsort(-sims)  # highest sim first
    _, first_idx = np.unique(pair_keys[order], return_index=True)
    keep_mask = np.zeros(len(anchors_df), dtype=bool)
    keep_mask[order[first_idx]] = True

    if keep_mask.sum() < min_size:
        return []

    deduped = anchors_df.loc[keep_mask].reset_index(drop=True)
    q_idx = deduped["query_idx"].values
    t_idx = deduped["target_idx"].values
    sims = deduped["similarity"].values
    orients = deduped["orientation"].values

    # Check if strand info is available
    has_strand = np.any(orients != 0)

    if has_strand:
        partitions = []
        fwd_mask = orients >= 0
        rev_mask = orients < 0
        if fwd_mask.sum() >= min_size:
            partitions.append((deduped.loc[fwd_mask].reset_index(drop=True), False))
        if rev_mask.sum() >= min_size:
            partitions.append((deduped.loc[rev_mask].reset_index(drop=True), True))
    else:
        partitions = [(deduped, False), (deduped, True)]

    all_chains = []

    for part_df, reverse_target in partitions:
        pq = part_df["query_idx"].values
        pt = part_df["target_idx"].values
        ps = part_df["similarity"].values

        # Deduplicate per query_idx (keep best match)
        q_order = np.argsort(-ps)
        _, q_first = np.unique(pq[q_order], return_index=True)
        q_keep = np.zeros(len(part_df), dtype=bool)
        q_keep[q_order[q_first]] = True

        if q_keep.sum() < min_size:
            continue

        sub_df = part_df.loc[q_keep].reset_index(drop=True)

        # Sort by query position
        sort_order = np.argsort(sub_df["query_idx"].values)
        sub_df = sub_df.iloc[sort_order].reset_index(drop=True)

        sq = sub_df["query_idx"].values
        st = sub_df["target_idx"].values
        ss = sub_df["similarity"].values

        if reverse_target:
            cmp_vals = -st
        else:
            cmp_vals = st.copy()

        n = len(sub_df)
        # Ensure contiguous int64/float64 for Numba
        sq_c = np.ascontiguousarray(sq, dtype=np.int64)
        st_c = np.ascontiguousarray(st, dtype=np.int64)
        ss_c = np.ascontiguousarray(ss, dtype=np.float64)
        cmp_c = np.ascontiguousarray(cmp_vals, dtype=np.int64)

        dp_len = np.ones(n, dtype=np.int32)
        dp_prev = np.full(n, -1, dtype=np.int32)
        dp_score = ss_c.copy()

        _max_gap = max_gap if max_gap is not None else -1

        if _NUMBA_AVAILABLE:
            _dp_fill(sq_c, st_c, ss_c, cmp_c, _max_gap, gap_penalty_scale,
                     dp_len, dp_prev, dp_score)
            extracted = _backtrack(dp_len, dp_prev, dp_score, min_size)
        else:
            # Pure-Python fallback
            for i in range(1, n):
                best_len, best_prev, best_score = 1, -1, ss_c[i]
                for j in range(i - 1, -1, -1):
                    gap_q = sq_c[i] - sq_c[j] - 1
                    if _max_gap >= 0 and gap_q > _max_gap:
                        break
                    if cmp_c[i] <= cmp_c[j]:
                        continue
                    gap_t = abs(st_c[i] - st_c[j]) - 1
                    if _max_gap >= 0 and gap_t > _max_gap:
                        continue
                    if gap_penalty_scale > 0:
                        gap_cost = gap_penalty_scale * math.log2(max(gap_q, gap_t) + 1)
                    else:
                        gap_cost = 0.0
                    new_len = dp_len[j] + 1
                    new_score = dp_score[j] + ss_c[i] - gap_cost
                    if new_len > best_len or (new_len == best_len and new_score > best_score):
                        best_len = new_len
                        best_prev = j
                        best_score = new_score
                dp_len[i] = best_len
                dp_prev[i] = best_prev
                dp_score[i] = best_score

            # Python backtracking
            used = set()
            indices = sorted(range(n), key=lambda i: (-dp_len[i], -dp_score[i]))
            extracted = []
            for i in indices:
                if i in used or dp_len[i] < min_size:
                    continue
                chain_idx = []
                j = i
                while j >= 0:
                    if j in used:
                        break
                    chain_idx.append(j)
                    used.add(j)
                    j = int(dp_prev[j])
                if len(chain_idx) >= min_size:
                    chain_idx.reverse()
                    extracted.append(np.array(chain_idx, dtype=np.int32))

        orientation = -1 if reverse_target else 1

        for chain_idx in extracted:
            chain_df = sub_df.iloc[chain_idx].copy()
            chain_df["orientation"] = orientation
            all_chains.append(chain_df.reset_index(drop=True))

    return all_chains


def extract_nonoverlapping_chains(
    chains: List[pd.DataFrame],
    block_id_start: int = 0,
) -> List[ChainedBlock]:
    """
    Extract non-overlapping blocks from chains using greedy selection.

    Args:
        chains: List of anchor DataFrames from chain_anchors_lis
        block_id_start: Starting block ID for numbering

    Returns:
        List of ChainedBlock objects
    """
    if not chains:
        return []

    # Score each chain
    scored = []
    for chain_df in chains:
        if chain_df.empty:
            continue
        q = chain_df["query_idx"].values
        t = chain_df["target_idx"].values
        s = chain_df["similarity"].values

        mean_sim = s.mean()
        score = len(chain_df) * mean_sim
        q_min, q_max = int(q.min()), int(q.max())
        t_min, t_max = int(t.min()), int(t.max())
        orientation = int(chain_df["orientation"].iloc[0])

        scored.append((score, chain_df, q_min, q_max, t_min, t_max, orientation))

    scored.sort(key=lambda x: -x[0])

    # Greedy selection avoiding overlaps
    blocks = []
    used_query: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    used_target: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

    def overlaps(intervals: List[Tuple[int, int]], start: int, end: int) -> bool:
        for s, e in intervals:
            if not (end < s or start > e):
                return True
        return False

    block_id = block_id_start
    for score, chain_df, q_min, q_max, t_min, t_max, orientation in scored:
        q_key = (str(chain_df["query_genome"].iloc[0]),
                 str(chain_df["query_contig"].iloc[0]))
        t_key = (str(chain_df["target_genome"].iloc[0]),
                 str(chain_df["target_contig"].iloc[0]))

        if overlaps(used_query.get(q_key, []), q_min, q_max):
            continue
        if overlaps(used_target.get(t_key, []), t_min, t_max):
            continue

        used_query.setdefault(q_key, []).append((q_min, q_max))
        used_target.setdefault(t_key, []).append((t_min, t_max))

        # Extract anchor info for the block
        qi = chain_df["query_idx"].values
        ti = chain_df["target_idx"].values
        q_sort = np.argsort(qi)
        t_sort = np.argsort(ti)

        block = ChainedBlock(
            block_id=block_id,
            query_genome=str(chain_df["query_genome"].iloc[0]),
            target_genome=str(chain_df["target_genome"].iloc[0]),
            query_contig=str(chain_df["query_contig"].iloc[0]),
            target_contig=str(chain_df["target_contig"].iloc[0]),
            query_start=q_min,
            query_end=q_max,
            target_start=t_min,
            target_end=t_max,
            anchor_query_ids=qi[q_sort].tolist(),
            anchor_target_ids=ti[t_sort].tolist(),
            anchor_query_gene_ids=chain_df["query_gene_id"].values[q_sort].tolist(),
            anchor_target_gene_ids=chain_df["target_gene_id"].values[t_sort].tolist(),
            n_anchors=len(chain_df),
            chain_score=score,
            orientation=orientation,
        )
        blocks.append(block)
        block_id += 1

    return blocks


# ---------------------------------------------------------------------------
# Preprocessing helper (shared logic for per-group and batched paths)
# ---------------------------------------------------------------------------
def _preprocess_group(anchors_df, min_size):
    """Preprocess a contig-pair group for DP chaining.

    Yields (sub_df, reverse_target) tuples — deduplicated, partitioned
    by orientation, and sorted by query position.
    """
    if len(anchors_df) < min_size:
        return

    q_idx = anchors_df["query_idx"].values
    t_idx = anchors_df["target_idx"].values
    sims = anchors_df["similarity"].values
    orients = anchors_df["orientation"].values

    # Deduplicate by (query_idx, target_idx) pair — keep highest similarity
    pair_keys = q_idx.astype(np.int64) * 1_000_000_000 + t_idx.astype(np.int64)
    order = np.argsort(-sims)
    _, first_idx = np.unique(pair_keys[order], return_index=True)
    keep_mask = np.zeros(len(anchors_df), dtype=bool)
    keep_mask[order[first_idx]] = True

    if keep_mask.sum() < min_size:
        return

    deduped = anchors_df.loc[keep_mask].reset_index(drop=True)
    orients = deduped["orientation"].values
    has_strand = np.any(orients != 0)

    if has_strand:
        partitions = []
        fwd_mask = orients >= 0
        rev_mask = orients < 0
        if fwd_mask.sum() >= min_size:
            partitions.append((deduped.loc[fwd_mask].reset_index(drop=True), False))
        if rev_mask.sum() >= min_size:
            partitions.append((deduped.loc[rev_mask].reset_index(drop=True), True))
    else:
        partitions = [(deduped, False), (deduped, True)]

    for part_df, reverse_target in partitions:
        pq = part_df["query_idx"].values
        ps = part_df["similarity"].values

        # Deduplicate per query_idx — keep best similarity
        q_order = np.argsort(-ps)
        _, q_first = np.unique(pq[q_order], return_index=True)
        q_keep = np.zeros(len(part_df), dtype=bool)
        q_keep[q_order[q_first]] = True

        if q_keep.sum() < min_size:
            continue

        sub_df = part_df.loc[q_keep].reset_index(drop=True)
        sort_order = np.argsort(sub_df["query_idx"].values)
        sub_df = sub_df.iloc[sort_order].reset_index(drop=True)

        yield sub_df, reverse_target


# ---------------------------------------------------------------------------
# Array-only preprocessing (avoids intermediate DataFrames)
# ---------------------------------------------------------------------------
def _preprocess_group_raw(q_idx, t_idx, sims, orients, min_size):
    """Preprocess a group from raw numpy arrays (no DataFrame).

    Yields (sub_idx, sq, st, ss, cmp, reverse_target) where sub_idx
    are indices into the input arrays.
    """
    n = len(q_idx)
    if n < min_size:
        return

    # Dedup by (qi, ti) pair — keep highest similarity
    pair_keys = q_idx.astype(np.int64) * 1_000_000_000 + t_idx.astype(np.int64)
    order = np.argsort(-sims)
    _, first_idx = np.unique(pair_keys[order], return_index=True)
    dedup_idx = order[first_idx]

    if len(dedup_idx) < min_size:
        return

    dedup_orients = orients[dedup_idx]
    has_strand = np.any(dedup_orients != 0)

    if has_strand:
        partitions = []
        fwd_mask = dedup_orients >= 0
        rev_mask = dedup_orients < 0
        if fwd_mask.sum() >= min_size:
            partitions.append((dedup_idx[fwd_mask], False))
        if rev_mask.sum() >= min_size:
            partitions.append((dedup_idx[rev_mask], True))
    else:
        partitions = [(dedup_idx, False), (dedup_idx.copy(), True)]

    for part_idx, reverse_target in partitions:
        pq = q_idx[part_idx]
        ps = sims[part_idx]

        q_order = np.argsort(-ps)
        _, q_first = np.unique(pq[q_order], return_index=True)
        sub_idx = part_idx[q_order[q_first]]

        if len(sub_idx) < min_size:
            continue

        sq = q_idx[sub_idx]
        sort_order = np.argsort(sq)
        sub_idx = sub_idx[sort_order]

        sq = q_idx[sub_idx]
        st = t_idx[sub_idx]
        ss = sims[sub_idx]
        cmp = -st if reverse_target else st.copy()

        yield sub_idx, sq, st, ss, cmp, reverse_target


def _preprocess_group_arrays(anchors_df, min_size):
    """Preprocess a group using only numpy index arrays.

    Yields (df_indices, sq, st, ss, cmp, reverse_target) where df_indices
    maps local partition positions back to rows in the original anchors_df.
    No intermediate DataFrames are created.
    """
    n = len(anchors_df)
    if n < min_size:
        return

    q_idx = anchors_df["query_idx"].values
    t_idx = anchors_df["target_idx"].values
    sims = anchors_df["similarity"].values
    orients = anchors_df["orientation"].values

    # Dedup by (qi, ti) pair — keep highest similarity
    pair_keys = q_idx.astype(np.int64) * 1_000_000_000 + t_idx.astype(np.int64)
    order = np.argsort(-sims)
    _, first_idx = np.unique(pair_keys[order], return_index=True)
    dedup_idx = order[first_idx]  # indices into original df

    if len(dedup_idx) < min_size:
        return

    dedup_orients = orients[dedup_idx]
    has_strand = np.any(dedup_orients != 0)

    if has_strand:
        partitions = []
        fwd_mask = dedup_orients >= 0
        rev_mask = dedup_orients < 0
        if fwd_mask.sum() >= min_size:
            partitions.append((dedup_idx[fwd_mask], False))
        if rev_mask.sum() >= min_size:
            partitions.append((dedup_idx[rev_mask], True))
    else:
        partitions = [(dedup_idx, False), (dedup_idx.copy(), True)]

    for part_idx, reverse_target in partitions:
        pq = q_idx[part_idx]
        ps = sims[part_idx]

        # Dedup per query_idx — keep best similarity
        q_order = np.argsort(-ps)
        _, q_first = np.unique(pq[q_order], return_index=True)
        sub_idx = part_idx[q_order[q_first]]

        if len(sub_idx) < min_size:
            continue

        # Sort by query position
        sq = q_idx[sub_idx]
        sort_order = np.argsort(sq)
        sub_idx = sub_idx[sort_order]

        sq = q_idx[sub_idx]
        st = t_idx[sub_idx]
        ss = sims[sub_idx]
        cmp = -st if reverse_target else st.copy()

        yield sub_idx, sq, st, ss, cmp, reverse_target


# ---------------------------------------------------------------------------
# Batched chaining — single Numba call for all contig-pair groups
# ---------------------------------------------------------------------------
def chain_groups_batched(
    groups,
    max_gap: int = 2,
    min_size: int = 2,
    gap_penalty_scale: float = 0.0,
) -> Dict:
    """Chain all contig-pair groups using batched Numba kernels.

    Accepts either a GroupedAnchors namedtuple (fast path — raw arrays +
    group boundaries, no DataFrame construction) or a legacy Dict mapping
    (qg, tg, qc, tc) keys to anchor DataFrames.

    Falls back to per-group chain_anchors_lis if Numba is not available.

    Returns:
        Dict mapping group keys to lists of chain DataFrames
    """
    # --- Detect input format ---
    is_grouped = hasattr(groups, 'order')

    if not _NUMBA_AVAILABLE:
        if is_grouped:
            raise RuntimeError("GroupedAnchors requires Numba for batched chaining")
        result = {}
        for key, anchors_df in groups.items():
            chains = chain_anchors_lis(anchors_df, max_gap, min_size, gap_penalty_scale)
            if chains:
                result[key] = chains
        return result

    # Phase 1: Preprocess all groups into partitions.
    #
    # Two-pass approach for GroupedAnchors (fast path):
    #   Pass 1 — count partition sizes (no large allocations)
    #   Pass 2 — write directly into pre-allocated flat arrays
    # This avoids collecting millions of small numpy arrays in Python lists,
    # which at 12M+ partitions consumes ~5 GB in object overhead alone.
    from tqdm import tqdm

    if is_grouped:
        # --- Fast path: GroupedAnchors (raw arrays + boundaries) ---
        #
        # Stream-to-flat: preprocess each group and write directly into
        # pre-allocated flat arrays. Avoids collecting millions of small
        # numpy arrays in Python lists (which at 12M+ partitions consumes
        # ~5 GB in object overhead alone).
        #
        # Two passes:
        #   Pass 1 — preprocess all groups, store only (sub_idx, rev, size)
        #            per partition + group index. Total memory: ~12M small
        #            index arrays (unavoidable) but NO sq/st/ss/cmp copies.
        #   Pass 2 — replay group slicing, use stored sub_idx to write
        #            sq/st/ss/cmp directly into flat arrays.
        ga = groups

        group_sizes = np.diff(ga.group_bounds)
        valid_groups = np.where(group_sizes >= min_size)[0]
        n_skipped = ga.n_groups - len(valid_groups)
        if n_skipped > 0:
            print(f"[Chain] Skipping {n_skipped:,} / {ga.n_groups:,} groups "
                  f"with < {min_size} anchors", file=sys.stderr, flush=True)

        # Pass 1: preprocess, store sub_idx + metadata (no flat copies)
        print(f"[Chain] Pass 1: preprocessing {len(valid_groups):,} groups...",
              file=sys.stderr, flush=True)
        # Each entry: (group_i, sub_idx_into_group_arrays, rev, size)
        part_specs = []
        total_n = 0
        for i in tqdm(valid_groups, desc="Preprocessing groups", file=sys.stderr):
            start = int(ga.group_bounds[i])
            end = int(ga.group_bounds[i + 1])
            idx = ga.order[start:end]

            g_qi = ga.query_idx[idx]
            g_ti = ga.target_idx[idx]
            g_sim = ga.similarity[idx]
            g_orient = ga.orientation[idx]

            for sub_idx, sq, st, ss, cmp, rev in _preprocess_group_raw(
                g_qi, g_ti, g_sim, g_orient, min_size
            ):
                sz = len(sq)
                # Store sub_idx (indices into the group's local arrays)
                part_specs.append((i, sub_idx, rev, sz))
                total_n += sz

        n_parts = len(part_specs)
        if n_parts == 0:
            del ga, groups
            return {}

        print(f"[Chain] {n_parts:,} partitions, {total_n:,} total elements — "
              f"allocating flat arrays...", file=sys.stderr, flush=True)

        # Pre-allocate flat arrays
        sizes = np.empty(n_parts, dtype=np.int64)
        offsets = np.empty(n_parts, dtype=np.int64)
        flat_sq = np.empty(total_n, dtype=np.int64)
        flat_st = np.empty(total_n, dtype=np.int64)
        flat_ss = np.empty(total_n, dtype=np.float64)
        flat_cmp = np.empty(total_n, dtype=np.int64)

        partitions = []  # (key, q_gids, t_gids, reverse_target) for Phase 5

        # Pass 2: re-slice group arrays using stored sub_idx, write to flat
        print(f"[Chain] Pass 2: filling flat arrays...",
              file=sys.stderr, flush=True)
        write_pos = 0
        for p_idx in tqdm(range(n_parts), desc="Packing partitions", file=sys.stderr):
            group_i, sub_idx, rev, sz = part_specs[p_idx]
            start = int(ga.group_bounds[group_i])
            end = int(ga.group_bounds[group_i + 1])
            idx = ga.order[start:end]

            # Reconstruct group key
            row0 = idx[0]
            key = (
                ga.qg_uniques[ga.qg_codes[row0]],
                ga.tg_uniques[ga.tg_codes[row0]],
                ga.qc_uniques[ga.qc_codes[row0]],
                ga.tc_uniques[ga.tc_codes[row0]],
            )

            # Slice raw arrays and apply sub_idx
            g_qi = ga.query_idx[idx]
            g_ti = ga.target_idx[idx]
            g_sim = ga.similarity[idx]
            g_qgid = ga.query_gene_id[idx]
            g_tgid = ga.target_gene_id[idx]

            sq = g_qi[sub_idx]
            st = g_ti[sub_idx]
            ss = g_sim[sub_idx]
            cmp = -st if rev else st.copy()

            flat_sq[write_pos:write_pos+sz] = sq
            flat_st[write_pos:write_pos+sz] = st
            flat_ss[write_pos:write_pos+sz] = ss
            flat_cmp[write_pos:write_pos+sz] = cmp
            sizes[p_idx] = sz
            offsets[p_idx] = write_pos
            partitions.append((key, g_qgid[sub_idx], g_tgid[sub_idx], rev))
            write_pos += sz

        del ga, groups, part_specs
        import gc; gc.collect()

    else:
        # --- Legacy path: Dict of DataFrames ---
        partitions = []
        sq_parts = []
        st_parts = []
        ss_parts = []
        cmp_parts = []
        part_sizes = []

        group_keys = list(groups.keys())
        for key in tqdm(group_keys, desc="Preprocessing groups", file=sys.stderr):
            anchors_df = groups.pop(key)
            q_gene_ids = anchors_df["query_gene_id"].values
            t_gene_ids = anchors_df["target_gene_id"].values

            for df_idx, sq, st, ss, cmp, rev in _preprocess_group_arrays(anchors_df, min_size):
                idx = df_idx.astype(np.intp)
                partitions.append((key, q_gene_ids[idx], t_gene_ids[idx], rev))
                sq_parts.append(np.ascontiguousarray(sq, dtype=np.int64))
                st_parts.append(np.ascontiguousarray(st, dtype=np.int64))
                ss_parts.append(np.ascontiguousarray(ss, dtype=np.float64))
                cmp_parts.append(np.ascontiguousarray(cmp, dtype=np.int64))
                part_sizes.append(len(sq))

            del anchors_df

        if not partitions:
            return {}

        n_parts = len(partitions)
        sizes = np.array(part_sizes, dtype=np.int64)
        offsets = np.zeros(n_parts, dtype=np.int64)
        if n_parts > 1:
            np.cumsum(sizes[:-1], out=offsets[1:])
        total_n = int(sizes.sum())

        flat_sq = np.empty(total_n, dtype=np.int64)
        flat_st = np.empty(total_n, dtype=np.int64)
        flat_ss = np.empty(total_n, dtype=np.float64)
        flat_cmp = np.empty(total_n, dtype=np.int64)

        for i in range(n_parts):
            o = int(offsets[i])
            n = int(sizes[i])
            flat_sq[o:o+n] = sq_parts[i]
            flat_st[o:o+n] = st_parts[i]
            flat_ss[o:o+n] = ss_parts[i]
            flat_cmp[o:o+n] = cmp_parts[i]

        del sq_parts, st_parts, ss_parts, cmp_parts, part_sizes

    # Phase 3: Batched DP fill — process groups in chunks for progress tracking.
    # Groups are independent so we can call the Numba kernel on sub-slices of
    # offsets/sizes while sharing the same flat arrays.
    dp_len = np.ones(total_n, dtype=np.int32)
    dp_prev = np.full(total_n, -1, dtype=np.int64)
    dp_score = flat_ss.copy()

    _max_gap = np.int64(max_gap if max_gap is not None else -1)
    _dp_func = _dp_fill_batched_parallel if total_n >= 100_000 else _dp_fill_batched

    DP_BATCH = 5_000   # groups per progress tick
    for b_start in tqdm(range(0, n_parts, DP_BATCH), desc="Chaining DP",
                        total=(n_parts + DP_BATCH - 1) // DP_BATCH, file=sys.stderr):
        b_end = min(b_start + DP_BATCH, n_parts)
        _dp_func(flat_sq, flat_st, flat_ss, flat_cmp,
                 offsets[b_start:b_end], sizes[b_start:b_end],
                 _max_gap, float(gap_penalty_scale),
                 dp_len, dp_prev, dp_score)

    # Free arrays no longer needed after DP
    del flat_cmp

    # Phase 4: Batched backtrack (single Numba call)
    chain_buf = np.empty(total_n, dtype=np.int64)
    chain_group_buf = np.empty(total_n, dtype=np.int64)
    chain_len_buf = np.empty(total_n, dtype=np.int64)

    n_chains, n_elements = _backtrack_batched(
        dp_len, dp_prev, dp_score,
        offsets, sizes, np.int32(min_size),
        chain_buf, chain_group_buf, chain_len_buf,
    )

    # Free DP arrays
    del dp_len, dp_prev, dp_score

    # Phase 5: Unpack — build chain DataFrames from flat arrays + gene IDs
    result = {}
    pos = 0
    for c in range(n_chains):
        g = int(chain_group_buf[c])
        cl = int(chain_len_buf[c])
        local_idx = chain_buf[pos:pos+cl].astype(np.intp)
        pos += cl

        key, q_gids, t_gids, rev = partitions[g]
        qg, tg, qc, tc = key
        off = int(offsets[g])
        glob_idx = off + local_idx
        orientation = -1 if rev else 1

        chain_df = pd.DataFrame({
            "query_idx": flat_sq[glob_idx],
            "target_idx": flat_st[glob_idx],
            "query_genome": qg,
            "target_genome": tg,
            "query_contig": qc,
            "target_contig": tc,
            "query_gene_id": q_gids[local_idx],
            "target_gene_id": t_gids[local_idx],
            "similarity": flat_ss[glob_idx],
            "orientation": orientation,
        })
        result.setdefault(key, []).append(chain_df)

    return result
