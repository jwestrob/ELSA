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
from typing import List, Dict

import os
import numpy as np
import pandas as pd
from ._log import tlog as _log

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
                if sq[i] <= sq[j]:
                    continue
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
                    if sq[ii] <= sq[jj]:
                        continue
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
                    if sq[ii] <= sq[jj]:
                        continue
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
    def _backtrack_ordered(dp_len, dp_prev, min_size, order):
        """Traverse endpoints in the supplied stable longest-first order."""
        n = len(dp_len)
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
    def _backtrack_batched_ordered(dp_len, dp_prev, offsets, sizes, min_size,
                                   chain_buf, chain_group_buf, chain_len_buf,
                                   endpoint_order):
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

            used = np.zeros(n, dtype=np.bool_)

            for idx in range(n):
                i_global = endpoint_order[off + idx]
                i_local = i_global - off
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

    @njit(cache=True, parallel=True)
    def _select_paired_overlap_numba(order, group_starts, group_ends,
                                     qmin, qmax, tmin, tmax):
        """Greedy rectangle selection, parallel over complete contig pairs.

        A static tree of query starts bounds the accepted rectangles beneath
        each node. Bounds prune the search; only an accepted leaf can reject
        a candidate, so both overlaps must belong to the same accepted block.
        """
        keep = np.zeros(len(order), dtype=np.bool_)
        maximum = np.iinfo(np.int64).max
        minimum = np.iinfo(np.int64).min
        for group in prange(len(group_starts)):
            start, end = group_starts[group], group_ends[group]
            ids = order[start:end]
            size = len(ids)
            if size == 0:
                continue
            coordinate_order = np.argsort(qmin[ids])
            ranks = np.empty(size, dtype=np.int64)
            leaves, height = 1, 0
            while leaves < size:
                leaves *= 2
                height += 1
            min_query = np.full(2 * leaves, maximum, dtype=np.int64)
            max_query = np.full(2 * leaves, minimum, dtype=np.int64)
            min_target = np.full(2 * leaves, maximum, dtype=np.int64)
            max_target = np.full(2 * leaves, minimum, dtype=np.int64)
            for rank in range(size):
                local = coordinate_order[rank]
                ranks[local] = rank
                min_query[leaves + rank] = qmin[ids[local]]
            for node in range(leaves - 1, 0, -1):
                min_query[node] = min(min_query[2 * node], min_query[2 * node + 1])
            stack = np.empty(height + 2, dtype=np.int64)
            for local in range(size):
                candidate = ids[local]
                stack[0], count, conflict = 1, 1, False
                while count:
                    count -= 1
                    node = stack[count]
                    if (min_query[node] > qmax[candidate]
                            or max_query[node] < qmin[candidate]
                            or min_target[node] > tmax[candidate]
                            or max_target[node] < tmin[candidate]):
                        continue
                    if node >= leaves:
                        conflict = True
                        break
                    stack[count], stack[count + 1] = 2 * node, 2 * node + 1
                    count += 2
                if conflict:
                    continue
                keep[candidate] = True
                node = leaves + ranks[local]
                max_query[node] = qmax[candidate]
                min_target[node], max_target[node] = tmin[candidate], tmax[candidate]
                node //= 2
                while node:
                    max_query[node] = max(max_query[2 * node], max_query[2 * node + 1])
                    min_target[node] = min(min_target[2 * node], min_target[2 * node + 1])
                    max_target[node] = max(max_target[2 * node], max_target[2 * node + 1])
                    node //= 2
        return keep

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def _backtrack(dp_len, dp_prev, dp_score, min_size):
    """Stable O(n log n) endpoint sort followed by compiled traversal."""
    order = np.lexsort((-dp_score, -dp_len))
    return _backtrack_ordered(dp_len, dp_prev, min_size, order)


def _backtrack_batched(dp_len, dp_prev, dp_score, offsets, sizes, min_size,
                       chain_buf, chain_group_buf, chain_len_buf):
    """Sort endpoints by partition, decreasing length and score, then traverse.

    NumPy's stable lexsort retains original endpoint order for exact ties.
    Traversal and claiming of truncated paths match the single-group path.
    """
    partition = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    order = np.lexsort((-dp_score, -dp_len, partition))
    return _backtrack_batched_ordered(
        dp_len, dp_prev, offsets, sizes, min_size,
        chain_buf, chain_group_buf, chain_len_buf, order)


def _select_paired_overlap(order, group_starts, group_ends,
                           qmin, qmax, tmin, tmax):
    """Keep candidates unless one accepted block overlaps both inclusive axes."""
    if _NUMBA_AVAILABLE:
        return _select_paired_overlap_numba(
            order, group_starts, group_ends, qmin, qmax, tmin, tmax)
    keep = np.zeros(len(order), dtype=np.bool_)
    for start, end in zip(group_starts, group_ends):
        accepted = []
        for candidate in order[start:end]:
            if any(qmin[candidate] <= qmax[previous]
                   and qmax[candidate] >= qmin[previous]
                   and tmin[candidate] <= tmax[previous]
                   and tmax[candidate] >= tmin[previous]
                   for previous in accepted):
                continue
            keep[candidate] = True
            accepted.append(candidate)
    return keep


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

    Accepts a DataFrame of anchors for a single contig pair. Exact duplicate
    pairs keep their strongest similarity; alternative targets remain
    available to the DP. Each chain progresses strictly on both gene axes.
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

    all_chains = []
    for sub_df, reverse_target in _preprocess_group(anchors_df, min_size):
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
                    if sq_c[i] <= sq_c[j]:
                        continue
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
    Extract blocks using greedy paired-span overlap selection.

    Within a contig pair, a candidate is suppressed when one accepted block
    overlaps both its query span and its target span, including boundaries.

    Args:
        chains: List of anchor DataFrames from chain_anchors_lis
        block_id_start: Starting block ID for numbering

    Returns:
        List of ChainedBlock objects
    """
    if not chains:
        return []

    blocks_df = extract_nonoverlapping_chains_df(chains, block_id_start)
    if blocks_df.empty:
        return []

    # Convert to ChainedBlock objects for backward compatibility
    import json as _json
    blocks = []
    for _, row in blocks_df.iterrows():
        blocks.append(ChainedBlock(
            block_id=int(row["block_id"]),
            query_genome=row["query_genome"],
            target_genome=row["target_genome"],
            query_contig=row["query_contig"],
            target_contig=row["target_contig"],
            query_start=int(row["query_start"]),
            query_end=int(row["query_end"]),
            target_start=int(row["target_start"]),
            target_end=int(row["target_end"]),
            anchor_query_ids=_json.loads(row["anchor_query_ids"]),
            anchor_target_ids=_json.loads(row["anchor_target_ids"]),
            anchor_query_gene_ids=_json.loads(row["anchor_query_gene_ids"]),
            anchor_target_gene_ids=_json.loads(row["anchor_target_gene_ids"]),
            n_anchors=int(row["n_anchors"]),
            chain_score=float(row["chain_score"]),
            orientation=int(row["orientation"]),
        ))
    return blocks


def extract_nonoverlapping_chains_df(
    chains: List[pd.DataFrame],
    block_id_start: int = 0,
) -> pd.DataFrame:
    """
    Extract blocks with paired-span overlap selection as a DataFrame.

    Same greedy algorithm as extract_nonoverlapping_chains but returns
    a DataFrame directly, avoiding 2M+ ChainedBlock object creations.
    """
    import json as _json

    if not chains:
        return pd.DataFrame()

    # Score each chain
    scored = []
    for chain_df in chains:
        if chain_df.empty:
            continue
        q = chain_df["query_idx"].values
        t = chain_df["target_idx"].values
        s = chain_df["similarity"].values

        # Match the flat-array extractor's float64 accumulation and ranking.
        score = np.add.reduceat(np.asarray(s, dtype=np.float64), [0])[0]
        q_min, q_max = int(q.min()), int(q.max())
        t_min, t_max = int(t.min()), int(t.max())
        orientation = int(chain_df["orientation"].iloc[0])

        scored.append((score, chain_df, q_min, q_max, t_min, t_max, orientation))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        return pd.DataFrame()

    # Complete contig-pair groups compete across both orientations. Grouping
    # for selection preserves the legacy global score order of emitted rows.
    key_codes = {}
    group_codes = np.empty(len(scored), dtype=np.int64)
    for i, (_, chain_df, *_) in enumerate(scored):
        key = tuple(str(chain_df[column].iloc[0]) for column in
                    ("query_genome", "target_genome", "query_contig", "target_contig"))
        if key not in key_codes:
            key_codes[key] = len(key_codes)
        group_codes[i] = key_codes[key]
    order = np.argsort(group_codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(group_codes[order])) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(scored)].astype(np.int64)
    bounds = np.asarray([item[2:6] for item in scored], dtype=np.int64)
    keep = _select_paired_overlap(order, starts, ends, *bounds.T)

    # Pre-allocate lists for columnar output
    out_bid = []
    out_qg = []; out_tg = []; out_qc = []; out_tc = []
    out_qs = []; out_qe = []; out_ts = []; out_te = []
    out_na = []; out_score = []; out_orient = []
    out_aqids = []; out_atids = []; out_aqgids = []; out_atgids = []

    block_id = block_id_start
    for i, (score, chain_df, q_min, q_max, t_min, t_max, orientation) in enumerate(scored):
        if not keep[i]:
            continue
        q_key = (str(chain_df["query_genome"].iloc[0]),
                 str(chain_df["query_contig"].iloc[0]))
        t_key = (str(chain_df["target_genome"].iloc[0]),
                 str(chain_df["target_contig"].iloc[0]))

        qi = chain_df["query_idx"].values
        ti = chain_df["target_idx"].values
        q_sort = np.argsort(qi)
        t_sort = np.argsort(ti)

        out_bid.append(block_id)
        out_qg.append(q_key[0]); out_tg.append(t_key[0])
        out_qc.append(q_key[1]); out_tc.append(t_key[1])
        out_qs.append(q_min); out_qe.append(q_max)
        out_ts.append(t_min); out_te.append(t_max)
        out_na.append(len(chain_df))
        out_score.append(round(score, 4))
        out_orient.append(orientation)
        out_aqids.append(_json.dumps(qi[q_sort].tolist()))
        out_atids.append(_json.dumps(ti[t_sort].tolist()))
        out_aqgids.append(_json.dumps(chain_df["query_gene_id"].values[q_sort].tolist()))
        out_atgids.append(_json.dumps(chain_df["target_gene_id"].values[t_sort].tolist()))
        block_id += 1

    if not out_bid:
        return pd.DataFrame()

    return pd.DataFrame({
        "block_id": out_bid,
        "query_genome": out_qg, "target_genome": out_tg,
        "query_contig": out_qc, "target_contig": out_tc,
        "query_start": out_qs, "query_end": out_qe,
        "target_start": out_ts, "target_end": out_te,
        "n_anchors": out_na, "chain_score": out_score,
        "orientation": out_orient,
        "anchor_query_ids": out_aqids, "anchor_target_ids": out_atids,
        "anchor_query_gene_ids": out_aqgids, "anchor_target_gene_ids": out_atgids,
    })


# ---------------------------------------------------------------------------
# Preprocessing helper (shared logic for per-group and batched paths)
# ---------------------------------------------------------------------------
def _preprocess_group(anchors_df, min_size):
    """Preprocess a contig-pair group for DP chaining.

    Yields (sub_df, reverse_target) tuples — deduplicated, partitioned
    by orientation, and sorted by query position.
    """
    for indices, _, _, _, _, reverse_target in _preprocess_group_arrays(anchors_df, min_size):
        yield anchors_df.iloc[indices].reset_index(drop=True), reverse_target


# ---------------------------------------------------------------------------
# Array-only preprocessing (avoids intermediate DataFrames)
# ---------------------------------------------------------------------------
def _preprocess_group_raw(q_idx, t_idx, sims, orients, min_size):
    """Preprocess a group from raw numpy arrays (no DataFrame).

    Yields (sub_idx, sq, st, ss, cmp, reverse_target) where sub_idx
    are indices into the input arrays.
    """
    n = len(q_idx)
    if n == 0 or n < min_size:
        return

    # Keep the strongest exact pair, retaining alternative targets per query.
    # Stable ties retain input row order; separate keys avoid integer packing.
    order = np.lexsort((-sims, t_idx, q_idx))
    first = np.empty(n, dtype=np.bool_)
    first[0] = True
    first[1:] = ((q_idx[order[1:]] != q_idx[order[:-1]])
                 | (t_idx[order[1:]] != t_idx[order[:-1]]))
    dedup_idx = order[first]

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
        sort_order = np.lexsort((t_idx[part_idx], q_idx[part_idx]))
        sub_idx = part_idx[sort_order]

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
    yield from _preprocess_group_raw(
        anchors_df["query_idx"].values, anchors_df["target_idx"].values,
        anchors_df["similarity"].values, anchors_df["orientation"].values,
        min_size)


# ---------------------------------------------------------------------------
# Vectorized preprocessing — replaces per-group Python loop with bulk numpy
# ---------------------------------------------------------------------------

def _preprocess_vectorized(ga, valid_groups, min_size):
    """Vectorized preprocessing of all valid groups at once.

    Replaces 12M+ Python loop iterations with ~5 vectorized numpy
    operations on the full anchor arrays.  Produces the same flat
    arrays + partition metadata as the per-group loop but 20-30x faster.

    Returns None if no valid partitions, otherwise a dict with:
        flat_sq, flat_st, flat_ss, flat_cmp, flat_fidx,
        sizes, offsets, n_parts, part_gids, part_revs, total_n
    """
    n_valid = len(valid_groups)
    if n_valid == 0:
        return None

    group_sizes = np.diff(ga.group_bounds)
    valid_sizes = group_sizes[valid_groups].astype(np.int64)
    total_valid = int(valid_sizes.sum())

    _log(f"[Chain] Extracting {total_valid:,} anchors from "
          f"{n_valid:,} valid groups...")

    # --- Step 1: Extract valid-group anchors into contiguous arrays ---
    # Fully vectorized: no Python loop
    starts = ga.group_bounds[valid_groups].astype(np.int64)
    cum_sizes = np.empty(n_valid, dtype=np.int64)
    cum_sizes[0] = 0
    np.cumsum(valid_sizes[:-1], out=cum_sizes[1:])
    group_start_rep = np.repeat(starts, valid_sizes)
    local_offset = np.arange(total_valid, dtype=np.int64) - np.repeat(cum_sizes, valid_sizes)
    positions = group_start_rep + local_offset
    del group_start_rep, local_offset, cum_sizes

    flat_idx = ga.order[positions]
    flat_gid = np.repeat(np.arange(n_valid, dtype=np.int64), valid_sizes)
    del positions

    qi = ga.query_idx[flat_idx]
    ti = ga.target_idx[flat_idx]
    sim = ga.similarity[flat_idx]
    orient = ga.orientation[flat_idx]

    # --- Step 2: Dedup by exact (group, qi, ti), keep highest sim ---
    _log(f"[Chain] Dedup by (group, gene-pair)...")
    sort1 = np.lexsort((-sim, ti, qi, flat_gid))
    s1_gid = flat_gid[sort1]
    s1_qi, s1_ti = qi[sort1], ti[sort1]
    first1 = np.empty(total_valid, dtype=np.bool_)
    first1[0] = True
    first1[1:] = ((s1_gid[1:] != s1_gid[:-1])
                  | (s1_qi[1:] != s1_qi[:-1]) | (s1_ti[1:] != s1_ti[:-1]))
    dedup = sort1[first1]
    del sort1, s1_gid, s1_qi, s1_ti, first1

    d_gid = flat_gid[dedup]
    d_qi = qi[dedup]
    d_ti = ti[dedup]
    d_sim = sim[dedup]
    d_orient = orient[dedup]
    d_fidx = flat_idx[dedup]
    n_dedup = len(dedup)
    del qi, ti, sim, orient, flat_idx, flat_gid, dedup
    _log(f"[Chain] {n_dedup:,} anchors after pair dedup")

    # --- Step 3: Per-group strand check ---
    group_has_strand = np.zeros(n_valid, dtype=np.bool_)
    nz = d_orient != 0
    if nz.any():
        np.maximum.at(group_has_strand, d_gid[nz], True)
    per_row_hs = group_has_strand[d_gid]
    del nz, group_has_strand

    # --- Step 4: Fwd/rev stream masks ---
    # fwd: orient >= 0 (includes orient==0 from no-strand and has-strand groups)
    # rev: orient < 0, OR all rows from no-strand groups (orient==0 AND no strand)
    in_fwd = d_orient >= 0
    in_rev = (d_orient < 0) | (~per_row_hs)
    del per_row_hs

    # --- Step 5: Process each stream ---
    results = []
    for mask, is_rev in [(in_fwd, False), (in_rev, True)]:
        if not mask.any():
            continue

        s_gid = d_gid[mask]
        s_qi = d_qi[mask]
        s_ti = d_ti[mask]
        s_sim = d_sim[mask]
        s_fidx = d_fidx[mask]

        # Retain all exact pairs and make coordinate ties deterministic.
        keep = np.lexsort((s_ti, s_qi, s_gid))

        # Result is sorted by (gid, qi, ti).
        k_gid = s_gid[keep]
        k_qi = s_qi[keep]
        k_ti = s_ti[keep]
        k_sim = s_sim[keep]
        k_fidx = s_fidx[keep]
        del s_gid, s_qi, s_ti, s_sim, s_fidx, keep

        # Filter groups with < min_size surviving anchors
        gcounts = np.zeros(n_valid, dtype=np.int64)
        np.add.at(gcounts, k_gid, 1)
        valid_pg = gcounts >= min_size
        row_valid = valid_pg[k_gid]
        del gcounts, valid_pg

        k_gid = k_gid[row_valid]
        k_qi = k_qi[row_valid]
        k_ti = k_ti[row_valid]
        k_sim = k_sim[row_valid]
        k_fidx = k_fidx[row_valid]
        del row_valid

        if len(k_gid) == 0:
            continue

        k_cmp = (-k_ti if is_rev else k_ti).copy()

        results.append((k_qi, k_ti, k_sim.astype(np.float64), k_cmp,
                         k_gid, k_fidx, is_rev))

    del d_gid, d_qi, d_ti, d_sim, d_orient, d_fidx

    if not results:
        return None

    # --- Step 6: Concatenate streams and compute partition boundaries ---
    all_sq = np.concatenate([r[0] for r in results])
    all_st = np.concatenate([r[1] for r in results])
    all_ss = np.concatenate([r[2] for r in results])
    all_cmp = np.concatenate([r[3] for r in results])
    all_gid = np.concatenate([r[4] for r in results])
    all_fidx = np.concatenate([r[5] for r in results])
    all_is_rev = np.concatenate([
        np.full(len(r[0]), r[6], dtype=np.bool_) for r in results
    ])
    del results

    total_n = len(all_sq)

    # Partition boundaries: where (gid, is_rev) changes
    boundaries = np.empty(total_n, dtype=np.bool_)
    boundaries[0] = True
    boundaries[1:] = (all_gid[1:] != all_gid[:-1]) | (all_is_rev[1:] != all_is_rev[:-1])
    part_starts = np.nonzero(boundaries)[0]
    n_parts = len(part_starts)
    del boundaries

    sizes = np.empty(n_parts, dtype=np.int64)
    sizes[:-1] = part_starts[1:] - part_starts[:-1]
    sizes[-1] = total_n - part_starts[-1]
    offsets = part_starts.astype(np.int64)

    part_gids = all_gid[part_starts]
    part_revs = all_is_rev[part_starts]
    del all_gid, all_is_rev

    stream_label = "fwd+rev" if len([1 for r in [in_fwd, in_rev] if r.any()]) == 2 else "single"
    _log(f"[Chain] Vectorized preprocessing done: {n_parts:,} partitions, "
          f"{total_n:,} elements ({stream_label})")

    return {
        "flat_sq": all_sq, "flat_st": all_st,
        "flat_ss": all_ss, "flat_cmp": all_cmp,
        "flat_fidx": all_fidx,
        "sizes": sizes, "offsets": offsets,
        "n_parts": n_parts, "part_gids": part_gids,
        "part_revs": part_revs, "total_n": total_n,
    }


# ---------------------------------------------------------------------------
# Flat-array block extraction (replaces Phase 5 + pipeline extraction loop)
# ---------------------------------------------------------------------------

def _dict_result_to_blocks_df(result: Dict, block_id_start: int = 0) -> pd.DataFrame:
    """Convert legacy Dict[key -> chain list] to blocks DataFrame.

    Fallback for the no-Numba path when extract_blocks=True.
    """
    all_dfs = []
    bid = block_id_start
    for key, chains in result.items():
        bdf = extract_nonoverlapping_chains_df(chains, block_id_start=bid)
        if not bdf.empty:
            all_dfs.append(bdf)
            bid += len(bdf)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def _extract_blocks_flat(
    chain_buf, chain_group_buf, chain_len_buf, n_chains, n_elements,
    flat_sq, flat_st, flat_ss,
    offsets, part_gids, part_revs,
    *,
    flat_fidx=None,
    ga_query_gene_id=None, ga_target_gene_id=None,
    ga_order=None, ga_bounds=None,
    ga_qg_codes=None, ga_tg_codes=None,
    ga_qc_codes=None, ga_tc_codes=None,
    ga_qg_uniques=None, ga_tg_uniques=None,
    ga_qc_uniques=None, ga_tc_uniques=None,
    valid_groups=None,
    partitions=None,
    block_id_start=0,
) -> pd.DataFrame:
    """Extract blocks with paired-span selection from flat backtrack arrays.

    Replaces Phase 5 (per-chain DataFrame creation) and the pipeline's
    per-group extract_nonoverlapping_chains_df loop with a single vectorized
    pass.  Builds exactly ONE DataFrame at the end for selected blocks only.

    Overlap checking spans both orientations within each contig pair.
    """
    import json as _json
    import time as _time

    _log(f"[Chain] Phase 5: extracting blocks from {n_chains:,} chains (flat-array path)...")
    _t5 = _time.time()

    if n_chains == 0:
        _log("[Chain] Phase 5: no chains to extract")
        return pd.DataFrame()

    is_ga = partitions is None  # GroupedAnchors vs legacy path

    # --- Step 1: Cumulative chain offsets + global element indices ---
    chain_lens = chain_len_buf[:n_chains].copy()
    chain_cum = np.zeros(n_chains + 1, dtype=np.int64)
    np.cumsum(chain_lens, out=chain_cum[1:])

    chain_part = chain_group_buf[:n_chains].astype(np.intp)
    all_glob = (np.repeat(offsets[chain_part], chain_lens)
                + chain_buf[:n_elements]).astype(np.intp)

    # --- Step 2: Per-chain bounds and scores via reduceat ---
    cstarts = chain_cum[:n_chains].astype(np.intp)
    q_all = flat_sq[all_glob]
    t_all = flat_st[all_glob]
    s_all = flat_ss[all_glob]

    q_mins = np.minimum.reduceat(q_all, cstarts)
    q_maxs = np.maximum.reduceat(q_all, cstarts)
    t_mins = np.minimum.reduceat(t_all, cstarts)
    t_maxs = np.maximum.reduceat(t_all, cstarts)
    scores = np.add.reduceat(s_all, cstarts)  # sum(sim) == len * mean_sim
    del q_all, t_all, s_all

    # --- Step 3: Per-chain group key + orientation ---
    if is_ga:
        chain_gvi = part_gids[chain_part]
        chain_gi = valid_groups[chain_gvi]
        r0 = ga_order[ga_bounds[chain_gi].astype(np.intp)]
        chain_qg = np.asarray(ga_qg_uniques[ga_qg_codes[r0]])
        chain_tg = np.asarray(ga_tg_uniques[ga_tg_codes[r0]])
        chain_qc = np.asarray(ga_qc_uniques[ga_qc_codes[r0]])
        chain_tc = np.asarray(ga_tc_uniques[ga_tc_codes[r0]])
        chain_key_code = chain_gvi  # same group index = same key
    else:
        chain_qg = np.empty(n_chains, dtype=object)
        chain_tg = np.empty(n_chains, dtype=object)
        chain_qc = np.empty(n_chains, dtype=object)
        chain_tc = np.empty(n_chains, dtype=object)
        key_map: Dict[tuple, int] = {}
        chain_key_code = np.empty(n_chains, dtype=np.int64)
        for c in range(n_chains):
            key = partitions[int(chain_part[c])][0]
            chain_qg[c], chain_tg[c], chain_qc[c], chain_tc[c] = key
            if key not in key_map:
                key_map[key] = len(key_map)
            chain_key_code[c] = key_map[key]

    chain_orient = np.where(part_revs[chain_part], np.int32(-1), np.int32(1))

    # --- Step 4: Sort by (key_code, -score) → groups ranked by score ---
    sort_idx = np.lexsort((-scores, chain_key_code))
    sorted_codes = chain_key_code[sort_idx]

    # Group boundaries within sorted order
    diffs = np.empty(n_chains, dtype=np.bool_)
    diffs[0] = True
    diffs[1:] = sorted_codes[1:] != sorted_codes[:-1]
    grp_starts = np.nonzero(diffs)[0]
    n_key_groups = len(grp_starts)
    grp_ends = np.empty(n_key_groups, dtype=np.int64)
    grp_ends[:-1] = grp_starts[1:]
    grp_ends[-1] = n_chains

    # --- Step 5: Greedy paired-span selection per complete contig pair ---
    keep = _select_paired_overlap(
        sort_idx, grp_starts, grp_ends, q_mins, q_maxs, t_mins, t_maxs)
    sel = sort_idx[keep[sort_idx]]
    n_blocks = len(sel)
    if n_blocks == 0:
        _log("[Chain] Phase 5: no blocks after greedy selection")
        return pd.DataFrame()

    _log(f"[Chain] Phase 5: selected {n_blocks:,} blocks with paired-span selection "
         f"from {n_chains:,} chains in {n_key_groups:,} groups")

    # --- Step 6: Build output — gene ID JSON only for selected blocks ---
    block_ids = np.arange(block_id_start, block_id_start + n_blocks, dtype=np.int64)
    aqids = [None] * n_blocks
    atids = [None] * n_blocks
    aqgids = [None] * n_blocks
    atgids = [None] * n_blocks

    for i, c in enumerate(sel):
        se, ee = int(chain_cum[c]), int(chain_cum[c + 1])
        glob = all_glob[se:ee]
        qi = flat_sq[glob]
        ti = flat_st[glob]
        qs = np.argsort(qi)
        ts = np.argsort(ti)

        if is_ga:
            rows = flat_fidx[glob]
            qg = ga_query_gene_id[rows]
            tg = ga_target_gene_id[rows]
        else:
            g = int(chain_part[c])
            _, p_qg, p_tg, _ = partitions[g]
            local = chain_buf[se:ee].astype(np.intp)
            qg = p_qg[local]
            tg = p_tg[local]

        aqids[i] = _json.dumps(qi[qs].tolist())
        atids[i] = _json.dumps(ti[ts].tolist())
        aqgids[i] = _json.dumps(qg[qs].tolist())
        atgids[i] = _json.dumps(tg[ts].tolist())

    blocks_df = pd.DataFrame({
        "block_id": block_ids,
        "query_genome": chain_qg[sel], "target_genome": chain_tg[sel],
        "query_contig": chain_qc[sel], "target_contig": chain_tc[sel],
        "query_start": q_mins[sel].astype(int),
        "query_end": q_maxs[sel].astype(int),
        "target_start": t_mins[sel].astype(int),
        "target_end": t_maxs[sel].astype(int),
        "n_anchors": chain_lens[sel].astype(int),
        "chain_score": np.round(scores[sel], 4),
        "orientation": chain_orient[sel],
        "anchor_query_ids": aqids, "anchor_target_ids": atids,
        "anchor_query_gene_ids": aqgids, "anchor_target_gene_ids": atgids,
    })

    _log(f"[Chain] Phase 5: {n_blocks:,} blocks extracted in {_time.time() - _t5:.1f}s")
    return blocks_df


# ---------------------------------------------------------------------------
# Batched chaining — single Numba call for all contig-pair groups
# ---------------------------------------------------------------------------
def chain_groups_batched(
    groups,
    max_gap: int = 2,
    min_size: int = 2,
    gap_penalty_scale: float = 0.0,
    extract_blocks: bool = False,
    block_id_start: int = 0,
):
    """Chain all contig-pair groups using batched Numba kernels.

    Accepts either a GroupedAnchors namedtuple (fast path — raw arrays +
    group boundaries, no DataFrame construction) or a legacy Dict mapping
    (qg, tg, qc, tc) keys to anchor DataFrames.

    Falls back to per-group chain_anchors_lis if Numba is not available.

    Args:
        extract_blocks: If True, return a single pd.DataFrame of
            blocks after paired-span selection (skips per-chain DataFrame
            creation). If False, return Dict of chain lists (legacy).
        block_id_start: Starting block_id when extract_blocks=True.

    Returns:
        Dict mapping group keys to lists of chain DataFrames (default),
        or pd.DataFrame of blocks when extract_blocks=True.
    """
    # --- Detect input format ---
    is_grouped = hasattr(groups, 'order')

    if not _NUMBA_AVAILABLE:
        if is_grouped:
            # Materialize one contig pair at a time for the Python fallback.
            # The public grouped-array input remains usable with optional JIT.
            def iter_groups():
                for g in range(groups.n_groups):
                    rows = groups.order[groups.group_bounds[g]:groups.group_bounds[g + 1]]
                    if len(rows) < min_size or len(rows) == 0:
                        continue
                    row = rows[0]
                    key = (groups.qg_uniques[groups.qg_codes[row]],
                           groups.tg_uniques[groups.tg_codes[row]],
                           groups.qc_uniques[groups.qc_codes[row]],
                           groups.tc_uniques[groups.tc_codes[row]])
                    frame = pd.DataFrame({
                        "query_idx": groups.query_idx[rows],
                        "target_idx": groups.target_idx[rows],
                        "similarity": groups.similarity[rows],
                        "orientation": groups.orientation[rows],
                        "query_gene_id": groups.query_gene_id[rows],
                        "target_gene_id": groups.target_gene_id[rows],
                        "query_genome": key[0], "target_genome": key[1],
                        "query_contig": key[2], "target_contig": key[3],
                    })
                    yield key, frame
            group_items = iter_groups()
        else:
            group_items = groups.items()
        result = {}
        for key, anchors_df in group_items:
            chains = chain_anchors_lis(anchors_df, max_gap, min_size, gap_penalty_scale)
            if chains:
                result[key] = chains
        if extract_blocks:
            return _dict_result_to_blocks_df(result, block_id_start)
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
        # --- Fast path: vectorized preprocessing ---
        ga = groups

        group_sizes = np.diff(ga.group_bounds)
        valid_groups = np.where(group_sizes >= min_size)[0]
        n_skipped = ga.n_groups - len(valid_groups)
        if n_skipped > 0:
            _log(f"[Chain] Skipping {n_skipped:,} / {ga.n_groups:,} groups "
                  f"with < {min_size} anchors")

        vp = _preprocess_vectorized(ga, valid_groups, min_size)
        if vp is None:
            del ga, groups
            return pd.DataFrame() if extract_blocks else {}

        flat_sq = vp["flat_sq"]
        flat_st = vp["flat_st"]
        flat_ss = vp["flat_ss"]
        flat_cmp = vp["flat_cmp"]
        flat_fidx = vp["flat_fidx"]
        sizes = vp["sizes"]
        offsets = vp["offsets"]
        n_parts = vp["n_parts"]
        total_n = vp["total_n"]
        part_gids = vp["part_gids"]
        part_revs = vp["part_revs"]
        del vp

        # Keep only what Phase 5 needs from GA
        _ga_query_gene_id = ga.query_gene_id
        _ga_target_gene_id = ga.target_gene_id
        _ga_order = ga.order
        _ga_bounds = ga.group_bounds
        _ga_qg_codes = ga.qg_codes; _ga_tg_codes = ga.tg_codes
        _ga_qc_codes = ga.qc_codes; _ga_tc_codes = ga.tc_codes
        _ga_qg_uniques = ga.qg_uniques; _ga_tg_uniques = ga.tg_uniques
        _ga_qc_uniques = ga.qc_uniques; _ga_tc_uniques = ga.tc_uniques
        _valid_groups = valid_groups

        partitions = None  # signal vectorized unpack path

        del ga, groups
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
            return pd.DataFrame() if extract_blocks else {}

        n_parts = len(partitions)
        # Build part_revs from partitions for extract_blocks path
        part_revs = np.array([p[3] for p in partitions], dtype=np.bool_)
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
    import time as _time
    _log(f"[Chain] Phase 4: backtracking {n_parts:,} partitions...")
    _t4 = _time.time()

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
    _log(f"[Chain] Phase 4: {n_chains:,} chains ({n_elements:,} elements) in {_time.time() - _t4:.1f}s")

    if extract_blocks:
        # Phase 5: Flat-array block extraction — no per-chain DataFrames.
        # Scores, bounds, and greedy paired-span selection all operate
        # on flat numpy arrays.  One DataFrame is built at the very end for
        # selected blocks only.
        blocks_df = _extract_blocks_flat(
            chain_buf, chain_group_buf, chain_len_buf, n_chains, n_elements,
            flat_sq, flat_st, flat_ss,
            offsets, part_gids if partitions is None else None, part_revs,
            # GA metadata (None for legacy path)
            flat_fidx=flat_fidx if partitions is None else None,
            ga_query_gene_id=_ga_query_gene_id if partitions is None else None,
            ga_target_gene_id=_ga_target_gene_id if partitions is None else None,
            ga_order=_ga_order if partitions is None else None,
            ga_bounds=_ga_bounds if partitions is None else None,
            ga_qg_codes=_ga_qg_codes if partitions is None else None,
            ga_tg_codes=_ga_tg_codes if partitions is None else None,
            ga_qc_codes=_ga_qc_codes if partitions is None else None,
            ga_tc_codes=_ga_tc_codes if partitions is None else None,
            ga_qg_uniques=_ga_qg_uniques if partitions is None else None,
            ga_tg_uniques=_ga_tg_uniques if partitions is None else None,
            ga_qc_uniques=_ga_qc_uniques if partitions is None else None,
            ga_tc_uniques=_ga_tc_uniques if partitions is None else None,
            valid_groups=_valid_groups if partitions is None else None,
            partitions=partitions,
            block_id_start=block_id_start,
        )
        return blocks_df

    # Phase 5: Legacy unpack — build per-chain DataFrames
    _log(f"[Chain] Phase 5: unpacking {n_chains:,} chains into DataFrames...")
    _t5 = _time.time()
    result = {}
    pos = 0

    if partitions is None:
        # Vectorized path: reconstruct keys + gene_ids from GA metadata
        for c in range(n_chains):
            g = int(chain_group_buf[c])
            cl = int(chain_len_buf[c])
            local_idx = chain_buf[pos:pos+cl].astype(np.intp)
            pos += cl

            off = int(offsets[g])
            glob_idx = off + local_idx

            # Reconstruct group key
            gvi = int(part_gids[g])
            gi = _valid_groups[gvi]
            row0 = _ga_order[int(_ga_bounds[gi])]
            qg = _ga_qg_uniques[_ga_qg_codes[row0]]
            tg = _ga_tg_uniques[_ga_tg_codes[row0]]
            qc = _ga_qc_uniques[_ga_qc_codes[row0]]
            tc = _ga_tc_uniques[_ga_tc_codes[row0]]
            key = (qg, tg, qc, tc)

            rev = bool(part_revs[g])
            orientation = -1 if rev else 1

            # Gene IDs via GA row indices
            ga_rows = flat_fidx[glob_idx]
            q_gids = _ga_query_gene_id[ga_rows]
            t_gids = _ga_target_gene_id[ga_rows]

            chain_df = pd.DataFrame({
                "query_idx": flat_sq[glob_idx],
                "target_idx": flat_st[glob_idx],
                "query_genome": qg,
                "target_genome": tg,
                "query_contig": qc,
                "target_contig": tc,
                "query_gene_id": q_gids,
                "target_gene_id": t_gids,
                "similarity": flat_ss[glob_idx],
                "orientation": orientation,
            })
            result.setdefault(key, []).append(chain_df)
    else:
        # Legacy path: partitions list
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

    _log(f"[Chain] Phase 5: unpacked into {len(result):,} groups in {_time.time() - _t5:.1f}s")
    return result
