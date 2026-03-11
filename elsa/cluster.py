"""
Overlap-based clustering of syntenic blocks.

Groups blocks that share genomic regions (gene overlap) using
contig-indexed interval sweep, mutual top-k filtering, and union-find.

The interval sweep exploits the fact that blocks are intervals on specific
contigs — two blocks can only share genes if they touch the same
(genome, contig). This gives O(n log n + k) complexity vs O(n²) for
the naive pairwise approach.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .chain import ChainedBlock
from ._log import tlog as _log

# ---------------------------------------------------------------------------
# Numba-accelerated interval sweep kernel (optional)
# ---------------------------------------------------------------------------
try:
    from numba import njit

    @njit(cache=True)
    def _sweep_contig(starts, ends, block_indices):
        """Find all overlapping interval pairs on a single contig.

        Intervals are sorted by start position. For each interval i,
        scan forward while starts[j] <= ends[i] to find overlaps.

        Args:
            starts: int32[n] sorted start positions
            ends: int32[n] corresponding end positions
            block_indices: int32[n] block row index for each interval

        Returns:
            (pairs_i, pairs_j): int32 arrays of overlapping block pairs
        """
        n = len(starts)
        # Pre-allocate output (worst case: all pairs, but typically much less)
        max_pairs = min(n * 20, n * (n - 1) // 2)  # heuristic cap
        out_i = np.empty(max_pairs, dtype=np.int32)
        out_j = np.empty(max_pairs, dtype=np.int32)
        count = np.int64(0)

        for i in range(n):
            end_i = ends[i]
            bi = block_indices[i]
            for j in range(i + 1, n):
                if starts[j] > end_i:
                    break  # sorted → all subsequent j start even later
                bj = block_indices[j]
                if bi == bj:
                    continue  # skip self-pairs (same block indexed twice)
                if count >= max_pairs:
                    # Grow output arrays
                    new_max = max_pairs * 2
                    new_i = np.empty(new_max, dtype=np.int32)
                    new_j = np.empty(new_max, dtype=np.int32)
                    new_i[:max_pairs] = out_i
                    new_j[:max_pairs] = out_j
                    out_i = new_i
                    out_j = new_j
                    max_pairs = new_max
                out_i[count] = bi
                out_j[count] = bj
                count += 1

        return out_i[:count], out_j[:count]

    _NUMBA_SWEEP = True
except ImportError:
    _NUMBA_SWEEP = False


def _sweep_contig_python(starts, ends, block_indices):
    """Pure-Python fallback for the interval sweep."""
    n = len(starts)
    pairs_i = []
    pairs_j = []
    for i in range(n):
        end_i = ends[i]
        bi = block_indices[i]
        for j in range(i + 1, n):
            if starts[j] > end_i:
                break
            bj = block_indices[j]
            if bi != bj:
                pairs_i.append(bi)
                pairs_j.append(bj)
    if pairs_i:
        return np.array(pairs_i, dtype=np.int32), np.array(pairs_j, dtype=np.int32)
    return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)


_sweep_fn = _sweep_contig if _NUMBA_SWEEP else _sweep_contig_python


# ---------------------------------------------------------------------------
# Columnar conversion
# ---------------------------------------------------------------------------

def _blocks_to_columnar(blocks) -> dict:
    """Convert blocks to columnar numpy arrays + string encodings.

    Accepts either a List[ChainedBlock] or a pd.DataFrame with the standard
    block columns. The DataFrame path is pure numpy/pandas — no Python object
    iteration, no GIL pressure.
    """
    if isinstance(blocks, pd.DataFrame):
        return _blocks_df_to_columnar(blocks)
    return _blocks_list_to_columnar(blocks)


def _blocks_df_to_columnar(df: pd.DataFrame) -> dict:
    """Zero-copy columnar conversion from DataFrame. No Python-object loop."""
    n = len(df)

    block_ids = df['block_id'].values.astype(np.int32)
    q_start = df['query_start'].values.astype(np.int32)
    q_end = df['query_end'].values.astype(np.int32)
    t_start = df['target_start'].values.astype(np.int32)
    t_end = df['target_end'].values.astype(np.int32)
    n_anchors = df['n_anchors'].values.astype(np.int32)

    # Pandas Categorical: C-level factorization, shared codes across q/t
    q_genome_strs = df['query_genome'].values
    t_genome_strs = df['target_genome'].values
    q_contig_strs = df['query_contig'].values
    t_contig_strs = df['target_contig'].values

    all_genome_cat = pd.Categorical(np.concatenate([q_genome_strs, t_genome_strs]))
    all_contig_cat = pd.Categorical(np.concatenate([q_contig_strs, t_contig_strs]))
    all_genomes = list(all_genome_cat.categories)
    all_contigs = list(all_contig_cat.categories)

    q_genome = all_genome_cat.codes[:n].astype(np.int32)
    t_genome = all_genome_cat.codes[n:].astype(np.int32)
    q_contig = all_contig_cat.codes[:n].astype(np.int32)
    t_contig = all_contig_cat.codes[n:].astype(np.int32)

    n_contigs = len(all_contigs)
    q_contig_key = q_genome.astype(np.int64) * n_contigs + q_contig
    t_contig_key = t_genome.astype(np.int64) * n_contigs + t_contig

    block_gene_count = (q_end - q_start + 1) + (t_end - t_start + 1)

    return {
        'n': n,
        'block_ids': block_ids,
        'q_start': q_start, 'q_end': q_end,
        't_start': t_start, 't_end': t_end,
        'q_contig_key': q_contig_key, 't_contig_key': t_contig_key,
        'q_genome': q_genome, 't_genome': t_genome,
        'q_contig': q_contig, 't_contig': t_contig,
        'n_anchors': n_anchors,
        'block_gene_count': block_gene_count.astype(np.float64),
        'all_genomes': all_genomes, 'all_contigs': all_contigs,
        'q_genome_strs': q_genome_strs, 't_genome_strs': t_genome_strs,
        'q_contig_strs': q_contig_strs, 't_contig_strs': t_contig_strs,
    }


def _blocks_list_to_columnar(blocks: List[ChainedBlock]) -> dict:
    """Columnar conversion from list of ChainedBlock objects.

    Single pass over blocks + pandas Categorical for O(n) string encoding.
    """
    n = len(blocks)

    # Single pass: extract all fields at once
    block_ids = np.empty(n, dtype=np.int32)
    q_start = np.empty(n, dtype=np.int32)
    q_end = np.empty(n, dtype=np.int32)
    t_start = np.empty(n, dtype=np.int32)
    t_end = np.empty(n, dtype=np.int32)
    n_anchors_arr = np.empty(n, dtype=np.int32)
    q_genome_strs = [None] * n
    t_genome_strs = [None] * n
    q_contig_strs = [None] * n
    t_contig_strs = [None] * n

    for i, b in enumerate(blocks):
        block_ids[i] = b.block_id
        q_start[i] = b.query_start
        q_end[i] = b.query_end
        t_start[i] = b.target_start
        t_end[i] = b.target_end
        n_anchors_arr[i] = b.n_anchors
        q_genome_strs[i] = b.query_genome
        t_genome_strs[i] = b.target_genome
        q_contig_strs[i] = b.query_contig
        t_contig_strs[i] = b.target_contig

    # Pandas Categorical: C-level factorization for string→int codes
    all_genome_cat = pd.Categorical(q_genome_strs + t_genome_strs)
    all_contig_cat = pd.Categorical(q_contig_strs + t_contig_strs)
    all_genomes = list(all_genome_cat.categories)
    all_contigs = list(all_contig_cat.categories)

    q_genome = all_genome_cat.codes[:n].astype(np.int32)
    t_genome = all_genome_cat.codes[n:].astype(np.int32)
    q_contig = all_contig_cat.codes[:n].astype(np.int32)
    t_contig = all_contig_cat.codes[n:].astype(np.int32)

    n_contigs = len(all_contigs)
    q_contig_key = q_genome.astype(np.int64) * n_contigs + q_contig
    t_contig_key = t_genome.astype(np.int64) * n_contigs + t_contig

    block_gene_count = (q_end - q_start + 1) + (t_end - t_start + 1)

    return {
        'n': n,
        'block_ids': block_ids,
        'q_start': q_start, 'q_end': q_end,
        't_start': t_start, 't_end': t_end,
        'q_contig_key': q_contig_key, 't_contig_key': t_contig_key,
        'q_genome': q_genome, 't_genome': t_genome,
        'q_contig': q_contig, 't_contig': t_contig,
        'n_anchors': n_anchors_arr,
        'block_gene_count': block_gene_count.astype(np.float64),
        'all_genomes': all_genomes, 'all_contigs': all_contigs,
        'q_genome_strs': q_genome_strs, 't_genome_strs': t_genome_strs,
        'q_contig_strs': q_contig_strs, 't_contig_strs': t_contig_strs,
    }


# ---------------------------------------------------------------------------
# Interval-sweep clustering
# ---------------------------------------------------------------------------

def _build_contig_index(col):
    """Build contig-key → sorted interval list for sweep.

    Each block appears twice: once for its query side, once for target side.
    Returns dict mapping contig_key → (starts, ends, block_row_indices).
    """
    n = col['n']
    # Stack query-side and target-side half-edges
    contig_keys = np.concatenate([col['q_contig_key'], col['t_contig_key']])
    starts = np.concatenate([col['q_start'], col['t_start']])
    ends = np.concatenate([col['q_end'], col['t_end']])
    block_rows = np.concatenate([
        np.arange(n, dtype=np.int32),
        np.arange(n, dtype=np.int32),
    ])

    # Sort by contig_key, then by start within each key
    order = np.lexsort((starts, contig_keys))
    contig_keys = contig_keys[order]
    starts = starts[order]
    ends = ends[order]
    block_rows = block_rows[order]

    # Find group boundaries
    diffs = np.empty(len(contig_keys), dtype=np.bool_)
    diffs[0] = True
    np.not_equal(contig_keys[1:], contig_keys[:-1], out=diffs[1:])
    group_starts = np.nonzero(diffs)[0]

    bounds = np.empty(len(group_starts) + 1, dtype=np.int64)
    bounds[:len(group_starts)] = group_starts
    bounds[len(group_starts)] = len(contig_keys)

    groups = {}
    for i in range(len(group_starts)):
        s, e = int(bounds[i]), int(bounds[i + 1])
        if e - s < 2:
            continue  # single interval → no pairs possible
        key = int(contig_keys[s])
        groups[key] = (
            starts[s:e].astype(np.int32),
            ends[s:e].astype(np.int32),
            block_rows[s:e].astype(np.int32),
        )

    return groups


def _sweep_one_group(args):
    """Sweep a single contig group. Designed for thread pool dispatch."""
    starts, ends, block_rows = args
    return _sweep_fn(starts, ends, block_rows)


def _sweep_all_contigs(contig_groups, n_workers=None):
    """Run interval sweeps for all contig bins in parallel.

    Uses ThreadPoolExecutor — the Numba sweep kernel releases the GIL,
    so true parallelism is achieved without pickle overhead.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, max(1, len(contig_groups)))

    all_pairs_i = []
    all_pairs_j = []

    if n_workers <= 1 or len(contig_groups) <= 1:
        # Serial path
        for key, group in contig_groups.items():
            pi, pj = _sweep_fn(group[0], group[1], group[2])
            if len(pi) > 0:
                all_pairs_i.append(pi)
                all_pairs_j.append(pj)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_sweep_one_group, group): key
                for key, group in contig_groups.items()
            }
            for future in as_completed(futures):
                pi, pj = future.result()
                if len(pi) > 0:
                    all_pairs_i.append(pi)
                    all_pairs_j.append(pj)

    if all_pairs_i:
        return (np.concatenate(all_pairs_i), np.concatenate(all_pairs_j))
    return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))


def _compute_jaccard_vectorized(pairs_i, pairs_j, col):
    """Compute exact Jaccard for candidate pairs using interval arithmetic.

    For each candidate pair, checks all 4 contig combinations
    (query×query, query×target, target×query, target×target) for overlap.
    The intersection is the sum of gene-level overlaps on matching contigs.
    """
    if len(pairs_i) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    # All 4 side combinations: (side_i, side_j)
    # side = 'q' means query_{start,end,contig_key}, 't' means target_
    sides = [
        ('q', 'q'), ('q', 't'), ('t', 'q'), ('t', 't'),
    ]

    intersection = np.zeros(len(pairs_i), dtype=np.float64)

    for si, sj in sides:
        ck_i = col[f'{si}_contig_key'][pairs_i]
        ck_j = col[f'{sj}_contig_key'][pairs_j]
        same_contig = ck_i == ck_j

        if not same_contig.any():
            continue

        start_i = col[f'{si}_start'][pairs_i]
        end_i = col[f'{si}_end'][pairs_i]
        start_j = col[f'{sj}_start'][pairs_j]
        end_j = col[f'{sj}_end'][pairs_j]

        overlap = np.maximum(0,
            np.minimum(end_i, end_j) - np.maximum(start_i, start_j) + 1
        ).astype(np.float64)

        intersection += overlap * same_contig

    # Jaccard = intersection / union, union = |A| + |B| - intersection
    size_i = col['block_gene_count'][pairs_i]
    size_j = col['block_gene_count'][pairs_j]
    union = size_i + size_j - intersection

    # Avoid division by zero
    valid = union > 0
    jaccard = np.zeros(len(pairs_i), dtype=np.float64)
    jaccard[valid] = intersection[valid] / union[valid]

    return pairs_i, pairs_j, jaccard


def cluster_blocks_by_overlap(
    blocks: Union[List[ChainedBlock], pd.DataFrame],
    jaccard_tau: float = 0.3,
    mutual_k: int = 5,
    min_genome_support: int = 2,
) -> Tuple[Dict[int, int], pd.DataFrame]:
    """
    Cluster blocks based on shared genomic regions (gene overlap).

    Uses contig-indexed interval sweep to find candidate overlapping pairs
    in O(n log n + k) time, then vectorized Jaccard, mutual top-k filtering,
    and union-find for connected components.

    Args:
        blocks: List of ChainedBlock objects or DataFrame with block columns
        jaccard_tau: Minimum Jaccard similarity for overlap edges
        mutual_k: Mutual top-k parameter for edge filtering
        min_genome_support: Minimum genomes per cluster

    Returns:
        block_to_cluster: mapping from block_id to cluster_id
        clusters_df: DataFrame with cluster metadata
    """
    if isinstance(blocks, pd.DataFrame):
        n_blocks = len(blocks)
    else:
        n_blocks = len(blocks)
    if n_blocks == 0:
        return {}, pd.DataFrame(columns=["cluster_id", "size", "genome_support",
                                          "mean_chain_length", "genes_json"])

    # --- Phase 1: Convert to columnar arrays ---
    _log(f"[Cluster] Converting {n_blocks:,} blocks to columnar format...")
    col = _blocks_to_columnar(blocks)

    # --- Phase 2: Build contig index ---
    contig_groups = _build_contig_index(col)
    _log(f"[Cluster] {len(contig_groups):,} contig bins for interval sweep")

    # --- Phase 3: Parallel interval sweep ---
    n_workers = min(os.cpu_count() or 4, len(contig_groups))
    _log(f"[Cluster] Running interval sweep ({n_workers} threads)...")
    raw_pairs_i, raw_pairs_j = _sweep_all_contigs(contig_groups, n_workers)
    del contig_groups

    _log(f"[Cluster] {len(raw_pairs_i):,} raw candidate pairs from sweep")

    # Deduplicate: encode (min, max) as single int64 for fast 1D unique
    if len(raw_pairs_i) > 0:
        _log(f"[Cluster] Deduplicating {len(raw_pairs_i):,} pairs...")
        lo = np.minimum(raw_pairs_i, raw_pairs_j).astype(np.int64)
        hi = np.maximum(raw_pairs_i, raw_pairs_j).astype(np.int64)
        # Composite key: lo * n_blocks + hi (fits in int64 for n < 2^31)
        composite = lo * np.int64(n_blocks) + hi
        del lo, hi, raw_pairs_i, raw_pairs_j
        composite = np.unique(composite)
        pairs_i = (composite // np.int64(n_blocks)).astype(np.int32)
        pairs_j = (composite % np.int64(n_blocks)).astype(np.int32)
        del composite
        _log(f"[Cluster] {len(pairs_i):,} unique candidate pairs after dedup")
    else:
        pairs_i = raw_pairs_i
        pairs_j = raw_pairs_j

    # --- Phase 4: Vectorized Jaccard ---
    _log(f"[Cluster] Computing Jaccard for {len(pairs_i):,} candidate pairs...")
    _, _, jaccard = _compute_jaccard_vectorized(pairs_i, pairs_j, col)

    mask = jaccard >= jaccard_tau
    n_edges = int(mask.sum())
    _log(f"[Cluster] {n_edges:,} overlap edges above Jaccard threshold {jaccard_tau}")

    # --- Phase 5: Build edges dict for mutual top-k ---
    block_ids_arr = col['block_ids']
    edges_by_u: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    valid_i = pairs_i[mask]
    valid_j = pairs_j[mask]
    valid_jac = jaccard[mask]
    del pairs_i, pairs_j, jaccard, mask

    t0 = time.time()
    _log(f"[Cluster] Building edge dict from {n_edges:,} edges...")

    # Vectorized edge building: map row indices → block_ids once
    edge_bids_i = block_ids_arr[valid_i].astype(int)
    edge_bids_j = block_ids_arr[valid_j].astype(int)
    edge_jacs = valid_jac.astype(float)
    del valid_i, valid_j, valid_jac

    for idx in range(n_edges):
        bi = int(edge_bids_i[idx])
        bj = int(edge_bids_j[idx])
        jac = edge_jacs[idx]
        edges_by_u[bi].append((bj, jac))
        edges_by_u[bj].append((bi, jac))
    del edge_bids_i, edge_bids_j, edge_jacs

    _log(f"[Cluster] Edge dict built in {time.time() - t0:.1f}s ({len(edges_by_u):,} nodes)")

    # Apply mutual top-k filter
    t0 = time.time()
    mutual_edges = _mutual_top_k(edges_by_u, mutual_k)
    _log(f"[Cluster] {len(mutual_edges):,} mutual top-{mutual_k} edges ({time.time() - t0:.1f}s)")

    # --- Phase 6: Connected components via union-find ---
    block_ids_list = [int(bid) for bid in block_ids_arr]
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in mutual_edges:
        union(a, b)

    for bid in block_ids_list:
        find(bid)

    # Group by root
    components: Dict[int, List[int]] = defaultdict(list)
    for bid in block_ids_list:
        components[find(bid)].append(bid)

    _log(f"[Cluster] {len(components):,} connected components, assigning cluster IDs...")

    # --- Phase 7: Assign cluster IDs with genome support filter ---
    # Lightweight pass: just assign block_to_cluster + count genomes.
    # genes_json is NOT built here — the pipeline reconstructs cluster
    # summaries post-merge anyway, so building them here is wasted work.
    t0 = time.time()
    bid_to_row = {int(block_ids_arr[i]): i for i in range(n_blocks)}
    all_genomes = col['all_genomes']
    n_anchors_arr = col.get('n_anchors')

    block_to_cluster: Dict[int, int] = {}
    cluster_rows = []
    cid = 1

    # Vectorized genome lookup: pre-extract string arrays
    q_genome_strs = col['q_genome_strs'] if 'q_genome_strs' in col else None
    t_genome_strs = col['t_genome_strs'] if 't_genome_strs' in col else None

    for root, members in tqdm(components.items(), desc="Assigning clusters",
                               total=len(components), file=sys.stderr):
        genomes: Set[str] = set()
        total_genes = 0

        for bid in members:
            row = bid_to_row[bid]
            if q_genome_strs is not None:
                genomes.add(q_genome_strs[row])
                genomes.add(t_genome_strs[row])
            else:
                genomes.add(all_genomes[col['q_genome'][row]])
                genomes.add(all_genomes[col['t_genome'][row]])

            if n_anchors_arr is not None:
                total_genes += int(n_anchors_arr[row])
            else:
                total_genes += (int(col['q_end'][row]) - int(col['q_start'][row]) + 1)

        if len(genomes) >= min_genome_support:
            for bid in members:
                block_to_cluster[bid] = cid

            mean_chain_len = total_genes / len(members) if members else 0.0
            cluster_rows.append({
                "cluster_id": cid,
                "size": len(members),
                "genome_support": len(genomes),
                "mean_chain_length": round(mean_chain_len, 2),
            })
            cid += 1
        else:
            for bid in members:
                block_to_cluster[bid] = 0

    clusters_df = pd.DataFrame(cluster_rows) if cluster_rows else pd.DataFrame(
        columns=["cluster_id", "size", "genome_support", "mean_chain_length"]
    )

    n_assigned = sum(1 for c in block_to_cluster.values() if c > 0)
    _log(f"[Cluster] Assigned {n_assigned:,} blocks to {cid - 1:,} clusters in {time.time() - t0:.1f}s")

    return block_to_cluster, clusters_df


def _build_cluster_footprints(block_to_cluster, blocks):
    """Build per-cluster gene footprints from blocks.

    Returns (cluster_positions, cluster_genome_set, blocks_per_cluster):
      cluster_positions: cid -> {genome: sorted int64 array of composite keys}
      cluster_genome_set: cid -> set of genome strings
      blocks_per_cluster: cid -> int
    """
    if isinstance(blocks, pd.DataFrame):
        bid_arr = blocks['block_id'].values
        qg_arr = blocks['query_genome'].values
        tg_arr = blocks['target_genome'].values
        qc_arr = blocks['query_contig'].values
        tc_arr = blocks['target_contig'].values
        qs_arr = blocks['query_start'].values.astype(np.int64)
        qe_arr = blocks['query_end'].values.astype(np.int64)
        ts_arr = blocks['target_start'].values.astype(np.int64)
        te_arr = blocks['target_end'].values.astype(np.int64)
    else:
        bid_arr = np.array([b.block_id for b in blocks])
        qg_arr = np.array([b.query_genome for b in blocks])
        tg_arr = np.array([b.target_genome for b in blocks])
        qc_arr = np.array([b.query_contig for b in blocks])
        tc_arr = np.array([b.target_contig for b in blocks])
        qs_arr = np.array([b.query_start for b in blocks], dtype=np.int64)
        qe_arr = np.array([b.query_end for b in blocks], dtype=np.int64)
        ts_arr = np.array([b.target_start for b in blocks], dtype=np.int64)
        te_arr = np.array([b.target_end for b in blocks], dtype=np.int64)

    # Factorize contigs → int codes for composite keys
    all_contigs_arr = np.concatenate([qc_arr, tc_arr])
    contig_uniques, contig_codes_all = np.unique(all_contigs_arr, return_inverse=True)
    n_blocks_total = len(bid_arr)
    qc_codes = contig_codes_all[:n_blocks_total].astype(np.int64)
    tc_codes = contig_codes_all[n_blocks_total:].astype(np.int64)
    MAX_POS = int(max(qe_arr.max(), te_arr.max())) + 1

    bid_to_row = {int(bid_arr[i]): i for i in range(n_blocks_total)}

    # Per-cluster: collect composite keys per genome, then sort+unique
    _positions: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    blocks_per_cluster: Dict[int, int] = defaultdict(int)
    cluster_genome_set: Dict[int, Set[str]] = defaultdict(set)

    for bid, cid in block_to_cluster.items():
        if cid == 0:
            continue
        blocks_per_cluster[cid] += 1
        row = bid_to_row[bid]
        qg = qg_arr[row]
        tg = tg_arr[row]
        qs, qe = int(qs_arr[row]), int(qe_arr[row])
        ts, te = int(ts_arr[row]), int(te_arr[row])
        qcc = int(qc_codes[row])
        tcc = int(tc_codes[row])

        cluster_genome_set[cid].add(qg)
        cluster_genome_set[cid].add(tg)

        base_q = qcc * MAX_POS
        _positions[cid][qg].extend(range(base_q + qs, base_q + qe + 1))
        base_t = tcc * MAX_POS
        _positions[cid][tg].extend(range(base_t + ts, base_t + te + 1))

    # Convert lists → sorted unique int64 arrays
    cluster_positions: Dict[int, Dict[str, np.ndarray]] = {}
    for cid, gdict in _positions.items():
        cluster_positions[cid] = {
            g: np.unique(np.array(vals, dtype=np.int64))
            for g, vals in gdict.items()
        }
    del _positions

    return cluster_positions, cluster_genome_set, blocks_per_cluster


def merge_contained_clusters(
    block_to_cluster: Dict[int, int],
    blocks: Union[List[ChainedBlock], pd.DataFrame],
    containment_threshold: float = 0.8,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Merge clusters whose genomic footprints are largely contained in a larger cluster.

    Uses sparse matrix M.T @ M to compute all pairwise overlaps in one
    vectorized pass, replacing the O(n²) Python loop over cluster pairs.
    Shared-genome denominators are computed via a genome×cluster size matrix S
    and its binarized form B, using sparse column extraction for the
    overlapping pairs only.

    Returns:
        block_to_cluster: updated mapping (block_id -> merged cluster_id)
        merge_map: mapping of child_cluster -> parent_cluster (pre-resolve)
    """
    from scipy.sparse import coo_matrix

    t_fp = time.time()
    cluster_positions, cluster_genome_set, blocks_per_cluster = \
        _build_cluster_footprints(block_to_cluster, blocks)
    _log(f"[Cluster] Built footprints for {len(cluster_positions):,} clusters "
         f"in {time.time() - t_fp:.1f}s")

    if not cluster_positions:
        return block_to_cluster, {}

    # Filter: only check clusters with 2+ blocks
    mergeable_cids = sorted(c for c, n in blocks_per_cluster.items() if n >= 2)
    n_skipped = len(cluster_positions) - len(mergeable_cids)

    # Genome-differentiated global position keys
    all_genomes_sorted = sorted(set(
        g for cid in mergeable_cids for g in cluster_positions.get(cid, {})))
    genome_to_code = {g: i for i, g in enumerate(all_genomes_sorted)}
    n_genomes = len(all_genomes_sorted)

    _log(f"[Cluster] Merge-check {len(mergeable_cids):,} clusters for containment "
         f"(skipped {n_skipped:,} singletons, {n_genomes:,} genomes)...")

    if len(mergeable_cids) < 2:
        return block_to_cluster, {}

    # --- Build sparse position×cluster matrix + genome×cluster size matrix ---
    cid_to_idx = {c: i for i, c in enumerate(mergeable_cids)}
    n_clust = len(mergeable_cids)

    max_composite = np.int64(1)
    for cid in mergeable_cids:
        for arr in cluster_positions.get(cid, {}).values():
            if len(arr) > 0:
                max_composite = max(max_composite, np.int64(arr.max()) + 1)

    # Collect position→cluster pairs AND genome→cluster sizes in one pass
    pos_chunks = []
    cidx_chunks = []
    cluster_sizes = np.zeros(n_clust, dtype=np.int64)
    sg_rows = []  # genome index
    sg_cols = []  # cluster index
    sg_vals = []  # per-genome cluster size

    for cid in mergeable_cids:
        idx = cid_to_idx[cid]
        for genome, positions in cluster_positions.get(cid, {}).items():
            gc = np.int64(genome_to_code[genome])
            global_pos = positions + gc * max_composite
            pos_chunks.append(global_pos)
            cidx_chunks.append(np.full(len(positions), idx, dtype=np.int32))
            n_pos_g = len(positions)
            cluster_sizes[idx] += n_pos_g
            sg_rows.append(genome_to_code[genome])
            sg_cols.append(idx)
            sg_vals.append(n_pos_g)

    del cluster_positions  # free footprint dicts

    all_pos = np.concatenate(pos_chunks)
    all_cidx = np.concatenate(cidx_chunks)
    del pos_chunks, cidx_chunks

    # Remap positions to dense row indices
    unique_pos, pos_rows = np.unique(all_pos, return_inverse=True)
    n_pos = len(unique_pos)
    del all_pos, unique_pos

    _log(f"[Cluster] Sparse overlap: {n_pos:,} positions × {n_clust:,} clusters, "
         f"nnz={len(pos_rows):,}")

    # Build sparse matrix M (positions × clusters) and compute M.T @ M
    M = coo_matrix(
        (np.ones(len(pos_rows), dtype=np.float32), (pos_rows, all_cidx)),
        shape=(n_pos, n_clust)
    ).tocsc()
    del pos_rows, all_cidx

    t0 = time.time()
    MtM = (M.T @ M).tocoo()
    del M
    _log(f"[Cluster] Sparse M.T @ M in {time.time() - t0:.1f}s "
         f"({MtM.nnz:,} nnz)")

    # Extract upper triangle off-diagonal (avoid double-counting)
    mask = MtM.row < MtM.col
    ci = MtM.row[mask]
    cj = MtM.col[mask]
    overlap_vals = MtM.data[mask].astype(np.float64)
    del MtM, mask

    if len(ci) == 0:
        return block_to_cluster, {}

    _log(f"[Cluster] {len(ci):,} overlapping cluster pairs")

    # --- Shared-genome sizes via genome×cluster size matrix ---
    # S[g,c] = number of positions of cluster c in genome g
    # B[g,c] = 1 if cluster c has positions in genome g
    # shared_size_i_wrt_j = (S[:,i] * B[:,j]).sum() = size of i in genomes shared with j
    S = coo_matrix(
        (np.array(sg_vals, dtype=np.float64),
         (np.array(sg_rows, dtype=np.int32),
          np.array(sg_cols, dtype=np.int32))),
        shape=(n_genomes, n_clust)
    ).tocsc()
    del sg_rows, sg_cols, sg_vals
    B = S.copy()
    B.data[:] = 1.0

    t0 = time.time()
    # Extract columns for pair indices and compute shared-genome sizes
    S_ci = S[:, ci]       # (n_genomes × n_pairs) sparse
    B_cj = B[:, cj]       # (n_genomes × n_pairs) sparse
    shared_size_ci = np.asarray(S_ci.multiply(B_cj).sum(axis=0)).ravel()

    S_cj = S[:, cj]
    B_ci = B[:, ci]
    shared_size_cj = np.asarray(S_cj.multiply(B_ci).sum(axis=0)).ravel()
    del S, B, S_ci, B_cj, S_cj, B_ci
    _log(f"[Cluster] Shared-genome sizes computed in {time.time() - t0:.1f}s")

    # Determine small/large by total footprint size
    small_mask = cluster_sizes[ci] <= cluster_sizes[cj]
    small_cidx = np.where(small_mask, ci, cj)
    large_cidx = np.where(small_mask, cj, ci)
    small_shared = np.where(small_mask, shared_size_ci, shared_size_cj)
    large_shared = np.where(small_mask, shared_size_cj, shared_size_ci)

    # Containment = overlap / shared-genome size (matches original semantics)
    containment = np.divide(overlap_vals, small_shared,
                            out=np.zeros(len(ci), dtype=np.float64),
                            where=small_shared > 0)
    reciprocal = np.divide(overlap_vals, large_shared,
                           out=np.zeros(len(ci), dtype=np.float64),
                           where=large_shared > 0)

    valid = (containment >= containment_threshold) & (reciprocal >= 0.5)
    n_valid = int(valid.sum())

    _log(f"[Cluster] {n_valid:,} merge candidates "
         f"(from {len(ci):,} overlapping pairs)")

    if n_valid == 0:
        return block_to_cluster, {}

    # For each small cluster, pick the largest container
    small_cids_v = small_cidx[valid]
    large_cids_v = large_cidx[valid]
    large_shared_v = large_shared[valid]

    best_merge: Dict[int, Tuple[int, float]] = {}
    for k in range(n_valid):
        sc = int(small_cids_v[k])
        lc = int(large_cids_v[k])
        ls = float(large_shared_v[k])
        if sc not in best_merge or ls > best_merge[sc][1]:
            best_merge[sc] = (lc, ls)

    merge_candidates = [
        (mergeable_cids[sc], mergeable_cids[lc])
        for sc, (lc, _) in best_merge.items()
    ]

    _log(f"[Cluster] {len(merge_candidates):,} unique merges after dedup")

    # --- Resolve merge chains ---
    merge_map: Dict[int, int] = {}

    def resolve(c: int) -> int:
        while c in merge_map:
            c = merge_map[c]
        return c

    n_applied = 0
    for small_cid, large_cid in merge_candidates:
        small_r = resolve(small_cid)
        large_r = resolve(large_cid)
        if small_r == large_r:
            continue
        ss = cluster_sizes[cid_to_idx.get(small_r, 0)]
        ls = cluster_sizes[cid_to_idx.get(large_r, 0)]
        if ss > ls:
            merge_map[large_r] = small_r
        else:
            merge_map[small_r] = large_r
        n_applied += 1

    _log(f"[Cluster] Applied {n_applied:,} merges")

    # Apply merges
    new_mapping = {}
    for bid, cid in block_to_cluster.items():
        if cid == 0:
            new_mapping[bid] = 0
        else:
            new_mapping[bid] = resolve(cid)
    return new_mapping, dict(merge_map)


def _mutual_top_k(edges_by_u: Dict[int, List[Tuple[int, float]]], k: int) -> Set[Tuple[int, int]]:
    """Return set of undirected edges that are mutual top-k by weight."""
    topk: Dict[int, Set[int]] = {}
    for u, neigh in edges_by_u.items():
        neigh_sorted = sorted(neigh, key=lambda x: (-x[1], x[0]))[:k]
        topk[u] = {v for v, _ in neigh_sorted}
    keep: Set[Tuple[int, int]] = set()
    for u, neigh in edges_by_u.items():
        for v, _w in neigh:
            if v in topk.get(u, set()) and u in topk.get(v, set()):
                a, b = (u, v) if u < v else (v, u)
                keep.add((a, b))
    return keep
