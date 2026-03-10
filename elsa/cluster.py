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
    print(f"[Cluster] Converting {n_blocks:,} blocks to columnar format...",
          file=sys.stderr, flush=True)
    col = _blocks_to_columnar(blocks)

    # --- Phase 2: Build contig index ---
    contig_groups = _build_contig_index(col)
    print(f"[Cluster] {len(contig_groups):,} contig bins for interval sweep",
          file=sys.stderr, flush=True)

    # --- Phase 3: Parallel interval sweep ---
    n_workers = min(os.cpu_count() or 4, len(contig_groups))
    print(f"[Cluster] Running interval sweep ({n_workers} threads)...",
          file=sys.stderr, flush=True)
    raw_pairs_i, raw_pairs_j = _sweep_all_contigs(contig_groups, n_workers)
    del contig_groups

    print(f"[Cluster] {len(raw_pairs_i):,} raw candidate pairs from sweep",
          file=sys.stderr, flush=True)

    # Deduplicate: encode (min, max) as single int64 for fast 1D unique
    if len(raw_pairs_i) > 0:
        print(f"[Cluster] Deduplicating {len(raw_pairs_i):,} pairs...",
              file=sys.stderr, flush=True)
        lo = np.minimum(raw_pairs_i, raw_pairs_j).astype(np.int64)
        hi = np.maximum(raw_pairs_i, raw_pairs_j).astype(np.int64)
        # Composite key: lo * n_blocks + hi (fits in int64 for n < 2^31)
        composite = lo * np.int64(n_blocks) + hi
        del lo, hi, raw_pairs_i, raw_pairs_j
        composite = np.unique(composite)
        pairs_i = (composite // np.int64(n_blocks)).astype(np.int32)
        pairs_j = (composite % np.int64(n_blocks)).astype(np.int32)
        del composite
        print(f"[Cluster] {len(pairs_i):,} unique candidate pairs after dedup",
              file=sys.stderr, flush=True)
    else:
        pairs_i = raw_pairs_i
        pairs_j = raw_pairs_j

    # --- Phase 4: Vectorized Jaccard ---
    print(f"[Cluster] Computing Jaccard for {len(pairs_i):,} candidate pairs...",
          file=sys.stderr, flush=True)
    _, _, jaccard = _compute_jaccard_vectorized(pairs_i, pairs_j, col)

    mask = jaccard >= jaccard_tau
    n_edges = int(mask.sum())
    print(f"[Cluster] {n_edges:,} overlap edges above Jaccard threshold {jaccard_tau}",
          file=sys.stderr, flush=True)

    # --- Phase 5: Build edges dict for mutual top-k ---
    block_ids_arr = col['block_ids']
    edges_by_u: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    valid_i = pairs_i[mask]
    valid_j = pairs_j[mask]
    valid_jac = jaccard[mask]
    del pairs_i, pairs_j, jaccard, mask

    t0 = time.time()
    print(f"[Cluster] Building edge dict from {n_edges:,} edges...",
          file=sys.stderr, flush=True)

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

    print(f"[Cluster] Edge dict built in {time.time() - t0:.1f}s ({len(edges_by_u):,} nodes)",
          file=sys.stderr, flush=True)

    # Apply mutual top-k filter
    t0 = time.time()
    mutual_edges = _mutual_top_k(edges_by_u, mutual_k)
    print(f"[Cluster] {len(mutual_edges):,} mutual top-{mutual_k} edges ({time.time() - t0:.1f}s)",
          file=sys.stderr, flush=True)

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

    print(f"[Cluster] {len(components):,} connected components, assigning cluster IDs...",
          file=sys.stderr, flush=True)

    # --- Phase 7: Assign cluster IDs with genome support filter ---
    # Use columnar arrays directly — no block_map[bid] lookups
    t0 = time.time()
    bid_to_row = {int(block_ids_arr[i]): i for i in range(n_blocks)}
    all_genomes = col['all_genomes']
    all_contigs = col['all_contigs']
    n_anchors_arr = col.get('n_anchors')

    block_to_cluster: Dict[int, int] = {}
    cluster_rows = []
    cid = 1

    for root, members in tqdm(components.items(), desc="Assigning clusters",
                               total=len(components), file=sys.stderr):
        genomes: Set[str] = set()
        total_genes = 0
        genes_by_genome: Dict[str, List[str]] = defaultdict(list)

        for bid in members:
            row = bid_to_row[bid]
            qg = all_genomes[col['q_genome'][row]]
            tg = all_genomes[col['t_genome'][row]]
            qc = all_contigs[col['q_contig'][row]]
            tc = all_contigs[col['t_contig'][row]]
            qs = int(col['q_start'][row])
            qe = int(col['q_end'][row])
            ts = int(col['t_start'][row])
            te = int(col['t_end'][row])

            genomes.add(qg)
            genomes.add(tg)

            for idx in range(qs, qe + 1):
                genes_by_genome[qg].append(f"{qc}:{idx}")
            for idx in range(ts, te + 1):
                genes_by_genome[tg].append(f"{tc}:{idx}")

            if n_anchors_arr is not None:
                total_genes += int(n_anchors_arr[row])
            else:
                total_genes += (qe - qs + 1)

        if len(genomes) >= min_genome_support:
            for bid in members:
                block_to_cluster[bid] = cid

            mean_chain_len = total_genes / len(members) if members else 0.0
            cluster_rows.append({
                "cluster_id": cid,
                "size": len(members),
                "genome_support": len(genomes),
                "mean_chain_length": round(mean_chain_len, 2),
                "genes_json": json.dumps({g: list(set(ids)) for g, ids in genes_by_genome.items()}),
            })
            cid += 1
        else:
            for bid in members:
                block_to_cluster[bid] = 0

    clusters_df = pd.DataFrame(cluster_rows) if cluster_rows else pd.DataFrame(
        columns=["cluster_id", "size", "genome_support", "mean_chain_length", "genes_json"]
    )

    n_assigned = sum(1 for c in block_to_cluster.values() if c > 0)
    print(f"[Cluster] Assigned {n_assigned:,} blocks to {cid - 1:,} clusters in {time.time() - t0:.1f}s",
          file=sys.stderr, flush=True)

    return block_to_cluster, clusters_df


def _build_cluster_footprints(block_to_cluster, blocks):
    """Build per-cluster gene footprints from blocks.

    Returns (cluster_genomes, blocks_per_cluster) where cluster_genomes
    maps cid -> {genome: set((contig, idx), ...)}.
    """
    if isinstance(blocks, pd.DataFrame):
        bid_arr = blocks['block_id'].values
        qg_arr = blocks['query_genome'].values
        tg_arr = blocks['target_genome'].values
        qc_arr = blocks['query_contig'].values
        tc_arr = blocks['target_contig'].values
        qs_arr = blocks['query_start'].values
        qe_arr = blocks['query_end'].values
        ts_arr = blocks['target_start'].values
        te_arr = blocks['target_end'].values
        bid_to_row = {int(bid_arr[i]): i for i in range(len(blocks))}
    else:
        n = len(blocks)
        bid_arr = np.array([b.block_id for b in blocks])
        qg_arr = np.array([b.query_genome for b in blocks])
        tg_arr = np.array([b.target_genome for b in blocks])
        qc_arr = np.array([b.query_contig for b in blocks])
        tc_arr = np.array([b.target_contig for b in blocks])
        qs_arr = np.array([b.query_start for b in blocks])
        qe_arr = np.array([b.query_end for b in blocks])
        ts_arr = np.array([b.target_start for b in blocks])
        te_arr = np.array([b.target_end for b in blocks])
        bid_to_row = {int(bid_arr[i]): i for i in range(n)}

    cluster_genomes: Dict[int, Dict[str, Set[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(set))
    blocks_per_cluster: Dict[int, int] = defaultdict(int)

    for bid, cid in block_to_cluster.items():
        if cid == 0:
            continue
        blocks_per_cluster[cid] += 1
        row = bid_to_row[bid]
        qg = str(qg_arr[row])
        tg = str(tg_arr[row])
        qc = str(qc_arr[row])
        tc = str(tc_arr[row])
        qs = int(qs_arr[row])
        qe = int(qe_arr[row])
        ts = int(ts_arr[row])
        te = int(te_arr[row])
        for idx in range(qs, qe + 1):
            cluster_genomes[cid][qg].add((qc, idx))
        for idx in range(ts, te + 1):
            cluster_genomes[cid][tg].add((tc, idx))

    return cluster_genomes, blocks_per_cluster


def merge_contained_clusters(
    block_to_cluster: Dict[int, int],
    blocks: Union[List[ChainedBlock], pd.DataFrame],
    containment_threshold: float = 0.8,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Merge clusters whose genomic footprints are largely contained in a larger cluster.

    Two-phase approach for parallelism:
      Phase 1 (parallel): Compute containment for each cluster against all larger
        candidates using frozen (original) footprints. Embarrassingly parallel.
      Phase 2 (serial): Process merge decisions in rank order, resolving conflicts.

    Accepts List[ChainedBlock] or DataFrame.

    Returns:
        block_to_cluster: updated mapping (block_id -> merged cluster_id)
        merge_map: mapping of child_cluster -> parent_cluster (pre-resolve)
    """
    # Build footprints
    cluster_genomes, blocks_per_cluster = _build_cluster_footprints(
        block_to_cluster, blocks)

    if not cluster_genomes:
        return block_to_cluster, {}

    # Compute footprint sizes for sorting
    cluster_size = {cid: sum(len(positions) for positions in gdict.values())
                    for cid, gdict in cluster_genomes.items()}

    # Sort by footprint size descending
    sorted_cids = sorted(cluster_size, key=lambda c: cluster_size[c], reverse=True)
    cid_to_rank = {cid: i for i, cid in enumerate(sorted_cids)}

    # Filter: only check clusters with 2+ blocks
    mergeable_cids = [c for c in sorted_cids if blocks_per_cluster.get(c, 0) >= 2]
    n_skipped = len(sorted_cids) - len(mergeable_cids)

    # Index clusters by genome — only include mergeable clusters
    genome_to_clusters: Dict[str, Set[int]] = defaultdict(set)
    for cid in mergeable_cids:
        for g in cluster_genomes[cid]:
            genome_to_clusters[g].add(cid)

    print(f"[Cluster] Merge-check {len(mergeable_cids):,} clusters for containment "
          f"(skipped {n_skipped:,} singletons, {len(genome_to_clusters):,} genomes)...",
          file=sys.stderr, flush=True)

    if not mergeable_cids:
        return block_to_cluster, {}

    n_total = len(sorted_cids)

    # --- Phase 1: Compute containment on frozen (original) footprints ---
    # Serial but fast: singleton skip eliminates ~90% of clusters,
    # and we use frozen footprints (no mutation during iteration).
    merge_candidates = []
    n_checked = 0

    for small_cid in tqdm(mergeable_cids, desc="Containment check",
                          file=sys.stderr):
        small_rank = cid_to_rank[small_cid]
        if small_rank == 0:
            continue  # largest cluster can't merge into anything

        small_genome_dict = cluster_genomes.get(small_cid)
        if not small_genome_dict:
            continue

        # Find candidates sharing at least one genome
        candidate_cids: Set[int] = set()
        for g in small_genome_dict:
            candidate_cids.update(genome_to_clusters.get(g, set()))

        # Only check larger clusters (lower rank = larger footprint)
        candidates_larger = [c for c in candidate_cids
                             if cid_to_rank.get(c, n_total) < small_rank
                             and c != small_cid]

        for large_cid in candidates_larger:
            large_genome_dict = cluster_genomes.get(large_cid)
            if not large_genome_dict:
                continue
            n_checked += 1

            shared_genomes = set(small_genome_dict) & set(large_genome_dict)
            if not shared_genomes:
                continue

            small_shared = set()
            large_shared = set()
            for g in shared_genomes:
                small_shared.update(small_genome_dict[g])
                large_shared.update(large_genome_dict[g])

            if not small_shared:
                continue

            overlap = len(small_shared & large_shared)
            containment = overlap / len(small_shared)
            reciprocal = overlap / len(large_shared) if large_shared else 0

            if containment >= containment_threshold and reciprocal >= 0.5:
                merge_candidates.append((small_cid, large_cid, containment))
                break  # one merge per small cluster

    print(f"[Cluster] Checked {n_checked:,} candidate pairs, "
          f"{len(merge_candidates):,} merge candidates",
          file=sys.stderr, flush=True)

    if not merge_candidates:
        return block_to_cluster, {}

    # --- Phase 2: Resolve merge chain ---
    merge_map: Dict[int, int] = {}

    def resolve(c: int) -> int:
        while c in merge_map:
            c = merge_map[c]
        return c

    n_applied = 0
    for small_cid, large_cid, containment in merge_candidates:
        small_r = resolve(small_cid)
        large_r = resolve(large_cid)
        if small_r == large_r:
            continue
        # Always merge smaller into larger (by rank)
        sr = cid_to_rank.get(small_r, n_total)
        lr = cid_to_rank.get(large_r, n_total)
        if sr < lr:
            merge_map[large_r] = small_r
        else:
            merge_map[small_r] = large_r
        n_applied += 1

    print(f"[Cluster] Applied {n_applied:,} merges",
          file=sys.stderr, flush=True)

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
