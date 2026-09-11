"""
Anchor seeding: find similar gene pairs across genomes.

Provides cross-genome anchor discovery via kNN and contig-pair grouping.
Anchors are represented as DataFrames for columnar efficiency.

Legacy GeneAnchor dataclass is retained for backward compatibility
(search.py, tests).
"""

from __future__ import annotations

import sys
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
from ._log import tlog as _log


# ---------------------------------------------------------------------------
# Grouped anchors — raw arrays + sort order + group boundaries
# ---------------------------------------------------------------------------
GroupedAnchors = namedtuple('GroupedAnchors', [
    'order',          # (n,) int64 — sort order into the raw arrays
    'group_bounds',   # (n_groups+1,) int64 — boundaries in sorted order
    'n_groups',       # int
    # Raw column arrays (original DataFrame order; use order[bounds[i]:bounds[i+1]] to index)
    'query_idx', 'target_idx', 'similarity', 'orientation',
    'query_gene_id', 'target_gene_id',
    # Integer codes + uniques for reconstructing group keys
    'qg_codes', 'tg_codes', 'qc_codes', 'tc_codes',
    'qg_uniques', 'tg_uniques', 'qc_uniques', 'tc_uniques',
])


# ---------------------------------------------------------------------------
# Anchor DataFrame column spec
# ---------------------------------------------------------------------------
ANCHOR_COLS = [
    "query_idx", "target_idx",
    "query_genome", "target_genome",
    "query_contig", "target_contig",
    "query_gene_id", "target_gene_id",
    "similarity", "orientation",
]


def _empty_anchor_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard anchor schema."""
    return pd.DataFrame(columns=ANCHOR_COLS)


# ---------------------------------------------------------------------------
# Legacy dataclass — kept for backward compat (search, tests, benchmarks)
# ---------------------------------------------------------------------------
@dataclass
class GeneAnchor:
    """A similar gene pair across two genomes."""
    query_idx: int
    target_idx: int
    query_genome: str
    target_genome: str
    query_contig: str
    target_contig: str
    query_gene_id: str
    target_gene_id: str
    similarity: float
    orientation: int = 0


# ---------------------------------------------------------------------------
# Main anchor discovery — returns DataFrame
# ---------------------------------------------------------------------------
def find_cross_genome_anchors(
    index_tuple: Tuple[str, Any],
    embeddings: np.ndarray,
    gene_info: pd.DataFrame,
    k: int = 50,
    similarity_threshold: float = 0.85,
    n_jobs: Optional[int] = None,
    query_batch_size: int = 100_000,
    neighborhood_retrieval: bool = True,
    max_gap: int = 2,
) -> pd.DataFrame:
    """
    Find similar gene pairs across genomes using kNN search.

    Args:
        index_tuple: (type, index) from build_gene_index
        embeddings: (n_genes, dim) array of gene embeddings
        gene_info: DataFrame with columns [gene_id, sample_id, contig_id, position_index]
        k: Number of neighbors to retrieve per gene
        similarity_threshold: Minimum cosine similarity for anchors
        n_jobs: Number of threads for HNSW/FAISS search (None = backend default)
        neighborhood_retrieval: Prioritize candidates supported by adjacent original seeds.
            Selection retains at most k + 10 targets per query before reciprocal dedup.
        max_gap: Neighbor steps span one through max_gap + 1 genes.
        query_batch_size: Maximum queries per HNSW/FAISS search call. Bounds
            neighbor-result scratch to O(query_batch_size * (k + 10)); retained
            anchors still occupy O(n_kept). Use len(gene_info) for one call.

    Returns:
        DataFrame with columns defined in ANCHOR_COLS
    """
    index_type, index = index_tuple
    if index is None or len(gene_info) == 0:
        return _empty_anchor_df()

    n = len(gene_info)
    if not isinstance(query_batch_size, (int, np.integer)) or query_batch_size < 1:
        raise ValueError("query_batch_size must be a positive integer")

    # Normalize in-place as float32 where possible — no second array.
    # `query = embeddings / norms` would allocate a full copy (~4.3 GB for 3.4M genes).
    #
    # In-place divide mutates the caller's array, so it is only valid on a
    # writable one. A read-only memory map — the natural way to hand over a
    # multi-gigabyte embedding matrix, and what a transform ablation produces —
    # must fall back to a copy rather than raising.
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32)
    np.maximum(norms, np.float32(1e-9), out=norms)
    writable = getattr(embeddings, "flags", None) is not None and embeddings.flags.writeable
    if writable:
        embeddings /= norms      # in-place divide — no copy
        query = embeddings       # alias, not a copy
    else:
        _log("[GeneAnchor] embeddings are read-only; copying to normalize "
             f"(~{embeddings.nbytes / 1e9:.1f} GB)")
        query = embeddings / norms
    del norms
    del embeddings               # drop the name; query is now the only reference

    k_query = min(k + 10, n)

    # Build lookup arrays (needed for per-chunk cross-genome filtering).
    #
    # The cross-genome test runs on integer codes rather than the genome strings.
    # ``genome_arr[a] != genome_arr[b]`` over an object array dispatches a Python
    # string compare per element, and this predicate is evaluated once per
    # retrieved neighbour — n * k_query times, ~200M at DPANN scale. Comparing
    # int32 codes is ~12x faster and the codes are reused below for the
    # Categorical columns, so factorizing here costs nothing extra.
    genome_arr = gene_info['sample_id'].values
    genome_codes, genome_cats = _category_codes(genome_arr)

    # --- Streaming anchor discovery: filter each search chunk immediately ---
    # Instead of materializing the full (n * k_query) result and then filtering,
    # we process each chunk in-place and keep only valid cross-genome pairs.
    # Neighbor-result storage is O(query_batch_size*k + n_kept).
    chunk_rows = []   # filtered (row_idx, col_idx, sims) per chunk
    n_raw_kept = 0

    if index_type in ("hnsw", "faiss"):
        if index_type == "faiss":
            import os, faiss as _faiss
            _ncpu = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
            _faiss.omp_set_num_threads(_ncpu)

        # Query rows and each row's neighbors retain their original order, so
        # the first-occurrence reciprocal-pair dedup below is batch invariant.
        hnsw_kwargs = {"num_threads": n_jobs} if n_jobs is not None else {}

        from tqdm import trange
        for start in trange(0, n, query_batch_size,
                            desc=f"{index_type.upper()} kNN search + filter"):
            end = min(start + query_batch_size, n)
            if index_type == "hnsw":
                chunk_labels, chunk_distances = index.knn_query(
                    query[start:end], k=k_query, **hnsw_kwargs,
                )
                chunk_sims = 1 - chunk_distances
                del chunk_distances
            else:
                chunk_sims, chunk_labels = index.search(query[start:end], k_query)
            chunk_labels = chunk_labels.astype(np.int32)

            # Flatten this chunk
            chunk_n = end - start
            ri = np.repeat(np.arange(start, end, dtype=np.int32), k_query)
            ci = chunk_labels.ravel()
            si = chunk_sims.ravel()

            # Filter: valid, above threshold, not self, cross-genome
            valid = (ci >= 0) & (si >= similarity_threshold) & (ri != ci)
            ri = ri[valid]; ci = ci[valid]; si = si[valid]
            cross = genome_codes[ri] != genome_codes[ci]
            ri = ri[cross]; ci = ci[cross]; si = si[cross]

            if len(ri) > 0 or index_type == "hnsw":
                # HNSW historically constructs typed empty columns even when
                # every neighbor is filtered; keep that return schema.
                chunk_rows.append((ri, ci, si))
                n_raw_kept += len(ri)
            del chunk_labels, chunk_sims, ri, ci, si, valid, cross

        # Note: do NOT call omp_set_num_threads(1) here. FAISS and Numba
        # may use separate libomp copies (KMP_DUPLICATE_LIB_OK=TRUE), and
        # mutating OpenMP global state between them causes thread-pool
        # corruption → SIGKILL on macOS. Numba now uses workqueue backend.

    else:  # sklearn
        distances, labels = index.kneighbors(query, n_neighbors=k_query)
        similarities = 1 - distances
        labels = labels.astype(np.int32)
        row_idx = np.repeat(np.arange(n, dtype=np.int32), k_query)
        col_idx = labels.ravel()
        sims_flat = similarities.ravel()
        valid = (col_idx >= 0) & (sims_flat >= similarity_threshold) & (row_idx != col_idx)
        row_idx = row_idx[valid]; col_idx = col_idx[valid]; sims_flat = sims_flat[valid]
        cross = genome_codes[row_idx] != genome_codes[col_idx]
        chunk_rows.append((row_idx[cross], col_idx[cross], sims_flat[cross]))
        n_raw_kept = int(cross.sum())
        del labels, distances, similarities

    # Concatenate filtered chunks
    if not chunk_rows:
        return _empty_anchor_df()

    row_idx = np.concatenate([c[0] for c in chunk_rows])
    col_idx = np.concatenate([c[1] for c in chunk_rows])
    sims = np.concatenate([c[2] for c in chunk_rows])
    del chunk_rows

    if neighborhood_retrieval and len(row_idx):
        from .neighborhood import select_neighborhood_candidates
        selected = select_neighborhood_candidates(
            query, query, row_idx, col_idx, sims,
            query_metadata=gene_info, reference_metadata=gene_info,
            k=k_query, threshold=similarity_threshold, max_gap=max_gap,
            n_jobs=n_jobs,
        )
        row_idx, col_idx, sims = selected.query_rows, selected.target_rows, selected.similarities
        _log(f"[GeneAnchor] Neighborhood selection retained {len(row_idx):,} directional pairs")
        del selected
    del query

    _log(f"[GeneAnchor] {n_raw_kept} cross-genome pairs before dedup (n={n}, k={k_query})")

    # Deduplicate: canonical pair (min, max) — keep first occurrence (highest sim from sorted kNN)
    lo = np.minimum(row_idx, col_idx)
    hi = np.maximum(row_idx, col_idx)

    # Pack the pair into one int64 rather than sorting a two-field structured
    # dtype. np.unique on a structured array sorts field-by-field through a
    # comparison path; on a plain int64 it takes the fast scalar path, which
    # measures ~1.6x quicker at the 200M-pair scale this reaches on DPANN.
    # n < 2^31 keeps both halves addressable, which every real corpus satisfies —
    # the fallback below is for completeness, not for a case anyone has hit.
    if n < 2_147_483_648:
        pairs = (lo.astype(np.int64) << np.int64(32)) | hi.astype(np.int64)
        del lo, hi
    else:
        pairs = np.empty(len(lo), dtype=[('lo', np.int64), ('hi', np.int64)])
        pairs['lo'] = lo
        pairs['hi'] = hi
        del lo, hi

    _, unique_idx = np.unique(pairs, return_index=True)
    del pairs
    unique_idx.sort()

    row_idx = row_idx[unique_idx]
    col_idx = col_idx[unique_idx]
    sims = sims[unique_idx]
    del unique_idx

    _log(f"[GeneAnchor] {len(row_idx)} unique cross-genome pairs after dedup")

    # Build remaining lookup arrays (deferred to reduce peak memory —
    # only needed for the final deduped set, not the full n*k flattened arrays)
    contig_arr = gene_info['contig_id'].values
    pos_arr = gene_info['position_index'].values
    gene_id_arr = gene_info['gene_id'].values
    strand_arr = gene_info['strand'].values if 'strand' in gene_info.columns else None

    # Canonical ordering: ensure query_genome <= target_genome (alphabetically).
    # Done here on raw index arrays (cheap int swaps) so the DataFrame
    # is already canonical and group_anchors_by_contig_pair avoids
    # expensive column-level copies that double memory at 153M rows.
    # Codes are assigned over lexicographically sorted categories, so comparing
    # them orders genomes exactly as comparing the strings would.
    q_gen = genome_codes[row_idx]
    t_gen = genome_codes[col_idx]
    swap_mask = q_gen > t_gen
    if swap_mask.any():
        temp = row_idx[swap_mask].copy()
        row_idx[swap_mask] = col_idx[swap_mask]
        col_idx[swap_mask] = temp
        del temp
    del q_gen, t_gen

    # Compute orientations vectorized (symmetric w.r.t. query/target swap)
    if strand_arr is not None:
        si = strand_arr[row_idx]
        sj = strand_arr[col_idx]
        known = np.isin(si, [-1, 1]) & np.isin(sj, [-1, 1])
        orientations = np.where(known, si * sj, 0).astype(np.int8)
    else:
        orientations = np.zeros(len(row_idx), dtype=np.int8)

    # Use Categorical for genome/contig columns: saves ~3 GB at 153M rows
    # (int32 codes vs 8-byte object pointers). Ordered + shared categories
    # enable safe column swaps downstream without category mismatches.
    #
    # Built from codes rather than from the strings. ``pd.Categorical(strings,
    # categories=...)`` hashes every element against the category index, and
    # there are four such columns over the deduped anchor set — 40M lookups at
    # DPANN scale. Factorizing the gene table once and gathering codes is ~12x
    # quicker and produces an identical column.
    contig_codes, contig_cats = _category_codes(contig_arr)

    def _cat(codes: np.ndarray, categories: np.ndarray) -> pd.Categorical:
        return pd.Categorical.from_codes(codes, categories=categories, ordered=True)

    anchors_df = pd.DataFrame({
        "query_idx": pos_arr[row_idx],
        "target_idx": pos_arr[col_idx],
        "query_genome": _cat(genome_codes[row_idx], genome_cats),
        "target_genome": _cat(genome_codes[col_idx], genome_cats),
        "query_contig": _cat(contig_codes[row_idx], contig_cats),
        "target_contig": _cat(contig_codes[col_idx], contig_cats),
        "query_gene_id": gene_id_arr[row_idx],
        "target_gene_id": gene_id_arr[col_idx],
        "similarity": sims,
        "orientation": orientations,
    })

    return anchors_df


# ---------------------------------------------------------------------------
# Grouping by contig pair — fully vectorized
# ---------------------------------------------------------------------------
def _category_codes(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer codes plus their sorted category array."""
    series = pd.Series(values)
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = np.asarray(series.cat.categories)
        order = np.argsort(categories, kind="stable")
        remap = np.empty(len(categories), dtype=np.int64)
        remap[order] = np.arange(len(categories))
        codes = series.cat.codes.to_numpy()
        return np.where(codes >= 0, remap[np.maximum(codes, 0)], -1), categories[order]
    raw = series.to_numpy()
    categories = np.unique(raw)
    return np.searchsorted(categories, raw), categories


def group_anchors_by_contig_pair(
    anchors_df: pd.DataFrame,
    *,
    canonicalize: bool = True,
) -> Dict[Tuple[str, str, str, str], pd.DataFrame]:
    """
    Group anchors by (query_genome, target_genome, query_contig, target_contig).

    Canonical ordering (smaller genome first) combines reciprocal discovery
    anchors. Pass canonicalize=False to retain external query direction.
    Returns GroupedAnchors arrays, or an empty dictionary for empty input.
    """
    if anchors_df.empty:
        return {}

    n = len(anchors_df)

    # Canonical ordering: swap rows where query_genome > target_genome.
    # Swap column values via numpy arrays to avoid a full DataFrame copy
    # (critical for 100M+ row DataFrames where .copy() doubles memory).
    needs_swap = (anchors_df["query_genome"].values > anchors_df["target_genome"].values
                  if canonicalize else np.zeros(n, dtype=bool))
    n_swap = int(needs_swap.sum())

    if n_swap > 0:
        _log(f"[GroupAnchors] Canonical swap on {n_swap:,} / {n:,} rows...")
        sw = needs_swap
        for qa, ta in [
            ("query_idx", "target_idx"),
            ("query_genome", "target_genome"),
            ("query_contig", "target_contig"),
            ("query_gene_id", "target_gene_id"),
        ]:
            q_vals = anchors_df[qa].values.copy()
            t_vals = anchors_df[ta].values.copy()
            q_vals[sw], t_vals[sw] = t_vals[sw], q_vals[sw]
            anchors_df[qa] = q_vals
            anchors_df[ta] = t_vals
    else:
        _log(f"[GroupAnchors] Already canonical ({n:,} rows), skipping swap")

    # --- Numpy sort-based grouping (replaces pandas groupby) ---
    # Encode each grouping column as integer codes, then combine into a
    # single composite key. numpy sort on int64 is vectorized and uses
    # BLAS threads — much faster than pandas' single-threaded hash table.
    _log(f"[GroupAnchors] Encoding group keys for {n:,} anchors...")

    def _get_codes_and_uniques(col):
        """Get integer codes for a column (fast path for Categorical)."""
        if hasattr(col, 'cat'):
            codes = col.cat.codes.values.astype(np.int64)
            uniques = col.cat.categories.values
        else:
            uniques, codes = np.unique(col.values, return_inverse=True)
            codes = codes.astype(np.int64)
        return codes, uniques

    qg_codes, qg_uniques = _get_codes_and_uniques(anchors_df["query_genome"])
    tg_codes, tg_uniques = _get_codes_and_uniques(anchors_df["target_genome"])
    qc_codes, qc_uniques = _get_codes_and_uniques(anchors_df["query_contig"])
    tc_codes, tc_uniques = _get_codes_and_uniques(anchors_df["target_contig"])

    _log(f"[GroupAnchors] Sorting {n:,} anchors by group key...")
    # Stable tuple ordering also handles large contig cardinalities safely.
    order = np.lexsort((tc_codes, qc_codes, tg_codes, qg_codes))

    # Find group boundaries
    diffs = np.empty(n, dtype=np.bool_)
    diffs[0] = True
    diffs[1:] = False
    for codes in (qg_codes, tg_codes, qc_codes, tc_codes):
        diffs[1:] |= codes[order[1:]] != codes[order[:-1]]
    group_starts = np.nonzero(diffs)[0]
    del diffs

    n_groups = len(group_starts)
    # Append sentinel for boundary slicing
    group_bounds = np.empty(n_groups + 1, dtype=np.int64)
    group_bounds[:n_groups] = group_starts
    group_bounds[n_groups] = n
    del group_starts

    _log(f"[GroupAnchors] {n_groups:,} contig pairs, extracting raw arrays...")

    return GroupedAnchors(
        order=order,
        group_bounds=group_bounds,
        n_groups=n_groups,
        query_idx=anchors_df["query_idx"].values,
        target_idx=anchors_df["target_idx"].values,
        similarity=anchors_df["similarity"].values,
        orientation=anchors_df["orientation"].values,
        query_gene_id=anchors_df["query_gene_id"].values,
        target_gene_id=anchors_df["target_gene_id"].values,
        qg_codes=qg_codes, tg_codes=tg_codes,
        qc_codes=qc_codes, tc_codes=tc_codes,
        qg_uniques=qg_uniques, tg_uniques=tg_uniques,
        qc_uniques=qc_uniques, tc_uniques=tc_uniques,
    )
