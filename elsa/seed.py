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
) -> pd.DataFrame:
    """
    Find similar gene pairs across genomes using kNN search.

    Args:
        index_tuple: (type, index) from build_gene_index
        embeddings: (n_genes, dim) array of gene embeddings
        gene_info: DataFrame with columns [gene_id, sample_id, contig_id, position_index]
        k: Number of neighbors to retrieve per gene
        similarity_threshold: Minimum cosine similarity for anchors
        n_jobs: Number of threads for FAISS search (None = all cores)

    Returns:
        DataFrame with columns defined in ANCHOR_COLS
    """
    index_type, index = index_tuple
    if index is None or len(gene_info) == 0:
        return _empty_anchor_df()

    n = len(gene_info)

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

    # --- Streaming anchor discovery: filter each FAISS chunk immediately ---
    # Instead of materializing the full (n * k_query) result and then filtering,
    # we process each chunk in-place and keep only valid cross-genome pairs.
    # This reduces peak memory from O(n*k) to O(CHUNK*k + n_kept).
    CHUNK = 100_000
    chunk_rows = []   # filtered (row_idx, col_idx, sims) per chunk
    n_raw_kept = 0

    if index_type == "hnsw":
        labels, distances = index.knn_query(query, k=k_query)
        similarities = 1 - distances
        labels = labels.astype(np.intp)
        del query  # free after search
        # Filter all at once (HNSW datasets are typically smaller)
        row_idx = np.repeat(np.arange(n), k_query)
        col_idx = labels.ravel()
        sims_flat = similarities.ravel()
        valid = (col_idx >= 0) & (sims_flat >= similarity_threshold) & (row_idx != col_idx)
        row_idx = row_idx[valid]; col_idx = col_idx[valid]; sims_flat = sims_flat[valid]
        cross = genome_codes[row_idx] != genome_codes[col_idx]
        chunk_rows.append((row_idx[cross].astype(np.int32),
                           col_idx[cross].astype(np.int32),
                           sims_flat[cross]))
        n_raw_kept = int(cross.sum())
        del labels, distances, similarities

    elif index_type == "faiss":
        import os, faiss as _faiss
        _ncpu = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
        _faiss.omp_set_num_threads(_ncpu)

        from tqdm import trange
        for start in trange(0, n, CHUNK, desc="FAISS kNN search + filter"):
            end = min(start + CHUNK, n)
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

            if len(ri) > 0:
                chunk_rows.append((ri, ci, si))
                n_raw_kept += len(ri)

        del query  # free 4.3 GB after all chunks processed
        # Note: do NOT call omp_set_num_threads(1) here. FAISS and Numba
        # may use separate libomp copies (KMP_DUPLICATE_LIB_OK=TRUE), and
        # mutating OpenMP global state between them causes thread-pool
        # corruption → SIGKILL on macOS. Numba now uses workqueue backend.

    else:  # sklearn
        distances, labels = index.kneighbors(query, n_neighbors=k_query)
        del query  # free after search
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
        si = strand_arr[row_idx].copy()
        sj = strand_arr[col_idx].copy()
        si[si == 0] = 1
        sj[sj == 0] = 1
        orientations = np.where(si == sj, 1, -1).astype(np.int8)
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
# Seed-and-extend: recover adjacencies that top-k retrieval could not see
# ---------------------------------------------------------------------------
def _contig_keys(gene_info: pd.DataFrame) -> np.ndarray:
    """Return an int32 contig key per gene row.

    ``gene_info`` is coordinate-sorted by (sample, contig, start), so a contig
    boundary is exactly where either identifier changes and the key is a running
    count of those boundaries.
    """
    sample = gene_info["sample_id"].values
    contig = gene_info["contig_id"].values
    n = len(gene_info)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    np.logical_or(sample[1:] != sample[:-1], contig[1:] != contig[:-1], out=changed[1:])
    return np.cumsum(changed, dtype=np.int32) - 1


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


def _resolve_gene_rows(
    gene_info: pd.DataFrame,
    contig_key: np.ndarray,
    genome_values: Any,
    contig_values: Any,
    position_values: np.ndarray,
) -> np.ndarray:
    """Map (genome, contig, contig-local position) to a global gene row.

    ``ANCHOR_COLS`` stores ``query_idx``/``target_idx`` as contig-local position
    indices, not row offsets into ``gene_info``. Extension needs the actual rows
    to gather embeddings, so they are reconstructed from each contig's first row
    plus the position. Rows resolve to -1 when the identifiers are unknown or the
    position falls outside the contig.
    """
    starts = np.flatnonzero(np.concatenate(([True], contig_key[1:] != contig_key[:-1])))
    lengths = np.diff(np.concatenate((starts, [len(gene_info)])))

    gi_genome = gene_info["sample_id"].to_numpy()[starts]
    gi_contig = gene_info["contig_id"].to_numpy()[starts]

    g_codes, g_cats = _category_codes(genome_values)
    c_codes, c_cats = _category_codes(contig_values)

    gi_g = np.searchsorted(g_cats, gi_genome)
    gi_c = np.searchsorted(c_cats, gi_contig)
    known = (
        (gi_g < len(g_cats)) & (gi_c < len(c_cats))
        & (g_cats[np.minimum(gi_g, len(g_cats) - 1)] == gi_genome)
        & (c_cats[np.minimum(gi_c, len(c_cats) - 1)] == gi_contig)
    )

    stride = np.int64(len(c_cats))
    gi_combined = gi_g.astype(np.int64) * stride + gi_c
    gi_combined = np.where(known, gi_combined, np.int64(-1))
    order = np.argsort(gi_combined, kind="stable")
    sorted_combined = gi_combined[order]

    wanted = g_codes.astype(np.int64) * stride + c_codes
    wanted = np.where((g_codes >= 0) & (c_codes >= 0), wanted, np.int64(-1))
    slot = np.searchsorted(sorted_combined, wanted)
    slot_clipped = np.minimum(slot, len(sorted_combined) - 1)
    hit = (slot < len(sorted_combined)) & (sorted_combined[slot_clipped] == wanted)

    contig_idx = order[slot_clipped]
    position = position_values.astype(np.int64)
    rows = starts[contig_idx] + position
    valid = hit & (position >= 0) & (position < lengths[contig_idx])
    return np.where(valid, rows, np.int64(-1))


def extend_anchor_adjacencies(
    anchors_df: pd.DataFrame,
    embeddings: np.ndarray,
    gene_info: pd.DataFrame,
    similarity_threshold: float = 0.953,
    max_radius: int = 2,
    min_chain_size: int = 2,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Recover collinear anchors that top-k retrieval structurally cannot find.

    Chaining requires two neighbouring genes to match into the *same* genome.
    With global top-k retrieval each gene samples an arbitrary k-subset of the
    cohort, so the genomes present in both lists number about ``k^2 / n_genomes``
    — a coincidence that collapses as cohorts grow. Measured on a 3,564-genome
    set at k=50, two adjacent genes shared a median of 4 genomes out of 50.

    No search is needed to close that gap. Once gene ``i`` anchors to position
    ``p`` of some genome, whether gene ``i+1`` matches position ``p+1`` is a
    single dot product against a known vector. This tests those neighbours
    directly for anchors whose contig pair holds too few anchors to chain,
    which is where the loss actually falls.

    Returns only the newly recovered anchors; the caller concatenates them.
    """
    if anchors_df.empty or max_radius < 1:
        return _empty_anchor_df()

    contig_key = _contig_keys(gene_info)
    n_genes = len(gene_info)

    # Anchor columns carry contig-local positions; resolve them to gene rows.
    query_idx = _resolve_gene_rows(
        gene_info, contig_key,
        anchors_df["query_genome"], anchors_df["query_contig"],
        anchors_df["query_idx"].to_numpy(),
    )
    target_idx = _resolve_gene_rows(
        gene_info, contig_key,
        anchors_df["target_genome"], anchors_df["target_contig"],
        anchors_df["target_idx"].to_numpy(),
    )
    resolved = (query_idx >= 0) & (target_idx >= 0)
    if not resolved.all():
        _log(
            f"[Extend] {int((~resolved).sum()):,} anchors did not resolve to a gene "
            "row and are skipped"
        )

    # Restrict to contig pairs that cannot currently form a chain.
    n_contigs = int(contig_key.max()) + 1 if n_genes else 1
    safe_q = np.where(resolved, query_idx, 0)
    safe_t = np.where(resolved, target_idx, 0)
    pair_key = contig_key[safe_q].astype(np.int64) * n_contigs + contig_key[safe_t]
    uniq, inverse, counts = np.unique(pair_key, return_inverse=True, return_counts=True)
    eligible = (counts[inverse] < min_chain_size) & resolved
    del pair_key, uniq, inverse, counts, safe_q, safe_t
    n_eligible = int(eligible.sum())
    _log(
        f"[Extend] {n_eligible:,}/{len(anchors_df):,} anchors sit on contig pairs "
        f"below min_chain_size={min_chain_size}"
    )
    if n_eligible == 0:
        return _empty_anchor_df()

    qi = query_idx[eligible]
    ti = target_idx[eligible]
    # Retain the existing pairs as one sorted int64 index so the recovered set
    # can be de-duplicated after the large index arrays are released.
    existing_pairs = np.sort(
        query_idx[resolved].astype(np.int64) * n_genes + target_idx[resolved]
    )
    del query_idx, target_idx, eligible, resolved

    # Offsets: same sign continues collinearity, opposite sign continues an
    # inverted block. Both sides of the seed are probed.
    offsets = []
    for radius in range(1, max_radius + 1):
        offsets.extend([(radius, radius), (radius, -radius),
                        (-radius, -radius), (-radius, radius)])

    found_q: List[np.ndarray] = []
    found_t: List[np.ndarray] = []
    found_s: List[np.ndarray] = []
    found_o: List[np.ndarray] = []

    for dq, dt in offsets:
        for start in range(0, len(qi), chunk_size):
            end = min(start + chunk_size, len(qi))
            q = qi[start:end] + dq
            t = ti[start:end] + dt
            # Clamp before indexing: numpy evaluates every operand of the
            # expression, so an in-range test cannot protect a gather placed
            # alongside it in the same conjunction.
            in_range = (q >= 0) & (q < n_genes) & (t >= 0) & (t < n_genes)
            if not in_range.any():
                continue
            q_safe = np.where(in_range, q, 0)
            t_safe = np.where(in_range, t, 0)
            valid = (
                in_range
                & (contig_key[q_safe] == contig_key[qi[start:end]])
                & (contig_key[t_safe] == contig_key[ti[start:end]])
            )
            if not valid.any():
                continue
            q = q[valid]
            t = t[valid]
            a = np.asarray(embeddings[q], dtype=np.float32)
            b = np.asarray(embeddings[t], dtype=np.float32)
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
            b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
            sim = np.einsum("ij,ij->i", a, b)
            del a, b
            keep = sim >= similarity_threshold
            if keep.any():
                found_q.append(q[keep])
                found_t.append(t[keep])
                found_s.append(sim[keep])
                # Same-sign offsets continue a forward block, opposite an
                # inverted one. The offset that produced the pair carries this
                # directly; gene positions alone cannot recover it.
                found_o.append(
                    np.full(int(keep.sum()), 1 if (dq > 0) == (dt > 0) else -1,
                            dtype=np.int32)
                )
            del q, t, sim, keep

    if not found_q:
        _log("[Extend] no adjacencies cleared the threshold")
        return _empty_anchor_df()

    new_q = np.concatenate(found_q)
    new_t = np.concatenate(found_t)
    new_s = np.concatenate(found_s)
    orientation = np.concatenate(found_o)
    del found_q, found_t, found_s, found_o

    # Drop duplicates among the recovered set, then anything already anchored.
    # Both passes stay in int64 array space: a Python set over 150M+ existing
    # pairs would cost tens of gigabytes.
    combined = new_q.astype(np.int64) * n_genes + new_t
    _, first = np.unique(combined, return_index=True)
    first.sort()
    new_q, new_t, new_s = new_q[first], new_t[first], new_s[first]
    orientation = orientation[first]
    combined = combined[first]

    existing = existing_pairs
    slot = np.searchsorted(existing, combined)
    slot_clipped = np.minimum(slot, len(existing) - 1)
    already = (slot < len(existing)) & (existing[slot_clipped] == combined)
    keep_new = ~already
    del existing, slot, slot_clipped, already, combined

    new_q, new_t, new_s = new_q[keep_new], new_t[keep_new], new_s[keep_new]
    orientation = orientation[keep_new]
    _log(f"[Extend] recovered {len(new_q):,} new anchors")
    if len(new_q) == 0:
        return _empty_anchor_df()

    genome_arr = gene_info["sample_id"].values
    contig_arr = gene_info["contig_id"].values
    gene_id_arr = gene_info["gene_id"].values
    pos_arr = gene_info["position_index"].values

    # Emit contig-local positions to match ANCHOR_COLS, not the gene rows used
    # internally for embedding lookups.
    return pd.DataFrame({
        "query_idx": pos_arr[new_q].astype(np.int32),
        "target_idx": pos_arr[new_t].astype(np.int32),
        "query_genome": genome_arr[new_q],
        "target_genome": genome_arr[new_t],
        "query_contig": contig_arr[new_q],
        "target_contig": contig_arr[new_t],
        "query_gene_id": gene_id_arr[new_q],
        "target_gene_id": gene_id_arr[new_t],
        "similarity": new_s,
        "orientation": orientation,
    })


# ---------------------------------------------------------------------------
# Grouping by contig pair — fully vectorized
# ---------------------------------------------------------------------------
def group_anchors_by_contig_pair(
    anchors_df: pd.DataFrame,
) -> Dict[Tuple[str, str, str, str], pd.DataFrame]:
    """
    Group anchors by (query_genome, target_genome, query_contig, target_contig).

    Uses canonical ordering (smaller genome first) to avoid duplicates.
    Operates on DataFrame; returns dict of sub-DataFrames.

    Uses numpy composite-key sort instead of pandas groupby for ~5-10x
    speedup on large DataFrames (153M+ rows).
    """
    if anchors_df.empty:
        return {}

    n = len(anchors_df)

    # Canonical ordering: swap rows where query_genome > target_genome.
    # Swap column values via numpy arrays to avoid a full DataFrame copy
    # (critical for 100M+ row DataFrames where .copy() doubles memory).
    needs_swap = anchors_df["query_genome"].values > anchors_df["target_genome"].values
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

    n_tg = len(tg_uniques)
    n_qc = len(qc_uniques)
    n_tc = len(tc_uniques)

    # Composite key — fits int64 as long as product of cardinalities < 2^63
    composite = ((qg_codes * n_tg + tg_codes) * n_qc + qc_codes) * n_tc + tc_codes

    _log(f"[GroupAnchors] Sorting {n:,} anchors by group key...")
    order = np.argsort(composite, kind='quicksort')
    sorted_keys = composite[order]
    del composite

    # Find group boundaries
    diffs = np.empty(n, dtype=np.bool_)
    diffs[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=diffs[1:])
    group_starts = np.nonzero(diffs)[0]
    del diffs, sorted_keys

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
