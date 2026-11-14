"""Chaining for the ws_chain ablation.

mode='length_first' reproduces the published chainer exactly: it calls the published
preprocessing, DP, backtracking and paired-overlap kernels of the private snapshot copy,
and mirrors `_extract_blocks_flat` for extraction (score = summed cosine, rounded to 4
decimals in the output exactly as published).

mode='score_first' (V3): anchor weight w = calibrated log-odds. The DP keeps, for each anchor,
the best-scoring chain ending there: score = sum(w) - sum(gap cost), starting fresh when no
predecessor improves on w alone (local-alignment form). Endpoints are claimed in order of
score (then length); a path stops at the first claimed anchor; each claimed path is trimmed
to its best-scoring suffix of >= min_size anchors. Blocks are selected per contig pair
greedily by that score with the published paired-overlap rule.

Every output block carries n_anchors, chain_score (the objective, rounded to 4 decimals),
sum_cos (summed raw cosine), sum_w (summed log-odds, the V4 statistic) and a long table of
its anchor pairs in paired order (query row, target row).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from numba import njit, prange

BLOCK_KEYS = ["query_genome", "target_genome", "query_contig", "target_contig"]


@njit(cache=True, parallel=True)
def _dp_score_first(sq, st, w, cmp_vals, offsets, sizes, max_gap, gap_scale, dp_len, dp_prev, dp_score):
    n_groups = len(sizes)
    for g in prange(n_groups):
        off = offsets[g]
        n = sizes[g]
        for i in range(1, n):
            ii = off + i
            best_len = np.int32(1)
            best_prev = np.int64(-1)
            best_score = w[ii]
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
                if gap_scale > 0.0:
                    gg = gap_q if gap_q > gap_t else gap_t
                    cost = gap_scale * math.log2(gg + 1)
                else:
                    cost = 0.0
                new_score = dp_score[jj] + w[ii] - cost
                new_len = dp_len[jj] + np.int32(1)
                if new_score > best_score or (new_score == best_score and new_len > best_len):
                    best_score = new_score
                    best_len = new_len
                    best_prev = jj
            dp_len[ii] = best_len
            dp_prev[ii] = best_prev
            dp_score[ii] = best_score


@njit(cache=True)
def _trim_chains(chain_buf, chain_len_buf, chain_group_buf, n_chains, offsets,
                 sq, st, w, gap_scale, min_size, out_buf, out_len, out_group, out_score):
    pos = 0
    wpos = 0
    nout = 0
    for c in range(n_chains):
        L = chain_len_buf[c]
        off = offsets[chain_group_buf[c]]
        best = -1e300
        best_s = -1
        S = 0.0
        for i in range(L - 1, -1, -1):
            gi = off + chain_buf[pos + i]
            if i == L - 1:
                S = w[gi]
            else:
                gn = off + chain_buf[pos + i + 1]
                cost = 0.0
                if gap_scale > 0.0:
                    gq = sq[gn] - sq[gi] - 1
                    gt = abs(st[gn] - st[gi]) - 1
                    gg = gq if gq > gt else gt
                    cost = gap_scale * math.log2(gg + 1)
                S = w[gi] + S - cost
            if L - i >= min_size and S >= best:
                best = S
                best_s = i
        if best_s >= 0:
            for i in range(best_s, L):
                out_buf[wpos] = chain_buf[pos + i]
                wpos += 1
            out_len[nout] = L - best_s
            out_group[nout] = chain_group_buf[c]
            out_score[nout] = best
            nout += 1
        pos += L
    return nout, wpos


def group_anchors(numeric, qmeta, rmeta, group_type, dp_values):
    """GroupedAnchors exactly as run_spacedust_scaling.chain_numeric, with `similarity`
    replaced by dp_values (cosine for length_first, weights for score_first)."""
    q = qmeta.iloc[numeric.query_row_index.to_numpy(dtype=np.int64)]
    t = rmeta.iloc[numeric.target_row_index.to_numpy(dtype=np.int64)]
    anchors = {}
    for side, meta in [("query", q), ("target", t)]:
        anchors[side + "_idx"] = meta.position_index.to_numpy(dtype=np.int64)
        anchors[side + "_gene_id"] = meta.gene_id.to_numpy()
        for suffix, source in [("genome", "sample_id"), ("contig", "contig_id")]:
            anchors[side + "_" + suffix] = meta[source].to_numpy()
    anchors["similarity"] = np.asarray(dp_values)
    anchors["orientation"] = (q.strand.to_numpy() * t.strand.to_numpy()).astype(np.int8)
    codes, categories = [], []
    for name in ("query_genome", "target_genome", "query_contig", "target_contig"):
        values, names = pd.factorize(anchors[name], sort=True)
        codes.append(values.astype(np.int64))
        categories.append(np.asarray(names))
    order = np.lexsort(tuple(reversed(codes)))
    different = np.zeros(len(order), dtype=bool)
    different[0] = True
    for values in codes:
        different[1:] |= values[order[1:]] != values[order[:-1]]
    bounds = np.r_[np.flatnonzero(different), len(order)].astype(np.int64)
    ga = group_type(order=order, group_bounds=bounds, n_groups=len(bounds) - 1,
                    **{name: anchors[name] for name in ["query_idx", "target_idx", "similarity", "orientation",
                                                        "query_gene_id", "target_gene_id"]},
                    **dict(zip(("qg_codes", "tg_codes", "qc_codes", "tc_codes"), codes)),
                    **dict(zip(("qg_uniques", "tg_uniques", "qc_uniques", "tc_uniques"), categories)))
    return ga


def chain(numeric, qmeta, rmeta, chain_mod, group_type, *, mode, cos, weights,
          gap_scale=0.0, max_gap=2, min_size=2, verify_columns=False, want_anchors=True):
    """Return (blocks, block_anchors). `cos` and `weights` are aligned to numeric rows."""
    assert mode in ("length_first", "score_first")
    cos = np.asarray(cos, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(numeric) == 0:
        return pd.DataFrame(), pd.DataFrame()
    dp_values = cos if mode == "length_first" else weights
    ga = group_anchors(numeric, qmeta, rmeta, group_type, dp_values)
    group_sizes = np.diff(ga.group_bounds)
    valid_groups = np.where(group_sizes >= min_size)[0]
    vp = chain_mod._preprocess_vectorized(ga, valid_groups, min_size)
    if vp is None:
        return pd.DataFrame(), pd.DataFrame()
    sq, st, ss, cmp_vals = vp["flat_sq"], vp["flat_st"], vp["flat_ss"], vp["flat_cmp"]
    fidx, sizes, offsets = vp["flat_fidx"], vp["sizes"], vp["offsets"]
    part_gids, part_revs = vp["part_gids"], vp["part_revs"]
    total_n = vp["total_n"]
    dp_len = np.ones(total_n, dtype=np.int32)
    dp_prev = np.full(total_n, -1, dtype=np.int64)
    dp_score = ss.copy()
    max_gap = np.int64(max_gap)
    chain_buf = np.empty(total_n, dtype=np.int64)
    chain_group_buf = np.empty(total_n, dtype=np.int64)
    chain_len_buf = np.empty(total_n, dtype=np.int64)
    if mode == "length_first":
        dp = chain_mod._dp_fill_batched_parallel if total_n >= 100_000 else chain_mod._dp_fill_batched
        dp(sq, st, ss, cmp_vals, offsets, sizes, max_gap, float(gap_scale), dp_len, dp_prev, dp_score)
        n_chains, n_el = chain_mod._backtrack_batched(dp_len, dp_prev, dp_score, offsets, sizes,
                                                      np.int32(min_size), chain_buf, chain_group_buf, chain_len_buf)
    else:
        _dp_score_first(sq, st, ss, cmp_vals, offsets, sizes, max_gap, float(gap_scale), dp_len, dp_prev, dp_score)
        partition = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
        order = np.lexsort((-dp_len, -dp_score, partition))
        n_raw, n_el_raw = chain_mod._backtrack_batched_ordered(dp_len, dp_prev, offsets, sizes, np.int32(min_size),
                                                               chain_buf, chain_group_buf, chain_len_buf, order)
        out_buf = np.empty(max(n_el_raw, 1), dtype=np.int64)
        out_len = np.empty(max(n_raw, 1), dtype=np.int64)
        out_group = np.empty(max(n_raw, 1), dtype=np.int64)
        out_score = np.empty(max(n_raw, 1), dtype=np.float64)
        n_chains, n_el = _trim_chains(chain_buf, chain_len_buf, chain_group_buf, n_raw, offsets,
                                      sq, st, ss, float(gap_scale), min_size, out_buf, out_len, out_group, out_score)
        chain_buf, chain_len_buf, chain_group_buf = out_buf, out_len, out_group
        trimmed_scores = out_score[:n_chains]
    if n_chains == 0:
        return pd.DataFrame(), pd.DataFrame()
    # ---- extraction (mirrors chain._extract_blocks_flat) ---------------------------
    chain_lens = chain_len_buf[:n_chains].copy()
    chain_cum = np.zeros(n_chains + 1, dtype=np.int64)
    np.cumsum(chain_lens, out=chain_cum[1:])
    chain_part = chain_group_buf[:n_chains].astype(np.intp)
    all_glob = (np.repeat(offsets[chain_part], chain_lens) + chain_buf[:n_el]).astype(np.intp)
    cstarts = chain_cum[:n_chains].astype(np.intp)
    q_all, t_all = sq[all_glob], st[all_glob]
    q_mins, q_maxs = np.minimum.reduceat(q_all, cstarts), np.maximum.reduceat(q_all, cstarts)
    t_mins, t_maxs = np.minimum.reduceat(t_all, cstarts), np.maximum.reduceat(t_all, cstarts)
    rows_all = fidx[all_glob]
    sum_cos = np.add.reduceat(cos[rows_all], cstarts)
    sum_w = np.add.reduceat(weights[rows_all], cstarts)
    scores = np.add.reduceat(ss[all_glob], cstarts) if mode == "length_first" else trimmed_scores
    chain_gvi = part_gids[chain_part]
    chain_gi = valid_groups[chain_gvi]
    r0 = ga.order[ga.group_bounds[chain_gi].astype(np.intp)]
    chain_orient = np.where(part_revs[chain_part], np.int32(-1), np.int32(1))
    sort_idx = np.lexsort((-scores, chain_gvi))
    sorted_codes = chain_gvi[sort_idx]
    diffs = np.empty(n_chains, dtype=np.bool_)
    diffs[0] = True
    diffs[1:] = sorted_codes[1:] != sorted_codes[:-1]
    grp_starts = np.nonzero(diffs)[0]
    grp_ends = np.r_[grp_starts[1:], n_chains].astype(np.int64)
    keep = chain_mod._select_paired_overlap(sort_idx, grp_starts.astype(np.int64), grp_ends,
                                            q_mins, q_maxs, t_mins, t_maxs)
    sel = sort_idx[keep[sort_idx]]
    nb = len(sel)
    blocks = pd.DataFrame({
        "block_id": np.arange(nb, dtype=np.int64),
        "query_genome": ga.qg_uniques[ga.qg_codes[r0[sel]]].astype(str),
        "target_genome": ga.tg_uniques[ga.tg_codes[r0[sel]]].astype(str),
        "query_contig": ga.qc_uniques[ga.qc_codes[r0[sel]]].astype(str),
        "target_contig": ga.tc_uniques[ga.tc_codes[r0[sel]]].astype(str),
        "query_start": q_mins[sel].astype(np.int64), "query_end": q_maxs[sel].astype(np.int64),
        "target_start": t_mins[sel].astype(np.int64), "target_end": t_maxs[sel].astype(np.int64),
        "n_anchors": chain_lens[sel].astype(np.int64),
        "chain_score": np.round(scores[sel], 4),
        "orientation": chain_orient[sel],
        "sum_cos": sum_cos[sel], "sum_w": sum_w[sel],
    })
    block_anchors = pd.DataFrame()
    if want_anchors or verify_columns:
        lens = chain_lens[sel]
        starts = chain_cum[sel]
        idx = np.repeat(starts, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        rows = rows_all[idx]
        block_anchors = pd.DataFrame({
            "block_id": np.repeat(np.arange(nb, dtype=np.int64), lens),
            "query_row_index": numeric.query_row_index.to_numpy(np.int64)[rows],
            "target_row_index": numeric.target_row_index.to_numpy(np.int64)[rows],
            "anchor_row": rows,
        })
        if verify_columns:
            qi, ti = sq[all_glob][idx], st[all_glob][idx]
            bid = block_anchors.block_id.to_numpy()
            aq, at = [], []
            bounds = np.r_[0, np.cumsum(lens)]
            for b in range(nb):
                lo, hi = bounds[b], bounds[b + 1]
                aq.append(json.dumps(np.sort(qi[lo:hi]).tolist()))
                at.append(json.dumps(np.sort(ti[lo:hi]).tolist()))
            blocks["anchor_query_ids"] = aq
            blocks["anchor_target_ids"] = at
    return blocks, block_anchors
