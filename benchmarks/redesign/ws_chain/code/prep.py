#!/usr/bin/env python
"""Stage A for one model.

1. Pair-score statistics: per query, seed-list stats (top-10 mean, mean, sd) and background
   stats (mean, sd of cosine to a fixed panel of 2,048 random reference proteins, top 1%
   trimmed); per reference protein, the same background stats.
2. Pool sample: full candidate pools (originals + every proposal, no cutoff, no cap) for
   2,500 training and 1,000 held-out queries (numpy seed 17), observed order.
3. V0 replay: published context60 + published chainer, observed and permutation seeds
   17/18/19; require the saved counts; require that chain_v3 length_first reproduces the
   published blocks exactly on the observed anchors.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HELDOUT, PERM_SEEDS, S, SAVED_V0, W, array_paths, base, load_meta, load_published, log,
                    model_dirs, ordered_metadata, write_json)

PANEL, TRIM, N_TRAIN, N_HELD = 2048, 20, 2500, 1000


def background(mat, panel_vectors, workers, chunk=131072):
    import torch
    torch.set_num_threads(workers)
    P = torch.from_numpy(np.ascontiguousarray(panel_vectors, dtype=np.float32))
    n = len(mat)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    K = P.shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        X = np.asarray(mat[lo:hi], dtype=np.float32)
        finite = np.isfinite(X).all(axis=1)
        X = np.where(finite[:, None], X, 0).astype(np.float32)
        C = torch.from_numpy(X) @ P.T
        top = torch.topk(C, TRIM, dim=1).values.double()
        Cd = C.double()
        s1 = Cd.sum(1) - top.sum(1)
        s2 = (Cd * Cd).sum(1) - (top * top).sum(1)
        m = s1 / (K - TRIM)
        v = torch.clamp(s2 / (K - TRIM) - m * m, min=0)
        o = np.stack([m.numpy(), v.sqrt().numpy()], axis=1)
        o[~finite] = np.nan
        out[lo:hi] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--skip", default="", help="comma list of stages to skip: stats,pool,v0")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    import numba
    numba.set_num_threads(a.workers)
    import run_spacedust_scaling as rss
    from spacedust_prepared_extension_executor import extend_prepared_queries
    import chain_v3
    import extend_v2

    m = a.model
    work, runs = model_dirs(m)
    tmp = W / "tmp" / m
    tmp.mkdir(parents=True, exist_ok=True)
    report = dict(model=m, workers=a.workers)
    t0 = time.time()
    paths = array_paths(m)
    qmeta, rmeta = load_meta()
    qv = np.load(paths["query_vectors"], mmap_mode="r")
    rv = np.load(paths["reference_vectors"], mmap_mode="r")
    offs = np.load(paths["seed_offsets"])
    sc = np.load(paths["seed_scores"])

    # ---- 1. statistics ----------------------------------------------------------
    if "stats" not in skip:
        t = time.time()
        rng = np.random.default_rng(17)
        cand = np.sort(rng.choice(len(rv), size=2 * PANEL, replace=False))
        pv = np.asarray(rv[cand], dtype=np.float32)
        ok = np.isfinite(pv).all(axis=1)
        panel = cand[ok][:PANEL]
        assert len(panel) == PANEL
        pv = np.asarray(rv[panel], dtype=np.float32)
        qbg = background(qv, pv, a.workers)
        log(m, "query background done", f"{time.time()-t:.0f}s")
        rbg = background(rv, pv, a.workers)
        log(m, "reference background done", f"{time.time()-t:.0f}s")
        nq = len(qv)
        qst = np.full((nq, 5), np.nan)
        for q in range(nq):
            s = np.sort(sc[offs[q]:offs[q + 1]].astype(np.float64))[::-1]
            if len(s):
                qst[q, 0] = s[:10].mean()
                qst[q, 1] = s.mean()
                qst[q, 2] = s.std()
        qst[:, 3:] = qbg
        np.save(work / "query_stats.npy", qst)
        np.save(work / "reference_stats.npy", rbg.astype(np.float32))
        report["stats"] = dict(panel_rows=panel.tolist(), trim=TRIM, seconds=time.time() - t,
                               query_bg_mean_quantiles=np.nanquantile(qbg[:, 0], [.01, .5, .99]).tolist(),
                               query_bg_sd_quantiles=np.nanquantile(qbg[:, 1], [.01, .5, .99]).tolist(),
                               ref_bg_mean_quantiles=np.nanquantile(rbg[:, 0], [.01, .5, .99]).tolist(),
                               ref_bg_sd_quantiles=np.nanquantile(rbg[:, 1], [.01, .5, .99]).tolist(),
                               query_nan=int(np.isnan(qbg[:, 0]).sum()), ref_nan=int(np.isnan(rbg[:, 0]).sum()))
        log(m, "stats", {k: v for k, v in report["stats"].items() if k != "panel_rows"})

    # ---- 2. pool sample -------------------------------------------------------------
    if "pool" not in skip:
        t = time.time()
        eligible = np.isfinite(np.asarray(qv[:, 0])) & (np.diff(offs) > 0)
        held = qmeta.sample_id.isin(HELDOUT).to_numpy()
        rng = np.random.default_rng(17)
        tr = np.sort(rng.choice(np.flatnonzero(eligible & ~held), N_TRAIN, replace=False))
        ho = np.sort(rng.choice(np.flatnonzero(eligible & held), N_HELD, replace=False))
        rows = np.r_[tr, ho]
        pool, info = extend_v2.run("pool_dump", paths, {}, {}, rows, a.workers)
        pool["heldout"] = held[pool.query_row_index.to_numpy()]
        pool.to_parquet(work / "pool.parquet", index=False)
        pairs = pool[pool.kind == "pair"]
        report["pool"] = dict(train_queries=len(tr), heldout_queries=len(ho), rows=len(pool),
                              pairs=len(pairs), seeds=int(pairs.is_original.sum()),
                              proposals_not_original=int((~pairs.is_original).sum()),
                              occurrences=int((pool.kind == "occ").sum()), seconds=time.time() - t,
                              pool_run=info)
        log(m, "pool", report["pool"])

    # ---- 3. V0 replay ---------------------------------------------------------------------
    if "v0" not in skip:
        chain_mod, group_type = load_published()
        saved = pd.read_parquet(base(m) / "all_blocks.parquet")
        pp = pd.read_parquet(S / "results/permuted_panels" / f"{m}_permuted_blocks.parquet", columns=["seed"])
        saved_perm = [int((pp.seed == s).sum()) for s in PERM_SEEDS]
        rows = np.arange(len(qmeta), dtype=np.int64)
        v0 = dict(saved_observed=len(saved), saved_permuted=saved_perm, panels_permuted=saved_perm)
        if m in SAVED_V0:
            assert SAVED_V0[m][0] == len(saved) and SAVED_V0[m][1] == saved_perm, (SAVED_V0[m], len(saved), saved_perm)
        for tag in ["observed"] + [f"order_seed{s}" for s in PERM_SEEDS]:
            t = time.time()
            p = dict(paths)
            meta = qmeta
            if tag != "observed":
                meta = ordered_metadata(qmeta, int(tag[-2:]))
                np.save(tmp / f"query_neighbors_{tag}.npy", rss.neighbors(meta))
                p["query_neighbors"] = tmp / f"query_neighbors_{tag}.npy"
            ext = extend_prepared_queries(p, rows, workers=a.workers)
            numeric = ext.pop("anchors")
            te = time.time() - t
            blocks = rss.chain_numeric(numeric, meta, rmeta, chain_mod, group_type)
            tc = time.time() - t - te
            v0[tag] = dict(anchors=len(numeric), blocks=len(blocks), anchor_pairs=int(blocks.n_anchors.sum()),
                           extension_seconds=te, chain_seconds=tc)
            log(m, "V0", tag, v0[tag])
            if tag == "observed":
                assert len(blocks) == len(saved) and int(blocks.n_anchors.sum()) == int(saved.n_anchors.sum())
                mine, _ = chain_v3.chain(numeric, meta, rmeta, chain_mod, group_type, mode="length_first",
                                         cos=numeric.similarity.to_numpy(), weights=np.zeros(len(numeric)),
                                         verify_columns=True, want_anchors=False)
                keys = ["query_genome", "target_genome", "query_contig", "target_contig", "query_start",
                        "query_end", "target_start", "target_end", "n_anchors", "orientation",
                        "anchor_query_ids", "anchor_target_ids", "chain_score"]
                A = blocks[keys].sort_values(keys).reset_index(drop=True)
                B = mine[keys].sort_values(keys).reset_index(drop=True)
                A["orientation"] = A.orientation.astype(np.int64)
                B["orientation"] = B.orientation.astype(np.int64)
                pd.testing.assert_frame_equal(A, B, check_dtype=False, check_exact=True)
                v0["chain_v3_length_first_identical"] = True
                log(m, "chain_v3 length_first reproduces the published blocks exactly")
            else:
                i = PERM_SEEDS.index(int(tag[-2:]))
                assert len(blocks) == saved_perm[i], (tag, len(blocks), saved_perm[i])
        report["v0"] = v0
    report["seconds"] = time.time() - t0
    write_json(work / "prep_report.json", report)
    log("done", m, report["seconds"])


if __name__ == "__main__":
    main()
