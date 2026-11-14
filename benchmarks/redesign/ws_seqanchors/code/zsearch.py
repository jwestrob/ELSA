#!/usr/bin/env python
"""Exact top-60 inner-product search of all 67,148 queries against the whole reference.

Spaces: zca10 (esm2_650m, esmc600m), zlin256_t10_vis_inb (esmc600m), raw (control).
Vectors are read with one sequential block read per side (the metric workstream saw
stalled mmap reads on some nodes). Rows without a vector are left out of both sides.
Output: work/zseeds/<space>__<model>.parquet (query_row_index, target_row_index, similarity, rank).
Replay: on the metric workstream's held-out queries, the top-60 sets must equal its hits.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
W = RD / "ws_seqanchors"
V = RD / "ws_metric/vectors"
NREF, NQ, K = 4190383, 67148, 60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def spec(name):
    space, model = name.split("__")
    arrays = SCALE / "results" / model / "analysis/full_workflow/arrays"
    if space == "raw":
        return dict(q=(arrays / "query_vectors.npy", 0), r=(arrays / "reference_vectors.npy", 0),
                    metric=(model, "raw"))
    f = {"zca10": f"{model}__zca10.npy", "zlin256": f"{model}__zlin256_t10_vis_inb.npy"}[space]
    arm = {"zca10": "zca10", "zlin256": "zlin256_t10_vis_inb"}[space]
    return dict(q=(V / f, NREF), r=(V / f, 0), metric=(model, arm))


def header(path):
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(fh)
        assert not fortran
        return fh.tell(), shape, dtype


def read_rows(path, start, n):
    off, shape, dtype = header(path)
    d = shape[1]
    assert start + n <= shape[0], (path, start, n, shape)
    a = np.fromfile(path, dtype=dtype, count=n * d, offset=off + start * d * dtype.itemsize)
    return a.reshape(n, d)


def finite_rows(x):
    fin = np.isfinite(x[:, 0])
    sub = x[fin][:: max(1, int(fin.sum()) // 20000)]
    assert np.isfinite(sub).all(), "partially finite rows"
    if (~fin).any():
        assert np.isnan(x[~fin][:200]).all(), "non-finite rows must be all-NaN"
    return fin


def run(name, threads, smoke=0):
    import faiss
    faiss.omp_set_num_threads(threads)
    s = spec(name)
    rec = dict(name=name, threads=threads, faiss=faiss.__version__)
    t0 = time.time()
    nref = smoke or NREF
    ref = read_rows(s["r"][0], s["r"][1], nref)
    rec["read_reference_seconds"] = round(time.time() - t0, 1)
    fin_r = finite_rows(ref)
    ref_rows = np.flatnonzero(fin_r).astype(np.int64)
    if not fin_r.all():
        ref = np.ascontiguousarray(ref[fin_r])
    t1 = time.time()
    q = read_rows(s["q"][0], s["q"][1], NQ)
    if smoke:
        q = q[:500]
    fin_q = finite_rows(q)
    q_rows = np.flatnonzero(fin_q).astype(np.int64)
    q = np.ascontiguousarray(q[fin_q])
    rec.update(read_query_seconds=round(time.time() - t1, 1), dim=int(ref.shape[1]),
               reference_rows=int(len(ref_rows)), reference_rows_without_vector=int((~fin_r).sum()),
               query_rows=int(len(q_rows)), query_rows_without_vector=int((~fin_q).sum()))
    rng = np.random.default_rng(0)
    for side, x in (("reference", ref), ("query", q)):
        nrm = np.linalg.norm(x[rng.integers(0, len(x), 20000)], axis=1)
        rec[f"{side}_norm_min_max"] = [float(nrm.min()), float(nrm.max())]
        assert np.abs(nrm - 1).max() < 1e-3, (side, nrm.min(), nrm.max())
    log(name, rec)
    t2 = time.time()
    D = np.empty((len(q), K), np.float32)
    I = np.empty((len(q), K), np.int64)
    step = 8192
    for lo in range(0, len(q), step):
        d, i = faiss.knn(q[lo:lo + step], ref, K, metric=faiss.METRIC_INNER_PRODUCT)
        D[lo:lo + step], I[lo:lo + step] = d, i
        log(name, "queries", min(lo + step, len(q)), "of", len(q), f"{time.time() - t2:.0f}s")
    rec["search_seconds"] = round(time.time() - t2, 1)
    assert (I >= 0).all()
    hits = pd.DataFrame(dict(query_row_index=np.repeat(q_rows, K), target_row_index=ref_rows[I.ravel()],
                             similarity=D.ravel(), rank=np.tile(np.arange(1, K + 1, dtype=np.int16), len(q))))
    if smoke:
        rec.update(hits=int(len(hits)), smoke=smoke, head=hits.head(3).to_dict("records"))
        return rec
    out = W / "work/zseeds" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    hits.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    del ref
    rec["hits"] = int(len(hits))
    rec["similarity_quantiles"] = {str(p): float(np.quantile(D, p)) for p in (0, .01, .1, .5, .9, 1)}
    # all hits are cross-genome: query and reference genomes are disjoint
    u = pd.read_parquet(RD / "universe/genes.parquet", columns=["gid", "genome"])
    qg = set(u.genome.to_numpy()[NREF + q_rows])
    rg = set(u.genome.to_numpy()[np.unique(ref_rows[I.ravel()])])
    assert not (qg & rg)
    rec.update(replay(name, s, hits))
    rec["total_seconds"] = round(time.time() - t0, 1)
    return rec


def replay(name, s, hits):
    model, arm = s["metric"]
    p = RD / "ws_metric/full/eval" / model / "hits" / f"{arm}.parquet"
    m = pd.read_parquet(p)
    m = m[m.search == "all"]
    m = pd.DataFrame(dict(query_row_index=m.qgid.to_numpy() - NREF, target_row_index=m.tgid.to_numpy(),
                          score=m.score.to_numpy()))
    mine = hits[hits.query_row_index.isin(np.unique(m.query_row_index))]
    j = mine.merge(m, on=["query_row_index", "target_row_index"], how="outer", indicator=True)
    per = j.groupby("query_row_index")._merge.agg(lambda x: (x == "both").sum() / len(x))
    both = j[j._merge == "both"]
    r = dict(replay_source=str(p), replay_queries=int(per.size), replay_mean_jaccard=float(per.mean()),
             replay_min_jaccard=float(per.min()), replay_queries_identical=int((per == 1).sum()),
             replay_max_score_diff=float((both.similarity - both.score).abs().max()))
    log(name, "replay", r)
    assert r["replay_mean_jaccard"] >= 0.999 and r["replay_max_score_diff"] <= 1e-5, r
    if name.startswith("raw__"):
        seeds = pd.read_parquet(SCALE / "results" / model / "analysis/full_workflow/seeds.parquet",
                                columns=["query_row_index", "target_row_index"])
        k = seeds.merge(hits[["query_row_index", "target_row_index"]], how="inner")
        r.update(ann60_seeds=int(len(seeds)), ann60_seeds_in_exact_top60=int(len(k)),
                 exact_hits_below_085=int((hits.similarity < 0.85).sum()))
        log(name, "ANN60 recall", len(k) / len(seeds), "exact hits < 0.85", r["exact_hits_below_085"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    out = W / "results/zsearch"
    out.mkdir(parents=True, exist_ok=True)
    for name in a.names.split(","):
        rec = run(name, a.threads, a.smoke)
        if a.smoke:
            log("SMOKE", rec)
            continue
        rec["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
        log("DONE", name, rec["search_seconds"], "s")


if __name__ == "__main__":
    main()
