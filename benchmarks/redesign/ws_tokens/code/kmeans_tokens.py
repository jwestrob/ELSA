#!/usr/bin/env python
"""Two-level spherical k-means tokens on protein embeddings, in gid order.

Coarse: faiss spherical k-means (--ncoarse cells) trained on a seeded sample, every
vector assigned by inner product. Fine: inside each cell, spherical Lloyd k-means with
k_c = max(1, round(K * n_c / N)) (capped at n_c). One coarse codebook serves every K in
--K; each arm is written to $RD/tokens/ the moment it is finished.

Transforms: raw (renormalized), globalcentred (subtract the mean of all vectors,
renormalize), genomecentred (subtract the mean vector of the protein's own genome,
renormalize). Proteins without a vector (NaN row) get token -1.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import duckdb
import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from common import NGENES, NREF, SCALE, UNIVERSE, WS, peak_rss_gb, stamp, write_tokens

BIG_CELL = 20000  # cells at least this large run one at a time with all BLAS threads


def load_vectors(model, vectors, threads):
    """Return (X float32 N x d in gid order, provenance dict)."""
    if vectors:
        X = np.load(vectors, mmap_mode="r")
        assert X.shape[0] == NGENES, X.shape
        out = np.empty(X.shape, dtype=np.float32)
        for s in range(0, NGENES, 500_000):
            out[s:s + 500_000] = X[s:s + 500_000]
        man = json.load(open(Path(vectors).with_suffix(".json"))) if Path(vectors).with_suffix(".json").exists() else {}
        return out, dict(vectors=str(vectors), vectors_manifest=man)
    p = json.load(open(SCALE / "results" / model / "analysis/full_workflow/protocol.json"))["vectors"]
    R = np.load(p["reference"]["path"], mmap_mode="r")
    Q = np.load(p["query"]["path"], mmap_mode="r")
    assert R.shape[0] == NREF and Q.shape[0] == NGENES - NREF and R.shape[1] == Q.shape[1]
    X = np.empty((NGENES, R.shape[1]), dtype=np.float32)
    for s in range(0, NREF, 500_000):
        X[s:min(s + 500_000, NREF)] = R[s:s + 500_000]
    X[NREF:] = Q[:]
    prov = {side: {k: p[side][k] for k in ("path", "sha256", "shape", "embedded_rows")} for side in ("reference", "query")}
    return X, dict(vectors=prov)


def normalize_inplace(X, valid, chunk=500_000):
    for s in range(0, X.shape[0], chunk):
        sl = slice(s, s + chunk)
        v = valid[sl]
        blk = X[sl]
        nrm = np.linalg.norm(blk, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        blk /= nrm
        blk[~v] = np.nan
        X[sl] = blk


def transform(X, valid, kind):
    info = {}
    if kind == "raw":
        pass
    elif kind == "globalcentred":
        mu = np.zeros(X.shape[1], dtype=np.float64)
        for s in range(0, X.shape[0], 500_000):
            blk = X[s:s + 500_000][valid[s:s + 500_000]]
            mu += blk.sum(0, dtype=np.float64)
        mu /= valid.sum()
        info["mean_norm"] = float(np.linalg.norm(mu))
        mu32 = mu.astype(np.float32)
        for s in range(0, X.shape[0], 500_000):
            X[s:s + 500_000] -= mu32
    elif kind == "genomecentred":
        g = duckdb.connect().execute(f"SELECT genome_code FROM read_parquet('{UNIVERSE}') ORDER BY gid").fetchnumpy()["genome_code"].astype(np.int64)
        G = int(g.max()) + 1
        sums = np.zeros((G, X.shape[1]), dtype=np.float64)
        cnt = np.zeros(G, dtype=np.int64)
        order = np.argsort(g, kind="stable")
        bounds = np.searchsorted(g[order], np.arange(G + 1))
        for c in range(G):
            idx = order[bounds[c]:bounds[c + 1]]
            idx = idx[valid[idx]]
            cnt[c] = len(idx)
            if len(idx):
                sums[c] = X[idx].sum(0, dtype=np.float64)
        means = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        info["genome_mean_norm_quantiles"] = np.quantile(np.linalg.norm(means, axis=1), [0, 0.5, 1]).round(4).tolist()
        for s in range(0, X.shape[0], 500_000):
            X[s:s + 500_000] -= means[g[s:s + 500_000]]
    else:
        raise ValueError(kind)
    normalize_inplace(X, valid)
    return info


def assign_ip(index, X, rows, chunk=200_000):
    lab = np.empty(len(rows), dtype=np.int64)
    sim = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        D, I = index.search(np.ascontiguousarray(X[rows[s:s + chunk]]), 1)
        lab[s:s + chunk] = I[:, 0]
        sim[s:s + chunk] = D[:, 0]
    return lab, sim


def _argmax_blocked(Xc, C):
    n, k = Xc.shape[0], C.shape[0]
    step = max(1, 16_000_000 // max(k, 1))
    a = np.empty(n, dtype=np.int64)
    best = np.empty(n, dtype=np.float32)
    for s in range(0, n, step):
        S = Xc[s:s + step] @ C.T
        a[s:s + step] = S.argmax(1)
        best[s:s + step] = S[np.arange(S.shape[0]), a[s:s + step]]
    return a, best


def kmeanspp(Xc, k, rng):
    """k-means++ seeding on the sphere (squared chord distance 2 - 2 cos)."""
    n = Xc.shape[0]
    idx = np.empty(k, dtype=np.int64)
    idx[0] = rng.integers(n)
    best = Xc @ Xc[idx[0]]
    u = rng.random(k)
    for j in range(1, k):
        d = np.maximum(2.0 - 2.0 * best, 0.0)
        cs = np.cumsum(d, dtype=np.float64)
        if cs[-1] <= 0:
            idx[j:] = rng.choice(n, k - j, replace=False)
            break
        idx[j] = min(int(np.searchsorted(cs, u[j] * cs[-1])), n - 1)
        np.maximum(best, Xc @ Xc[idx[j]], out=best)
    return Xc[idx].copy()


def sph_kmeans(Xc, k, seed, niter=25):
    """Spherical Lloyd k-means. Returns labels in [0, k) and the number of iterations."""
    n = Xc.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=np.int64), 0
    if k >= n:
        return np.arange(n, dtype=np.int64), 0
    rng = np.random.default_rng(seed)
    C = kmeanspp(Xc, k, rng)
    prev = None
    it = 0
    for it in range(1, niter + 1):
        a, best = _argmax_blocked(Xc, C)
        if prev is not None and np.array_equal(a, prev):
            break
        prev = a
        order = np.argsort(a, kind="stable")
        counts = np.bincount(a, minlength=k)
        nz = np.flatnonzero(counts)
        starts = np.concatenate([[0], np.cumsum(counts[nz])[:-1]])
        newC = np.zeros_like(C)
        newC[nz] = np.add.reduceat(Xc[order], starts, axis=0)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            far = np.argsort(best, kind="stable")[:len(empty)]
            newC[empty] = Xc[far]
        nrm = np.linalg.norm(newC, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        C = newC / nrm
    else:
        a, _ = _argmax_blocked(Xc, C)
    return a, it


_G = {}  # inherited by forked workers (copy-on-write): X, rows_sorted, bounds, kc, seed, niter


def _worker_init():
    threadpool_limits(1)


def _run_cells(cells):
    X, rows_sorted, bounds, kc = _G["X"], _G["rows_sorted"], _G["bounds"], _G["kc"]
    out = []
    for c in cells:
        Xc = np.ascontiguousarray(X[rows_sorted[bounds[c]:bounds[c + 1]]])
        a, it = sph_kmeans(Xc, int(kc[c]), _G["seed"] * 1_000_003 + c, _G["niter"])
        out.append((c, a.astype(np.int32), it))
    return out


def fine_stage(X, rows_sorted, bounds, K, n_valid, seed, threads, niter):
    """Per-cell spherical k-means in a fork-based process pool (the per-cell work is small,
    so threads are GIL-bound); cells >= BIG_CELL run in the parent with all BLAS threads."""
    ncell = len(bounds) - 1
    sizes = np.diff(bounds)
    kc = np.maximum(1, np.rint(K * sizes / n_valid)).astype(np.int64)
    kc = np.minimum(kc, np.maximum(sizes, 1))
    kc[sizes == 0] = 0
    offs = np.concatenate([[0], np.cumsum(kc)[:-1]])
    labels = np.empty(len(rows_sorted), dtype=np.int64)
    iters = np.zeros(ncell, dtype=np.int64)
    _G.update(X=X, rows_sorted=rows_sorted, bounds=bounds, kc=kc, seed=seed, niter=niter)
    order = [int(c) for c in np.argsort(-sizes, kind="stable") if sizes[c] > 0]
    big = [c for c in order if sizes[c] >= BIG_CELL]
    small = [c for c in order if sizes[c] < BIG_CELL]
    with threadpool_limits(threads):
        for c, a, it in _run_cells(big):
            labels[bounds[c]:bounds[c + 1]] = offs[c] + a
            iters[c] = it
    # batches of cells, largest first, about 8 batches per worker for load balance
    nb = max(1, threads * 8)
    batches = [small[i::nb] for i in range(nb)]
    batches = [b for b in batches if b]
    ctx = mp.get_context("fork")
    with ctx.Pool(threads, initializer=_worker_init) as pool:
        for res in pool.imap_unordered(_run_cells, batches):
            for c, a, it in res:
                labels[bounds[c]:bounds[c + 1]] = offs[c] + a
                iters[c] = it
    _G.clear()
    return labels, dict(k_total=int(kc.sum()), k_cell_quantiles=np.quantile(kc[sizes > 0], [0, 0.5, 0.9, 0.99, 1]).tolist(),
                        n_big_cells=len(big), iters_quantiles=np.quantile(iters[sizes > 0], [0, 0.5, 0.9, 1]).tolist(),
                        cells_at_iter_cap=int((iters >= niter).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transform", default="raw", choices=["raw", "globalcentred", "genomecentred"])
    ap.add_argument("--K", default="262144")
    ap.add_argument("--ncoarse", type=int, default=4096)
    ap.add_argument("--sample-per-cell", type=int, default=100)
    ap.add_argument("--coarse-iter", type=int, default=20)
    ap.add_argument("--fine-iter", type=int, default=25)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--vectors", default="", help="gid-ordered float32 .npy (ws_metric); overrides --model vectors")
    ap.add_argument("--name", default="", help="model field of the arm name when --vectors is used")
    a = ap.parse_args()
    threads = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
    faiss.omp_set_num_threads(threads)
    model_name = a.name or a.model
    tag = f"{model_name}__{a.transform}"
    work = WS / "work" / "kmeans" / tag
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stamp("start", tag, "threads", threads)

    with threadpool_limits(threads):
        X, prov = load_vectors(a.model, a.vectors, threads)
    valid = np.empty(NGENES, dtype=bool)
    for s in range(0, NGENES, 200_000):
        valid[s:s + 200_000] = ~np.isnan(X[s:s + 200_000]).any(axis=1)
    n_valid = int(valid.sum())
    t_load = time.time() - t0
    stamp("loaded", X.shape, "valid", n_valid, f"{t_load:.0f}s", "rss", peak_rss_gb())

    with threadpool_limits(threads):
        tinfo = transform(X, valid, a.transform)
    t_tr = time.time() - t0 - t_load
    stamp("transformed", tinfo, f"{t_tr:.0f}s")

    vrows = np.flatnonzero(valid)
    rng = np.random.default_rng(a.seed)
    ns = min(n_valid, a.ncoarse * a.sample_per_cell)
    samp = np.sort(rng.choice(vrows, ns, replace=False))
    km = faiss.Kmeans(X.shape[1], a.ncoarse, niter=a.coarse_iter, seed=a.seed, spherical=True,
                      verbose=False, max_points_per_centroid=a.sample_per_cell)
    km.train(np.ascontiguousarray(X[samp]))
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(km.centroids)
    coarse, csim = assign_ip(index, X, vrows)
    np.save(work / "coarse_centroids.npy", km.centroids)
    cfull = np.full(NGENES, -1, dtype=np.int32)
    cfull[vrows] = coarse
    np.save(work / "coarse_labels.npy", cfull)
    order = np.argsort(coarse, kind="stable")
    rows_sorted = vrows[order]
    bounds = np.searchsorted(coarse[order], np.arange(a.ncoarse + 1))
    sizes = np.diff(bounds)
    t_coarse = time.time() - t0 - t_load - t_tr
    cinfo = dict(ncoarse=a.ncoarse, sample=int(ns), coarse_iter=a.coarse_iter,
                 cell_size_quantiles=np.quantile(sizes, [0, 0.01, 0.5, 0.99, 1]).tolist(),
                 empty_cells=int((sizes == 0).sum()), mean_coarse_sim=float(csim.mean()))
    stamp("coarse", cinfo, f"{t_coarse:.0f}s")
    shared = time.time() - t0

    for K in [int(k) for k in a.K.split(",")]:
        arm = f"kmeans__{model_name}__{a.transform}__K{K}"
        t1 = time.time()
        lab, finfo = fine_stage(X, rows_sorted, bounds, K, n_valid, a.seed, threads, a.fine_iter)
        tok = np.full(NGENES, -1, dtype=np.int64)
        uniq, dense = np.unique(lab, return_inverse=True)
        tok[rows_sorted] = dense
        t_fine = time.time() - t1
        stamp(arm, finfo, f"fine {t_fine:.0f}s")
        man = dict(source="kmeans", model=model_name, transform=a.transform, detail=f"K{K}", K_target=K,
                   method="two-level spherical k-means: faiss coarse codebook, numpy Lloyd per cell, "
                          "k_c = max(1, round(K n_c / N)) capped at n_c; tokens renumbered densely over non-empty fine centroids",
                   seed=a.seed, fine_iter_max=a.fine_iter, n_valid=n_valid, provenance=prov,
                   transform_info=tinfo, coarse=cinfo, fine=finfo,
                   seconds_shared_load_transform_coarse=round(shared, 1), seconds_fine=round(t_fine, 1),
                   seconds_load=round(t_load, 1), coarse_work=str(work))
        write_tokens(arm, tok, man, shared + t_fine)
    stamp("end", f"{time.time() - t0:.0f}s", "rss", peak_rss_gb())


if __name__ == "__main__":
    main()
