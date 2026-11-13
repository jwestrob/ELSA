#!/usr/bin/env python
"""Stage-2 alphabet: a vocabulary tree of whitened ESM++ small vectors (no families, no taxonomy).

Unique sequences (uid rows of the embed_queue run), zca10 transform exactly as softrec (embed_tokens.transform),
unit-normalised. Nested spherical k-means: level 1 k = K1 over all sequences; each level-1 cell split into up to B2
cells; each level-2 cell into up to B3 cells (a cell of n sequences gets min(B, n // 4) children, at least 1).
A unit is a cell; its prototype is the normalised mean of its members (the k-means centroid). Every gene carries one
unit per level (coarse -> fine); genes without a vector get -1.
Writes $RD/vocab/units/<tag>/: labels_gene.npy (n_genes x 3 int64, global ids, level offsets), labels_uid.npy,
centroids_l{1,2,3}.npy, units.json (counts, sizes, timings, and descriptive agreement with MMseqs2 rescue3 families).
usage: units.py --tag vt1k --k1 1024 --b2 16 --b3 8 --threads N
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
VD = RD / "vocab"
sys.path.insert(0, str(RD / "softrec/code"))
from embed_tokens import load_run, transform  # noqa: E402  (read-only import)

X = None  # shared with forked workers


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kmeans(P, k, seed, niter=20):
    import faiss
    d = P.shape[1]
    if k <= 1 or len(P) <= k:
        c = P.mean(0, keepdims=True) if k <= 1 else P.copy()
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        lab = np.zeros(len(P), dtype=np.int64) if k <= 1 else np.arange(len(P))
        return c.astype(np.float32), lab
    km = faiss.Kmeans(d, k, niter=niter, seed=seed, spherical=True, min_points_per_centroid=1,
                      max_points_per_centroid=1_000_000_000, verbose=False)
    km.train(np.ascontiguousarray(P))
    idx = faiss.IndexFlatIP(d)
    idx.add(km.centroids)
    _, I = idx.search(np.ascontiguousarray(P), 1)
    lab = I[:, 0].astype(np.int64)
    # recompute centroids as normalised means of final members (empty cells dropped below)
    c = np.zeros((k, d), dtype=np.float64)
    np.add.at(c, lab, P)
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c.astype(np.float32), lab


def split_cell(args):
    """Split one parent cell (member row indices) into up to b children. Returns (rows, child labels, centroids)."""
    import faiss
    faiss.omp_set_num_threads(1)
    rows, b, seed = args
    k = int(min(b, max(1, len(rows) // 4)))
    c, lab = kmeans(X[rows], k, seed)
    used = np.unique(lab)
    remap = np.full(len(c), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return rows, remap[lab], c[used]


def split_level(parent, b, procs, seed):
    import multiprocessing as mp
    order = np.argsort(parent, kind="stable")
    ps = parent[order]
    cuts = np.flatnonzero(np.diff(ps)) + 1
    groups = np.split(order, cuts)
    jobs = [(g, b, seed + i) for i, g in enumerate(groups)]
    jobs.sort(key=lambda j: -len(j[0]))
    child = np.full(len(parent), -1, dtype=np.int64)
    cents = []
    base = 0
    with mp.get_context("fork").Pool(procs) as pool:
        for rows, lab, c in pool.imap_unordered(split_cell, jobs, chunksize=4):
            child[rows] = base + lab
            cents.append((base, c))
            base += len(c)
    C = np.zeros((base, X.shape[1]), dtype=np.float32)
    for b0, c in cents:
        C[b0:b0 + len(c)] = c
    return child, C


def main():
    global X
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vt1k")
    ap.add_argument("--k1", type=int, default=1024)
    ap.add_argument("--b2", type=int, default=16)
    ap.add_argument("--b3", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    out = VD / "units" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xall, u, valid, cnt = load_run(EMB)
    info = transform(Xall, valid, cnt, "zca10")
    log("vectors", Xall.shape, "valid", int(valid.sum()), "genes", len(u), info, f"{time.time()-t0:.0f}s")
    vrows = np.flatnonzero(valid & (cnt > 0))
    X = Xall[vrows]
    del Xall
    # level 1
    rng = np.random.default_rng(a.seed)
    samp = np.sort(rng.choice(len(X), min(len(X), a.k1 * 512), replace=False))
    km = faiss.Kmeans(X.shape[1], a.k1, niter=25, seed=a.seed, spherical=True, max_points_per_centroid=512, verbose=False)
    km.train(np.ascontiguousarray(X[samp]))
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(km.centroids)
    l1 = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), 200_000):
        l1[s:s + 200_000] = idx.search(np.ascontiguousarray(X[s:s + 200_000]), 1)[1][:, 0]
    used = np.unique(l1)
    remap = np.full(a.k1, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    l1 = remap[l1]
    C1 = np.zeros((len(used), X.shape[1]), dtype=np.float64)
    np.add.at(C1, l1, X)
    C1 = (C1 / np.maximum(np.linalg.norm(C1, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    t1 = time.time() - t0
    log("level 1", len(used), f"{t1:.0f}s")
    faiss.omp_set_num_threads(1)
    l2, C2 = split_level(l1, a.b2, a.threads, a.seed + 1000)
    t2 = time.time() - t0
    log("level 2", len(C2), f"{t2:.0f}s")
    l3, C3 = split_level(l2, a.b3, a.threads, a.seed + 100000)
    t3 = time.time() - t0
    log("level 3", len(C3), f"{t3:.0f}s")
    n1, n2 = len(C1), len(C2)
    lab_uid = np.full((len(valid), 3), -1, dtype=np.int64)
    lab_uid[vrows, 0] = l1
    lab_uid[vrows, 1] = n1 + l2
    lab_uid[vrows, 2] = n1 + n2 + l3
    lab_gene = np.full((len(u), 3), -1, dtype=np.int64)
    ok = u >= 0
    lab_gene[ok] = lab_uid[u[ok]]
    np.save(out / "labels_uid.npy", lab_uid)
    np.save(out / "labels_gene.npy", lab_gene)
    np.save(out / "centroids_l1.npy", C1)
    np.save(out / "centroids_l2.npy", C2)
    np.save(out / "centroids_l3.npy", C3)
    # within-cell cosine to own prototype (radius summary)
    rad = {}
    for lev, (lab, C) in enumerate(((l1, C1), (l2, C2), (l3, C3)), start=1):
        cs = np.empty(len(X), dtype=np.float32)
        for s in range(0, len(X), 200_000):
            cs[s:s + 200_000] = (X[s:s + 200_000] * C[lab[s:s + 200_000]]).sum(1)
        sizes = np.bincount(lab)
        rad[f"l{lev}"] = dict(units=int(len(C)), size_q=np.quantile(sizes, [.1, .5, .9, .99, 1]).tolist(),
                              member_cos_q=np.quantile(cs, [.05, .25, .5, .75]).tolist())
    # descriptive agreement with MMseqs2 rescue3 families (evaluation only)
    import pandas as pd
    from sklearn.metrics import homogeneity_completeness_v_measure
    t = pd.read_parquet(RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet")
    tok = np.full(len(u), -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    agree = {}
    m = lab_gene[:, 0] >= 0
    for lev in range(3):
        h, c, v = homogeneity_completeness_v_measure(tok[m], lab_gene[m, lev])
        agree[f"l{lev+1}"] = dict(homogeneity_vs_family=float(h), completeness_vs_family=float(c), v=float(v))
    res = dict(tag=a.tag, k1=a.k1, b2=a.b2, b3=a.b3, sequences=int(len(X)), genes=int(len(u)),
               genes_with_vector=int(m.sum()), transform=info, levels=rad, agreement_with_rescue3=agree,
               seconds=dict(level1=round(t1, 1), level2=round(t2, 1), level3=round(t3, 1), total=round(time.time() - t0, 1)))
    with open(out / "units.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
