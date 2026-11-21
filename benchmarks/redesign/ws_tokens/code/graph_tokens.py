#!/usr/bin/env python
"""Graph tokens: Leiden communities of a mutual-kNN cosine graph over all genes.

Vectors are loaded and transformed exactly as in kmeans_tokens.py. kNN by faiss IVF
(inner product; the coarse centroids of the matching k-means run are reused as the IVF
quantizer when present, so no retraining), k neighbours per gene excluding itself. An
edge is kept when each gene is in the other's k list (mutual kNN); weight = cosine.
Communities: igraph Leiden, modularity objective, one run per --resolution, seed fixed.
Genes without a vector get token -1; genes with no mutual edge become singleton tokens.
Run with the workstream venv ($RD/ws_tokens/env, igraph 0.11.8 + leidenalg 0.10.2).
"""
import argparse
import os
import time

import faiss
import igraph as ig
import numpy as np
from threadpoolctl import threadpool_limits

from common import NGENES, WS, peak_rss_gb, stamp, write_tokens
from kmeans_tokens import load_vectors, transform


def mutual_edges(I, D):
    """Undirected mutual-kNN edges (i < j) from neighbour lists I (positions, -1 = none)."""
    n = I.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), I.shape[1])
    dst = I.ravel().astype(np.int64)
    w = D.ravel()
    keep = (dst >= 0) & (dst != src)
    src, dst, w = src[keep], dst[keep], w[keep]
    key_f = src * n + dst
    key_r = dst * n + src
    key_sorted = np.sort(key_f)
    pos = np.minimum(np.searchsorted(key_sorted, key_r), len(key_sorted) - 1)
    sel = (key_sorted[pos] == key_r) & (src < dst)
    return src[sel], dst[sel], w[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transform", default="raw")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--resolution", default="1.0")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    threads = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
    faiss.omp_set_num_threads(threads)
    t0 = time.time()
    tag = f"{a.model}__{a.transform}"
    work = WS / "work" / "graph" / tag
    work.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(threads):
        X, prov = load_vectors(a.model, "", threads)
        valid = np.empty(NGENES, dtype=bool)
        for s in range(0, NGENES, 200_000):
            valid[s:s + 200_000] = ~np.isnan(X[s:s + 200_000]).any(axis=1)
        tinfo = transform(X, valid, a.transform)
    vrows = np.flatnonzero(valid)
    d = X.shape[1]
    knn_f = work / f"knn_k{a.k}_np{a.nprobe}.npz"
    if knn_f.exists():
        z = np.load(knn_f)
        I, D = z["I"], z["D"]
    else:
        cen = WS / "work" / "kmeans" / tag / "coarse_centroids.npy"
        quant = faiss.IndexFlatIP(d)
        C = np.load(cen)
        quant.add(C)
        index = faiss.IndexIVFFlat(quant, d, C.shape[0], faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True
        for s in range(0, len(vrows), 500_000):
            index.add(np.ascontiguousarray(X[vrows[s:s + 500_000]]))
        index.nprobe = a.nprobe
        I = np.empty((len(vrows), a.k + 1), dtype=np.int64)
        D = np.empty((len(vrows), a.k + 1), dtype=np.float32)
        for s in range(0, len(vrows), 200_000):
            D[s:s + 200_000], I[s:s + 200_000] = index.search(np.ascontiguousarray(X[vrows[s:s + 200_000]]), a.k + 1)
        np.savez(knn_f, I=I, D=D)
        del index
    t_knn = time.time() - t0
    stamp("knn", I.shape, f"{t_knn:.0f}s", "rss", peak_rss_gb())
    n = len(vrows)
    es, ed, ew = mutual_edges(I, D)
    deg = np.bincount(np.concatenate([es, ed]), minlength=n)
    ginfo = dict(nodes=int(n), mutual_edges=int(len(es)), isolated=int((deg == 0).sum()),
                 degree_mean=float(deg.mean()), edge_cos_quantiles=np.quantile(ew, [0.01, 0.5, 0.99]).round(4).tolist())
    stamp("graph", ginfo)
    G = ig.Graph(n=n, edges=np.column_stack([es, ed]).tolist(), directed=False)
    G.es["weight"] = ew.astype(float).tolist()
    t_graph = time.time() - t0
    for r in [float(x) for x in a.resolution.split(",")]:
        t1 = time.time()
        import random
        random.seed(a.seed)
        ig.set_random_number_generator(random)
        part = G.community_leiden(objective_function="modularity", weights="weight", resolution=r,
                                  n_iterations=-1)
        memb = np.asarray(part.membership, dtype=np.int64)
        tok = np.full(NGENES, -1, dtype=np.int64)
        tok[vrows] = memb
        rs = f"{r:g}".replace(".", "p")
        arm = f"graph__{a.model}__{a.transform}__mknn{a.k}r{rs}"
        t_leiden = time.time() - t1
        man = dict(source="graph", model=a.model, transform=a.transform, detail=f"mknn{a.k}r{rs}",
                   method="mutual kNN cosine graph (faiss IVFFlat, coarse k-means centroids as quantizer, "
                          "per-query probing), igraph Leiden modularity with edge weight = cosine",
                   k=a.k, nprobe=a.nprobe, resolution=r, seed=a.seed, provenance=prov, transform_info=tinfo,
                   graph=ginfo, modularity=float(part.modularity), seconds_knn=round(t_knn, 1),
                   seconds_graph=round(t_graph - t_knn, 1), seconds_leiden=round(t_leiden, 1),
                   igraph=ig.__version__)
        write_tokens(arm, tok, man, t_graph + t_leiden)
    stamp("end", f"{time.time() - t0:.0f}s", "rss", peak_rss_gb())


if __name__ == "__main__":
    main()
