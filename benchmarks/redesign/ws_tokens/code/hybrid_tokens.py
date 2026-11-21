#!/usr/bin/env python
"""Hybrid tokens: sequence families (MMseqs arm) merged by embedding similarity.

1. Base families = tokens of --base (default mmseqs__seq__id30c80).
2. Family centroid = L2-normalized mean of its members' transformed vectors
   (members without a vector are skipped; a family with none keeps its own token).
3. Mutual k-nearest-neighbour pairs among family centroids (faiss IVF, inner product).
4. Merge threshold tau from a null: cosines of 2,000,000 random pairs of family centroids
   (seeded). For each null tail probability p in --null-p, tau_p = the (1 - p) quantile;
   mutual pairs with cosine >= tau_p are joined by union-find (connected components).
One arm per p: hybrid__<model>__<transform>__mm30merge<p>.

--merge cpm (added after the union-find arms percolated into tokens of up to 266,762 genes):
the same thresholded mutual-kNN family graph is partitioned by igraph Leiden with the CPM
objective at resolution tau_p (edge weight = cosine). A group of families then needs mean
pairwise weight above tau_p, counting absent edges as 0, so chains cannot form.
Arms: hybrid__<model>__<transform>__mm30cpm<p>. The expensive part (centroids, null, kNN) is
cached in $RD/ws_tokens/work/hybrid/<model>__<transform>/edges.npz.
"""
import argparse
import os
import time

import duckdb
import faiss
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from common import NGENES, TOK, WS, peak_rss_gb, stamp, write_tokens
from graph_tokens import mutual_edges
from kmeans_tokens import load_vectors, transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transform", default="raw")
    ap.add_argument("--base", default="mmseqs__seq__id30c80")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--null-p", default="1e-3,1e-4,1e-5")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--merge", default="cc", choices=["cc", "cpm"])
    ap.add_argument("--cpm-res", default="", help="extra CPM runs on all mutual edges at these resolutions (arm tag r<res>)")
    a = ap.parse_args()
    threads = a.threads or int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
    faiss.omp_set_num_threads(threads)
    t0 = time.time()
    base = duckdb.connect().execute(f"SELECT token FROM read_parquet('{TOK}/{a.base}.parquet') ORDER BY gid").fetchnumpy()["token"]
    uniq, fam = np.unique(base, return_inverse=True)   # dense family id per gene
    F = len(uniq)
    ps = [float(p) for p in a.null_p.split(",") if p]
    cache = WS / "work" / "hybrid" / f"{a.model}__{a.transform}" / f"edges_k{a.k}_np{a.nprobe}_{a.base}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        es, ed, ew, fidx, null = z["es"], z["ed"], z["ew"], z["fidx"], z["null"]
        prov, tinfo = z["prov"].item(), z["tinfo"].item()
        taus = {p: float(np.quantile(null, 1 - p)) for p in ps}
        keep = np.ones(len(null), dtype=bool)
        t_base = time.time() - t0
        stamp("cached edges", len(es), "taus", taus)
    else:
        with threadpool_limits(threads):
            X, prov = load_vectors(a.model, "", threads)
            valid = np.empty(NGENES, dtype=bool)
            for s in range(0, NGENES, 200_000):
                valid[s:s + 200_000] = ~np.isnan(X[s:s + 200_000]).any(axis=1)
            tinfo = transform(X, valid, a.transform)
            d = X.shape[1]
            # family centroids: sparse (F x N) indicator times X, in row blocks
            vr = np.flatnonzero(valid)
            C = np.zeros((F, d), dtype=np.float32)
            for s in range(0, len(vr), 500_000):
                r = vr[s:s + 500_000]
                M = coo_matrix((np.ones(len(r), dtype=np.float32), (fam[r], np.arange(len(r)))), shape=(F, len(r))).tocsr()
                C += M @ X[r]
            del X
            has = np.linalg.norm(C, axis=1) > 0
            C[has] /= np.linalg.norm(C[has], axis=1, keepdims=True)
        fidx = np.flatnonzero(has)
        Cv = np.ascontiguousarray(C[fidx])
        stamp("families", F, "with vectors", len(fidx), f"{time.time() - t0:.0f}s", "rss", peak_rss_gb())
        rng = np.random.default_rng(a.seed)
        i, j = rng.integers(0, len(fidx), 2_000_000), rng.integers(0, len(fidx), 2_000_000)
        keep = i != j
        null = np.einsum("ij,ij->i", Cv[i[keep]], Cv[j[keep]])
        taus = {p: float(np.quantile(null, 1 - p)) for p in ps}
        stamp("null cos quantiles 0.5/0.99/0.999/0.9999", np.quantile(null, [0.5, 0.99, 0.999, 0.9999]).round(4), "taus", taus)
        quant = faiss.IndexFlatIP(d)
        km = faiss.Kmeans(d, a.nlist, niter=20, seed=a.seed, spherical=True, max_points_per_centroid=64)
        km.train(Cv[np.sort(rng.choice(len(Cv), min(len(Cv), a.nlist * 64), replace=False))])
        quant.add(km.centroids)
        index = faiss.IndexIVFFlat(quant, d, a.nlist, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True
        index.add(Cv)
        index.nprobe = a.nprobe
        D = np.empty((len(Cv), a.k + 1), dtype=np.float32)
        I = np.empty((len(Cv), a.k + 1), dtype=np.int64)
        for s in range(0, len(Cv), 200_000):
            D[s:s + 200_000], I[s:s + 200_000] = index.search(Cv[s:s + 200_000], a.k + 1)
        es, ed, ew = mutual_edges(I, D)
        t_base = time.time() - t0
        stamp("mutual family edges", len(es), "cos q50/q90/q99", np.quantile(ew, [0.5, 0.9, 0.99]).round(4), f"{t_base:.0f}s")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, es=es, ed=ed, ew=ew, fidx=fidx, null=null, prov=np.array(prov, dtype=object),
                 tinfo=np.array(tinfo, dtype=object))
    runs = [(f"{p:.0e}".replace("-0", "-").replace("e-", "em"), taus[p], taus[p], p) for p in ps]
    if a.cpm_res:
        assert a.merge == "cpm"
        runs += [("r" + f"{float(g):g}".replace(".", "p"), -1.0, float(g), None) for g in a.cpm_res.split(",")]
        if not a.null_p:
            runs = runs[len(ps):]
    for ptag, thr, res, p in runs:
        t1 = time.time()
        sel = ew >= thr
        if a.merge == "cc":
            A = coo_matrix((np.ones(int(sel.sum())), (es[sel], ed[sel])), shape=(len(fidx), len(fidx)))
            _, comp = connected_components(A, directed=False)
        else:
            import random
            import igraph as ig
            G = ig.Graph(n=len(fidx), edges=np.column_stack([es[sel], ed[sel]]).tolist(), directed=False)
            G.es["weight"] = ew[sel].astype(float).tolist()
            random.seed(a.seed)
            ig.set_random_number_generator(random)
            comp = np.asarray(G.community_leiden(objective_function="CPM", weights="weight", resolution=res,
                                                 n_iterations=-1).membership, dtype=np.int64)
        merged = np.arange(F, dtype=np.int64)          # families without vectors stay alone
        merged[fidx] = F + comp
        _, merged = np.unique(merged, return_inverse=True)
        tok = merged[fam].astype(np.int64)
        csize = np.bincount(comp)
        mtag = "mm30merge" if a.merge == "cc" else "mm30cpm"
        arm = f"hybrid__{a.model}__{a.transform}__{mtag}{ptag}"
        man = dict(source="hybrid", model=a.model, transform=a.transform, detail=f"{mtag}{ptag}", merge=a.merge,
                   method="MMseqs 30/80 families merged over mutual-kNN family-centroid pairs with cosine >= tau, "
                          "tau = (1 - p) quantile of random family-centroid pair cosines; merge=cc: union-find; "
                          "merge=cpm: Leiden CPM at resolution tau",
                   base=a.base, k=a.k, nlist=a.nlist, nprobe=a.nprobe, null_p=p, tau=thr, cpm_resolution=res,
                   null_pairs=int(keep.sum()), families=F, families_with_vector=int(len(fidx)),
                   merge_edges=int(sel.sum()), merged_groups_max_families=int(csize.max()),
                   provenance=prov, transform_info=tinfo, seconds_base=round(t_base, 1))
        write_tokens(arm, tok, man, t_base + time.time() - t1)
    stamp("end", f"{time.time() - t0:.0f}s", "rss", peak_rss_gb())


if __name__ == "__main__":
    main()
