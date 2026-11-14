#!/usr/bin/env python
"""Locus space, part 3: a map of every called locus (workstream ws_locusspace).

  lspace_map.py vectors --threads T --rep REP   locus vectors for every nested-reading (DH) locus (cores and
                                                neighbourhoods); REP = MEAN or SK_K<K>_r<r>
  lspace_map.py map     --threads T --rep REP   kNN graph over the core loci (rule H inside DH), Leiden at the
                                                resolution chosen on odd genomes, gold / census / dark-cluster tables
Predeclaration: predeclared.json part_3_map.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lspace as LS  # noqa: E402
from lspace import log, dump, OUT, RD, PKG, SEED, DH, HCORE  # noqa: E402

MAP = OUT / "map"
RES_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
K_NN = 15
CENSUS = RD / "cofold/s3/census"


@njit(parallel=True, cache=True)
def span_sums(X, s, e, out):
    for i in prange(s.shape[0]):
        for d in range(X.shape[1]):
            out[i, d] = 0.0
        for g in range(s[i], e[i] + 1):
            for d in range(X.shape[1]):
                out[i, d] += X[g, d]


def load_X():
    t0 = time.time()
    X = np.load(LS.ZROW)
    log(f"zca10 vectors in RAM {X.shape} {time.time() - t0:.0f}s")
    return X


def dh_table():
    D = pd.read_parquet(DH)
    H = pd.read_parquet(HCORE, columns=["gid_s", "gid_e"])
    key = D.gid_s.to_numpy() * 10_000_000 + D.gid_e.to_numpy()
    hk = set((H.gid_s.to_numpy() * 10_000_000 + H.gid_e.to_numpy()).tolist())
    D["is_core"] = [k in hk for k in key]
    D["span"] = D.gid_e - D.gid_s + 1
    return D.reset_index(drop=True)


# =========================================================================== vectors
def stage_vectors(a):
    import faiss
    faiss.omp_set_num_threads(a.threads)
    MAP.mkdir(parents=True, exist_ok=True)
    D = dh_table()
    log(f"DH loci {len(D):,} (cores {int(D.is_core.sum()):,}), span q50/q90/q99/max "
        f"{np.quantile(D.span, [.5, .9, .99, 1]).tolist()}")
    D.to_parquet(MAP / "dh_loci.parquet", index=False)
    X = load_X()
    t0 = time.time()
    if a.rep == "MEAN":
        V = np.empty((len(D), X.shape[1]), np.float32)
        span_sums(X, D.gid_s.to_numpy(np.int64), D.gid_e.to_numpy(np.int64), V)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        np.save(MAP / "V_MEAN.npy", V.astype(np.float16))
    else:
        import scipy.sparse as sp
        K, r = (int(x[1:]) for x in a.rep.split("_")[1:])
        z = np.load(OUT / "codebook" / f"K{K}.npz")
        cent, idf = z["cent"], z["idf"]
        ix = faiss.IndexFlatIP(cent.shape[1])
        ix.add(cent)
        I = np.empty((X.shape[0], r), np.int32)
        W = np.empty((X.shape[0], r), np.float32)
        for c0 in range(0, X.shape[0], 200_000):
            Dd, Ii = ix.search(X[c0:c0 + 200_000], r)
            w = np.exp(LS.SK_BETA * (Dd - Dd[:, :1]))
            w /= w.sum(1, keepdims=True)
            I[c0:c0 + 200_000] = Ii
            W[c0:c0 + 200_000] = w * idf[Ii]
        np.save(MAP / f"gene_I_{a.rep}.npy", I)
        np.save(MAP / f"gene_W_{a.rep}.npy", W)
        s, e = D.gid_s.to_numpy(np.int64), D.gid_e.to_numpy(np.int64)
        lens = e - s + 1
        rows = np.repeat(np.arange(len(D)), lens * r)
        g = np.concatenate([np.arange(x, y + 1) for x, y in zip(s, e)])
        V = sp.csr_matrix((W[g].ravel(), (rows, I[g].ravel().astype(np.int64))), shape=(len(D), K))
        V.sum_duplicates()
        nrm = np.sqrt(np.asarray(V.multiply(V).sum(1)).ravel())
        V = (sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ V).tocsr().astype(np.float32)
        sp.save_npz(MAP / f"V_{a.rep}.npz", V)
    log(f"locus vectors {a.rep}: {time.time() - t0:.0f}s")


def load_V(rep, rows=None):
    if rep in ("MEAN", "LZ", "LW"):
        V = np.load(MAP / f"V_{rep}.npy").astype(np.float32)
    else:
        import scipy.sparse as sp
        V = sp.load_npz(MAP / f"V_{rep}.npz").tocsr()
    return V if rows is None else V[rows]


# =========================================================================== kNN graph
def knn_dense(V, k, threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    ix = faiss.IndexFlatIP(V.shape[1])
    ix.add(V)
    n = V.shape[0]
    I = np.empty((n, k + 1), np.int64)
    S = np.empty((n, k + 1), np.float32)
    t0 = time.time()
    for c0 in range(0, n, 20_000):
        Ss, Ii = ix.search(V[c0:c0 + 20_000], k + 1)
        I[c0:c0 + 20_000], S[c0:c0 + 20_000] = Ii, Ss
        if (c0 // 20_000) % 5 == 0:
            log(f"  knn {c0 + 20_000:,}/{n:,} {time.time() - t0:.0f}s")
    return I, S


_SPK = {}


def _knn_sparse_block(c0):
    V, VT, k = _SPK["V"], _SPK["VT"], _SPK["k"]
    B = (V[c0:c0 + 2000] @ VT).tocsr()
    n = B.shape[0]
    I = np.full((n, k + 1), -1, np.int64)
    S = np.zeros((n, k + 1), np.float32)
    for i in range(n):
        a, b = B.indptr[i], B.indptr[i + 1]
        if b == a:
            continue
        idx, val = B.indices[a:b], B.data[a:b]
        kk = min(k + 1, len(idx))
        top = np.argpartition(-val, kk - 1)[:kk]
        o = top[np.argsort(-val[top], kind="stable")]
        I[i, :kk], S[i, :kk] = idx[o], val[o]
    return c0, I, S


def knn_sparse(V, k, threads):
    import multiprocessing as mp
    _SPK.update(V=V, VT=V.T.tocsc().tocsr() if False else V.T.tocsr(), k=k)
    n = V.shape[0]
    I = np.empty((n, k + 1), np.int64)
    S = np.empty((n, k + 1), np.float32)
    t0 = time.time()
    with mp.get_context("fork").Pool(threads) as pool:
        for j, (c0, Ii, Ss) in enumerate(pool.imap_unordered(_knn_sparse_block, range(0, n, 2000))):
            I[c0:c0 + len(Ii)], S[c0:c0 + len(Ii)] = Ii, Ss
            if j % 25 == 0:
                log(f"  sparse knn block {j} {time.time() - t0:.0f}s")
    return I, S


def graph_from_knn(I, S, k):
    import igraph as ig
    import scipy.sparse as sp
    n = I.shape[0]
    rows, cols, w = [], [], []
    for j in range(I.shape[1]):
        ok = (I[:, j] >= 0) & (I[:, j] != np.arange(n))
        rows.append(np.flatnonzero(ok))
        cols.append(I[ok, j])
        w.append(np.maximum(S[ok, j], 0) + 1e-12)
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    # keep the k best non-self neighbours per row
    A = sp.csr_matrix((w, (r, c)), shape=(n, n))
    A = A.maximum(A.T).tocoo()
    m = A.row < A.col
    g = ig.Graph(n=n, edges=np.c_[A.row[m], A.col[m]].tolist(), edge_attrs={"weight": A.data[m].tolist()})
    return g


_LG = {}


def _leiden_task(res):
    return res, LS.leiden(_LG["g"], res)


# =========================================================================== gold / annotation context
def gold_members():
    I = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type",
                                                                   "system_subtype", "genome", "member_gids"])
    return I


def best_core_for_gold(U, I, C):
    """For each >= 2-member clean gold instance: the core locus (row of C) with the best member F1, coverage >= 0.5."""
    cc = U.cc[C.gid_s.to_numpy()]
    order = np.argsort(cc, kind="stable")
    cc_s = cc[order]
    rows = []
    for k, mem in enumerate(I.mem):
        c = int(U.cc[mem[0]])
        a, b = np.searchsorted(cc_s, c, "left"), np.searchsorted(cc_s, c, "right")
        best = None
        for r in order[a:b]:
            s, e = int(C.gid_s.iat[r]), int(C.gid_e.iat[r])
            ov = int(np.searchsorted(mem, e, "right") - np.searchsorted(mem, s, "left"))
            if ov == 0:
                continue
            cov, prec = ov / len(mem), ov / (e - s + 1)
            f1 = 2 * cov * prec / (cov + prec)
            key = (f1, -(e - s), -s)
            if best is None or key > best[0]:
                best = (key, r, cov, f1)
        if best is not None and best[2] >= 0.5:
            rows.append((k, best[1], best[2], best[3]))
    return pd.DataFrame(rows, columns=["gi", "core_row", "cov", "f1"])


def stage_map(a):
    import faiss  # noqa: F401
    import numba
    numba.set_num_threads(a.threads)
    T0 = time.time()
    res = dict(stage="map", rep=a.rep)
    D = pd.read_parquet(MAP / "dh_loci.parquet")
    core_rows = np.flatnonzero((D.is_core & (D.span >= 2)).to_numpy())
    C = D.iloc[core_rows].reset_index(drop=True)
    res["n_cores"] = int(len(C))
    log(f"core loci (span >= 2): {len(C):,}")
    U = LS.Universe(vectors=False)
    t0 = time.time()
    if a.reuse_knn and (MAP / f"knn_I_{a.rep}.npy").exists():
        I = np.load(MAP / f"knn_I_{a.rep}.npy").astype(np.int64)
        S = np.load(MAP / f"knn_S_{a.rep}.npy").astype(np.float32)
        log("kNN reused")
    else:
        V = load_V(a.rep, core_rows)
        if a.rep in ("MEAN", "LZ", "LW"):
            I, S = knn_dense(V, K_NN, a.threads)
        else:
            I, S = knn_sparse(V, K_NN, a.threads)
        np.save(MAP / f"knn_I_{a.rep}.npy", I.astype(np.int32))
        np.save(MAP / f"knn_S_{a.rep}.npy", S.astype(np.float16))
        del V
    res["seconds_knn"] = time.time() - t0
    log(f"kNN {time.time() - t0:.0f}s")
    RES = tuple(float(x) for x in a.grid.split("/")) if a.grid else RES_GRID
    t0 = time.time()
    g = graph_from_knn(I, S, K_NN)
    res["graph"] = dict(nodes=g.vcount(), edges=g.ecount(), seconds=time.time() - t0)
    log(f"graph {g.vcount():,} nodes {g.ecount():,} edges")
    # gold labels on cores
    Gd = LS.load_gold(U)
    B = best_core_for_gold(U, Gd, C)
    B["key"] = Gd.key.to_numpy()[B.gi]
    B["hikey"] = Gd.hikey.to_numpy()[B.gi]
    B["split"] = Gd.split.to_numpy()[B.gi]
    B["genome_code"] = Gd.genome_code.to_numpy()[B.gi]
    B["g075"] = Gd.g075.to_numpy()[B.gi]
    B["caller"] = Gd.caller.to_numpy()[B.gi]
    B = B.sort_values(["core_row", "f1"], ascending=[True, False]).drop_duplicates("core_row").reset_index(drop=True)
    res["gold_cores"] = B.groupby(["caller", "split"]).size().reset_index().values.tolist()
    # Leiden grid (parallel over resolutions)
    import multiprocessing as mp
    _LG["g"] = g
    t0 = time.time()
    parts = {}
    prev = MAP / f"cores_clusters_{a.rep}.parquet"
    if a.reuse_knn and prev.exists():
        P0 = pd.read_parquet(prev)
        for col in P0.columns:
            if col.startswith("cl_r"):
                parts[float(col[4:])] = P0[col].to_numpy()
        log(f"reused partitions {sorted(parts)}")
    todo = [r_ for r_ in RES if r_ not in parts]
    with mp.get_context("fork").Pool(max(1, min(a.threads, len(todo)))) as pool:
        for r_, memb in pool.imap_unordered(_leiden_task, todo):
            parts[r_] = memb
            log(f"  leiden res {r_}: {len(np.unique(memb)):,} clusters ({time.time() - t0:.0f}s)")
    lab = pd.factorize(B.key)[0]
    hil = pd.factorize(B.hikey)[0]
    grp = B.g075.to_numpy()
    gcode = B.genome_code.to_numpy()
    odd = np.flatnonzero(B.split.eq("odd").to_numpy())
    even = np.flatnonzero(B.split.eq("even").to_numpy())
    grid = {}
    for r_, memb in sorted(parts.items()):
        cl = memb[B.core_row.to_numpy()]
        grid[r_] = dict(clusters=int(len(np.unique(memb))), odd=LS.cluster_metrics(lab, cl, grp, odd),
                        odd_hi=LS.cluster_metrics(hil, cl, grp, odd))
    best_r = max(grid, key=lambda r_: grid[r_]["odd"]["ami"])
    res["resolution_grid"] = grid
    res["resolution"] = best_r
    log(f"selected resolution {best_r} (odd AMI {grid[best_r]['odd']['ami']:.3f})")
    memb = parts[best_r]
    cl = memb[B.core_row.to_numpy()]
    ev = LS.cluster_metrics(lab, cl, grp, even)
    ev["ci"] = LS.boot_ami(lab, cl, gcode, even)
    res["even"] = ev
    res["even_hi"] = LS.cluster_metrics(hil, cl, grp, even)
    # engine module labels on the same cores, same metrics (the token baseline partition)
    mcl = C.module_id.to_numpy()[B.core_row.to_numpy()]
    res["even_module_labels"] = LS.cluster_metrics(lab, mcl, grp, even)
    res["even_module_labels"]["ci"] = LS.boot_ami(lab, mcl, gcode, even)
    res["even_module_labels_hi"] = LS.cluster_metrics(hil, mcl, grp, even)
    # hierarchy: coarser resolution (best / 10 if on the grid, else the smallest)
    coarse = [r_ for r_ in sorted(parts) if r_ < best_r]
    if coarse:
        rc = coarse[0] if best_r / 10 <= coarse[0] else max(r_ for r_ in coarse if r_ <= best_r / 10)
        clc = parts[rc][B.core_row.to_numpy()]
        res["coarse"] = dict(resolution=rc, even_label=LS.cluster_metrics(lab, clc, grp, even),
                             even_hi=LS.cluster_metrics(hil, clc, grp, even))
    C["cluster"] = memb
    for r_ in sorted(parts):
        C[f"cl_r{r_:g}"] = parts[r_]
    C.to_parquet(MAP / f"cores_clusters_{a.rep}.parquet", index=False)
    B.to_parquet(MAP / f"gold_cores_{a.rep}.parquet", index=False)
    res["seconds"] = time.time() - T0
    dump(res, MAP / f"map_{a.rep}.json")
    log(f"map stage {res['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vectors", "map"])
    ap.add_argument("--rep", default="MEAN")
    ap.add_argument("--grid", default="", help="resolutions separated by '/', e.g. 100/200/500/1000")
    ap.add_argument("--reuse-knn", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 4)))
    a = ap.parse_args()
    {"vectors": stage_vectors, "map": stage_map}[a.stage](a)


if __name__ == "__main__":
    main()
