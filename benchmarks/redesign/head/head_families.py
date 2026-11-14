#!/usr/bin/env python
"""Workstream head, step 4: families from the chosen student's kNN for the recurrence engine.

Student exact kNN over unique sequences (k 50, self excluded; head_train.py) -> gene-level cross-genome partner lists
with a cap of 64 per gene (identical-sequence partners first at cosine 1, then kNN order; context_kernel 'expand') ->
undirected union -> Leiden modularity (n_iterations -1, seed 17), weight max(student cosine, 0), at each resolution,
one process per resolution sharing the graph by fork -> token arms $RD/engine/dpann_tokens/head__<cfg>__r<res>.parquet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
PKG = RD / "ws_dpann/package"
H = RD / "head"
TOK = RD / "engine/dpann_tokens"
CAP = 64
SEED = 17


def stamp(*a):
    print(dt.datetime.now().isoformat(timespec="seconds"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


@njit(parallel=True, cache=True)
def expand(uq, gcode, I, D, indptr, members, cap, out_b, out_s, n_avail):
    """As context_kernel.expand: own unique sequence first (cosine 1), then kNN order; other genomes only."""
    n = len(uq)
    for a in prange(n):
        U = uq[a]
        if U < 0:
            continue
        ga = gcode[a]
        k = 0
        avail = 0
        for t in range(indptr[U], indptr[U + 1]):
            b = members[t]
            if gcode[b] != ga:
                avail += 1
                if k < cap:
                    out_b[a, k] = b
                    out_s[a, k] = 1.0
                    k += 1
        for j in range(I.shape[1]):
            V = I[U, j]
            if V < 0:
                continue
            s = D[U, j]
            for t in range(indptr[V], indptr[V + 1]):
                b = members[t]
                if gcode[b] != ga:
                    avail += 1
                    if k < cap:
                        out_b[a, k] = b
                        out_s[a, k] = s
                        k += 1
        n_avail[a] = avail


_G = None
_W = None


def _leiden(res):
    import random
    import igraph as ig
    t = time.time()
    random.seed(SEED)
    ig.set_random_number_generator(random)
    part = _G.community_leiden(objective_function="modularity", weights=_W, resolution=res, n_iterations=-1)
    return res, np.asarray(part.membership, dtype=np.int64), round(time.time() - t, 1)


def token_stats(memb):
    sizes = np.bincount(np.unique(memb, return_inverse=True)[1])
    return dict(n_tokens=int(len(sizes)), singleton_tokens=int((sizes == 1).sum()), token_size_max=int(sizes.max()),
                token_size_q50_q90_q99=np.quantile(sizes, [0.5, 0.9, 0.99]).tolist())


def main():
    import igraph as ig
    import pyarrow as pa
    import pyarrow.parquet as pq
    global _G, _W
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="300,1000,3000,7500,11000,19000,30000,50000")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    a = ap.parse_args()
    t0 = time.time()
    sel = json.loads((H / "student/selection.json").read_text())
    pick = sel["selected"]
    short = pick.split("_")[0]
    z = np.load(H / "student" / pick / "knn_k50.npz")
    I, D = z["I"], z["D"]
    g = pq.read_table(PKG / "universe/genes.parquet", columns=["gid", "genome_code"])
    assert (g["gid"].to_numpy() == np.arange(g.num_rows)).all()
    gcode = g["genome_code"].to_numpy().astype(np.int64)
    uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    n = len(uq)
    rng = np.random.default_rng(SEED)
    order = np.lexsort((rng.random(n), uq))
    indptr = np.r_[0, np.cumsum(np.bincount(uq, minlength=I.shape[0]))].astype(np.int64)
    members = order.astype(np.int64)
    out_b = np.full((n, CAP), -1, dtype=np.int32)
    out_s = np.zeros((n, CAP), dtype=np.float32)
    n_avail = np.zeros(n, dtype=np.int64)
    expand(uq, gcode, I, D, indptr, members, CAP, out_b, out_s, n_avail)
    kept = (out_b >= 0).sum(1)
    info = dict(student=pick, cap=CAP, genes=n, directed_edges=int(kept.sum()),
                cap_binds_fraction=float((n_avail > CAP).mean()),
                available_partners_q10_q50_q90=np.quantile(n_avail, [.1, .5, .9]).tolist(),
                kept_partners_q10_q50_q90=np.quantile(kept, [.1, .5, .9]).tolist(),
                genes_without_partner=int((kept == 0).sum()))
    rows = np.repeat(np.arange(n, dtype=np.int64), CAP)
    b = out_b.ravel().astype(np.int64)
    s = out_s.ravel()
    m = b >= 0
    rows, b, s = rows[m], b[m], s[m]
    del out_b, out_s, m
    lo, hi = np.minimum(rows, b), np.maximum(rows, b)
    key = lo * n + hi
    del rows, b, lo, hi
    o = np.argsort(key, kind="stable")
    key, s = key[o], s[o]
    del o
    first = np.r_[True, key[1:] != key[:-1]]
    ukey, usim = key[first], s[first]
    del key, s, first
    ea, eb = (ukey // n).astype(np.int64), (ukey % n).astype(np.int64)
    del ukey
    info.update(undirected_edges=int(len(ea)), sim_q10_q50_q90=np.quantile(usim, [.1, .5, .9]).tolist(),
                negative_weight_edges=int((usim < 0).sum()), expand_union_seconds=round(time.time() - t0, 1))
    stamp("edges", json.dumps(info))
    t1 = time.time()
    _G = ig.Graph(n=n, edges=np.column_stack([ea, eb]), directed=False)
    _W = np.maximum(usim, 0).astype(np.float64)
    del ea, eb, usim
    info["graph_seconds"] = round(time.time() - t1, 1)
    stamp(f"graph built {info['graph_seconds']}s rss {rss_gb()}")
    res = [float(r) for r in a.res.split(",")]
    grid = []
    with mp.get_context("fork").Pool(min(len(res), a.procs)) as pool:
        for r, memb, secs in pool.imap_unordered(_leiden, res):
            _, tok = np.unique(memb, return_inverse=True)
            arm = f"head__{short}__r{r:g}"
            st = token_stats(memb)
            man = dict(arm=arm, source="head", student=pick, resolution=r, objective="modularity", n_iterations=-1,
                       seed=SEED, weight="max(student cosine, 0)", leiden_seconds=secs, rows=int(n), **st, edges=info,
                       code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       written=dt.datetime.now().isoformat(timespec="seconds"))
            tmp = TOK / f".{arm}.parquet.tmp{os.getpid()}"
            pq.write_table(pa.table({"gid": np.arange(n, dtype=np.int64), "token": tok.astype(np.int64)}), tmp,
                           compression="zstd")
            os.replace(tmp, TOK / f"{arm}.parquet")
            (TOK / f"{arm}.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
            grid.append(dict(arm=arm, resolution=r, leiden_seconds=secs, **st))
            stamp("leiden", json.dumps(grid[-1]))
    grid.sort(key=lambda x: x["resolution"])
    (H / "engine").mkdir(exist_ok=True)
    (H / "engine/families.json").write_text(json.dumps(dict(edges=info, grid=grid, seconds=round(time.time() - t0, 1)),
                                                       indent=1, default=float) + "\n")
    with open(H / "engine/arms.txt", "w") as fh:
        for x in grid:
            fh.write(x["arm"] + "\n")
    stamp("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
