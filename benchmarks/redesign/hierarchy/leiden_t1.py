#!/usr/bin/env python
"""Task 1 Leiden runs (workstream hierarchy).

graphs
  v0      the context_kernel candidate graph: 164,863,660 undirected gene-level edges (ea, eb), weight = cosine
          clipped at 0; nodes = genes (gid order). Identical to context_kernel.py families --edge-set all, weights sim.
  mutual  v0 restricted to its mutual edges (edges/mutual.npy).
  g1      the round-1 graph: mutual 20-NN of unique sequences from the IVF k50 kNN file, weight = cosine; nodes =
          unique sequences; a gene's token is the community of its sequence. Identical to lab embed_tokens.py graph.
Each task = (resolution, seed): Leiden modularity, n_iterations -1, python `random` seeded as in the earlier code.
Writes $RD/hierarchy/families/<graph>_r<res>_s<seed>.npy (node memberships, int32), a grid TSV, and a gene-level
token arm $RD/engine/dpann_tokens/hier__<graph>__r<res>_s<seed>.parquet (+ .json).
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

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
EDGES = RD / "context_kernel/edges"
KNN_IVF = RD / "lab/knn/knn__esmpp_small__zca10__k50.npz"
FAM = RD / "hierarchy/families"
TOK = RD / "engine/dpann_tokens"


def stamp(*a):
    print(dt.datetime.now().isoformat(timespec="seconds"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)


def mutual_edges(I, D):
    """Copy of lab/code/embed_tokens.py mutual_edges (round-1 graph)."""
    n = I.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), I.shape[1])
    dst = I.ravel().astype(np.int64)
    w = D.ravel()
    keep = (dst >= 0) & (dst != src)
    src, dst, w = src[keep], dst[keep], w[keep]
    key_f = np.sort(src * n + dst)
    key_r = dst * n + src
    pos = np.minimum(np.searchsorted(key_f, key_r), len(key_f) - 1)
    sel = (key_f[pos] == key_r) & (src < dst)
    return src[sel], dst[sel], w[sel]


_G = None
_W = None
_GN = None


def _leiden(task):
    import random
    import igraph as ig
    res, seed = task
    t = time.time()
    random.seed(seed)
    ig.set_random_number_generator(random)
    if _GN == "g1":
        part = _G.community_leiden(objective_function="modularity", weights="weight", resolution=res, n_iterations=-1)
    else:
        part = _G.community_leiden(objective_function="modularity", weights=_W, resolution=res, n_iterations=-1)
    return res, seed, np.asarray(part.membership, dtype=np.int64), round(time.time() - t, 1)


def gene_tokens(memb, u):
    """Gene-level tokens 0..T-1 (np.unique inverse, as context_kernel cmd_arms); -1 for genes without a node."""
    if u is None:
        _, tok = np.unique(memb, return_inverse=True)
        return tok.astype(np.int64)
    comp = np.where(u >= 0, memb[np.maximum(u, 0)], -1)
    tok = np.full(len(u), -1, np.int64)
    ok = comp >= 0
    _, tok[ok] = np.unique(comp[ok], return_inverse=True)
    return tok


def write_arm(arm, tok, man):
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert arm.startswith("hier__")
    tmp = TOK / f".{arm}.parquet.tmp{os.getpid()}"
    pq.write_table(pa.table({"gid": np.arange(len(tok), dtype=np.int64), "token": tok}), tmp, compression="zstd")
    os.replace(tmp, TOK / f"{arm}.parquet")
    tj = TOK / f".{arm}.json.tmp{os.getpid()}"
    tj.write_text(json.dumps(man, indent=1, default=float) + "\n")
    os.replace(tj, TOK / f"{arm}.json")


def main():
    global _G, _W, _GN
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=("v0", "mutual", "g1"), required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated res:seed")
    ap.add_argument("--procs", type=int, required=True)
    a = ap.parse_args()
    import igraph as ig
    t0 = time.time()
    _GN = a.graph
    FAM.mkdir(parents=True, exist_ok=True)
    u = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    n_genes = len(u)
    if a.graph in ("v0", "mutual"):
        ea = np.load(EDGES / "ea.npy").astype(np.int64)
        eb = np.load(EDGES / "eb.npy").astype(np.int64)
        sim = np.maximum(np.load(EDGES / "sim.npy").astype(np.float64), 0)
        if a.graph == "mutual":
            m = np.load(EDGES / "mutual.npy").astype(bool)
            ea, eb, sim = ea[m], eb[m], sim[m]
        _W = sim
        _G = ig.Graph(n=n_genes, edges=np.column_stack([ea, eb]), directed=False)
        node_u = None
        info = dict(edges_file=str(EDGES), n_nodes=n_genes, n_edges=int(_G.ecount()))
        del ea, eb
    else:
        z = np.load(KNN_IVF, allow_pickle=True)
        I, D = z["I"][:, :20], z["D"][:, :20]
        es, ed, ew = mutual_edges(I, D)
        _G = ig.Graph(n=I.shape[0], edges=np.column_stack([es, ed]).tolist(), directed=False)
        _G.es["weight"] = ew.astype(float).tolist()
        node_u = u
        info = dict(knn=str(KNN_IVF), k=20, n_nodes=int(I.shape[0]), n_edges=int(len(es)))
        del I, D, z
    stamp(f"graph {a.graph}: {info} built {time.time() - t0:.0f}s rss {rss_gb()}")
    tasks = [(float(x.split(":")[0]), int(x.split(":")[1])) for x in a.tasks.split(",")]
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    with mp.get_context("fork").Pool(min(a.procs, len(tasks))) as pool:
        for res, seed, memb, secs in pool.imap_unordered(_leiden, tasks):
            tag = f"{a.graph}_r{res:g}_s{seed}"
            np.save(FAM / f"{tag}.npy", memb.astype(np.int32))
            tok = gene_tokens(memb, node_u)
            sizes = np.bincount(tok[tok >= 0])
            row = dict(tag=tag, graph=a.graph, resolution=res, seed=seed, leiden_seconds=secs, n_tokens=int(len(sizes)),
                       singleton_tokens=int((sizes == 1).sum()), token_size_max=int(sizes.max()),
                       genes_assigned=int((tok >= 0).sum()), **info)
            arm = f"hier__{a.graph}__r{res:g}_s{seed}"
            write_arm(arm, tok, dict(row, arm=arm, objective="modularity", n_iterations=-1, code_sha256=code_sha,
                                     slurm_job=os.environ.get("SLURM_JOB_ID"),
                                     written=dt.datetime.now().isoformat(timespec="seconds")))
            rows.append(row)
            stamp("leiden", json.dumps(row))
    import pandas as pd
    tsv = FAM / f"grid_{a.graph}.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False, mode="a", header=not tsv.exists())
    stamp("end", f"{time.time() - t0:.0f}s rss {rss_gb()}")


if __name__ == "__main__":
    main()
