#!/usr/bin/env python
"""long_loci part 3: deep cosine kNN of the Spacedust collection's unique sequences (copy of softrec/code/knn_deep.py,
logic unchanged; vectors = transfer_spacedust/emb/zca10_f32__uid.npy, i.e. its zca10 refit of the stored ESM-C 300M
vectors, uid order = fam/gid2uid). FAISS HNSW (M 32, efConstruction 200, efSearch = k + 1), self excluded. Writes
  $RD/long_loci/sd_knn/zca10__k<k>__I.npy  int32 (U x k, -1 pad)
  $RD/long_loci/sd_knn/zca10__k<k>__D.npy  float16 cosine, descending per row
  $RD/long_loci/sd_knn/zca10__k<k>.json    timings, recall against exact search on a query sample
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VEC = RD / "transfer_spacedust/emb/zca10_f32__uid.npy"
OUT = RD / "long_loci/sd_knn"
BLK = 200_000


def stamp(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


def load_vectors(source):
    assert source == "zca10"
    X = np.load(VEC)
    valid = np.isfinite(X).all(1)
    nrm = np.linalg.norm(X[valid], axis=1)
    info = dict(transform="zca10 (transfer_spacedust refit)", vectors=str(VEC), norm_q=np.quantile(nrm, [0, 0.5, 1]).tolist())
    return X, valid, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("zca10",), default="zca10")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--block", type=int, default=50_000)
    ap.add_argument("--exact-n", type=int, default=2000)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    X, valid, info = load_vectors(a.source)
    U, d = X.shape
    vi = np.flatnonzero(valid)
    assert len(vi) == U, f"{U - len(vi)} unique sequences without a vector"
    info.update(source=a.source, k=a.k, hnsw_m=a.m, ef_construction=a.efc, ef_search=a.k + 1, unique=U, dim=d,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), threads=a.threads)
    stamp("vectors", X.shape, f"{time.time() - t0:.0f}s rss {rss_gb()}")

    rng = np.random.default_rng(a.seed)
    i, j = rng.integers(0, U, 2_000_000), rng.integers(0, U, 2_000_000)
    null = np.einsum("ij,ij->i", X[i], X[j])[i != j]
    info["random_pair_cosine_q50_q99_q999_q9999"] = np.quantile(null, [0.5, 0.99, 0.999, 0.9999]).round(4).tolist()

    t1 = time.time()
    index = faiss.IndexHNSWFlat(d, a.m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = a.efc
    for s in range(0, U, BLK):
        index.add(np.ascontiguousarray(X[s:s + BLK]))
    info["build_seconds"] = round(time.time() - t1, 1)
    stamp("hnsw built", info["build_seconds"], f"rss {rss_gb()}")
    index.hnsw.efSearch = a.k + 1

    tag = f"{a.source}__k{a.k}"
    fI = OUT / f".{tag}__I.npy.tmp"
    fD = OUT / f".{tag}__D.npy.tmp"
    Iout = np.lib.format.open_memmap(fI, mode="w+", dtype=np.int32, shape=(U, a.k))
    Dout = np.lib.format.open_memmap(fD, mode="w+", dtype=np.float16, shape=(U, a.k))
    t1 = time.time()
    n_self_missing = 0
    for s in range(0, U, a.block):
        e = min(U, s + a.block)
        D, I = index.search(np.ascontiguousarray(X[s:e]), a.k + 1)
        rows = np.arange(s, e)
        is_self = I == rows[:, None]
        has_self = is_self.any(1)
        n_self_missing += int((~has_self).sum())
        # drop self where present, else drop the last column
        drop = np.where(has_self, is_self.argmax(1), a.k)
        keep = np.ones_like(I, dtype=bool)
        keep[np.arange(len(rows)), drop] = False
        Iout[s:e] = I[keep].reshape(len(rows), a.k).astype(np.int32)
        Dout[s:e] = D[keep].reshape(len(rows), a.k).astype(np.float16)
        if (s // a.block) % 10 == 0:
            el = time.time() - t1
            stamp(f"search {e:,}/{U:,} {el:.0f}s ({e - 0:,} q, {(e) / max(el, 1e-9):.0f} q/s) rss {rss_gb()}")
    Iout.flush()
    Dout.flush()
    info["search_seconds"] = round(time.time() - t1, 1)
    info["queries_without_self_in_result"] = n_self_missing
    stamp("search done", info["search_seconds"])

    # recall against brute force
    q = np.sort(rng.choice(U, min(a.exact_n, U), replace=False))
    t1 = time.time()
    flat = faiss.IndexFlatIP(d)
    for s in range(0, U, BLK):
        flat.add(np.ascontiguousarray(X[s:s + BLK]))
    De, Ie = flat.search(np.ascontiguousarray(X[q]), a.k + 1)
    rec = {}
    Ih = np.asarray(Iout[q])
    for kk in (50, 200, 1000, a.k):
        if kk > a.k:
            continue
        hits = []
        for r in range(len(q)):
            ex = Ie[r][Ie[r] != q[r]][:kk]
            hits.append(len(np.intersect1d(Ih[r, :kk], ex)) / kk)
        rec[f"recall@{kk}"] = round(float(np.mean(hits)), 4)
    info["recall_vs_exact"] = rec
    info["exact_seconds"] = round(time.time() - t1, 1)
    Dq = np.asarray(Dout[q]).astype(np.float32)
    info["cosine_q10_q50_q90_at_rank"] = {str(r): np.quantile(Dq[:, r - 1], [0.1, 0.5, 0.9]).round(3).tolist()
                                         for r in (1, 50, 200, 1000, a.k) if r <= a.k}
    stamp("recall", rec)
    del Iout, Dout
    os.replace(fI, OUT / f"{tag}__I.npy")
    os.replace(fD, OUT / f"{tag}__D.npy")
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_gb"] = rss_gb()
    (OUT / f"{tag}.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    stamp("end", f"{info['seconds']}s rss {rss_gb()}")


if __name__ == "__main__":
    main()
