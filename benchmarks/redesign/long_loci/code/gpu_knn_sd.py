#!/usr/bin/env python
"""long_loci part 3: exact cosine kNN of the Spacedust unique zca10 vectors on one A5000 (replaces the CPU HNSW run,
which projected to ~4 h on 64 CPUs and was cancelled under the compute limit). Same output layout as softrec
knn_deep.py: I int32 (U x k, -1 pad), D float16 cosine descending; self dropped where present, else the last column.
Exact search (no HNSW approximation); fp16 products, fp32 accumulation inside the matmul.

usage: gpu_knn_sd.py --k 2000 [--limit N]   (--limit: time the first N queries only, write nothing but the json)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VEC = RD / "transfer_spacedust/emb/zca10_f16__uid.npy"
OUT = RD / "long_loci/sd_knn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--qc", type=int, default=2048)
    ap.add_argument("--tb", type=int, default=262144)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    vec = np.load(VEC, mmap_mode="r")
    U, dim = vec.shape
    X = torch.empty((U, dim), dtype=torch.float16, device=dev)
    for s in range(0, U, 200_000):
        X[s:s + 200_000] = torch.from_numpy(np.array(vec[s:s + 200_000])).to(dev)
    nrm = X[:100_000].float().norm(dim=1)
    info = dict(gpu=torch.cuda.get_device_name(0), U=int(U), dim=int(dim), k=a.k, qc=a.qc, tb=a.tb, limit=a.limit,
                norm_q=[float(nrm.min()), float(nrm.median()), float(nrm.max())], load_seconds=round(time.time() - t0, 1),
                slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    print(json.dumps(info), flush=True)
    K1 = a.k + 1
    nq = a.limit if a.limit else U
    tag = f"zca10__k{a.k}"
    if not a.limit:
        Iout = np.lib.format.open_memmap(OUT / f".{tag}__I.npy.tmp", mode="w+", dtype=np.int32, shape=(U, a.k))
        Dout = np.lib.format.open_memmap(OUT / f".{tag}__D.npy.tmp", mode="w+", dtype=np.float16, shape=(U, a.k))
    blocks = [(s, min(U, s + a.tb)) for s in range(0, U, a.tb)]
    n_self_missing = 0
    t1 = time.time()
    for q0 in range(0, nq, a.qc):
        q1 = min(nq, q0 + a.qc)
        Q = X[q0:q1]
        cv, ci = [], []
        for s, e in blocks:
            S = Q @ X[s:e].T
            v, i = torch.topk(S, min(K1, e - s), dim=1)
            cv.append(v)
            ci.append(i + s)
            del S
        v = torch.cat(cv, 1)
        i = torch.cat(ci, 1)
        v, o = torch.topk(v.float(), K1, dim=1)
        i = torch.gather(i, 1, o)
        rows = torch.arange(q0, q1, device=dev)[:, None]
        is_self = i == rows
        has_self = is_self.any(1)
        n_self_missing += int((~has_self).sum())
        drop = torch.where(has_self, is_self.int().argmax(1), torch.full_like(has_self, a.k, dtype=torch.int64))
        keep = torch.ones_like(i, dtype=torch.bool)
        keep[torch.arange(q1 - q0, device=dev), drop] = False
        I = i[keep].view(q1 - q0, a.k)
        D = v[keep].view(q1 - q0, a.k)
        if not a.limit:
            Iout[q0:q1] = I.to(torch.int32).cpu().numpy()
            Dout[q0:q1] = D.half().cpu().numpy()
        if (q0 // a.qc) % 50 == 0 or q1 == nq:
            el = time.time() - t1
            print(f"{q1:,}/{nq:,} {el:.0f}s eta_full {el / q1 * (U - q1):.0f}s", flush=True)
    info["search_seconds"] = round(time.time() - t1, 1)
    info["projected_full_search_seconds"] = round(info["search_seconds"] / nq * U, 1)
    info["queries_without_self_in_result"] = n_self_missing
    info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if not a.limit:
        Iout.flush()
        Dout.flush()
        qs = np.random.default_rng(17).choice(U, 2000, replace=False)
        Dq = np.asarray(Dout[np.sort(qs)]).astype(np.float32)
        info["cosine_q10_q50_q90_at_rank"] = {str(r): np.quantile(Dq[:, r - 1], [0.1, 0.5, 0.9]).round(3).tolist()
                                             for r in (1, 50, 200, 1000, a.k)}
        del Iout, Dout
        os.replace(OUT / f".{tag}__I.npy.tmp", OUT / f"{tag}__I.npy")
        os.replace(OUT / f".{tag}__D.npy.tmp", OUT / f"{tag}__D.npy")
    info["seconds"] = round(time.time() - t0, 1)
    fn = OUT / (f"{tag}.json" if not a.limit else f"{tag}__timing_test.json")
    fn.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
