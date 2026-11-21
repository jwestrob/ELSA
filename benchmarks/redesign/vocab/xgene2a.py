#!/usr/bin/env python
"""Round 2A: zca10 gene vectors (softrec transform of ESM++ small residue means), one row per gid, float32, for
memory-mapped context sums and pooled word vectors. Writes $RD/vocab/ctx2a/xgene_zca10_f32.npy (+ .json).
usage: xgene2a.py --threads N"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
OUT = RD / "vocab/ctx2a"
sys.path.insert(0, str(RD / "softrec/code"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    from threadpoolctl import threadpool_limits
    from embed_tokens import load_run, transform  # read-only import
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=a.threads):
        X, u, valid, cnt = load_run(EMB)
        info = transform(X, valid, cnt, "zca10")
    t1 = time.time() - t0
    assert (u >= 0).all() and valid[u].all()
    tmp = OUT / ".xgene_zca10_f32.npy.tmp"
    G = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(u), X.shape[1]))
    for s in range(0, len(u), 200_000):
        G[s:s + 200_000] = X[u[s:s + 200_000]]
    G.flush()
    del G
    tmp.rename(OUT / "xgene_zca10_f32.npy")
    G = np.load(OUT / "xgene_zca10_f32.npy", mmap_mode="r")
    nrm = np.linalg.norm(G[:100_000], axis=1)
    res = dict(genes=int(len(u)), dim=int(X.shape[1]), transform=info, norm_q=np.quantile(nrm, [0, .5, 1]).tolist(),
               load_transform_s=round(t1, 1), total_s=round(time.time() - t0, 1))
    json.dump(res, open(OUT / "xgene_zca10_f32.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
