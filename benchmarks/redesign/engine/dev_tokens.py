#!/usr/bin/env python
"""Coarse single-level k-means tokens, only to smoke-test the engine before the
tokens workstream delivers its arms. Arm name carries the `dev__` prefix."""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
SCALE = ROOT / "model_scaling_20260913"
NREF = 4190383


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_8m")
    ap.add_argument("--k", type=int, default=16384)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    t0 = time.time()
    proto = json.load(open(SCALE / "results" / a.model / "analysis/full_workflow/protocol.json"))
    rv = np.load(proto["vectors"]["reference"]["path"], mmap_mode="r")
    qv = np.load(proto["vectors"]["query"]["path"], mmap_mode="r")
    x = np.concatenate([np.asarray(rv, dtype=np.float32), np.asarray(qv, dtype=np.float32)])
    ok = np.isfinite(x).all(axis=1)
    rng = np.random.default_rng(17)
    sample = rng.choice(np.flatnonzero(ok), size=min(1_000_000, int(ok.sum())), replace=False)
    km = faiss.Kmeans(x.shape[1], a.k, niter=12, spherical=True, seed=17, verbose=True,
                      max_points_per_centroid=256)
    km.train(x[np.sort(sample)])
    token = np.full(len(x), -1, dtype=np.int64)
    idx = faiss.IndexFlatIP(x.shape[1])
    idx.add(km.centroids)
    good = np.flatnonzero(ok)
    for s in range(0, len(good), 500_000):
        rows = good[s:s + 500_000]
        token[rows] = idx.search(x[rows], 1)[1][:, 0]
    arm = f"dev__{a.model}__raw__K{a.k}"
    tmp = RD / "tokens" / f".{arm}.parquet.tmp"
    pd.DataFrame({"gid": np.arange(len(x), dtype=np.int64), "token": token}).to_parquet(tmp, index=False)
    tmp.rename(RD / "tokens" / f"{arm}.parquet")
    manifest = dict(arm=arm, model=a.model, k=a.k, rows=len(x), assigned=int(ok.sum()),
                    tokens_used=int(len(np.unique(token[ok]))), seconds=round(time.time() - t0, 1),
                    note="development arm: single-level spherical k-means trained on a 1,000,000-row sample")
    json.dump(manifest, open(RD / "tokens" / f"{arm}.json", "w"), indent=1)
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
