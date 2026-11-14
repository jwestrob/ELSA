#!/usr/bin/env python
"""Assemble reembed_esm2.py shards into one gid-ordered float32 matrix (NaN for > 1,022 aa)."""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=Path, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--rows", type=int, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
M = np.full((a.rows, 320), np.nan, dtype=np.float32)
seen = np.zeros(a.rows, bool)
for i in range(a.n):
    g = np.load(a.shards / f"shard_{i}_of_{a.n}.gids.npy")
    M[g] = np.load(a.shards / f"shard_{i}_of_{a.n}.npy")
    seen[g] = True
assert seen.all(), f"{(~seen).sum()} gids missing"
tmp = a.out.with_name("." + a.out.name + ".tmp.npy")
np.save(tmp, M)
tmp.rename(a.out)
ok = ~np.isnan(M[:, 0])
print(json.dumps(dict(rows=a.rows, embedded=int(ok.sum()), skipped_gt1022=int((~ok).sum()),
                      norm_min=float(np.linalg.norm(M[ok], axis=1).min()))))
