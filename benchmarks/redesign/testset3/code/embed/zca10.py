#!/usr/bin/env python
"""zca10 whitening of the ESM++ small vectors of this data set, by the DPANN recipe, fitted here with no labels.

Recipe: lab/code/embed_tokens.py transform(kind="zca10") (sha256 87acde26...), materialised as in
softrec/softrec3.py cmd_prep (unique sequences, float16, unit norm):
  residue-mean vector of each unique sequence -> L2 normalise -> subtract the mean weighted by the number of rows
  (genes) carrying the sequence -> soft-ZCA with alpha 0.1 (lambda' = 0.9 lambda + 0.1 mean(lambda)) of the
  weighted covariance -> L2 normalise. Failed sequences stay zero rows (valid = False).
The vectors come from embed_tokens.transform itself; the fitted mean and matrix are recomputed with the same
helpers in a second pass, saved, and checked to reproduce the transformed vectors on 20,000 sampled rows.

Writes in --out: zca10_f16__uid.npy, zca10_transform.npz (mu, W, eigenvalues, alpha, valid), zca10.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_tokens as E  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
X, uu, valid, cnt = E.load_run(a.run)
info = E.transform(X, valid, cnt, "zca10")          # in place, the DPANN function
E.stamp("zca10", json.dumps(info), f"{time.time() - t0:.0f}s rss {E.rss_gb()}")
U, dim = X.shape
fn = a.out / "zca10_f16__uid.npy"
tmp = a.out / ".zca10_f16__uid.tmp.npy"
out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(U, dim))
for s in range(0, U, 200_000):
    out[s:s + 200_000] = X[s:s + 200_000].astype(np.float16)
out.flush()
del out
os.replace(tmp, fn)
# second pass: the fitted parameters, with the same helpers in the same order
Y, _, valid2, cnt2 = E.load_run(a.run)
assert np.array_equal(valid, valid2) and np.array_equal(cnt, cnt2)
E.unit(Y)
mu = E.weighted_mean(Y, cnt2)
for s in range(0, U, E.BLK):
    Y[s:s + E.BLK] -= mu.astype(np.float32)
Y[~valid2] = 0
W, ev = E.soft_zca(E.weighted_cov(Y, cnt2))
rng = np.random.default_rng(17)
pick = np.sort(rng.choice(np.flatnonzero(valid), size=min(20000, int(valid.sum())), replace=False))
Z = Y[pick] @ W
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
f16 = np.load(fn, mmap_mode="r")
check = dict(rows=int(len(pick)), max_abs_diff_vs_transform=float(np.abs(Z - X[pick]).max()),
             min_cosine_vs_transform=float((Z * X[pick]).sum(1).min()),
             min_cosine_f16_vs_f32=float((np.asarray(f16[pick], dtype=np.float32) * X[pick]).sum(1).min()))
assert check["min_cosine_vs_transform"] > 0.99999, check
np.savez(a.out / "zca10_transform.npz", mu=mu, W=W, eigenvalues=ev, alpha=np.float64(0.1), valid=valid,
         weights_rows_per_uid=cnt)
nrm = np.linalg.norm(np.asarray(f16[pick], dtype=np.float32), axis=1)
res = dict(run=str(a.run), unique=int(U), dim=int(dim), valid=int(valid.sum()), invalid=int((~valid).sum()),
           rows_weighted=int(cnt.sum()), transform=info,
           participation_ratio_after=float((lambda r: r.sum() ** 2 / (r ** 2).sum())(
               np.maximum(ev, 0) / (0.9 * np.maximum(ev, 0) + 0.1 * np.maximum(ev, 0).mean()))),
           eigenvalue_quantiles_0_50_90_100=np.quantile(ev, [0, 0.5, 0.9, 1]).tolist(), check=check,
           f16_norm_q=np.quantile(nrm, [0, 0.5, 1]).tolist(),
           recipe="lab/code/embed_tokens.py transform('zca10'), stored as softrec/softrec3.py cmd_prep (float16, per unique)",
           embed_tokens_sha256=sha(Path(E.__file__)), code_sha256=sha(Path(__file__)),
           zca10_f16_sha256=sha(fn), seconds=round(time.time() - t0, 1), date=time.strftime("%Y-%m-%d"))
(a.out / "zca10.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
