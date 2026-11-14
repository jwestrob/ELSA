#!/usr/bin/env python
"""Tiny synthetic check of ctxrec kernels against a direct numpy computation (seconds; login-node safe)."""
import math
import sys
import types
from pathlib import Path

import numpy as np

# import only the kernels: stub the biotite-side modules ctxrec imports at top level
for name in ("recurrence_lf", "blib", "rules", "embed_tokens"):
    m = types.ModuleType(name)
    m.CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
    m.load_run = m.transform = None
    sys.modules[name] = m
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctxrec as C  # noqa: E402

rng = np.random.default_rng(0)
d = 16
nu = 50
X = rng.normal(size=(nu, d)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
c_start = np.array([0, 7, 9, 30], np.int64)
c_end = np.array([7, 9, 30, 41], np.int64)
n = 41
uid = rng.integers(0, nu, n).astype(np.int64)
cid = np.repeat(np.arange(4), c_end - c_start)
gen = np.array([0] * 9 + [1] * 21 + [2] * 11, np.int64)
P = np.empty((n + 4, d), np.float32)
C.build_prefix(X, uid, c_start, c_end, P)
mu_g = np.stack([X[uid[gen == g]].mean(0) for g in range(3)]).astype(np.float64)


def ctx_direct(lo, hi, W, center):
    c = cid[lo]
    idx = [k for k in range(max(c_start[c], lo - W), lo)] + [k for k in range(hi + 1, min(c_end[c], hi + W + 1))]
    if not idx:
        return None
    s = X[uid[idx]].astype(np.float64).sum(0)
    if center:
        s = s - len(idx) * mu_g[gen[lo]]
    return s / np.linalg.norm(s)


# instances of one "pair": (lo, hi) in genomes 0, 0, 1, 2
lo = np.array([1, 3, 12, 33], np.int64)
hi = np.array([2, 5, 13, 34], np.int64)
g_i = gen[lo]
Ws = np.array([2, 5], np.int64)
for center in (False, True):
    cbar = np.empty((1, 2))
    kk = np.zeros(1, np.int64)
    C.pooled_cbar(np.array([0, 4], np.int64), g_i, 0, Ws, P, c_start, c_end, lo, hi, cid[lo], g_i, mu_g, center, X, uid,
                  lo, hi, np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, d), np.float32), 2, cbar, kk)
    for wi, W in enumerate(Ws):
        Bg = {}
        for j in range(4):
            v = ctx_direct(lo[j], hi[j], W, center)
            Bg.setdefault(g_i[j], np.zeros(d))
            Bg[g_i[j]] = Bg[g_i[j]] + v
        Bs = [b / np.linalg.norm(b) for b in Bg.values()]
        k = len(Bs)
        s2 = np.linalg.norm(np.sum(Bs, 0)) ** 2
        ref = (s2 - k) / (k * (k - 1))
        pair_mean = np.mean([Bs[a] @ Bs[b] for a in range(k) for b in range(k) if a != b])
        print("center", center, "W", W, "kernel", round(cbar[0, wi], 6), "direct", round(ref, 6), "pairmean",
              round(pair_mean, 6), "k", kk[0])
        assert abs(cbar[0, wi] - ref) < 1e-5 and abs(ref - pair_mean) < 1e-9
# floor sample
ia = np.array([0, 1], np.int64)
ib = np.array([2, 3], np.int64)
od = np.empty((2, 2))
on = np.empty((2, 2), np.int64)
C.floor_sample(ia, ib, 0, Ws, P, c_start, c_end, lo, hi, cid[lo], g_i, mu_g, False, X, uid, lo, hi,
               np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, d), np.float32), od, on)
for s in range(2):
    for wi, W in enumerate(Ws):
        r = ctx_direct(lo[ia[s]], hi[ia[s]], W, False) @ ctx_direct(lo[ib[s]], hi[ib[s]], W, False)
        assert abs(od[s, wi] - r) < 1e-5, (od[s, wi], r)
# member mode
tok = rng.integers(0, 5, n).astype(np.int64)
mu_tok = C.token_means(tok, uid, X, 5)
cbar = np.empty((1, 1))
kk = np.zeros(1, np.int64)
C.pooled_cbar(np.array([0, 4], np.int64), g_i, 1, np.array([0], np.int64), P, c_start, c_end, lo, hi, cid[lo], g_i, mu_g,
              False, X, uid, lo, hi, tok[lo], tok[hi], mu_tok, 2, cbar, kk)
Bg = {}
for j in range(4):
    v = X[uid[lo[j]]] - mu_tok[tok[lo[j]]] + X[uid[hi[j]]] - mu_tok[tok[hi[j]]]
    v = v / np.linalg.norm(v)
    Bg.setdefault(g_i[j], np.zeros(d))
    Bg[g_i[j]] = Bg[g_i[j]] + v
Bs = [b / np.linalg.norm(b) for b in Bg.values()]
k = len(Bs)
ref = (np.linalg.norm(np.sum(Bs, 0)) ** 2 - k) / (k * (k - 1))
assert abs(cbar[0, 0] - ref) < 1e-5, (cbar, ref)
print("all kernel checks passed")
