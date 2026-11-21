#!/usr/bin/env python
"""Synthetic check of cuts.py cohesion kernels against a direct computation (seconds; login-node safe)."""
import sys, types
from pathlib import Path
import numpy as np
for name in ("recurrence_lf", "blib", "rules", "embed_tokens"):
    m = types.ModuleType(name); m.CALLERS = ("a", "b", "c"); m.load_run = m.transform = None; sys.modules[name] = m
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctxrec  # noqa
ctxrec.B = sys.modules["blib"]; ctxrec.BR = sys.modules["rules"]; ctxrec.RL = sys.modules["recurrence_lf"]
import cuts as K  # noqa
rng = np.random.default_rng(1)
d, nu, n = 12, 40, 60
X = rng.normal(size=(nu, d)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
uid = rng.integers(0, nu, n).astype(np.int64)
tok = rng.integers(0, 4, n).astype(np.int64)
strand = rng.choice([-1, 1], n).astype(np.int64)
cid = np.repeat(np.arange(3), [25, 20, 15]).astype(np.int64)
A, cnt = K.side_sums(tok, uid, X, strand, cid, 4)
nA = K.side_norms(A)
kr = np.empty(n); kl = np.empty(n)
K.gene_cohesion(tok, uid, X, strand, cid, A, cnt, nA, 0.0, kr, kl)
def nbr(m, side):  # side 0 downstream, 1 upstream
    step = 1 if (strand[m] == 1) == (side == 0) else -1
    j = m + step
    return j if 0 <= j < n and cid[j] == cid[m] else -1
for i in range(n):
    for which, out in ((0, kr), (1, kl)):
        nb = i + 1 if which == 0 else i - 1
        if nb < 0 or nb >= n or cid[nb] != cid[i]:
            assert np.isnan(out[i]); continue
        side = 0 if (strand[i] == 1) == (which == 0) else 1
        ys = [X[uid[nbr(m, side)]] for m in range(n) if tok[m] == tok[i] and m != i and nbr(m, side) >= 0]
        if len(ys) < 2:
            assert np.isnan(out[i]); continue
        Y = np.array(ys, dtype=np.float64); Cm = Y @ Y.T
        ref = (Cm.sum() - np.trace(Cm)) / (len(ys) * (len(ys) - 1))
        assert abs(out[i] - ref) < 1e-4, (i, which, out[i], ref)
print("cohesion kernel matches direct computation")
b = np.ones(n, np.float32); kap = np.full(n, 0.5); kap[5] = 0.0; kap[10] = np.nan
s, e = K.var_cores(b, kap, np.array([2], np.int64), np.array([12], np.int64), 0.1)
print("var_cores", list(zip(s, e)))
s, e = K.cont_cores(b, kap, np.array([2], np.int64), np.array([12], np.int64), 0.1)
print("cont_cores", list(zip(s, e)))
