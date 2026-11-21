#!/usr/bin/env python
"""Login-node check: zwa kernels at window 3 equal vl.adjacency + vl.pass_count on a synthetic universe; the adaptive
support at distance d equals a brute-force count for sampled adjacencies."""
import sys
from pathlib import Path
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zwa
import vl
rng = np.random.default_rng(5)
ng, nc, L = 30, 2, 40
n = ng * nc * L
gc = np.repeat(np.arange(ng), nc * L).astype(np.int64)
cid = np.repeat(np.arange(ng * nc), L).astype(np.int64)
grp = (np.arange(ng) // 2).astype(np.int64)
mg = np.full(ng, nc * L, np.float64)
fam = rng.integers(0, 50, n)
nb_l, cnt = [], np.zeros(n, np.int64)
for g in range(n):
    p = np.flatnonzero((fam == fam[g]) & (grp[gc] != grp[gc[g]]))
    p = rng.permutation(p)[: rng.integers(0, len(p) + 1)]
    nb_l.append(p); cnt[g] = len(p)
lstart = np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64)
nbs = np.concatenate(nb_l).astype(np.int32)
nbg = gc[nbs].astype(np.int16)
tail = np.ones(len(nbs), np.uint16)
c_end = np.r_[np.flatnonzero(np.diff(cid)) + 1, n]
def inst(w):
    ce = c_end[np.searchsorted(c_end, np.arange(n), side="right")]
    nd = np.maximum(np.minimum(w, ce - 1 - np.arange(n)), 0)
    off = np.r_[0, np.cumsum(nd)[:-1]].astype(np.int64)
    left = np.repeat(np.arange(n), nd); d = np.arange(int(nd.sum())) - off[left] + 1
    return nd, off, left, d
nd3, off3, left3, d3 = inst(3)
nd8, off8, left8, d8 = inst(8)
ex8 = rng.random(len(left8)) < 0.05
ex3 = ex8[off8[left3] + d3 - 1]
ns = np.arange(n, dtype=np.int64); nterm = np.ones(n, np.bool_)
aptr, ak = vl.adjacency(n, ns, nterm, cid, cnt, ex3, off3, True)
aptr2, ak2 = zwa.adjacency_w(n, cid, cnt, ex3, off3, 3)
assert np.array_equal(aptr, aptr2) and np.array_equal(ak, ak2)
R = len(ak)
o1 = [np.zeros(R, np.int64), np.zeros(R, np.int64), np.zeros(R), np.zeros(R)]
z = np.zeros(1, np.int32)
vl.pass_count(n, ns, nterm, np.full(n, -1, np.int64), aptr, ak, ns, lstart, cnt, nbs, tail, nbg, np.zeros(1, np.int64),
              z, z, z, np.zeros(1, np.int16), cid, grp, ng, int(grp.max()) + 1, mg, *o1, 4)
o2 = [np.zeros(R, np.int64), np.zeros(R, np.int64), np.zeros(R), np.zeros(R)]
zwa.pass_count_w(n, aptr2, ak2, ns, lstart, cnt, nbs, nbg, cid, grp, ng, int(grp.max()) + 1, mg, 3, 3, *o2, 4)
eq = [np.array_equal(a, b) for a, b in zip(o1, o2)]
print("window-3 equality n/ng/E/Eg:", eq, "rows", R)
assert all(eq)
# adaptive vs brute force
aptr8, ak8 = zwa.adjacency_w(n, cid, cnt, ex8, off8, 8)
R8 = len(ak8)
o3 = [np.zeros(R8, np.int64), np.zeros(R8, np.int64), np.zeros(R8), np.zeros(R8)]
zwa.pass_count_w(n, aptr8, ak8, ns, lstart, cnt, nbs, nbg, cid, grp, ng, int(grp.max()) + 1, mg, 8, 0, *o3, 4)
rj = np.repeat(np.arange(n), np.diff(aptr8))
bad = 0
for r in rng.choice(R8, 300, replace=False):
    j, k = rj[r], ak8[r]; d = k - j
    X = nbs[lstart[j]:lstart[j] + cnt[j]]; Y = nbs[lstart[k]:lstart[k] + cnt[k]]
    sup = set()
    for q in Y:
        if any((cid[p] == cid[q]) and (0 < abs(int(q) - int(p)) <= d) for p in X):
            sup.add(int(gc[q]))
    bad += len(sup) != o3[0][r]
print("adaptive support mismatches", bad, "of 300; rows", R8, "by d", np.bincount(ak8 - rj, minlength=9)[1:].tolist())
assert bad == 0
print("zwa smoke OK")
