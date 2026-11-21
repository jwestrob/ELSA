#!/usr/bin/env python
"""Login-node smoke test (2 threads, tiny synthetic data): numba typing of the round-4 kernels, mode 0 of soft_pass4
equal to softrec2.soft_pass2, modes 1 and 2 equal to the brute recount, support_edges / overlap_edges / critic_fill run."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import softrec4 as S4  # noqa: E402
import readings4 as R4m  # noqa: E402

t0 = time.time()
rng = np.random.default_rng(5)
n, ng, nl, K, w = 400, 8, 4, 30, 3
gc = np.repeat(np.arange(ng), n // ng).astype(np.int32)
cc = np.repeat(np.arange(ng * 2), n // (ng * 2))
brk = np.flatnonzero(np.diff(cc)) + 1
c_start = np.r_[0, brk].astype(np.int64)
c_end = np.r_[brk, n].astype(np.int64)
cid = np.repeat(np.arange(len(c_start)), c_end - c_start).astype(np.int32)
lin_of_genome = (np.arange(ng) % nl).astype(np.int32)
lin = lin_of_genome[gc]
strand = rng.choice(np.array([-1, 1], np.int8), n)
# lists: random genes outside the genus, ranked; ties via tail
nb = np.full((n, K), -1, np.int32)
tail = np.zeros((n, K), np.uint16)
llen = np.zeros(n, np.int64)
for a in range(n):
    pool = np.flatnonzero(lin != lin[a])
    k = rng.integers(5, K + 1)
    pick = rng.choice(pool, k, replace=False)
    nb[a, :k] = pick
    t = 0
    while t < k:
        blk = int(rng.integers(1, 3))
        e = min(k, t + blk)
        tail[a, t:e] = e
        t = e
    llen[a] = k
nbf, tailf = nb.ravel(), tail.ravel()
nbg = np.where(nbf >= 0, gc[np.maximum(nbf, 0)], 0).astype(np.int16)
lstart = np.arange(n, dtype=np.int64) * K
lend = lstart + llen
V = 2
cap = np.stack([np.minimum(llen, 10), np.minimum(llen, K)]).astype(np.int32)
capmax = cap.max(0).astype(np.int32)
nd = np.minimum(w, c_end[cid] - 1 - np.arange(n)).astype(np.int32)
nd = np.maximum(nd, 0)
off = np.r_[0, np.cumsum(nd)[:-1]].astype(np.int64)
n_inst = int(nd.sum())
excl = np.zeros((V, n_inst), np.bool_)
n_g = np.bincount(gc, minlength=ng).astype(np.float64)
wtype = np.array([x[1] for x in S4.WEIGHTS], np.int64)
wpar = np.array([x[2] for x in S4.WEIGHTS], np.float64)
ident = np.arange(n, dtype=np.int64)


def run(mode, fn=None):
    oL = np.zeros((V, n_inst), np.int16)
    oG = np.zeros((V, n_inst), np.int16)
    oE = np.zeros((V, n_inst), np.float32)
    oW = np.zeros((V, len(wtype), n_inst), np.float32)
    if fn is None:
        S4.soft_pass4(n, w, V, ident, cid, lstart, lend, nbf, nbg, tailf, gc, lin_of_genome, cap, capmax, excl, off, nd, ng, nl,
                      n_g, wtype, wpar, strand, mode, oL, oG, oE, oW, 8)
    else:
        fn(n, w, V, ident, cid, lstart, lend, nbf, nbg, tailf, gc, lin_of_genome, cap, capmax, excl, off, nd, ng, nl, n_g,
           wtype, wpar, oL, oG, oE, oW, 8)
    return dict(L=oL, G=oG, E=oE, W=oW)


p0 = run(0)
p2ref = run(0, S4.S2.soft_pass2)
print("mode0 == soft_pass2:", all(np.array_equal(p0[k], p2ref[k]) for k in ("L", "G", "E", "W")), f"{time.time() - t0:.1f}s")
u = SimpleNamespace(n=n, lin=lin, cid=cid)
Lo = SimpleNamespace(V=V, cap=cap, nb=nbf, lstart=lstart, lend=lend, tail=tailf)
for mode in (0, 1, 2):
    p = run(mode)
    br = S4.brute4(u, Lo, p, nd, off, strand, mode, np.random.default_rng(1), m=min(300, n_inst))
    print("mode", mode, "brute", br, "mean L", float(p["L"][1].mean()))
# support edges on all instances with L >= 1 (mode 2)
left = np.repeat(np.arange(n, dtype=np.int64), nd)
d = (np.arange(n_inst) - off[left] + 1).astype(np.int64)
p = run(2)
sel = np.flatnonzero(p["L"][1] >= 1)
sl, sd = left[sel], d[sel]
inst_pos = np.full(n * w, -1, np.int64)
inst_pos[sl * w + sd - 1] = np.arange(len(sel))
M = len(sel)
dst = np.full((M, 10), -1, np.int32)
mm = np.zeros((M, 10), np.int32)
cnt = np.zeros(M, np.int8)
ngen = np.zeros(M, np.int32)
S4.support_edges(sl, sd, K, n, w, cid, lstart, lend, nbf, tailf, nbg, lin_of_genome, strand, 2, inst_pos, nl, 10, dst, mm, cnt, ngen, 8)
print("support edges: instances", M, "with any", int((cnt > 0).sum()), "self", int((dst == np.arange(M)[:, None]).sum()))
oa, ob = R4m.overlap_edges(sl, sd, inst_pos, n, w)
print("overlap edges", len(oa), bool((oa < ob).all()))
b = np.zeros(n, np.float32)
b[rng.choice(n - 1, 100, replace=False)] = 3.0
on = rng.random(n) < 0.5
b2, ne, nbr, na = R4m.critic_fill(b, on, c_start, c_end, 2, False, np.float32(2.0))
b3, ne3, nbr3, na3 = R4m.critic_fill(b, on, c_start, c_end, 1, True, np.float32(2.0))
print("critic fill x m2:", ne, nbr, na, int((b2 != b).sum()), "| b m1:", ne3, nbr3, na3, int((b3 != b).sum()))
assert int((b2 != b).sum()) == na and int((b3 != b).sum()) == na3
assert ((b2 != b) <= (on & (b == 0))).all()
print("smoke done", f"{time.time() - t0:.1f}s")
