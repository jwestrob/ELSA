#!/usr/bin/env python
"""Toy compile-and-consistency check of softctx kernels (login node, seconds): with z large (rho = 0) N_eff = 1 + supporting
genomes and W_ctx = W; brute-force support count on a toy universe."""
import sys, types
from pathlib import Path
import numpy as np
for name in ("recurrence_lf", "blib", "rules", "embed_tokens", "softrec", "softrec2", "readings2", "softrec_lf"):
    m = types.ModuleType(name); m.CALLERS = ("a", "b", "c"); m.load_run = m.transform = None; m.RD = Path("/tmp"); m.W_T = (1,)
    sys.modules[name] = m
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softctx as S
rng = np.random.default_rng(3)
# toy: 6 genomes x 1 contig x 12 genes; gene families f in 0..5 arranged so that families (0,1) are adjacent in genomes 0..4
ng, per = 6, 12
n = ng * per
gc = np.repeat(np.arange(ng), per).astype(np.int32)
cid = gc.copy()
fam = rng.integers(2, 6, n)
for g in range(5):
    fam[g * per + 4] = 0; fam[g * per + 5] = 1
D = 8
F = rng.normal(size=(6, D)); F /= np.linalg.norm(F, axis=1, keepdims=True)
Cx = rng.normal(size=(n, D)).astype(np.float32); Cx /= np.linalg.norm(Cx, axis=1, keepdims=True)
# lists: genes of the same family in other genomes, depth K
K = 10
nb = np.full((n, K), -1, np.int32); tail = np.zeros((n, K), np.uint16); llen = np.zeros(n, np.int64)
for a in range(n):
    c = [b for b in range(n) if fam[b] == fam[a] and gc[b] != gc[a]][:K]
    nb[a, :len(c)] = c; tail[a, :len(c)] = len(c); llen[a] = len(c)
nbf = nb.ravel(); tailf = tail.ravel(); nbg = np.where(nbf >= 0, gc[np.maximum(nbf, 0)], 0).astype(np.int16)
lstart = np.arange(n, dtype=np.int64) * K; lend = lstart + llen
cap = llen.astype(np.int32)[None, :]; capmax = cap[0].copy()
w = 3
nd = np.array([max(0, min(w, (gc[i] + 1) * per - 1 - i)) for i in range(n)], np.int32)
off = np.r_[0, np.cumsum(nd)[:-1]].astype(np.int64); n_inst = int(nd.sum())
excl = np.zeros((1, n_inst), np.bool_)
for i in range(n):
    for d in range(1, nd[i] + 1):
        excl[0, off[i] + d - 1] = fam[i] == fam[i + d]
outs = [np.zeros((1, n_inst), t) for t in (np.int16, np.int16, np.float32, np.float32, np.float32, np.float32)]
ident = np.arange(n, dtype=np.int64)
S.soft_pass_ctx(n, w, 1, ident, cid, lstart, lend, nbf, nbg, tailf, gc, cap, capmax, excl, off, nd, ng,
                np.full(ng, float(per)), 50.0, Cx, 0.0, 0.05, 1e9, *outs, 4)
oL, oG, oE, oW, oN, oWc = outs
for i in range(n):
    for d in range(1, nd[i] + 1):
        inst = off[i] + d - 1
        if excl[0, inst]:
            assert oL[0, inst] == -1; continue
        a, b = i, i + d
        sup = set()
        for x in nb[a, :llen[a]]:
            for y in nb[b, :llen[b]]:
                if x != y and gc[x] == gc[y] and abs(x - y) <= w and gc[x] != gc[a]:
                    sup.add(gc[x])
        assert oG[0, inst] == len(sup), (i, d, oG[0, inst], sup)
        assert abs(oN[0, inst] - (len(sup) + 1)) < 1e-5 and abs(oWc[0, inst] - oW[0, inst]) < 1e-5
inst01 = [off[g * per + 4] for g in range(5)]
print("supporting genomes of the (0,1) instances:", [int(oG[0, k]) for k in inst01], "N_eff (z huge):", [float(oN[0, k]) for k in inst01])
# identical contexts -> N_eff = 1
Cx2 = np.tile(Cx[:1], (n, 1))
outs2 = [np.zeros((1, n_inst), t) for t in (np.int16, np.int16, np.float32, np.float32, np.float32, np.float32)]
S.soft_pass_ctx(n, w, 1, ident, cid, lstart, lend, nbf, nbg, tailf, gc, cap, capmax, excl, off, nd, ng,
                np.full(ng, float(per)), 50.0, Cx2, 0.0, 0.05, 0.0, *outs2, 4)
print("identical contexts N_eff:", [float(outs2[4][0, k]) for k in inst01])
assert all(abs(outs2[4][0, k] - 1.0) < 1e-4 for k in inst01)
# position contexts and list expansion compile
P = np.zeros((n + ng, D), np.float32)
Xp = rng.normal(size=(n, D)).astype(np.float32)
import ctxrec as C
C.build_prefix(Xp, np.arange(n, dtype=np.int64), np.arange(ng, dtype=np.int64) * per, (np.arange(ng, dtype=np.int64) + 1) * per, P)
out = np.zeros((n, D), np.float32); on = np.zeros(n, np.int64)
S.position_contexts(P, np.arange(ng, dtype=np.int64) * per, (np.arange(ng, dtype=np.int64) + 1) * per, cid.astype(np.int64),
                    gc.astype(np.int64), np.zeros((ng, D)), False, 5, out, on)
p = 20
idx = [q for q in range(max(12, p - 5), min(24, p + 6)) if abs(q - p) > 3]
ref = Xp[idx].sum(0); ref /= np.linalg.norm(ref)
assert np.allclose(out[p], ref, atol=1e-5) and on[p] == len(idx)
I = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1]], np.int64); Dd = np.array([[0.9, 0.5, 0], [0.9, 0.4, 0], [0.5, 0.4, 0]], np.float32)
uq = np.array([0, 0, 1, 2, 2, 1], np.int64); gen = np.array([0, 1, 2, 3, 4, 5], np.int64)
order = np.argsort(uq, kind="stable"); mptr = np.r_[0, np.cumsum(np.bincount(uq, minlength=3))].astype(np.int64)
Cz = np.eye(6, D, dtype=np.float32)
nb2 = np.full((6, 3), -1, np.int32); t2 = np.zeros((6, 3), np.uint16); l2 = np.zeros(6, np.int64); nx = np.zeros(6, np.int64); cc = np.zeros(6, np.float32)
S.expand_knn_ctx(uq, gen, Cz, np.float32(0.5), I, Dd, mptr, order.astype(np.int64), 3, nb2, t2, l2, nx, cc)
print("lists", nb2.tolist(), "capcos", cc.tolist())
print("softctx kernel checks passed")
