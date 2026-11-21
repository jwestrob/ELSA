#!/usr/bin/env python
"""Login-node check of lb.count_lists against a brute-force window scan on a random synthetic universe."""
import math, sys
import numpy as np
sys.argv = sys.argv[:1]
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("lbk", str(Path(__file__).resolve().parent / "lb.py"))
rng = np.random.default_rng(3)
# synthetic: 40 genomes x 3 contigs x 30 genes
ng, nc, L = 40, 3, 30
n = ng * nc * L
gc = np.repeat(np.arange(ng), nc * L).astype(np.int64)
cid = np.repeat(np.arange(ng * nc), L).astype(np.int64)
grp = (np.arange(ng) // 3).astype(np.int64)
mg = np.full(ng, nc * L, np.float64)
fam = rng.integers(0, 60, n)
# lists: partners = genes of the same family in other groups (+ some -1 padding), random order
lists = []
for g in range(n):
    p = np.flatnonzero((fam == fam[g]) & (grp[gc] != grp[gc[g]]))
    p = rng.permutation(p)[: rng.integers(0, len(p) + 1)]
    lists.append(p)
K = max(len(x) for x in lists) + 2
nb = np.full((n, K), -1, np.int32)
cnt = np.zeros(n, np.int64)
for g, p in enumerate(lists):
    nb[g, :len(p)] = p
    cnt[g] = len(p) + (1 if len(p) and rng.random() < .3 else 0)   # sometimes include a -1 inside the count
fstart = (np.arange(n) * K).astype(np.int64)
import lb
lo = rng.integers(0, n - 6, 300)
hi = lo + rng.integers(1, 6, 300)
ok = cid[lo] == cid[hi]
lo, hi = lo[ok].astype(np.int64), hi[ok].astype(np.int64)
own = grp[gc[lo]]
nO, eO, nI = lb.count_lists(lo, hi, own, fstart, cnt, nb.ravel(), cid, gc, grp, int(grp.max()) + 1, mg, 16)
bad = 0
for i in range(len(lo)):
    genes = [g for g in range(lo[i], hi[i] + 1) if (nb[g, :cnt[g]] >= 0).any()]
    kf = len(genes)
    if kf < 2:
        bad += nI[i] != kf
        continue
    q = max(2, math.ceil(.5 * kf)); W = 2 * kf
    lab = {}
    for li, g in enumerate(genes):
        for x in nb[g, :cnt[g]]:
            if x >= 0 and x not in lab:
                lab[int(x)] = li
    hit = set()
    for st in sorted(lab):
        win = [x for x in lab if st <= x < st + W and cid[x] == cid[st]]
        if len({lab[x] for x in win}) >= q:
            hit.add(int(grp[gc[st]]))
    nb_ = len(hit) - (1 if own[i] in hit else 0)
    bad += (nb_ != nO[i]) or (nI[i] != kf)
print("nodes", len(lo), "mismatches", bad, "n q50/q90", np.quantile(nO, [.5, .9]), "E q50", np.median(eO))
assert bad == 0
print("lb smoke OK")
