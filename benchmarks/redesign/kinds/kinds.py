#!/usr/bin/env python
"""Workstream ws_kinds: recurrence of kinds (kind recurrence kappa).

For every pair of consecutive genes (a, b) on a contig, count how many independent realisations of
"a protein like a beside a protein like b" exist in other genomes, against the same count with gene
order permuted within each genome.

  lists     N_D(x): the D nearest unique sequences of x by whitened (zca10) cosine (stored HNSW top-1000)
            plus x's own sequence; a gene's list = every gene carrying one of them, minus the query's genome
  support   consecutive pairs (a', b') with seq(a') in N_D(a), seq(b') in N_D(b), any orientation;
            level of a support pair = max(rank of a' in a's list, rank of b' in b's list) (0 = identical)
  units     realisation units = embedding family pairs (f(a'), f(b')), f = Leiden communities of the zca10
            mutual-kNN graph (graph r3000); MMseqs2 family pairs are computed for evaluation only
  counts    per depth D (support level <= D):
              0 A   support pairs (descent counted)
              1 G   distinct genomes
              2 F   distinct embedding-family pairs
              3 FX  distinct embedding-family pairs with f(a'), f(b') both outside {f(a), f(b)}
              4 M   distinct MMseqs2-family pairs (evaluation)
              5 X   distinct MMseqs2-family pairs with both families outside the query's (evaluation)
              6 FXo FX restricted to support pairs in the query's arrangement
              7 Xo  X  restricted to support pairs in the query's arrangement
  null      gene order permuted within each genome (contig sizes kept), seeds 1000-1002; queries and their
            lists fixed

Subcommands
  prep      per-gene arrays, adjacencies, random same-genome pairs, whitened cosine of every adjacency
  main      all genomes as reference: observed + 3 null passes over adjacencies and random pairs
  bench     grammar benchmark: reference = training side of the split holding each adjacency
  withheld  curated system types with their families withheld (L1, L2, L2 control)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
KNN = RD / "softrec/knn/zca10__k1000__I.npy"
TOK_M = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
TOK_F = RD / "engine/dpann_tokens/graph__esmpp_small_zca10__mknn20r3000.parquet"
TOK_C = RD / "engine/dpann_tokens/graph__esmpp_small_zca10__mknn20r100.parquet"
OUT = RD / "kinds"
PREP = OUT / "prep"
DEPTHS = np.array([30, 100, 300, 1000], np.int64)
STATS = ["A", "G", "F", "FX", "M", "X", "FXo", "Xo"]
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


def rss_gb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


# ============================================================================== kernel
@njit(cache=True)
def relation(sx, sy, y_right):
    """Arrangement of x and its neighbour y: 1 x upstream of y, 2 x downstream, 3 divergent, 4 convergent."""
    if sx == sy:
        if (sx > 0 and y_right) or (sx < 0 and not y_right):
            return 1
        return 2
    ls = sx if y_right else sy
    rs = sy if y_right else sx
    if ls < 0 and rs > 0:
        return 3
    return 4


@njit(cache=True)
def distinct_by_depth(tmp, m, depths, out_row):
    """tmp[:m] = key << 10 | level; count distinct keys whose minimum level <= depths[v]."""
    if m == 0:
        return
    s = np.sort(tmp[:m])
    prev = -1
    V = len(depths)
    for t in range(m):
        key = s[t] >> 10
        if key != prev:
            lev = s[t] & 1023
            for v in range(V):
                if lev <= depths[v]:
                    out_row[v] += 1
            prev = key


@njit(parallel=True, cache=True)
def kind_pass(qa, qb, qref, ref_ok2, uq, I, Dmax, depths, mptr, members, gc, slot_of, gene_at, cid, strand,
              tokM, tokF, TF, TM, excl_own_m, mask, mask_id, U, out, hom, overflow, nchunks, bufcap):
    nq = len(qa)
    n = len(gene_at)
    V = len(depths)
    for ch in prange(nchunks):
        lo = ch * nq // nchunks
        hi = (ch + 1) * nq // nchunks
        stamp = np.full(U, -1, np.int64)
        rk = np.zeros(U, np.int32)
        b_lev = np.empty(bufcap, np.int64)
        b_g = np.empty(bufcap, np.int64)
        b_kF = np.empty(bufcap, np.int64)
        b_kM = np.empty(bufcap, np.int64)
        b_fl = np.empty(bufcap, np.int8)
        tmp = np.empty(bufcap, np.int64)
        row = np.zeros(V, np.int64)
        for q in range(lo, hi):
            a = qa[q]
            b = qb[q]
            Ua = uq[a]
            Ub = uq[b]
            for v in range(V):
                for s_ in range(8):
                    out[q, v, s_] = 0
            if Ua < 0 or Ub < 0:
                for v in range(V):
                    for s_ in range(8):
                        out[q, v, s_] = -1
                hom[q, 0] = -1
                hom[q, 1] = -1
                continue
            stamp[Ub] = q
            rk[Ub] = 0
            for j in range(Dmax):
                x = I[Ub, j]
                if x < 0:
                    break
                if stamp[x] != q:
                    stamp[x] = q
                    rk[x] = j + 1
            hom[q, 1] = rk[Ua] if stamp[Ua] == q else Dmax + 1      # a's sequence in b's list
            hom_ba = Dmax + 1
            ga = gc[a]
            fa = tokF[a]
            fb = tokF[b]
            ma = tokM[a]
            mb = tokM[b]
            rq = relation(strand[a], strand[b], True)
            r_ = qref[q]
            mi = mask_id[q]
            k = 0
            for jj in range(Dmax + 1):
                if jj == 0:
                    Up = Ua
                    ra = 0
                else:
                    Up = I[Ua, jj - 1]
                    ra = jj
                    if Up < 0:
                        break
                if Up == Ub and ra < hom_ba:
                    hom_ba = ra
                for t in range(mptr[Up], mptr[Up + 1]):
                    a2 = members[t]
                    g2 = gc[a2]
                    if g2 == ga or not ref_ok2[r_, g2]:
                        continue
                    s = slot_of[a2]
                    for dd in range(2):
                        s2 = s - 1 if dd == 0 else s + 1
                        if s2 < 0 or s2 >= n:
                            continue
                        if cid[s2] != cid[s]:
                            continue
                        b2 = gene_at[s2]
                        U2 = uq[b2]
                        if U2 < 0 or stamp[U2] != q:
                            continue
                        rb = rk[U2]
                        lev = ra if ra > rb else rb
                        m1 = tokM[a2]
                        m2 = tokM[b2]
                        if excl_own_m and (m1 == ma or m1 == mb or m2 == ma or m2 == mb):
                            continue
                        if mi >= 0 and (mask[mi, m1] or mask[mi, m2]):
                            continue
                        if k >= bufcap:
                            overflow[q] = 1
                            continue
                        f1 = tokF[a2]
                        f2 = tokF[b2]
                        fl = 0
                        if f1 != fa and f1 != fb and f2 != fa and f2 != fb:
                            fl |= 1
                        if m1 != ma and m1 != mb and m2 != ma and m2 != mb:
                            fl |= 2
                        if relation(strand[a2], strand[b2], dd == 1) == rq:
                            fl |= 4
                        b_lev[k] = lev
                        b_g[k] = g2
                        b_kF[k] = f1 * TF + f2
                        b_kM[k] = m1 * TM + m2
                        b_fl[k] = fl
                        k += 1
            hom[q, 0] = hom_ba                                      # b's sequence in a's list
            # A: support pairs
            for t in range(k):
                for v in range(V):
                    if b_lev[t] <= depths[v]:
                        out[q, v, 0] += 1
            # distinct counts
            for st in range(1, 8):
                m = 0
                for t in range(k):
                    fl = b_fl[t]
                    if st == 1:
                        key = b_g[t]
                    elif st == 2:
                        key = b_kF[t]
                    elif st == 3:
                        if (fl & 1) == 0:
                            continue
                        key = b_kF[t]
                    elif st == 4:
                        key = b_kM[t]
                    elif st == 5:
                        if (fl & 2) == 0:
                            continue
                        key = b_kM[t]
                    elif st == 6:
                        if (fl & 1) == 0 or (fl & 4) == 0:
                            continue
                        key = b_kF[t]
                    else:
                        if (fl & 2) == 0 or (fl & 4) == 0:
                            continue
                        key = b_kM[t]
                    tmp[m] = (key << 10) | b_lev[t]
                    m += 1
                for v in range(V):
                    row[v] = 0
                distinct_by_depth(tmp, m, depths, row)
                for v in range(V):
                    out[q, v, st] = row[v]


@njit(parallel=True, cache=True)
def list_gene_bound(I, mptr, Dmax):
    U = I.shape[0]
    out = np.zeros(U, np.int64)
    for u in prange(U):
        s = mptr[u + 1] - mptr[u]
        for j in range(Dmax):
            x = I[u, j]
            if x < 0:
                break
            s += mptr[x + 1] - mptr[x]
        out[u] = s
    return out


# ============================================================================== data
class Data:
    def __init__(self, load_knn=True, unit=None):
        p = PREP
        self.gc = np.load(p / "gc.npy")
        self.cid = np.load(p / "cid.npy")
        self.strand = np.load(p / "strand.npy")
        self.uq = np.load(p / "uq.npy")
        self.mptr = np.load(p / "mptr.npy")
        self.members = np.load(p / "members.npy")
        self.tokM = np.load(p / "tokM.npy")
        self.tokF = np.load(p / "tokF.npy")
        if unit:                                    # post hoc: another embedding-family resolution as the unit
            t = pd.read_parquet(RD / f"engine/dpann_tokens/graph__esmpp_small_zca10__mknn20{unit}.parquet")
            x = np.full(len(self.gc), -1, np.int64)
            x[t.gid.to_numpy()] = t.token.to_numpy()
            assert (x >= 0).all()
            self.tokF = x.astype(np.int32)
        self.TM = int(self.tokM.max()) + 1
        self.TF = int(self.tokF.max()) + 1
        self.n = len(self.gc)
        self.ng = int(self.gc.max()) + 1
        self.U = len(self.mptr) - 1
        self.qa = np.load(p / "adj_a.npy")
        self.qb = self.qa + 1
        meta = json.load(open(p / "prep.json"))
        self.bufcap = int(meta["bufcap"])
        if load_knn:
            t = time.time()
            self.I = np.load(KNN)                     # into RAM (10 GB); a memmap on the shared FS is I/O bound
            log("kNN loaded", self.I.shape, f"{time.time() - t:.0f}s rss {rss_gb()}")
        gs = np.r_[0, np.flatnonzero(np.diff(self.gc)) + 1]
        self.g_start = gs.astype(np.int64)
        self.g_end = np.r_[gs[1:], self.n].astype(np.int64)

    def order(self, r):
        """r < 0: observed order; else within-genome permutation with seed 1000 + r (softrec / engine)."""
        if r < 0:
            gene_at = np.arange(self.n, dtype=np.int64)
        else:
            rng = np.random.default_rng(1000 + r)
            gene_at = np.arange(self.n, dtype=np.int64)
            for s, e in zip(self.g_start, self.g_end):
                gene_at[s:e] = s + rng.permutation(e - s)
        slot_of = np.empty_like(gene_at)
        slot_of[gene_at] = np.arange(self.n)
        return gene_at, slot_of

    def run(self, qa, qb, qref, ref_ok2, r, threads, excl_own_m=False, mask=None, mask_id=None, label=""):
        t = time.time()
        gene_at, slot_of = self.order(r)
        nq = len(qa)
        out = np.zeros((nq, len(DEPTHS), 8), np.int32)
        hom = np.zeros((nq, 2), np.int32)
        ovf = np.zeros(nq, np.int8)
        if mask is None:
            mask = np.zeros((1, 1), np.bool_)
            mask_id = np.full(nq, -1, np.int32)
        kind_pass(qa.astype(np.int64), qb.astype(np.int64), qref.astype(np.int32), ref_ok2, self.uq, self.I,
                  int(DEPTHS.max()), DEPTHS, self.mptr, self.members, self.gc, slot_of, gene_at, self.cid,
                  self.strand, self.tokM, self.tokF, self.TF, self.TM, excl_own_m, mask, mask_id.astype(np.int32),
                  self.U, out, hom, ovf, threads * 16, self.bufcap)
        log(f"pass {label} r={r}: {nq:,} queries {time.time() - t:.0f}s overflow {int(ovf.sum())} rss {rss_gb()}")
        return out, hom, ovf


# ============================================================================== prep
def cmd_prep(a):
    PREP.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(PKG / "universe/genes.parquet").sort_values("gid").reset_index(drop=True)
    n = len(g)
    assert (g.gid.to_numpy() == np.arange(n)).all()
    gc = g.genome_code.to_numpy(np.int32)
    cc = g.contig_code.to_numpy(np.int64)
    assert (np.diff(gc) >= 0).all() and (np.diff(cc) >= 0).all()
    cid = np.r_[0, np.cumsum(np.diff(cc) != 0)].astype(np.int32)
    strand = g.strand.to_numpy(np.int8)
    uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    assert len(uq) == n
    U = int(uq.max()) + 1
    ok = uq >= 0
    members = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")].astype(np.int64)
    mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=U))].astype(np.int64)
    toks = {}
    for nm, path in (("tokM", TOK_M), ("tokF", TOK_F), ("tokC", TOK_C)):
        t = pd.read_parquet(path)
        x = np.full(n, -1, np.int64)
        x[t.gid.to_numpy()] = t.token.to_numpy()
        assert (x >= 0).all(), nm
        toks[nm] = x.astype(np.int32)
        np.save(PREP / f"{nm}.npy", toks[nm])
    i = np.flatnonzero(cc[:-1] == cc[1:])
    adj = pd.read_parquet(RD / "grammar/prep/adj1.parquet", columns=["i", "j"])
    assert np.array_equal(adj.i.to_numpy(), i) and np.array_equal(adj.j.to_numpy(), i + 1), "adjacency order != adj1"
    for nm, x in (("gc", gc), ("cid", cid), ("strand", strand), ("uq", uq), ("mptr", mptr), ("members", members),
                  ("adj_a", i.astype(np.int64)), ("length", g.length.to_numpy(np.int32))):
        np.save(PREP / f"{nm}.npy", x)
    # random same-genome pairs (the critic's null: different contig or >= 50 genes apart), seed 17
    rng = np.random.default_rng(17)
    ra = rng.integers(0, n, 600_000)
    gs = np.r_[0, np.flatnonzero(np.diff(gc)) + 1]
    ge = np.r_[gs[1:], n]
    lo, hi = gs[gc[ra]], ge[gc[ra]]
    rb = lo + (rng.random(len(ra)) * (hi - lo)).astype(np.int64)
    keep = (rb != ra) & ((cc[ra] != cc[rb]) | (np.abs(ra - rb) >= 50))
    ra, rb = ra[keep][:300_000], rb[keep][:300_000]
    np.save(PREP / "rnd_a.npy", ra.astype(np.int64))
    np.save(PREP / "rnd_b.npy", rb.astype(np.int64))
    log("genes", n, "unique", U, "adjacencies", len(i), "random pairs", len(ra))
    # buffer bound
    I = np.ascontiguousarray(np.load(KNN, mmap_mode="r"))
    lg = list_gene_bound(I, mptr, int(DEPTHS.max()))
    bufcap = int(2 * lg.max() + 64)
    log("list genes per unique sequence q50/q99/max", np.quantile(lg, [0.5, 0.99]).tolist(), int(lg.max()), "bufcap", bufcap)
    # whitened cosine of every adjacency and random pair
    sys.path.insert(0, str(RD / "softrec/code"))
    from embed_tokens import load_run, transform
    X, _, valid, cnt = load_run(EMB)
    transform(X, valid, cnt, "zca10")
    def cos(x, y):
        o = np.empty(len(x), np.float32)
        for s in range(0, len(x), 500_000):
            ux, uy = uq[x[s:s + 500_000]], uq[y[s:s + 500_000]]
            o[s:s + 500_000] = np.einsum("ij,ij->i", X[ux], X[uy])
        return o
    np.save(PREP / "adj_cos.npy", cos(i, i + 1))
    np.save(PREP / "rnd_cos.npy", cos(ra, rb))
    json.dump(dict(bufcap=bufcap, n=n, U=U, n_adj=int(len(i)), n_rnd=int(len(ra)), TM=int(toks["tokM"].max()) + 1,
                   TF=int(toks["tokF"].max()) + 1, TC=int(toks["tokC"].max()) + 1,
                   list_genes_q50_q99_max=[float(np.quantile(lg, .5)), float(np.quantile(lg, .99)), int(lg.max())],
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
              open(PREP / "prep.json", "w"), indent=1)
    log("prep done")


# ============================================================================== passes
def save(path, **kw):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def cmd_main(a):
    d = Data()
    out = OUT / "main"
    out.mkdir(parents=True, exist_ok=True)
    ra, rb = np.load(PREP / "rnd_a.npy"), np.load(PREP / "rnd_b.npy")
    qa = np.r_[d.qa, ra]
    qb = np.r_[d.qb, rb]
    ref = np.ones((1, d.ng), np.bool_)
    qref = np.zeros(len(qa), np.int32)
    for r in [-1, 0, 1, 2]:
        f = out / f"pass_{'obs' if r < 0 else 'null%d' % r}.npz"
        if f.exists():
            log("exists", f)
            continue
        o, h, ov = d.run(qa, qb, qref, ref, r, a.threads, label="main")
        save(f, out=o, hom=h, ovf=ov, n_adj=len(d.qa))


def cmd_bench(a):
    d = Data(unit=a.unit)
    out = OUT / ("bench" if not a.unit else f"bench_{a.unit}")
    out.mkdir(parents=True, exist_ok=True)
    gm = pd.read_parquet(RD / "grammar/prep/genes_meta.parquet", columns=["genome_code", "role_cv0", "role_cv1",
                                                                            "role_cv2", "role_cv3", "role_cv4"])
    gm = gm.drop_duplicates("genome_code").sort_values("genome_code")
    assert np.array_equal(gm.genome_code.to_numpy(), np.arange(d.ng))
    roles = np.stack([gm[f"role_cv{k}"].to_numpy() for k in range(5)])        # 5 x ng
    ref = roles != 2                                                            # training side = train + val
    test_split = np.argmax(roles == 2, axis=0)
    assert ((roles == 2).sum(0) == 1).all()
    qref = test_split[d.gc[d.qa]].astype(np.int32)
    for r in [-1, 0, 1, 2]:
        f = out / f"pass_{'obs' if r < 0 else 'null%d' % r}.npz"
        if f.exists():
            log("exists", f)
            continue
        o, h, ov = d.run(d.qa, d.qb, qref, ref, r, a.threads, label="bench")
        save(f, out=o, hom=h, ovf=ov, qref=qref)


def cmd_withheld(a):
    d = Data()
    out = OUT / "withheld"
    out.mkdir(parents=True, exist_ok=True)
    W = np.load(out / "design.npz")                  # written by withheld_design.py
    qa, qb = W["qa"], W["qb"]
    mask_l2, mid_l2 = W["mask_l2"], W["mid_l2"]
    mask_ct, mid_ct = W["mask_ct"], W["mid_ct"]
    ref = np.ones((1, d.ng), np.bool_)
    qref = np.zeros(len(qa), np.int32)
    for lvl in a.levels.split(","):
        for r in [-1, 0, 1, 2]:
            f = out / f"pass_{lvl}_{'obs' if r < 0 else 'null%d' % r}.npz"
            if f.exists():
                log("exists", f)
                continue
            if lvl == "full":
                o, h, ov = d.run(qa, qb, qref, ref, r, a.threads, label=lvl)
            elif lvl == "L1":
                o, h, ov = d.run(qa, qb, qref, ref, r, a.threads, excl_own_m=True, label=lvl)
            elif lvl == "L2":
                o, h, ov = d.run(qa, qb, qref, ref, r, a.threads, mask=mask_l2, mask_id=mid_l2, label=lvl)
            elif lvl == "L2ctl":
                o, h, ov = d.run(qa, qb, qref, ref, r, a.threads, mask=mask_ct, mask_id=mid_ct, label=lvl)
            else:
                raise SystemExit(lvl)
            save(f, out=o, hom=h, ovf=ov)


def cmd_support(a):
    """Support lists of selected queries at one depth (architecture graph): for each query, the left gene of
    every support pair (observed order, all genomes as reference)."""
    d = Data()
    sel = np.load(a.queries)                         # int64 adjacency rows or (a, b) pairs
    qa, qb = sel["qa"], sel["qb"]
    D = int(a.depth)
    res = support_lists(qa.astype(np.int64), qb.astype(np.int64), d.uq, d.I, D, d.mptr, d.members, d.gc, d.cid,
                        d.U, d.n, d.bufcap)
    ptr, lst = res
    np.savez(a.out, qa=qa, qb=qb, ptr=ptr, left=lst)
    log("support lists", len(qa), "entries", len(lst))


@njit(cache=True)
def _support_one(a, b, uq, I, D, mptr, members, gc, cid, U, n, stamp, rk, q, buf):
    Ua, Ub = uq[a], uq[b]
    k = 0
    if Ua < 0 or Ub < 0:
        return 0
    stamp[Ub] = q
    for j in range(D):
        x = I[Ub, j]
        if x < 0:
            break
        stamp[x] = q
    ga = gc[a]
    for jj in range(D + 1):
        Up = Ua if jj == 0 else I[Ua, jj - 1]
        if Up < 0:
            break
        for t in range(mptr[Up], mptr[Up + 1]):
            a2 = members[t]
            if gc[a2] == ga:
                continue
            for s2 in (a2 - 1, a2 + 1):
                if s2 < 0 or s2 >= n or cid[s2] != cid[a2]:
                    continue
                U2 = uq[s2]
                if U2 >= 0 and stamp[U2] == q and k < len(buf):
                    buf[k] = min(a2, s2)
                    k += 1
    return k


def support_lists(qa, qb, uq, I, D, mptr, members, gc, cid, U, n, bufcap):
    stamp = np.full(U, -1, np.int64)
    rk = np.zeros(U, np.int32)
    buf = np.empty(bufcap, np.int64)
    ptr = [0]
    out = []
    for q in range(len(qa)):
        k = _support_one(qa[q], qb[q], uq, I, D, mptr, members, gc, cid, U, n, stamp, rk, q, buf)
        x = np.unique(buf[:k])
        out.append(x)
        ptr.append(ptr[-1] + len(x))
    return np.array(ptr, np.int64), (np.concatenate(out) if out else np.zeros(0, np.int64))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("prep", "main", "bench", "withheld", "support"):
        p = sub.add_parser(nm)
        p.add_argument("--threads", type=int, required=True)
        if nm == "bench":
            p.add_argument("--unit", default=None, help="post hoc: graph resolution tag, e.g. r100")
        if nm == "withheld":
            p.add_argument("--levels", default="full,L1,L2,L2ctl")
        if nm == "support":
            p.add_argument("--queries", required=True)
            p.add_argument("--depth", required=True)
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"prep": cmd_prep, "main": cmd_main, "bench": cmd_bench, "withheld": cmd_withheld, "support": cmd_support}[a.cmd](a)
    log("end", f"rss {rss_gb()}")


if __name__ == "__main__":
    main()
