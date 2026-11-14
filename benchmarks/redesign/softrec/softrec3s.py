#!/usr/bin/env python
"""softrec3: stratified lists, the statistic on them, candidate rules and validated edges.

usage
  softrec3s.py lists --src ex|kn --threads T
      ex: exact per-genome best matches (r3/exact, GPU job, stored at cosine >= 0.32); kn: per-genome bests inside the
      stored zca10 top-2,000 unique kNN (the cheap version). Per unique sequence: one entry per genome (best gene, cosine
      >= 0.32); level = rank of the genome among the sequence's genomes of the same genus by cosine (0 = the genus's
      best). Floors are global cosines (0.32, 0.45, 0.6), so the level among entries above a higher floor equals the level
      among all entries. Tail rank t = 1 + N_genes x P(cos >= c over all pairs): the cosine's expected position in a
      global ranked list. Gene lists = the sequence's entries outside the gene's genus. Writes r3/lists/<src>/ and
      lists.json (Task 1: genera reached vs genera holding an MMseqs2 family member).
  softrec3s.py stats --src ex|kn --threads T
      observed pass + 3 gene-order nulls (seeds 1000 + r as the engine): per instance and variant (floor f, per-genus
      cap m) L = genera other than the own genus with a supporting pair (a', b'): a' = a's entry in genome G, b' = b's entry
      in G, a' != b', within 3 genes on one contig, both cosines >= f, min(level a', level b') < m; W_j = sum over those
      genera of exp(-(t_g - 1) / r_j), t_g = min over the genus's supporting pairs of max(t_a', t_b').
  softrec3s.py cands --src ex|kn --threads T [--edges NAMES]
      rule selection per (variant, statistic) with the engine's procedure at the instance level (FDR 0.01, top 12 by
      count, most genes covered); r3/cand/<name>/ in the round-2 format; validated edges for NAMES.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import softrec3 as S3  # noqa: E402
SR = S3.SR
from embed_tokens import stamp, rss_gb  # noqa: E402

RD, R3, EMB = S3.RD, S3.R3, S3.EMB
W = 3
FLOORS = (0.32, 0.45, 0.6)     # global cosine floors of the variants (0.32 = the brief's start = storage floor)
FBIN = np.array([int(round(f * 1000)) for f in FLOORS], np.int64)
MCAPS = (1, 3, 255)            # per-genus cap on min(level a', level b'); 255 = every genome
RW = (100.0, 1000.0)           # rank-weight scales r_j
COUNT_M = SR.COUNT_M
W_T = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 260, 330, 400)
FDR = 0.01
CH = 100_000_000


def variants():
    return [(fi, m) for fi in range(len(FLOORS)) for m in MCAPS]


def vname(fi, m):
    return f"c{int(round(FLOORS[fi] * 100))}_m{'all' if m == 255 else m}"


def tail_table(n_genes):
    """t[k] = 1 + N_genes x P(cos >= k / 1000) over all pairs of the calibration (10,000 queries x all genes)."""
    h = np.load(R3 / "calib/hist.npz")
    hall = (h["same"] + h["diff"]).astype(np.float64)
    up = np.cumsum(hall[::-1])[::-1] / hall.sum()            # up[b] = P(bin >= b) = P(cos >= b/1000 - 1)
    return np.array([1.0 + n_genes * up[min(1000 + k, 1999)] for k in range(1001)], np.float64)


def to_kbin(v16):
    out = np.empty(len(v16), np.uint16)
    for s0 in range(0, len(v16), CH):
        x = v16[s0:s0 + CH].view(np.float16).astype(np.float32)
        out[s0:s0 + CH] = np.clip(np.floor(x * 1000.0 + 1e-4), 0, 1000).astype(np.uint16)
    return out


# ============================================================================== unique-level entries (cheap source)
@njit(parallel=True, cache=True)
def kn_count(Ic, Dc, u0, mptr, members, gc, n_genome, fl, out_cnt, nch):
    m = Ic.shape[0]
    for ch in prange(nch):
        lo = ch * m // nch
        hi = (ch + 1) * m // nch
        st = np.full(n_genome, -1, np.int64)
        for r in range(lo, hi):
            Uq = u0 + r
            k = 0
            for j in range(-1, Ic.shape[1]):
                V = Uq if j == -1 else Ic[r, j]
                if V < 0:
                    break
                if j >= 0 and V == Uq:
                    continue
                s = 1.0 if j == -1 else Dc[r, j]
                if s < fl:
                    break
                for t in range(mptr[V], mptr[V + 1]):
                    G = gc[members[t]]
                    if st[G] != Uq:
                        st[G] = Uq
                        k += 1
            out_cnt[Uq] = k


@njit(parallel=True, cache=True)
def kn_fill(Ic, Dc, u0, mptr, members, gc, n_genome, fl, uptr, o_gen, o_par, o_kb, o_ord, nch):
    m = Ic.shape[0]
    for ch in prange(nch):
        lo = ch * m // nch
        hi = (ch + 1) * m // nch
        st = np.full(n_genome, -1, np.int64)
        for r in range(lo, hi):
            Uq = u0 + r
            k = uptr[Uq]
            for j in range(-1, Ic.shape[1]):
                V = Uq if j == -1 else Ic[r, j]
                if V < 0:
                    break
                if j >= 0 and V == Uq:
                    continue
                s = 1.0 if j == -1 else Dc[r, j]
                if s < fl:
                    break
                kb = int(math.floor(s * 1000.0 + 1e-4))
                if kb > 1000:
                    kb = 1000
                for t in range(mptr[V], mptr[V + 1]):
                    b = members[t]
                    G = gc[b]
                    if st[G] != Uq:
                        st[G] = Uq
                        o_gen[k] = G
                        o_par[k] = b
                        o_kb[k] = kb
                        o_ord[k] = 65535 - (k - uptr[Uq]) if k - uptr[Uq] < 65535 else 0
                        k += 1


@njit(parallel=True, cache=True)
def unique_levels(uptr, gen, key, lin_of_genome, n_lin, lev, nch):
    """lev[e] = rank of entry e's genome among the sequence's genomes of the same genus by key (descending; ties by
    genome order), capped at 254."""
    U = len(uptr) - 1
    for ch in prange(nch):
        lo = ch * U // nch
        hi = (ch + 1) * U // nch
        cnt = np.zeros(n_lin, np.int64)
        for q in range(lo, hi):
            e0, e1 = uptr[q], uptr[q + 1]
            if e1 == e0:
                continue
            kk = np.empty(e1 - e0, np.int64)
            for k in range(e1 - e0):
                kk[k] = -np.int64(key[e0 + k])
            order = np.argsort(kk, kind="mergesort")
            for k in range(e1 - e0):
                e = e0 + order[k]
                lg = lin_of_genome[gen[e]]
                c = cnt[lg]
                lev[e] = c if c < 254 else 254
                cnt[lg] = c + 1
            for k in range(e1 - e0):
                cnt[lin_of_genome[gen[e0 + k]]] = 0


@njit(parallel=True, cache=True)
def gene_count(uq, lin, uptr, gen, lin_of_genome, out):
    n = len(uq)
    for a in prange(n):
        U = uq[a]
        la = lin[a]
        k = 0
        for e in range(uptr[U], uptr[U + 1]):
            if lin_of_genome[gen[e]] != la:
                k += 1
        out[a] = k


@njit(parallel=True, cache=True)
def gene_fill(uq, lin, uptr, gen, par, lev, kb, lin_of_genome, lptr, nb, nbg, glev, gkb):
    n = len(uq)
    for a in prange(n):
        U = uq[a]
        la = lin[a]
        k = lptr[a]
        for e in range(uptr[U], uptr[U + 1]):
            if lin_of_genome[gen[e]] == la:
                continue
            nb[k] = par[e]
            nbg[k] = gen[e]
            glev[k] = lev[e]
            gkb[k] = kb[e]
            k += 1


@njit(parallel=True, cache=True)
def gene_reach(lptr, nbg, gkb, glev, fbin, lin_of_genome, lin, tokg, famlin_ptr, famlin, n_lin, out, nch):
    """per gene and floor f: genera reached, family genera, family genera reached, entries kept, level-0 entries."""
    n = len(lptr) - 1
    F = len(fbin)
    for ch in prange(nch):
        lo = ch * n // nch
        hi = (ch + 1) * n // nch
        st = np.full((F, n_lin), -1, np.int64)
        for a in range(lo, hi):
            la = lin[a]
            for f in range(F):
                nr = 0
                ne = 0
                n0 = 0
                for e in range(lptr[a], lptr[a + 1]):
                    if gkb[e] < fbin[f]:
                        continue
                    ne += 1
                    if glev[e] == 0:
                        n0 += 1
                    lg = lin_of_genome[nbg[e]]
                    if st[f, lg] != a:
                        st[f, lg] = a
                        nr += 1
                t0 = tokg[a]
                nf = 0
                ri = 0
                for z in range(famlin_ptr[t0], famlin_ptr[t0 + 1]):
                    lz = famlin[z]
                    if lz == la:
                        continue
                    nf += 1
                    if st[f, lz] == a:
                        ri += 1
                out[a, f, 0] = nr
                out[a, f, 1] = nf
                out[a, f, 2] = ri
                out[a, f, 3] = ne
                out[a, f, 4] = n0


def cmd_lists(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    u = SR.Universe()
    meta = dict(np.load(R3 / "vec/meta.npz"))      # materialise: npz members re-read on every access otherwise
    uq = meta["uq"].astype(np.int64)
    U = int(uq.max()) + 1
    out = R3 / "lists" / a.src
    out.mkdir(parents=True, exist_ok=True)
    info = dict(src=a.src, floors=list(FLOORS))
    if a.src == "ex":
        ej = json.loads((R3 / "exact/exact.json").read_text())
        assert abs(float(ej["store_floor"]) - FLOORS[0]) < 1e-6, ej["store_floor"]
        counts = np.load(R3 / "exact/counts.npy").astype(np.int64)
        uptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        gen = np.fromfile(R3 / "exact/genome.bin", dtype=np.int16)
        par = np.fromfile(R3 / "exact/local.bin", dtype=np.int16).astype(np.int32)
        assert len(gen) == len(par) == uptr[-1], (len(gen), uptr[-1])
        sizes = (meta["g_end"] - meta["g_start"]).astype(np.int32)
        for s0 in range(0, len(par), CH):
            g_ = gen[s0:s0 + CH]
            assert (par[s0:s0 + CH] < sizes[g_]).all()
            par[s0:s0 + CH] += meta["g_start"][g_].astype(np.int32)
        key = np.fromfile(R3 / "exact/value.bin", dtype=np.uint16)        # float16 bit patterns (positive: ordered)
        kb = to_kbin(key)
        info["exact"] = ej
        # agreement with the float32 CPU calibration (10,000 queries x all genomes)
        qids = np.load(R3 / "calib/qids.npy")
        omax = np.load(R3 / "calib/omax.npy")
        oarg = np.load(R3 / "calib/oarg.npy")
        n_exp = n_found = n_arg = 0
        dmax = 0.0
        miss = []
        for r, q in enumerate(qids):
            e0, e1 = uptr[q], uptr[q + 1]
            cs = key[e0:e1].view(np.float16).astype(np.float32)
            got = {int(g): (int(p_) - int(meta["g_start"][g]), float(c_)) for g, p_, c_ in zip(gen[e0:e1], par[e0:e1], cs)}
            want = np.flatnonzero(omax[r] >= FLOORS[0] + 0.002)
            n_exp += len(want)
            for G in want:
                if int(G) in got:
                    n_found += 1
                    l_, c_ = got[int(G)]
                    dmax = max(dmax, abs(c_ - float(omax[r, G])))
                    n_arg += int(l_ == int(oarg[r, G]))
                else:
                    miss.append(float(omax[r, G]))
        info["gpu_vs_cpu"] = dict(queries=int(len(qids)), cpu_entries_above_0p322=int(n_exp), found=int(n_found),
                                  same_argmax=int(n_arg), max_abs_cos_diff=dmax, missing_cpu_cos_max=(max(miss) if miss else None))
        stamp("gpu vs cpu", info["gpu_vs_cpu"])
        del omax, oarg
    else:
        order = np.argsort(uq, kind="stable").astype(np.int64)
        mptr = np.r_[0, np.cumsum(np.bincount(uq, minlength=U))].astype(np.int64)
        I = np.load(RD / "softrec/knn/zca10__k2000__I.npy", mmap_mode="r")
        D = np.load(RD / "softrec/knn/zca10__k2000__D.npy", mmap_mode="r")
        counts = np.zeros(U, np.int64)
        CU = 250_000
        gc64 = u.gc.astype(np.int64)
        for u0 in range(0, U, CU):
            Ic = np.ascontiguousarray(I[u0:u0 + CU])
            Dc = np.ascontiguousarray(D[u0:u0 + CU]).astype(np.float32)
            kn_count(Ic, Dc, u0, mptr, order, gc64, u.ng, FLOORS[0], counts, a.threads * 8)
        uptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        gen = np.empty(uptr[-1], np.int16)
        par = np.empty(uptr[-1], np.int32)
        kb = np.empty(uptr[-1], np.uint16)
        key = np.empty(uptr[-1], np.uint16)             # list order (cosine order of the kNN) as a descending key
        for u0 in range(0, U, CU):
            Ic = np.ascontiguousarray(I[u0:u0 + CU])
            Dc = np.ascontiguousarray(D[u0:u0 + CU]).astype(np.float32)
            kn_fill(Ic, Dc, u0, mptr, order, gc64, u.ng, FLOORS[0], uptr, gen, par, kb, key, a.threads * 8)
        stamp("kn entries", f"{uptr[-1]:,}", f"{time.time() - t0:.0f}s")
    stamp("unique entries", f"{uptr[-1]:,}", f"{time.time() - t0:.0f}s rss {rss_gb()}")
    lev = np.empty(len(gen), np.uint8)
    unique_levels(uptr, gen, key, u.lin_of_genome.astype(np.int64), u.nl, lev, a.threads * 8)
    del key
    stamp("levels", f"{time.time() - t0:.0f}s")
    gcnt = np.zeros(u.n, np.int64)
    gene_count(uq, u.lin.astype(np.int64), uptr, gen, u.lin_of_genome.astype(np.int64), gcnt)
    lptr = np.r_[0, np.cumsum(gcnt)].astype(np.int64)
    E = int(lptr[-1])
    nb = np.empty(E, np.int32)
    nbg = np.empty(E, np.int16)
    glev = np.empty(E, np.uint8)
    gkb = np.empty(E, np.uint16)
    gene_fill(uq, u.lin.astype(np.int64), uptr, gen, par, lev, kb, u.lin_of_genome.astype(np.int64), lptr, nb, nbg, glev, gkb)
    del gen, par, lev, kb
    for nm, x in (("lptr", lptr), ("nb", nb), ("nbg", nbg), ("lev", glev), ("kb", gkb)):
        np.save(out / f"{nm}.npy", x)
    stamp("gene lists", f"{E:,} entries", f"{time.time() - t0:.0f}s rss {rss_gb()}")
    # Task 1: genera reached vs genera holding an MMseqs2 family member
    tok = S3.rescue3_tokens(u.n)
    keyf = np.unique(tok * u.nl + u.lin)
    kt, kl = keyf // u.nl, keyf % u.nl
    T = int(tok.max()) + 1
    famlin_ptr = np.searchsorted(kt, np.arange(T + 1)).astype(np.int64)
    R = np.zeros((u.n, len(FLOORS), 5), np.int32)
    gene_reach(lptr, nbg, gkb, glev, FBIN, u.lin_of_genome.astype(np.int64), u.lin.astype(np.int64), tok, famlin_ptr,
               kl.astype(np.int64), u.nl, R, a.threads * 8)
    np.save(out / "reach.npy", R)
    info["unique_entries"] = int(uptr[-1])
    info["unique_entries_per_sequence_q50_q90_q99"] = np.quantile(counts, [.5, .9, .99]).tolist()
    info["gene_entries"] = E
    info["gene_entries_per_gene_q50_q90_q99_max"] = np.quantile(gcnt, [.5, .9, .99, 1]).tolist()
    tt = tail_table(u.n)
    info["tail_rank_at_cos"] = {f"{c:g}": float(tt[int(round(c * 1000))]) for c in (0.32, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95)}
    for fi, fl in enumerate(FLOORS):
        info[f"floor_{fl:g}"] = dict(entries_per_gene_q50_q90=np.quantile(R[:, fi, 3], [.5, .9]).tolist(),
                                     entries_per_gene_mean=float(R[:, fi, 3].mean()),
                                     genera_per_gene_q50_mean=[float(np.median(R[:, fi, 0])), float(R[:, fi, 0].mean())],
                                     genomes_per_genus_reached_mean=float(R[:, fi, 3].sum() / max(1, R[:, fi, 0].sum())),
                                     reach=S3.reach_summary(R[:, fi, :3]))
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_gb"] = rss_gb()
    S3.jdump(out / "lists.json", info)
    stamp("lists", json.dumps(info, default=float))


class Lists3:
    def __init__(self, src):
        d = R3 / "lists" / src
        self.lptr = np.load(d / "lptr.npy")
        self.nb = np.load(d / "nb.npy")
        self.nbg = np.load(d / "nbg.npy")
        self.lev = np.load(d / "lev.npy")
        self.kb = np.load(d / "kb.npy")


# ============================================================================== passes
@njit(parallel=True, cache=True)
def excl3(n, gene_at, nd_arr, inst_off, uq, X, floors, out):
    dim = X.shape[1]
    F = len(floors)
    for i in prange(n):
        nd = nd_arr[i]
        if nd == 0:
            continue
        a = gene_at[i]
        for d in range(1, nd + 1):
            b = gene_at[i + d]
            inst = inst_off[i] + d - 1
            if uq[a] == uq[b]:
                for f in range(F):
                    out[f, inst] = True
                continue
            c = 0.0
            ua, ub = uq[a], uq[b]
            for k in range(dim):
                c += X[ua, k] * X[ub, k]
            for f in range(F):
                out[f, inst] = c >= floors[f]


@njit(parallel=True, cache=True)
def pass3(n, w, gene_at, slot_of, cid, lptr, nb, nbg, lev, kb, ttab, lin_of_genome, vf, vm, fbin, rw, excl, inst_off,
          nd_arr, n_genome, n_lin, out_L, out_W, nchunks):
    V = len(vf)
    J = len(rw)
    for ch in prange(nchunks):
        lo = ch * n // nchunks
        hi = (ch + 1) * n // nchunks
        Ast = np.full(n_genome, -1, np.int64)
        Aslot = np.zeros(n_genome, np.int64)
        Alev = np.zeros(n_genome, np.int64)
        Akb = np.zeros(n_genome, np.int64)
        lst = np.full((V, n_lin), -1, np.int64)
        lmin = np.zeros((V, n_lin), np.float64)
        linlist = np.zeros((V, n_lin), np.int32)
        cntL = np.zeros(V, np.int32)
        for i in range(lo, hi):
            nd = nd_arr[i]
            if nd == 0:
                continue
            a = gene_at[i]
            for e in range(lptr[a], lptr[a + 1]):
                G = nbg[e]
                Ast[G] = i
                Aslot[G] = slot_of[nb[e]]
                Alev[G] = lev[e]
                Akb[G] = kb[e]
            for d in range(1, nd + 1):
                b = gene_at[i + d]
                inst = inst_off[i] + d - 1
                for v in range(V):
                    cntL[v] = 0
                for e in range(lptr[b], lptr[b + 1]):
                    G = nbg[e]
                    if Ast[G] != i:
                        continue
                    sb = slot_of[nb[e]]
                    sa = Aslot[G]
                    if sa == sb:
                        continue
                    dd = sa - sb
                    if dd > w or dd < -w:
                        continue
                    if cid[sa] != cid[sb]:
                        continue
                    lg = lin_of_genome[G]
                    ka = Akb[G]
                    kbb = np.int64(kb[e])
                    kmin = ka if ka < kbb else kbb
                    tmx = ttab[kmin]
                    la_ = Alev[G]
                    lb_ = np.int64(lev[e])
                    mn = la_ if la_ < lb_ else lb_
                    for v in range(V):
                        if excl[vf[v], inst]:
                            continue
                        if kmin < fbin[vf[v]]:
                            continue
                        if mn >= vm[v]:
                            continue
                        if lst[v, lg] != inst:
                            lst[v, lg] = inst
                            lmin[v, lg] = tmx
                            linlist[v, cntL[v]] = lg
                            cntL[v] += 1
                        elif tmx < lmin[v, lg]:
                            lmin[v, lg] = tmx
                for v in range(V):
                    if excl[vf[v], inst]:
                        out_L[v, inst] = -1
                        for j in range(J):
                            out_W[v, j, inst] = 0.0
                        continue
                    out_L[v, inst] = cntL[v]
                    for j in range(J):
                        s = 0.0
                        for k in range(cntL[v]):
                            s += math.exp(-(lmin[v, linlist[v, k]] - 1.0) / rw[j])
                        out_W[v, j, inst] = s


def load_X():
    return np.load(R3 / "vec/zca10_f16.npy").astype(np.float32)


def run_pass(u, L, X, uq, ttab, gene_at, slot_of, nd, off, n_inst, threads, label):
    t0 = time.time()
    excl = np.zeros((len(FLOORS), n_inst), np.bool_)
    excl3(u.n, gene_at, nd, off, uq, X, np.array(FLOORS, np.float64), excl)
    vs = variants()
    vf = np.array([x[0] for x in vs], np.int64)
    vm = np.array([x[1] for x in vs], np.int64)
    oL = np.zeros((len(vs), n_inst), np.int16)
    oW = np.zeros((len(vs), len(RW), n_inst), np.float32)
    pass3(u.n, W, gene_at, slot_of, u.cid.astype(np.int64), L.lptr, L.nb, L.nbg, L.lev, L.kb, ttab,
          u.lin_of_genome.astype(np.int64), vf, vm, FBIN, np.array(RW, np.float64), excl, off, nd, u.ng, u.nl, oL, oW,
          threads * 8)
    stamp(f"pass {label}: {time.time() - t0:.0f}s; excluded per floor", [int(x.sum()) for x in excl], f"rss {rss_gb()}")
    return dict(L=oL, W=oW, excl=excl)


def brute3(u, L, ttab, P, nd, off, rng, m=300):
    """python recount of L and W for random observed instances."""
    n_inst = P["L"].shape[1]
    left_all = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    pick = rng.choice(n_inst, m, replace=False)
    bad, maxd = 0, 0.0
    vs = variants()
    for inst in pick:
        i = int(left_all[inst])
        a, b = i, i + int(inst - off[i] + 1)
        la = int(u.lin[a])
        A = {int(L.nbg[e]): e for e in range(L.lptr[a], L.lptr[a + 1])}
        for v, (fi, mcap) in enumerate(vs):
            if P["L"][v, inst] < 0:
                continue
            lins = {}
            for e in range(L.lptr[b], L.lptr[b + 1]):
                G = int(L.nbg[e])
                if G not in A:
                    continue
                ea = A[G]
                sa, sb = int(L.nb[ea]), int(L.nb[e])
                if sa == sb or abs(sa - sb) > W or u.cid[sa] != u.cid[sb]:
                    continue
                if min(int(L.kb[ea]), int(L.kb[e])) < FBIN[fi] or min(int(L.lev[ea]), int(L.lev[e])) >= mcap:
                    continue
                lg = int(u.lin_of_genome[G])
                assert lg != la
                tm = float(ttab[min(int(L.kb[ea]), int(L.kb[e]))])
                lins[lg] = min(lins.get(lg, 1e30), tm)
            if len(lins) != int(P["L"][v, inst]):
                bad += 1
                continue
            for j, r0 in enumerate(RW):
                want = sum(math.exp(-(x - 1.0) / r0) for x in lins.values())
                got = float(P["W"][v, j, inst])
                maxd = max(maxd, abs(got - want) / max(1.0, want))
                if abs(got - want) > 1e-3 * max(1.0, want):
                    bad += 1
    return dict(instances=int(m), mismatches=bad, max_rel_diff=maxd)


def cmd_stats(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    out = R3 / "stats" / a.src
    out.mkdir(parents=True, exist_ok=True)
    u = SR.Universe()
    nd, off, left, d = u.instances(W)
    n_inst = len(left)
    uq = np.load(R3 / "vec/meta.npz")["uq"].astype(np.int64)
    ttab = tail_table(u.n)
    L = Lists3(a.src)
    X = load_X()
    stamp("loaded", f"{time.time() - t0:.0f}s rss {rss_gb()}")
    ident = np.arange(u.n, dtype=np.int64)
    man = dict(src=a.src, variants=[vname(*x) for x in variants()], rw=list(RW), n_instances=int(n_inst),
               slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, threads=a.threads)
    t1 = time.time()
    obs = run_pass(u, L, X, uq, ttab, ident, ident, nd, off, n_inst, a.threads, "observed")
    man["seconds_observed_pass"] = round(time.time() - t1, 1)
    man["brute_observed"] = brute3(u, L, ttab, obs, nd, off, np.random.default_rng(17), 300)
    stamp("brute", man["brute_observed"])
    np.savez(out / "obs.npz", L=obs["L"], W=obs["W"], excl=obs["excl"])
    nullL = []
    for r in range(a.perms):
        t1 = time.time()
        ga, so = SR.null_order(u, r)
        p = run_pass(u, L, X, uq, ttab, ga, so, nd, off, n_inst, a.threads, f"null {r}")
        np.savez(out / f"null{r}.npz", L=p["L"], W=p["W"], excl=p["excl"])
        man[f"seconds_null_{r}"] = round(time.time() - t1, 1)
        nullL.append(p["L"])
        del p
    man["support"] = {}
    for v, x in enumerate(variants()):
        o = obs["L"][v].astype(np.int64)
        ok = o >= 0
        row = dict(excluded=int((~ok).sum()), obs_Lo_mean=float(o[ok].mean()), obs_Lo_q50_q90_q99=np.quantile(o[ok], [.5, .9, .99]).tolist())
        nl = [n_[v].astype(np.int64) for n_ in nullL]
        row["null_Lo_mean"] = float(np.mean([z[z >= 0].mean() for z in nl]))
        row["null_Lo_q99_q999"] = np.quantile(nl[0][nl[0] >= 0], [.99, .999]).tolist()
        for m_ in (1, 2, 3, 4, 5, 8, 15, 30):
            row[f"obs_Lo_ge{m_}"] = int((o >= m_).sum())
            row[f"null_Lo_ge{m_}"] = float(np.mean([(z >= m_).sum() for z in nl]))
        man["support"][vname(*x)] = row
    man["seconds"] = round(time.time() - t0, 1)
    man["peak_rss_gb"] = rss_gb()
    S3.jdump(out / "stats.json", man)
    stamp("end stats", json.dumps(man["support"]), f"rss {rss_gb()}")


# ============================================================================== candidates
def load_pass(fn):
    z = np.load(fn)
    return dict(L=z["L"], W=z["W"])


def rule_masks(kind, P, v, j=None, only=None):
    Lo = P["L"][v].astype(np.int64)
    Lt = np.where(Lo >= 0, Lo + 1, -1)
    if kind == "cnt":
        for m in COUNT_M:
            name = f"L>={m}"
            if only is None or name == only:
                yield name, Lt >= m
    else:
        Wv = P["W"][v, j]
        for t in W_T:
            name = f"W>={t:g}"
            if only is None or name == only:
                yield name, (Lt >= 2) & (Wv >= t)


def select(n, obs, nulls, kind, v, j, left, d):
    nc = [{name: int(m.sum()) for name, m in rule_masks(kind, p, v, j)} for p in nulls]
    rows, masks = [], {}
    for name, m in rule_masks(kind, obs, v, j):
        n_obs = int(m.sum())
        n_null = float(np.mean([c[name] for c in nc]))
        rows.append(dict(rule=name, observed=n_obs, null_mean=n_null, fdr=(n_null / n_obs if n_obs else np.nan)))
        masks[name] = m
    rules = pd.DataFrame(rows)
    cand = rules[(rules.fdr <= FDR) & (rules.observed > 0)].sort_values("observed", ascending=False).head(12)
    best, best_cov = None, -1
    for name in cand.rule:
        g = np.zeros(n, bool)
        g[left[masks[name]]] = True
        g[left[masks[name]] + d[masks[name]]] = True
        cov = int(g.sum())
        rules.loc[rules.rule == name, "genes_covered"] = cov
        if cov > best_cov:
            best, best_cov = name, cov
    return rules, best, (masks[best] if best else None)


def all_cands(src):
    out = {}
    for v, (fi, m) in enumerate(variants()):
        out[f"{src}_{vname(fi, m)}_cnt"] = dict(v=v, fi=fi, m=m, kind="cnt", j=None)
        for j, r0 in enumerate(RW):
            out[f"{src}_{vname(fi, m)}_rw{r0:g}"] = dict(v=v, fi=fi, m=m, kind="rw", j=j)
    return out


@njit(parallel=True, cache=True)
def validated3(sel_left, sel_d, fb, mcap, n, w, cid, lptr, nb, nbg, lev, kb, n_genome, valid, nchunks):
    m = len(sel_left)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        Ast = np.full(n_genome, -1, np.int64)
        Aent = np.zeros(n_genome, np.int64)
        for q in range(lo, hi):
            a = sel_left[q]
            b = a + sel_d[q]
            for e in range(lptr[a], lptr[a + 1]):
                Ast[nbg[e]] = q
                Aent[nbg[e]] = e
            for e in range(lptr[b], lptr[b + 1]):
                G = nbg[e]
                if Ast[G] != q:
                    continue
                ea = Aent[G]
                sa = nb[ea]
                sb = nb[e]
                if sa == sb or sa - sb > w or sb - sa > w or cid[sa] != cid[sb]:
                    continue
                if kb[ea] < fb or kb[e] < fb:
                    continue
                if (lev[ea] if lev[ea] < lev[e] else lev[e]) >= mcap:
                    continue
                valid[ea] = True
                valid[e] = True


@njit(parallel=True, cache=True)
def fill_keys3(valid, lptr, nb, uq, U, off, out):
    n = len(lptr) - 1
    for a in prange(n):
        k = off[a]
        ua = uq[a]
        for e in range(lptr[a], lptr[a + 1]):
            if valid[e]:
                ub = uq[nb[e]]
                if ua < ub:
                    out[k] = ua * U + ub
                elif ub < ua:
                    out[k] = ub * U + ua
                else:
                    out[k] = -1
                k += 1


@njit(parallel=True, cache=True)
def row_valid(valid, lptr, out):
    n = len(lptr) - 1
    for a in prange(n):
        k = 0
        for e in range(lptr[a], lptr[a + 1]):
            if valid[e]:
                k += 1
        out[a] = k


def extract_edges3(u, L, uq, X, sidx, left, d, fi, mcap, threads, out):
    from softrec2 import edge_cos, union_find_cc
    t0 = time.time()
    valid = np.zeros(len(L.nb), np.bool_)
    validated3(left[sidx].astype(np.int64), d[sidx].astype(np.int64), int(FBIN[fi]), int(mcap), u.n, W, u.cid.astype(np.int64),
               L.lptr, L.nb, L.nbg, L.lev, L.kb, u.ng, valid, threads * 8)
    per = np.zeros(u.n, np.int64)
    row_valid(valid, L.lptr, per)
    off = np.r_[0, np.cumsum(per)].astype(np.int64)
    U = int(uq.max()) + 1
    keys = np.empty(int(off[-1]), np.int64)
    fill_keys3(valid, L.lptr, L.nb, uq, U, off[:-1].copy(), keys)
    del valid
    keys = keys[keys >= 0]
    n_dir = int(len(keys))
    keys = np.unique(keys)
    ea = (keys // U).astype(np.int32)
    eb = (keys % U).astype(np.int32)
    del keys
    wgt = np.empty(len(ea), np.float32)
    edge_cos(ea, eb, X, wgt)
    cc = union_find_cc(U, ea.astype(np.int64), eb.astype(np.int64))
    _, cc = np.unique(cc, return_inverse=True)
    sizes = np.bincount(cc)
    deg = np.bincount(ea, minlength=U) + np.bincount(eb, minlength=U)
    np.save(out / "edges_a.npy", ea)
    np.save(out / "edges_b.npy", eb)
    np.save(out / "edges_w.npy", wgt)
    np.save(out / "fam_cc.npy", cc.astype(np.int32))
    np.save(out / "valid_per_gene.npy", per.astype(np.int32))
    _, first = np.unique(uq, return_index=True)
    lin_u = u.lin[first]
    info = dict(valid_entries=int(off[-1]), genes_with_valid_entry=int((per > 0).sum()),
                directed_unique_level_nonself=n_dir, undirected_unique_edges=int(len(ea)), unique_nodes_with_edge=int((deg > 0).sum()),
                degree_q50_q90_q99_max=np.quantile(deg[deg > 0], [0.5, 0.9, 0.99, 1]).tolist() if len(ea) else [],
                cosine_q01_q10_q50=np.quantile(wgt, [0.01, 0.1, 0.5]).tolist() if len(wgt) else [],
                cc_components=int(len(sizes)), cc_nontrivial=int((sizes >= 2).sum()), cc_largest=int(sizes.max()),
                cc_largest_share_of_nodes_with_edge=float(sizes.max() / max(1, (deg > 0).sum())),
                edges_same_genus_share=float((lin_u[ea] == lin_u[eb]).mean()) if len(ea) else None,
                seconds=round(time.time() - t0, 1))
    stamp("edges", json.dumps(info), f"rss {rss_gb()}")
    return info


def cmd_cands(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    u = SR.Universe()
    nd, off, left, d = u.instances(W)
    sdir = R3 / "stats" / a.src
    obs = load_pass(sdir / "obs.npz")
    nulls = [load_pass(sdir / f"null{r}.npz") for r in range(3)]
    allc = all_cands(a.src)
    names = [x for x in a.names.split(",") if x] or list(allc)
    edge_names = [x for x in a.edges.split(",") if x]
    L = X = uq = None
    if edge_names:
        L = Lists3(a.src)
        X = load_X()
        uq = np.load(R3 / "vec/meta.npz")["uq"].astype(np.int64)
    for name in names:
        spec = allc[name]
        cd = R3 / "cand" / name
        if (cd / "cand.json").exists() and name not in edge_names and not a.redo:
            continue
        t1 = time.time()
        rules, best, sig = select(u.n, obs, nulls, spec["kind"], spec["v"], spec["j"], left, d)
        if best is None:
            stamp(name, "no rule passes")
            continue
        cd.mkdir(parents=True, exist_ok=True)
        Lo = obs["L"][spec["v"]].astype(np.int64)
        Lt = np.where(Lo >= 0, Lo + 1, -1)
        sidx = np.flatnonzero(sig)
        sidx = sidx[np.lexsort((left[sidx], d[sidx]))]
        val = Lt[sidx].astype(np.float32)
        assert (val >= 2).all()
        b, arg = SR.boundary_support_argmax(left[sidx], d[sidx], val, np.arange(len(sidx), dtype=np.int64), u.n)
        g = np.zeros(u.n, bool)
        g[left[sidx]] = True
        g[left[sidx] + d[sidx]] = True
        row = rules[rules.rule == best].iloc[0]
        info = dict(name=name, src=a.src, variant=vname(spec["fi"], spec["m"]), floor=FLOORS[spec["fi"]],
                    mcap=spec["m"], kind=spec["kind"], rw=(RW[spec["j"]] if spec["j"] is not None else None), rule=best,
                    fdr=float(row.fdr), null_mean=float(row.null_mean), significant_instances=int(len(sidx)),
                    genes_covered=int(g.sum()), genes_covered_frac=float(g.mean()), supported_boundaries=int((b > 0).sum()),
                    L_q10_q50_q90=np.quantile(val, [0.1, 0.5, 0.9]).tolist(), seconds_select=round(time.time() - t1, 1))
        for k_, x in (("sidx", sidx), ("value", val.astype(np.int16)), ("cap", np.zeros(len(sidx), np.int16)), ("b", b), ("arg", arg)):
            np.save(cd / f"{k_}.npy", x)
        rules.to_csv(cd / "rules.tsv", sep="\t", index=False)
        if name in edge_names:
            info["edges"] = extract_edges3(u, L, uq, X, sidx, left, d, spec["fi"], spec["m"], a.threads, cd)
        S3.jdump(cd / "cand.json", info)
        stamp("candidate", json.dumps({k_: v_ for k_, v_ in info.items() if k_ != "edges"}, default=float))
    stamp("end cands", round(time.time() - t0, 1), f"rss {rss_gb()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("lists", "stats", "cands"))
    ap.add_argument("--src", default="ex")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--names", default="")
    ap.add_argument("--edges", default="")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    {"lists": cmd_lists, "stats": cmd_stats, "cands": cmd_cands}[a.cmd](a)


if __name__ == "__main__":
    main()
