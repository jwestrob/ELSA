#!/usr/bin/env python
"""ws_vocab2b: words whose identity is the set of instances aligned to them (DPANN).

Genomes are unsegmented text, but no node has a label. A node (a gene, later a word instance) carries a LIST: its aligned
instances in other genomes. A terminal gene's list is its zca10 neighbour list (softrec Lists, depth K, own genome or own
group excluded). Two adjacent live nodes x, y (<= 2 intervening single genes) are supported by
    S(x, y) = {(p, q): p in list(x), q in list(y), p, q on one contig of another genome, disjoint, <= 2 genes apart}
(per q the best-ranked p). n = 1 + supporting genomes, E = own term + sum_G min(1, 6 c_x(G) c_y(G) / m_G),
Lambda = n ln(n/E) - n + E (stage 1's code-length saving). Evidence units:
    ctx      Lambda_eff = Lambda * N_eff / n; N_eff = n / (1 + (n-1) rho), rho from the mean pairwise cosine of genomic-context
             vectors (W 50 window sums, own genes excluded) of the instance and <= 32 supporting genomes, noise floor c0, sd0
             removed (z 2); N_eff >= 2
    ctxshuf  as ctx with the context of a random slot (control)
    gen      Lambda over genomes, no correction (diagnostic)
    grp      lists outside the own group; Lambda over data-derived genome groups (reference)
A committed merge becomes a word node whose list is S (spans in other genomes; rank = worse rank of its parts), so the next
round aligns words against all genomes at once. Greedy commit in decreasing Lambda_eff, rounds until nothing commits.
tau: smallest ladder value with round-0 instance FDR (3 gene-order shuffles) <= 0.01.

List sources (amendment_1): emb (zca10 kNN lists, depth 1,000 / 200), aln (MMseqs2 hits, control), z10.6 / z13.7
(ws_calib per-genome best matches above a z cut). Contexts, N_eff and the round-0 homolog exclusion mask always come from
the zca10 vectors / embedding lists, so arms differ only in list membership and ranks.

usage (biotite):
  vl.py smoke                                   synthetic universe; kernels vs brute force (login node, seconds)
  vl.py refsr --threads T                       genus lists: round-0 pass == $RD/softrec/r2/stats/obs.npz (K1000)
  vl.py job --excl genome --arms ctx,ctxshuf,gen,ctx200 --threads T [--full-nulls ctx,ctx200]   (writes exclusion masks)
  vl.py job --excl group --arms grp --refsr --threads T --full-nulls grp
  vl.py job --excl genome --arms z10.6,z13.7 --threads T          (after the genome job: reads its exclusion masks)
  vl.py job --excl genome --arms aln --threads T                  (after the MMseqs2 search)
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

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VL = RD / "vocab_lists"
GROUPS = RD / "lineagefree/groups/emb_hom_average_t0.75.parquet"
GENOME_MAP = RD / "lineagefree/groups/genome_map.parquet"
WGAP = 3            # positions: <= 2 intervening genes
CTX_W = 50
M_SAMPLE = 32
ZNOISE = 2.0
LADDER = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
FDR = 0.01
MIN_NEFF = 2.0
MAXR = 200


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


def lam_np(n, E):
    n = np.asarray(n, np.float64)
    E = np.maximum(np.asarray(E, np.float64), 1e-300)
    out = n * np.log(np.maximum(n, 1e-300) / E) - n + E
    return np.where(n > E, out, 0.0)


# ============================================================================== list access
@njit(cache=True)
def node_list(j, nterm, ns, lslot, gene_at, lstart, tcnt, wptr):
    if nterm[j]:
        g = gene_at[ns[j]]
        return True, lstart[g], np.int64(tcnt[g])
    q = lslot[j]
    return False, wptr[q], wptr[q + 1] - wptr[q]


@njit(cache=True)
def entry(isterm, base, t, nbs, tail, nbg, ws, we, wr, wg):
    if isterm:
        s = np.int64(nbs[base + t])
        return s, s, np.int64(tail[base + t]), np.int64(nbg[base + t])
    return np.int64(ws[base + t]), np.int64(we[base + t]), np.int64(wr[base + t]), np.int64(wg[base + t])


@njit(cache=True)
def mark(stv, isx, bx, cx, nbs, tail, nbg, ws, we, wr, wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast):
    """Mark the positions right of / left of every entry of x (<= WGAP away, same contig). First (best-ranked) wins."""
    for t in range(cx):
        s, e, r, G = entry(isx, bx, t, nbs, tail, nbg, ws, we, wr, wg)
        c = cid[s]
        u = e + 1
        while u <= e + 3 and u < n:
            if cid[u] != c:
                break
            if Rst[u] != stv:
                Rst[u] = stv
                Rsrc[u] = t
            u += 1
        u = s - 1
        while u >= s - 3 and u >= 0:
            if cid[u] != c:
                break
            if Lst[u] != stv:
                Lst[u] = stv
                Lsrc[u] = t
            u -= 1
        if cast[G] != stv:
            cast[G] = stv
            ca[G] = 0
        ca[G] += 1


@njit(cache=True)
def best_p(stv, s, e, Rst, Rsrc, Lst, Lsrc, isx, bx, nbs, tail, nbg, ws, we, wr, wg):
    """For q = [s, e]: the best-ranked p of x adjacent to q (p left of q via R marks at s, right of q via L marks at e)."""
    tR = Rsrc[s] if Rst[s] == stv else -1
    tL = Lsrc[e] if Lst[e] == stv else -1
    if tR < 0 and tL < 0:
        return -1, 0, 0, 0
    if tR >= 0 and tL >= 0:
        _, _, rR, _ = entry(isx, bx, tR, nbs, tail, nbg, ws, we, wr, wg)
        _, _, rL, _ = entry(isx, bx, tL, nbs, tail, nbg, ws, we, wr, wg)
        t = tR if rR <= rL else tL
    else:
        t = tR if tR >= 0 else tL
    ps, pe, pr, _ = entry(isx, bx, t, nbs, tail, nbg, ws, we, wr, wg)
    return t, ps, pe, pr


# ============================================================================== pass A: counts
@njit(parallel=True, cache=True)
def pass_count(m, ns, nterm, lslot, aptr, ak, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr, wg,
               cid, grp, ngenome, ngroup, mg, out_n, out_ng, out_E, out_Eg, nchunks):
    n = len(cid)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        Rst = np.full(n, -1, np.int64)
        Lst = np.full(n, -1, np.int64)
        Rsrc = np.zeros(n, np.int32)
        Lsrc = np.zeros(n, np.int32)
        ca = np.zeros(ngenome, np.int64)
        cast = np.full(ngenome, -1, np.int64)
        cb = np.zeros(ngenome, np.int64)
        cbst = np.full(ngenome, -1, np.int64)
        tb = np.zeros(ngenome, np.int64)
        gst = np.full(ngenome, -1, np.int64)
        grst = np.full(ngroup, -1, np.int64)
        grst2 = np.full(ngroup, -1, np.int64)
        grlp = np.zeros(ngroup, np.float64)
        grt = np.zeros(ngroup, np.int64)
        for j in range(lo, hi):
            if aptr[j + 1] == aptr[j]:
                continue
            isx, bx, cx = node_list(j, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
            mark(j, isx, bx, cx, nbs, tail, nbg, ws, we, wr, wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast)
            for r in range(aptr[j], aptr[j + 1]):
                k = ak[r]
                isy, by, cy = node_list(k, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
                ntb = 0
                nsup = 0
                ngs = 0
                for t in range(cy):
                    s, e, rk, G = entry(isy, by, t, nbs, tail, nbg, ws, we, wr, wg)
                    if cbst[G] != r:
                        cbst[G] = r
                        cb[G] = 0
                        tb[ntb] = G
                        ntb += 1
                    cb[G] += 1
                    if Rst[s] == j or Lst[e] == j:
                        if gst[G] != r:
                            gst[G] = r
                            nsup += 1
                            gg = grp[G]
                            if grst2[gg] != r:
                                grst2[gg] = r
                                ngs += 1
                Es = 0.0
                ngt = 0
                for q in range(ntb):
                    G = tb[q]
                    if cast[G] == j and ca[G] > 0:
                        x = 2.0 * WGAP * ca[G] * cb[G] / mg[G]
                        p = x if x < 1.0 else 1.0
                        Es += p
                        gg = grp[G]
                        if grst[gg] != r:
                            grst[gg] = r
                            grlp[gg] = 0.0
                            grt[ngt] = gg
                            ngt += 1
                        if p >= 1.0:
                            grlp[gg] = -1e300
                        else:
                            grlp[gg] += math.log1p(-p)
                Eg = 0.0
                for q in range(ngt):
                    Eg += 1.0 - math.exp(grlp[grt[q]])
                out_n[r] = nsup
                out_ng[r] = ngs
                out_E[r] = Es
                out_Eg[r] = Eg


# ============================================================================== contexts
@njit(cache=True)
def ctx_vec(s, e, W, cid, c_start, c_end, P, sig, shuf, v):
    """Window sum of gene vectors [s-W, s-1] u [e+1, e+W] inside the contig (own genes excluded). Returns n_ctx."""
    if shuf:
        s = sig[s]
        e = s
    c = cid[s]
    cs = c_start[c]
    ce = c_end[c]
    d = P.shape[1]
    for k in range(d):
        v[k] = 0.0
    nctx = 0
    a0 = s - W
    if a0 < cs:
        a0 = cs
    a1 = s - 1
    if a1 >= a0:
        r1 = a1 + 1 + c
        r0 = a0 + c
        for k in range(d):
            v[k] += P[r1, k] - P[r0, k]
        nctx += a1 - a0 + 1
    b0 = e + 1
    b1 = e + W
    if b1 > ce - 1:
        b1 = ce - 1
    if b1 >= b0:
        r1 = b1 + 1 + c
        r0 = b0 + c
        for k in range(d):
            v[k] += P[r1, k] - P[r0, k]
        nctx += b1 - b0 + 1
    return nctx


@njit(cache=True)
def add_unit(v, T):
    nn = 0.0
    for k in range(len(v)):
        nn += v[k] * v[k]
    if nn <= 0.0:
        return False
    inv = 1.0 / math.sqrt(nn)
    for k in range(len(v)):
        T[k] += v[k] * inv
    return True


@njit(parallel=True, cache=True)
def prefix_sums(X, uq, gene_at, c_start, c_end, P):
    d = X.shape[1]
    for c in prange(len(c_start)):
        row = c_start[c] + c
        for k in range(d):
            P[row, k] = 0.0
        for i in range(c_start[c], c_end[c]):
            u = uq[gene_at[i]]
            if u >= 0:
                for k in range(d):
                    P[row + 1, k] = P[row, k] + X[u, k]
            else:
                for k in range(d):
                    P[row + 1, k] = P[row, k]
            row += 1


@njit(parallel=True, cache=True)
def floor_cos(ia, ib, W, cid, c_start, c_end, P, sig, shuf, out, nchunks):
    m = len(ia)
    d = P.shape[1]
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        va = np.zeros(d)
        vb = np.zeros(d)
        for t in range(lo, hi):
            na = ctx_vec(ia[t], ia[t] + 1, W, cid, c_start, c_end, P, sig, shuf, va)
            nb = ctx_vec(ib[t], ib[t] + 1, W, cid, c_start, c_end, P, sig, shuf, vb)
            if na == 0 or nb == 0:
                out[t] = np.nan
                continue
            sa = 0.0
            sb = 0.0
            sab = 0.0
            for k in range(d):
                sa += va[k] * va[k]
                sb += vb[k] * vb[k]
                sab += va[k] * vb[k]
            if sa <= 0 or sb <= 0:
                out[t] = np.nan
            else:
                out[t] = sab / math.sqrt(sa * sb)


# ============================================================================== pass B: N_eff
@njit(parallel=True, cache=True)
def pass_neff(m, ns, ne, nterm, lslot, aptr, ak, need, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr, wg,
              cid, c_start, c_end, ngenome, prio, P, W, sig, shuf, msamp, c0, sd0, z, kmax,
              out_neff, out_k, out_cbar, nchunks):
    n = len(cid)
    d = P.shape[1]
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        Rst = np.full(n, -1, np.int64)
        Lst = np.full(n, -1, np.int64)
        Rsrc = np.zeros(n, np.int32)
        Lsrc = np.zeros(n, np.int32)
        ca = np.zeros(ngenome, np.int64)
        cast = np.full(ngenome, -1, np.int64)
        gst = np.full(ngenome, -1, np.int64)
        gbs = np.zeros(ngenome, np.int64)
        gbe = np.zeros(ngenome, np.int64)
        gbr = np.zeros(ngenome, np.int64)
        glist = np.zeros(ngenome, np.int64)
        gp = np.zeros(ngenome, np.int64)
        T = np.zeros(d)
        v = np.zeros(d)
        for j in range(lo, hi):
            anyneed = False
            for r in range(aptr[j], aptr[j + 1]):
                if need[r]:
                    anyneed = True
                    break
            if not anyneed:
                continue
            isx, bx, cx = node_list(j, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
            mark(j, isx, bx, cx, nbs, tail, nbg, ws, we, wr, wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast)
            for r in range(aptr[j], aptr[j + 1]):
                if not need[r]:
                    continue
                k = ak[r]
                isy, by, cy = node_list(k, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
                ng = 0
                for t in range(cy):
                    s, e, rk, G = entry(isy, by, t, nbs, tail, nbg, ws, we, wr, wg)
                    tp, ps, pe, pr = best_p(j, s, e, Rst, Rsrc, Lst, Lsrc, isx, bx, nbs, tail, nbg, ws, we, wr, wg)
                    if tp < 0:
                        continue
                    ms = ps if ps < s else s
                    me = pe if pe > e else e
                    mr = pr if pr > rk else rk
                    if gst[G] != r:
                        gst[G] = r
                        glist[ng] = G
                        ng += 1
                        gbs[G] = ms
                        gbe[G] = me
                        gbr[G] = mr
                    elif mr < gbr[G] or (mr == gbr[G] and ms < gbs[G]):
                        gbs[G] = ms
                        gbe[G] = me
                        gbr[G] = mr
                nn = ng + 1
                # genome sample: the msamp supporting genomes of smallest priority
                if ng > msamp:
                    for q in range(ng):
                        gp[q] = prio[glist[q]]
                    o = np.argsort(gp[:ng])
                    sel = glist[:ng][o[:msamp]].copy()
                else:
                    sel = glist[:ng].copy()
                for kk in range(d):
                    T[kk] = 0.0
                kinf = 0
                nc = ctx_vec(ns[j], ne[k], W, cid, c_start, c_end, P, sig, shuf, v)
                if nc > 0 and add_unit(v, T):
                    kinf += 1
                for q in range(len(sel)):
                    G = sel[q]
                    nc = ctx_vec(gbs[G], gbe[G], W, cid, c_start, c_end, P, sig, shuf, v)
                    if nc > 0 and add_unit(v, T):
                        kinf += 1
                rho = 0.0
                cbar = np.nan
                if kinf >= 2:
                    tt = 0.0
                    for kk in range(d):
                        tt += T[kk] * T[kk]
                    cbar = (tt - kinf) / (kinf * (kinf - 1.0))
                    rho = (cbar - c0 - z * sd0 * math.sqrt(2.0 / (kinf * (kinf - 1.0)))) / (1.0 - c0)
                    if rho < 0.0:
                        rho = 0.0
                    if rho > 1.0:
                        rho = 1.0
                out_neff[r] = nn / (1.0 + (nn - 1.0) * rho)
                out_k[r] = kinf
                out_cbar[r] = cbar


# ============================================================================== syn: synteny-retention N_eff
@njit(parallel=True, cache=True)
def syn_matrix(ng, g_start, g_end, aptr, ak, ns, nterm, lslot, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr, wg,
               cid, out_cnt, out_tot, nchunks):
    """Round 0 (nodes = slots): per genome g, out_tot[g] = adjacencies with >= 1 supporting genome, out_cnt[g, h] =
    those supported by genome h."""
    n = len(cid)
    for ch in prange(nchunks):
        lo = ch * ng // nchunks
        hi = (ch + 1) * ng // nchunks
        Rst = np.full(n, -1, np.int64)
        Lst = np.full(n, -1, np.int64)
        Rsrc = np.zeros(n, np.int32)
        Lsrc = np.zeros(n, np.int32)
        ca = np.zeros(ng, np.int64)
        cast = np.full(ng, -1, np.int64)
        gst = np.full(ng, -1, np.int64)
        for g in range(lo, hi):
            for j in range(g_start[g], g_end[g]):
                if aptr[j + 1] == aptr[j]:
                    continue
                isx, bx, cx = node_list(j, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
                mark(j, isx, bx, cx, nbs, tail, nbg, ws, we, wr, wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast)
                for r in range(aptr[j], aptr[j + 1]):
                    k = ak[r]
                    isy, by, cy = node_list(k, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
                    anys = False
                    for t in range(cy):
                        s, e, rk, G = entry(isy, by, t, nbs, tail, nbg, ws, we, wr, wg)
                        if Rst[s] == j or Lst[e] == j:
                            if gst[G] != r:
                                gst[G] = r
                                out_cnt[g, G] += 1
                                anys = True
                    if anys:
                        out_tot[g] += 1


@njit(parallel=True, cache=True)
def pass_synneff(m, ns, nterm, lslot, aptr, ak, need, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr, wg,
                 cid, gs, ngenome, prio, S, msamp, out_neff, out_sbar, nchunks):
    n = len(cid)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        Rst = np.full(n, -1, np.int64)
        Lst = np.full(n, -1, np.int64)
        Rsrc = np.zeros(n, np.int32)
        Lsrc = np.zeros(n, np.int32)
        ca = np.zeros(ngenome, np.int64)
        cast = np.full(ngenome, -1, np.int64)
        gst = np.full(ngenome, -1, np.int64)
        glist = np.zeros(ngenome, np.int64)
        gp = np.zeros(ngenome, np.int64)
        for j in range(lo, hi):
            anyneed = False
            for r in range(aptr[j], aptr[j + 1]):
                if need[r]:
                    anyneed = True
                    break
            if not anyneed:
                continue
            isx, bx, cx = node_list(j, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
            mark(j, isx, bx, cx, nbs, tail, nbg, ws, we, wr, wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast)
            own = gs[ns[j]]
            for r in range(aptr[j], aptr[j + 1]):
                if not need[r]:
                    continue
                k = ak[r]
                isy, by, cy = node_list(k, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
                ng = 0
                for t in range(cy):
                    s, e, rk, G = entry(isy, by, t, nbs, tail, nbg, ws, we, wr, wg)
                    if Rst[s] == j or Lst[e] == j:
                        if gst[G] != r:
                            gst[G] = r
                            glist[ng] = G
                            ng += 1
                nn = ng + 1
                if ng > msamp - 1:
                    for q in range(ng):
                        gp[q] = prio[glist[q]]
                    o = np.argsort(gp[:ng])
                    sel = glist[:ng][o[:msamp - 1]].copy()
                else:
                    sel = glist[:ng].copy()
                kk = len(sel) + 1
                tot = 0.0
                for x in range(len(sel)):
                    tot += S[own, sel[x]]
                    for y in range(x + 1, len(sel)):
                        tot += S[sel[x], sel[y]]
                sbar = tot / (kk * (kk - 1) / 2.0) if kk >= 2 else 0.0
                out_sbar[r] = sbar
                out_neff[r] = nn / (1.0 + (nn - 1.0) * sbar)


# ============================================================================== pass C: word lists
@njit(cache=True)
def gather_all(j, k, ns, nterm, lslot, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr, wg, cid, n,
               Rst, Rsrc, Lst, Lsrc, ca, cast, os_, oe, orr, og):
    isx, bx, cx = node_list(j, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
    mark(j, isx, bx, cx, nbs, tail, nbg, ws, we, wr, wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast)
    isy, by, cy = node_list(k, nterm, ns, lslot, gene_at, lstart, tcnt, wptr)
    c = 0
    for t in range(cy):
        s, e, rk, G = entry(isy, by, t, nbs, tail, nbg, ws, we, wr, wg)
        tp, ps, pe, pr = best_p(j, s, e, Rst, Rsrc, Lst, Lsrc, isx, bx, nbs, tail, nbg, ws, we, wr, wg)
        if tp < 0:
            continue
        os_[c] = ps if ps < s else s
        oe[c] = pe if pe > e else e
        orr[c] = pr if pr > rk else rk
        og[c] = G
        c += 1
    if c == 0:
        return 0
    # dedupe identical spans (keep best rank), then order by rank, genome, start
    key = np.empty(c, np.int64)
    for t in range(c):
        key[t] = os_[t] * 8388608 + oe[t]
    o = np.argsort(key * 65536 + np.minimum(orr[:c], 65535))
    us = np.empty(c, np.int64)
    ue = np.empty(c, np.int64)
    ur = np.empty(c, np.int64)
    ug = np.empty(c, np.int64)
    u = 0
    last = -1
    for t in range(c):
        q = o[t]
        if key[q] != last:
            us[u] = os_[q]
            ue[u] = oe[q]
            ur[u] = orr[q]
            ug[u] = og[q]
            u += 1
            last = key[q]
    o2 = np.argsort((ur[:u] * 4096 + ug[:u]) * 8388608 + us[:u])
    for t in range(u):
        q = o2[t]
        os_[t] = us[q]
        oe[t] = ue[q]
        orr[t] = ur[q]
        og[t] = ug[q]
    return u


@njit(parallel=True, cache=True)
def pass_build(cj, ck, phase, optr, ns, nterm, lslot, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr, wg,
               cid, ngenome, kmax, out_cnt, nws, nwe, nwr, nwg, nchunks):
    n = len(cid)
    m = len(cj)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        Rst = np.full(n, -1, np.int64)
        Lst = np.full(n, -1, np.int64)
        Rsrc = np.zeros(n, np.int32)
        Lsrc = np.zeros(n, np.int32)
        ca = np.zeros(ngenome, np.int64)
        cast = np.full(ngenome, -1, np.int64)
        os_ = np.zeros(kmax, np.int64)
        oe = np.zeros(kmax, np.int64)
        orr = np.zeros(kmax, np.int64)
        og = np.zeros(kmax, np.int64)
        for c in range(lo, hi):
            u = gather_all(cj[c], ck[c], ns, nterm, lslot, gene_at, lstart, tcnt, nbs, tail, nbg, wptr, ws, we, wr,
                           wg, cid, n, Rst, Rsrc, Lst, Lsrc, ca, cast, os_, oe, orr, og)
            # stamps: mark() uses the node index j as stamp; reset per committed pair is implicit (unique j)
            if phase == 0:
                out_cnt[c] = u
            else:
                b = optr[c]
                for t in range(u):
                    nws[b + t] = os_[t]
                    nwe[b + t] = oe[t]
                    nwr[b + t] = orr[t]
                    nwg[b + t] = og[t]


# ============================================================================== adjacency, commit, rebuild
@njit(cache=True)
def adjacency(m, ns, nterm, ncid, nlen, excl, off, use_excl):
    cnt = 0
    aptr = np.zeros(m + 1, np.int64)
    ak = np.empty(m * 3, np.int64)
    for j in range(m):
        aptr[j] = cnt
        if nlen[j] == 0:
            continue
        for d in range(1, 4):
            k = j + d
            if k >= m or ncid[k] != ncid[j]:
                break
            ok = True
            for q in range(j + 1, k):
                if not nterm[q]:
                    ok = False
                    break
            if not ok:
                break
            if nlen[k] == 0:
                continue
            if use_excl and nterm[j] and nterm[k]:
                i = ns[j]
                dd = ns[k] - i
                if excl[off[i] + dd - 1]:
                    continue
            ak[cnt] = k
            cnt += 1
    aptr[m] = cnt
    return aptr, ak[:cnt].copy()


@njit(cache=True)
def commit(order, rj, rk, m):
    claimed = np.zeros(m, np.bool_)
    cj = np.empty(len(order), np.int64)
    ck = np.empty(len(order), np.int64)
    cr = np.empty(len(order), np.int64)
    c = 0
    for t in range(len(order)):
        r = order[t]
        j = rj[r]
        k = rk[r]
        free = True
        for q in range(j, k + 1):
            if claimed[q]:
                free = False
                break
        if free:
            for q in range(j, k + 1):
                claimed[q] = True
            cj[c] = j
            ck[c] = k
            cr[c] = r
            c += 1
    return cj[:c], ck[:c], cr[:c]


@njit(cache=True)
def rebuild(ns, ne, nterm, ninst, lslot, cj, ck, inst_base, lslot_base):
    """cj sorted. Live arrays after merging nodes cj..ck into one word node each (new instance ids and list slots in
    commit order). Child codes: >= 0 instance id, < 0 terminal slot -(s+1)."""
    m = len(ns)
    drop = 0
    for c in range(len(cj)):
        drop += ck[c] - cj[c]
    m2 = m - drop
    ns2 = np.empty(m2, np.int64)
    ne2 = np.empty(m2, np.int64)
    nt2 = np.empty(m2, np.bool_)
    ni2 = np.empty(m2, np.int64)
    ls2 = np.empty(m2, np.int64)
    ca_ = np.empty(len(cj), np.int64)
    cb_ = np.empty(len(cj), np.int64)
    i = 0
    c = 0
    o = 0
    while i < m:
        if c < len(cj) and cj[c] == i:
            j = cj[c]
            k = ck[c]
            ns2[o] = ns[j]
            ne2[o] = ne[k]
            nt2[o] = False
            ni2[o] = inst_base + c
            ls2[o] = lslot_base + c
            ca_[c] = ninst[j] if not nterm[j] else -(ns[j] + 1)
            cb_[c] = ninst[k] if not nterm[k] else -(ns[k] + 1)
            o += 1
            c += 1
            i = k + 1
        else:
            ns2[o] = ns[i]
            ne2[o] = ne[i]
            nt2[o] = nterm[i]
            ni2[o] = ninst[i]
            ls2[o] = lslot[i]
            o += 1
            i += 1
    return ns2, ne2, nt2, ni2, ls2, ca_, cb_


@njit(parallel=True, cache=True)
def copy_ranges(src_ptr, keep, dst_ptr, ws, we, wr, wg, nws, nwe, nwr, nwg):
    for t in prange(len(keep)):
        q = keep[t]
        b = src_ptr[q]
        L = src_ptr[q + 1] - b
        d0 = dst_ptr[t]
        for x in range(L):
            nws[d0 + x] = ws[b + x]
            nwe[d0 + x] = we[b + x]
            nwr[d0 + x] = wr[b + x]
            nwg[d0 + x] = wg[b + x]


# ============================================================================== list sources
class ListSet:
    """Gene lists in CSR form: entries nb (gene ids), tail (rank, 1-based), nbg (genome); per depth K the entry count."""

    def __init__(self, name, nb, tail, nbg, lstart, tcnt, info):
        self.name, self.nb, self.tail, self.nbg, self.lstart, self.tcnt, self.info = name, nb, tail, nbg, lstart, tcnt, info

    def kmax(self, K):
        return int(self.tcnt[K].max())


class Vectors:
    def __init__(self, X, uq):
        self.X, self.uq = X, uq


def load_vectors():
    import softrec as SR
    from embed_tokens import load_run, transform
    X, uu, valid, cnt = load_run(SR.EMB)
    transform(X, valid, cnt, "zca10")
    uq = np.load(SR.EMB / "uid_of_row.npy").astype(np.int64)
    return Vectors(X, uq)


def emb_lists(U):
    """softrec round-2 lists (zca10 kNN k 2,000, depths 200 / 1,000) outside the universe's lineage unit."""
    import softrec as SR
    import softrec2 as S2
    L = SR.Lists(U.u, S2.ListArgs([200, 1000]))
    tc = {K: L.cap[L.variants.index(f"K{K}_fnone")].astype(np.int64) for K in (200, 1000)}
    return ListSet("emb", L.nb, L.tail, L.nbg, L.lstart, tc, L.info), L


def aln_lists(U, V):
    """MMseqs2 hits (bit-score ranked, kNN layout) expanded to genes outside the lineage unit, depth 1,000."""
    import softrec as SR
    t0 = time.time()
    I = np.load(VL / "mm/aln__k1000__I.npy")
    D = np.load(VL / "mm/aln__k1000__B.npy").astype(np.float32)
    u = U.u
    uq = V.uq
    ok = uq >= 0
    order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
    mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=I.shape[0]))].astype(np.int64)
    K = 1000
    n = u.n
    nb = np.full((n, K), -1, np.int32)
    tail = np.zeros((n, K), np.uint16)
    llen = np.zeros(n, np.int64)
    ntie = np.zeros(n, np.int64)
    rf = np.zeros((n, 1), np.int64)
    cos_at = np.zeros((n, 1), np.float64)
    SR.expand_knn(uq, u.lin.astype(np.int64), I, D, mptr, order.astype(np.int64), K, np.array([-2.0]), np.array([K], np.int64),
                  nb, tail, llen, ntie, rf, cos_at)
    del I, D
    nbf = nb.ravel()
    nbg = np.where(nbf >= 0, u.gc[np.maximum(nbf, 0)], 0).astype(np.int16)
    info = dict(source="aln", genes_with_list=int((llen > 0).sum()), list_len_q01_q10_q50_q90=np.quantile(llen, [.01, .1, .5, .9]).tolist(),
                genes_list_shorter_than_1000=int((llen < K).sum()), seconds=round(time.time() - t0, 1))
    log("aln lists", info)
    return ListSet("aln", nbf, tail.ravel(), nbg, np.arange(n, dtype=np.int64) * K, {1000: llen}, info)


@njit(parallel=True, cache=True)
def z_count(uq, gc, lin, zoff, zG, out):
    for g in prange(len(uq)):
        u = uq[g]
        c = 0
        if u >= 0:
            for t in range(zoff[u], zoff[u + 1]):
                if lin[zG[t]] != lin[gc[g]]:
                    c += 1
        out[g] = c


@njit(parallel=True, cache=True)
def z_fill(uq, gc, lin, zoff, zG, zL, zz, g_start, lptr, nb, tail, nbg):
    for g in prange(len(uq)):
        u = uq[g]
        if u < 0:
            continue
        m = 0
        for t in range(zoff[u], zoff[u + 1]):
            if lin[zG[t]] != lin[gc[g]]:
                m += 1
        idx = np.empty(m, np.int64)
        key = np.empty(m, np.float64)
        m = 0
        for t in range(zoff[u], zoff[u + 1]):
            if lin[zG[t]] != lin[gc[g]]:
                idx[m] = t
                key[m] = -zz[t]
                m += 1
        o = np.argsort(key, kind="mergesort")          # entries are in genome order: ties keep genome order
        b = lptr[g]
        for r in range(m):
            t = idx[o[r]]
            nb[b + r] = g_start[zG[t]] + zL[t]
            tail[b + r] = r + 1
            nbg[b + r] = zG[t]


def z_lists(U, V, cut):
    """ws_calib table: per gene, the best gene of every genome outside its lineage unit (own genome or own group) with
    within-genome z >= cut, ranked by z."""
    t0 = time.time()
    T = RD / "calib/table"
    cnt = np.load(T / "counts.npy").astype(np.int64)
    off = np.r_[0, np.cumsum(cnt)]
    N = len(cnt)
    zG_all = np.memmap(T / "genome.bin", np.int16, "r")
    zL_all = np.memmap(T / "local.bin", np.int16, "r")
    zz_all = np.memmap(T / "z.bin", np.float16, "r")
    keep_cnt = np.zeros(N, np.int64)
    parts = []
    step = 100_000
    for a in range(0, N, step):
        b = min(N, a + step)
        s, e = off[a], off[b]
        z = np.asarray(zz_all[s:e]).astype(np.float32)
        k = z >= cut
        uid_of = np.repeat(np.arange(a, b), cnt[a:b])
        keep_cnt[a:b] = np.bincount(uid_of[k] - a, minlength=b - a)
        parts.append((np.asarray(zG_all[s:e])[k], np.asarray(zL_all[s:e])[k], z[k]))
    zG = np.concatenate([p[0] for p in parts]).astype(np.int16)
    zL = np.concatenate([p[1] for p in parts]).astype(np.int16)
    zz = np.concatenate([p[2] for p in parts]).astype(np.float32)
    del parts
    zoff = np.r_[0, np.cumsum(keep_cnt)].astype(np.int64)
    u = U.u
    gcount = np.zeros(u.n, np.int64)
    lin = u.lin_of_genome.astype(np.int64)
    z_count(V.uq, u.gc.astype(np.int64), lin, zoff, zG, gcount)
    lptr = np.r_[0, np.cumsum(gcount)].astype(np.int64)
    nb = np.full(lptr[-1], -1, np.int32)
    tail = np.zeros(lptr[-1], np.uint16)
    nbg = np.zeros(lptr[-1], np.int16)
    z_fill(V.uq, u.gc.astype(np.int64), lin, zoff, zG, zL, zz, u.g_start.astype(np.int64), lptr, nb, tail, nbg)
    assert (nb >= 0).all()
    assert np.array_equal(u.gc[nb], nbg.astype(np.int32))
    info = dict(source=f"z{cut:g}", entries=int(lptr[-1]), uid_entries=int(zoff[-1]),
                list_len_q01_q10_q50_q90_max=np.quantile(gcount, [.01, .1, .5, .9, 1]).tolist(),
                genes_empty=int((gcount == 0).sum()), seconds=round(time.time() - t0, 1))
    log("z lists", info)
    return ListSet(f"z{cut:g}", nb, tail, nbg, lptr[:-1].copy(), {0: gcount}, info)


# ============================================================================== one order (observed or shuffled)
class Order:
    """Slot space of one gene order: gene_at, slot_of, list members as slots, context prefix sums, exclusion, floor."""

    def __init__(self, U, V, lsets, r, threads, Lemb=None, excl_tag="genome", shuf_needed=True):
        import softrec as SR
        t0 = time.time()
        self.r = r
        self.cache = {}
        u = U.u
        if r < 0:
            self.gene_at = np.arange(u.n, dtype=np.int64)
            self.slot_of = self.gene_at.copy()
        else:
            self.gene_at, self.slot_of = SR.null_order(u, r)
        self.nbs = {}
        for LS in lsets:
            x = np.empty_like(LS.nb)
            SR.map_slots(LS.nb, self.slot_of.astype(np.int32), x)
            self.nbs[LS.name] = x
        nd, off, left, d = u.instances(WGAP)
        self.off = off
        ed = VL / "excl"
        ed.mkdir(exist_ok=True)
        fn = ed / f"{excl_tag}_order{r}.npz"
        if Lemb is not None:
            ex = np.zeros((Lemb.V, len(left)), np.bool_)
            SR.exclusion(u.n, Lemb.V, self.gene_at, nd, off, Lemb.excl_mode, Lemb.tok, Lemb.X, Lemb.uq, Lemb.capcos, ex)
            self.excl = {200: ex[Lemb.variants.index("K200_fnone")], 1000: ex[Lemb.variants.index("K1000_fnone")]}
            np.savez(fn, K200=self.excl[200], K1000=self.excl[1000])
        else:
            z = np.load(fn)
            self.excl = {200: z["K200"], 1000: z["K1000"]}
        self.P = np.zeros((u.n + len(u.c_start), V.X.shape[1]), np.float32)
        prefix_sums(V.X, V.uq, self.gene_at, u.c_start, u.c_end, self.P)
        # noise floor: 2,000,000 random pairs of round-0 instance contexts in different genomes
        rng = np.random.default_rng(17 if r < 0 else 2000 + r)
        okslot = np.flatnonzero(u.cid[:-1] == u.cid[1:])
        ia = rng.choice(okslot, 2_400_000)
        ib = rng.choice(okslot, 2_400_000)
        keep = u.gc[ia] != u.gc[ib]
        ia, ib = ia[keep][:2_000_000], ib[keep][:2_000_000]
        self.floor = {}
        for shuf in ((False, True) if shuf_needed else (False,)):
            out = np.zeros(len(ia))
            floor_cos(ia, ib, CTX_W, u.cid.astype(np.int64), u.c_start, u.c_end, self.P, U.sig, shuf, out, threads * 4)
            ok = np.isfinite(out)
            self.floor[shuf] = dict(c0=float(out[ok].mean()), sd0=float(out[ok].std()), n=int(ok.sum()),
                                    q50_q99_q999=np.quantile(out[ok], [0.5, 0.99, 0.999]).tolist())
        log(f"order {r}: slots, exclusion, prefix sums, floor {self.floor} ({time.time() - t0:.0f}s)")


class Univ:
    def __init__(self, lineage_map, genus=False):
        import softrec as SR
        if genus:
            self.u = SR.Universe()
        else:
            self.u = lf_universe(lineage_map)
        u = self.u
        gm = pd.read_parquet(GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(u.ng))
        self.grp = pd.factorize(gm.group.to_numpy())[0].astype(np.int64)
        if genus:
            self.grp = u.lin_of_genome.astype(np.int64)
        self.ngroup = int(self.grp.max()) + 1
        self.mg = u.n_g.astype(np.float64)
        self.prio = np.random.default_rng(17).permutation(u.ng).astype(np.int64)
        self.sig = np.random.default_rng(17).permutation(u.n).astype(np.int64)


def lf_universe(lineage_map):
    """softrec.Universe with the lineage unit from a genome -> group map (copied from lineagefree.softrec_lf; no
    taxonomy column is read)."""
    import softrec as SR

    class LFU(SR.Universe):
        def __init__(self):
            g = pd.read_parquet(SR.PKG / "universe/genes.parquet",
                                columns=["gid", "genome", "genome_code", "contig", "contig_code", "position_index"])
            assert (g.gid.to_numpy() == np.arange(len(g))).all()
            self.n = n = len(g)
            self.genome_name = g.genome.to_numpy()
            self.contig_name = g.contig.to_numpy()
            self.pos = g.position_index.to_numpy(np.int64)
            self.gc = g.genome_code.to_numpy(np.int32)
            cc = g.contig_code.to_numpy(np.int64)
            assert (np.diff(self.gc) >= 0).all() and (np.diff(cc) >= 0).all()
            brk = np.flatnonzero(np.diff(cc)) + 1
            self.c_start = np.r_[0, brk].astype(np.int64)
            self.c_end = np.r_[brk, n].astype(np.int64)
            self.cid = np.repeat(np.arange(len(self.c_start)), self.c_end - self.c_start).astype(np.int32)
            gm = pd.read_parquet(SR.PKG / "universe/genomes.parquet", columns=["genome", "genome_code", "n_genes"])
            gm = gm.sort_values("genome_code").reset_index(drop=True)
            assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
            self.ng = len(gm)
            self.n_g = gm.n_genes.to_numpy(np.float64)
            lm = pd.read_parquet(lineage_map, columns=["genome_code", "group"]).sort_values("genome_code")
            assert np.array_equal(lm.genome_code.to_numpy(), np.arange(self.ng))
            self.lin_of_genome = pd.factorize(lm.group)[0].astype(np.int32)
            self.nl = int(self.lin_of_genome.max()) + 1
            self.lin = self.lin_of_genome[self.gc]
            gchange = np.flatnonzero(np.diff(self.gc)) + 1
            self.g_start = np.r_[0, gchange].astype(np.int64)
            self.g_end = np.r_[gchange, n].astype(np.int64)
            self.genes_in_lin = np.bincount(self.lin, minlength=self.nl).astype(np.int64)
            self.gm = gm
    return LFU()


# ============================================================================== the learner
class Learner:
    def __init__(self, U, LS, O, K, arm, threads, excl_K=1000):
        self.U, self.LS, self.O, self.K, self.arm, self.threads = U, LS, O, K, arm, threads
        self.tcnt = LS.tcnt[K]
        self.kmax = max(int(self.tcnt.max()), 1)
        self.nbs = O.nbs[LS.name]
        self.excl = O.excl[excl_K]
        self.shuf = arm == "ctxshuf"
        fl = O.floor[self.shuf]
        self.c0, self.sd0 = fl["c0"], fl["sd0"]
        u = U.u
        self.cid = u.cid.astype(np.int64)
        self.gs = u.gc.astype(np.int64)        # genome of slot (fixed)
        self.own = np.minimum(1.0, 2.0 * WGAP / U.mg)

    def evaluate(self, st, aptr, ak, tau):
        """n, E, Lambda_eff (and parts) for every adjacency row."""
        U, LS, O = self.U, self.LS, self.O
        R = len(ak)
        on = np.zeros(R, np.int64)
        ong = np.zeros(R, np.int64)
        oE = np.zeros(R, np.float64)
        oEg = np.zeros(R, np.float64)
        t0 = time.time()
        pass_count(st["m"], st["ns"], st["nterm"], st["lslot"], aptr, ak, O.gene_at, LS.lstart, self.tcnt, self.nbs, LS.tail,
                   LS.nbg, st["wptr"], st["ws"], st["we"], st["wr"], st["wg"], self.cid, U.grp, U.u.ng, U.ngroup, U.mg,
                   on, ong, oE, oEg, self.threads * 4)
        ta = time.time() - t0
        rj = np.repeat(np.arange(st["m"], dtype=np.int64), np.diff(aptr))
        own = self.own[self.gs[st["ns"][rj]]]
        n = on + 1.0
        E = oE + own
        lamG = lam_np(n, E)
        neff = n.copy()
        kinf = np.zeros(R, np.int64)
        cbar = np.full(R, np.nan)
        tb = 0.0
        if self.arm == "grp":
            nu = ong + 1.0
            Eu = oEg + own
            lam = lam_np(nu, Eu)
            ok = (nu >= 2) & (lam >= tau)
        elif self.arm == "gen":
            lam = lamG
            ok = (n >= 2) & (lam >= tau)
        elif self.arm == "syn":
            need = (n >= 2) & (lamG >= tau)
            t1 = time.time()
            S = self.syn_S()
            ne_ = np.zeros(R, np.float64)
            pass_synneff(st["m"], st["ns"], st["nterm"], st["lslot"], aptr, ak, need, O.gene_at, LS.lstart, self.tcnt,
                         self.nbs, LS.tail, LS.nbg, st["wptr"], st["ws"], st["we"], st["wr"], st["wg"], self.cid, self.gs,
                         U.u.ng, U.prio, S, 64, ne_, cbar, self.threads * 4)
            tb = time.time() - t1
            neff = np.where(need, ne_, n)
            lam = np.where(need, lamG * neff / n, 0.0)
            ok = need & (neff >= MIN_NEFF) & (lam >= tau)
        else:
            need = (n >= 2) & (lamG >= tau)
            t1 = time.time()
            ne_ = np.zeros(R, np.float64)
            pass_neff(st["m"], st["ns"], st["ne"], st["nterm"], st["lslot"], aptr, ak, need, O.gene_at, LS.lstart,
                      self.tcnt, self.nbs, LS.tail, LS.nbg, st["wptr"], st["ws"], st["we"], st["wr"], st["wg"], self.cid,
                      U.u.c_start, U.u.c_end, U.u.ng, U.prio, O.P, CTX_W, U.sig, self.shuf, M_SAMPLE, self.c0, self.sd0,
                      ZNOISE, self.kmax, ne_, kinf, cbar, self.threads * 4)
            tb = time.time() - t1
            neff = np.where(need, ne_, n)
            lam = np.where(need, lamG * neff / n, 0.0)
            ok = need & (neff >= MIN_NEFF) & (lam >= tau)
        return dict(rj=rj, n=n, E=E, ng=ong + 1.0, Eg=oEg + own, lamG=lamG, neff=neff, kinf=kinf, cbar=cbar, lam=lam,
                    ok=ok, sec_A=round(ta, 1), sec_B=round(tb, 1))

    def syn_S(self):
        """Synteny-retention matrix of this gene order (round-0 adjacencies, this learner's lists), cached on the order."""
        key = ("S", self.LS.name, self.K)
        if key in self.O.cache:
            return self.O.cache[key]
        t0 = time.time()
        st = self.init_state()
        aptr, ak = self.adj(st)
        u = self.U.u
        ng = u.ng
        cnt = np.zeros((ng, ng), np.int64)
        tot = np.zeros(ng, np.int64)
        syn_matrix(ng, u.g_start.astype(np.int64), u.g_end.astype(np.int64), aptr, ak, st["ns"], st["nterm"], st["lslot"],
                   self.O.gene_at, self.LS.lstart, self.tcnt, self.nbs, self.LS.tail, self.LS.nbg, st["wptr"], st["ws"],
                   st["we"], st["wr"], st["wg"], self.cid, cnt, tot, self.threads * 4)
        Sd = cnt / np.maximum(tot, 1)[:, None]
        S = (Sd + Sd.T) / 2.0
        np.fill_diagonal(S, 1.0)
        off = S[~np.eye(ng, dtype=bool)]
        grp = self.U.grp
        same = (grp[:, None] == grp[None, :]) & ~np.eye(ng, dtype=bool)
        info = dict(seconds=round(time.time() - t0, 1), supported_adjacencies_q50=float(np.median(tot)),
                    offdiag_mean=float(off.mean()), offdiag_q50_q90_q99_max=np.quantile(off, [.5, .9, .99, 1]).tolist(),
                    same_group_mean=float(S[same].mean()), other_group_mean=float(S[~same & ~np.eye(ng, dtype=bool)].mean()))
        log("synteny matrix", f"order {self.O.r}", info)
        self.O.cache[key] = S
        self.O.cache[("Sinfo", self.LS.name, self.K)] = info
        if self.O.r < 0:
            np.save(VL / "results" / f"syn_S_{self.LS.name}_K{self.K}.npy", S.astype(np.float32))
        return S

    def init_state(self):
        n = self.U.u.n
        z = np.zeros(0, np.int32)
        return dict(m=n, ns=np.arange(n, dtype=np.int64), ne=np.arange(n, dtype=np.int64), nterm=np.ones(n, np.bool_),
                    ninst=np.full(n, -1, np.int64), lslot=np.full(n, -1, np.int64), wptr=np.zeros(1, np.int64),
                    ws=z, we=z.copy(), wr=z.copy(), wg=np.zeros(0, np.int16))

    def nlen(self, st):
        out = self.tcnt[self.O.gene_at[st["ns"]]].copy()
        w = ~st["nterm"]
        q = st["lslot"][w]
        out[w] = st["wptr"][q + 1] - st["wptr"][q]
        return out

    def adj(self, st):
        return adjacency(st["m"], st["ns"], st["nterm"], self.cid[st["ns"]], self.nlen(st), self.excl, self.O.off, True)

    def round0(self, tau_min):
        st = self.init_state()
        aptr, ak = self.adj(st)
        ev = self.evaluate(st, aptr, ak, tau_min)
        return st, aptr, ak, ev

    def run(self, tau, save_dir=None):
        t0 = time.time()
        st = self.init_state()
        inst = dict(s=[], e=[], a=[], b=[], rnd=[], n=[], E=[], neff=[], lam=[], lamG=[], ng=[], llen=[])
        rounds = []
        n_inst = 0
        for rnd in range(MAXR):
            tr = time.time()
            aptr, ak = self.adj(st)
            if len(ak) == 0:
                break
            ev = self.evaluate(st, aptr, ak, tau)
            cand = np.flatnonzero(ev["ok"])
            if len(cand) == 0:
                rounds.append(dict(round=rnd, adjacencies=int(len(ak)), candidates=0, committed=0,
                                   seconds=round(time.time() - tr, 1)))
                break
            rj = ev["rj"]
            dist = ak - rj
            order = cand[np.lexsort((dist[cand], st["ns"][rj[cand]], -ev["lam"][cand]))]
            cj, ck, cr = commit(order, rj, ak, st["m"])
            o = np.argsort(cj, kind="stable")
            cj, ck, cr = cj[o], ck[o], cr[o]
            nc = len(cj)
            cnt = np.zeros(nc, np.int64)
            dummy = np.zeros(1, np.int64)
            args = (st["ns"], st["nterm"], st["lslot"], self.O.gene_at, self.LS.lstart, self.tcnt, self.nbs, self.LS.tail,
                    self.LS.nbg, st["wptr"], st["ws"], st["we"], st["wr"], st["wg"], self.cid, self.U.u.ng, self.kmax)
            d32 = np.zeros(1, np.int32)
            pass_build(cj, ck, 0, dummy, *args, cnt, d32, d32, d32, np.zeros(1, np.int16), self.threads * 4)
            optr = np.r_[0, np.cumsum(cnt)].astype(np.int64)
            nws = np.zeros(optr[-1], np.int32)
            nwe = np.zeros(optr[-1], np.int32)
            nwr = np.zeros(optr[-1], np.int32)
            nwg = np.zeros(optr[-1], np.int16)
            pass_build(cj, ck, 1, optr, *args, cnt, nws, nwe, nwr, nwg, self.threads * 4)
            # surviving words' lists + new lists -> new CSR
            claimed = np.zeros(st["m"] + 1, np.int64)
            np.add.at(claimed, cj, 1)
            np.add.at(claimed, ck + 1, -1)
            claimed = np.cumsum(claimed)[:st["m"]] > 0
            surv = np.flatnonzero(~st["nterm"] & ~claimed)
            sl = st["lslot"][surv]
            slen = st["wptr"][sl + 1] - st["wptr"][sl]
            sptr = np.r_[0, np.cumsum(slen)].astype(np.int64)
            tot = sptr[-1] + optr[-1]
            ws2 = np.zeros(tot, np.int32)
            we2 = np.zeros(tot, np.int32)
            wr2 = np.zeros(tot, np.int32)
            wg2 = np.zeros(tot, np.int16)
            copy_ranges(st["wptr"], sl, sptr, st["ws"], st["we"], st["wr"], st["wg"], ws2, we2, wr2, wg2)
            ws2[sptr[-1]:], we2[sptr[-1]:], wr2[sptr[-1]:], wg2[sptr[-1]:] = nws, nwe, nwr, nwg
            wptr2 = np.r_[sptr, sptr[-1] + optr[1:]].astype(np.int64)
            lslot = st["lslot"].copy()
            lslot[surv] = np.arange(len(surv))
            ns2, ne2, nt2, ni2, ls2, ca_, cb_ = rebuild(st["ns"], st["ne"], st["nterm"], st["ninst"], lslot, cj, ck,
                                                        n_inst, len(surv))
            for key, val in (("s", st["ns"][cj]), ("e", st["ne"][ck]), ("a", ca_), ("b", cb_),
                             ("rnd", np.full(nc, rnd, np.int64)), ("n", ev["n"][cr]), ("E", ev["E"][cr]),
                             ("neff", ev["neff"][cr]), ("lam", ev["lam"][cr]), ("lamG", ev["lamG"][cr]),
                             ("ng", ev["ng"][cr]), ("llen", cnt)):
                inst[key].append(val)
            n_inst += nc
            st = dict(m=len(ns2), ns=ns2, ne=ne2, nterm=nt2, ninst=ni2, lslot=ls2, wptr=wptr2, ws=ws2, we=we2, wr=wr2,
                      wg=wg2)
            rounds.append(dict(round=rnd, adjacencies=int(len(ak)), candidates=int(len(cand)), committed=int(nc),
                               live=int(st["m"]), list_entries=int(len(ws2)), empty_lists=int((cnt == 0).sum()),
                               max_lam=float(ev["lam"][cand].max()), sec_A=ev["sec_A"], sec_B=ev["sec_B"],
                               seconds=round(time.time() - tr, 1)))
            log(self.arm, self.LS.name, f"K{self.K}", f"order {self.O.r}", f"tau {tau:g}", rounds[-1],
                f"{time.time() - t0:.0f}s")
        inst = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in inst.items()}
        top = ~st["nterm"]
        cov = int((st["ne"][top] - st["ns"][top] + 1).sum())
        summ = dict(arm=self.arm, lists=self.LS.name, K=self.K, order=self.O.r, tau=tau, instances=int(len(inst["s"])),
                    top_instances=int(top.sum()), genes_in_top=cov, genes_in_top_frac=cov / self.U.u.n,
                    rounds=len(rounds), seconds=round(time.time() - t0, 1),
                    top_span_q=np.quantile((st["ne"] - st["ns"] + 1)[top], [.5, .9, .99, 1]).tolist() if top.any() else [])
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(inst).to_parquet(save_dir / "inst.parquet", index=False)
            np.savez(save_dir / "live.npz", ns=st["ns"], ne=st["ne"], nterm=st["nterm"], ninst=st["ninst"],
                     lslot=st["lslot"])
            np.savez(save_dir / "toplists.npz", wptr=st["wptr"], ws=st["ws"], we=st["we"], wr=st["wr"], wg=st["wg"])
            jdump(save_dir / "run.json", dict(summary=summ, rounds=rounds, floor=self.O.floor[self.shuf],
                                              K=self.K, arm=self.arm, lists=self.LS.name))
        log("RUN", summ)
        return summ, rounds, st, inst


def ladder_counts(ev):
    return {str(t): int((ev["ok"] & (ev["lam"] >= t)).sum()) for t in LADDER}


# ============================================================================== commands
def cmd_refsr(a):
    """Round-0 pass with softrec round 2's genus lists reproduces the stored L, G, E (K1000)."""
    import numba
    numba.set_num_threads(a.threads)
    U = Univ(None, genus=True)
    LS, Lemb = emb_lists(U)
    u = U.u
    gene_at = np.arange(u.n, dtype=np.int64)
    nbs = LS.nb
    nd, off, left, d = u.instances(WGAP)
    ex = np.zeros((Lemb.V, len(left)), np.bool_)
    import softrec as SR
    SR.exclusion(u.n, Lemb.V, gene_at, nd, off, Lemb.excl_mode, Lemb.tok, Lemb.X, Lemb.uq, Lemb.capcos, ex)
    excl = ex[Lemb.variants.index("K1000_fnone")]
    tcnt = LS.tcnt[1000]
    m = u.n
    ns = np.arange(m, dtype=np.int64)
    nterm = np.ones(m, np.bool_)
    lslot = np.full(m, -1, np.int64)
    z = np.zeros(1, np.int32)
    cid = u.cid.astype(np.int64)
    aptr, ak = adjacency(m, ns, nterm, cid, np.ones(m, np.int64), excl, off, True)
    R = len(ak)
    on, ong, oE, oEg = np.zeros(R, np.int64), np.zeros(R, np.int64), np.zeros(R), np.zeros(R)
    t0 = time.time()
    pass_count(m, ns, nterm, lslot, aptr, ak, gene_at, LS.lstart, tcnt, nbs, LS.tail, LS.nbg, np.zeros(1, np.int64), z, z, z,
               np.zeros(1, np.int16), cid, U.grp,
               u.ng, U.ngroup, U.mg, on, ong, oE, oEg, a.threads * 4)
    sec = time.time() - t0
    rj = np.repeat(np.arange(m, dtype=np.int64), np.diff(aptr))
    inst = off[rj] + (ak - rj) - 1
    zz = np.load(RD / "softrec/r2/stats/obs.npz")
    v = list(zz["variants"]).index("K1000_fnone")
    sL, sG, sE = zz["L"][v], zz["G"][v], zz["E"][v]
    mine_excl = np.ones(len(sL), bool)
    mine_excl[inst] = False
    dE = np.abs(sE[inst].astype(np.float64) - oE)
    res = dict(rows=int(R), stored_instances=int(len(sL)), seconds_pass=round(sec, 1),
               excluded_equal=bool(np.array_equal(sL < 0, mine_excl)),
               L_equal=bool(np.array_equal(sL[inst].astype(np.int64), ong)),
               G_equal=bool(np.array_equal(sG[inst].astype(np.int64), on)),
               E_maxabs=float(dE.max()), E_maxrel=float((dE / np.maximum(oE, 1e-6)).max()),
               E_equal_as_float32=bool(np.array_equal(sE[inst], oE.astype(np.float32))))
    res["reproduced"] = bool(res["excluded_equal"] and res["L_equal"] and res["G_equal"] and res["E_maxrel"] < 1e-5)
    log("REFSR", res)
    jdump(VL / "results" / "refsr.json", res)


# arm -> (evidence unit, list source, depth). amendment_2: grp is the main unit; aln and z use it (run with --excl group);
# syn / ctx / ctxshuf / gen use genome-excluded lists (--excl genome).
SPEC = {"grp": ("grp", "emb", 1000), "grp200": ("grp", "emb", 200), "syn": ("syn", "emb", 1000),
        "ctx": ("ctx", "emb", 1000), "ctxshuf": ("ctxshuf", "emb", 1000), "gen": ("gen", "emb", 1000),
        "ctx200": ("ctx", "emb", 200), "aln": ("grp", "aln", 1000), "z10.6": ("grp", "z10.6", 0),
        "z13.7": ("grp", "z13.7", 0)}


def cmd_job(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    arms = [x for x in a.arms.split(",") if x]
    full_nulls = [x for x in a.full_nulls.split(",") if x]
    if a.refsr:
        cmd_refsr(a)
    lmap = GENOME_MAP if a.excl == "genome" else GROUPS
    U = Univ(lmap)
    srcs = sorted({SPEC[x][1] for x in arms})
    lsets, Lemb, V = {}, None, None
    for s in srcs:
        if s == "emb":
            lsets[s], Lemb = emb_lists(U)
            V = Vectors(Lemb.X, Lemb.uq)
    if V is None:
        V = load_vectors()
    for s in srcs:
        if s == "aln":
            lsets[s] = aln_lists(U, V)
        elif s.startswith("z"):
            lsets[s] = z_lists(U, V, float(s[1:]))
    log("lists", {k: v.info for k, v in lsets.items()}, f"{time.time() - t0:.0f}s")
    tag = a.tag or a.arms.replace(",", "+")
    res_fn = VL / "results" / f"job_{tag}.json"
    res = dict(arms=arms, excl=a.excl, lists={k: v.info for k, v in lsets.items()}, ladder={}, floors={})
    excl_tag = a.excl
    shuf_needed = "ctxshuf" in arms
    for r in (-1, 0, 1, 2):
        O = Order(U, V, list(lsets.values()), r, a.threads, Lemb=Lemb, excl_tag=excl_tag, shuf_needed=shuf_needed)
        res["floors"][str(r)] = {str(k): v for k, v in O.floor.items()}
        for arm in arms:
            kind, src, K = SPEC[arm]
            Lr = Learner(U, lsets[src], O, K, kind, a.threads, excl_K=(200 if K == 200 else 1000))
            st, aptr, ak, ev = Lr.round0(min(LADDER))
            cnts = ladder_counts(ev)
            res["ladder"].setdefault(arm, {})[str(r)] = dict(counts=cnts, adjacencies=int(len(ak)), sec_A=ev["sec_A"],
                                                             sec_B=ev["sec_B"])
            log("LADDER", arm, "order", r, cnts, ev["sec_A"], ev["sec_B"])
            if kind == "syn":
                res.setdefault("syn_S", {})[str(r)] = O.cache.get(("Sinfo", src, K))
            if r < 0:
                pas = ev["ok"] & (ev["lam"] >= 10)
                res.setdefault("round0_obs", {})[arm] = dict(
                    lamG_ge10=int((ev["lamG"] >= 10).sum()), pass_tau10=int(pas.sum()),
                    neff_over_n_q=np.quantile((ev["neff"] / ev["n"])[pas], [.1, .25, .5, .75, .9]).tolist() if pas.any() else [],
                    n_q=np.quantile(ev["n"][pas], [.1, .5, .9]).tolist() if pas.any() else [],
                    neff_q=np.quantile(ev["neff"][pas], [.1, .5, .9]).tolist() if pas.any() else [],
                    kinf_q=np.quantile(ev["kinf"][pas], [.1, .5, .9]).tolist() if pas.any() else [])
            del Lr, ev, st
        jdump(res_fn, res)
        del O
    for arm in arms:
        lad = res["ladder"][arm]
        rows = []
        for t in LADDER:
            obs = lad["-1"]["counts"][str(t)]
            nul = float(np.mean([lad[str(r)]["counts"][str(t)] for r in (0, 1, 2)]))
            rows.append(dict(tau=t, observed=obs, null_mean=nul, fdr=(nul / obs if obs else None)))
        ok = [x["tau"] for x in rows if x["fdr"] is not None and x["fdr"] <= FDR]
        res.setdefault("tau", {})[arm] = dict(table=rows, tau_star=(min(ok) if ok else None))
        log("TAU", arm, rows, "tau*", res["tau"][arm]["tau_star"])
    jdump(res_fn, res)
    O = Order(U, V, list(lsets.values()), -1, a.threads, Lemb=None, excl_tag=excl_tag, shuf_needed=shuf_needed)
    for arm in arms:
        kind, src, K = SPEC[arm]
        ts = res["tau"][arm]["tau_star"]
        if ts is None:
            continue
        Lr = Learner(U, lsets[src], O, K, kind, a.threads, excl_K=(200 if K == 200 else 1000))
        summ, rounds, st, inst = Lr.run(ts, save_dir=VL / "runs" / arm / "obs")
        res.setdefault("full", {})[arm] = dict(obs=summ, rounds=rounds)
        if a.recip and arm == arms[0]:
            res["full"][arm]["reciprocity"] = reciprocity(U, lsets[src], Lr, inst)
        jdump(res_fn, res)
        del Lr, st, inst
    del O
    if full_nulls:
        O = Order(U, V, list(lsets.values()), 0, a.threads, Lemb=None, excl_tag=excl_tag,
                  shuf_needed=("ctxshuf" in full_nulls))
        for arm in full_nulls:
            kind, src, K = SPEC[arm]
            ts = res["tau"][arm]["tau_star"]
            if ts is None:
                continue
            Lr = Learner(U, lsets[src], O, K, kind, a.threads, excl_K=(200 if K == 200 else 1000))
            summ, rounds, st, inst = Lr.run(ts, save_dir=None)
            res["full"][arm]["null0"] = summ
            res["full"][arm]["word_fdr"] = summ["instances"] / max(res["full"][arm]["obs"]["instances"], 1)
            jdump(res_fn, res)
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res_fn, res)
    log("JOB DONE", res_fn, f"{time.time() - t0:.0f}s")


def cmd_diag(a):
    """Junction diagnosis of a saved run (evaluation; reads the ribosomal runs and gold members).
    For every adjacency of the final parse: n, E, Lambda at the word level; for junctions whose two nodes lie in one
    ribosomal run or one gold instance, the same quantities for the two genes that meet at the junction as terminals.
    Also list size against word span for all word instances."""
    import numba
    numba.set_num_threads(a.threads)
    arm = a.arms.split(",")[0]
    kind, src, K = SPEC[arm]
    lmap = GENOME_MAP if a.excl == "genome" else GROUPS
    U = Univ(lmap)
    lsets, Lemb, V = {}, None, None
    if src == "emb":
        lsets[src], Lemb = emb_lists(U)
        V = Vectors(Lemb.X, Lemb.uq)
    else:
        V = load_vectors()
        lsets[src] = aln_lists(U, V) if src == "aln" else z_lists(U, V, float(src[1:]))
    O = Order(U, V, list(lsets.values()), -1, a.threads, Lemb=None, excl_tag=a.excl, shuf_needed=False)
    tau = json.loads((VL / "runs" / arm / "obs" / "run.json").read_text())["summary"]["tau"]
    Lr = Learner(U, lsets[src], O, K, kind, a.threads, excl_K=(200 if K == 200 else 1000))
    d = VL / "runs" / arm / "obs"
    lv = np.load(d / "live.npz")
    tl = np.load(d / "toplists.npz")
    st = dict(m=len(lv["ns"]), ns=lv["ns"].astype(np.int64), ne=lv["ne"].astype(np.int64), nterm=lv["nterm"],
              ninst=lv["ninst"].astype(np.int64), lslot=lv["lslot"].astype(np.int64), wptr=tl["wptr"].astype(np.int64),
              ws=tl["ws"], we=tl["we"], wr=tl["wr"], wg=tl["wg"])
    aptr, ak = Lr.adj(st)
    ev = Lr.evaluate(st, aptr, ak, tau)
    rj = ev["rj"]
    # gene-level counterfactual: the two genes that meet at the junction, as terminals
    gl = st["ne"][rj]
    gr = st["ns"][ak]
    st0 = Lr.init_state()
    m0 = st0["m"]
    pair_ok = (U.u.cid[gl] == U.u.cid[gr]) & (gr - gl >= 1) & (gr - gl <= 3)
    aptr0 = np.zeros(m0 + 1, np.int64)
    sel = np.flatnonzero(pair_ok)
    o = np.argsort(gl[sel], kind="stable")
    sel = sel[o]
    np.add.at(aptr0, gl[sel] + 1, 1)
    aptr0 = np.cumsum(aptr0)
    ak0 = gr[sel].astype(np.int64)
    ev0 = Lr.evaluate(st0, aptr0, ak0, tau)
    gene_lam = np.full(len(rj), np.nan)
    gene_n = np.full(len(rj), np.nan)
    gene_lam[sel] = ev0["lam"]
    gene_n[sel] = (ev0["ng"] if kind == "grp" else ev0["n"])
    import softrec as SR  # noqa: F401
    sys.path.insert(0, str(RD / "boundary/code"))
    import blib as BL
    runs = BL.ribo_runs()
    rid = np.full(U.u.n, -1, np.int64)
    for i, (s_, e_) in enumerate(zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy())):
        rid[s_:e_ + 1] = i
    inside = (rid[gl] >= 0) & (rid[gl] == rid[gr])
    word_n = ev["ng"] if kind == "grp" else ev["n"]
    nx = Lr.nlen(st)
    res = dict(arm=arm, tau=tau, adjacencies=int(len(rj)), passing=int(ev["ok"].sum()))

    def q(x):
        x = x[np.isfinite(x)]
        return np.quantile(x, [.1, .25, .5, .75, .9]).tolist() if len(x) else []
    for name, mk in (("ribosomal_junctions", inside), ("all_junctions", np.ones(len(rj), bool))):
        mk = mk & ~(st["nterm"][rj] & st["nterm"][ak])
        res[name] = dict(n=int(mk.sum()), word_level_n_q=q(word_n[mk].astype(float)), word_level_lam_q=q(ev["lam"][mk]),
                         gene_pair_n_q=q(gene_n[mk]), gene_pair_lam_q=q(gene_lam[mk]),
                         gene_pair_passes_tau=float(np.mean(gene_lam[mk] >= tau)) if mk.any() else None,
                         list_len_left_q=q(nx[rj][mk].astype(float)), list_len_right_q=q(nx[ak][mk].astype(float)),
                         parv=None)
    fam = runs.family.to_numpy()
    pv = inside & (fam[np.maximum(rid[gl], 0)] == "Parvarchaeaceae")
    res["parv_junctions"] = dict(n=int(pv.sum()), word_level_n_q=q(word_n[pv].astype(float)), gene_pair_n_q=q(gene_n[pv]),
                                 gene_pair_passes_tau=float(np.mean(gene_lam[pv] >= tau)) if pv.any() else None)
    inst = pd.read_parquet(d / "inst.parquet")
    span = (inst.e - inst.s + 1).to_numpy()
    res["list_len_by_span"] = {str(k): np.quantile(inst.llen.to_numpy()[span == k], [.1, .5, .9]).tolist()
                               for k in (2, 3, 4, 5, 6, 8, 10, 15, 20) if (span == k).any()}
    gene_ok = np.zeros(len(rj), bool)
    gene_ok[sel] = ev0["ok"]
    np.savez(VL / "runs" / arm / "obs" / "junctions.npz", j=rj, k=ak, gene_ok=gene_ok, gene_lam=gene_lam,
             word_lam=ev["lam"], word_n=word_n.astype(np.float64))
    log("DIAG", json.dumps(res)[:3000])
    jdump(VL / "results" / f"diag_{arm}.json", res)


def reciprocity(U, LS, Lr, inst, m=2000):
    """Share of supporting pairs (a', b') of committed round-0 merges with a in list(a') and b in list(b')."""
    rng = np.random.default_rng(17)
    r0 = np.flatnonzero((np.asarray(inst["rnd"]) == 0) & (np.asarray(inst["a"]) < 0) & (np.asarray(inst["b"]) < 0))
    pick = rng.choice(r0, min(m, len(r0)), replace=False)
    tc = Lr.tcnt

    def lst(g):
        return LS.nb[LS.lstart[g]:LS.lstart[g] + tc[g]]
    cid = U.u.cid
    tot, ra_, rb_, both = 0, 0, 0, 0
    for i in pick:
        a_, b_ = -int(inst["a"][i]) - 1, -int(inst["b"][i]) - 1
        sb = set(lst(b_).tolist())
        for x in lst(a_).tolist():
            for y in range(x - 3, x + 4):
                if y == x or y < 0 or y >= U.u.n or cid[y] != cid[x] or y not in sb:
                    continue
                tot += 1
                ra = a_ in set(lst(x).tolist())
                rb = b_ in set(lst(y).tolist())
                ra_ += ra
                rb_ += rb
                both += (ra and rb)
    return dict(merges=int(len(pick)), supporting_pairs=int(tot), a_reciprocal=ra_ / max(tot, 1),
                b_reciprocal=rb_ / max(tot, 1), both=both / max(tot, 1))


# ============================================================================== smoke test (synthetic, brute force)
def cmd_smoke(a):
    """Synthetic universe: kernels against direct Python implementations of the definitions."""
    import numba
    numba.set_num_threads(2)
    rng = np.random.default_rng(5)
    ng = 12
    genes_per = rng.integers(30, 60, ng)
    n = int(genes_per.sum())
    gc = np.repeat(np.arange(ng), genes_per)
    cid = np.zeros(n, np.int64)
    c = 0
    starts = []
    for g in range(ng):
        idx = np.flatnonzero(gc == g)
        cuts = sorted(rng.choice(np.arange(5, len(idx) - 5), rng.integers(0, 3), replace=False).tolist())
        bounds = [0] + cuts + [len(idx)]
        for x, y in zip(bounds[:-1], bounds[1:]):
            cid[idx[x:y]] = c
            starts.append(idx[x])
            c += 1
    c_start = np.array(starts, np.int64)
    c_end = np.r_[c_start[1:], n].astype(np.int64)
    fam = rng.integers(0, 60, n)
    for g in range(ng):
        idx = np.flatnonzero(gc == g)
        if g % 3 != 2:
            s = rng.integers(0, len(idx) - 8)
            fam[idx[s:s + 6]] = 1000 + np.arange(6)
    K = 8
    nb = np.full((n, K), -1, np.int32)
    tail = np.zeros((n, K), np.uint16)
    llen = np.zeros(n, np.int64)
    for i in range(n):
        cand = np.flatnonzero((fam == fam[i]) & (gc != gc[i]))
        cand = cand[rng.permutation(len(cand))][:K]
        nb[i, :len(cand)] = cand
        tail[i, :len(cand)] = np.arange(1, len(cand) + 1)
        llen[i] = len(cand)
    nbf = nb.ravel()
    lstart = np.arange(n, dtype=np.int64) * K
    tcnt = llen.copy()
    nbg = np.where(nbf >= 0, gc[np.maximum(nbf, 0)], 0).astype(np.int16)
    tailf = tail.ravel()
    grp = (np.arange(ng) // 2).astype(np.int64)
    mg = np.bincount(gc, minlength=ng).astype(np.float64)
    gene_at = np.arange(n, dtype=np.int64)
    z0 = np.zeros(0, np.int32)
    z16 = np.zeros(0, np.int16)
    ns0 = np.arange(n, dtype=np.int64)
    nterm0 = np.ones(n, np.bool_)
    lslot0 = np.full(n, -1, np.int64)
    wptr0 = np.zeros(1, np.int64)
    nd = np.zeros(n, np.int64)
    for i in range(n):
        k = 0
        while k < 3 and i + k + 1 < n and cid[i + k + 1] == cid[i]:
            k += 1
        nd[i] = k
    off = np.r_[0, np.cumsum(nd)[:-1]].astype(np.int64)
    excl = np.zeros(int(nd.sum()), np.bool_)
    aptr, ak = adjacency(n, ns0, nterm0, cid, tcnt, excl, off, True)
    R = len(ak)
    on, ong = np.zeros(R, np.int64), np.zeros(R, np.int64)
    oE, oEg = np.zeros(R), np.zeros(R)
    pass_count(n, ns0, nterm0, lslot0, aptr, ak, gene_at, lstart, tcnt, nbf, tailf, nbg, wptr0, z0, z0, z0, z16, cid, grp,
               ng, int(grp.max()) + 1, mg, on, ong, oE, oEg, 3)
    rj = np.repeat(np.arange(n), np.diff(aptr))
    bad = 0
    for r in range(R):
        a_, b_ = rj[r], ak[r]
        la = list(nb[a_, :llen[a_]])
        lb = list(nb[b_, :llen[b_]])
        sup = set()
        for x in la:
            for y in lb:
                if x != y and cid[x] == cid[y] and abs(x - y) <= 3:
                    sup.add(gc[y])
        E = 0.0
        Eg = 0.0
        lps = {}
        for G in range(ng):
            ca_ = sum(gc[x] == G for x in la)
            cb_ = sum(gc[y] == G for y in lb)
            if ca_ and cb_:
                p = min(1.0, 6.0 * ca_ * cb_ / mg[G])
                E += p
                lps[grp[G]] = lps.get(grp[G], 0.0) + (math.log1p(-p) if p < 1 else -1e300)
        Eg = sum(1.0 - math.exp(v) for v in lps.values())
        grs = {grp[G] for G in sup}
        if len(sup) != on[r] or len(grs) != ong[r] or abs(E - oE[r]) > 1e-9 or abs(Eg - oEg[r]) > 1e-9:
            bad += 1
    log("smoke pass_count rows", R, "mismatches", bad)
    assert bad == 0
    d = 16
    X = rng.normal(size=(n, d)).astype(np.float32)
    uq = np.arange(n, dtype=np.int64)
    P = np.zeros((n + len(c_start), d), np.float32)
    prefix_sums(X, uq, gene_at, c_start, c_end, P)
    v = np.zeros(d)
    sig = rng.permutation(n).astype(np.int64)
    for _ in range(200):
        s = int(rng.integers(0, n - 3))
        e = s + int(rng.integers(0, 3))
        if cid[e] != cid[s]:
            continue
        nc = ctx_vec(s, e, 4, cid, c_start, c_end, P, sig, False, v)
        cs, ce = c_start[cid[s]], c_end[cid[s]]
        win = [x for x in range(s - 4, s) if x >= cs] + [x for x in range(e + 1, e + 5) if x < ce]
        ref = X[win].sum(0) if win else np.zeros(d)
        assert nc == len(win) and np.allclose(ref, v, atol=1e-4), (s, e)
    log("smoke ctx_vec OK")
    need = np.ones(R, np.bool_)
    neff = np.zeros(R)
    kinf = np.zeros(R, np.int64)
    cbar = np.zeros(R)
    prio = rng.permutation(ng).astype(np.int64)
    pass_neff(n, ns0, ns0, nterm0, lslot0, aptr, ak, need, gene_at, lstart, tcnt, nbf, tailf, nbg, wptr0, z0, z0, z0, z16,
              cid, c_start, c_end, ng, prio, P, 4, sig, False, 5, 0.01, 0.05, 2.0, K, neff, kinf, cbar, 3)
    bad = 0
    for r in range(R):
        a_, b_ = rj[r], ak[r]
        best = {}
        for tq, y in enumerate(nb[b_, :llen[b_]]):
            cands = [(int(tail[a_, tp]), tp, int(x)) for tp, x in enumerate(nb[a_, :llen[a_]])
                     if x != y and cid[x] == cid[y] and abs(int(x) - int(y)) <= 3]
            if not cands:
                continue
            right = [c_ for c_ in cands if c_[2] < y]
            left = [c_ for c_ in cands if c_[2] > y]
            opts = []
            if right:
                opts.append(min(right, key=lambda zz: zz[1]))
            if left:
                opts.append(min(left, key=lambda zz: zz[1]))
            p = opts[0] if (len(opts) == 1 or opts[0][0] <= opts[1][0]) else opts[1]
            ms, me = min(p[2], int(y)), max(p[2], int(y))
            mr = max(p[0], int(tail[b_, tq]))
            G = gc[y]
            if G not in best or (mr, ms) < (best[G][2], best[G][0]):
                best[G] = (ms, me, mr)
        nn = len(best) + 1
        gl = sorted(best.keys(), key=lambda G: prio[G])[:5]
        vecs = []
        for s, e in [(a_, b_)] + [(best[G][0], best[G][1]) for G in gl]:
            nc = ctx_vec(s, e, 4, cid, c_start, c_end, P, sig, False, v)
            if nc > 0 and np.linalg.norm(v) > 0:
                vecs.append(v / np.linalg.norm(v))
        k = len(vecs)
        rho = 0.0
        if k >= 2:
            Tv = np.sum(vecs, 0)
            cb = (Tv @ Tv - k) / (k * (k - 1))
            rho = min(1.0, max(0.0, (cb - 0.01 - 2.0 * 0.05 * math.sqrt(2 / (k * (k - 1)))) / (1 - 0.01)))
        ref = nn / (1 + (nn - 1) * rho)
        if abs(ref - neff[r]) > 1e-6 or k != kinf[r]:
            bad += 1
    log("smoke pass_neff mismatches", bad)
    assert bad == 0

    class NS:
        pass
    LS = ListSet("emb", nbf, tailf, nbg, lstart, {8: tcnt}, {})
    Of = NS()
    Of.gene_at, Of.nbs, Of.off, Of.excl, Of.P, Of.r = gene_at, {"emb": nbf}, off, {1000: excl, 200: excl}, P, -1
    Of.floor = {False: dict(c0=0.01, sd0=0.05), True: dict(c0=0.01, sd0=0.05)}
    Uf = NS()
    Uf.u = NS()
    Uf.u.n, Uf.u.ng, Uf.u.cid, Uf.u.gc, Uf.u.c_start, Uf.u.c_end = n, ng, cid, gc, c_start, c_end
    Uf.grp, Uf.ngroup, Uf.mg, Uf.prio, Uf.sig = grp, int(grp.max()) + 1, mg, prio, sig
    gch = np.flatnonzero(np.diff(gc)) + 1
    Uf.u.g_start, Uf.u.g_end = np.r_[0, gch].astype(np.int64), np.r_[gch, n].astype(np.int64)
    Of.cache = {}
    global CTX_W, VL
    CTX_W = 4
    VL = Path(os.environ.get("TMPDIR", "/tmp")) / "vl_smoke"
    (VL / "results").mkdir(parents=True, exist_ok=True)
    # synteny matrix and N_eff vs brute force
    Lr = Learner(Uf, LS, Of, 8, "syn", 1)
    S = Lr.syn_S()
    cntb = np.zeros((ng, ng))
    totb = np.zeros(ng)
    sups = []
    for r in range(R):
        a_, b_ = rj[r], ak[r]
        la, lb = list(nb[a_, :llen[a_]]), list(nb[b_, :llen[b_]])
        sup = {gc[y] for x in la for y in lb if x != y and cid[x] == cid[y] and abs(int(x) - int(y)) <= 3}
        sups.append(sup)
        if sup:
            totb[gc[a_]] += 1
            for h in sup:
                cntb[gc[a_], h] += 1
    Sd = cntb / np.maximum(totb, 1)[:, None]
    Sb = (Sd + Sd.T) / 2
    np.fill_diagonal(Sb, 1.0)
    assert np.allclose(S, Sb), "syn_matrix"
    st0 = Lr.init_state()
    ev = Lr.evaluate(st0, aptr, ak, 0.0)
    bad = 0
    for r in range(R):
        sup = sorted(sups[r], key=lambda G: prio[G])[:63]
        if not sups[r] or ev["lamG"][r] < 0.0:
            continue
        C = [gc[rj[r]]] + sup
        k = len(C)
        sb = np.mean([Sb[C[x], C[y]] for x in range(k) for y in range(x + 1, k)])
        nn = len(sups[r]) + 1
        if abs(nn / (1 + (nn - 1) * sb) - ev["neff"][r]) > 1e-9:
            bad += 1
    log("smoke syn mismatches", bad, "S offdiag mean", float(S[~np.eye(ng, dtype=bool)].mean()))
    assert bad == 0
    for arm in ("ctx", "gen", "grp", "syn"):
        Lr = Learner(Uf, LS, Of, 8, arm, 1)
        summ, rounds, stt, inst = Lr.run(2.0)
        assert stt["ns"][0] == 0 and stt["ne"][-1] == n - 1
        assert np.all(stt["ns"][1:] == stt["ne"][:-1] + 1)
        assert np.all(cid[stt["ns"]] == cid[stt["ne"]])
        for i in range(len(inst["s"])):
            a_, b_ = int(inst["a"][i]), int(inst["b"][i])
            sa = inst["s"][a_] if a_ >= 0 else -a_ - 1
            ea = inst["e"][a_] if a_ >= 0 else -a_ - 1
            sb = inst["s"][b_] if b_ >= 0 else -b_ - 1
            eb = inst["e"][b_] if b_ >= 0 else -b_ - 1
            assert sa == inst["s"][i] and eb == inst["e"][i] and 0 <= sb - ea - 1 <= 2, i
        top = ~stt["nterm"]
        for q in stt["lslot"][top]:
            for t in range(stt["wptr"][q], stt["wptr"][q + 1]):
                assert cid[stt["ws"][t]] == cid[stt["we"][t]] and stt["ws"][t] <= stt["we"][t]
                assert gc[stt["ws"][t]] == stt["wg"][t]
        # a word's list holds only spans in genomes other than its own
        for j in np.flatnonzero(top):
            q = stt["lslot"][j]
            assert not np.any(stt["wg"][stt["wptr"][q]:stt["wptr"][q + 1]] == gc[stt["ns"][j]])
        log("smoke learner", arm, summ, [x["committed"] for x in rounds])
    # a planted 6-gene block in 8 of 12 genomes must become one word in most of them
    Lr = Learner(Uf, LS, Of, 8, "gen", 1)
    summ, rounds, stt, inst = Lr.run(2.0)
    full = 0
    for g in range(ng):
        idx = np.flatnonzero((gc == g) & (fam >= 1000))
        if len(idx) == 6:
            s0, e0 = idx.min(), idx.max()
            full += int(any((stt["ns"][j] <= s0) and (stt["ne"][j] >= e0) for j in np.flatnonzero(~stt["nterm"])))
    log("planted blocks covered by one top word:", full, "of", sum(g % 3 != 2 for g in range(ng)))
    log("SMOKE OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("smoke", "refsr", "job", "diag"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--excl", default="genome", choices=("genome", "group"))
    ap.add_argument("--arms", default="ctx")
    ap.add_argument("--full-nulls", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--refsr", action="store_true")
    ap.add_argument("--recip", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(RD / "softrec/code"))
    {"smoke": cmd_smoke, "refsr": cmd_refsr, "job": cmd_job, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
