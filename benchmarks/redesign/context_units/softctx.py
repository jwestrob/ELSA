#!/usr/bin/env python
"""context_units Test 2: the embedding-only kernel (softrec round 2 = lineagefree softrec_lf configuration) with genomic
context in place of the lineage unit. No lineage array and no taxonomy are read.

Changes against softrec_lf (everything else is round 2's code, imported read-only):
  lists     zca10 kNN (k 2,000 unique) expanded to genes, depth 1,000; a gene b is excluded from a's list when b is in a's
            genome or when cos(C(a), C(b)) >= tau (same context: a relative or the same locus), in place of 'b in a's genus'.
  context   C(p) = unit sum of the projected zca10 vectors (random Gaussian 960 -> 128, seed 17; variant gc subtracts the
            genome mean) over positions [p - W, p + W] of p's contig minus [p - 3, p + 3]. Position-indexed (real order),
            so a null instance at slot s carries the real context of position s.
  tau       q99.9 of cos(C(p), C(q)) over 2,000,000 random position pairs in different genomes.
  statistic per instance (a at slot i, b at i + d): supporting genomes as in round 2 (lineage = genome); k = 1 + their
            count; c_bar = mean pairwise cosine of {C(i)} and the context of the first supporting b' in each genome;
            rho_z = clip((c_bar - c0 - z sd0 sqrt(2 / (k_inf (k_inf - 1)))) / (1 - c0), 0, 1) (amendment_1, z = 2);
            N_eff = k / (1 + (k - 1) rho_z); W_ctx = W_e50 / (1 + (k - 1) rho_z) (rank-weighted genome sum, deflated by
            the same design effect).
  rule      W_ctx >= t & N_eff >= 2 (round 2: W >= t & L >= 2), instance-level FDR <= 0.01 over three gene-order nulls,
            top 12 by count, most genes covered. Landscape value = N_eff.
  modules   Leiden families (validated edges, r2000) -> engine pair statistics on family tokens -> Louvain 0.3 with
            weights J * log2(max(N_eff, 2)), N_eff of each token pair from the same instance contexts (ctxrec pooling).
Readings R1 and DH (readings2), full-package scoring, paired genome bootstrap against round 2 with the genus (stored)
and with lineagefree emb_hom g0.75. Logs print odd numbers only.

usage: softctx.py run --variant gc --W 25 --z 2 --tag TAG --threads T
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctxrec as C  # noqa: E402
sys.path.insert(0, str(RD / "softrec/code"))
import softrec as SR  # noqa: E402
import softrec2 as S2  # noqa: E402
import readings2 as RG  # noqa: E402
sys.path.insert(0, str(RD / "lineagefree/code"))
import softrec_lf as SLF  # noqa: E402

B, BR = C.B, C.BR
CU = C.CU
LF = RD / "lineagefree"
R2 = RD / "softrec/r2"
W3 = 3
DP = 128
CAND = "rw1000_e50"
log = C.log


# ============================================================================== contexts
@njit(parallel=True, cache=True)
def position_contexts(P, c_start, c_end, cid, gc, mu_g, center, W, out, out_n):
    n = out.shape[0]
    d = out.shape[1]
    for p in prange(n):
        c = cid[p]
        cs, ce = c_start[c], c_end[c]
        a0 = max(cs, p - W)
        a1 = max(cs, p - 3)
        b0 = min(ce, p + 4)
        b1 = min(ce, p + W + 1)
        nl = max(a1 - a0, 0)
        nr = max(b1 - b0, 0)
        m = nl + nr
        out_n[p] = m
        s2 = 0.0
        for j in range(d):
            x = 0.0
            if nl > 0:
                x += P[a1 + c, j] - P[a0 + c, j]
            if nr > 0:
                x += P[b1 + c, j] - P[b0 + c, j]
            if center:
                x -= m * mu_g[gc[p], j]
            out[p, j] = x
            s2 += x * x
        if m == 0 or s2 <= 0:
            for j in range(d):
                out[p, j] = 0.0
        else:
            s = math.sqrt(s2)
            for j in range(d):
                out[p, j] /= s


@njit(parallel=True, cache=True)
def pos_cos(ia, ib, Cx, out):
    for s in prange(len(ia)):
        x = 0.0
        for j in range(Cx.shape[1]):
            x += Cx[ia[s], j] * Cx[ib[s], j]
        out[s] = x


@njit(parallel=True, cache=True)
def expand_knn_ctx(uq, gen, Cx, tau, I, D, mptr, members, Kmax, out_nb, out_tail, llen, n_ctx_excl, capcos):
    """softrec.expand_knn (one depth, no floors) with the genus test replaced by: other genome and context cosine < tau.
    capcos = cosine of the unique neighbour at which the list reached depth Kmax (as softrec's cos_at), else the last
    cosine consumed (-1 when the kNN row ran out)."""
    n = len(uq)
    Ku = I.shape[1]
    d = Cx.shape[1]
    for a in prange(n):
        U = uq[a]
        llen[a] = 0
        n_ctx_excl[a] = 0
        capcos[a] = 2.0
        if U < 0:
            continue
        ga = gen[a]
        k = 0
        j = -1
        last = 1.0
        exhausted = False
        reached = False
        while k < Kmax and j < Ku:
            if j < 0:
                V = U
                sim = 1.0
            else:
                V = I[U, j]
                if V < 0:
                    exhausted = True
                    break
                sim = np.float64(D[U, j])
            last = sim
            if j < 0 or V != U:
                k0 = k
                for t in range(mptr[V], mptr[V + 1]):
                    b = members[t]
                    if gen[b] == ga or k >= Kmax:
                        continue
                    c = 0.0
                    for q in range(d):
                        c += Cx[a, q] * Cx[b, q]
                    if c >= tau:
                        n_ctx_excl[a] += 1
                        continue
                    out_nb[a, k] = b
                    k += 1
                for t in range(k0, k):
                    out_tail[a, t] = k
                if k >= Kmax and not reached:
                    capcos[a] = sim
                    reached = True
            j += 1
        llen[a] = k
        if not reached:
            capcos[a] = -1.0 if exhausted else last


class CtxLists:
    """Dense n x 1000 lists (attributes used by softrec2.run_pass2-like passes and extract_edges)."""

    def __init__(self, u, X, uq, Cx, tau, threads):
        t0 = time.time()
        I = np.ascontiguousarray(np.load(SR.WORK / "knn/zca10__k2000__I.npy", mmap_mode="r"))
        D = np.ascontiguousarray(np.load(SR.WORK / "knn/zca10__k2000__D.npy", mmap_mode="r")).astype(np.float32)
        log("kNN loaded", I.shape, f"{time.time()-t0:.0f}s")
        U = I.shape[0]
        ok = uq >= 0
        order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
        mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=U))].astype(np.int64)
        K = 1000
        n = u.n
        nb = np.full((n, K), -1, np.int32)
        tail = np.zeros((n, K), np.uint16)
        llen = np.zeros(n, np.int64)
        nex = np.zeros(n, np.int64)
        ccos = np.zeros(n, np.float32)
        expand_knn_ctx(uq, u.gc.astype(np.int64), Cx, np.float32(tau), I, D, mptr, order.astype(np.int64), K, nb, tail, llen,
                       nex, ccos)
        del I, D
        self.nb = nb.ravel()
        self.tail = tail.ravel()
        self.nbg = np.where(self.nb >= 0, u.gc[np.maximum(self.nb, 0)], 0).astype(np.int16)
        self.lstart = np.arange(n, dtype=np.int64) * K
        self.lend = self.lstart + llen
        self.V = 1
        self.variants = ["K1000_fnone"]
        self.cap = np.minimum(llen, K).astype(np.int32)[None, :]
        self.capmax = self.cap[0].copy()
        # exclusion of homologous instance pairs (as round 2): identical sequence or cosine >= max(0.32, capcos)
        self.capcos = ccos[None, :]
        self.excl_mode = 1
        self.X = X
        self.uq = uq
        self.tok = np.zeros(1, np.int64)
        self.info = dict(depth=K, list_len_q01_q10_q50=np.quantile(llen, [0.01, 0.1, 0.5]).tolist(),
                         genes_list_shorter_than_depth=int((llen < K).sum()),
                         context_excluded_per_gene_q50_q90_q99=np.quantile(nex, [0.5, 0.9, 0.99]).tolist(),
                         context_excluded_total=int(nex.sum()), seconds=round(time.time() - t0, 1))
        log("lists", self.info)


# ============================================================================== kernel
@njit(parallel=True, cache=True)
def soft_pass_ctx(n, w, V, gene_at, cid, lstart, lend, nbs, nbg, tail, genome_gene, cap, capmax, excl,
                  inst_off, nd_arr, n_genome, ngenes_g, wpar, Cx, c0, sd0, z, out_L, out_G, out_E, out_W, out_N,
                  out_Wc, nchunks):
    """softrec2.soft_pass2 with lineage = genome (single variant, weight e50) plus context pooling over the instance's own
    context and the context of the first supporting b' in each supporting genome."""
    D = Cx.shape[1]
    for ch in prange(nchunks):
        lo = ch * n // nchunks
        hi = (ch + 1) * n // nchunks
        mstamp = np.full(n, -1, np.int32)
        mrank = np.zeros(n, np.int32)
        ca = np.zeros(n_genome, np.int32)
        cb = np.zeros(n_genome, np.int32)
        seen_a = np.full(n_genome, -1, np.int64)
        seen_b = np.full(n_genome, -1, np.int64)
        gst = np.full(n_genome, -1, np.int64)
        gmin = np.zeros(n_genome, np.int64)
        glist = np.zeros(n_genome, np.int32)
        tch_a = np.zeros(n_genome, np.int32)
        tch_b = np.zeros(n_genome, np.int32)
        m = np.zeros(D)
        for i in range(lo, hi):
            nd = nd_arr[i]
            if nd == 0:
                continue
            a = gene_at[i]
            ga = genome_gene[a]
            e0 = lstart[a]
            e1 = lend[a]
            if e1 > e0 + capmax[a]:
                e1 = e0 + capmax[a]
            nta = 0
            for e in range(e0, e1):
                G = nbg[e]
                r = e - e0
                s = nbs[e]
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                for uu in range(ulo, uhi):
                    if uu != s and cid[uu] == c and mstamp[uu] != i:
                        mstamp[uu] = i
                        mrank[uu] = r
                if seen_a[G] != i:
                    seen_a[G] = i
                    tch_a[nta] = G
                    nta += 1
                if r < cap[0, a]:
                    ca[G] += 1
            for d in range(1, nd + 1):
                inst = inst_off[i] + d - 1
                if excl[0, inst]:
                    out_L[0, inst] = -1
                    out_G[0, inst] = -1
                    out_E[0, inst] = 0.0
                    out_W[0, inst] = 0.0
                    out_N[0, inst] = 0.0
                    out_Wc[0, inst] = 0.0
                    continue
                b = gene_at[i + d]
                f0 = lstart[b]
                f1 = lend[b]
                if f1 > f0 + capmax[b]:
                    f1 = f0 + capmax[b]
                ntb = 0
                cntG = 0
                kinf = 0
                for q in range(D):
                    m[q] = Cx[i, q]
                x0 = 0.0
                for q in range(D):
                    x0 += m[q] * m[q]
                if x0 > 0:
                    kinf = 1
                for e in range(f0, f1):
                    G = nbg[e]
                    r = e - f0
                    if seen_b[G] != inst:
                        seen_b[G] = inst
                        tch_b[ntb] = G
                        ntb += 1
                    if r < cap[0, b]:
                        cb[G] += 1
                    s = nbs[e]
                    if mstamp[s] != i:
                        continue
                    ra = mrank[s]
                    mab = np.int64(tail[e0 + ra])
                    tb_ = np.int64(tail[e])
                    if tb_ > mab:
                        mab = tb_
                    if ra < cap[0, a] and r < cap[0, b] and G != ga:
                        if gst[G] != inst:
                            gst[G] = inst
                            gmin[G] = mab
                            glist[cntG] = G
                            cntG += 1
                            xs = 0.0
                            for q in range(D):
                                xs += Cx[s, q] * Cx[s, q]
                            if xs > 0:
                                kinf += 1
                                for q in range(D):
                                    m[q] += Cx[s, q]
                        elif mab < gmin[G]:
                            gmin[G] = mab
                esum = 0.0
                for kk in range(ntb):
                    G = tch_b[kk]
                    if ca[G] > 0 and cb[G] > 0:
                        x = 2.0 * w * ca[G] * cb[G] / ngenes_g[G]
                        esum += x if x < 1.0 else 1.0
                ssum = 0.0
                for kk in range(cntG):
                    ssum += math.exp(-(gmin[glist[kk]] - 1.0) / wpar)
                k = cntG + 1
                rho = 0.0
                if kinf >= 2:
                    s2 = 0.0
                    for q in range(D):
                        s2 += m[q] * m[q]
                    cbar = (s2 - kinf) / (kinf * (kinf - 1.0))
                    rho = (cbar - c0 - z * sd0 * math.sqrt(2.0 / (kinf * (kinf - 1.0)))) / (1.0 - c0)
                    if rho < 0.0:
                        rho = 0.0
                    if rho > 1.0:
                        rho = 1.0
                de = 1.0 + (k - 1.0) * rho
                out_L[0, inst] = cntG
                out_G[0, inst] = cntG
                out_E[0, inst] = esum
                out_W[0, inst] = ssum
                out_N[0, inst] = k / de
                out_Wc[0, inst] = ssum / de
                for kk in range(ntb):
                    cb[tch_b[kk]] = 0
            for kk in range(nta):
                ca[tch_a[kk]] = 0


def run_pass(u, L, gene_at, slot_of, nd, off, n_inst, threads, label, Cx, c0, sd0, z):
    t0 = time.time()
    nbs = np.empty_like(L.nb)
    SR.map_slots(L.nb, slot_of.astype(np.int32), nbs)
    excl = np.zeros((1, n_inst), np.bool_)
    SR.exclusion(u.n, 1, gene_at, nd, off, L.excl_mode, L.tok, L.X, L.uq, L.capcos, excl)
    oL = np.zeros((1, n_inst), np.int16)
    oG = np.zeros((1, n_inst), np.int16)
    oE = np.zeros((1, n_inst), np.float32)
    oW = np.zeros((1, n_inst), np.float32)
    oN = np.zeros((1, n_inst), np.float32)
    oWc = np.zeros((1, n_inst), np.float32)
    soft_pass_ctx(u.n, W3, 1, gene_at, u.cid, L.lstart, L.lend, nbs, L.nbg, L.tail, u.gc, L.cap, L.capmax, excl, off, nd,
                  u.ng, u.n_g, 50.0, Cx, c0, sd0, z, oL, oG, oE, oW, oN, oWc, threads * 8)
    del nbs
    log(f"pass {label}: {time.time()-t0:.0f}s excluded {int((oL < 0).sum())}; N_eff>=2 {int((oN >= 2).sum())}, "
        f"supporting genomes>=1 {int((oG >= 1).sum())}")
    return dict(L=oL[0], G=oG[0], N=oN[0], Wc=oWc[0], W=oW[0], E=oE[0])


W_T = S2.W_T


def select(n, obs, nulls, left, d):
    rows, masks = [], {}
    for t in W_T:
        m = (obs["L"] >= 0) & (obs["N"] >= 2) & (obs["Wc"] >= t)
        nn = float(np.mean([((p["L"] >= 0) & (p["N"] >= 2) & (p["Wc"] >= t)).sum() for p in nulls]))
        no = int(m.sum())
        rows.append(dict(rule=f"Wctx>={t:g}&Neff>=2", observed=no, null_mean=nn, fdr=nn / no if no else np.nan))
        masks[rows[-1]["rule"]] = m
    rules = pd.DataFrame(rows)
    cand = rules[(rules.fdr <= 0.01) & (rules.observed > 0)].sort_values("observed", ascending=False).head(12)
    best, best_cov = None, -1
    for name in cand.rule:
        cov = int(S2.covered(n, masks[name], left, d).sum())
        rules.loc[rules.rule == name, "genes_covered"] = cov
        if cov > best_cov:
            best, best_cov = name, cov
    return rules, best, (masks[best] if best else None)


def modules_ctx(ctx, land, tok, Pp, Xp, mu_gp, center, W, c0i, sd0i, z, threads, res=0.3, seed=17):
    """readings2.modules with the pair statistic's L replaced by N_eff of the token pair's instance contexts."""
    import networkx as nx
    t0 = time.time()
    u = ctx.u
    key, left, dist, T = B.R.colocalization(u, tok, W3)
    st = B.R.pair_statistics(u, tok, key, left, W3, T)
    ta = tok[land.sig_left]
    tb = tok[land.sig_left + land.sig_d]
    lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
    ns = lo != hi
    skeys = np.unique(lo[ns] * T + hi[ns])
    idx = np.searchsorted(st["key"], skeys)
    assert (idx < len(st["key"])).all() and (st["key"][idx] == skeys).all()
    # N_eff per significant token pair
    pos = np.minimum(np.searchsorted(skeys, key), len(skeys) - 1)
    found = skeys[pos] == key
    ii = np.flatnonzero(found)
    pidx = pos[found]
    gen = u.gc[left[ii]].astype(np.int64)
    o = np.lexsort((gen, pidx))
    ii, pidx, gen = ii[o], pidx[o], gen[o]
    ptr = np.searchsorted(pidx, np.arange(len(skeys) + 1)).astype(np.int64)
    ilo, ihi = left[ii], left[ii] + dist[ii]
    cid = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start).astype(np.int64)
    cbar = np.empty((len(skeys), 1))
    kk = np.zeros(len(skeys), np.int64)
    z1 = np.zeros(1, np.int64)
    C.pooled_cbar(ptr, gen, 0, np.array([W], np.int64), Pp, u.c_start, u.c_end, ilo, ihi, cid[ilo], gen, mu_gp, center,
                  Xp, np.zeros(1, np.int64), ilo, ihi, z1, z1, np.zeros((1, Xp.shape[1]), np.float32), threads * 16, cbar, kk)
    G = st["G"][idx].astype(np.float64)
    se = sd0i * np.sqrt(2.0 / np.maximum(kk * (kk - 1.0), 1.0))
    rho = np.clip((cbar[:, 0] - c0i - z * se) / (1.0 - c0i), 0.0, 1.0)
    rho[kk < 2] = 0.0
    Lp = G / (1.0 + (G - 1.0) * rho)
    Jp = st["J"][idx]
    slo, shi = skeys // T, skeys % T
    H = nx.Graph()
    wt = Jp * np.log2(np.maximum(Lp, 2).astype(np.float64))
    H.add_weighted_edges_from(zip(slo.tolist(), shi.tolist(), wt.tolist()))
    tm = np.full(T, -1, np.int64)
    for i, c in enumerate(nx.community.louvain_communities(H, weight="weight", resolution=res, seed=seed)):
        if len(c) >= 2:
            tm[np.fromiter(c, dtype=np.int64)] = i
    pm = tm[lo].copy()
    pm[tm[hi] != pm] = -1
    sizes = np.bincount(tm[tm >= 0]) if (tm >= 0).any() else np.zeros(1, np.int64)
    sizes = sizes[sizes > 0]
    info = dict(tokens=int(T), significant_token_pairs=int(len(skeys)), modules=int(len(sizes)),
                tokens_in_modules=int((tm >= 0).sum()), neff_over_G_q50=float(np.median(Lp / G)),
                louvain_res=res, seconds=round(time.time() - t0, 1))
    log("modules", info)
    return dict(tm=tm, pm=pm, gene_module=tm[tok], info=info, sp=dict(lo=slo, hi=shi, J=Jp, L=Lp))


def evaluate_odd_log(ctx, tables, land, threads):
    res, pers = {}, {}
    for name, df in tables.items():
        t0 = time.time()
        per, _ = B.score_table(C.PKG, df, threads)
        g = per.genome.map(ctx.gcode).to_numpy()
        summ = {sp: B.summarize_per(per[g % 2 == (1 if sp == "odd" else 0)].reset_index(drop=True)) for sp in ("odd", "even")}
        sc = B.ribo_score(ctx.runs_all, df.gid_s.to_numpy(), df.gid_e.to_numpy())
        er = ctx.runs_even_mask
        rib = dict(scored=B.ribo_summary(ctx.runs_all, sc),
                   odd_runs=B.ribo_summary(ctx.runs_all[~er].reset_index(drop=True), sc[~er].reset_index(drop=True)),
                   even_runs=B.ribo_summary(ctx.runs_all[er].reset_index(drop=True), sc[er].reset_index(drop=True)))
        res[name] = dict(summary=summ, ribo=rib, loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), ctx.u.n),
                         seconds=round(time.time() - t0, 1))
        pers[name] = (per, sc)
        od = summ["odd"]
        log(land.name, name, "ODD", round(od["summed_DF_TXSS_excess"], 4), {c[:4]: round(od[c]["f1"], 3) for c in C.CALLERS},
            "ribo odd", rib["odd_runs"]["recovered"], rib["odd_runs"]["resolved"], rib["odd_runs"]["parvarchaeaceae"])
    return res, pers


def cmd_run(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    out = CU / "softctx" / a.tag
    cd = out / "cand"
    cd.mkdir(parents=True, exist_ok=True)
    res = dict(tag=a.tag, variant=a.variant, W=a.W, z=a.z, timing={})
    u = SLF.LFUniverse(LF / "groups/genome_map.parquet")       # every genome its own unit (lineage array unused below)
    nd, off, left, d = u.instances(W3)
    n_inst = len(left)
    X, uq, _ = C.load_vectors(type("b", (), {"u": u})(), a.threads)
    rng = np.random.default_rng(17)
    R = (rng.standard_normal((X.shape[1], DP)) / math.sqrt(DP)).astype(np.float32)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=a.threads):
            Xp = np.ascontiguousarray(X @ R)
    except ImportError:
        Xp = np.ascontiguousarray(X @ R)
    nc = len(u.c_start)
    Pp = np.empty((u.n + nc, DP), np.float32)
    C.build_prefix(Xp, uq, u.c_start, u.c_end, Pp)
    mu_gp = np.zeros((u.ng, DP))
    cs_ = Pp[u.c_end + np.arange(nc)].astype(np.float64) - Pp[u.c_start + np.arange(nc)].astype(np.float64)
    np.add.at(mu_gp, u.gc[u.c_start], cs_)
    mu_gp /= np.bincount(u.gc, minlength=u.ng)[:, None]
    center = a.variant == "gc"
    cid64 = u.cid.astype(np.int64)
    Cx = np.zeros((u.n, DP), np.float32)
    ncx = np.zeros(u.n, np.int64)
    position_contexts(Pp, u.c_start, u.c_end, cid64, u.gc.astype(np.int64), mu_gp, center, a.W, Cx, ncx)
    # floors: position contexts (list exclusion, kernel) and instance contexts (module N_eff), different genomes
    ia = rng.integers(0, u.n, 2_000_000)
    ib = rng.integers(0, u.n, 2_000_000)
    ok = (u.gc[ia] != u.gc[ib]) & (ncx[ia] > 0) & (ncx[ib] > 0)
    pc = np.empty(int(ok.sum()))
    pos_cos(ia[ok], ib[ok], Cx, pc)
    c0, sd0, tau = float(pc.mean()), float(pc.std()), float(np.quantile(pc, 0.999))
    lo_all, hi_all = left, left + d.astype(np.int64)
    ja = rng.integers(0, n_inst, 2_000_000)
    jb = rng.integers(0, n_inst, 2_000_000)
    okj = u.gc[lo_all[ja]] != u.gc[lo_all[jb]]
    ja, jb = ja[okj], jb[okj]
    od = np.empty((len(ja), 1))
    on = np.empty((len(ja), 1), np.int64)
    z1 = np.zeros(1, np.int64)
    gen_inst = u.gc[lo_all].astype(np.int64)
    C.floor_sample(ja, jb, 0, np.array([a.W], np.int64), Pp, u.c_start, u.c_end, lo_all, hi_all, cid64[lo_all], gen_inst,
                   mu_gp, center, Xp, z1, lo_all, hi_all, z1, z1, np.zeros((1, DP), np.float32), od, on)
    c0i, sd0i = float(np.nanmean(od)), float(np.nanstd(od))
    res["floors"] = dict(position=dict(c0=c0, sd=sd0, tau_q999=tau, zero_context=float((ncx == 0).mean())),
                         instance=dict(c0=c0i, sd=sd0i))
    log("contexts", res["floors"], f"{time.time()-t0:.0f}s")
    # ---- lists, passes ---------------------------------------------------------------------------------------------
    t1 = time.time()
    L = CtxLists(u, X, uq, Cx, tau, a.threads)
    res["lists"] = L.info
    ident = np.arange(u.n, dtype=np.int64)
    obs = run_pass(u, L, ident, ident, nd, off, n_inst, a.threads, "observed", Cx, c0, sd0, a.z)
    nulls = []
    for r in range(3):
        ga, so = SR.null_order(u, r)
        nulls.append(run_pass(u, L, ga, so, nd, off, n_inst, a.threads, f"null {r}", Cx, c0, sd0, a.z))
    res["timing"]["lists_and_passes"] = round(time.time() - t1, 1)
    rules, best, sig = select(u.n, obs, nulls, left, d)
    rules.to_csv(cd / "rules.tsv", sep="\t", index=False)
    assert best is not None, "no admissible rule"
    row = rules[rules.rule == best].iloc[0]
    sidx = np.flatnonzero(sig)
    sidx = sidx[np.lexsort((left[sidx], d[sidx]))]
    val = obs["N"][sidx].astype(np.float32)
    b, arg = SR.boundary_support_argmax(left[sidx], d[sidx], val, np.arange(len(sidx), dtype=np.int64), u.n)
    genes = S2.covered(u.n, sig, left, d)
    res["candidate"] = dict(rule=best, fdr=float(row.fdr), significant_instances=int(len(sidx)),
                            genes_covered=int(genes.sum()), N_q10_q50_q90=np.quantile(val, [.1, .5, .9]).tolist(),
                            supporting_genomes_q50=float(np.median(obs["G"][sidx])))
    log("candidate", res["candidate"])
    for k_, v_ in (("sidx", sidx), ("value", val), ("b", b), ("arg", arg)):
        np.save(cd / f"{k_}.npy", v_)
    del obs, nulls
    # ---- validated edges, Leiden families --------------------------------------------------------------------------
    t1 = time.time()
    cap = np.full(len(left), 1000, np.int16)
    res["edges"] = S2.extract_edges(u, L, sidx, left, d, cap[sidx], a.threads, cd)
    del L
    ea, eb, ew = np.load(cd / "edges_a.npy"), np.load(cd / "edges_b.npy"), np.load(cd / "edges_w.npy")
    U = int(uq.max()) + 1
    memb, t_build, t_leiden = SLF.leiden(ea, eb, ew, U)
    del ea, eb, ew
    np.save(cd / "fam_l2000.npy", memb)
    sz = np.bincount(memb)
    res["families"] = dict(communities=int(len(sz)), nontrivial=int((sz >= 2).sum()), largest=int(sz.max()))
    res["timing"]["edges_leiden"] = round(time.time() - t1, 1)
    log("families", res["families"], f"{time.time()-t0:.0f}s")
    # ---- modules, readings, scoring --------------------------------------------------------------------------------
    t1 = time.time()
    ctx = RG.Ctx()
    land = RG.Land(a.tag, ctx.left[sidx], ctx.d[sidx], val, b, arg, info=res["candidate"])
    g = memb.astype(np.int64)[ctx.uq]
    _, tok = np.unique(g, return_inverse=True)
    mod = modules_ctx(ctx, land, tok.astype(np.int64), np.ascontiguousarray(Pp), Xp, mu_gp, center, a.W, c0i, sd0i, a.z,
                      a.threads)
    res["modules"] = mod["info"]
    tabs = RG.readings(ctx, land, mod, ["R1", "DH"])
    ev, pers = evaluate_odd_log(ctx, tabs, land, a.threads)
    res["readings"] = ev
    res["timing"]["modules_readings_scoring"] = round(time.time() - t1, 1)
    res["paired"] = {}
    refs = {"genus_round2": lambda rn: R2 / "scores/all" / f"selected__{CAND}__l2000__0.3__{rn}.parquet",
            "lf_emb_hom_g0.75": lambda rn: LF / "scores" / f"sr_emb_hom_g0.75__{rn}.parquet",
            "lf_genome": lambda rn: LF / "scores" / f"sr_genome__{rn}.parquet"}
    for rn, (per, sc) in pers.items():
        per.to_parquet(CU / "scores" / f"sr_{a.tag}__{rn}.parquet", index=False)
        tabs[rn].to_parquet(CU / "loci" / f"sr_{a.tag}__{rn}.parquet", index=False)
        pi = per.set_index("instance_id")
        for cname, fn in refs.items():
            f = fn(rn)
            if not f.exists():
                continue
            ref = pd.read_parquet(f)
            if "excess" not in ref:
                ref["excess"] = ref.f1 - ref.control_mean_f1
            ref = ref.set_index("instance_id")
            for sp in ("even", "odd"):
                res["paired"][f"{sp}:{rn}:{cname}"] = BR.paired(pi, ref, ctx.gcode, sp)
        pr = res["paired"][f"odd:{rn}:genus_round2"]
        log(a.tag, rn, "ODD d summed vs genus round 2", [round(x, 4) for x in pr["summed"]],
            {k[:4]: round(pr[k]["d_f1"][0], 4) for k in C.CALLERS})
    res["seconds"] = round(time.time() - t0, 1)
    C.jdump(CU / "results" / "full" / f"softctx_{a.tag}.json", res)
    odd = {k: v for k, v in res.items() if k not in ("readings", "paired")}
    odd["readings_odd"] = {k: dict(summary_odd=v["summary"]["odd"], ribo_odd=v["ribo"]["odd_runs"], loci=v["loci"])
                           for k, v in res["readings"].items()}
    odd["paired_odd"] = {k: v for k, v in res["paired"].items() if k.startswith("odd:")}
    C.jdump(CU / "results" / f"softctx_{a.tag}.odd.json", odd)
    log("done", res["seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--variant", choices=("raw", "gc"), required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    for sub in ("results/full", "scores", "loci", "softctx"):
        (CU / sub).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
