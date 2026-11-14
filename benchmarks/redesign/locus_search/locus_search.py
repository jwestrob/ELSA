#!/usr/bin/env python
"""Locus search: query a set of genes (a locus, or a profile of slots) against every gene of every
other genome, and return windows that co-localize matches to several slots.

Evidence. For query slot i and database gene g, s_i(g) = -log p, where p is the fraction of
database genes that match slot i at least as well:
  token (M0)      p = n_db(slot tokens) / N_db, for every gene carrying one of the slot's tokens
  embedding (M1)  p = (number of database genes with cosine >= cos(i, g)) / N_db, top-K genes only
  combined (M2)   s = max(s_token, w * s_embedding)
N_db = genes outside the query genome. Both scales are -log of a per-gene tail probability, so the
evidence of different slots and different methods is comparable.

Hits. A hit is a window of <= W genes on one contig. Inside a window one target gene serves one
slot (greedy matching in descending evidence). Window score
    S = sum over filled slots of (s - log L),  counting slots with s - log L >= tau,
where L is the window span in genes (log L is the Sidak correction for taking the best of L genes).
Windows whose first or last gene contributes nothing are dominated by a shorter window and are not
scored. Non-overlapping hits by greedy selection on S.

E-values. R replicates of a co-localization null: each slot's evidence track is circularly shifted
by an independent random offset along the genome-ordered gene array (gid order), which preserves
each slot's marginal score distribution and its own local clustering and destroys co-localization
between slots. E(S) = mean number of null hits with score >= S per replicate.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
# environment overrides exist for synthetic tests only
PKG = Path(os.environ.get("LS_PKG", RD / "ws_dpann/package"))
TOKENS = Path(os.environ.get("LS_TOKENS", RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet"))
EMB = Path(os.environ.get("LS_EMB", "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/"
                                    "dpann__esmpp_small__20261111/final/l2_f32__row.npy"))

LAB_UNLABELLED, LAB_CORRECT, LAB_WRONG, LAB_SAMETYPE = 0, 1, 2, 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =========================================================================== kernels
@njit(parallel=True, cache=True)
def topk_rows(S, K, lo, hi, out_idx, out_cos, out_s):
    """Row-wise top-K of a score block S (n x N), with columns [lo[r], hi[r]) masked (own genome).
    out_s = log(N_db / tail rank), tail rank = number of unmasked columns with value >= the value."""
    n, N = S.shape
    for r in prange(n):
        row = S[r]
        for j in range(lo[r], hi[r]):
            row[j] = -np.inf
        n_db = N - (hi[r] - lo[r])
        kth = np.partition(row, N - K)[N - K]
        idx = np.empty(K, np.int64)
        val = np.empty(K, np.float32)
        m = 0
        for j in range(N):
            if row[j] > kth:
                idx[m] = j
                val[m] = row[j]
                m += 1
        m_gt = m
        n_eq = 0
        for j in range(N):
            if row[j] == kth:
                n_eq += 1
                if m < K:
                    idx[m] = j
                    val[m] = kth
                    m += 1
        o = np.argsort(-val[:m], kind="mergesort")
        cur = 0
        for t in range(m - 1, -1, -1):
            v = val[o[t]]
            if t == m - 1 or v != val[o[t + 1]]:
                cur = (t + 1) if v > kth else (m_gt + n_eq)
            out_idx[r, t] = idx[o[t]]
            out_cos[r, t] = v
            out_s[r, t] = math.log(n_db / cur)
        for t in range(m, K):
            out_idx[r, t] = -1
            out_cos[r, t] = -np.inf
            out_s[r, t] = 0.0


@njit(nogil=True, cache=True)
def scan(e_gid, e_slot, e_s, n_slots, cc, pos, W, tau, top_n):
    """Windows of <= W genes over candidate entries sorted by gid; greedy one-to-one slot filling;
    non-overlapping selection by score. Returns kept (first gid, last gid, score, slots filled),
    in descending score; top_n > 0 stops after that many kept windows."""
    n = e_gid.shape[0]
    ug = np.empty(n + 1, np.int64)
    uptr = np.empty(n + 1, np.int64)
    nu = 0
    for k in range(n):
        if k == 0 or e_gid[k] != e_gid[k - 1]:
            ug[nu] = e_gid[k]
            uptr[nu] = k
            nu += 1
    uptr[nu] = n
    cap = 1024
    wa = np.empty(cap, np.int64)
    wb = np.empty(cap, np.int64)
    ws = np.empty(cap, np.float64)
    wf = np.empty(cap, np.int64)
    nw = 0
    slot_st = np.full(max(n_slots, 1), -1, np.int64)
    gene_st = np.full(max(nu, 1), -1, np.int64)
    bufv = np.empty(max(n, 1), np.float64)
    bufs = np.empty(max(n, 1), np.int64)
    bufg = np.empty(max(n, 1), np.int64)
    stamp = 0
    for a in range(nu):
        ca = cc[ug[a]]
        pa = pos[ug[a]]
        m = 0
        b = a
        while b < nu:
            gb = ug[b]
            if cc[gb] != ca:
                break
            L = pos[gb] - pa + 1
            if L > W:
                break
            for k in range(uptr[b], uptr[b + 1]):
                v = e_s[k]
                j = m
                while j > 0 and bufv[j - 1] < v:
                    bufv[j] = bufv[j - 1]
                    bufs[j] = bufs[j - 1]
                    bufg[j] = bufg[j - 1]
                    j -= 1
                bufv[j] = v
                bufs[j] = e_slot[k]
                bufg[j] = b
                m += 1
            corr = math.log(L)
            thr = tau + corr
            stamp += 1
            sc = 0.0
            nf = 0
            ua = False
            ub = False
            for j in range(m):
                v = bufv[j]
                if v < thr:
                    break
                sl = bufs[j]
                gg = bufg[j]
                if slot_st[sl] == stamp or gene_st[gg] == stamp:
                    continue
                slot_st[sl] = stamp
                gene_st[gg] = stamp
                sc += v - corr
                nf += 1
                if gg == a:
                    ua = True
                if gg == b:
                    ub = True
            if ua and ub and sc > 0.0:
                if nw == cap:
                    cap *= 2
                    wa2 = np.empty(cap, np.int64)
                    wb2 = np.empty(cap, np.int64)
                    ws2 = np.empty(cap, np.float64)
                    wf2 = np.empty(cap, np.int64)
                    wa2[:nw] = wa[:nw]
                    wb2[:nw] = wb[:nw]
                    ws2[:nw] = ws[:nw]
                    wf2[:nw] = wf[:nw]
                    wa, wb, ws, wf = wa2, wb2, ws2, wf2
                wa[nw] = a
                wb[nw] = b
                ws[nw] = sc
                wf[nw] = nf
                nw += 1
            b += 1
    order = np.argsort(-ws[:nw], kind="mergesort")
    used = np.zeros(max(nu, 1), np.bool_)
    lim = nw if top_n <= 0 else min(nw, top_n)
    ka = np.empty(lim, np.int64)
    kb = np.empty(lim, np.int64)
    ks = np.empty(lim, np.float64)
    kf = np.empty(lim, np.int64)
    nk = 0
    for t in range(nw):
        i = order[t]
        a = wa[i]
        b = wb[i]
        ok = True
        for j in range(a, b + 1):
            if used[j]:
                ok = False
                break
        if not ok:
            continue
        for j in range(a, b + 1):
            used[j] = True
        ka[nk] = ug[a]
        kb[nk] = ug[b]
        ks[nk] = ws[i]
        kf[nk] = wf[i]
        nk += 1
        if nk >= lim:
            break
    return ka[:nk], kb[:nk], ks[:nk], kf[:nk]


@njit(nogil=True, cache=True)
def null_tail(e_gid, e_slot, e_s, n_slots, cc, pos, N, own_lo, own_hi, W, tau, R, seed, top_n):
    """Top `top_n` hit scores in each of R replicates of the circular-shift null (R x top_n, -inf pad)."""
    np.random.seed(seed)
    out = np.full((R, top_n), -np.inf)
    n = e_gid.shape[0]
    g2 = np.empty(n, np.int64)
    offs = np.empty(max(n_slots, 1), np.int64)
    for r in range(R):
        for i in range(n_slots):
            offs[i] = np.random.randint(0, N)
        for k in range(n):
            g = (e_gid[k] + offs[e_slot[k]]) % N
            if g >= own_lo and g < own_hi:
                g = -1
            g2[k] = g
        order = np.argsort(g2, kind="mergesort")
        st = 0
        while st < n and g2[order[st]] < 0:
            st += 1
        idx = order[st:]
        _, _, sc, _ = scan(g2[idx], e_slot[idx], e_s[idx], n_slots, cc, pos, W, tau, top_n)
        for t in range(sc.shape[0]):
            out[r, t] = sc[t]
    return out


@njit(nogil=True, cache=True)
def assign_windows(e_gid, e_slot, e_s, n_slots, pos, ha, hb, tau):
    """Slot assignment (greedy, as in `scan`) inside given windows. Returns (hit index, slot, gid)."""
    out_h = []
    out_s = []
    out_g = []
    for h in range(ha.shape[0]):
        lo = np.searchsorted(e_gid, ha[h], "left")
        hi = np.searchsorted(e_gid, hb[h], "right")
        m = hi - lo
        if m <= 0:
            continue
        L = pos[hb[h]] - pos[ha[h]] + 1
        corr = math.log(L)
        v = e_s[lo:hi]
        o = np.argsort(-v, kind="mergesort")
        used_s = np.zeros(max(n_slots, 1), np.bool_)
        used_g = np.zeros(m, np.bool_)
        for t in range(m):
            k = o[t]
            if v[k] - corr < tau:
                break
            sl = e_slot[lo + k]
            # gene identity inside the window: entries of one gene are adjacent in gid order
            gk = k
            while gk > 0 and e_gid[lo + gk - 1] == e_gid[lo + k]:
                gk -= 1
            if used_s[sl] or used_g[gk]:
                continue
            used_s[sl] = True
            used_g[gk] = True
            out_h.append(h)
            out_s.append(sl)
            out_g.append(e_gid[lo + k])
    return np.array(out_h, np.int64), np.array(out_s, np.int64), np.array(out_g, np.int64)


@njit(nogil=True, cache=True)
def label_hits(ha, hb, qtype, gptr, ginst, inst_type, inst_nmem, inst_target):
    """Per hit: 1 correct (covers >= 50% of the members of a same-type target instance), 2 wrong
    (otherwise overlaps a member of a different-type instance), 3 same-type other (overlaps a
    same-type instance below 50% or a non-target same-type instance), 0 unlabelled."""
    n = ha.shape[0]
    lab = np.zeros(n, np.int8)
    match = np.full(n, -1, np.int64)
    cov = np.zeros(n, np.float64)
    ids = np.empty(8192, np.int64)
    cnt = np.zeros(8192, np.int64)
    for h in range(n):
        m = 0
        for g in range(ha[h], hb[h] + 1):
            for k in range(gptr[g], gptr[g + 1]):
                t = ginst[k]
                f = -1
                for j in range(m):
                    if ids[j] == t:
                        f = j
                        break
                if f < 0:
                    if m < 8192:
                        ids[m] = t
                        cnt[m] = 1
                        m += 1
                else:
                    cnt[f] += 1
        best = -1.0
        bt = -1
        wrong = False
        same = False
        for j in range(m):
            t = ids[j]
            if inst_type[t] == qtype:
                c = cnt[j] / inst_nmem[t]
                if inst_target[t] and c >= 0.5:
                    if c > best or (c == best and t < bt):
                        best = c
                        bt = t
                else:
                    same = True
            else:
                wrong = True
        if bt >= 0:
            lab[h] = 1
            match[h] = bt
            cov[h] = best
        elif wrong:
            lab[h] = 2
        elif same:
            lab[h] = 3
    return lab, match, cov


# =========================================================================== database
class Database:
    """Gene universe in gid order, token index and (optionally) the embedding matrix."""

    def __init__(self, pkg=PKG, tokens=TOKENS, emb=EMB, vectors="ram"):
        pkg = Path(pkg)
        g = pd.read_parquet(pkg / "universe/genes.parquet",
                            columns=["gid", "genome_code", "contig_code", "position_index"])
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g))), "gid must be 0..N-1 in order"
        self.N = len(g)
        self.cc = g.contig_code.to_numpy(np.int64)
        self.pos = g.position_index.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int64)
        assert np.all(np.diff(self.gc) >= 0), "genomes must be contiguous in gid order"
        assert np.all(np.diff(self.cc) >= 0), "contigs must be contiguous in gid order"
        self.n_genomes = int(self.gc.max()) + 1
        ar = np.arange(self.n_genomes)
        self.g_lo = np.searchsorted(self.gc, ar, "left").astype(np.int64)
        self.g_hi = np.searchsorted(self.gc, ar, "right").astype(np.int64)
        self.contigs = pd.read_parquet(pkg / "universe/contigs.parquet").set_index("contig_code")
        gm = pd.read_parquet(pkg / "universe/genomes.parquet", columns=["genome", "genome_code"])
        self.genome_name = gm.sort_values("genome_code").genome.to_numpy()
        t = pd.read_parquet(tokens)
        tok = np.full(self.N, -1, np.int64)
        tok[t.gid.to_numpy()] = t.token.to_numpy()
        self.tok = tok
        order = np.argsort(tok, kind="stable")
        order = order[tok[order] >= 0]
        self.n_tokens = int(tok.max()) + 1
        self.tok_order = order.astype(np.int64)                     # gids sorted by (token, gid)
        self.tok_ptr = np.searchsorted(tok[order], np.arange(self.n_tokens + 1)).astype(np.int64)
        self.tok_n = np.diff(self.tok_ptr)
        self.X = None
        if vectors == "ram":
            self.X = np.load(emb)
        elif vectors == "mmap":
            self.X = np.load(emb, mmap_mode="r")

    def own_range(self, genome_code):
        return int(self.g_lo[genome_code]), int(self.g_hi[genome_code])

    def n_db(self, genome_code):
        lo, hi = self.own_range(genome_code)
        return self.N - (hi - lo)

    def token_members(self, token, genome_code):
        """Sorted gids carrying `token`, outside genome `genome_code`."""
        if token < 0:
            return np.empty(0, np.int64)
        m = self.tok_order[self.tok_ptr[token]:self.tok_ptr[token + 1]]
        lo, hi = self.own_range(genome_code)
        a, b = np.searchsorted(m, lo), np.searchsorted(m, hi)
        return np.concatenate([m[:a], m[b:]])

    def window_rows(self, ha, hb):
        """Hit table columns (genome, contig, start, end in position_index) for gid windows."""
        cc = self.cc[ha]
        c = self.contigs.loc[cc]
        return pd.DataFrame(dict(genome=c.genome.to_numpy(), contig=c.contig.to_numpy(),
                                 start=self.pos[ha], end=self.pos[hb], start_gid=ha, end_gid=hb))


def topk_cosine(db, vectors, own_genomes, K=20000, chunk=256, log_every=20):
    """Exact top-K cosine neighbours of each vector among database genes outside its own genome.
    Returns idx (int32, -1 pad), cos (float16), s = log(N_db / tail rank) (float32)."""
    assert db.X is not None
    n = len(vectors)
    idx = np.empty((n, K), np.int32)
    cos = np.empty((n, K), np.float16)
    s = np.empty((n, K), np.float32)
    lo = db.g_lo[own_genomes].astype(np.int64)
    hi = db.g_hi[own_genomes].astype(np.int64)
    t0 = time.time()
    oi = np.empty((chunk, K), np.int64)
    oc = np.empty((chunk, K), np.float32)
    os_ = np.empty((chunk, K), np.float32)
    for c, a in enumerate(range(0, n, chunk)):
        b = min(n, a + chunk)
        S = np.ascontiguousarray(vectors[a:b], dtype=np.float32) @ db.X.T
        topk_rows(S, K, lo[a:b], hi[a:b], oi[:b - a], oc[:b - a], os_[:b - a])
        idx[a:b] = oi[:b - a]
        cos[a:b] = oc[:b - a]
        s[a:b] = os_[:b - a]
        del S
        if log_every and c % log_every == 0:
            log(f"top-K {b:,}/{n:,} vectors, {time.time() - t0:.0f}s")
    return idx, cos, s


# =========================================================================== queries and profiles
def slot_evidence(db, own, tokens, nbr_gid, nbr_s, mode, w=1.0):
    """Evidence list (sorted gids, s) of one slot. tokens: the slot's token set; nbr_*: its
    embedding neighbours (gid, s). mode: 'tok', 'emb' or 'comb' (max of token and w * embedding)."""
    N_db = db.n_db(own)
    gt = np.empty(0, np.int64)
    if mode in ("tok", "comb"):
        parts = [db.token_members(int(t), own) for t in np.unique(np.asarray(tokens, np.int64)) if t >= 0]
        if parts:
            gt = np.sort(np.concatenate(parts))
    st = np.full(len(gt), math.log(N_db / len(gt)) if len(gt) else 0.0)
    if mode == "tok":
        return gt, st
    keep = nbr_gid >= 0
    ge = nbr_gid[keep].astype(np.int64)
    se = nbr_s[keep].astype(np.float64) * (w if mode == "comb" else 1.0)
    if mode == "emb":
        o = np.argsort(ge, kind="stable")
        return ge[o], se[o]
    g = np.concatenate([gt, ge])
    s = np.concatenate([st, se])
    o = np.lexsort((-s, g))
    g, s = g[o], s[o]
    first = np.r_[True, g[1:] != g[:-1]] if len(g) else np.zeros(0, bool)
    return g[first], s[first]


def assemble(slot_lists):
    """Concatenate per-slot (gid, s) lists into entries sorted by gid."""
    if not slot_lists:
        z = np.empty(0, np.int64)
        return z, z.copy(), np.empty(0, np.float64)
    g = np.concatenate([x[0] for x in slot_lists]).astype(np.int64)
    sl = np.concatenate([np.full(len(x[0]), i, np.int64) for i, x in enumerate(slot_lists)])
    s = np.concatenate([x[1] for x in slot_lists]).astype(np.float64)
    o = np.argsort(g, kind="stable")
    return g[o], sl[o], s[o]


def evalues(scores, tail, R):
    """E(S) = (null hits with score >= S) / R, and whether the stored tails make it exact."""
    flat = np.sort(tail[np.isfinite(tail)])
    cnt = len(flat) - np.searchsorted(flat, scores, side="left")
    full = np.isfinite(tail[:, -1])
    floor = tail[full, -1].max() if full.any() else -np.inf
    return cnt / R, scores > floor


def run_entries(db, own, entries, n_slots, W, tau, R=200, seed=17, top_n=128):
    """Scan + null calibration for prepared entries. Returns dict of hit arrays (descending score)."""
    eg, es_, ev = entries
    ha, hb, sc, nf = scan(eg, es_, ev, n_slots, db.cc, db.pos, W, tau, 0)
    out = dict(ha=ha, hb=hb, score=sc, nfill=nf)
    if R:
        lo, hi = db.own_range(own)
        tail = null_tail(eg, es_, ev, n_slots, db.cc, db.pos, db.N, lo, hi, W, tau, R, seed, top_n)
        out["E"], out["E_exact"] = evalues(sc, tail, R)
        out["null_top"] = tail[:, 0]
    return out


def search(db, query_gids, method="comb", W=None, tau=1.0, w=0.75, K=20000, R=200, seed=17,
           top_n=128, nbrs=None, max_E=10.0):
    """Search one locus (genes of one genome) against every gene of every other genome.

    method: 'tok' (M0), 'emb' (M1) or 'comb' (M2). W defaults to the query span + 2 genes (cap 60).
    nbrs: optional precomputed (idx, s) top-K rows aligned with the sorted unique query gids.
    Returns a DataFrame of hits with E <= max_E, in descending score."""
    q = np.unique(np.asarray(query_gids, np.int64))
    own = int(db.gc[q[0]])
    assert np.all(db.gc[q] == own), "query genes must come from one genome"
    if W is None:
        W = min(60, int(db.pos[q].max() - db.pos[q].min() + 1) + 2)
    if method != "tok" and nbrs is None:
        i, _, s = topk_cosine(db, db.X[q], np.full(len(q), own), K=K, log_every=0)
        nbrs = (i, s)
    lists = []
    for k, g in enumerate(q):
        ni, ns = (nbrs[0][k], nbrs[1][k]) if nbrs is not None else (np.empty(0, np.int64), np.empty(0))
        lists.append(slot_evidence(db, own, [db.tok[g]], ni, ns, method, w))
    res = run_entries(db, own, assemble(lists), len(q), W, tau, R, seed, top_n)
    keep = res["E"] <= max_E if R else np.ones(len(res["score"]), bool)
    df = db.window_rows(res["ha"][keep], res["hb"][keep])
    df["score"] = res["score"][keep]
    df["n_slots_filled"] = res["nfill"][keep]
    if R:
        df["evalue"] = res["E"][keep]
        df["evalue_exact"] = res["E_exact"][keep]
    return df.reset_index(drop=True)


# =========================================================================== evaluation
def evaluate(score, lab, match, E, rel_rows, rel_vis, thresholds=(0.01, 1.0)):
    """Per-query retrieval metrics. Hits in any order; ranked by score (descending, stable)."""
    n = len(score)
    o = np.argsort(-score, kind="stable")
    lab, match = lab[o], match[o]
    E = E[o] if E is not None else None
    corr = lab == LAB_CORRECT
    new = np.zeros(n, bool)
    if corr.any():
        ic = np.flatnonzero(corr)
        _, first = np.unique(match[ic], return_index=True)
        new[ic[first]] = True
    nrel = len(rel_rows)

    def ap(neg):
        cons = new | neg
        k = np.cumsum(cons)
        tp = np.cumsum(new)
        return float((tp[new] / k[new]).sum() / nrel) if nrel else np.nan

    r = dict(n_rel=nrel, n_rel_vis=int(rel_vis.sum()), n_rel_inv=int((~rel_vis).sum()), n_hits=n,
             ap=ap(lab == LAB_WRONG), ap_strict=ap((lab == LAB_WRONG) | (lab == LAB_UNLABELLED)),
             recall_any=float(new.sum() / nrel) if nrel else np.nan)
    if E is not None:
        for thr in thresholds:
            m = E <= thr
            found = np.isin(rel_rows, match[new & m])
            t = f"E{thr:g}"
            r[f"recall_{t}"] = float(found.mean()) if nrel else np.nan
            r[f"found_vis_{t}"] = int(found[rel_vis].sum())
            r[f"found_inv_{t}"] = int(found[~rel_vis].sum())
            r[f"recall_vis_{t}"] = float(found[rel_vis].mean()) if rel_vis.any() else np.nan
            r[f"recall_inv_{t}"] = float(found[~rel_vis].mean()) if (~rel_vis).any() else np.nan
            r[f"n_correct_{t}"] = int((new & m).sum())
            r[f"n_wrong_{t}"] = int(((lab == LAB_WRONG) & m).sum())
            r[f"n_unlabelled_{t}"] = int(((lab == LAB_UNLABELLED) & m).sum())
            r[f"n_sametype_{t}"] = int(((lab == LAB_SAMETYPE) & m).sum())
            r[f"n_dup_{t}"] = int((corr & ~new & m).sum())
            nc, nw = r[f"n_correct_{t}"], r[f"n_wrong_{t}"]
            r[f"precision_{t}"] = nc / (nc + nw) if nc + nw else np.nan
            nu_ = r[f"n_unlabelled_{t}"]
            r[f"precision_strict_{t}"] = nc / (nc + nw + nu_) if nc + nw + nu_ else np.nan
    return r


def boot_mean(values, groups, draws=2000, seed=17):
    """Mean with a 95% interval, resampling whole groups (genomes)."""
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(codes))
    n = np.bincount(inv, minlength=len(codes)).astype(float)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    m = s[pick].sum(1) / np.maximum(n[pick].sum(1), 1)
    return float(values.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
