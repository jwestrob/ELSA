#!/usr/bin/env python
"""Token-free recurrence (workstream softrec): the recurrence engine with per-gene ranked neighbour lists
in place of family tokens.

Instance  every pair (a, b) of genes within w positions on one contig (the engine's instances).
Lists     N(a) = ranked genes outside a's genus (or outside a's genome, `--skip genome`), either
          - knn : unique-sequence kNN expanded to genes (identical sequences first, tail rank = number of
                  identical copies; members of one unique sequence share the tail rank of their block),
                  capped at depth K and an optional cosine floor f  -> variants (K, f)
          - fam : the members of a's family token (all tied) = the degenerate list; the token engine is
                  this special case.
Statistic per instance and variant
          L_o  genera other than a's genus that contain a pair (a', b') within w on one contig,
               a' in N(a), b' in N(b), a' != b'
          G_o  genomes (other than a's genome) with such a pair
          E_o  sum over genomes G of min(1, 2w c_a(G) c_b(G) / n_G)  (c = list members in G)
          S    graded: sum over those genera of the best pair's min(s_a(a'), s_b(b')), s = log(N_db / tail rank)
          Engine convention: L = L_o + 1, G = G_o + 1 (the instance's own genome / genus count), E = E_o + own
          genome term, P = Poisson P(X >= G | E).
Null      gene order shuffled within each genome (contig sizes kept), seeds 1000 + r as the engine; every
          order-dependent step (instances and target co-location) runs on the shuffled order.
Rules     count L >= m; poisson P <= 1e-e & L >= m; graded S >= t. Instance-level FDR = mean null instances
          passing / observed instances passing. Per statistic, the engine's selection: among rules with
          FDR <= --fdr, the top 12 by observed count, the one covering most genes.
Loci      boundary support = largest L of a significant instance spanning the boundary; loci read with the
          engine's rules (conserved levels 2, 5, 12, 30, 80 at gaps 0 and 2; profile units, Jaccard 0.5 of the
          genome sets of the best instances). Written as an engine result directory.
"""
from __future__ import annotations

import argparse
import hashlib
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
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
HEAD = RD / "head/student/B_linear256_full/student_f32__uid.npy"
WORK = RD / "softrec"
RESULTS = RD / "engine/results_dpann_softrec"
LADDER = (2, 5, 12, 30, 80)
EXCL_FLOOR = 0.32      # random-pair cosine q99.9 (zca10): homolog-pair exclusion never fires below it

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_tokens import load_run, transform, stamp, rss_gb  # noqa: E402


def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ============================================================================== universe
class Universe:
    def __init__(self):
        g = pd.read_parquet(PKG / "universe/genes.parquet",
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
        assert ((self.pos - self.pos[self.c_start[self.cid]]) == (np.arange(n) - self.c_start[self.cid])).all()
        gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
        self.ng = len(gm)
        self.n_g = gm.n_genes.to_numpy(np.float64)
        assert np.array_equal(np.bincount(self.gc, minlength=self.ng), gm.n_genes.to_numpy())
        genus = gm.genus.where(gm.genus.notna() & (gm.genus != ""), "genome:" + gm.genome)
        self.lin_of_genome = pd.factorize(genus)[0].astype(np.int32)       # engine's lineage = genus
        self.nl = int(self.lin_of_genome.max()) + 1
        self.lin = self.lin_of_genome[self.gc]
        gchange = np.flatnonzero(np.diff(self.gc)) + 1
        self.g_start = np.r_[0, gchange].astype(np.int64)
        self.g_end = np.r_[gchange, n].astype(np.int64)
        self.genes_in_lin = np.bincount(self.lin, minlength=self.nl).astype(np.int64)
        # instances: (left slot i, d), d = 1..w, same contig
        self.gm = gm

    def instances(self, w):
        nd = np.minimum(w, self.c_end[self.cid] - 1 - np.arange(self.n)).astype(np.int32)
        nd = np.maximum(nd, 0)
        off = np.r_[0, np.cumsum(nd)[:-1]].astype(np.int64)
        n_inst = int(nd.sum())
        left = np.repeat(np.arange(self.n, dtype=np.int64), nd)
        d = (np.arange(n_inst) - off[left] + 1).astype(np.int32)
        return nd, off, left, d


# ============================================================================== list construction
@njit(parallel=True, cache=True)
def expand_knn(uq, lin, I, D, mptr, members, Kmax, floors, depths, out_nb, out_tail, llen, ntie, rf, cos_at):
    """Gene-level ranked lists outside the gene's genus from a unique-sequence kNN (rows sorted by cosine)."""
    n = len(uq)
    Ku = I.shape[1]
    F = len(floors)
    ND = len(depths)
    for a in prange(n):
        U = uq[a]
        llen[a] = 0
        ntie[a] = 0
        for f in range(F):
            rf[a, f] = 0
        for di in range(ND):
            cos_at[a, di] = 2.0
        if U < 0:
            continue
        la = lin[a]
        k = 0
        for t in range(mptr[U], mptr[U + 1]):
            b = members[t]
            if lin[b] != la and k < Kmax:
                out_nb[a, k] = b
                k += 1
        ntie[a] = k
        for t in range(k):
            out_tail[a, t] = k
        for f in range(F):
            rf[a, f] = k
        for di in range(ND):
            if depths[di] - 1 < k:
                cos_at[a, di] = 1.0
        full = k >= Kmax
        j = 0
        last = 1.0
        exhausted_row = False
        while (not full) and j < Ku:
            V = I[U, j]
            if V < 0:
                exhausted_row = True
                break
            s = np.float64(D[U, j])
            last = s
            if V != U:
                k0 = k
                for t in range(mptr[V], mptr[V + 1]):
                    b = members[t]
                    if lin[b] != la and k < Kmax:
                        out_nb[a, k] = b
                        k += 1
                for t in range(k0, k):
                    out_tail[a, t] = k
                for f in range(F):
                    if s >= floors[f]:
                        rf[a, f] = k
                for di in range(ND):
                    if k0 <= depths[di] - 1 and depths[di] - 1 < k:
                        cos_at[a, di] = s
                if k >= Kmax:
                    full = True
            j += 1
        llen[a] = k
        # cosine above which every gene outside the genus is in the list (up to index recall)
        if full:
            cstar = last
        elif exhausted_row:
            cstar = -1.0
        else:
            cstar = last          # all Ku unique neighbours consumed
        for di in range(ND):
            if depths[di] > k:
                cos_at[a, di] = cstar


@njit(parallel=True, cache=True)
def map_slots(nb, slot_of, out):
    for e in prange(len(nb)):
        x = nb[e]
        out[e] = slot_of[x] if x >= 0 else -1


# ============================================================================== kernels
@njit(parallel=True, cache=True)
def exclusion(n, V, gene_at, nd_arr, inst_off, mode, tok, X, uq, capcos, out):
    """Homologous instance pairs, excluded as the engine excludes ta == tb.
    mode 0 (fam): same token. mode 1 (knn): identical sequence, or cosine(a, b) >= max(EXCL_FLOOR,
    min(capcos[v, a], capcos[v, b])) (b would enter a's list of variant v, or a would enter b's)."""
    dim = X.shape[1]
    for i in prange(n):
        nd = nd_arr[i]
        if nd == 0:
            continue
        a = gene_at[i]
        for d in range(1, nd + 1):
            b = gene_at[i + d]
            inst = inst_off[i] + d - 1
            if mode == 0:
                ex = tok[a] >= 0 and tok[a] == tok[b]
                for v in range(V):
                    out[v, inst] = ex
            else:
                ua = uq[a]
                ub = uq[b]
                if ua < 0 or ub < 0:
                    for v in range(V):
                        out[v, inst] = False
                    continue
                if ua == ub:
                    for v in range(V):
                        out[v, inst] = True
                    continue
                c = 0.0
                for k in range(dim):
                    c += X[ua, k] * X[ub, k]
                for v in range(V):
                    t = capcos[v, a] if capcos[v, a] < capcos[v, b] else capcos[v, b]
                    if t < 0.32:
                        t = 0.32
                    out[v, inst] = c >= t


@njit(parallel=True, cache=True)
def soft_pass(n, w, V, gene_at, cid, lstart, lend, nbs, nbg, tail, tail_gene, rank_mode, skip, genome_gene,
              lin_of_genome, ndb_log, cap, capmax, excl, inst_off, nd_arr, n_genome, n_lin, ngenes_g,
              out_L, out_G, out_E, out_S, nchunks):
    """One pass over all instances in slot space (gene_at: slot -> gene; nbs: list entries as slots).
    rank_mode 0: rank = position in the gene's list, evidence uses tail[entry]; rank_mode 1: all entries tied,
    rank 0, evidence uses tail_gene[gene]. skip 0: none; 1: skip entries in the gene's genus; 2: in its genome."""
    for ch in prange(nchunks):
        lo = ch * n // nchunks
        hi = (ch + 1) * n // nchunks
        mstamp = np.full(n, -1, np.int32)
        mrank = np.zeros(n, np.int32)
        ca = np.zeros((V, n_genome), np.int32)
        cb = np.zeros((V, n_genome), np.int32)
        seen_a = np.full(n_genome, -1, np.int64)
        seen_b = np.full(n_genome, -1, np.int64)
        gst = np.full((V, n_genome), -1, np.int64)
        lst = np.full((V, n_lin), -1, np.int64)
        lbest = np.zeros((V, n_lin), np.float64)
        linlist = np.zeros((V, n_lin), np.int32)
        tch_a = np.zeros(n_genome, np.int32)
        tch_b = np.zeros(n_genome, np.int32)
        cntL = np.zeros(V, np.int32)
        cntG = np.zeros(V, np.int32)
        for i in range(lo, hi):
            nd = nd_arr[i]
            if nd == 0:
                continue
            a = gene_at[i]
            ga = genome_gene[a]
            la = lin_of_genome[ga]
            e0 = lstart[a]
            e1 = lend[a]
            if rank_mode == 0:
                if e1 > e0 + capmax[a]:
                    e1 = e0 + capmax[a]
            nta = 0
            for e in range(e0, e1):
                G = nbg[e]
                if skip == 1 and lin_of_genome[G] == la:
                    continue
                if skip == 2 and G == ga:
                    continue
                r = (e - e0) if rank_mode == 0 else 0
                s = nbs[e]
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                for u in range(ulo, uhi):
                    if u != s and cid[u] == c and mstamp[u] != i:
                        mstamp[u] = i
                        mrank[u] = r
                if seen_a[G] != i:
                    seen_a[G] = i
                    tch_a[nta] = G
                    nta += 1
                for v in range(V):
                    if r < cap[v, a]:
                        ca[v, G] += 1
            for d in range(1, nd + 1):
                b = gene_at[i + d]
                inst = inst_off[i] + d - 1
                any_ok = False
                for v in range(V):
                    cntL[v] = 0
                    cntG[v] = 0
                    if not excl[v, inst]:
                        any_ok = True
                if not any_ok:
                    for v in range(V):
                        out_L[v, inst] = -1
                        out_G[v, inst] = -1
                        out_E[v, inst] = 0.0
                        out_S[v, inst] = 0.0
                    continue
                f0 = lstart[b]
                f1 = lend[b]
                if rank_mode == 0:
                    if f1 > f0 + capmax[b]:
                        f1 = f0 + capmax[b]
                ntb = 0
                for e in range(f0, f1):
                    G = nbg[e]
                    if skip == 1 and lin_of_genome[G] == la:
                        continue
                    if skip == 2 and G == ga:
                        continue
                    r = (e - f0) if rank_mode == 0 else 0
                    if seen_b[G] != inst:
                        seen_b[G] = inst
                        tch_b[ntb] = G
                        ntb += 1
                    for v in range(V):
                        if r < cap[v, b]:
                            cb[v, G] += 1
                    s = nbs[e]
                    if mstamp[s] != i:
                        continue
                    ra = mrank[s]
                    lg = lin_of_genome[G]
                    for v in range(V):
                        if excl[v, inst]:
                            continue
                        if ra < cap[v, a] and r < cap[v, b]:
                            if gst[v, G] != inst:
                                gst[v, G] = inst
                                cntG[v] += 1
                            if lg != la:
                                if rank_mode == 0:
                                    ta_ = tail[e0 + ra]
                                    tb_ = tail[e]
                                else:
                                    ta_ = tail_gene[a]
                                    tb_ = tail_gene[b]
                                ev_a = ndb_log[a] - math.log(ta_)
                                ev_b = ndb_log[b] - math.log(tb_)
                                ev = ev_a if ev_a < ev_b else ev_b
                                if lst[v, lg] != inst:
                                    lst[v, lg] = inst
                                    lbest[v, lg] = ev
                                    linlist[v, cntL[v]] = lg
                                    cntL[v] += 1
                                elif ev > lbest[v, lg]:
                                    lbest[v, lg] = ev
                for v in range(V):
                    if excl[v, inst]:
                        out_L[v, inst] = -1
                        out_G[v, inst] = -1
                        out_E[v, inst] = 0.0
                        out_S[v, inst] = 0.0
                        continue
                    ssum = 0.0
                    for k in range(cntL[v]):
                        ssum += lbest[v, linlist[v, k]]
                    esum = 0.0
                    for k in range(ntb):
                        G = tch_b[k]
                        if ca[v, G] > 0 and cb[v, G] > 0:
                            x = 2.0 * w * ca[v, G] * cb[v, G] / ngenes_g[G]
                            esum += x if x < 1.0 else 1.0
                    out_L[v, inst] = cntL[v]
                    out_G[v, inst] = cntG[v]
                    out_E[v, inst] = esum
                    out_S[v, inst] = ssum
                for k in range(ntb):
                    G = tch_b[k]
                    for v in range(V):
                        cb[v, G] = 0
            for k in range(nta):
                G = tch_a[k]
                for v in range(V):
                    ca[v, G] = 0


@njit(parallel=True, cache=True)
def profile_bits(sel_left, sel_d, v, n, w, cid, lstart, lend, nbs, nbg, rank_mode, skip, genome_gene, lin_of_genome,
                 cap, capmax, n_words, out, nchunks):
    """Genome set (bitset) of selected observed instances under variant v: own genome plus every genome with a
    supporting pair (as counted in G)."""
    m = len(sel_left)
    one = np.uint64(1)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        mstamp = np.full(n, -1, np.int64)
        mrank = np.zeros(n, np.int32)
        for q in range(lo, hi):
            a = sel_left[q]
            b = a + sel_d[q]
            ga = genome_gene[a]
            la = lin_of_genome[ga]
            out[q, ga // 64] |= one << np.uint64(ga % 64)
            e0 = lstart[a]
            e1 = lend[a]
            if rank_mode == 0 and e1 > e0 + capmax[a]:
                e1 = e0 + capmax[a]
            for e in range(e0, e1):
                G = nbg[e]
                if skip == 1 and lin_of_genome[G] == la:
                    continue
                if skip == 2 and G == ga:
                    continue
                r = (e - e0) if rank_mode == 0 else 0
                s = nbs[e]
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                for u in range(ulo, uhi):
                    if u != s and cid[u] == c and mstamp[u] != q:
                        mstamp[u] = q
                        mrank[u] = r
            f0 = lstart[b]
            f1 = lend[b]
            if rank_mode == 0 and f1 > f0 + capmax[b]:
                f1 = f0 + capmax[b]
            for e in range(f0, f1):
                G = nbg[e]
                if skip == 1 and lin_of_genome[G] == la:
                    continue
                if skip == 2 and G == ga:
                    continue
                r = (e - f0) if rank_mode == 0 else 0
                s = nbs[e]
                if mstamp[s] == q and mrank[s] < cap[v, a] and r < cap[v, b]:
                    out[q, G // 64] |= one << np.uint64(G % 64)


@njit(parallel=True, cache=True)
def fold_bits(bits, lin_of_genome, n_lin_words, out):
    """Genome bitsets -> genus bitsets (exploratory unit reading on genus profiles)."""
    one = np.uint64(1)
    for q in prange(bits.shape[0]):
        for wd in range(bits.shape[1]):
            x = bits[q, wd]
            k = 0
            while x:
                if x & one:
                    G = wd * 64 + k
                    lg = lin_of_genome[G]
                    out[q, lg // 64] |= one << np.uint64(lg % 64)
                x >>= one
                k += 1


# ---------------------------------------------------------------------- engine copies (recurrence.py v1.5)
@njit(cache=True)
def boundary_support_argmax(left, dist, value, sig_pos, n):
    b = np.zeros(n, dtype=np.float32)
    arg = np.full(n, -1, dtype=np.int64)
    for t in range(len(left)):
        v = value[t]
        for k in range(left[t], left[t] + dist[t]):
            if v > b[k]:
                b[k] = v
                arg[k] = sig_pos[t]
    return b, arg


@njit(cache=True)
def runs_at_least(b, starts, ends, level, gap):
    out_s, out_e = [], []
    for c in range(len(starts)):
        s, e = starts[c], ends[c]
        k = s
        while k < e - 1:
            if b[k] >= level:
                k0 = k
                last = k
                k += 1
                while k < e - 1:
                    if b[k] >= level:
                        last = k
                        k += 1
                    elif k - last <= gap:
                        k += 1
                    else:
                        break
                out_s.append(k0)
                out_e.append(last + 1)
                k = last + 1
            else:
                k += 1
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@njit(cache=True)
def _jaccard(a, b):
    inter, union = 0, 0
    for w in range(len(a)):
        inter += _popcount(a[w] & b[w])
        union += _popcount(a[w] | b[w])
    return inter / union if union else 0.0


@njit(cache=True)
def profile_units(arg, bits, starts, ends, tau):
    out_s, out_e = [], []
    for c in range(len(starts)):
        s, e = starts[c], ends[c]
        k = s
        while k < e - 1:
            if arg[k] < 0:
                k += 1
                continue
            k0 = k
            while (k + 1 < e - 1 and arg[k + 1] >= 0
                   and _jaccard(bits[arg[k]], bits[arg[k + 1]]) >= tau):
                k += 1
            out_s.append(k0)
            out_e.append(k + 1)
            k += 1
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


# ============================================================================== lists
class Lists:
    """Ranked lists for every gene plus the variant caps."""

    def __init__(self, u, a):
        self.u = u
        self.source = a.source
        t0 = time.time()
        if a.source.startswith("fam"):
            self._fam(u, a)
        else:
            self._knn(u, a)
        stamp("lists", self.info, f"{time.time() - t0:.0f}s rss {rss_gb()}")

    def _fam(self, u, a):
        t = pd.read_parquet(a.tokens)
        tok = np.full(u.n, -1, np.int64)
        tok[t.gid.to_numpy()] = t.token.to_numpy()
        assert (tok >= 0).all()
        self.tok = tok
        order = np.argsort(tok, kind="stable")
        T = int(tok.max()) + 1
        fptr = np.searchsorted(tok[order], np.arange(T + 1)).astype(np.int64)
        self.nb = order.astype(np.int32)
        self.nbg = u.gc[order].astype(np.int16)
        self.lstart = fptr[tok]
        self.lend = fptr[tok + 1]
        self.rank_mode = 1
        self.skip = 1 if a.skip == "genus" else 2
        fam_size = np.diff(fptr)[tok]
        # family members outside the own genus / genome
        if self.skip == 1:
            key = tok * u.nl + u.lin
        else:
            key = tok * u.ng + u.gc
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        same = cnt[inv]
        self.tail_gene = np.maximum(fam_size - same, 1).astype(np.float64)
        self.own_copies = None
        if self.skip == 2:
            self.own_copies = same            # copies of the token in the gene's own genome (incl. itself)
        self.tail = np.ones(1, np.uint16)
        self.V = 1
        self.variants = ["fam"]
        self.cap = np.ones((1, u.n), np.int32)
        self.capmax = np.ones(u.n, np.int32)
        self.capcos = np.ones((1, u.n), np.float32)
        self.excl_mode = 0
        self.X = np.zeros((1, 1), np.float32)
        self.uq = np.zeros(u.n, np.int64)
        excl_n = (u.n - u.genes_in_lin[u.lin]) if self.skip == 1 else (u.n - u.n_g[u.gc])
        self.ndb_log = np.log(np.maximum(excl_n, 1)).astype(np.float64)
        self.info = dict(source=a.source, tokens=str(a.tokens), n_tokens=T, skip=a.skip,
                         list_len_q50_q90_q99_max=np.quantile(self.tail_gene, [0.5, 0.9, 0.99, 1]).tolist(),
                         entries_scanned_per_gene_q50_q99=np.quantile(fam_size, [0.5, 0.99]).tolist())

    def _knn(self, u, a):
        assert a.skip == "genus"
        uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
        assert len(uq) == u.n
        src = a.source
        if src == "zca10_k50":
            z = np.load(RD / "lab/knn/knn__esmpp_small__zca10__k50__hnsw.npz", allow_pickle=True)
            I, D = z["I"], z["D"].astype(np.float16)
            vec = "zca10"
        elif src == "zca10":
            I = np.load(WORK / f"knn/zca10__k{a.ku}__I.npy", mmap_mode="r")
            D = np.load(WORK / f"knn/zca10__k{a.ku}__D.npy", mmap_mode="r")
            vec = "zca10"
        elif src == "head":
            I = np.load(WORK / f"knn/head__k{a.ku}__I.npy", mmap_mode="r")
            D = np.load(WORK / f"knn/head__k{a.ku}__D.npy", mmap_mode="r")
            vec = "head"
        else:
            raise SystemExit(src)
        t0 = time.time()
        I = np.ascontiguousarray(I)
        D = np.ascontiguousarray(D).astype(np.float32)
        stamp("kNN loaded", I.shape, f"{time.time() - t0:.0f}s rss {rss_gb()}")
        U = I.shape[0]
        self.ku = I.shape[1]
        ok = uq >= 0
        order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
        mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=U))].astype(np.int64)
        depths = np.array(a.depths, np.int64)
        floors = np.array(a.floors, np.float64)
        Kmax = int(depths.max())
        n = u.n
        nb = np.full((n, Kmax), -1, np.int32)
        tail = np.zeros((n, Kmax), np.uint16)
        llen = np.zeros(n, np.int64)
        ntie = np.zeros(n, np.int64)
        rf = np.zeros((n, len(floors)), np.int64)
        cos_at = np.zeros((n, len(depths)), np.float64)
        t0 = time.time()
        expand_knn(uq, u.lin.astype(np.int64), I, D, mptr, order.astype(np.int64), Kmax, floors, depths, nb, tail,
                   llen, ntie, rf, cos_at)
        stamp("expand", f"{time.time() - t0:.0f}s")
        del I, D
        self.nb = nb.ravel()
        self.tail = tail.ravel()
        self.nbg = np.where(self.nb >= 0, u.gc[np.maximum(self.nb, 0)], 0).astype(np.int16)
        self.lstart = np.arange(n, dtype=np.int64) * Kmax
        self.lend = self.lstart + llen
        self.rank_mode = 0
        self.skip = 0
        self.tail_gene = np.ones(n, np.float64)
        self.own_copies = None
        self.variants, caps, ccos = [], [], []
        for di, K in enumerate(depths):
            for fi, f in enumerate(floors):
                self.variants.append(f"K{K}_f{'none' if f <= -1 else ('%g' % f).replace('.', 'p')}")
                caps.append(np.minimum(np.minimum(llen, K), rf[:, fi]))
                ccos.append(np.maximum(cos_at[:, di], f))
        self.V = len(self.variants)
        self.cap = np.stack(caps).astype(np.int32)
        self.capmax = self.cap.max(0).astype(np.int32)
        self.capcos = np.stack(ccos).astype(np.float32)
        self.excl_mode = 1
        if vec == "zca10":
            X, uu, valid, cnt = load_run(EMB)
            transform(X, valid, cnt, "zca10")
        else:
            X = np.load(HEAD)
        self.X = X
        self.uq = uq
        self.tok = np.zeros(1, np.int64)
        self.ndb_log = np.log(np.maximum(u.n - u.genes_in_lin[u.lin], 1)).astype(np.float64)
        has = llen > 0
        self.info = dict(source=src, ku=int(self.ku), depths=depths.tolist(), floors=floors.tolist(), variants=self.variants,
                         genes_with_list=int(has.sum()),
                         list_len_q01_q10_q50=np.quantile(llen, [0.01, 0.1, 0.5]).tolist(),
                         genes_list_shorter_than_depth={int(K): int((llen < K).sum()) for K in depths},
                         identical_copies_q50_q99=np.quantile(ntie, [0.5, 0.99]).tolist(),
                         cap_q10_q50_q90={v: np.quantile(self.cap[i], [0.1, 0.5, 0.9]).tolist() for i, v in enumerate(self.variants)},
                         capcos_q10_q50_q90={v: np.quantile(self.capcos[i], [0.1, 0.5, 0.9]).round(3).tolist() for i, v in enumerate(self.variants)})


# ============================================================================== passes
def run_pass(u, L, gene_at, slot_of, w, nd, off, n_inst, threads, label):
    t0 = time.time()
    nbs = np.empty_like(L.nb)
    map_slots(L.nb, slot_of.astype(np.int32), nbs)
    excl = np.zeros((L.V, n_inst), np.bool_)
    exclusion(u.n, L.V, gene_at, nd, off, L.excl_mode, L.tok, L.X, L.uq, L.capcos, excl)
    oL = np.zeros((L.V, n_inst), np.int16)
    oG = np.zeros((L.V, n_inst), np.int16)
    oE = np.zeros((L.V, n_inst), np.float32)
    oS = np.zeros((L.V, n_inst), np.float32)
    soft_pass(u.n, w, L.V, gene_at, u.cid, L.lstart, L.lend, nbs, L.nbg, L.tail, L.tail_gene, L.rank_mode, L.skip,
              u.gc, u.lin_of_genome, L.ndb_log, L.cap, L.capmax, excl, off, nd, u.ng, u.nl, u.n_g,
              oL, oG, oE, oS, threads * 8)
    del nbs
    # own-genome terms (engine convention)
    left = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    a_gene = gene_at[left]
    gi = u.gc[a_gene]
    if L.own_copies is not None:
        dd = np.arange(n_inst) - off[left] + 1
        b_gene = gene_at[left + dd]
        own = np.minimum(1.0, 2.0 * w * L.own_copies[a_gene] * L.own_copies[b_gene] / u.n_g[gi])
    else:
        own = np.minimum(1.0, 2.0 * w / u.n_g[gi])
    res = dict(L=oL, G=oG, E=oE, S=oS, own=own.astype(np.float32), excl=(oL < 0))
    stamp(f"pass {label}: {time.time() - t0:.0f}s; excluded per variant",
          {v: int(res['excl'][i].sum()) for i, v in enumerate(L.variants)}, f"rss {rss_gb()}")
    return res


def null_order(u, r):
    rng = np.random.default_rng(1000 + r)
    gene_at = np.arange(u.n, dtype=np.int64)
    for s, e in zip(u.g_start, u.g_end):
        gene_at[s:e] = s + rng.permutation(e - s)
    slot_of = np.empty_like(gene_at)
    slot_of[gene_at] = np.arange(u.n)
    return gene_at, slot_of


# ============================================================================== rules
def derived(p, v):
    L = p["L"][v].astype(np.int64)
    ok = L >= 0
    Lt = np.where(ok, L + 1, -1)
    G = p["G"][v].astype(np.int64) + 1
    E = p["E"][v].astype(np.float64) + p["own"]
    return ok, Lt, G, E, p["S"][v]


def logp_poisson(G, E, mask):
    from scipy.stats import poisson
    from scipy.special import gammaln
    out = np.zeros(len(G))
    g, e = G[mask].astype(np.float64), np.maximum(E[mask], 1e-12)
    pv = poisson.sf(g - 1, e)
    lp = np.log(np.maximum(pv, 1e-300))
    tiny = pv < 1e-290
    if tiny.any():
        lp[tiny] = g[tiny] * np.log(e[tiny]) - e[tiny] - gammaln(g[tiny] + 1) - np.log1p(-np.minimum(e[tiny] / (g[tiny] + 1), 0.999))
    out[mask] = lp
    return out


COUNT_M = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
POIS_E = (2, 3, 4, 6, 8, 10, 15, 20, 30)
GRADED_T = (8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 260, 330, 400, 500, 650, 800, 1000, 1300, 1600, 2000)


def rule_masks(Lt, logP, S, stat):
    if stat == "count":
        for m in COUNT_M:
            yield f"L>={m}", Lt >= m
    elif stat == "poisson":
        for e in POIS_E:
            for m in (2, 3):
                yield f"P<=1e-{e}&L>={m}", (Lt >= m) & (logP <= -e * math.log(10))
    elif stat == "graded":
        for t in GRADED_T:
            yield f"S>={t}", (Lt >= 2) & (S >= t)


def covered(u, mask, left, d):
    g = np.zeros(u.n, bool)
    g[left[mask]] = True
    g[left[mask] + d[mask]] = True
    return g


def select(u, obs, nulls, v, stat, left, d, fdr, force=None):
    ok, Lt, G, E, S = derived(obs, v)
    logP = logp_poisson(G, E, ok & (Lt >= 2)) if stat == "poisson" else None
    null_counts = []
    for p in nulls:
        okn, Ltn, Gn, En, Sn = derived(p, v)
        lPn = logp_poisson(Gn, En, okn & (Ltn >= 2)) if stat == "poisson" else None
        null_counts.append({name: int(mm.sum()) for name, mm in rule_masks(Ltn, lPn, Sn, stat)})
    rows, masks = [], {}
    for name, m in rule_masks(Lt, logP, S, stat):
        n_obs = int(m.sum())
        n_null = float(np.mean([c[name] for c in null_counts]))
        rows.append(dict(rule=name, observed=n_obs, null_mean=n_null, fdr=(n_null / n_obs if n_obs else np.nan)))
        masks[name] = m
    rules = pd.DataFrame(rows)
    if force:
        best = force
    else:
        cand = rules[(rules.fdr <= fdr) & (rules.observed > 0)].sort_values("observed", ascending=False).head(12)
        best, best_cov = None, -1
        for name in cand.rule:
            cov = int(covered(u, masks[name], left, d).sum())
            rules.loc[rules.rule == name, "genes_covered"] = cov
            if cov > best_cov:
                best, best_cov = name, cov
    return rules, best, (masks[best] if best else None), dict(Lt=Lt, G=G, E=E, S=S, logP=logP)


# ============================================================================== loci
def write_result(u, L, a, v, stat, rules, best, sig, st, left, d, w, extra):
    vname = L.variants[v]
    usuf = "" if a.unit_mode == "genome" and a.unit_tau == 0.5 else f"__u{a.unit_mode}{('%g' % a.unit_tau).replace('.', 'p')}"
    arm = f"dpann__softrec__{a.name}__{vname}__{stat}{usuf}"
    out = RESULTS / f"{arm}__w{w}"
    out.mkdir(parents=True, exist_ok=True)
    rules.to_csv(out / "rules.tsv", sep="\t", index=False)
    row = rules[rules.rule == best].iloc[0]
    # the engine's instance order (d, then left gene) so that ties in L pick the same best instance per boundary
    sidx = np.flatnonzero(sig)
    sidx = sidx[np.lexsort((left[sidx], d[sidx]))]
    li, di = left[sidx], d[sidx]
    value = st["Lt"][sidx].astype(np.float32)
    ids = np.arange(len(li), dtype=np.int64)
    b, arg = boundary_support_argmax(li, di, value, ids, u.n)
    np.save(out / "boundary_support.npy", b)
    # genome profiles of the best instance at each boundary, for the unit reading
    ua = np.unique(arg[arg >= 0])
    remap = np.full(len(li), -1, np.int64)
    remap[ua] = np.arange(len(ua))
    arg_c = np.where(arg >= 0, remap[np.maximum(arg, 0)], -1)
    n_words = (u.ng + 63) // 64
    bits = np.zeros((len(ua), n_words), np.uint64)
    nbs = L.nb  # observed order: slot == gene
    profile_bits(li[ua].astype(np.int64), di[ua].astype(np.int64), v, u.n, w, u.cid, L.lstart, L.lend, nbs, L.nbg,
                 L.rank_mode, L.skip, u.gc, u.lin_of_genome, L.cap, L.capmax, n_words, bits, a.threads * 8)
    pc = np.array([bin(int(x)).count("1") for x in bits[: min(2000, len(bits))].ravel()]).reshape(-1, n_words).sum(1)
    chk = st["G"][sidx][ua[: len(pc)]]
    bits_ok = bool(np.array_equal(pc, chk))
    cs, ce = u.c_start, u.c_end
    parts = []

    def frame(s_idx, e_idx, kind, level):
        return pd.DataFrame(dict(genome=u.genome_name[s_idx], contig=u.contig_name[s_idx], start=u.pos[s_idx],
                                 end=u.pos[e_idx], kind=kind, level=float(level), module_id=-1))
    for gap in (0, 2):
        for level in LADDER:
            s_idx, e_idx = runs_at_least(b, cs, ce, float(level), gap)
            parts.append(frame(s_idx, e_idx, "conserved" if gap == 0 else f"conserved_g{gap}", level))
    ubits = bits
    if a.unit_mode == "genus":
        ubits = np.zeros((len(bits), (u.nl + 63) // 64), np.uint64)
        fold_bits(bits, u.lin_of_genome, ubits.shape[1], ubits)
    s_idx, e_idx = profile_units(arg_c, ubits, cs, ce, float(a.unit_tau))
    parts.append(frame(s_idx, e_idx, "unit", 0.5))
    loci = pd.concat(parts, ignore_index=True)
    loci = loci[loci.end > loci.start].drop_duplicates(["genome", "contig", "start", "end", "kind"]).reset_index(drop=True)
    loci.to_parquet(out / "loci_all_genomes.parquet", index=False)
    genes = covered(u, sig, left, d)
    span = (loci.end - loci.start + 1)
    summ = dict(engine="softrec", arm=arm, w=w, source=a.name, variant=vname, statistic=stat, rule=best,
                fdr=float(row.fdr), significant_pairs=int(sig.sum()), significant_instances=int(sig.sum()),
                null_mean_passing=float(row.null_mean), genes=u.n,
                genes_covered=int(genes.sum()), genes_covered_frac=float(genes.mean()),
                L_quantiles=np.quantile(st["Lt"][sig], [.1, .25, .5, .75, .9, .99]).tolist() if sig.any() else [],
                profile_bits_match_G=bits_ok, arg_instances=int(len(ua)),
                loci_all_genomes={k: int(x) for k, x in loci.groupby("kind").size().items()},
                locus_span_median={k: float(x) for k, x in span.groupby(loci.kind).median().items()},
                modules=dict(modules=0, note="token-free: no modules"), lists=L.info, **extra)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    stamp("result", arm, {k: summ[k] for k in ("rule", "fdr", "significant_instances", "genes_covered", "profile_bits_match_G")},
          summ["loci_all_genomes"])
    return out


# ============================================================================== validation
def brute_check(u, L, gene_at, slot_of, w, res, off, nd, rng, m=1500):
    """Pure-python recount of L_o, G_o, E_o, S for random instances of a pass."""
    n_inst = res["L"].shape[1]
    left_all = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    pick = rng.choice(n_inst, m, replace=False)
    bad = 0
    maxdiff = 0.0
    for inst in pick:
        i = int(left_all[inst])
        dd = int(inst - off[i] + 1)
        a, b = int(gene_at[i]), int(gene_at[i + dd])
        ga = int(u.gc[a]); la = int(u.lin_of_genome[ga])

        def entries(g):
            e0, e1 = int(L.lstart[g]), int(L.lend[g])
            out = []
            for e in range(e0, e1):
                x = int(L.nb[e]); G = int(u.gc[x])
                if L.skip == 1 and u.lin_of_genome[G] == la:
                    continue
                if L.skip == 2 and G == ga:
                    continue
                r = (e - e0) if L.rank_mode == 0 else 0
                tl = float(L.tail[e]) if L.rank_mode == 0 else float(L.tail_gene[g])
                out.append((x, r, tl))
            return out
        Ea, Eb = entries(a), entries(b)
        for v in range(L.V):
            if res["excl"][v, inst]:
                continue
            ca = [(x, r, tl) for (x, r, tl) in Ea if r < L.cap[v, a]]
            cb = [(x, r, tl) for (x, r, tl) in Eb if r < L.cap[v, b]]
            # best evidence of a's list members at each slot (independent of the kernel's window marks)
            best_a = {}
            for (x, r, tl) in ca:
                sx = int(slot_of[x])
                ev = L.ndb_log[a] - math.log(tl)
                best_a[sx] = max(best_a.get(sx, -1e9), ev)
            gens, lins = set(), {}
            for (y, r2, tl2) in cb:
                sy = int(slot_of[y])
                eva = -1e9
                for t in range(-w, w + 1):
                    sx = sy + t
                    if t != 0 and 0 <= sx < u.n and sx in best_a and u.cid[sx] == u.cid[sy]:
                        eva = max(eva, best_a[sx])
                if eva > -1e9:
                    G = int(u.gc[y]); gens.add(G)
                    lg = int(u.lin_of_genome[G])
                    if lg != la:
                        ev = min(eva, L.ndb_log[b] - math.log(tl2))
                        lins[lg] = max(lins.get(lg, -1e9), ev)
            from collections import Counter
            na, nb_ = Counter(int(u.gc[x]) for (x, _, _) in ca), Counter(int(u.gc[y]) for (y, _, _) in cb)
            E = sum(min(1.0, 2.0 * w * na[G] * nb_[G] / u.n_g[G]) for G in na if G in nb_)
            got = (int(res["L"][v, inst]), int(res["G"][v, inst]), float(res["E"][v, inst]), float(res["S"][v, inst]))
            want = (len(lins), len(gens), E, sum(lins.values()))
            dE, dS = abs(got[2] - want[2]), abs(got[3] - want[3])
            maxdiff = max(maxdiff, dE, dS / max(1.0, want[3]))
            if got[0] != want[0] or got[1] != want[1] or dE > 1e-3 or dS > 1e-3 * max(1.0, want[3]):
                bad += 1
                if bad <= 5:
                    stamp("BRUTE MISMATCH", inst, L.variants[v], got, want)
    return dict(instances=int(m), mismatches=bad, max_rel_diff_E_S=maxdiff)


def engine_check(u, L, obs, nd, off, w, threads):
    """Degenerate family lists, skip genome: per-instance L, G, E, P against the engine's token-pair statistics."""
    sys.path.insert(0, str(RD / "engine/code"))
    import recurrence as R
    import numba
    eu = R.Universe(PKG / "universe")
    assert (eu.gid == np.arange(u.n)).all()
    tok = L.tok
    key, left, dist, T = R.colocalization(eu, tok, w)
    st = R.pair_statistics(eu, tok, key, left, w, T)
    idx = np.searchsorted(st["key"], key)
    inst = off[left] + dist - 1
    ok, Lt, G, E, S = derived(obs, 0)
    res = dict(engine_instances=int(len(key)), soft_instances_not_excluded=int(ok.sum()))
    res["excluded_match"] = bool(ok.sum() == len(key) and ok[inst].all())
    res["L_equal"] = int((Lt[inst] == st["L"][idx]).sum())
    res["G_equal"] = int((G[inst] == st["G"][idx]).sum())
    dE = np.abs(E[inst] - st["E"][idx])
    res["E_absdiff_max"] = float(dE.max())
    res["E_absdiff_q999"] = float(np.quantile(dE, 0.999))
    logP = logp_poisson(G, E, ok & (Lt >= 2))
    eng_sig = (st["P"][idx] <= 1e-4) & (st["L"][idx] >= 2)
    soft_sig = (logP[inst] <= -4 * math.log(10)) & (Lt[inst] >= 2)
    res["sig_engine_rule_P1e-4_L2"] = dict(engine=int(eng_sig.sum()), soft=int(soft_sig.sum()),
                                           both=int((eng_sig & soft_sig).sum()))
    stamp("ENGINE CHECK", res)
    return res


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="fam | zca10_k50 | zca10 | head")
    ap.add_argument("--name", required=True, help="result-directory tag, e.g. fam_e3c50s75_genus, zca10")
    ap.add_argument("--tokens", type=Path)
    ap.add_argument("--skip", choices=("genus", "genome"), default="genus")
    ap.add_argument("--ku", type=int, default=2000)
    ap.add_argument("--depths", default="50,200,1000")
    ap.add_argument("--floors", default="none,0.4")
    ap.add_argument("--stats", default="count,poisson,graded")
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--force-rule", default="", help="also write a result with this poisson rule (degenerate check)")
    ap.add_argument("--engine-check", action="store_true")
    ap.add_argument("--brute", type=int, default=600)
    ap.add_argument("--unit-mode", choices=("genome", "genus"), default="genome",
                    help="exploratory: profile units on genome sets (engine) or genus sets")
    ap.add_argument("--unit-tau", type=float, default=0.5, help="exploratory: Jaccard threshold of the unit reading")
    ap.add_argument("--only", default="", help="variant:stat pairs to write, comma-separated (default: all)")
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    a.depths = [int(x) for x in a.depths.split(",")]
    a.floors = [(-2.0 if x == "none" else float(x)) for x in a.floors.split(",")]
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    run_dir = WORK / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    man = dict(args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, code_sha256=code_sha(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    u = Universe()
    nd, off, left, d = u.instances(a.w)
    n_inst = len(left)
    stamp("universe", u.n, "genes", u.ng, "genomes", u.nl, "genera", n_inst, "instances")
    L = Lists(u, a)
    man["lists"] = L.info
    rng = np.random.default_rng(17)
    ident = np.arange(u.n, dtype=np.int64)
    obs = run_pass(u, L, ident, ident, a.w, nd, off, n_inst, a.threads, "observed")
    man["seconds_observed_pass"] = round(time.time() - t0, 1)
    man["brute_observed"] = brute_check(u, L, ident, ident, a.w, obs, off, nd, rng, a.brute)
    stamp("brute observed", man["brute_observed"])
    if a.engine_check:
        man["engine_check"] = engine_check(u, L, obs, nd, off, a.w, a.threads)
    nulls = []
    for r in range(a.perms):
        t1 = time.time()
        ga, so = null_order(u, r)
        nulls.append(run_pass(u, L, ga, so, a.w, nd, off, n_inst, a.threads, f"null {r}"))
        man[f"seconds_null_{r}"] = round(time.time() - t1, 1)
        if r == 0:
            man["brute_null0"] = brute_check(u, L, ga, so, a.w, nulls[-1], off, nd, rng, max(200, a.brute // 3))
            stamp("brute null", man["brute_null0"])
    # per-variant observed statistics for later analysis
    np.savez(run_dir / "observed_stats.npz", L=obs["L"], G=obs["G"], E=obs["E"], S=obs["S"], own=obs["own"],
             variants=np.array(L.variants))
    man["per_variant"] = {}
    for v, vname in enumerate(L.variants):
        okv, Lt, G, E, S = derived(obs, v)
        man["per_variant"][vname] = dict(
            excluded=int((~okv).sum()), L_ge2=int((Lt >= 2).sum()),
            null_L_ge2=float(np.mean([(derived(p, v)[1] >= 2).sum() for p in nulls])),
            E_q50_q90_q99=np.quantile(E[okv], [0.5, 0.9, 0.99]).round(4).tolist(),
            null_E_q50_q90_q99=np.quantile(derived(nulls[0], v)[3][derived(nulls[0], v)[0]], [0.5, 0.9, 0.99]).round(4).tolist())
        only = [x for x in a.only.split(",") if x]
        for stat in [s for s in a.stats.split(",") if s and s != "none"]:
            if only and f"{vname}:{stat}" not in only:
                continue
            rules, best, sig, st = select(u, obs, nulls, v, stat, left, d, a.fdr)
            man["per_variant"][vname][stat] = dict(rule=best, rules=rules.to_dict("records"))
            stamp(vname, stat, "rule", best)
            if best is None:
                continue
            write_result(u, L, a, v, stat, rules, best, sig, st, left, d, a.w, dict(seconds_total=round(time.time() - t0, 1)))
        if a.force_rule:
            rules, best, sig, st = select(u, obs, nulls, v, "poisson", left, d, a.fdr, force=a.force_rule)
            write_result(u, L, a, v, "forced", rules, best, sig, st, left, d, a.w, dict(forced=True))
    man["seconds"] = round(time.time() - t0, 1)
    man["peak_rss_gb"] = rss_gb()
    (run_dir / "run.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    stamp("end", man["seconds"], f"rss {rss_gb()}")


if __name__ == "__main__":
    main()
