#!/usr/bin/env python
"""long_loci core: bonds, L1 / L2 / L3 loci, persistence tree with approximate joint recurrence (L4, L4a, L4+).

Definitions: long_loci/predeclared.json.
  bonds   round 0 of the word learner (words_ts.run_words, copied): labelled gene pairs at distance 1-3 on one contig,
          n units, E (w = 3), Lambda; significant if n >= 2 and Lambda >= tau (syn: Lambda_eff, N_eff >= 2)
  s_k     largest Lambda of a bond spanning boundary k (between genes k and k+1)
  L1      runs of s_k > 0;  L2(theta) L1 cut where kappa < theta;  L3(theta) L2 UNION word loci (V-mob)
  L4      tie-collapsed min-rooted Cartesian tree of s'_k (= s_k, 0 where kappa < theta); node kept if persistence
          >= delta, n >= 2, Lambda(n, E) >= tau_node; n = other groups holding >= q of the node's distinct
          non-singleton families within <= 2|I| genes; E literal (3 shuffled corpora) or closed form (L4a)
  L4+     tree nodes with persistence >= delta, no recurrence test

usage
  ll.py build --data dpann|sd --count group [--tau T] --threads N      (Slurm)
  ll.py smoke --data dpann --genomes 150                                (login node, 1 thread)
Writes $RD/long_loci/loci/<data>_<count>/<output>.parquet (columns s, e = gene ids, inclusive), nodes/, results/.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import words_ts as W  # noqa: E402  (copied learner kernels)

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
LL = RD / "long_loci"
TS = RD / "transfer_spacedust"
THETAS = (0.02, 0.05, 0.1, 0.2)
DELTAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
TAUN = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
PHI, CWIN = 0.5, 2
NULLS = 3
SHIFT, MASK = W.SHIFT, W.MASK

DATA = {
    "dpann": dict(genes=RD / "ws_dpann/package/universe/genes.parquet",
                  tok=RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet",
                  groups=RD / "lineagefree/groups/emb_hom_average_t0.75.parquet",
                  kappa=RD / "context_units/ctx/kappa_var_tokens.npy",
                  S_obs=RD / "vocab/ctx2a/synS_obs.npy",
                  words={"group": ("loci", RD / "vocab/loci/main_tau10_V-mob.parquet"),
                         "syn": ("loci", RD / "vocab/loci/runs2a_neff_syn_tau8_V-mob.parquet")},
                  tau={"group": 10.0, "syn": 8.0}),
    "sd": dict(genes=RD / "universe/genes.parquet",
               tok=TS / "tok/mmseqs__seq__e3c50s75_rescue3__spacedust.parquet",
               groups=TS / "groups/emb_hom_average_t0.75.parquet",
               kappa=TS / "emb/kappa_var_tokens.npy",
               S_obs=TS / "results/synS_obs.npy",
               words={"group": ("run", LL / "runs/group/tau{tau}/obs"),
                      "syn": ("run", TS / "runs/syn/tau8/obs")},
               tau={"group": None, "syn": 8.0}),
}


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def jdump(obj, fn):
    Path(fn).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


# =========================================================================== data
def load(data, max_genomes=None):
    D = DATA[data]
    g = pd.read_parquet(D["genes"], columns=["gid", "genome_code", "contig_code"])
    assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
    cc = g.contig_code.to_numpy(np.int64)
    gc = g.genome_code.to_numpy(np.int64)
    assert np.all(np.diff(cc) >= 0) and np.all(np.diff(gc) >= 0)
    t = pd.read_parquet(D["tok"])
    tok = np.full(len(g), -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    assert (tok >= 0).all()
    ng = int(gc.max()) + 1
    grp = W.load_groups(D["groups"], ng)
    kappa = np.load(D["kappa"]).astype(np.float64)
    assert len(kappa) == len(tok)
    if max_genomes is not None:
        keep = gc < max_genomes
        tok, cc, gc, kappa = tok[keep], cc[keep], gc[keep], kappa[keep]
        grp = pd.factorize(grp[:max_genomes])[0].astype(np.int64)
    return tok, cc, gc, grp, kappa


# =========================================================================== bonds (learner round 0)
def bonds(tok, cc, gc, grp_in, mode, tau, S=None, w=3, min_n=2):
    """Round 0 of words_ts.run_words: returns the bonds (k, d, Lambda of their key) and key statistics."""
    n = len(tok)
    n_genomes = int(gc.max()) + 1
    grp = grp_in if mode == "group" else np.arange(n_genomes, dtype=np.int64)
    ng = int(grp.max()) + 1
    order = np.lexsort((np.arange(n_genomes), grp))
    rank_of_genome = np.empty(n_genomes, dtype=np.int64)
    rank_of_genome[order] = np.arange(n_genomes)
    grp_of_rank = grp[order].astype(np.int64)
    T = int(tok.max()) + 1
    tg = np.unique(tok * ng + grp[gc])
    tok_groups = np.bincount(tg // ng, minlength=T)
    sym = np.where(tok_groups[tok] >= min_n, tok, -1).astype(np.int64)
    k, d, key = W.occurrences(np.arange(n, dtype=np.int64), cc, sym, np.ones(n, dtype=np.bool_), w)
    ukey, kinv = np.unique(key, return_inverse=True)
    g_occ = grp[gc[k]]
    kg = np.unique(kinv * ng + g_occ)
    nG = np.bincount(kg // ng, minlength=len(ukey))
    cand = np.flatnonzero(nG >= min_n)
    lo_s, hi_s = ukey[cand] >> SHIFT, ukey[cand] & MASK
    live = sym >= 0
    sr = np.unique(sym[live] * n_genomes + rank_of_genome[gc[live]], return_counts=True)
    s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
    indptr = np.searchsorted(s_of, np.arange(T + 1)).astype(np.int64)
    m_rank = np.bincount(rank_of_genome[gc], minlength=n_genomes).astype(np.float64)
    E = W.expected_groups(lo_s, hi_s, indptr, r_of.astype(np.int64), sr[1].astype(np.float64), m_rank, grp_of_rank,
                          float(w))
    nc_ = nG[cand]
    lam = np.where(nc_ > E, nc_ * np.log(nc_ / np.maximum(E, 1e-300)) - nc_ + E, 0.0)
    if mode == "syn":
        pre = np.flatnonzero(lam >= tau)
        rank_k = np.full(len(ukey), -1, dtype=np.int64)
        rank_k[cand[pre]] = np.arange(len(pre))
        orr = rank_k[kinv]
        sel = np.flatnonzero(orr >= 0)
        o = np.lexsort((gc[k[sel]], orr[sel]))
        sel = sel[o]
        ptr = np.searchsorted(orr[sel], np.arange(len(pre) + 1)).astype(np.int64)
        nn = nc_[pre].astype(np.float64)
        neff = np.minimum(W.syn_neff_keys(ptr, gc[k[sel]], S), nn)
        fct = np.zeros(len(cand))
        fct[pre] = neff / nn
        lam = lam * fct
        pas = (lam >= tau) & (nc_ * fct >= 2)
    else:
        pas = lam >= tau
    key_lam = np.zeros(len(ukey))
    key_lam[cand[pas]] = lam[pas]
    occ_lam = key_lam[kinv]
    si = occ_lam > 0
    info = dict(occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)), significant_keys=int(pas.sum()),
                bonds=int(si.sum()), bond_lambda_q10_q50_q90_q99=np.quantile(occ_lam[si], [.1, .5, .9, .99]).tolist()
                if si.any() else [])
    return k[si], d[si], occ_lam[si], info


@njit(cache=True)
def boundary_score(k, d, lam, n):
    s = np.zeros(n)
    for i in range(len(k)):
        for b in range(k[i], k[i] + d[i]):
            if lam[i] > s[b]:
                s[b] = lam[i]
    return s


def runs(mask):
    """Maximal runs of True boundaries -> gene intervals (st, en): boundaries st..en-1 -> genes st..en."""
    x = np.r_[False, mask, False].astype(np.int8)
    st = np.flatnonzero(np.diff(x) == 1)
    en = np.flatnonzero(np.diff(x) == -1)
    return st.astype(np.int64), en.astype(np.int64)


# =========================================================================== persistence tree
@njit(cache=True)
def _find(uf, x):
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


@njit(cache=True)
def build_tree(s, order):
    """Tie-collapsed min-rooted Cartesian tree over every maximal run of positive boundaries.
    order = positive boundaries by decreasing s. Returns lo, hi, m, parent (compact ids, -1 = run root)."""
    n = len(s)
    nb = len(order)
    uf = np.arange(n)
    comp_node = np.full(n, -1, dtype=np.int64)
    clo = np.arange(n)
    chi = np.arange(n)
    nlo = np.empty(nb, dtype=np.int64)
    nhi = np.empty(nb, dtype=np.int64)
    nm = np.empty(nb, dtype=np.float64)
    npar = np.full(nb, -1, dtype=np.int64)
    alias = np.arange(nb)
    nn = 0
    for t in range(nb):
        b = order[t]
        v = s[b]
        ra = _find(uf, b)
        rb = _find(uf, b + 1)
        na = comp_node[ra]
        nbn = comp_node[rb]
        a_same = na >= 0 and nm[na] == v
        b_same = nbn >= 0 and nm[nbn] == v
        if a_same and b_same:
            alias[nbn] = na
            node = na
        elif a_same:
            node = na
            if nbn >= 0:
                npar[nbn] = na
        elif b_same:
            node = nbn
            if na >= 0:
                npar[na] = nbn
        else:
            node = nn
            nn += 1
            nm[node] = v
            if na >= 0:
                npar[na] = node
            if nbn >= 0:
                npar[nbn] = node
        uf[rb] = ra
        if clo[rb] < clo[ra]:
            clo[ra] = clo[rb]
        if chi[rb] > chi[ra]:
            chi[ra] = chi[rb]
        comp_node[ra] = node
        nlo[node] = clo[ra]
        nhi[node] = chi[ra]
    # resolve aliases, compact
    newid = np.full(nn, -1, dtype=np.int64)
    m = 0
    for i in range(nn):
        if alias[i] == i:
            newid[i] = m
            m += 1
    lo = np.empty(m, dtype=np.int64)
    hi = np.empty(m, dtype=np.int64)
    mm = np.empty(m, dtype=np.float64)
    par = np.full(m, -1, dtype=np.int64)
    for i in range(nn):
        if alias[i] != i:
            continue
        j = newid[i]
        lo[j] = nlo[i]
        hi[j] = nhi[i]
        mm[j] = nm[i]
        p = npar[i]
        if p >= 0:
            while alias[p] != p:
                p = alias[p]
            par[j] = newid[p]
    return lo, hi, mm, par


def tree(s):
    idx = np.flatnonzero(s > 0)
    order = idx[np.argsort(-s[idx], kind="mergesort")].astype(np.int64)
    lo, hi, m, par = build_tree(s, order)
    e = np.where(par >= 0, m[np.maximum(par, 0)], 0.0)
    return pd.DataFrame(dict(lo=lo, hi=hi, m=m, e=e, pers=m - e, par=par))


# =========================================================================== node keys
@njit(cache=True)
def _mix(h, x):
    h ^= np.uint64(x) + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2))
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(31)
    return h


@njit(parallel=True, cache=True)
def node_keys(lo, hi, tokc, fsize):
    """Per node: nI (non-singleton genes), kF (distinct non-singleton families), 64-bit hash of (nI, sorted families)."""
    nn = len(lo)
    nI = np.zeros(nn, dtype=np.int64)
    kF = np.zeros(nn, dtype=np.int64)
    hsh = np.zeros(nn, dtype=np.uint64)
    for i in prange(nn):
        L = hi[i] - lo[i] + 1
        buf = np.empty(L, dtype=np.int64)
        c = 0
        for g in range(lo[i], hi[i] + 1):
            f = tokc[g]
            if fsize[f] >= 2:
                buf[c] = f
                c += 1
        nI[i] = c
        b = np.sort(buf[:c])
        h = _mix(np.uint64(1469598103934665603), c)
        u = 0
        for j in range(c):
            if j == 0 or b[j] != b[j - 1]:
                h = _mix(h, b[j])
                u += 1
        kF[i] = u
        hsh[i] = h
    return nI, kF, hsh


@njit(cache=True)
def key_families(rep_lo, rep_hi, tokc, fsize, kF):
    kptr = np.zeros(len(rep_lo) + 1, dtype=np.int64)
    for u in range(len(rep_lo)):
        kptr[u + 1] = kptr[u] + kF[u]
    kfam = np.empty(kptr[-1], dtype=np.int64)
    for u in range(len(rep_lo)):
        L = rep_hi[u] - rep_lo[u] + 1
        buf = np.empty(L, dtype=np.int64)
        c = 0
        for g in range(rep_lo[u], rep_hi[u] + 1):
            f = tokc[g]
            if fsize[f] >= 2:
                buf[c] = f
                c += 1
        b = np.sort(buf[:c])
        p = kptr[u]
        for j in range(c):
            if j == 0 or b[j] != b[j - 1]:
                kfam[p] = b[j]
                p += 1
    return kptr, kfam


def make_keys(lo, hi, own, tokc, fsize):
    """Unique keys over the given intervals of one corpus; instance CSR (own group per instance)."""
    nI, kF, hsh = node_keys(lo, hi, tokc, fsize)
    uh, first, inv = np.unique(hsh, return_index=True, return_inverse=True)
    kq = np.maximum(2, np.ceil(PHI * kF[first]).astype(np.int64))
    kW = (CWIN * nI[first]).astype(np.int64)
    kptr, kfam = key_families(lo[first], hi[first], tokc, fsize, kF[first])
    o = np.argsort(inv, kind="stable")
    iptr = np.searchsorted(inv[o], np.arange(len(uh) + 1)).astype(np.int64)
    return dict(nI=nI, kF=kF, inv=inv, order=o, kptr=kptr, kfam=kfam, kq=kq, kW=kW, iptr=iptr,
                iown=own[o].astype(np.int64), valid=(kF[first] >= 2) & (nI[first] >= 2))


def fam_csr(tokc, T):
    o = np.argsort(tokc, kind="stable").astype(np.int64)
    ptr = np.searchsorted(tokc[o], np.arange(T + 1)).astype(np.int64)
    return ptr, o


# =========================================================================== recurrence count
@njit(parallel=True, cache=True)
def count_keys(kptr, kfam, kq, kW, valid, iptr, iown, fptr, fgen, cc, gc, grp, n_groups, mg, do_ana, nblocks):
    """Per key: n = groups with a genome holding >= q distinct key families within <= W genes on one contig.
    Per instance: n minus its own group (if hit); closed-form E over other groups (do_ana)."""
    nk = len(kq)
    ni = len(iown)
    n_out = np.zeros(ni, dtype=np.int64)
    e_out = np.zeros(ni, dtype=np.float64)
    bs = (nk + nblocks - 1) // nblocks
    for blk in prange(nblocks):
        u0 = blk * bs
        u1 = min(nk, u0 + bs)
        if u0 >= u1:
            continue
        stamp = np.full(n_groups, -1, dtype=np.int64)
        acc = np.zeros(n_groups, dtype=np.float64)
        touched = np.empty(n_groups, dtype=np.int64)
        for u in range(u0, u1):
            if not valid[u]:
                continue
            a = kptr[u]
            b = kptr[u + 1]
            kf = b - a
            q = kq[u]
            Wn = kW[u]
            tot = 0
            for t in range(a, b):
                f = kfam[t]
                tot += fptr[f + 1] - fptr[f]
            buf = np.empty(tot, dtype=np.int64)
            p = 0
            for t in range(a, b):
                f = kfam[t]
                for x in range(fptr[f], fptr[f + 1]):
                    buf[p] = fgen[x] * kf + (t - a)
                    p += 1
            buf.sort()
            cnt = np.zeros(kf, dtype=np.int64)
            nh = 0
            nt = 0
            j0 = 0
            while j0 < tot:
                g0 = gc[buf[j0] // kf]
                j1 = j0 + 1
                while j1 < tot and gc[buf[j1] // kf] == g0:
                    j1 += 1
                G = grp[g0]
                if do_ana:
                    # closed form: counts per family in this genome
                    pres = 0
                    for x in range(j0, j1):
                        gx = buf[x] // kf
                        lx = buf[x] - gx * kf
                        if cnt[lx] == 0:
                            pres += 1
                        cnt[lx] += 1
                    if pres >= q:
                        e = np.zeros(q + 1)
                        e[0] = 1.0
                        lsc = 0.0
                        cur = 0
                        for x in range(j0, j1):
                            gx = buf[x] // kf
                            lx = buf[x] - gx * kf
                            c = cnt[lx]
                            if c > 0:
                                top = cur + 1 if cur + 1 < q else q
                                for tt in range(top, 0, -1):
                                    e[tt] += e[tt - 1] * c
                                cur += 1
                                cnt[lx] = 0
                                mx = 0.0
                                for tt in range(q + 1):
                                    if e[tt] > mx:
                                        mx = e[tt]
                                if mx > 1e250:
                                    for tt in range(q + 1):
                                        e[tt] *= 1e-250
                                    lsc += 250.0 * math.log(10.0)
                        if e[q] > 0:
                            m = mg[g0]
                            lmu = (math.log(m) + math.lgamma(Wn) - math.lgamma(q) - math.lgamma(Wn - q + 1)
                                   + math.lgamma(q + 1) + math.log(e[q]) + lsc - q * math.log(m))
                            mu = math.exp(min(lmu, 50.0))
                            if acc[G] == 0.0:
                                touched[nt] = G
                                nt += 1
                            acc[G] += mu
                    else:
                        for x in range(j0, j1):
                            gx = buf[x] // kf
                            cnt[buf[x] - gx * kf] = 0
                if stamp[G] != u and j1 - j0 >= q:
                    i = j0
                    dist = 0
                    hit = False
                    jl = j1 - 1
                    for j in range(j0, j1):
                        gj = buf[j] // kf
                        lj = buf[j] - gj * kf
                        while i < j:
                            gi = buf[i] // kf
                            if cc[gi] != cc[gj] or gj - gi >= Wn:
                                li = buf[i] - gi * kf
                                cnt[li] -= 1
                                if cnt[li] == 0:
                                    dist -= 1
                                i += 1
                            else:
                                break
                        cnt[lj] += 1
                        if cnt[lj] == 1:
                            dist += 1
                        if dist >= q:
                            hit = True
                            jl = j
                            break
                    for x in range(i, jl + 1):
                        gx = buf[x] // kf
                        cnt[buf[x] - gx * kf] = 0
                    if hit:
                        stamp[G] = u
                        nh += 1
                j0 = j1
            Eall = 0.0
            for t in range(nt):
                Eall += 1.0 - math.exp(-acc[touched[t]])
            for r in range(iptr[u], iptr[u + 1]):
                own = iown[r]
                n_out[r] = nh - (1 if stamp[own] == u else 0)
                if do_ana:
                    e_out[r] = Eall - (1.0 - math.exp(-acc[own]))
            for t in range(nt):
                acc[touched[t]] = 0.0
    return n_out, e_out


def recurrence(lo, hi, own, tokc, fsize, corpora, own_corpus, cc, gc, grp, mg, nblocks):
    """n in the node's own corpus, closed-form E, and the count in every other corpus (literal shuffled E).
    corpora: dict r -> (fptr, fgen); own_corpus = key of the corpus the nodes live in."""
    K = make_keys(lo, hi, own, tokc, fsize)
    ngr = int(grp.max()) + 1
    args = (K["kptr"], K["kfam"], K["kq"], K["kW"], K["valid"], K["iptr"], K["iown"])
    out = {}
    fptr, fgen = corpora[own_corpus]
    n_o, e_o = count_keys(*args, fptr, fgen, cc, gc, grp, ngr, mg, True, nblocks)
    inst = K["order"]
    res = pd.DataFrame(dict(nI=K["nI"], kF=K["kF"]))
    res["q"] = np.maximum(2, np.ceil(PHI * res.kF.to_numpy()).astype(np.int64))
    res["W"] = CWIN * res.nI.to_numpy()
    nn = np.zeros(len(lo), dtype=np.int64)
    ee = np.zeros(len(lo))
    nn[inst] = n_o
    ee[inst] = e_o
    res["n"] = nn
    res["E_ana"] = np.maximum(ee, 0.0)
    shuf = []
    for r, (fp, fg) in corpora.items():
        if r == own_corpus:
            continue
        n_r, _ = count_keys(*args, fp, fg, cc, gc, grp, ngr, mg, False, nblocks)
        x = np.zeros(len(lo), dtype=np.int64)
        x[inst] = n_r
        res[f"n_shuf{r}"] = x
        shuf.append(x)
    res["E_shuf"] = np.mean(shuf, axis=0) if shuf else np.nan
    res.loc[(res.kF < 2) | (res.nI < 2), ["n", "E_ana", "E_shuf"]] = [0, 0.0, 0.0]
    out["keys"] = int(len(K["kq"]))
    return res, out


def lam_of(n, E):
    n = n.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(n > E, n * np.log(n / E) - n + E, 0.0)
    L = np.where((E <= 0) & (n > 0), np.inf, L)
    return L


def kept(df, delta, taun, which):
    lam = df[f"lam_{which}"].to_numpy()
    return (df.pers.to_numpy() >= delta) & (df.n.to_numpy() >= 2) & (lam >= taun) & (df.nI.to_numpy() >= 2)


def calibrate(obs, nulls, which):
    """FDR = mean null kept / observed kept over the (delta, tau_node) ladder; choose the pair with the most observed
    kept nodes among FDR <= 0.01 (ties: larger delta, then larger tau_node)."""
    rows = []
    for dl in DELTAS:
        for tn in TAUN:
            o = int(kept(obs, dl, tn, which).sum())
            nl = [int(kept(x, dl, tn, which).sum()) for x in nulls]
            fdr = float(np.mean(nl)) / max(o, 1)
            rows.append(dict(delta=dl, tau_node=tn, observed=o, null=nl, fdr=fdr))
    ok = [r for r in rows if r["fdr"] <= 0.01 and r["observed"] > 0]
    best = max(ok, key=lambda r: (r["observed"], r["delta"], r["tau_node"])) if ok else None
    return rows, best


# =========================================================================== words (L3)
def word_loci(data, count, tau, gc, n):
    kind, path = DATA[data]["words"][count]
    if kind == "loci":
        df = pd.read_parquet(path, columns=["gid_s", "gid_e"])
        return df.gid_s.to_numpy(np.int64), df.gid_e.to_numpy(np.int64), str(path)
    import wscore_ts as WS
    d = Path(str(path).format(tau=f"{tau:g}"))
    run = WS.Run(d)
    ng = int(gc.max()) + 1
    if count == "group":
        unit = W.load_groups(DATA[data]["groups"], ng)
    else:
        unit = np.arange(ng, dtype=np.int64)
    I = run.inst
    lab = I["label"]
    S = int(max(lab.max() if len(lab) else 0, run.top_sym.max() if len(run.top_sym) else 0, run.T)) + 1
    nu = int(unit.max()) + 1
    tg = np.unique(run.top_sym * nu + unit[gc[run.top_s]])
    mobile = np.bincount(tg // nu, minlength=S) >= 2
    mc = WS.mobile_cores(run.top_inst.astype(np.int64), I["a"], I["b"], lab, mobile)
    s = np.r_[run.top_s, I["s"][mc]].astype(np.int64)
    e = np.r_[run.top_e, I["e"][mc]].astype(np.int64)
    return s, e, str(d)


# =========================================================================== build
def save_loci(out_dir, name, s, e):
    df = pd.DataFrame(dict(s=np.asarray(s, np.int64), e=np.asarray(e, np.int64))).drop_duplicates()
    df = df[df.e > df.s].sort_values(["s", "e"]).reset_index(drop=True)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return df


def loci_info(df, n):
    s, e = df.s.to_numpy(), df.e.to_numpy()
    cov = np.zeros(n + 1, np.int32)
    np.add.at(cov, s, 1)
    np.add.at(cov, e + 1, -1)
    sp = e - s + 1
    return dict(loci=int(len(s)), genes_in_loci_frac=float((np.cumsum(cov)[:n] > 0).mean()),
                span_q50_q90_q99_max=np.quantile(sp, [.5, .9, .99, 1]).tolist() if len(sp) else [])


def corpus_tokens(tok, gc, r):
    return tok if r < 0 else W.shuffled(tok, gc, r)


def cmd_build(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    tok, cc, gc, grp, kappa = load(a.data, a.genomes)
    n = len(tok)
    ng = int(gc.max()) + 1
    tau = a.tau if a.tau is not None else DATA[a.data]["tau"][a.count]
    assert tau is not None
    tag = f"{a.data}_{a.count}{a.suffix}"
    od = LL / "loci" / tag
    nd = LL / "nodes" / tag
    od.mkdir(parents=True, exist_ok=True)
    nd.mkdir(parents=True, exist_ok=True)
    T = int(tok.max()) + 1
    fsize = np.bincount(tok, minlength=T)
    mg = np.bincount(gc, minlength=ng).astype(np.float64)
    res = dict(data=a.data, count=a.count, tau=tau, genes=n, genomes=ng, groups=int(grp.max()) + 1,
               singleton_family_genes=int((fsize[tok] == 1).sum()), outputs={}, timing={})
    log(tag, "genes", n, "genomes", ng, "groups", res["groups"], "tau", tau)
    # corpora
    toks = {r: corpus_tokens(tok, gc, r) for r in [-1] + list(range(NULLS))}
    S = {}
    if a.count == "syn":
        for r in toks:
            S[r] = W.synteny_matrix(toks[r], cc, gc)
        if a.genomes is None:
            st = np.load(DATA[a.data]["S_obs"])
            res["S_obs_equal_stored"] = bool(np.allclose(st, S[-1], atol=1e-6))
            log("S obs equal stored:", res["S_obs_equal_stored"])
    corpora = {r: fam_csr(toks[r], T) for r in toks}
    res["timing"]["prep"] = round(time.time() - t0, 1)
    # bonds and boundary scores per corpus
    sc = {}
    for r in toks:
        tb = time.time()
        k, d, lam, info = bonds(toks[r], cc, gc, grp, a.count, tau, S.get(r))
        sc[r] = boundary_score(k.astype(np.int64), d.astype(np.int64), lam.astype(np.float64), n)
        info["boundaries_positive"] = int((sc[r] > 0).sum())
        info["seconds"] = round(time.time() - tb, 1)
        res[f"bonds_{'obs' if r < 0 else f'null{r}'}"] = info
        log("bonds", r, info)
        if r < 0:
            np.savez(nd / "bonds_obs.npz", k=k, d=d, lam=lam)
    s = sc[-1]
    # L1
    st, en = runs(s > 0)
    L1 = save_loci(od, "L1", st, en)
    res["outputs"]["L1"] = loci_info(L1, n)
    if a.data == "sd" and a.count == "syn" and a.genomes is None:
        sig0 = dict(np.load(TS / "runs/syn/tau8/obs/sig0.npz"))
        m = sig0["d"] <= 3
        cov = np.zeros(n + 1, dtype=np.int32)
        np.add.at(cov, sig0["k"][m], 1)
        np.add.at(cov, sig0["k"][m] + sig0["d"][m], -1)
        b0 = np.cumsum(cov)[:n] > 0
        res["L1_equals_stored_CC2_boundaries"] = bool(np.array_equal(b0, s > 0))
        log("L1 == stored CC2 boundaries:", res["L1_equals_stored_CC2_boundaries"])
    # words
    ws, we, wsrc = word_loci(a.data, a.count, tau, gc, n)
    res["words_source"] = wsrc
    # L2, L3, trees per theta
    trees = {}
    for th in THETAS:
        cut = kappa < th
        st, en = runs((s > 0) & ~cut)
        L2 = save_loci(od, f"L2_t{th:g}", st, en)
        L3 = save_loci(od, f"L3_t{th:g}", np.r_[L2.s.to_numpy(), ws], np.r_[L2.e.to_numpy(), we])
        res["outputs"][f"L2_t{th:g}"] = loci_info(L2, n)
        res["outputs"][f"L3_t{th:g}"] = loci_info(L3, n)
        s2 = np.where(cut, 0.0, s)
        trees[th] = tree(s2)
        log("theta", th, "L2", res["outputs"][f"L2_t{th:g}"], "tree nodes", len(trees[th]))
    # null trees (no cohesion cut)
    ntrees = {r: tree(sc[r]) for r in range(NULLS)}
    # recurrence for the union of observed tree nodes
    tr = time.time()
    allnodes = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
    own = grp[gc[allnodes.lo.to_numpy()]]
    rec, kinfo = recurrence(allnodes.lo.to_numpy(), allnodes.hi.to_numpy(), own, toks[-1], fsize, corpora, -1, cc, gc,
                            grp, mg, a.blocks)
    rec = pd.concat([allnodes, rec], axis=1)
    res["timing"]["recurrence_obs"] = round(time.time() - tr, 1)
    res["recurrence_obs"] = dict(unique_intervals=int(len(allnodes)), **kinfo)
    log("recurrence obs", res["recurrence_obs"], res["timing"]["recurrence_obs"], "s")
    # null nodes
    nrec = {}
    for r in range(NULLS):
        tr = time.time()
        t = ntrees[r]
        own = grp[gc[t.lo.to_numpy()]]
        cor = {x: corpora[x] for x in range(NULLS)}
        rr, ki = recurrence(t.lo.to_numpy(), t.hi.to_numpy(), own, toks[r], fsize, cor, r, cc, gc, grp, mg, a.blocks)
        nrec[r] = pd.concat([t.reset_index(drop=True), rr], axis=1)
        res[f"null{r}_nodes"] = dict(nodes=int(len(t)), keys=ki["keys"], seconds=round(time.time() - tr, 1))
        log("null", r, res[f"null{r}_nodes"])
    for r in nrec:
        x = nrec[r]
        x["lam_shuf"] = lam_of(x.n.to_numpy(), x.E_shuf.to_numpy())
        x["lam_ana"] = lam_of(x.n.to_numpy(), x.E_ana.to_numpy())
        x.to_parquet(nd / f"null{r}.parquet", index=False)
    res["calibration"] = {}
    for th in THETAS:
        t = trees[th].merge(rec, on=["lo", "hi"], how="left")
        assert len(t) == len(trees[th])
        t["lam_shuf"] = lam_of(t.n.to_numpy(), t.E_shuf.to_numpy())
        t["lam_ana"] = lam_of(t.n.to_numpy(), t.E_ana.to_numpy())
        t.to_parquet(nd / f"obs_t{th:g}.parquet", index=False)
        cal = {}
        for which, nm in (("shuf", "L4"), ("ana", "L4a")):
            rows, best = calibrate(t, [nrec[r] for r in range(NULLS)], which)
            cal[nm] = dict(best=best, ladder=rows)
            if best is None:
                log("theta", th, nm, "no ladder pair meets FDR <= 0.01")
                continue
            km = kept(t, best["delta"], best["tau_node"], which)
            df = save_loci(od, f"{nm}_t{th:g}", t.lo.to_numpy()[km], t.hi.to_numpy()[km])
            res["outputs"][f"{nm}_t{th:g}"] = loci_info(df, n)
            pm = (t.pers.to_numpy() >= best["delta"]) & (t.hi.to_numpy() > t.lo.to_numpy())
            dfp = save_loci(od, f"{nm}p_t{th:g}", t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
            res["outputs"][f"{nm}p_t{th:g}"] = loci_info(dfp, n)
            log("theta", th, nm, "best", best, "kept", res["outputs"][f"{nm}_t{th:g}"], "plus",
                res["outputs"][f"{nm}p_t{th:g}"])
        sz = t.hi - t.lo + 1
        cal["nodes"] = dict(nodes=int(len(t)), roots=int((t.par < 0).sum()), size_q50_q90_q99_max=np.quantile(
            sz, [.5, .9, .99, 1]).tolist(), pers_q10_q50_q90=np.quantile(t.pers, [.1, .5, .9]).tolist(),
            n_ge2=int((t.n >= 2).sum()), E_shuf_zero_frac=float((t.E_shuf == 0).mean()),
            E_ana_q50_q90=np.quantile(t.E_ana, [.5, .9]).tolist(),
            shuf_vs_ana_sum=[float(t.E_shuf.sum()), float(t.E_ana.sum())])
        res["calibration"][f"{th:g}"] = cal
    res["timing"]["total"] = round(time.time() - t0, 1)
    jdump(res, LL / "results" / f"build_{tag}.json")


def cmd_build4b(a):
    """L4b (amendment_2): tree over L1 runs with low-cohesion junctions as the first splits (w_k = 1e-6 + max(kappa, 0)
    where kappa < theta), node kept if n >= 2 and Lambda(n, E_closed_form) >= the bond tau."""
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    tok, cc, gc, grp, kappa = load(a.data, a.genomes)
    n = len(tok)
    ng = int(gc.max()) + 1
    tag = f"{a.data}_{a.count}{a.suffix}"
    bj = json.load(open(LL / "results" / f"build_{tag}.json"))
    tau = float(bj["tau"])
    od = LL / "loci" / tag
    nd = LL / "nodes" / tag
    z = np.load(nd / "bonds_obs.npz")
    s = boundary_score(z["k"].astype(np.int64), z["d"].astype(np.int64), z["lam"].astype(np.float64), n)
    T = int(tok.max()) + 1
    fsize = np.bincount(tok, minlength=T)
    mg = np.bincount(gc, minlength=ng).astype(np.float64)
    corpora = {-1: fam_csr(tok, T)}
    res = dict(data=a.data, count=a.count, tau=tau, outputs={}, nodes={})
    trees = {}
    for th in THETAS:
        low = (kappa < th) & (s > 0)
        w = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kappa, nan=0.0), 0.0), s)
        trees[th] = tree(w)
    allnodes = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
    own = grp[gc[allnodes.lo.to_numpy()]]
    rec, kinfo = recurrence(allnodes.lo.to_numpy(), allnodes.hi.to_numpy(), own, tok, fsize, corpora, -1, cc, gc, grp,
                            mg, a.blocks)
    rec = pd.concat([allnodes, rec], axis=1)
    res["recurrence"] = dict(unique_intervals=int(len(allnodes)), seconds=round(time.time() - t0, 1), **kinfo)
    for th in THETAS:
        t = trees[th].merge(rec, on=["lo", "hi"], how="left")
        t["lam_ana"] = lam_of(t.n.to_numpy(), t.E_ana.to_numpy())
        t.to_parquet(nd / f"obs4b_t{th:g}.parquet", index=False)
        km = (t.n.to_numpy() >= 2) & (t.lam_ana.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
        df = save_loci(od, f"L4b_t{th:g}", t.lo.to_numpy()[km], t.hi.to_numpy()[km])
        res["outputs"][f"L4b_t{th:g}"] = loci_info(df, n)
        sz = (t.hi - t.lo + 1).to_numpy()
        res["nodes"][f"{th:g}"] = dict(nodes=int(len(t)), kept=int(km.sum()), roots=int((t.par < 0).sum()),
                                       kept_low_cohesion_parents=int((km & (t.m.to_numpy() < 1)).sum()),
                                       low_cohesion_parents=int((t.m.to_numpy() < 1).sum()),
                                       kept_size_q50_q90_q99_max=np.quantile(sz[km], [.5, .9, .99, 1]).tolist())
        log("L4b theta", th, res["outputs"][f"L4b_t{th:g}"], res["nodes"][f"{th:g}"])
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, LL / "results" / f"build4b_{tag}.json")


# =========================================================================== smoke
def cmd_smoke(a):
    """Small checks on the first genomes (login node): tree properties against a brute-force recursion, counts against a
    brute-force window scan."""
    tok, cc, gc, grp, kappa = load(a.data, a.genomes)
    n = len(tok)
    tau = DATA[a.data]["tau"]["group"]
    k, d, lam, info = bonds(tok, cc, gc, grp, "group", tau)
    s = boundary_score(k.astype(np.int64), d.astype(np.int64), lam.astype(np.float64), n)
    log("smoke bonds", info, "positive boundaries", int((s > 0).sum()))
    s2 = np.where(kappa < 0.05, 0.0, s)
    t = tree(s2)
    # brute force: every node interval [lo, hi] must satisfy min internal = m, max(delimiting) = e
    bad = 0
    for r in t.itertuples():
        inner = s2[r.lo:r.hi]
        left = s2[r.lo - 1] if r.lo > 0 else 0.0
        right = s2[r.hi] if r.hi < n else 0.0
        if not (inner.min() == r.m and max(left, right) == r.e and r.pers > 0 and (inner > 0).all()):
            bad += 1
    st, en = runs(s2 > 0)
    roots = t[t.par < 0]
    same_roots = set(zip(roots.lo, roots.hi)) == set(zip(st.tolist(), en.tolist()))
    log("smoke tree nodes", len(t), "bad", bad, "roots == runs", same_roots)
    # recurrence: brute force for 200 random nodes
    T = int(tok.max()) + 1
    fsize = np.bincount(tok, minlength=T)
    mg = np.bincount(gc).astype(np.float64)
    corpora = {-1: fam_csr(tok, T), 0: fam_csr(W.shuffled(tok, gc, 0), T)}
    toks = {-1: tok, 0: W.shuffled(tok, gc, 0)}
    rng = np.random.default_rng(1)
    pick = rng.choice(len(t), size=min(60, len(t)), replace=False)
    lo, hi = t.lo.to_numpy()[pick], t.hi.to_numpy()[pick]
    own = grp[gc[lo]]
    rec, ki = recurrence(lo, hi, own, tok, fsize, corpora, -1, cc, gc, grp, mg, 8)
    mism = 0
    for i in range(len(lo)):
        fam = [f for f in tok[lo[i]:hi[i] + 1] if fsize[f] >= 2]
        F = set(fam)
        if len(F) < 2 or len(fam) < 2:
            continue
        q = max(2, math.ceil(PHI * len(F)))
        Wn = CWIN * len(fam)
        for nm, tk, col in (("obs", toks[-1], "n"), ("shuf", toks[0], "n_shuf0")):
            hitg = set()
            pos = np.flatnonzero(np.isin(tk, list(F)))
            for st_ in pos:                      # a qualifying window can always start at a member gene
                if grp[gc[st_]] in hitg:
                    continue
                win = np.arange(st_, min(st_ + Wn, len(tk)))
                win = win[cc[win] == cc[st_]]
                if len(set(tk[win].tolist()) & F) >= q:
                    hitg.add(grp[gc[st_]])
            nb = len(hitg) - (1 if own[i] in hitg else 0)
            if nb != rec[col].iloc[i]:
                mism += 1
    log("smoke recurrence: brute-force mismatches", mism, "of", 2 * len(lo), "| E_ana q50/q90",
        np.quantile(rec.E_ana, [.5, .9]).tolist(), "n q50/q90", np.quantile(rec.n, [.5, .9]).tolist(),
        "n_shuf0 q50/q90", np.quantile(rec.n_shuf0, [.5, .9]).tolist())
    assert bad == 0 and same_roots and mism == 0
    log("smoke OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build4b", "smoke"])
    ap.add_argument("--data", default="dpann")
    ap.add_argument("--count", default="group")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--genomes", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"build": cmd_build, "build4b": cmd_build4b, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
