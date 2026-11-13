#!/usr/bin/env python
"""Round 2A of ws_vocab: the stage-1 token learner with (1) independence measured from genomic context (N_eff) and
(2) word classes, alone and together (predeclared2a.json).

Independence modes
  group   stage 1: n = lineage groups (emb_hom 0.75), E over groups
  genome  n = genomes, E over genomes
  neff    genome statistic scaled: Lambda_eff = Lambda_genome x N_eff / n, with N_eff = n / (1 + (n - 1) rho) from the
          similarity of the instances' genomic contexts (vector: zca10 sums; token: MinHash Jaccard of token sets)
  neffshuf  the same with every occurrence's context taken from a random other occurrence (control)
Classes (optional): after each phase (rounds until nothing commits) live word nodes are relabelled by class
  (leader clustering of compositions by token Jaccard or pooled-vector cosine at theta); phases alternate until a
  phase commits nothing.

usage (biotite):
  vocab2a.py ladder --specs SPEC[,SPEC...] --taus 6,8,10,12 --nulls 3 --procs P --threads N
     SPEC = mode:ctx:W:cls:theta  e.g. neff:vec:25:none:0 | neff:tok:50:none:0 | group:none:0:vec:0.75 | neffshuf:vec:25:none:0
  vocab2a.py smoke --genomes 300
Run dirs: $RD/vocab/runs2a/<spec tag>/tau<t>/<obs|nullr>/ (words.parquet, inst.npz, live.npz, b0.npy, run.json).
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
sys.path.insert(0, str(VD / "code"))
import vocab as V  # noqa: E402
from vocab import gain as vgain  # noqa: E402

XGENE = VD / "ctx2a" / "xgene_zca10_f32.npy"
SHIFT = np.int64(31)
MASK = (np.int64(1) << SHIFT) - 1
NH = 64
PRIME = np.int64(2147483647)
WORKER_THREADS = 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


# --------------------------------------------------------------------------- context kernels
@njit(parallel=True, cache=True)
def ctx_vec_keys(ptr, lo, hi, gen, cstart, cend, X, perm, W):
    """Per key (occurrences ptr[r]:ptr[r+1], sorted by genome): mean pairwise cosine c_bar of the per-genome pooled
    context vectors, and k (genomes with an informative context)."""
    nk = len(ptr) - 1
    D = X.shape[1]
    cbar = np.full(nk, np.nan)
    kout = np.zeros(nk, dtype=np.int64)
    for r in prange(nk):
        tot = np.zeros(D, dtype=np.float64)
        g = np.zeros(D, dtype=np.float64)
        S = np.zeros(D, dtype=np.float64)
        curg = -1
        gn = 0
        k = 0
        for o in range(ptr[r], ptr[r + 1]):
            a = max(cstart[lo[o]], lo[o] - W)
            b = min(cend[hi[o]], hi[o] + W)
            S[:] = 0.0
            cnt = 0
            for j in range(a, lo[o]):
                S += X[perm[j]]
                cnt += 1
            for j in range(hi[o] + 1, b + 1):
                S += X[perm[j]]
                cnt += 1
            if cnt == 0:
                continue
            nrm = np.sqrt((S * S).sum())
            if nrm == 0.0:
                continue
            if gen[o] != curg:
                if gn > 0:
                    gg = np.sqrt((g * g).sum())
                    if gg > 0:
                        tot += g / gg
                        k += 1
                g[:] = 0.0
                gn = 0
                curg = gen[o]
            g += S / nrm
            gn += 1
        if gn > 0:
            gg = np.sqrt((g * g).sum())
            if gg > 0:
                tot += g / gg
                k += 1
        kout[r] = k
        if k >= 2:
            cbar[r] = ((tot * tot).sum() - k) / (k * (k - 1.0))
    return cbar, kout


@njit(cache=True)
def _mh(t, s, A, B):
    return (A[s] * (t + 1) + B[s]) % PRIME


@njit(parallel=True, cache=True)
def ctx_tok_keys(ptr, lo, hi, gen, cstart, cend, tok, W, A, B):
    """Token analogue: per-genome pooled MinHash sketch of the window token sets (union = elementwise min);
    c_bar = mean pairwise Jaccard estimate over the k genome sketches."""
    nk = len(ptr) - 1
    cbar = np.full(nk, np.nan)
    kout = np.zeros(nk, dtype=np.int64)
    for r in prange(nk):
        # count genomes first
        ng = 0
        last = -1
        for o in range(ptr[r], ptr[r + 1]):
            if gen[o] != last:
                ng += 1
                last = gen[o]
        sk = np.full((ng, NH), PRIME, dtype=np.int64)
        has = np.zeros(ng, dtype=np.bool_)
        gi = -1
        last = -1
        for o in range(ptr[r], ptr[r + 1]):
            if gen[o] != last:
                gi += 1
                last = gen[o]
            a = max(cstart[lo[o]], lo[o] - W)
            b = min(cend[hi[o]], hi[o] + W)
            for j in range(a, b + 1):
                if lo[o] <= j <= hi[o]:
                    continue
                t = tok[j]
                has[gi] = True
                for s in range(NH):
                    h = _mh(t, s, A, B)
                    if h < sk[gi, s]:
                        sk[gi, s] = h
        k = 0
        for i in range(ng):
            if has[i]:
                k += 1
        kout[r] = k
        if k < 2:
            continue
        rows = np.empty((k, NH), dtype=np.int64)
        q = 0
        for i in range(ng):
            if has[i]:
                rows[q] = sk[i]
                q += 1
        pairs = 0.0
        for s in range(NH):
            col = np.sort(rows[:, s])
            run = 1
            for i in range(1, k):
                if col[i] == col[i - 1]:
                    run += 1
                else:
                    pairs += run * (run - 1.0)
                    run = 1
            pairs += run * (run - 1.0)
        cbar[r] = pairs / (NH * k * (k - 1.0))
    return cbar, kout


@njit(cache=True)
def syn_overlap(kptr, kgen, n_genomes):
    """O[g, h] = number of adjacency keys shared by genomes g and h (keys given as CSR of sorted genome lists)."""
    O = np.zeros((n_genomes, n_genomes), dtype=np.int32)
    for r in range(len(kptr) - 1):
        a, b = kptr[r], kptr[r + 1]
        for i in range(a, b):
            gi = kgen[i]
            O[gi, gi] += 1
            for j in range(i + 1, b):
                O[gi, kgen[j]] += 1
    return O


@njit(parallel=True, cache=True)
def syn_neff_keys(ptr, gen, S):
    """Per key (occurrences sorted by genome): n^2 / sum_{g,h in G} S(g,h) over the distinct carrier genomes G."""
    nk = len(ptr) - 1
    out = np.zeros(nk, dtype=np.float64)
    for r in prange(nk):
        m = 0
        last = -1
        for o in range(ptr[r], ptr[r + 1]):
            if gen[o] != last:
                m += 1
                last = gen[o]
        G = np.empty(m, dtype=np.int64)
        q = 0
        last = -1
        for o in range(ptr[r], ptr[r + 1]):
            if gen[o] != last:
                G[q] = gen[o]
                q += 1
                last = gen[o]
        tot = float(m)
        for i in range(m):
            for j in range(i + 1, m):
                tot += 2.0 * S[G[i], G[j]]
        out[r] = m * m / tot
    return out


@njit(cache=True)
def assign_f(occ_k, occ_d, occ_r, occ_g, key_E, key_f, tau, m, n_groups, min_n, min_neff):
    """vocab.assign with a per-key evidence factor f = N_eff / n: commit if n_claimed >= min_n,
    n_claimed * f >= min_neff and gain(n_claimed, E) * f >= tau."""
    claimed = np.zeros(m, dtype=np.bool_)
    stamp = np.full(n_groups, -1, dtype=np.int64)
    nk = len(key_E)
    key_ok = np.zeros(nk, dtype=np.bool_)
    key_n = np.zeros(nk, dtype=np.int64)
    key_lam = np.zeros(nk, dtype=np.float64)
    key_occ = np.zeros(nk, dtype=np.int64)
    ck = np.empty(len(occ_k), dtype=np.int64)
    cd = np.empty(len(occ_k), dtype=np.int64)
    cr = np.empty(len(occ_k), dtype=np.int64)
    tmp = np.empty(len(occ_k), dtype=np.int64)
    nc = 0
    i = 0
    N = len(occ_k)
    while i < N:
        r = occ_r[i]
        j = i
        nt = 0
        ng = 0
        while j < N and occ_r[j] == r:
            k = occ_k[j]
            d = occ_d[j]
            free = True
            for t in range(k, k + d + 1):
                if claimed[t]:
                    free = False
                    break
            if free:
                for t in range(k, k + d + 1):
                    claimed[t] = True
                tmp[nt] = j
                nt += 1
                g = occ_g[j]
                if stamp[g] != r:
                    stamp[g] = r
                    ng += 1
            j += 1
        lam = vgain(float(ng), key_E[r]) * key_f[r]
        if ng >= min_n and ng * key_f[r] >= min_neff and lam >= tau:
            key_ok[r] = True
            key_n[r] = ng
            key_lam[r] = lam
            key_occ[r] = nt
            for q in range(nt):
                ck[nc] = occ_k[tmp[q]]
                cd[nc] = occ_d[tmp[q]]
                cr[nc] = r
                nc += 1
        else:
            for q in range(nt):
                k = occ_k[tmp[q]]
                d = occ_d[tmp[q]]
                for t in range(k, k + d + 1):
                    claimed[t] = False
        i = j
    return ck[:nc], cd[:nc], cr[:nc], key_ok, key_n, key_lam, key_occ


@njit(cache=True)
def leaves_of(top_inst, ia, ib):
    """CSR of leaf gene positions (slots) of each top instance."""
    cnt = np.zeros(len(top_inst) + 1, dtype=np.int64)
    stack = np.empty(len(ia) + 2, dtype=np.int64)
    out = []
    for t in range(len(top_inst)):
        sp = 0
        stack[sp] = top_inst[t]
        sp += 1
        c = 0
        while sp > 0:
            sp -= 1
            x = stack[sp]
            for ch in (ia[x], ib[x]):
                if ch >= 0:
                    stack[sp] = ch
                    sp += 1
                else:
                    out.append(-ch - 1)
                    c += 1
        cnt[t + 1] = cnt[t] + c
    arr = np.empty(len(out), dtype=np.int64)
    for i in range(len(out)):
        arr[i] = out[i]
    return cnt, arr


@njit(cache=True)
def mix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def comp_hash(ptr, leaf, tok):
    """Order-free hash of the slot-token SET of each top instance."""
    out = np.zeros(len(ptr) - 1, dtype=np.uint64)
    for t in range(len(ptr) - 1):
        u = np.unique(tok[leaf[ptr[t]:ptr[t + 1]]])
        h = np.uint64(len(u))
        for x in u:
            h += mix64(np.uint64(x))
        out[t] = h
    return out


@njit(parallel=True, cache=True)
def pooled_vectors(ptr, leaf, X, perm, comp_of, ncomp):
    """unit(sum over a composition's instances of unit(sum of its slot-gene vectors))."""
    D = X.shape[1]
    nt = len(ptr) - 1
    iv = np.zeros((nt, D), dtype=np.float32)
    for t in prange(nt):
        s = np.zeros(D, dtype=np.float64)
        for q in range(ptr[t], ptr[t + 1]):
            s += X[perm[leaf[q]]]
        n = np.sqrt((s * s).sum())
        if n > 0:
            iv[t] = s / n
    out = np.zeros((ncomp, D), dtype=np.float64)
    for t in range(nt):
        out[comp_of[t]] += iv[t]
    for c in range(ncomp):
        n = np.sqrt((out[c] * out[c]).sum())
        if n > 0:
            out[c] /= n
    return out.astype(np.float32)


@njit(cache=True)
def leader_cluster(order, nbr, nsim, theta):
    """Compositions in `order`; each joins the most similar earlier leader among its neighbours with sim >= theta."""
    n = len(order)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    cls = np.full(n, -1, dtype=np.int64)
    is_leader = np.zeros(n, dtype=np.bool_)
    nc = 0
    for i in range(n):
        c = order[i]
        best, bs = -1, -1.0
        for q in range(nbr.shape[1]):
            o = nbr[c, q]
            if o < 0:
                continue
            if rank[o] < i and is_leader[o] and nsim[c, q] >= theta and nsim[c, q] > bs:
                best, bs = o, nsim[c, q]
        if best >= 0:
            cls[c] = cls[best]
        else:
            cls[c] = nc
            nc += 1
            is_leader[c] = True
    return cls, nc


# --------------------------------------------------------------------------- learner
class Corpus:
    def __init__(self, cc, gc, tok, grp, perm, X):
        self.cc, self.gc, self.tok, self.grp, self.perm, self.X = cc, gc, tok, grp, perm, X
        n = len(cc)
        ch = np.flatnonzero(np.diff(cc)) + 1
        starts = np.r_[0, ch]
        ends = np.r_[ch, n] - 1
        lens = ends - starts + 1
        self.cstart = np.repeat(starts, lens).astype(np.int64)
        self.cend = np.repeat(ends, lens).astype(np.int64)


def noise_floor(C, kind, W, A, B, npairs=200_000, seed=17):
    """Similarity between contexts of random 2-gene windows in different genomes (k = 2 keys)."""
    rng = np.random.default_rng(seed)
    n = len(C.cc)
    i = rng.integers(0, n - 1, size=3 * npairs)
    i = i[C.cc[i] == C.cc[i + 1]]
    a, b = i[: len(i) // 2], i[len(i) // 2: 2 * (len(i) // 2)]
    ok = C.gc[a] != C.gc[b]
    a, b = a[ok][:npairs], b[ok][:npairs]
    lo = np.empty(2 * len(a), dtype=np.int64)
    lo[0::2], lo[1::2] = a, b
    hi = lo + 1
    gen = np.empty(2 * len(a), dtype=np.int64)
    gen[0::2], gen[1::2] = 0, 1
    ptr = np.arange(0, 2 * len(a) + 1, 2, dtype=np.int64)
    if kind == "vec":
        cb, k = ctx_vec_keys(ptr, lo, hi, gen, C.cstart, C.cend, C.X, C.perm, W)
    else:
        cb, k = ctx_tok_keys(ptr, lo, hi, gen, C.cstart, C.cend, C.tok, W, A, B)
    cb = cb[k == 2]
    return float(np.mean(cb)), float(np.std(cb)), int(len(cb))


def synteny_matrix(C, null, min_n=2, w=3):
    """S(g,h): share of g's round-0 adjacent token pairs also adjacent in h, symmetrised, diagonal 1. Cached per corpus."""
    fn = VD / "ctx2a" / f"synS_{'obs' if null < 0 else f'null{null}'}.npy"
    if fn.exists():
        return np.load(fn)
    tok, cc, gc = C.tok, C.cc, C.gc
    n_genomes = int(gc.max()) + 1
    T = int(tok.max()) + 1
    tg = np.unique(tok * n_genomes + gc)
    cnt = np.bincount(tg // n_genomes, minlength=T)
    sym = np.where(cnt[tok] >= min_n, tok, -1).astype(np.int64)
    n = len(tok)
    k, d, key = V.occurrences(np.arange(n, dtype=np.int64), cc, sym, np.ones(n, dtype=np.bool_), w)
    kg = np.unique(np.c_[key, gc[k]], axis=0)
    o = np.lexsort((kg[:, 1], kg[:, 0]))
    kg = kg[o]
    ch = np.flatnonzero(np.diff(kg[:, 0])) + 1
    kptr = np.r_[0, ch, len(kg)].astype(np.int64)
    O = syn_overlap(kptr, kg[:, 1].astype(np.int64), n_genomes).astype(np.float64)
    O = O + np.triu(O, 1).T
    diag = np.diag(O).copy()
    Sm = O / np.maximum(diag[:, None], 1.0)
    Sm = 0.5 * (Sm + Sm.T)
    np.fill_diagonal(Sm, 1.0)
    tmp = fn.with_suffix(".tmp.npy")
    np.save(tmp, Sm.astype(np.float32))
    tmp.rename(fn)
    return Sm.astype(np.float32)


def similarity_top(kind, comp_tok_ptr, comp_tok, ncomp, vecs, topk=50):
    """Top-k most similar compositions (excluding self): token Jaccard (sparse) or pooled-vector cosine (dense)."""
    nbr = np.full((ncomp, topk), -1, dtype=np.int64)
    sim = np.zeros((ncomp, topk), dtype=np.float64)
    if ncomp < 2:
        return nbr, sim
    if kind == "tok":
        import scipy.sparse as sp
        rows = np.repeat(np.arange(ncomp), np.diff(comp_tok_ptr))
        M = sp.csr_matrix((np.ones(len(comp_tok), dtype=np.float32), (rows, comp_tok)), shape=(ncomp, int(comp_tok.max()) + 1))
        sz = np.diff(comp_tok_ptr).astype(np.float64)
        for s in range(0, ncomp, 5000):
            I = (M[s:s + 5000] @ M.T).tocsr()
            for r in range(I.shape[0]):
                c = s + r
                cols, inter = I.indices[I.indptr[r]:I.indptr[r + 1]], I.data[I.indptr[r]:I.indptr[r + 1]]
                keep = cols != c
                cols, inter = cols[keep], inter[keep]
                if len(cols) == 0:
                    continue
                j = inter / (sz[c] + sz[cols] - inter)
                o = np.argsort(-j, kind="stable")[:topk]
                nbr[c, :len(o)] = cols[o]
                sim[c, :len(o)] = j[o]
    else:
        for s in range(0, ncomp, 4096):
            Sm = vecs[s:s + 4096] @ vecs.T
            for r in range(Sm.shape[0]):
                Sm[r, s + r] = -2
            kk = min(topk, ncomp - 1)
            idx = np.argpartition(-Sm, kk - 1, axis=1)[:, :kk]
            vals = np.take_along_axis(Sm, idx, axis=1)
            o = np.argsort(-vals, axis=1, kind="stable")
            nbr[s:s + len(Sm), :kk] = np.take_along_axis(idx, o, axis=1)
            sim[s:s + len(Sm), :kk] = np.take_along_axis(vals, o, axis=1)
    return nbr, sim


def run_vocab2a(C, mode, tau, ctx="none", W=0, cls="none", theta=0.0, w=3, max_rounds=500, min_n=2, seed_ctl=17,
                verbose=True):
    t0 = time.time()
    tok, cc, gc = C.tok, C.cc, C.gc
    n = len(tok)
    n_genomes = int(gc.max()) + 1
    grp = C.grp if mode == "group" else np.arange(n_genomes, dtype=np.int64)
    ng = int(grp.max()) + 1
    order = np.lexsort((np.arange(n_genomes), grp))
    rank_of_genome = np.empty(n_genomes, dtype=np.int64)
    rank_of_genome[order] = np.arange(n_genomes)
    grp_of_rank = grp[order].astype(np.int64)
    T = int(tok.max()) + 1
    tg = np.unique(tok * ng + grp[gc])
    tok_groups = np.bincount(tg // ng, minlength=T)
    sym = np.where(tok_groups[tok] >= min_n, tok, -1).astype(np.int64)
    use_neff = mode in ("neff", "neffshuf")
    rngA = np.random.default_rng(12345)
    A = rngA.integers(1, PRIME, size=NH).astype(np.int64)
    Bh = rngA.integers(0, PRIME, size=NH).astype(np.int64)
    floor = None
    Ssyn = None
    if use_neff and ctx == "syn":
        Ssyn = C.S
        off = Ssyn[~np.eye(len(Ssyn), dtype=bool)]
        floor = dict(S_offdiag_q=np.quantile(off, [.5, .9, .99]).tolist())
    elif use_neff:
        c0, sd0, nf = noise_floor(C, ctx, W, A, Bh)
        floor = dict(c0=c0, sd0=sd0, pairs=nf)
        if verbose:
            log("noise floor", ctx, W, floor)
    ctl_rng = np.random.default_rng(seed_ctl)
    ns = np.arange(n, dtype=np.int64)
    ne = ns.copy()
    term = np.ones(n, dtype=np.bool_)
    ninst = np.full(n, -1, dtype=np.int64)
    ncont, ngen = cc.copy(), gc.copy()
    S = T
    words, inst_parts, rounds, phases = [], [], [], []
    inst_label = []          # per instance: last class label while live (or its own symbol)
    n_inst = 0
    b0 = np.zeros(n, dtype=np.bool_)
    phase = 0
    phase_commits = 0
    class_cost = 0.0
    paid = set()             # composition hashes already charged as class members (followers)
    rnd = 0
    while rnd < max_rounds:
        tr = time.time()
        k, d, key = V.occurrences(ns, ncont, sym, term, w)
        committed_round = 0
        if len(k):
            ukey, kinv = np.unique(key, return_inverse=True)
            g_occ = grp[ngen[k]]
            kg = np.unique(kinv * ng + g_occ)
            nG = np.bincount(kg // ng, minlength=len(ukey))
            cand = np.flatnonzero(nG >= min_n)
            lo_s, hi_s = ukey[cand] >> SHIFT, ukey[cand] & MASK
            live = sym >= 0
            sr = np.unique(sym[live] * n_genomes + rank_of_genome[ngen[live]], return_counts=True)
            s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
            indptr = np.searchsorted(s_of, np.arange(S + 1)).astype(np.int64)
            m_rank = np.bincount(rank_of_genome[ngen], minlength=n_genomes).astype(np.float64)
            E = V.expected_groups(lo_s, hi_s, indptr, r_of.astype(np.int64), sr[1].astype(np.float64), m_rank,
                                  grp_of_rank, float(w))
            nc_ = nG[cand]
            lam = np.where(nc_ > E, nc_ * np.log(nc_ / np.maximum(E, 1e-300)) - nc_ + E, 0.0)
            fct = np.ones(len(cand))
            neff_info = {}
            if use_neff:
                pre = np.flatnonzero(lam >= tau)          # Lambda_eff <= Lambda_genome
                rank_k = np.full(len(ukey), -1, dtype=np.int64)
                rank_k[cand[pre]] = np.arange(len(pre))
                orr = rank_k[kinv]
                sel = np.flatnonzero(orr >= 0)
                o = np.lexsort((ngen[k[sel]], orr[sel]))
                sel = sel[o]
                ptr = np.searchsorted(orr[sel], np.arange(len(pre) + 1)).astype(np.int64)
                lo_i, hi_i = ns[k[sel]], ne[k[sel] + d[sel]]
                if mode == "neffshuf":
                    p = ctl_rng.permutation(len(sel))
                    lo_i, hi_i = lo_i[p], hi_i[p]
                gen_i = ngen[k[sel]]
                nn = nc_[pre].astype(np.float64)
                if ctx == "syn":
                    if mode == "neffshuf":
                        gen_i = gen_i[ctl_rng.permutation(len(gen_i))]
                        oo = np.lexsort((gen_i, orr[sel]))
                        gen_i = gen_i[oo]
                    neff = np.minimum(syn_neff_keys(ptr, gen_i, Ssyn), nn)
                    rho = np.zeros(len(pre))
                else:
                    if ctx == "vec":
                        cb, kk = ctx_vec_keys(ptr, lo_i, hi_i, gen_i, C.cstart, C.cend, C.X, C.perm, W)
                    else:
                        cb, kk = ctx_tok_keys(ptr, lo_i, hi_i, gen_i, C.cstart, C.cend, tok, W, A, Bh)
                    kkf = kk.astype(np.float64)
                    se = floor["sd0"] * np.sqrt(2.0 / np.maximum(kkf * (kkf - 1), 1.0))
                    rho = np.where(kk >= 2, np.clip((cb - floor["c0"] - 2 * se) / (1 - floor["c0"]), 0, 1), 0.0)
                    rho = np.nan_to_num(rho, nan=0.0)
                    neff = nn / (1 + (nn - 1) * rho)
                fct[:] = 0.0
                fct[pre] = neff / nn
                lam = lam * fct
                neff_info = dict(pre=int(len(pre)), rho_q=np.quantile(rho, [.1, .5, .9]).tolist() if len(rho) else [],
                                 neff_q=np.quantile(neff, [.1, .5, .9]).tolist() if len(neff) else [])
                pas = (lam >= tau) & (nc_ * fct >= 2)
            else:
                pas = lam >= tau
            if rnd == 0 or (phase == 0 and rnd == 0):
                # round-0 significant adjacencies for the phrase reading
                sig_key = np.zeros(len(ukey), dtype=bool)
                sig_key[cand[pas]] = True
                si = sig_key[kinv]
                cov = np.zeros(n + 1, dtype=np.int32)
                np.add.at(cov, ns[k[si]], 1)
                np.add.at(cov, ne[k[si] + d[si]], -1)
                b0 = np.cumsum(cov)[:n] > 0
            if pas.any():
                ck_idx = cand[pas]
                ordk = np.lexsort((ck_idx, -lam[pas]))
                ck_idx = ck_idx[ordk]
                cE, clam0, clo, chi = E[pas][ordk], lam[pas][ordk], lo_s[pas][ordk], hi_s[pas][ordk]
                cf, cn0 = fct[pas][ordk], nG[ck_idx]
                rank_of_key = np.full(len(ukey), -1, dtype=np.int64)
                rank_of_key[ck_idx] = np.arange(len(ck_idx))
                orr = rank_of_key[kinv]
                sel = np.flatnonzero(orr >= 0)
                o = np.lexsort((k[sel], d[sel], orr[sel]))
                sel = sel[o]
                c_k, c_d, c_r, key_ok, key_n, key_lam, key_occ = assign_f(
                    k[sel], d[sel], orr[sel], g_occ[sel], cE, cf, float(tau), len(ns), ng, min_n,
                    2.0 if use_neff else 0.0)
                committed_round = int(key_ok.sum())
                if committed_round:
                    new_sym = np.full(len(ck_idx), -1, dtype=np.int64)
                    okr = np.flatnonzero(key_ok)
                    new_sym[okr] = S + np.arange(len(okr))
                    words.append(pd.DataFrame(dict(sym=new_sym[okr], a=clo[okr], b=chi[okr], lam=key_lam[okr],
                                                   n=key_n[okr], E=cE[okr], f=cf[okr], lam0=clam0[okr], n0=cn0[okr],
                                                   occ=key_occ[okr], round=rnd, phase=phase)))
                    S += len(okr)
                    oc = np.argsort(c_k, kind="stable")
                    c_k, c_d, c_s = c_k[oc], c_d[oc], new_sym[c_r[oc]]
                    ns, ne, sym, term, ninst, i_sym, i_s, i_e, i_a, i_b = V.rebuild(ns, ne, sym, term, ninst, c_k, c_d,
                                                                                      c_s, n_inst)
                    ncont, ngen = cc[ns], gc[ns]
                    inst_parts.append((i_sym, i_s, i_e, i_a, i_b, np.full(len(i_sym), rnd, dtype=np.int64)))
                    inst_label.append(i_sym.copy())
                    n_inst += len(i_sym)
            rounds.append(dict(round=rnd, phase=phase, occurrences=int(len(k)), keys=int(len(ukey)),
                               keys_n2=int(len(cand)), candidates=int(pas.sum()), committed=committed_round,
                               live_nodes=int(len(ns)), rss_gb=rss_gb(), seconds=round(time.time() - tr, 1), **neff_info))
            if verbose:
                log("round", rnd, {k_: v for k_, v in rounds[-1].items() if k_ != "round"}, f"{time.time()-t0:.0f}s")
        rnd += 1
        phase_commits += committed_round
        if committed_round:
            continue
        # phase end
        if cls == "none" or phase_commits == 0:
            break
        tc = time.time()
        I_a = np.concatenate([p[3] for p in inst_parts])
        I_b = np.concatenate([p[4] for p in inst_parts])
        lab_all = np.concatenate(inst_label)
        topn = np.flatnonzero(~term)
        top_inst = ninst[topn]
        lptr, leaf = leaves_of(top_inst, I_a, I_b)
        h = comp_hash(lptr, leaf, tok)
        uh, comp_of = np.unique(h, return_inverse=True)
        ncomp = len(uh)
        gcount = np.bincount(np.unique(comp_of * ng + grp[ngen[topn]]) // ng, minlength=ncomp)
        icount = np.bincount(comp_of, minlength=ncomp)
        first = np.unique(comp_of, return_index=True)[1]
        if cls == "tok":
            parts = [np.unique(tok[leaf[lptr[t]:lptr[t + 1]]]) for t in first]
            ctp = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
            ctk = np.concatenate(parts).astype(np.int64)
            nbr, nsim = similarity_top("tok", ctp, ctk, ncomp, None)
        else:
            vecs = pooled_vectors(lptr, leaf, C.X, C.perm, comp_of.astype(np.int64), ncomp)
            nbr, nsim = similarity_top("vec", None, None, ncomp, vecs)
        order_c = np.lexsort((np.arange(ncomp), -icount, -gcount))
        clsid, nclass = leader_cluster(order_c.astype(np.int64), nbr, nsim, float(theta))
        sizes = np.bincount(clsid, minlength=nclass)
        # each follower composition (a member that is not its class leader) is a rule costing tau, charged once
        leader_of_class = np.full(nclass, -1, dtype=np.int64)
        for c_ in order_c:
            if leader_of_class[clsid[c_]] < 0:
                leader_of_class[clsid[c_]] = c_
        followers = np.flatnonzero(leader_of_class[clsid] != np.arange(ncomp))
        new_f = [int(x) for x in uh[followers] if int(x) not in paid]
        paid.update(new_f)
        class_cost += len(new_f) * tau
        newlab = S + clsid[comp_of]
        S += nclass
        sym[topn] = newlab
        lab_all[top_inst] = newlab
        # write labels back into the per-round label arrays
        off = 0
        for p_i, arr in enumerate(inst_label):
            inst_label[p_i] = lab_all[off:off + len(arr)]
            off += len(arr)
        phases.append(dict(phase=phase, commits=int(phase_commits), live_words=int(len(topn)), compositions=int(ncomp),
                           classes=int(nclass), multi_classes=int((sizes > 1).sum()),
                           compositions_in_multi=int(sizes[sizes > 1].sum()), seconds=round(time.time() - tc, 1)))
        if verbose:
            log("phase", phases[-1])
        phase += 1
        phase_commits = 0
    words = pd.concat(words, ignore_index=True) if words else pd.DataFrame(
        columns=["sym", "a", "b", "lam", "n", "E", "f", "lam0", "n0", "occ", "round", "phase"])
    if inst_parts:
        inst = {nm: np.concatenate([p[i] for p in inst_parts]) for i, nm in enumerate(("sym", "s", "e", "a", "b", "round"))}
        inst["label"] = np.concatenate(inst_label)
    else:
        inst = {nm: np.zeros(0, dtype=np.int64) for nm in ("sym", "s", "e", "a", "b", "round", "label")}
    D = float((words.lam - tau).sum() - class_cost) if len(words) else 0.0
    return dict(words=words, inst=inst, live=dict(ns=ns, ne=ne, sym=sym, term=term, ninst=ninst), rounds=rounds,
                phases=phases, b0=b0, T=T, seconds=round(time.time() - t0, 1), floor=floor, D=D,
                class_cost=class_cost, rss_gb=rss_gb())


def summary(res, n):
    live = res["live"]
    top = ~live["term"]
    cov = int((live["ne"][top] - live["ns"][top] + 1).sum())
    return dict(word_types=int(len(res["words"])), instances=int(len(res["inst"]["sym"])), top_instances=int(top.sum()),
                top_labels=int(len(np.unique(live["sym"][top]))), genes_in_top=cov, genes_in_top_frac=cov / n,
                rounds=len(res["rounds"]), phases=len(res["phases"]), D=res["D"], seconds=res["seconds"],
                rss_gb=res["rss_gb"], floor=res["floor"])


# --------------------------------------------------------------------------- specs, runs, ladder
def parse_spec(s):
    mode, ctx, W, cls, theta = s.split(":")
    return dict(mode=mode, ctx=ctx, W=int(W), cls=cls, theta=float(theta))


def spec_tag(sp):
    t = sp["mode"]
    if sp["mode"].startswith("neff"):
        t += f"_{sp['ctx']}" + (f"{sp['W']}" if sp["ctx"] != "syn" else "")
    if sp["cls"] != "none":
        t += f"_cls{sp['cls']}{sp['theta']:g}"
    return t


def run_dir(sp, tau, null):
    return VD / "runs2a" / spec_tag(sp) / f"tau{tau:g}" / ("obs" if null < 0 else f"null{null}")


def build_corpus(null, arm="main", with_syn=False):
    cc, gc, _ = V.load_universe()
    tok = V.load_tokens(len(cc))
    grp = V.load_groups(arm, int(gc.max()) + 1)
    perm = np.arange(len(cc), dtype=np.int64)
    if null >= 0:
        rng = np.random.default_rng(1000 + null)
        gs, ge = V.genome_bounds(gc)
        for s, e in zip(gs, ge):
            perm[s:e] = s + rng.permutation(e - s)
        tok = tok[perm]
    X = np.asarray(np.load(XGENE, mmap_mode="r")) if XGENE.exists() else None
    C = Corpus(cc, gc, tok, grp, perm, X)
    C.S = synteny_matrix(C, null) if with_syn else None
    return C


def one(args):
    spec, tau, null = args
    import numba
    numba.set_num_threads(WORKER_THREADS)
    sp = parse_spec(spec)
    C = build_corpus(null, with_syn=(sp["ctx"] == "syn"))
    res = run_vocab2a(C, sp["mode"], tau, sp["ctx"], sp["W"], sp["cls"], sp["theta"], verbose=False)
    s = summary(res, len(C.cc))
    out = run_dir(sp, tau, null)
    out.mkdir(parents=True, exist_ok=True)
    res["words"].to_parquet(out / "words.parquet", index=False)
    if null < 0:
        np.savez(out / "inst.npz", **res["inst"])
        np.savez(out / "live.npz", **res["live"])
        np.save(out / "b0.npy", res["b0"])
    with open(out / "run.json", "w") as fh:
        json.dump(dict(spec=spec, tau=tau, null=null, summary=s, rounds=res["rounds"], phases=res["phases"], T=res["T"]),
                  fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log(spec, tau, "obs" if null < 0 else f"null{null}", s)
    return dict(spec=spec, tau=tau, null=null, **{k: v for k, v in s.items() if k != "floor"})


def cmd_ladder(a):
    import multiprocessing as mp
    global WORKER_THREADS
    specs = a.specs.split(",")
    taus = [float(x) for x in a.taus.split(",")]
    jobs = [(s, t, r) for s in specs for t in taus for r in [-1] + list(range(a.nulls))]
    jobs.sort(key=lambda j: (j[1], j[2]))
    procs = min(a.procs or a.threads, len(jobs))
    WORKER_THREADS = max(1, a.threads // procs)
    log("ladder2a", specs, taus, len(jobs), "runs on", procs, "processes x", WORKER_THREADS, "threads")
    with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    out = []
    for s in specs:
        summ = []
        for t, g in df[df.spec == s].groupby("tau"):
            o = g[g.null < 0].iloc[0]
            nl = g[g.null >= 0]
            nm = float(nl.word_types.mean()) if len(nl) else float("nan")
            summ.append(dict(tau=t, obs_types=int(o.word_types), null_types_mean=nm, fdr_types=nm / max(o.word_types, 1),
                             obs_genes=int(o.genes_in_top), genes_frac=float(o.genes_in_top_frac), D=float(o.D),
                             phases=int(o.phases), rounds=int(o.rounds), obs_seconds=float(o.seconds),
                             obs_rss_gb=float(o.rss_gb)))
        sel = [x for x in summ if x["fdr_types"] <= 0.01]
        res = dict(spec=s, tag=spec_tag(parse_spec(s)), ladder=summ, tau_star=(min(x["tau"] for x in sel) if sel else None),
                   written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        out.append(res)
        for x in summ:
            log(s, x)
        log(s, "tau*", res["tau_star"])
    fn = VD / "results" / f"ladder2a_{a.tag}.json"
    with open(fn, "w") as fh:
        json.dump(dict(results=out, runs=rows), fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log("wrote", fn)


def _synS(null):
    C = build_corpus(null)
    t0 = time.time()
    Sm = synteny_matrix(C, null)
    off = Sm[~np.eye(len(Sm), dtype=bool)]
    log("synS", null, "offdiag q50/q90/q99/max", np.quantile(off, [.5, .9, .99, 1]).round(4).tolist(), f"{time.time()-t0:.0f}s")
    return null


def cmd_synS(a):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4) as pool:
        pool.map(_synS, [-1, 0, 1, 2])


def cmd_smoke(a):
    cc, gc, _ = V.load_universe()
    keep = gc < a.genomes
    cc, gc = cc[keep], gc[keep]
    tok = V.load_tokens(len(keep))[keep]
    grp = pd.factorize(V.load_groups("main", 3564)[: a.genomes])[0].astype(np.int64)
    X = np.load(XGENE, mmap_mode="r")[: len(cc)] if XGENE.exists() else \
        np.random.default_rng(0).standard_normal((len(cc), 16)).astype(np.float32)
    C = Corpus(cc, gc, tok, grp, np.arange(len(cc), dtype=np.int64), np.ascontiguousarray(X))
    # 1) group mode without classes reproduces vocab.run_vocab
    r1 = V.run_vocab(tok, cc, gc, grp, a.tau, verbose=False)
    r2 = run_vocab2a(C, "group", a.tau, verbose=False)
    w1, w2 = r1["words"].sort_values("sym").reset_index(drop=True), r2["words"].sort_values("sym").reset_index(drop=True)
    same = len(w1) == len(w2) and all(np.array_equal(w1[c].to_numpy(), w2[c].to_numpy()) for c in ("sym", "a", "b", "n", "occ"))
    same &= all(np.array_equal(r1["inst"][c], r2["inst"][c]) for c in ("s", "e", "a", "b"))
    log("group mode == vocab.py:", same, len(w1), len(w2))
    # 2) genome mode == vocab.py with genome units
    r3 = V.run_vocab(tok, cc, gc, np.arange(a.genomes, dtype=np.int64), a.tau, verbose=False)
    r4 = run_vocab2a(C, "genome", a.tau, verbose=False)
    log("genome mode == vocab.py genome units:", len(r3["words"]) == len(r4["words"]) and
        np.array_equal(r3["inst"]["s"], r4["inst"]["s"]), len(r3["words"]), len(r4["words"]))
    # 3) context kernels: direct numpy check on a few keys
    rng = np.random.default_rng(1)
    lo = np.sort(rng.integers(0, len(cc) - 2, 40))
    hi = lo + 1
    gen = gc[lo]
    o = np.lexsort((gen, np.repeat(np.arange(4), 10)))
    lo, hi, gen = lo[o], hi[o], gen[o]
    ptr = np.arange(0, 41, 10, dtype=np.int64)
    cb, kk = ctx_vec_keys(ptr, lo, hi, gen, C.cstart, C.cend, C.X, C.perm, 25)
    ref = []
    for r in range(4):
        G = {}
        for q in range(ptr[r], ptr[r + 1]):
            a_, b_ = max(C.cstart[lo[q]], lo[q] - 25), min(C.cend[hi[q]], hi[q] + 25)
            idx = [j for j in range(a_, b_ + 1) if not lo[q] <= j <= hi[q]]
            if not idx:
                continue
            s_ = C.X[idx].astype(np.float64).sum(0)
            G.setdefault(gen[q], np.zeros_like(s_))
            G[gen[q]] += s_ / np.linalg.norm(s_)
        Bs = np.array([v / np.linalg.norm(v) for v in G.values()])
        kq = len(Bs)
        ref.append(((Bs.sum(0) ** 2).sum() - kq) / (kq * (kq - 1)) if kq >= 2 else np.nan)
    log("vector c_bar kernel vs numpy max |d|:", float(np.nanmax(np.abs(cb - np.array(ref)))), kk.tolist())
    A = np.random.default_rng(12345).integers(1, PRIME, size=NH).astype(np.int64)
    Bh = np.random.default_rng(54321).integers(0, PRIME, size=NH).astype(np.int64)
    cbt, kt = ctx_tok_keys(ptr, lo, hi, gen, C.cstart, C.cend, tok, 25, A, Bh)
    reft = []
    for r in range(4):
        G = {}
        for q in range(ptr[r], ptr[r + 1]):
            a_, b_ = max(C.cstart[lo[q]], lo[q] - 25), min(C.cend[hi[q]], hi[q] + 25)
            st = {int(tok[j]) for j in range(a_, b_ + 1) if not lo[q] <= j <= hi[q]}
            if st:
                G.setdefault(gen[q], set()).update(st)
        Ss = list(G.values())
        js = [len(x & y) / len(x | y) for i_, x in enumerate(Ss) for y in Ss[i_ + 1:]]
        reft.append(np.mean(js) if js else np.nan)
    log("token c_bar (MinHash) vs exact Jaccard:", np.round(cbt, 3).tolist(), np.round(reft, 3).tolist())
    # 4) neff and class runs complete and tile the genes
    for mode, ctx, cl, th in (("neff", "vec", "none", 0), ("neff", "tok", "none", 0), ("neffshuf", "vec", "none", 0),
                              ("group", "none", "tok", 0.5), ("group", "none", "vec", 0.75), ("neff", "vec", "vec", 0.75)):
        r = run_vocab2a(C, mode, a.tau, ctx, 25, cl, th, verbose=False)
        lv = r["live"]
        assert np.all(lv["ns"][1:] == lv["ne"][:-1] + 1) and np.all(cc[lv["ns"]] == cc[lv["ne"]])
        log(mode, ctx, cl, th, summary(r, len(tok)), r["phases"][:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ladder", "smoke", "synS"])
    ap.add_argument("--specs", default="")
    ap.add_argument("--taus", default="6,8,10,12")
    ap.add_argument("--nulls", type=int, default=3)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--tag", default="part1")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--genomes", type=int, default=300)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"ladder": cmd_ladder, "smoke": cmd_smoke, "synS": cmd_synS}[a.cmd](a)


if __name__ == "__main__":
    main()
