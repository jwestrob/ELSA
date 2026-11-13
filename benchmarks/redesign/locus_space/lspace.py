#!/usr/bin/env python
"""Locus space (workstream ws_locusspace): a locus as a point / point cloud of whitened gene vectors.

  lspace.py gold   --threads T   parts 1-2: representations of gold loci and of the nested-reading (DH) loci that
                                 overlap gold; leave-one-genome-group-out typing, clustering, disjoint-family and
                                 sibling-subtype tests
Further stages (map, search) live in lspace_map.py / lspace_search.py.
Predeclaration: predeclared.json (part_1_gold_typing).
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
PKG = RD / "ws_dpann/package"
OUT = Path(os.environ.get("LSP_OUT", RD / "locus_space/out"))
TOKENS = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
ENG = RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"
DH = RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet"
HCORE = RD / "boundary/loci/rescue3/H_res0.3_f0.5.parquet"
ZROW = RD / "locus_search/out3/vec/zca10_f32__row.npy"
TOPK = RD / "head/ls_zca10/topk_student"
GROUPS = {"g075": RD / "lineagefree/groups/emb_hom_average_t0.75.parquet",
          "g070": RD / "lineagefree/groups/emb_hom_average_t0.7.parquet"}
SEED = 17
DRAWS = 2000
THETA, GAP, SPEN = 0.14, 0.3, 0.3
SK_GRID = [(4096, 1), (4096, 8), (32768, 1), (32768, 8)]
SK_BETA = 20.0
RFF_GRID = [0.5, 2.0]
RFF_D = 8192
LEIDEN_RES = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
KNN_K = 15
FAM_PREFIX = ("RM_", "Lamassu-", "pAgo_", "Gao_", "CBASS_", "BREX_", "Thoeris_", "PARIS_", "Mokosh_", "Zorya_", "Wadjet_",
              "Druantia_", "Retron_", "Avs_", "Rst_")
CALLERS_Q = ("DefenseFinder", "CRISPRCasTyper")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jd(o):
    if isinstance(o, dict):
        return {str(k): jd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jd(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return jd(o.tolist())
    return o


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump(jd(obj), open(tmp, "w"), indent=1)
    tmp.replace(path)


# =========================================================================== universe
class Universe:
    def __init__(self, vectors=True):
        g = pd.read_parquet(PKG / "universe/genes.parquet",
                            columns=["gid", "genome_code", "contig_code", "position_index", "strand", "length"])
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
        self.N = len(g)
        self.gc = g.genome_code.to_numpy(np.int64)
        self.cc = g.contig_code.to_numpy(np.int64)
        self.pos = g.position_index.to_numpy(np.int64)
        self.strand = g.strand.to_numpy(np.int8)
        self.length = g.length.to_numpy(np.int64)
        self.n_genomes = int(self.gc.max()) + 1
        gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
        self.genome_code = dict(zip(gm.genome, gm.genome_code))
        t = pd.read_parquet(TOKENS)
        tok = np.full(self.N, -1, np.int64)
        tok[t.gid.to_numpy()] = t.token.to_numpy()
        assert (tok >= 0).all()
        self.tok = tok
        tm = pd.read_parquet(ENG / "token_modules.parquet")
        mt = np.full(int(tok.max()) + 1, -1, np.int64)
        mt[tm.token.to_numpy()] = tm.module.to_numpy()
        self.mod = mt[tok]
        self.groups = {}
        for k, p in GROUPS.items():
            G = pd.read_parquet(p, columns=["genome_code", "group"])
            a = np.full(self.n_genomes, -1, np.int64)
            a[G.genome_code.to_numpy()] = G.group.to_numpy()
            assert (a >= 0).all()
            self.groups[k] = a
        self.X = np.load(ZROW, mmap_mode="r") if vectors else None

    def vecs(self, gids):
        gids = np.asarray(gids, np.int64)
        o = np.argsort(gids, kind="stable")
        out = np.empty((len(gids), self.X.shape[1]), np.float32)
        out[o] = self.X[gids[o]]
        return out


# =========================================================================== gold
def df_family(t):
    for p in FAM_PREFIX:
        if t.startswith(p):
            return p.rstrip("_-")
    return t


def load_gold(U):
    I = pd.read_parquet(PKG / "gold/instances.parquet",
                        columns=["instance_id", "caller", "system_type", "system_subtype", "genome", "member_gids"])
    I["mem"] = [np.unique(np.asarray(m, np.int64)) for m in I.member_gids]
    I = I[I.mem.map(len) >= 2].copy()
    I["label"] = np.where(I.caller.eq("CRISPRCasTyper"), I.system_subtype, I.system_type).astype(str)
    amb = I.caller.eq("CRISPRCasTyper") & I.label.str.contains(",|Hybrid", regex=True)
    log(f"gold >= 2 members: {len(I):,}; dropped ambiguous CRISPRCasTyper calls: {int(amb.sum())}")
    I = I[~amb].reset_index(drop=True)
    hi = []
    for c, l in zip(I.caller, I.label):
        hi.append(df_family(l) if c == "DefenseFinder" else (l.split("-")[0] if c == "CRISPRCasTyper" else l))
    I["hi"] = hi
    I["key"] = I.caller + ":" + I.label
    I["hikey"] = I.caller + ":" + I.hi
    I["genome_code"] = I.genome.map(U.genome_code).astype(np.int64)
    I["split"] = np.where(I.genome_code % 2 == 1, "odd", "even")
    for k, a in U.groups.items():
        I[k] = a[I.genome_code.to_numpy()]
    I["n_genes"] = I.mem.map(len)
    return I.drop(columns=["member_gids"])


def plurality(vals):
    v = vals[vals >= 0]
    if not len(v):
        return -1
    u, c = np.unique(v, return_counts=True)
    return int(u[np.argmax(c)])          # ties -> smallest id (np.unique sorts)


# =========================================================================== locus sets
class LocusSet:
    """Loci as ordered gene lists. ptr / pos_idx index into the unique gene array ug."""

    def __init__(self, U, gene_lists, mods=None):
        lens = np.array([len(g) for g in gene_lists], np.int64)
        self.n = len(gene_lists)
        self.ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
        flat = np.concatenate(gene_lists).astype(np.int64)
        self.ug, self.idx = np.unique(flat, return_inverse=True)
        self.idx = self.idx.astype(np.int64)
        self.gid = flat
        self.strand = U.strand[flat].astype(np.int64)
        self.tok = U.tok[flat]
        self.gmod = U.mod[flat]
        self.lens = lens
        self.maxlen = int(lens.max())
        self.mod = np.array([plurality(self.gmod[self.ptr[i]:self.ptr[i + 1]]) for i in range(self.n)], np.int64) \
            if mods is None else np.asarray(mods, np.int64)

    def incidence(self, values):
        import scipy.sparse as sp
        rows = np.repeat(np.arange(self.n), self.lens)
        ok = values >= 0
        u, inv = np.unique(values[ok], return_inverse=True)
        M = sp.csr_matrix((np.ones(ok.sum(), np.float32), (rows[ok], inv)), shape=(self.n, len(u)))
        M.data[:] = 1.0
        M.sum_duplicates()
        M.data[:] = 1.0
        return M


# =========================================================================== kernels
@njit(parallel=True, cache=True)
def chamfer_all(G, ptr, idx):
    n = ptr.shape[0] - 1
    S = np.zeros((n, n), np.float32)
    for i in prange(n):
        a0, a1 = ptr[i], ptr[i + 1]
        for j in range(i, n):
            b0, b1 = ptr[j], ptr[j + 1]
            sa = 0.0
            for p in range(a0, a1):
                m = -1e30
                gp = idx[p]
                for q in range(b0, b1):
                    v = G[gp, idx[q]]
                    if v > m:
                        m = v
                sa += m
            sb = 0.0
            for q in range(b0, b1):
                m = -1e30
                gq = idx[q]
                for p in range(a0, a1):
                    v = G[idx[p], gq]
                    if v > m:
                        m = v
                sb += m
            v = 0.5 * (sa / (a1 - a0) + sb / (b1 - b0))
            S[i, j] = v
            S[j, i] = v
    return S


@njit(cache=True)
def _sw(G, idx, strand, a0, a1, b0, b1, theta, gap, spen, rev, prev, cur):
    nb = b1 - b0
    for y in range(nb + 1):
        prev[y] = 0.0
    best = 0.0
    for x in range(a1 - a0):
        ga = idx[a0 + x]
        sa = strand[a0 + x]
        cur[0] = 0.0
        for y in range(1, nb + 1):
            q = b0 + (nb - y if rev else y - 1)
            sub = G[ga, idx[q]] - theta
            if spen > 0.0:
                sb = strand[q]
                if rev:
                    sb = -sb
                if sa != sb:
                    sub -= spen
            h = prev[y - 1] + sub
            t = prev[y] - gap
            if t > h:
                h = t
            t = cur[y - 1] - gap
            if t > h:
                h = t
            if h < 0.0:
                h = 0.0
            cur[y] = h
            if h > best:
                best = h
        for y in range(nb + 1):
            prev[y] = cur[y]
    return best


@njit(parallel=True, cache=True)
def aln_all(G, ptr, idx, strand, theta, gap, spen, maxlen):
    n = ptr.shape[0] - 1
    S = np.zeros((n, n), np.float32)
    selfs = np.zeros(n)
    for i in prange(n):
        prev = np.zeros(maxlen + 1)
        cur = np.zeros(maxlen + 1)
        selfs[i] = _sw(G, idx, strand, ptr[i], ptr[i + 1], ptr[i], ptr[i + 1], theta, gap, spen, False, prev, cur)
    for i in prange(n):
        prev = np.zeros(maxlen + 1)
        cur = np.zeros(maxlen + 1)
        for j in range(i, n):
            f = _sw(G, idx, strand, ptr[i], ptr[i + 1], ptr[j], ptr[j + 1], theta, gap, spen, False, prev, cur)
            r = _sw(G, idx, strand, ptr[i], ptr[i + 1], ptr[j], ptr[j + 1], theta, gap, spen, True, prev, cur)
            d = math.sqrt(selfs[i] * selfs[j])
            v = (f if f > r else r) / d if d > 0 else 0.0
            S[i, j] = v
            S[j, i] = v
    return S


# =========================================================================== representations
def rep_tokens(L):
    M = L.incidence(L.tok)
    inter = (M @ M.T).toarray().astype(np.float32)
    sz = np.asarray(M.sum(1)).ravel().astype(np.float32)
    J = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
    Mm = L.incidence(L.gmod)
    im = (Mm @ Mm.T).toarray().astype(np.float32)
    szm = np.asarray(Mm.sum(1)).ravel().astype(np.float32)
    MJ = im / np.maximum(szm[:, None] + szm[None, :] - im, 1)
    return J, MJ, inter > 0


def rep_mean(L, Xu):
    Xm = np.add.reduceat(Xu[L.idx], L.ptr[:-1], axis=0)
    Xm /= np.linalg.norm(Xm, axis=1, keepdims=True)
    return (Xm @ Xm.T).astype(np.float32)


def evidence_matrix(L):
    """E[a, b] = s_a(b) = log(N_db / rank of b in a's zca10 top-20,000 list), 0 outside the list (gold genes)."""
    gids = np.load(TOPK / "gids.npy")
    idx = np.load(TOPK / "idx.npy", mmap_mode="r")
    s = np.load(TOPK / "s.npy", mmap_mode="r")
    big = np.full(3_600_000, -1, np.int64)
    big[L.ug] = np.arange(len(L.ug))
    E = np.zeros((len(L.ug), len(L.ug)), np.float32)
    rows = big[gids]
    have = np.flatnonzero(rows >= 0)
    for c0 in range(0, len(have), 512):
        sel = have[c0:c0 + 512]
        I_ = np.asarray(idx[sel]).astype(np.int64)
        S_ = np.asarray(s[sel]).astype(np.float32)
        ok = I_ >= 0
        b = np.where(ok, big[np.where(ok, I_, 0)], -1)
        r, cidx = np.nonzero(b >= 0)
        E[rows[sel][r], b[r, cidx]] = S_[r, cidx]
    log(f"evidence matrix: {len(have):,} of {len(L.ug):,} genes have lists; nonzero {np.count_nonzero(E):,}")
    return 0.5 * (E + E.T)


def codebook(U, K, nsamp=1_000_000, threads=24):
    import faiss
    d = OUT / "codebook"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"K{K}.npz"
    if p.exists():
        z = np.load(p)
        return z["cent"], z["idf"]
    t0 = time.time()
    faiss.omp_set_num_threads(threads)
    odd = np.flatnonzero(U.gc % 2 == 1)
    rng = np.random.default_rng(SEED)
    smp = np.sort(rng.choice(odd, nsamp, replace=False))
    Xs = np.ascontiguousarray(U.X[smp], dtype=np.float32)
    nrm = np.linalg.norm(Xs, axis=1)
    log(f"codebook K={K}: sample {nsamp:,} odd-genome genes, norm q01/q50/q99 {np.quantile(nrm, [.01, .5, .99]).round(4)}")
    km = faiss.Kmeans(Xs.shape[1], K, niter=10, spherical=True, seed=SEED, verbose=False, max_points_per_centroid=1024)
    km.train(Xs)
    _, a = km.index.search(Xs, 1)
    cnt = np.bincount(a.ravel(), minlength=K)
    idf = np.log((nsamp + 1) / (cnt + 1)).astype(np.float32)
    cent = km.centroids.astype(np.float32)
    np.savez(p, cent=cent, idf=idf, counts=cnt, sample=smp)
    log(f"codebook K={K}: {time.time() - t0:.0f}s; empty {int((cnt == 0).sum())}, max cluster {int(cnt.max())}")
    return cent, idf


def sk_gene_features(Xu, cent, idf, r, threads=24):
    import faiss
    faiss.omp_set_num_threads(threads)
    ix = faiss.IndexFlatIP(cent.shape[1])
    ix.add(cent)
    D, I = ix.search(np.ascontiguousarray(Xu), r)
    w = np.exp(SK_BETA * (D - D[:, :1]))
    w /= w.sum(1, keepdims=True)
    w = (w * idf[I]).astype(np.float32)
    return I.astype(np.int64), w


def sk_locus_matrix(L, I, w, K):
    import scipy.sparse as sp
    r = I.shape[1]
    rows = np.repeat(np.repeat(np.arange(L.n), L.lens), r)
    V = sp.csr_matrix((w[L.idx].ravel(), (rows, I[L.idx].ravel())), shape=(L.n, K))
    V.sum_duplicates()
    nrm = np.sqrt(np.asarray(V.multiply(V).sum(1)).ravel())
    V = sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ V
    return V.tocsr()


def rff_gene_features(Xu, gamma, D=RFF_D):
    rng = np.random.default_rng(SEED)
    W = rng.normal(0.0, math.sqrt(2.0 * gamma), (Xu.shape[1], D)).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, D).astype(np.float32)
    Z = np.empty((Xu.shape[0], D), np.float32)
    for a in range(0, Xu.shape[0], 8192):
        Z[a:a + 8192] = np.cos(Xu[a:a + 8192] @ W + b) * math.sqrt(2.0 / D)
    return Z


def rff_locus_matrix(L, Z):
    V = np.add.reduceat(Z[L.idx], L.ptr[:-1], axis=0)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def build_reps(U, L, threads, with_chr=True, tag=""):
    reps, secs = {}, {}
    t0 = time.time()
    J, MJ, Sh = rep_tokens(L)
    reps["J"], reps["MODJ"] = J, MJ
    secs["J+MODJ"] = time.time() - t0
    t0 = time.time()
    Xu = U.vecs(L.ug)
    nrm = np.linalg.norm(Xu, axis=1)
    log(f"{tag} vectors {Xu.shape}, norm q01/q50/q99 {np.quantile(nrm, [.01, .5, .99]).round(4)}")
    Xu /= np.maximum(nrm[:, None], 1e-12)
    G = (Xu @ Xu.T).astype(np.float32)
    secs["gene_gram"] = time.time() - t0
    t0 = time.time()
    reps["MEAN"] = rep_mean(L, Xu)
    secs["MEAN"] = time.time() - t0
    t0 = time.time()
    reps["CH"] = chamfer_all(G, L.ptr, L.idx)
    secs["CH"] = time.time() - t0
    log(f"{tag} CH {secs['CH']:.1f}s")
    if with_chr:
        t0 = time.time()
        E = evidence_matrix(L)
        reps["CHr"] = chamfer_all(E, L.ptr, L.idx)
        del E
        secs["CHr"] = time.time() - t0
    t0 = time.time()
    reps["ALN"] = aln_all(G, L.ptr, L.idx, L.strand, THETA, GAP, 0.0, L.maxlen)
    secs["ALN"] = time.time() - t0
    t0 = time.time()
    reps["ALNs"] = aln_all(G, L.ptr, L.idx, L.strand, THETA, GAP, SPEN, L.maxlen)
    secs["ALNs"] = time.time() - t0
    log(f"{tag} ALN {secs['ALN']:.1f}s ALNs {secs['ALNs']:.1f}s")
    del G
    for K in sorted({k for k, _ in SK_GRID}):
        cent, idf = codebook(U, K, threads=threads)
        rmax = max(r for k, r in SK_GRID if k == K)
        t0 = time.time()
        I8, w8 = sk_gene_features(Xu, cent, idf, rmax, threads)
        for k, r in SK_GRID:
            if k != K:
                continue
            I_, w_ = I8[:, :r], w8[:, :r]
            if r < rmax:          # renormalise softmax weights over the kept codewords
                base = w_ / idf[I_]
                base /= base.sum(1, keepdims=True)
                w_ = (base * idf[I_]).astype(np.float32)
            V = sk_locus_matrix(L, I_, w_, K)
            reps[f"SK_K{K}_r{r}"] = (V @ V.T).toarray().astype(np.float32)
        secs[f"SK_K{K}"] = time.time() - t0
    for gm in RFF_GRID:
        t0 = time.time()
        Z = rff_gene_features(Xu, gm)
        V = rff_locus_matrix(L, Z)
        reps[f"RFF_g{gm:g}"] = (V @ V.T).astype(np.float32)
        secs[f"RFF_g{gm:g}"] = time.time() - t0
        del Z
    return reps, Sh, secs


# =========================================================================== evaluation
def majority(codes):
    u, c = np.unique(codes, return_counts=True)
    return int(u[np.argmax(c)])


def nn_eval(S, key, grp, q, zero_none, mod=None, restrict=None, k5=True):
    """Leave-one-group-out 1-NN over references in other groups. Returns pred (1-NN) and pred5 (weighted 5-NN)."""
    pred = np.full(len(q), -1, np.int64)
    pred5 = np.full(len(q), -1, np.int64)
    for t, i in enumerate(q):
        ok = grp != grp[i]
        if restrict is not None:
            ok &= restrict[t]
        if mod is not None:
            if mod[i] >= 0:
                m = ok & (mod == mod[i])
                if m.any():
                    pred[t] = majority(key[m])
                    pred5[t] = pred[t]
            continue
        cand = np.flatnonzero(ok)
        if not len(cand):
            continue
        s = S[i, cand]
        mx = s.max()
        if zero_none and mx <= 0:
            continue
        tied = cand[s >= mx - 1e-6 * max(1.0, abs(mx))]
        pred[t] = majority(key[tied])
        if k5:
            kk = min(5, len(cand))
            top = np.argpartition(-s, kk - 1)[:kk]
            wv = np.maximum(s[top], 0) + 1e-9
            kc = key[cand[top]]
            u = np.unique(kc)
            sc = np.array([wv[kc == x].sum() for x in u])
            pred5[t] = int(u[np.argmax(sc)])
    return pred, pred5


def boot_mean(values, groups, draws=DRAWS, seed=SEED):
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if not len(values):
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(codes))
    n = np.bincount(inv, minlength=len(codes)).astype(float)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    m = s[pick].sum(1) / np.maximum(n[pick].sum(1), 1)
    return [float(values.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def macro(correct, labels, min_n=5):
    df = pd.DataFrame(dict(c=correct, l=labels))
    g = df.groupby("l").c.agg(["mean", "size"])
    g = g[g["size"] >= min_n]
    return float(g["mean"].mean()) if len(g) else np.nan


def typing_eval(name, reps, mods, L_info, Sh, groups_key="g075"):
    """Per representation: per-query correctness for DF and CCT queries, both splits."""
    I = L_info
    key = pd.factorize(I.key)[0].astype(np.int64)
    hikey_of = dict(zip(I.key, I.hikey))
    keys_u = pd.factorize(I.key)[1]
    grp = I[groups_key].to_numpy(np.int64)
    q = np.flatnonzero(I.caller.isin(CALLERS_Q).to_numpy())
    rows = []
    for rep, S in list(reps.items()) + [("MOD", None)]:
        t0 = time.time()
        if rep == "MOD":
            pred, pred5 = nn_eval(None, key, grp, q, False, mod=mods)
        else:
            pred, pred5 = nn_eval(S, key, grp, q, rep in ("J", "MODJ"))
        hi_true = I.hikey.to_numpy()[q]
        hi_pred = np.array([hikey_of[keys_u[p]] if p >= 0 else "" for p in pred])
        rows.append(pd.DataFrame(dict(rep=rep, qi=q, caller=I.caller.to_numpy()[q], split=I.split.to_numpy()[q],
                                      genome_code=I.genome_code.to_numpy()[q], key=I.key.to_numpy()[q],
                                      correct=(pred == key[q]).astype(float), correct5=(pred5 == key[q]).astype(float),
                                      correct_hi=(hi_pred == hi_true).astype(float), has_pred=(pred >= 0).astype(float))))
        log(f"{name} typing {rep}: {time.time() - t0:.1f}s")
    return pd.concat(rows, ignore_index=True)


def summarize_typing(P):
    out = {}
    for (rep, caller, split), g in P.groupby(["rep", "caller", "split"]):
        d = dict(n=int(len(g)), acc=boot_mean(g.correct, g.genome_code) if split == "even" else [float(g.correct.mean())],
                 acc5=boot_mean(g.correct5, g.genome_code) if split == "even" else [float(g.correct5.mean())],
                 acc_hi=boot_mean(g.correct_hi, g.genome_code) if split == "even" else [float(g.correct_hi.mean())],
                 macro=macro(g.correct.to_numpy(), g.key.to_numpy()), coverage=float(g.has_pred.mean()))
        out.setdefault(rep, {}).setdefault(caller, {})[split] = d
    return out


def paired(P, x, y, caller, col="correct", split="even"):
    a = P[(P.rep == x) & (P.caller == caller) & (P.split == split)].set_index("qi")
    b = P[(P.rep == y) & (P.caller == caller) & (P.split == split)].set_index("qi")
    ids = a.index.intersection(b.index)
    return boot_mean(a.loc[ids, col].to_numpy() - b.loc[ids, col].to_numpy(), b.loc[ids, "genome_code"].to_numpy())


def select_best(summary, names):
    sc = {n: 0.5 * (summary[n]["DefenseFinder"]["odd"]["acc"][0] + summary[n]["CRISPRCasTyper"]["odd"]["acc"][0])
          for n in names if n in summary}
    return max(sc, key=sc.get), sc


# --------------------------------------------------------------------------- clustering
def knn_graph(S, k=KNN_K):
    import igraph as ig
    import scipy.sparse as sp
    n = S.shape[0]
    S2 = S.copy()
    np.fill_diagonal(S2, -np.inf)
    nb = np.argpartition(-S2, k, axis=1)[:, :k]
    r = np.repeat(np.arange(n), k)
    c = nb.ravel()
    w = np.maximum(S[r, c], 0).astype(np.float64)
    A = sp.csr_matrix((w + 1e-12, (r, c)), shape=(n, n))
    A = A.maximum(A.T).tocoo()
    m = A.row < A.col
    g = ig.Graph(n=n, edges=list(zip(A.row[m].tolist(), A.col[m].tolist())), edge_attrs={"weight": A.data[m].tolist()})
    return g


def leiden(g, res):
    import leidenalg
    p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=res, seed=SEED, n_iterations=-1)
    return np.array(p.membership, np.int64)


def cluster_metrics(lab, cl, grp, idx):
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure
    a = lab[idx]
    c = cl[idx]
    ami = adjusted_mutual_info_score(a, c)
    h, cm, _ = homogeneity_completeness_v_measure(a, c)
    g = grp[idx]
    same_l = a[:, None] == a[None, :]
    same_c = c[:, None] == c[None, :]
    cross = g[:, None] != g[None, :]
    tri = np.triu(np.ones(len(idx), bool), 1)
    m = cross & tri
    tp = (same_l & same_c & m).sum()
    prec = tp / max((same_c & m).sum(), 1)
    rec = tp / max((same_l & m).sum(), 1)
    return dict(ami=float(ami), homogeneity=float(h), completeness=float(cm), clusters=int(len(np.unique(c))),
                pair_precision_cross_group=float(prec), pair_recall_cross_group=float(rec))


def boot_ami(lab, cl, gcode, idx, draws=DRAWS):
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure
    rng = np.random.default_rng(SEED)
    g = gcode[idx]
    codes, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(codes))]
    vals = np.empty((draws, 3))
    for d in range(draws):
        pick = rng.integers(0, len(codes), len(codes))
        sel = idx[np.concatenate([members[k] for k in pick])]
        a, c = lab[sel], cl[sel]
        h, cm, _ = homogeneity_completeness_v_measure(a, c)
        vals[d] = (adjusted_mutual_info_score(a, c), h, cm)
    return {k: [float(np.quantile(vals[:, j], .025)), float(np.quantile(vals[:, j], .975))]
            for j, k in enumerate(("ami", "homogeneity", "completeness"))}


_CL = {}


def _cluster_task(rep):
    lab, grp, gcode, odd, even, modcl = (_CL[k] for k in ("lab", "grp", "gcode", "odd", "even", "modcl"))
    S = _CL["reps"].get(rep)
    if rep == "MOD":
        cl = modcl
        best = dict(res=None)
    else:
        g = knn_graph(S)
        best = None
        for res in LEIDEN_RES:
            cl_ = leiden(g, res)
            m = cluster_metrics(lab, cl_, grp, odd)
            if best is None or m["ami"] > best["odd"]["ami"]:
                best = dict(res=res, odd=m, cl=cl_)
        cl = best.pop("cl")
    ev = cluster_metrics(lab, cl, grp, even)
    ev["ci"] = boot_ami(lab, cl, gcode, even)
    return rep, dict(resolution=best.get("res"), odd=best.get("odd"), even=ev), cl


def clustering(reps, I, mods, threads):
    import multiprocessing as mp
    lab = pd.factorize(I.key)[0].astype(np.int64)
    grp = I.g075.to_numpy(np.int64)
    gcode = I.genome_code.to_numpy(np.int64)
    odd = np.flatnonzero(I.split.eq("odd").to_numpy())
    even = np.flatnonzero(I.split.eq("even").to_numpy())
    modcl = mods.copy()
    neg = np.flatnonzero(modcl < 0)
    modcl[neg] = modcl.max() + 1 + np.arange(len(neg))      # no module -> own singleton cluster
    _CL.clear()
    _CL.update(reps=reps, lab=lab, grp=grp, gcode=gcode, odd=odd, even=even, modcl=modcl)
    tasks = list(reps) + ["MOD"]
    out, cls = {}, {}
    with mp.get_context("fork").Pool(min(threads, len(tasks))) as pool:
        for rep, d, cl in pool.imap_unordered(_cluster_task, tasks):
            out[rep] = d
            cls[rep] = cl
            log(f"cluster {rep}: res {d['resolution']} even AMI {d['even']['ami']:.3f} clusters {d['even']['clusters']}")
    return out, cls


# --------------------------------------------------------------------------- disjoint-family and sibling tests
def disjoint_eval(reps, mods, I, Sh):
    key = pd.factorize(I.key)[0].astype(np.int64)
    hik = pd.factorize(I.hikey)[0].astype(np.int64)
    kc2h = np.full(key.max() + 1, -1, np.int64)
    kc2h[key] = hik                                   # label code -> higher-level code
    grp = I.g075.to_numpy(np.int64)
    q = np.flatnonzero((I.caller.isin(CALLERS_Q) & I.split.eq("even")).to_numpy())
    rows = []
    for i in q:
        other = grp != grp[i]
        cand = other & ~Sh[i]
        same = cand & (key == key[i])
        if same.any():
            cn = np.flatnonzero(cand)
            r = dict(qi=i, test="disjoint_same_label", caller=I.caller.iat[i], genome_code=int(I.genome_code.iat[i]),
                     chance=float(same.sum() / cand.sum()), n_cand=int(len(cn)))
            for rep, S in reps.items():
                s = S[i, cn]
                mx = s.max()
                if rep in ("J", "MODJ") and mx <= 0:
                    r[rep] = 0.0
                    continue
                tied = cn[s >= mx - 1e-6 * max(1.0, abs(mx))]
                r[rep] = float(majority(key[tied]) == key[i])
            m = cand & (mods == mods[i]) if mods[i] >= 0 else np.zeros_like(cand)
            r["MOD"] = float(majority(key[m]) == key[i]) if m.any() else 0.0
            rows.append(r)
        for variant, base in (("sibling_all", other), ("sibling_disjoint", other & ~Sh[i])):
            foreign = base & (key != key[i])
            sib = foreign & (hik == hik[i])
            if not sib.any():
                continue
            fn = np.flatnonzero(foreign)
            r = dict(qi=i, test=variant, caller=I.caller.iat[i], genome_code=int(I.genome_code.iat[i]),
                     chance=float(sib.sum() / foreign.sum()), n_cand=int(len(fn)))
            for rep, S in reps.items():
                s = S[i, fn]
                mx = s.max()
                if rep in ("J", "MODJ") and mx <= 0:
                    r[rep] = 0.0
                    continue
                tied = fn[s >= mx - 1e-6 * max(1.0, abs(mx))]
                r[rep] = float(kc2h[majority(key[tied])] == hik[i])
            m = foreign & (mods == mods[i]) if mods[i] >= 0 else np.zeros_like(foreign)
            r["MOD"] = float(kc2h[majority(key[m])] == hik[i]) if m.any() else 0.0
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_disjoint(D, reps):
    out = {}
    for (test, caller), g in D.groupby(["test", "caller"]):
        d = dict(n=int(len(g)), chance=boot_mean(g.chance, g.genome_code))
        for rep in list(reps) + ["MOD"]:
            d[rep] = boot_mean(g[rep], g.genome_code)
            d[rep + "_minus_chance"] = boot_mean(g[rep] - g.chance, g.genome_code)
        out.setdefault(test, {})[caller] = d
    return out


# =========================================================================== engine (DH) loci overlapping gold
def best_dh_loci(U, I):
    D = pd.read_parquet(DH)
    D["cc"] = U.cc[D.gid_s.to_numpy()]
    D["span"] = D.gid_e - D.gid_s + 1
    by = {c: g[["gid_s", "gid_e", "module_id", "span"]].to_numpy() for c, g in D.groupby("cc")}
    rows = []
    for i, mem in enumerate(I.mem):
        c = int(U.cc[mem[0]])
        arr = by.get(c)
        best = None
        if arr is not None:
            for s, e, mid, sp_ in arr:
                ov = int(np.searchsorted(mem, e, "right") - np.searchsorted(mem, s, "left"))
                if ov == 0:
                    continue
                cov, prec = ov / len(mem), ov / sp_
                f1 = 2 * cov * prec / (cov + prec)
                k = (f1, -sp_, -s)
                if best is None or k > best[0]:
                    best = (k, s, e, mid, cov, prec, f1)
        if best is None:
            rows.append(dict(i=i, gid_s=-1, gid_e=-1, module_id=-1, cov=0.0, prec=0.0, f1=0.0))
        else:
            _, s, e, mid, cov, prec, f1 = best
            rows.append(dict(i=i, gid_s=int(s), gid_e=int(e), module_id=int(mid), cov=cov, prec=prec, f1=f1))
    return pd.DataFrame(rows)


# =========================================================================== stage: gold
def stage_gold(a):
    import numba
    numba.set_num_threads(a.threads)
    out = OUT / "gold"
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    U = Universe()
    I = load_gold(U)
    log("instances", I.groupby(["caller", "split"]).size().to_dict(), "labels", I.groupby("caller").key.nunique().to_dict())
    res = dict(stage="gold", n_instances=I.groupby(["caller", "split"]).size().reset_index().values.tolist(),
               n_groups_g075=int(I.g075.nunique()), timings={})
    for setname in ("gold", "engine"):
        t1 = time.time()
        if setname == "gold":
            Iset = I.copy()
            L = LocusSet(U, [m for m in Iset.mem])
            mods = L.mod
        else:
            B = best_dh_loci(U, I)
            B.to_parquet(out / "engine_best_dh.parquet", index=False)
            keep = (B["cov"] >= 0.5).to_numpy()
            res["engine_coverage50"] = {c: float(keep[(I.caller == c).to_numpy()].mean()) for c in I.caller.unique()}
            res["engine_f1_median"] = {c: float(B.f1[(I.caller == c).to_numpy()].median()) for c in I.caller.unique()}
            Iset = I[keep].reset_index(drop=True)
            Bk = B[keep].reset_index(drop=True)
            spans = (Bk.gid_e - Bk.gid_s + 1).to_numpy()
            res["engine_span_q"] = np.quantile(spans, [.5, .9, .99, 1]).tolist()
            L = LocusSet(U, [np.arange(s, e + 1) for s, e in zip(Bk.gid_s, Bk.gid_e)], mods=Bk.module_id.to_numpy())
            mods = L.mod
        log(f"[{setname}] loci {L.n:,}, unique genes {len(L.ug):,}, max length {L.maxlen}")
        reps, Sh, secs = build_reps(U, L, a.threads, with_chr=(setname == "gold"), tag=setname)
        res["timings"][setname] = secs
        log(f"[{setname}] representations built in {time.time() - t1:.0f}s")
        P = typing_eval(setname, reps, mods, Iset, Sh)
        # stringency: coarser genome groups (g070) for every representation
        P2 = typing_eval(setname + "/g070", reps, mods, Iset, Sh, groups_key="g070")
        P.to_parquet(out / f"{setname}_typing_perquery.parquet", index=False)
        P2.to_parquet(out / f"{setname}_typing_perquery_g070.parquet", index=False)
        summ = summarize_typing(P)
        summ70 = summarize_typing(P2)
        names = list(reps) + ["MOD"]
        sk_names = [n for n in names if n.startswith("SK_")]
        rff_names = [n for n in names if n.startswith("RFF_")]
        sel = {}
        sel["SK"], sel["SK_scores"] = select_best(summ, sk_names)
        sel["RFF"], sel["RFF_scores"] = select_best(summ, rff_names)
        main = ["J", "MOD", "MODJ", "MEAN", "CH", "ALN", "ALNs", sel["SK"], sel["RFF"]] + (["CHr"] if "CHr" in reps else [])
        sel["best"], sel["best_scores"] = select_best(summ, main)
        sel["best_embedding"], _ = select_best(summ, [n for n in main if n not in ("J", "MOD", "MODJ")])
        sel["indexable"], sel["indexable_scores"] = select_best(summ, ["MEAN", sel["SK"], sel["RFF"]])
        log(f"[{setname}] selection (odd queries): {sel}")
        pairs = {}
        be, ix = sel["best_embedding"], sel["indexable"]
        cmp = [(be, "J"), (be, "MOD"), ("CH", "MEAN"), ("ALN", "CH"), ("ALNs", "ALN"), ("CH", ix), (ix, "J"), (ix, "MOD"),
               ("MEAN", "J"), ("CH", "J"), ("CH", "MOD"), ("J", "MOD")]
        if "CHr" in reps:
            cmp += [("CHr", "CH")]
        for x, y in cmp:
            for c in CALLERS_Q:
                for col in ("correct", "correct_hi"):
                    pairs[f"{x}-{y}|{c}|{col}"] = paired(P, x, y, c, col)
        res[setname] = dict(summary=summ, summary_g070=summ70, selection=sel, paired=pairs)
        dump(res, out / "gold_results.json")
        if setname == "gold":
            t1 = time.time()
            D = disjoint_eval(reps, mods, Iset, Sh)
            D.to_parquet(out / "disjoint_perquery.parquet", index=False)
            res["disjoint"] = summarize_disjoint(D, reps)
            res["timings"]["disjoint"] = time.time() - t1
            log(f"disjoint tests {time.time() - t1:.0f}s")
            dump(res, out / "gold_results.json")
            Iset.drop(columns=["mem"]).assign(mod=mods).to_parquet(out / "gold_instances.parquet", index=False)
            keep_reps = {k: v for k, v in reps.items() if k in set(main)}
        else:
            keep_reps = {k: v for k, v in reps.items() if k in set(main)}
        t1 = time.time()
        cl, cls = clustering(keep_reps, Iset, mods, a.threads)
        res[setname]["clustering"] = cl
        pd.DataFrame(cls).to_parquet(out / f"{setname}_clusters.parquet", index=False)
        res["timings"][setname + "_clustering"] = time.time() - t1
        dump(res, out / "gold_results.json")
        for k in ("MEAN", "CH", "J", sel["SK"]):
            np.save(out / f"{setname}_S_{k}.npy", reps[k].astype(np.float16))
        del reps
    res["seconds"] = time.time() - T0
    dump(res, out / "gold_results.json")
    log(f"gold stage done in {res['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gold"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 4)))
    a = ap.parse_args()
    {"gold": stage_gold}[a.stage](a)


if __name__ == "__main__":
    main()
