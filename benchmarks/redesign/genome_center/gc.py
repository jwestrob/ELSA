#!/usr/bin/env python
"""genome_center: remove each genome's own component before zca10 whitening (predeclared.json, sha256 in NOTEBOOK.txt).

Arms (all on the stored ESM++ small residue means, fitted on all 3,585,303 genes like REPORT3's zca10):
  Z    zca10 (REPORT3)                       G1 / G05   v - alpha * genome mean, then zca10 (alpha 1 / 0.5)
  B8 / B32   project out the top 8 / 32 principal directions of the genome-mean matrix (genomes weighted equally), then zca10

Stages (run on biotite; $RD/genome_center):
  gc.py smoke  --n 512             Z on 512 member genes: list identity vs the stored head lists, readout kernel, timing;
                                   search3 grid / null machinery vs REPORT3 rows (exact)
  gc.py lists  --arms Z,G1         transforms, variance shares (S2), top-20,000 lists for the 18,414 member genes,
                                   z >= 10.6 per-genome lists and top-1,000 group-excluded lists (S3, S4) -> lists/<arm>/
  gc.py grid                       scan-only AP grid (W extra x tau) on DefenseFinder, 5-fold CV, reproduction gate,
                                   primary readout, frozen-configuration scans (DF / TXSScan / CRISPRCasTyper even)
  gc.py nullx  --part i/n          200-replicate null runs, CRISPRCasTyper even queries, frozen configurations
  gc.py final                      all result tables, results/summary.json, results/tables.txt, STATUS
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
import scipy.sparse as sp
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
GC = Path(os.environ.get("GC_OUT", RD / "genome_center"))
os.environ.setdefault("LS_OUT3", str(GC / "ls_out3_unused"))      # search3 writes nothing here; guard against out3 writes
sys.path.insert(0, str(RD / "locus_search/code"))
sys.path.insert(0, str(RD / "lab/code"))
import bench  # noqa: E402
import embed_tokens as et  # noqa: E402
import locus_search as ls  # noqa: E402
import search3 as s3  # noqa: E402
from locus_search import log  # noqa: E402

OUT1 = s3.OUT1
OUT3REF = RD / "locus_search/out3"
H = s3.H
EMBRUN = s3.EMBRUN
ARMS = ["Z", "G1", "G05", "B8", "B32"]
ALPHA = {"G1": 1.0, "G05": 0.5}
KPC = {"B8": 8, "B32": 32}
K = 20000
K1 = 1000
ZCUT = 10.6
SEED = 17
CH = 1024
SUB = 64
BLK = et.BLK
GROUPS = RD / "lineagefree/groups/emb_hom_average_t0.75.parquet"
TOK_R3 = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
REF_M1Z_AP = 0.7961079531037453
RD_KEYS = ("zlen", "zgrp", "zhit", "k1grp", "k1hit", "k1first", "k1fam", "k1ocas")


def jd(o):
    return s3.jd(o)


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    s3.dump(obj, path)


def boot(v, g):
    return ls.boot_mean(np.asarray(v, float), np.asarray(g))


# =========================================================================== auxiliary tables (S2-S4 only)
def load_aux(db, qg):
    """Genome groups, rescue3 tokens, same-component target sets T_i (CSR over query rows), Cas flags."""
    t0 = time.time()
    N, NG = db.N, db.n_genomes
    t = pd.read_parquet(TOK_R3).sort_values("gid")
    assert np.array_equal(t.gid.to_numpy(), np.arange(N)), "rescue3 token table is not one row per gid"
    tok3 = t.token.to_numpy(np.int64)
    gm = pd.read_parquet(GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(gm.genome_code.to_numpy(), np.arange(NG))
    grp_genome = pd.factorize(gm.group.to_numpy())[0].astype(np.int32)        # as frozen3 fz3.groups()
    ngrp = int(grp_genome.max()) + 1
    grp_gene = grp_genome[db.gc].astype(np.int32)
    callers = bench.load_gold(db)
    recs = []
    for c in bench.CALLERS:
        C = callers[c]
        for r, m in enumerate(C["mem"]):
            lab = str(C["tlabels"][C["tcode"][r]])
            for g in m:
                recs.append((int(g), c, lab, bool(C["target"][r])))
    M = pd.DataFrame(recs, columns=["gid", "caller", "label", "target"])
    iscas = np.zeros(N, np.uint8)
    iscas[M.gid[M.caller == "CRISPRCasTyper"].to_numpy()] = 1
    row_of = {int(g): k for k, g in enumerate(qg)}
    Q = M[M.target & M.gid.isin(list(row_of))].drop_duplicates(["gid", "caller", "label"]).copy()
    Q["qrow"] = Q.gid.map(row_of).astype(np.int64)
    nq = len(qg)
    qcaller = {c: np.zeros(nq, bool) for c in bench.CALLERS}
    for c in bench.CALLERS:
        qcaller[c][Q.qrow[Q.caller == c].to_numpy()] = True
    Ma = M.drop_duplicates(["gid", "caller", "label"]).copy()
    Ma["tok"] = tok3[Ma.gid.to_numpy()]
    Ma["grp"] = grp_gene[Ma.gid.to_numpy()]
    Ma = Ma[Ma.tok >= 0]
    Q["tok"] = tok3[Q.gid.to_numpy()]
    Q["grp"] = grp_gene[Q.gid.to_numpy()]
    Q = Q[Q.tok >= 0]
    P = Q[["qrow", "caller", "label", "tok", "grp"]].merge(Ma[["gid", "caller", "label", "tok", "grp"]],
                                                           on=["caller", "label", "tok"], suffixes=("_q", ""))
    P = P[P.grp != P.grp_q][["qrow", "gid"]].drop_duplicates().sort_values(["qrow", "gid"])
    qr = P.qrow.to_numpy(np.int64)
    tg = P.gid.to_numpy(np.int64)
    tptr = np.searchsorted(qr, np.arange(nq + 1)).astype(np.int64)
    tlen = np.diff(tptr)
    tgen = np.zeros(nq, np.int64)
    if len(P):
        gq = pd.DataFrame(dict(q=qr, g=db.gc[tg])).drop_duplicates()
        tgen = np.bincount(gq.q.to_numpy(), minlength=nq).astype(np.int64)
    # family genes outside the query's own genome group (S4)
    ok = tok3 >= 0
    tot = np.bincount(tok3[ok])
    key = tok3[ok] * ngrp + grp_gene[ok]
    uk, uc = np.unique(key, return_counts=True)
    qtok = tok3[qg]
    qgrp = grp_gene[qg]
    qkey = qtok * ngrp + qgrp
    pos = np.searchsorted(uk, qkey)
    own = np.where((pos < len(uk)) & (uk[np.minimum(pos, len(uk) - 1)] == qkey), uc[np.minimum(pos, len(uk) - 1)], 0)
    famout = np.where(qtok >= 0, tot[np.maximum(qtok, 0)] - own, 0).astype(np.int64)
    qcas = qcaller["CRISPRCasTyper"].copy()
    aux = dict(tok3=tok3, grp_genome=grp_genome, grp_gene=grp_gene, ngrp=ngrp, iscas=iscas, tptr=tptr, tg=tg,
               qtok=qtok, qgrp=qgrp)
    den = dict(qg=qg, qgc=db.gc[qg], tlen=tlen, tgen=tgen, famout=famout, qcas=qcas, qtok=qtok, qgrp=qgrp,
               **{f"q_{c}": v for c, v in qcaller.items()})
    log(f"aux: {ngrp} genome groups, {len(M):,} curated memberships, T pairs {len(tg):,}, query genes with T {int((tlen > 0).sum()):,}, "
        f"Cas query genes {int(qcas.sum()):,}, {time.time() - t0:.0f}s")
    return aux, den


# =========================================================================== variance shares (S2)
@njit(parallel=True, cache=True)
def nb_unit(R):
    n, d = R.shape
    for i in prange(n):
        t = 0.0
        for k in range(d):
            t += R[i, k] * R[i, k]
        nr = np.float32(math.sqrt(t)) if t > 0 else np.float32(1.0)
        for k in range(d):
            R[i, k] = R[i, k] / nr


@njit(parallel=True, cache=True)
def nb_genome_sums(R, g_lo, g_hi, gs, sq):
    NG = len(g_lo)
    d = R.shape[1]
    for g in prange(NG):
        acc = np.zeros(d)
        q = 0.0
        for i in range(g_lo[g], g_hi[g]):
            for k in range(d):
                v = np.float64(R[i, k])
                acc[k] += v
                q += v * v
        gs[g, :] = acc
        sq[g] = q


@njit(parallel=True, cache=True)
def nb_family_sums(R, fptr, frows, S):
    nf = len(fptr) - 1
    d = R.shape[1]
    for f in prange(nf):
        acc = np.zeros(d)
        for t in range(fptr[f], fptr[f + 1]):
            i = frows[t]
            for k in range(d):
                acc[k] += R[i, k]
        for k in range(d):
            S[f, k] = acc[k]


@njit(parallel=True, cache=True)
def nb_resid(R, S, g_lo, g_hi, elig, fam_of, m_of, B, ssg):
    """Per genome: B[g] = sum of LOO residuals r_i = (m x_i - S_F) / (m - 1) over eligible genes, ssg[g] = sum |r_i|^2."""
    NG = len(g_lo)
    d = R.shape[1]
    for g in prange(NG):
        acc = np.zeros(d)
        q = 0.0
        for i in range(g_lo[g], g_hi[g]):
            if not elig[i]:
                continue
            m = m_of[i]
            a = m / (m - 1.0)
            b = 1.0 / (m - 1.0)
            f = fam_of[i]
            for k in range(d):
                r = R[i, k] * a - S[f, k] * b
                acc[k] += r
                q += r * r
        B[g, :] = acc
        ssg[g] = q


def genome_sums(R, db):
    NG, d = db.n_genomes, R.shape[1]
    gs = np.zeros((NG, d))
    sq = np.zeros(NG)
    nb_genome_sums(R, db.g_lo.astype(np.int64), db.g_hi.astype(np.int64), gs, sq)
    return gs, float(sq.sum())


def total_share(R, db):
    gs, sq = genome_sums(R, db)
    n = (db.g_hi - db.g_lo).astype(np.float64)
    N = n.sum()
    mu = gs.sum(0) / N
    ss_tot = sq - N * float(mu @ mu)
    k = n > 0
    mg = gs[k] / n[k, None]
    ss_b = float((n[k] * ((mg - mu) ** 2).sum(1)).sum())
    return gs, dict(total_share=ss_b / ss_tot, total_expected=float((k.sum() - 1) / (N - 1)),
                    global_mean_norm=float(np.linalg.norm(mu)),
                    genome_mean_dev_norm_q10_50_90=np.quantile(np.linalg.norm(mg - mu, axis=1), [.1, .5, .9]).tolist())


def resid_share(R, db, tok, u, mode):
    """Family-centered LOO residual share of genome means (signals/prep.py). mode 'row': family sums over genes;
    'uid': over distinct uids (signals' own definition; valid only when all rows of a uid are identical)."""
    N, d = R.shape
    NG = db.n_genomes
    ok = tok >= 0
    U = int(u.max()) + 1
    key = np.where(ok, tok * U + u, -1)
    uk, first = np.unique(key[ok], return_index=True)
    rep = np.flatnonzero(ok)[first]                     # first row of each distinct (token, uid)
    T = int(tok.max()) + 1
    m_uid = np.bincount(uk // U, minlength=T)
    elig_tok = m_uid >= 3
    nf = int(elig_tok.sum())
    fam_row = np.full(T, -1, np.int64)
    fam_row[elig_tok] = np.arange(nf)
    elig = ok & elig_tok[np.maximum(tok, 0)]
    ge = np.flatnonzero(elig)
    if mode == "uid":
        src = rep[elig_tok[tok[rep]]]
        m = m_uid[elig_tok].astype(np.float64)
    else:
        src = ge
        m = np.bincount(fam_row[tok[ge]], minlength=nf).astype(np.float64)
    fs = fam_row[tok[src]]
    o = np.argsort(fs, kind="stable")
    fptr = np.searchsorted(fs[o], np.arange(nf + 1)).astype(np.int64)
    S = buf("Sfam", (nf, d), np.float32)
    nb_family_sums(R, fptr, src[o].astype(np.int64), S)
    fam_of = np.where(elig, fam_row[np.maximum(tok, 0)], 0).astype(np.int64)
    m_of = np.where(elig, m[fam_of], 3.0)
    B = np.zeros((NG, d))
    ssg = np.zeros(NG)
    nb_resid(R, S, db.g_lo.astype(np.int64), db.g_hi.astype(np.int64), elig, fam_of, m_of, B, ssg)
    ss = float(ssg.sum())
    ng = np.bincount(db.gc[ge], minlength=NG).astype(np.float64)
    k = ng > 0
    share = float(((B[k] ** 2).sum(1) / ng[k]).sum() / ss)
    return {f"resid_share_{mode}": share, f"resid_expected_{mode}": float(k.sum() / len(ge)),
            f"resid_eligible_genes_{mode}": int(len(ge)), f"resid_families_{mode}": int(nf)}


def shares(R, db, tok, u, uid_mode=False):
    t0 = time.time()
    gs, out = total_share(R, db)
    out.update(resid_share(R, db, tok, u, "row"))
    if uid_mode:
        out.update(resid_share(R, db, tok, u, "uid"))
    out["seconds"] = round(time.time() - t0, 1)
    return gs, out


# =========================================================================== transforms (in place, preallocated buffers)
@njit(parallel=True, cache=True)
def nb_gather_unit(X, u, R):
    """R[i] = X[u[i]] / |X[u[i]]|; returns the number of rows with a non-finite value."""
    n = len(u)
    d = X.shape[1]
    bad = np.zeros(n, np.int64)
    for i in prange(n):
        s = u[i]
        t = 0.0
        for k in range(d):
            v = np.float64(X[s, k])
            t += v * v
        if not np.isfinite(t):
            bad[i] = 1
        nr = np.float32(math.sqrt(t)) if t > 0 else np.float32(1.0)
        for k in range(d):
            R[i, k] = X[s, k] / nr
    return bad.sum()


@njit(parallel=True, cache=True)
def nb_sub_genome(R, g_lo, g_hi, M):
    for g in prange(len(g_lo)):
        for i in range(g_lo[g], g_hi[g]):
            for k in range(R.shape[1]):
                R[i, k] -= M[g, k]


@njit(parallel=True, cache=True)
def nb_project_out(R, P):
    n, d = R.shape
    kk = P.shape[1]
    for i in prange(n):
        c = np.zeros(kk)
        for t in range(d):
            x = np.float64(R[i, t])
            for j in range(kk):
                c[j] += x * P[t, j]
        for t in range(d):
            s = 0.0
            for j in range(kk):
                s += P[t, j] * c[j]
            R[i, t] = np.float32(R[i, t] - s)


@njit(parallel=True, cache=True)
def nb_f16_sub(R):
    """float16 rounding (round to nearest even) for |x| < 2**-14 (float16 subnormals, spacing 2**-24)."""
    n, d = R.shape
    for i in prange(n):
        for k in range(d):
            x = R[i, k]
            if abs(x) < 6.103515625e-05:
                R[i, k] = np.float32(np.rint(np.float64(x) * 16777216.0) / 16777216.0)


@njit(parallel=True, cache=True)
def nb_f16_norm(Ru):
    """float16 rounding (round to nearest even, 10 mantissa bits) on the float32 bit patterns of normal-range values."""
    n, d = Ru.shape
    for i in prange(n):
        for k in range(d):
            b = Ru[i, k]
            if ((b >> np.uint32(23)) & np.uint32(0xFF)) >= np.uint32(113):
                Ru[i, k] = (b + np.uint32(0xFFF) + ((b >> np.uint32(13)) & np.uint32(1))) & np.uint32(0xFFFFE000)


def f16_round(R):
    nb_f16_sub(R)
    nb_f16_norm(R.view(np.uint32))


BUF = {}


def buf(name, shape, dtype):
    b = BUF.get(name)
    if b is None or b.shape != shape or b.dtype != dtype:
        BUF[name] = b = np.empty(shape, dtype)
    return b


def zca10_rows(R, renorm_first, db):
    """zca10 on gene rows (gene weights = REPORT3's count weights over unique sequences), float16 storage."""
    t0 = time.time()
    if renorm_first:
        nb_unit(R)
    gs, _ = genome_sums(R, db)
    mu = gs.sum(0) / len(R)                                    # gene-weighted global mean (float64)
    mu32 = mu.astype(np.float32)
    d = R.shape[1]
    C = np.zeros((d, d))
    b64 = buf("b64", (BLK, d), np.float64)
    b32 = buf("b32", (BLK, d), np.float32)
    for s0 in range(0, len(R), BLK):
        blk = R[s0:s0 + BLK]
        blk -= mu32
        bb = b64[:len(blk)]
        np.copyto(bb, blk)
        C += bb.T @ bb
    C /= len(R)
    t1 = time.time()
    W, ev = et.soft_zca(C)
    for s0 in range(0, len(R), BLK):
        blk = R[s0:s0 + BLK]
        o = b32[:len(blk)]
        np.matmul(blk, W, out=o)
        blk[...] = o
    nb_unit(R)
    t2 = time.time()
    f16_round(R)
    return dict(global_mean_norm=float(np.linalg.norm(mu)), eigenvalue_q0_50_90_100=np.quantile(ev, [0, .5, .9, 1]).tolist(),
                participation_ratio_before=float(ev.sum() ** 2 / (ev ** 2).sum()),
                seconds=dict(mean_cov=round(t1 - t0, 1), apply=round(t2 - t1, 1), f16=round(time.time() - t2, 1)))


def removal(R, arm, gs, db):
    """In place: the arm's removal step on unit-L2 rows; gs = genome sums of the unit rows."""
    n = (db.g_hi - db.g_lo).astype(np.float64)
    k = n > 0
    mg = np.zeros_like(gs)
    mg[k] = gs[k] / n[k, None]
    info = {}
    if arm in ALPHA:
        a = ALPHA[arm]
        nb_sub_genome(R, db.g_lo.astype(np.int64), db.g_hi.astype(np.int64), (a * mg).astype(np.float32))
        info["alpha"] = a
    elif arm in KPC:
        Mc = mg[k] - mg[k].mean(0)
        _, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
        kk = KPC[arm]
        nb_project_out(R, np.ascontiguousarray(Vt[:kk].T))
        info.update(k=kk, genome_mean_variance_in_top_k=float((sv[:kk] ** 2).sum() / (sv ** 2).sum()),
                    sv_share_top10=((sv[:10] ** 2) / (sv ** 2).sum()).tolist())
    nb_unit(R)
    return info


# =========================================================================== list pass: top-20,000 + readouts (S3, S4)
@njit(cache=True)
def _in(tg, a, b, x):
    lo, hi = a, b
    while lo < hi:
        mid = (lo + hi) >> 1
        if tg[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo < b and tg[lo] == x


@njit(cache=True)
def kth_smallest(a, n, k):
    """In place (a[:n] is reordered): the value at position k of the sorted a[:n] (Hoare quickselect, median of 3)."""
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        x = a[lo]
        y = a[mid]
        z = a[hi]
        if x > y:
            x, y = y, x
        if y > z:
            y, z = z, y
        if x > y:
            x, y = y, x
        p = y
        i = lo
        j = hi
        while i <= j:
            while a[i] < p:
                i += 1
            while a[j] > p:
                j -= 1
            if i <= j:
                t = a[i]
                a[i] = a[j]
                a[j] = t
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return a[k]
    return a[lo]


@njit(parallel=True, cache=True)
def query_rows(S, Wb, a0, qgrp, qtok, olo, ohi, tptr, tg, g_lo, g_hi, grp_genome, grp_gene, tok3, iscas, ngrp, zcut, k1, kk,
               c1, v1, cK, vK, out_idx, out_s, zlen, zgrp, zhit, k1grp, k1hit, k1first, k1fam, k1ocas):
    """Per query row: (S3a) z >= zcut per-genome best-match list outside the own genome group; (S3b/S4) top k1 outside
    the own genome group; top kk outside the own genome, exactly as locus_search.topk_rows (idx, s = log(N_db / tail rank)).
    Wb, c1, v1, cK, vK: preallocated scratch rows; S is left unchanged."""
    n, N = S.shape
    NG = len(g_lo)
    for r in prange(n):
        q = a0 + r
        row = S[r]
        w = Wb[r]
        gq = qgrp[q]
        t0 = tptr[q]
        t1 = tptr[q + 1]
        # ---- S3a
        mark = np.zeros(ngrp, np.uint8)
        zl = 0
        zg = 0
        zh = 0
        for G in range(NG):
            if grp_genome[G] == gq:
                continue
            lo = g_lo[G]
            hi = g_hi[G]
            nn = hi - lo
            if nn < 2:
                continue
            mx = -np.inf
            am = -1
            s1 = 0.0
            s2 = 0.0
            for j in range(lo, hi):
                v = row[j]
                s1 += v
                s2 += v * v
                if v > mx:
                    mx = v
                    am = j
            mean = s1 / nn
            var = (s2 - nn * mean * mean) / (nn - 1)
            sd = math.sqrt(var) if var > 0 else 0.0
            if sd < 1e-4:
                sd = 1e-4
            if (mx - mean) / sd >= zcut:
                zl += 1
                g2 = grp_genome[G]
                if mark[g2] == 0:
                    mark[g2] = 1
                    zg += 1
                if t1 > t0 and _in(tg, t0, t1, am):
                    zh += 1
        zlen[q] = zl
        zgrp[q] = zg
        zhit[q] = zh
        # ---- top k1 outside the own genome group
        for j in range(N):
            w[j] = -np.inf if grp_gene[j] == gq else row[j]
        kth = kth_smallest(w, N, N - k1)
        cand = c1[r]
        val = v1[r]
        m = 0
        for j in range(N):
            if grp_gene[j] != gq and row[j] > kth:
                cand[m] = j
                val[m] = row[j]
                m += 1
        for j in range(N):
            if m >= k1:
                break
            if grp_gene[j] != gq and row[j] == kth:
                cand[m] = j
                val[m] = kth
                m += 1
        o = np.argsort(-val[:m], kind="mergesort")
        mark2 = np.zeros(ngrp, np.uint8)
        kg = 0
        kh = 0
        first = 0
        fam = 0
        oc = 0
        tq = qtok[q]
        for t in range(m):
            j = cand[o[t]]
            g2 = grp_gene[j]
            if mark2[g2] == 0:
                mark2[g2] = 1
                kg += 1
            if t1 > t0 and _in(tg, t0, t1, j):
                kh += 1
            if tq >= 0 and tok3[j] == tq:
                fam += 1
                if first == 0:
                    first = t + 1
            elif iscas[j] == 1:
                oc += 1
        k1grp[q] = kg
        k1hit[q] = kh
        k1first[q] = first
        k1fam[q] = fam
        k1ocas[q] = oc
        # ---- top kk outside the own genome (locus_search.topk_rows semantics)
        lo = olo[q]
        hi = ohi[q]
        for j in range(N):
            w[j] = -np.inf if (j >= lo and j < hi) else row[j]
        n_db = N - (hi - lo)
        kth = kth_smallest(w, N, N - kk)
        cand = cK[r]
        val = vK[r]
        m = 0
        for j in range(N):
            if (j < lo or j >= hi) and row[j] > kth:
                cand[m] = j
                val[m] = row[j]
                m += 1
        m_gt = m
        n_eq = 0
        for j in range(N):
            if (j < lo or j >= hi) and row[j] == kth:
                n_eq += 1
                if m < kk:
                    cand[m] = j
                    val[m] = kth
                    m += 1
        o = np.argsort(-val[:m], kind="mergesort")
        cur = 0
        for t in range(m - 1, -1, -1):
            v = val[o[t]]
            if t == m - 1 or v != val[o[t + 1]]:
                cur = (t + 1) if v > kth else (m_gt + n_eq)
            out_idx[q, t] = cand[o[t]]
            out_s[q, t] = math.log(n_db / cur)
        for t in range(m, kk):
            out_idx[q, t] = -1
            out_s[q, t] = 0.0


def list_pass(R, db, qg, aux):
    n, N = len(qg), db.N
    own = db.gc[qg]
    olo, ohi = db.g_lo[own].astype(np.int64), db.g_hi[own].astype(np.int64)
    idx = buf("idx", (n, K), np.int32)
    sv = buf("sv", (n, K), np.float32)
    rd = {k_: np.zeros(n, np.int32) for k_ in RD_KEYS}
    Sb = buf("S", (CH, N), np.float32)
    Wb = buf("W", (SUB, N), np.float32)
    c1, v1 = buf("c1", (SUB, K1), np.int64), buf("v1", (SUB, K1), np.float32)
    cK, vK = buf("cK", (SUB, K), np.int64), buf("vK", (SUB, K), np.float32)
    g_lo, g_hi = db.g_lo.astype(np.int64), db.g_hi.astype(np.int64)
    tm = dict(matmul=0.0, rows=0.0)
    check = None
    for c, a in enumerate(range(0, n, CH)):
        b = min(n, a + CH)
        t0 = time.time()
        S = Sb[:b - a]
        np.matmul(R[qg[a:b]], R.T, out=S)
        t1 = time.time()
        for s0 in range(0, b - a, SUB):
            s1 = min(b - a, s0 + SUB)
            query_rows(S[s0:s1], Wb, a + s0, aux["qgrp"], aux["qtok"], olo, ohi, aux["tptr"], aux["tg"], g_lo, g_hi,
                       aux["grp_genome"], aux["grp_gene"], aux["tok3"], aux["iscas"], aux["ngrp"], ZCUT, K1, K, c1, v1, cK, vK,
                       idx, sv, rd["zlen"], rd["zgrp"], rd["zhit"], rd["k1grp"], rd["k1hit"], rd["k1first"], rd["k1fam"],
                       rd["k1ocas"])
        if c == 0:                                  # exactness check against locus_search.topk_rows on 4 rows
            T = np.array(S[:4])
            oi, oc, os_ = np.empty((4, K), np.int64), np.empty((4, K), np.float32), np.empty((4, K), np.float32)
            ls.topk_rows(T, K, olo[a:a + 4], ohi[a:a + 4], oi, oc, os_)
            check = dict(idx_equal=bool((oi == idx[a:a + 4]).all()), max_abs_s=float(np.abs(os_ - sv[a:a + 4]).max()))
            log("topk_rows check", check)
            assert check["idx_equal"] and check["max_abs_s"] < 1e-6, "list kernel differs from locus_search.topk_rows"
        t2 = time.time()
        tm["matmul"] += t1 - t0
        tm["rows"] += t2 - t1
        log(f"  lists {b:,}/{n:,} " + " ".join(f"{k_} {v:.0f}s" for k_, v in tm.items()))
    return idx, sv, rd, {k_: round(v, 1) for k_, v in tm.items()} | {"topk_rows_check": check}


def compare_stored(idx, sv, rows):
    """Z lists against the head workstream's stored zca10 lists (the lists behind REPORT3's M1z)."""
    d = H / "ls_zca10/topk_student"
    assert np.array_equal(np.load(d / "gids.npy"), np.load(OUT1 / "topk/gids.npy"))
    ri = np.load(d / "idx.npy", mmap_mode="r")
    rs = np.load(d / "s.npy", mmap_mode="r")
    same = np.zeros(len(rows), bool)
    ov = []
    mx = 0.0
    for a in range(0, len(rows), 2048):
        b = min(len(rows), a + 2048)
        A, B = idx[a:b], np.asarray(ri[rows[a:b]])
        same[a:b] = (A == B).all(1)
        mx = max(mx, float(np.abs(sv[a:b] - np.asarray(rs[rows[a:b]])).max()))
        for k_ in np.flatnonzero(~same[a:b])[:200]:
            ov.append(len(np.intersect1d(A[k_], B[k_])) / K)
    return dict(rows=int(len(rows)), rows_identical=int(same.sum()), share_identical=float(same.mean()),
                max_abs_s_diff=mx, overlap_of_differing_rows_q0_50=(np.quantile(ov, [0, .5]).tolist() if ov else None),
                differing_rows_sampled=len(ov))


def run_arms(arms, qrows, outroot, info_extra=None, smoke_mode=False, threads=24):
    t00 = time.time()
    db = ls.Database(vectors=None)
    g0 = np.load(OUT1 / "topk/gids.npy")
    qg = g0[qrows]
    u = np.load(EMBRUN / "uid_of_row.npy").astype(np.int64)
    aux, den = load_aux(db, qg)
    tmp = outroot / f"aux_den.tmp{os.getpid()}.npz"
    np.savez(tmp, qrows=qrows, **den)
    os.replace(tmp, outroot / "aux_den.npz")
    t0 = time.time()
    Xm = np.asarray(np.load(EMBRUN / "final" / "mean_raw_f32__uid.npy", mmap_mode="r"))
    assert len(u) == db.N and u.min() >= 0 and u.max() < len(Xm)
    R0 = np.empty((db.N, Xm.shape[1]), np.float32)
    nbad = int(nb_gather_unit(Xm, u, R0))
    assert nbad == 0, f"{nbad} genes with non-finite vectors"
    del Xm
    R = np.empty_like(R0)
    log(f"vectors: {R0.shape}, {time.time() - t0:.0f}s")
    tok = aux["tok3"]
    if smoke_mode:
        gs, sh_in = total_share(R0, db)
    else:
        gs, sh_in = shares(R0, db, tok, u, uid_mode=("Z" in arms))
    log("input shares", json.dumps(sh_in))
    for arm in arms:
        ta = time.time()
        od = outroot / "lists" / arm
        od.mkdir(parents=True, exist_ok=True)
        info = dict(arm=arm, query_genes=int(len(qg)), share_input=sh_in, **(info_extra or {}))
        np.copyto(R, R0)
        if arm != "Z":
            info["removal"] = removal(R, arm, gs, db)
            _, info["share_pre_whitening"] = shares(R, db, tok, u)
        info["zca10"] = zca10_rows(R, (arm != "Z"), db)
        _, info["share_final"] = shares(R, db, tok, u, uid_mode=(arm == "Z"))
        info["transform_seconds"] = round(time.time() - ta, 1)
        log(arm, "zca10", json.dumps(info["zca10"]), "removal", json.dumps(info.get("removal")))
        log(arm, "shares", json.dumps({k_: info[k_] for k_ in info if k_.startswith("share")}))
        idx, sv, rd, tm = list_pass(R, db, qg, aux)
        info["list_seconds"] = tm
        np.save(od / "gids.npy", qg)
        np.save(od / "idx.npy", idx)
        np.save(od / "s.npy", sv)
        np.savez(od / "readouts.npz", **rd)
        if arm == "Z":
            info["vs_stored_head_lists"] = compare_stored(idx, sv, qrows)
            log("Z vs stored", json.dumps(info["vs_stored_head_lists"]))
        info["arm_seconds"] = round(time.time() - ta, 1)
        dump(info, od / "info.json")
        log(f"arm {arm} done {info['arm_seconds']}s")
        del idx, sv
    log(f"run_arms done {time.time() - t00:.0f}s")


def lists(a):
    g0 = np.load(OUT1 / "topk/gids.npy")
    run_arms(a.arms.split(","), np.arange(len(g0)), GC, threads=a.threads)


# =========================================================================== smoke test
def smoke(a):
    t0 = time.time()
    out = GC / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    g0 = np.load(OUT1 / "topk/gids.npy")
    qrows = np.sort(np.random.default_rng(SEED).choice(len(g0), a.n, replace=False))
    run_arms(["Z"], qrows, out, dict(smoke=True), smoke_mode=True, threads=a.threads)
    res = dict(lists_seconds=round(time.time() - t0, 1))
    # search3 machinery against REPORT3 (exact): grid rows of M1z and one null run
    db, callers, queries = s3.setup(a, ["zca10"])
    bench.CTX["grid_fams"] = ["M1z"]
    Q = queries[queries.caller == "DefenseFinder"]
    qids = Q.qid.to_numpy()[np.random.default_rng(SEED).choice(len(Q), 40, replace=False)].tolist()
    rows = [r for ch in bench.pool_map(s3.grid_task, qids, a.threads, chunksize=1) for r in ch]
    G = pd.DataFrame(rows, columns=["qid", "fam", "w", "wx", "tau", "ap", "ap_strict", "n_rel", "n_hits"])
    ref = pd.read_parquet(OUT3REF / "grid_main.parquet", filters=[("fam", "==", "M1z")])
    mg = G.merge(ref[["qid", "wx", "tau", "ap"]].rename(columns={"ap": "ap_ref"}), on=["qid", "wx", "tau"])
    res["grid_vs_report3"] = dict(rows=int(len(mg)), max_abs_diff=float((mg.ap.fillna(-1) - mg.ap_ref.fillna(-1)).abs().max()))
    X = pd.read_parquet(OUT3REF / "final_x_metrics.parquet")
    X = X[(X.fam == "M1z") & (X.caller == "CRISPRCasTyper")].sort_values("n_rel")
    pick = X.qid.to_numpy()[[len(X) // 2, len(X) // 2 + 1]]
    rr = bench.pool_map(s3.final_task, [(int(q), "M1z", 1.0, 8, 3.0) for q in pick], 2, chunksize=1)
    cmp = []
    for m, _, _ in rr:
        x = X[X.qid == m["qid"]].iloc[0]
        cmp.append({c: float(abs(m[c] - x[c])) for c in ("ap", "recall_E0.01", "n_wrong_E0.01", "n_unlabelled_E0.01")})
    res["null_vs_report3"] = cmp
    res["seconds"] = round(time.time() - t0, 1)
    dump(res, out / "smoke.json")
    log("smoke", json.dumps(res))


# =========================================================================== grid, CV, gate, primary
def register():
    for arm in ARMS:
        nm = "gc" + arm
        s3.SRC_DIR[nm] = GC / "lists" / arm
        s3.FAM[nm] = [(False, 0.0, nm, 1.0)]
    return ["gc" + x for x in ARMS]


def grid(a):
    t0 = time.time()
    out = GC / "out"
    out.mkdir(parents=True, exist_ok=True)
    fams = register()
    db, callers, queries = s3.setup(a, fams)
    bench.CTX["grid_fams"] = fams
    Q = queries[queries.caller == "DefenseFinder"]
    qids = sorted(Q.qid.tolist(), key=lambda x: -queries.n_members[x])
    rows = [r for ch in bench.pool_map(s3.grid_task, qids, a.threads, chunksize=2) for r in ch]
    G = pd.DataFrame(rows, columns=["qid", "fam", "w", "wx", "tau", "ap", "ap_strict", "n_rel", "n_hits"])
    G = G.merge(queries[["qid", "genome_code", "split", "n_members", "type"]], on="qid")
    G.to_parquet(out / "grid.parquet", index=False)
    log(f"grid: {len(qids):,} queries, {len(G):,} rows, {time.time() - t0:.0f}s")
    G = G[G.n_rel > 0].copy()
    fold_of = s3.folds_of(G[G.fam == "gcZ"])
    ref_sel = json.load(open(OUT3REF / "selection3.json"))
    folds_same = fold_of == {int(k_): v for k_, v in ref_sel["fold_of"].items()}
    G["fold"] = G.genome_code.map(fold_of)
    chosen = {f: [s3.choose(G[(G.fam == f) & (G.fold != k_)]) for k_ in range(5)] for f in fams}
    odd = {f: s3.choose(G[(G.fam == f) & (G.split == "odd")]) for f in fams}
    held = []
    for f in fams:
        for k_ in range(5):
            c = chosen[f][k_]
            held.append(G[(G.fam == f) & (G.fold == k_) & (G.w == c["w"]) & (G.wx == c["wx"]) & (G.tau == c["tau"])])
    Hd = pd.concat(held, ignore_index=True)
    Hd.to_parquet(out / "cv_heldout.parquet", index=False)
    X = Hd.pivot_table(index=["qid", "genome_code", "split", "n_members", "fold"], columns="fam",
                       values=["ap", "ap_strict"]).reset_index()
    X.columns = ["_".join(c).strip("_") for c in X.columns]
    # reproduction gate
    ref = pd.read_parquet(OUT3REF / "cv_heldout.parquet")
    ref = ref[ref.fam == "M1z"].set_index("qid")
    xz = X.set_index("qid")
    ids = xz.index.intersection(ref.index)
    dq = (xz.loc[ids, "ap_gcZ"] - ref.loc[ids, "ap"]).abs()
    apz = float(X.ap_gcZ.mean())
    gate = dict(Z_heldout_ap=apz, report3_M1z=REF_M1Z_AP, abs_diff=abs(apz - REF_M1Z_AP), cond_a=bool(abs(apz - REF_M1Z_AP) <= 0.002),
                queries=int(len(X)), matched_queries=int(len(ids)), share_query_absdiff_le_0_01=float((dq <= 0.01).mean()),
                max_query_absdiff=float(dq.max()), share_query_identical=float((dq == 0).mean()),
                cond_b=bool((dq <= 0.01).mean() >= 0.99 and len(ids) == len(X)), folds_same_as_report3=bool(folds_same),
                Z_chosen=chosen["gcZ"], Z_odd=odd["gcZ"])
    gate["pass"] = bool(gate["cond_a"] and gate["cond_b"])
    dump(gate, out / "gate.json")
    log("gate", json.dumps(gate, default=jd))
    res = dict(n_queries=int(len(X)), n_genomes=int(X.genome_code.nunique()), chosen=chosen, odd=odd, tables={}, paired={},
               per_fold={}, gate=gate)
    strata = {"all": np.ones(len(X), bool), "even": (X.split == "even").to_numpy(), "odd": (X.split == "odd").to_numpy(),
              "min3": (X.n_members >= 3).to_numpy()}
    for nm, msk in strata.items():
        S = X[msk]
        res["tables"][nm] = {f: dict(queries=int(S[f"ap_{f}"].notna().sum()), ap=boot(S[f"ap_{f}"], S.genome_code),
                                     ap_strict=boot(S[f"ap_strict_{f}"], S.genome_code)) for f in fams}
        res["paired"][nm] = {f"{f}-gcZ": {c: boot(S[f"{c}_{f}"] - S[f"{c}_gcZ"], S.genome_code) for c in ("ap", "ap_strict")}
                             for f in fams if f != "gcZ"}
    res["per_fold"] = {f: {int(k_): float((g[f"ap_{f}"] - g["ap_gcZ"]).mean()) for k_, g in X.groupby("fold")}
                       for f in fams if f != "gcZ"}
    res["criterion"] = {}
    for f in fams:
        if f == "gcZ":
            continue
        d = res["paired"]["all"][f"{f}-gcZ"]["ap"]
        res["criterion"][f] = dict(diff=d, helps=bool(d[1] > 0 and d[0] >= 0.010))
    res["seconds_grid"] = round(time.time() - t0, 1)
    dump(res, out / "cv.json")
    dump(dict(fold_of={str(k_): v for k_, v in fold_of.items()}, chosen=chosen, odd=odd), out / "selection.json")
    log("criterion", json.dumps(res["criterion"], default=jd))
    # frozen-configuration scans (S1): DF / TXSScan / CRISPRCasTyper even queries with n_rel > 0
    rel = bench.CTX["rel"]
    E = queries[(queries.split == "even")]
    E = E[[len(rel[int(q)][0]) > 0 for q in E.qid]]
    tasks = [(int(q), f, odd[f]["w"], odd[f]["wx"], odd[f]["tau"]) for f in fams for q in E.qid]
    tasks.sort(key=lambda t: -int(queries.n_members[t[0]]))
    # ap_task returns (qid, ap, ap_strict, n_rel) without the family and imap_unordered loses task order: run per family
    parts = []
    for f in fams:
        tf = [t for t in tasks if t[1] == f]
        for q, ap, aps, nr in bench.pool_map(s3.ap_task, tf, a.threads, chunksize=4):
            parts.append(dict(qid=int(q), fam=f, ap=ap, ap_strict=aps, n_rel=int(nr)))
    F = pd.DataFrame(parts).merge(queries[["qid", "caller", "genome_code", "split", "n_members"]], on="qid")
    F.to_parquet(out / "frozen_scan.parquet", index=False)
    log(f"grid stage done {time.time() - t0:.0f}s")


# =========================================================================== null runs (S1, CRISPRCasTyper)
def nullx(a):
    t0 = time.time()
    out = GC / "out"
    gate = json.load(open(out / "gate.json"))
    i, n = (int(x) for x in a.part.split("/"))
    if not gate["pass"]:
        log("reproduction gate failed: no null runs")
        return
    sel = json.load(open(out / "selection.json"))
    fams = register()
    db, callers, queries = s3.setup(a, fams)
    rel = bench.CTX["rel"]
    Q = queries[(queries.caller == "CRISPRCasTyper") & (queries.split == "even")]
    Q = Q[[len(rel[int(q)][0]) > 0 for q in Q.qid]]
    tasks = [(int(q), f, sel["odd"][f]["w"], sel["odd"][f]["wx"], sel["odd"][f]["tau"]) for f in fams for q in Q.qid]
    tasks.sort(key=lambda t: (-int(queries.n_members[t[0]]), t))
    tasks = tasks[i::n]
    log(f"nullx part {i}/{n}: {len(tasks):,} runs ({len(Q):,} queries x {len(fams)} arms overall)")
    res = bench.pool_map(s3.final_task, tasks, a.threads, chunksize=1)
    M = pd.DataFrame([m for m, _, _ in res]).merge(queries, on="qid")
    M.to_parquet(out / f"nullx_{i}.parquet", index=False)
    log(f"nullx part {i} done {time.time() - t0:.0f}s")


# =========================================================================== final tables
def final(a):
    t0 = time.time()
    out = GC / "out"
    resd = GC / "results"
    resd.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in [GC / "lists" / x / "info.json" for x in ARMS] + [GC / "lists" / x / "readouts.npz" for x in ARMS]
               + [out / "gate.json", out / "cv.json", out / "frozen_scan.parquet"] if not p.exists()]
    if missing:
        (GC / "STATUS").write_text("FAILED missing outputs: " + " ".join(missing) + "\n")
        log("missing", missing)
        return
    gate = json.load(open(out / "gate.json"))
    cv = json.load(open(out / "cv.json"))
    summ = dict(gate=gate, primary=dict(criterion=cv["criterion"], tables=cv["tables"], paired=cv["paired"],
                                        per_fold=cv["per_fold"], chosen=cv["chosen"], odd=cv["odd"]))
    fams = ["gc" + x for x in ARMS]
    # S1
    F = pd.read_parquet(out / "frozen_scan.parquet")
    s1 = {}
    for c in bench.CALLERS:
        Fc = F[F.caller == c]
        P = Fc.pivot_table(index=["qid", "genome_code"], columns="fam", values=["ap", "ap_strict"]).reset_index()
        P.columns = ["_".join(x).strip("_") for x in P.columns]
        s1[c] = dict(queries=int(len(P)), tables={f: {m: boot(P[f"{m}_{f}"], P.genome_code) for m in ("ap", "ap_strict")} for f in fams},
                     paired={f"{f}-gcZ": {m: boot(P[f"{m}_{f}"] - P[f"{m}_gcZ"], P.genome_code) for m in ("ap", "ap_strict")}
                             for f in fams if f != "gcZ"})
    parts = sorted(out.glob("nullx_*.parquet"))
    if gate["pass"]:
        if len(parts) < a.nparts:
            (GC / "STATUS").write_text(f"FAILED null runs incomplete: {len(parts)} of {a.nparts} parts\n")
            return
        M = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        M = M[M.n_rel > 0]
        cols = ["ap", "ap_strict", "recall_E0.01", "n_wrong_E0.01", "n_unlabelled_E0.01"]
        P = M.pivot_table(index=["qid", "genome_code"], columns="fam", values=cols).reset_index()
        P.columns = ["_".join(x).strip("_") for x in P.columns]
        s1["CRISPRCasTyper_null200"] = dict(queries=int(len(P)),
                                            tables={f: {m: boot(P[f"{m}_{f}"], P.genome_code) for m in cols} for f in fams},
                                            paired={f"{f}-gcZ": {m: boot(P[f"{m}_{f}"] - P[f"{m}_gcZ"], P.genome_code) for m in cols}
                                                    for f in fams if f != "gcZ"})
        # Z against REPORT3's M1z null runs (same queries, same seeds)
        X = pd.read_parquet(OUT3REF / "final_x_metrics.parquet")
        X = X[(X.fam == "M1z") & (X.caller == "CRISPRCasTyper") & (X.n_rel > 0)].set_index("qid")
        z = M[M.fam == "gcZ"].set_index("qid")
        ids = z.index.intersection(X.index)
        s1["CRISPRCasTyper_null200"]["Z_vs_report3_M1z"] = {m: dict(z=float(z.loc[ids, m].mean()), report3=float(X.loc[ids, m].mean()),
                                                                     max_abs=float((z.loc[ids, m] - X.loc[ids, m]).abs().max()))
                                                             for m in cols} | {"queries": int(len(ids))}
    summ["S1"] = s1
    # S2
    infos = {x: json.load(open(GC / "lists" / x / "info.json")) for x in ARMS}
    summ["S2"] = {x: {k_: v for k_, v in infos[x].items() if k_.startswith("share") or k_ in ("removal", "zca10")} for x in ARMS}
    summ["Z_vs_stored_head_lists"] = infos["Z"].get("vs_stored_head_lists")
    summ["timing"] = {x: dict(transform=infos[x]["transform_seconds"], lists=infos[x]["list_seconds"], arm=infos[x]["arm_seconds"])
                      for x in ARMS}
    # S3 / S4
    den = np.load(GC / "aux_den.npz")
    rd = {x: dict(np.load(GC / "lists" / x / "readouts.npz")) for x in ARMS}
    qgc = den["qgc"]
    sets = {"all": np.ones(len(qgc), bool), **{c: den[f"q_{c}"] for c in bench.CALLERS}}
    tgen, tlen = den["tgen"].astype(float), den["tlen"].astype(float)
    s3r = {}
    for nm, msk in sets.items():
        d = {"genes": int(msk.sum()), "genes_with_T": int((msk & (tlen > 0)).sum())}
        per = {}
        for x in ARMS:
            r = rd[x]
            mt = msk & (tgen > 0)
            per[x] = dict(zlen=r["zlen"][msk].astype(float), zgrp=r["zgrp"][msk].astype(float),
                          zrec=(r["zhit"][mt] / tgen[mt]), k1grp=r["k1grp"][msk].astype(float),
                          k1rec=(r["k1hit"][msk & (tlen > 0)] / tlen[msk & (tlen > 0)]))
        g = {"zlen": qgc[msk], "zgrp": qgc[msk], "zrec": qgc[msk & (tgen > 0)], "k1grp": qgc[msk], "k1rec": qgc[msk & (tlen > 0)]}
        d["tables"] = {x: {k_: boot(v, g[k_]) for k_, v in per[x].items()} for x in ARMS}
        d["medians"] = {x: {k_: float(np.median(v)) if len(v) else None for k_, v in per[x].items()} for x in ARMS}
        d["paired"] = {f"{x}-Z": {k_: boot(per[x][k_] - per["Z"][k_], g[k_]) for k_ in per[x]} for x in ARMS if x != "Z"}
        s3r[nm] = d
    summ["S3"] = s3r
    cas = den["qcas"] & (den["famout"] >= 1)
    fo = den["famout"].astype(float)
    s4 = dict(genes=int(cas.sum()), cas_query_genes=int(den["qcas"].sum()), per_arm={}, paired={})
    vals = {}
    for x in ARMS:
        r = rd[x]
        br = np.where(r["k1first"] > 0, r["k1first"], K1 + 1)[cas].astype(float)
        share = (r["k1fam"][cas] / np.minimum(K1, fo[cas]))
        ocas = r["k1ocas"][cas].astype(float)
        vals[x] = dict(best_rank=br, capped_family_share=share, other_family_cas=ocas)
        s4["per_arm"][x] = dict(best_rank_median=float(np.median(br)), best_rank_mean=float(br.mean()),
                                best_rank_q75=float(np.quantile(br, .75)), best_rank_q90=float(np.quantile(br, .9)),
                                share_absent=float((br > K1).mean()), share_rank_gt_10=float((br > 10).mean()),
                                capped_family_share_mean=float(share.mean()), other_family_cas_mean=float(ocas.mean()),
                                other_family_cas_median=float(np.median(ocas)))
    for x in ARMS:
        if x != "Z":
            s4["paired"][f"{x}-Z"] = {k_: boot(vals[x][k_] - vals["Z"][k_], qgc[cas]) for k_ in vals[x]}
    summ["S4"] = s4
    summ["seconds"] = round(time.time() - t0, 1)
    dump(summ, resd / "summary.json")
    write_tables(summ, resd / "tables.txt")
    crit = summ["primary"]["criterion"]
    line = ("DONE gate " + ("pass" if gate["pass"] else "FAIL") + "; " +
            "; ".join(f"{f[2:]} {v['diff'][0]:+.4f} [{v['diff'][1]:+.4f}, {v['diff'][2]:+.4f}] helps={'yes' if v['helps'] else 'no'}"
                      for f, v in crit.items()))
    if not gate["pass"]:
        line = "FAILED reproduction gate: " + json.dumps({k_: gate[k_] for k_ in ("Z_heldout_ap", "share_query_absdiff_le_0_01", "folds_same_as_report3")})
    (GC / "STATUS").write_text(line + "\n")
    log(line)


def write_tables(s, path):
    def f(v, n=3):
        return f"{v[0]:+.{n}f} [{v[1]:+.{n}f}, {v[2]:+.{n}f}]" if isinstance(v, (list, tuple)) else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

    def m(v, n=3):
        return f"{v[0]:.{n}f}" if isinstance(v, (list, tuple)) else f(v, n)
    L = []
    g = s["gate"]
    L.append(f"GATE pass={g['pass']} Z AP {g['Z_heldout_ap']:.5f} (REPORT3 {g['report3_M1z']:.5f}); queries |dAP|<=0.01 "
             f"{g['share_query_absdiff_le_0_01']:.4f}, identical {g['share_query_identical']:.4f}, max {g['max_query_absdiff']:.4f}; "
             f"folds same {g['folds_same_as_report3']}")
    L.append("PRIMARY (5-fold CV, 2,447 DefenseFinder queries)")
    T = s["primary"]["tables"]["all"]
    for fam in T:
        cr = s["primary"]["criterion"].get(fam)
        L.append(f"  {fam[2:]:4s} AP {m(T[fam]['ap'], 4)} strict {m(T[fam]['ap_strict'], 4)}" +
                 (f"  d {f(cr['diff'], 4)} helps={cr['helps']}" if cr else ""))
    for st in ("even", "odd", "min3"):
        L.append(f"  stratum {st}: " + "; ".join(f"{k[2:]} {f(v['ap'], 4)}" for k, v in s["primary"]["paired"][st].items()))
    L.append("  per fold: " + json.dumps({k[2:]: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in s["primary"]["per_fold"].items()}))
    L.append("  chosen (w, wx, tau) per fold / odd: " + "; ".join(
        f"{k[2:]} {[(c['wx'], c['tau']) for c in v]} / {(s['primary']['odd'][k]['wx'], s['primary']['odd'][k]['tau'])}"
        for k, v in s["primary"]["chosen"].items()))
    L.append("S1 frozen configurations, even genomes")
    for c, d in s["S1"].items():
        cols = list(next(iter(d["tables"].values())).keys())
        L.append(f"  {c} ({d['queries']} queries)")
        for fam, t in d["tables"].items():
            L.append(f"    {fam[2:]:4s} " + " ".join(f"{k}={m(t[k])}" for k in cols))
        for k, t in d["paired"].items():
            L.append(f"    {k[2:]:8s} " + " ".join(f"{kk}={f(t[kk])}" for kk in cols))
        if "Z_vs_report3_M1z" in d:
            L.append("    Z vs REPORT3 M1z: " + json.dumps(d["Z_vs_report3_M1z"]))
    L.append("S2 variance shares (total / residual row-mode [/ residual uid-mode])")
    for x, d in s["S2"].items():
        for st in ("share_input", "share_pre_whitening", "share_final"):
            if st in d:
                v = d[st]
                L.append(f"  {x:4s} {st:20s} total {100 * v['total_share']:.3f}% (exp {100 * v['total_expected']:.3f}%) resid "
                         f"{100 * v['resid_share_row']:.3f}% (exp {100 * v['resid_expected_row']:.3f}%)" +
                         (f" uid {100 * v['resid_share_uid']:.3f}%" if "resid_share_uid" in v else ""))
        if "removal" in d:
            L.append(f"       removal {json.dumps(d['removal'])}")
    L.append("Z vs stored head lists: " + json.dumps(s["Z_vs_stored_head_lists"]))
    L.append("S3 neighbor composition (means; zlen/zgrp/zrec = z>=10.6 per-genome lists; k1grp/k1rec = top 1,000 outside own group)")
    for nm, d in s["S3"].items():
        L.append(f"  {nm} genes {d['genes']} with T {d['genes_with_T']}")
        for x, t in d["tables"].items():
            L.append(f"    {x:4s} " + " ".join(f"{k}={m(v)}" for k, v in t.items()))
        for k, t in d["paired"].items():
            L.append(f"    {k:6s} " + " ".join(f"{kk}={f(v)}" for kk, v in t.items()))
    d = s["S4"]
    L.append(f"S4 Cas genes (family outside own group) {d['genes']} of {d['cas_query_genes']}")
    for x, t in d["per_arm"].items():
        L.append(f"    {x:4s} " + " ".join(f"{k}={v:.3f}" for k, v in t.items()))
    for k, t in d["paired"].items():
        L.append(f"    {k:6s} " + " ".join(f"{kk}={f(v)}" for kk, v in t.items()))
    L.append("timing " + json.dumps(s["timing"]))
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "lists", "grid", "nullx", "final"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    ap.add_argument("--arms", default="Z")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--nparts", type=int, default=3)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    {"smoke": smoke, "lists": lists, "grid": grid, "nullx": nullx, "final": final}[a.stage](a)
    log(f"done {a.stage} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
