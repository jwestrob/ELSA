#!/usr/bin/env python
"""Context-kernel families on DPANN (workstream context_kernel).

K(a, a') = k(x_a, x_a') * sum_{d,d'} k(x_{a+d}, x_{a'+d'}): similarity of two genes times the agreement
of their genomic neighbourhoods. With a dot-product k the neighbourhood term equals c_a . c_a' with
c_a the sum of the neighbours' vectors, so context support costs one extra dot product per edge.

edges     cross-genome gene pairs from the unique-sequence kNN (identical sequences at cosine 1, cap per
          gene), context scores (dense, matched) for w = 1, 3, 5, a per-edge null (partner replaced by a
          random gene of its genome), FDR-1% support, the dense identity check, gold-pair diagnostics
families  Leiden communities on candidate / supported edges with weight sim, or sim x ctx -> token arms
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
KNN = RD / "lab/knn/knn__esmpp_small__zca10__k50__hnsw.npz"
WORK = RD / "context_kernel"
WS = (1, 3, 5)
VARIANTS = [f"{f}_w{w}" for f in ("dense", "matched") for w in WS]      # column order of score matrices
THR = 0.14                                                              # random-pair cosine q99 (zca10)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_tokens import load_run, transform, stamp, rss_gb             # noqa: E402


def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_universe():
    import pyarrow.parquet as pq
    t = pq.read_table(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "position_index"])
    gid = t["gid"].to_numpy()
    n = len(gid)
    assert (gid == np.arange(n)).all()
    gcode = t["genome_code"].to_numpy().astype(np.int64)
    contig = t["contig_code"].to_numpy()
    pos = t["position_index"].to_numpy()
    brk = np.flatnonzero(np.diff(contig) != 0) + 1
    st, en = np.r_[0, brk], np.r_[brk, n]
    assert len(st) == len(np.unique(contig)), "contigs are not contiguous gid ranges"
    cstart, cend = np.repeat(st, en - st), np.repeat(en, en - st)
    assert ((pos - pos[cstart]) == (np.arange(n) - cstart)).all(), "position_index is not consecutive within contigs"
    assert (np.diff(gcode) >= 0).all()
    gn = np.bincount(gcode)
    gstart = np.r_[0, np.cumsum(gn)[:-1]]
    return n, gcode, cstart.astype(np.int64), cend.astype(np.int64), gstart.astype(np.int64), gn.astype(np.int64)


# ----------------------------------------------------------------------------------------- kernels
@njit(parallel=True, cache=True)
def expand(uq, gcode, I, D, indptr, members, cap, out_b, out_s, n_avail):
    """Directed partner lists: own unique sequence first (cosine 1), then kNN order; other genomes only."""
    n = len(uq)
    for a in prange(n):
        U = uq[a]
        if U < 0:
            continue
        ga = gcode[a]
        k = 0
        avail = 0
        for t in range(indptr[U], indptr[U + 1]):
            b = members[t]
            if gcode[b] != ga:
                avail += 1
                if k < cap:
                    out_b[a, k] = b
                    out_s[a, k] = 1.0
                    k += 1
        for j in range(I.shape[1]):
            V = I[U, j]
            if V < 0:
                continue
            s = D[U, j]
            for t in range(indptr[V], indptr[V + 1]):
                b = members[t]
                if gcode[b] != ga:
                    avail += 1
                    if k < cap:
                        out_b[a, k] = b
                        out_s[a, k] = s
                        k += 1
        n_avail[a] = avail


@njit(parallel=True, fastmath=True, cache=True)
def context_kernel(ea, eb, X, uq, cstart, cend, thr, out):
    """out[e] = dense_w1, dense_w3, dense_w5, matched_w1, matched_w3, matched_w5 for the pair (ea[e], eb[e]).
    Edges must be sorted by ea (the a-side neighbour block is cached)."""
    E = len(ea)
    d = X.shape[1]
    Wm = 5
    BS = 4096
    nblk = (E + BS - 1) // BS
    for blk in prange(nblk):
        A = np.empty((2 * Wm, d), np.float32)
        Aoff = np.empty(2 * Wm, np.int64)
        Bu = np.empty(2 * Wm, np.int64)
        Boff = np.empty(2 * Wm, np.int64)
        G = np.empty((2 * Wm, 2 * Wm), np.float32)
        last = -1
        na = 0
        for e in range(blk * BS, min(E, (blk + 1) * BS)):
            a = ea[e]
            b = eb[e]
            if a != last:
                na = 0
                for g in range(max(cstart[a], a - Wm), min(cend[a], a + Wm + 1)):
                    if g == a or uq[g] < 0:
                        continue
                    A[na, :] = X[uq[g]]
                    Aoff[na] = abs(g - a)
                    na += 1
                last = a
            nb = 0
            for g in range(max(cstart[b], b - Wm), min(cend[b], b + Wm + 1)):
                if g == b or uq[g] < 0:
                    continue
                Bu[nb] = uq[g]
                Boff[nb] = abs(g - b)
                nb += 1
            for i in range(na):
                for j in range(nb):
                    xb = X[Bu[j]]
                    s = np.float32(0.0)
                    for k in range(d):
                        s += A[i, k] * xb[k]
                    G[i, j] = s
            for wi in range(3):
                w = 1 if wi == 0 else (3 if wi == 1 else 5)
                ca = 0
                cb = 0
                for i in range(na):
                    if Aoff[i] <= w:
                        ca += 1
                for j in range(nb):
                    if Boff[j] <= w:
                        cb += 1
                dense = 0.0
                m1 = 0.0
                m2 = 0.0
                if ca > 0 and cb > 0:
                    for i in range(na):
                        if Aoff[i] > w:
                            continue
                        best = 0.0
                        for j in range(nb):
                            if Boff[j] > w:
                                continue
                            dense += G[i, j]
                            r = G[i, j] - thr
                            if r > best:
                                best = r
                        m1 += best
                    for j in range(nb):
                        if Boff[j] > w:
                            continue
                        best = 0.0
                        for i in range(na):
                            if Aoff[i] > w:
                                continue
                            r = G[i, j] - thr
                            if r > best:
                                best = r
                        m2 += best
                    dense /= ca * cb
                out[e, wi] = dense
                out[e, 3 + wi] = 0.5 * (m1 + m2)


def neighbours(g, w, uq, cstart, cend):
    return [x for x in range(max(cstart[g], g - w), min(cend[g], g + w + 1)) if x != g and uq[x] >= 0]


def identity_check(ea, eb, out, X, uq, cstart, cend, rng, m=5000):
    """dense via summed neighbour vectors (c_a . c_b) against the explicit double sum, float64; and the
    kernel's float32 output against both."""
    idx = rng.choice(len(ea), m, replace=False)
    err_c, err_k, err_m = 0.0, 0.0, 0.0
    for e in idx:
        a, b = int(ea[e]), int(eb[e])
        for wi, w in enumerate(WS):
            na, nb = neighbours(a, w, uq, cstart, cend), neighbours(b, w, uq, cstart, cend)
            if not na or not nb:
                continue
            A = X[uq[na]].astype(np.float64)
            B = X[uq[nb]].astype(np.float64)
            Gm = A @ B.T
            explicit = Gm.sum() / (len(na) * len(nb))
            cform = A.sum(0) @ B.sum(0) / (len(na) * len(nb))
            R = np.maximum(Gm - THR, 0)
            matched = 0.5 * (R.max(1).sum() + R.max(0).sum())
            err_c = max(err_c, abs(cform - explicit))
            err_k = max(err_k, abs(out[e, wi] - explicit))
            err_m = max(err_m, abs(out[e, 3 + wi] - matched))
    return dict(edges=int(m), max_abs_c_form_vs_double_sum=err_c, max_abs_kernel_dense_vs_double_sum=err_k,
                max_abs_kernel_matched_vs_numpy=err_m)


def fdr_threshold(obs, nul, alpha=0.01, grid=200_000):
    """Smallest t > 0 with N_null(>= t) / N_obs(>= t) <= alpha, on a grid of observed quantiles."""
    o = np.sort(obs)
    nl = np.sort(nul)
    pos = o[o > 0]
    if len(pos) == 0:
        return dict(t=None)
    cand = np.unique(pos[np.linspace(0, len(pos) - 1, min(grid, len(pos))).astype(np.int64)])
    n_obs = len(o) - np.searchsorted(o, cand, "left")
    n_nul = len(nl) - np.searchsorted(nl, cand, "left")
    fdr = n_nul / np.maximum(n_obs, 1)
    ok = np.flatnonzero(fdr <= alpha)
    res = dict(obs_positive=int(len(pos)), null_positive=int((nl > 0).sum()),
               obs_q50_q90_q99=np.quantile(o, [0.5, 0.9, 0.99]).round(4).tolist(),
               null_q50_q90_q99=np.quantile(nl, [0.5, 0.9, 0.99]).round(4).tolist())
    if len(ok) == 0:
        return dict(res, t=None)
    i = ok[0]
    return dict(res, t=float(cand[i]), fdr_at_t=float(fdr[i]), n_obs_ge_t=int(n_obs[i]), n_null_ge_t=int(n_nul[i]))


# ----------------------------------------------------------------------------------------- AUROC
def auc_groups(score, y):
    """Sort order and tie-group starts for repeated weighted AUROC."""
    o = np.argsort(score, kind="mergesort")
    s = score[o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return o, starts, y[o].astype(bool)


def auc_w(o, starts, yo, w):
    wo = w[o]
    P = np.add.reduceat(np.where(yo, wo, 0.0), starts)
    N = np.add.reduceat(np.where(yo, 0.0, wo), starts)
    cumN = np.cumsum(N) - N
    tp, tn = P.sum(), N.sum()
    return float((P * (cumN + 0.5 * N)).sum() / (tp * tn)) if tp > 0 and tn > 0 else float("nan")


def auc_table(scores: dict, y, clus, rng, boot=200, pairs=()):
    """AUROC of each score, and genome-bootstrap 95% intervals for the differences in `pairs`."""
    ones = np.ones(len(y))
    prep = {k: auc_groups(v, y) for k, v in scores.items()}
    res = {k: auc_w(*prep[k], ones) for k in scores}
    if pairs and len(y) > 0:
        cu, inv = np.unique(clus, return_inverse=True)
        diffs = {p: [] for p in pairs}
        for _ in range(boot):
            cnt = np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu)).astype(np.float64)
            w = cnt[inv]
            cache = {}
            for p in pairs:
                for k in p:
                    if k not in cache:
                        cache[k] = auc_w(*prep[k], w)
                diffs[p].append(cache[p[0]] - cache[p[1]])
        for p in pairs:
            d = np.array(diffs[p])
            res[f"{p[0]} - {p[1]}"] = [res[p[0]] - res[p[1]], float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]
    return res


# ----------------------------------------------------------------------------------------- edges
def cmd_edges(a):
    import numba
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    out = WORK / "edges"
    out.mkdir(parents=True, exist_ok=True)
    man = dict(code_sha256=code_sha(), slurm_job=os.environ.get("SLURM_JOB_ID"), threads=numba.get_num_threads(),
               cap=a.cap, seed=a.seed, knn=str(KNN), emb=str(EMB), thr=THR, variants=VARIANTS)
    n, gcode, cstart, cend, gstart, gn = load_universe()
    stamp("universe", n, "genes", len(gn), "genomes", f"{time.time() - t0:.0f}s")

    X, u, valid, cnt = load_run(EMB)
    info = transform(X, valid, cnt, "zca10")
    assert len(u) == n
    uq = np.where((u >= 0) & valid[np.maximum(u, 0)], u, -1).astype(np.int64)
    man.update(transform=info, unique_sequences=int(len(X)), genes_with_vector=int((uq >= 0).sum()))
    stamp("vectors", X.shape, "genes with vector", int((uq >= 0).sum()), f"{time.time() - t0:.0f}s rss {rss_gb()}")

    z = np.load(KNN, allow_pickle=True)
    I, D = z["I"], z["D"]
    assert I.shape[0] == len(X)
    # the transform must reproduce the space of the kNN file
    q = rng.choice(np.flatnonzero(I[:, 0] >= 0), 20000, replace=False)
    rec = np.einsum("ij,ij->i", X[q], X[I[q, 0]])
    man["knn_cosine_recompute_max_abs_err"] = float(np.abs(rec - D[q, 0]).max())
    stamp("kNN", I.shape, "recomputed rank-1 cosine max abs err", man["knn_cosine_recompute_max_abs_err"])
    assert man["knn_cosine_recompute_max_abs_err"] < 1e-3

    # unique sequence -> its genes, in random order (ties among copies broken at random)
    g_ok = np.flatnonzero(uq >= 0)
    order = g_ok[np.lexsort((rng.random(len(g_ok)), uq[g_ok]))]
    indptr = np.r_[0, np.cumsum(np.bincount(uq[g_ok], minlength=len(X)))].astype(np.int64)
    members = order.astype(np.int64)

    t1 = time.time()
    out_b = np.full((n, a.cap), -1, dtype=np.int32)
    out_s = np.zeros((n, a.cap), dtype=np.float32)
    n_avail = np.zeros(n, dtype=np.int64)
    expand(uq, gcode, I, D, indptr, members, a.cap, out_b, out_s, n_avail)
    kept = (out_b >= 0).sum(1)
    has = uq >= 0
    capinfo = dict(
        seconds=round(time.time() - t1, 1),
        genes_with_vector=int(has.sum()),
        genes_with_any_partner=int((kept > 0).sum()),
        directed_edges=int(kept.sum()),
        cap_binds_genes=int((n_avail > a.cap).sum()),
        cap_binds_fraction_of_genes_with_vector=float((n_avail[has] > a.cap).mean()),
        available_partners_q10_q50_q90_q99=np.quantile(n_avail[has], [0.1, 0.5, 0.9, 0.99]).tolist(),
        available_partners_max=int(n_avail.max()),
        candidates_dropped_by_cap=int(np.maximum(n_avail - a.cap, 0).sum()),
        kept_partners_q10_q50_q90=np.quantile(kept[has], [0.1, 0.5, 0.9]).tolist(),
        directed_identical_sequence_fraction=float((out_s[out_b >= 0] >= 0.99999).mean()),
        sim_at_last_kept_partner_q10_q50_q90_when_cap_binds=np.quantile(out_s[n_avail > a.cap, a.cap - 1], [0.1, 0.5, 0.9]).tolist()
        if (n_avail > a.cap).any() else None,
    )
    del I, D, z
    stamp("expand", capinfo, f"rss {rss_gb()}")

    # undirected union
    t1 = time.time()
    rows = np.repeat(np.arange(n, dtype=np.int64), a.cap)
    b = out_b.ravel().astype(np.int64)
    s = out_s.ravel()
    m = b >= 0
    rows, b, s = rows[m], b[m], s[m]
    del out_b, out_s, m
    lo, hi = np.minimum(rows, b), np.maximum(rows, b)
    key = lo * n + hi
    del rows, b, lo, hi
    o = np.argsort(key, kind="stable")
    key, s = key[o], s[o]
    del o
    first = np.r_[True, key[1:] != key[:-1]]
    mult = np.diff(np.r_[np.flatnonzero(first), len(key)])
    ukey, usim = key[first], s[first]
    del key, s
    ea = (ukey // n).astype(np.int32)
    eb = (ukey % n).astype(np.int32)
    del ukey
    E = len(ea)
    capinfo.update(undirected_edges=int(E), mutual_edges=int((mult == 2).sum()), union_seconds=round(time.time() - t1, 1),
                   genes_with_edge=int(len(np.unique(np.r_[ea, eb]))),
                   sim_q10_q50_q90=np.quantile(usim, [0.1, 0.5, 0.9]).round(4).tolist())
    stamp("undirected", E, "mutual", capinfo["mutual_edges"], f"rss {rss_gb()}")
    man["edges"] = capinfo

    # context scores, observed and null
    t1 = time.time()
    ctx = np.zeros((E, 6), dtype=np.float32)
    context_kernel(ea, eb, X, uq, cstart, cend, np.float32(THR), ctx)
    man["context_seconds"] = round(time.time() - t1, 1)
    stamp("context", f"{man['context_seconds']}s")
    t1 = time.time()
    gb = gcode[eb]
    nb_ = gstart[gb] + (rng.random(E) * gn[gb]).astype(np.int64)
    same = nb_ == eb
    nb_[same] = gstart[gb[same]] + (nb_[same] - gstart[gb[same]] + 1) % gn[gb[same]]
    del gb, same
    nul = np.zeros((E, 6), dtype=np.float32)
    context_kernel(ea, nb_.astype(np.int32), X, uq, cstart, cend, np.float32(THR), nul)
    man["null_seconds"] = round(time.time() - t1, 1)
    stamp("null", f"{man['null_seconds']}s")
    man["identity_check"] = identity_check(ea, eb, ctx, X, uq, cstart, cend, rng)
    stamp("identity", man["identity_check"])
    del X

    # support at empirical FDR 1%
    sup = {}
    gene_cov = {}
    for v, name in enumerate(VARIANTS):
        r = fdr_threshold(ctx[:, v], nul[:, v], a.alpha)
        msk = ctx[:, v] >= r["t"] if r["t"] is not None else np.zeros(E, bool)
        sup[name] = msk
        gs = np.unique(np.r_[ea[msk], eb[msk]])
        r.update(supported_edges=int(msk.sum()), supported_fraction=float(msk.mean()), genes_with_supported_edge=int(len(gs)),
                 genes_with_supported_edge_fraction_of_genes_with_edge=float(len(gs) / capinfo["genes_with_edge"]),
                 obs_zero_fraction=float((ctx[:, v] <= 0).mean()), null_zero_fraction=float((nul[:, v] <= 0).mean()))
        gene_cov[name] = r
        stamp("support", name, r)
    man["support"] = gene_cov

    # save edge arrays
    np.save(out / "ea.npy", ea)
    np.save(out / "eb.npy", eb)
    np.save(out / "sim.npy", usim)
    np.save(out / "mutual.npy", (mult == 2))
    np.save(out / "ctx.npy", ctx)
    np.save(out / "null.npy", nul)
    np.save(out / "null_partner.npy", nb_.astype(np.int32))
    np.save(out / "support.npy", np.column_stack([sup[v] for v in VARIANTS]))
    stamp("saved edges", f"{time.time() - t0:.0f}s rss {rss_gb()}")

    # gold diagnostics
    man["gold"] = gold_diagnostics(ea, eb, usim, ctx, sup, gcode, n, rng)
    tune = man["gold"]["pooled"]["tune"]
    best = {}
    for form in ("dense", "matched"):
        cands = [v for v in VARIANTS if v.startswith(form)]
        best[form] = max(cands, key=lambda v: tune[f"simctx:{v}"])
    primary = max(best.values(), key=lambda v: tune[f"simctx:{v}"])
    secondary = [v for v in best.values() if v != primary][0]
    man["selection"] = dict(rule="highest tune pooled AUROC of sim x ctx", primary=primary, secondary=secondary,
                            tune_auroc={v: tune[f"simctx:{v}"] for v in VARIANTS}, tune_auroc_sim=tune["sim"])
    stamp("selection", man["selection"])
    man["seconds"] = round(time.time() - t0, 1)
    man["peak_rss_gb"] = rss_gb()
    (out / "edges.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    stamp("end", f"{time.time() - t0:.0f}s")


def gold_diagnostics(ea, eb, sim, ctx, sup, gcode, n, rng):
    import pandas as pd
    mem = pd.read_parquet(PKG / "gold/members.parquet", columns=["instance_id", "caller", "gid"])
    ins = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "system_type"])
    mem = mem.merge(ins, on="instance_id")[["gid", "caller", "system_type"]].drop_duplicates()
    isgold = np.zeros(n, bool)
    isgold[mem.gid.to_numpy()] = True
    res = dict(gold_genes=int(isgold.sum()))
    idx = np.flatnonzero(isgold[ea] & isgold[eb])
    df = pd.DataFrame({"e": idx, "a": ea[idx].astype(np.int64), "b": eb[idx].astype(np.int64)})
    ma = df.merge(mem.rename(columns={"gid": "a", "system_type": "ta"}), on="a")
    mab = ma.merge(mem.rename(columns={"gid": "b", "system_type": "tb"}), on=["b", "caller"])
    mab["same"] = mab.ta == mab.tb
    per = mab.groupby(["e", "caller"], as_index=False).same.any()
    pooled = per.groupby("e").same.any()
    lab = pd.Series(False, index=idx)
    lab.loc[pooled.index] = pooled.to_numpy()

    def scores(e):
        d = {"sim": sim[e].astype(np.float64)}
        for v, name in enumerate(VARIANTS):
            c = ctx[e, v].astype(np.float64)
            d[f"ctx:{name}"] = c
            d[f"simctx:{name}"] = sim[e] * np.maximum(c, 0)
        return d

    def splits(e):
        ga, gb = gcode[ea[e]], gcode[eb[e]]
        return {"all": np.ones(len(e), bool), "tune": (ga % 2 == 1) & (gb % 2 == 1), "test": (ga % 2 == 0) & (gb % 2 == 0)}

    def block(e, y, boot):
        out = {}
        sc = scores(e)
        sp = splits(e)
        for k, m in sp.items():
            pairs = [(f"simctx:{v}", "sim") for v in VARIANTS] if (boot and k == "test") else []
            t = auc_table({kk: vv[m] for kk, vv in sc.items()}, y[m], gcode[ea[e[m]]], rng, pairs=pairs)
            t.update(n=int(m.sum()), positives=int(y[m].sum()))
            for v in VARIANTS:
                t[f"supported_fraction_pos:{v}"] = float(sup[v][e[m]][y[m]].mean()) if y[m].any() else None
                t[f"supported_fraction_neg:{v}"] = float(sup[v][e[m]][~y[m]].mean()) if (~y[m]).any() else None
            out[k] = t
        return out

    e_all = lab.index.to_numpy()
    res["pooled"] = block(e_all, lab.to_numpy(), True)
    res["per_caller"] = {}
    for caller, g in per.groupby("caller"):
        e = g.e.to_numpy()
        res["per_caller"][caller] = block(e, g.same.to_numpy(), True)
    # TXSScan members against same-type TXSScan partners (positive) and non-gold partners (negative)
    tx = mem[mem.caller == "TXSScan"]
    istx = np.zeros(n, bool)
    istx[tx.gid.to_numpy()] = True
    e_ng = np.flatnonzero((istx[ea] & ~isgold[eb]) | (istx[eb] & ~isgold[ea]))
    t_same = per[(per.caller == "TXSScan") & per.same].e.to_numpy()
    e_tx = np.r_[t_same, e_ng]
    y_tx = np.r_[np.ones(len(t_same), bool), np.zeros(len(e_ng), bool)]
    res["txss_vs_nongold"] = block(e_tx, y_tx, True)
    # where the edges of each caller's members go, and how many members keep a supported edge
    has_edge = np.zeros(n, bool)
    has_edge[ea] = True
    has_edge[eb] = True
    has_sup = {}
    for v in VARIANTS:
        h = np.zeros(n, bool)
        h[ea[sup[v]]] = True
        h[eb[sup[v]]] = True
        has_sup[v] = h
    for caller in ("TXSScan", "DefenseFinder", "CRISPRCasTyper", "nongold"):
        isc = ~isgold if caller == "nongold" else np.zeros(n, bool)
        if caller != "nongold":
            isc[mem[mem.caller == caller].gid.to_numpy()] = True
        ec = np.flatnonzero(isc[ea] | isc[eb])
        other_gold = isgold[ea[ec]] & isgold[eb[ec]]
        d = dict(genes=int(isc.sum()), genes_with_edge=int((isc & has_edge).sum()), edges=int(len(ec)),
                 to_gold_fraction=float(other_gold.mean()))
        for v in VARIANTS:
            ms = sup[v][ec]
            d[f"supported_fraction:{v}"] = float(ms.mean())
            d[f"to_gold_fraction_among_supported:{v}"] = float(other_gold[ms].mean()) if ms.any() else None
            d[f"genes_with_supported_edge_fraction:{v}"] = float((isc & has_sup[v]).sum() / max(1, (isc & has_edge).sum()))
        res[f"edges_of_{caller}_members"] = d
    stamp("gold", json.dumps({k: res["pooled"][k].get("n") for k in res["pooled"]}))
    return res


# ----------------------------------------------------------------------------------------- families
_G = None
_W = {}


def _leiden(task):
    import random
    import igraph as ig
    wkind, res, seed = task
    t = time.time()
    random.seed(seed)
    ig.set_random_number_generator(random)
    part = _G.community_leiden(objective_function="modularity", weights=_W[wkind], resolution=res, n_iterations=-1)
    memb = np.asarray(part.membership, dtype=np.int64)
    return wkind, res, memb, round(time.time() - t, 1)


def write_tokens(arm, tokens, man):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = RD / "engine/dpann_tokens"
    assert arm.startswith("ctx__")
    tokens = np.asarray(tokens, dtype=np.int64)
    tmp = out / f".{arm}.parquet.tmp{os.getpid()}"
    pq.write_table(pa.table({"gid": np.arange(len(tokens), dtype=np.int64), "token": tokens}), tmp, compression="zstd")
    os.replace(tmp, out / f"{arm}.parquet")
    tj = out / f".{arm}.json.tmp{os.getpid()}"
    tj.write_text(json.dumps(man, indent=1, default=float) + "\n")
    os.replace(tj, out / f"{arm}.json")


def token_stats(memb):
    sizes = np.bincount(np.unique(memb, return_inverse=True)[1])
    return dict(n_tokens=int(len(sizes)), singleton_tokens=int((sizes == 1).sum()), token_size_max=int(sizes.max()),
                genes_in_tokens_ge2=int(sizes[sizes >= 2].sum()), token_size_q50_q90_q99=np.quantile(sizes, [0.5, 0.9, 0.99]).tolist())


def cmd_families(a):
    import multiprocessing as mp
    import igraph as ig
    global _G, _W
    t0 = time.time()
    ed = WORK / "edges"
    man = json.loads((ed / "edges.json").read_text())
    sel = man["selection"]
    variant = sel["primary"] if a.variant == "primary" else (sel["secondary"] if a.variant == "secondary" else a.variant)
    v = VARIANTS.index(variant)
    ea = np.load(ed / "ea.npy").astype(np.int64)
    eb = np.load(ed / "eb.npy").astype(np.int64)
    sim = np.load(ed / "sim.npy").astype(np.float64)
    ctx = np.load(ed / "ctx.npy", mmap_mode="r")[:, v].astype(np.float64)
    n = len(np.load(EMB / "uid_of_row.npy", mmap_mode="r"))
    if a.edge_set == "sup":
        m = np.load(ed / "support.npy", mmap_mode="r")[:, v].astype(bool)
        ea, eb, sim, ctx = ea[m], eb[m], sim[m], ctx[m]
    elif a.edge_set == "permtop":
        # control: the same number of edges as the true-order support set, ranked by the context score
        # computed after permuting gene order within genomes; weights use that permuted-order score
        cp = np.load(ed / "ctx_perm.npy", mmap_mode="r")[:, v].astype(np.float64)
        K = int(np.load(ed / "support.npy", mmap_mode="r")[:, v].sum())
        idx = np.sort(np.argpartition(-cp, K - 1)[:K])
        ea, eb, sim, ctx = ea[idx], eb[idx], sim[idx], cp[idx]
        stamp(f"permtop: {K:,} edges, permuted-order {variant} score >= {cp[idx].min():.4f}")
    sim = np.maximum(sim, 0)
    _W = {"sim": sim, "simctx": sim * np.maximum(ctx, 0)}
    kinds = a.weights.split(",")
    for k in list(_W):
        if k not in kinds:
            del _W[k]
    for k in kinds:          # zero-weight edges carry no information for modularity; igraph requires weights >= 0
        assert (_W[k] >= 0).all()
    t1 = time.time()
    _G = ig.Graph(n=n, edges=np.column_stack([ea, eb]), directed=False)
    del ea, eb
    stamp(f"graph {a.edge_set} ({variant}): {n:,} nodes {_G.ecount():,} edges, built {time.time() - t1:.0f}s rss {rss_gb()}")
    res_grid = [float(r) for r in a.res.split(",")]
    tasks = [(k, r, a.seed) for k in kinds for r in res_grid]
    procs = min(len(tasks), a.procs)
    fam = WORK / "families"
    fam.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("fork").Pool(procs) as pool:
        for wkind, r, memb, secs in pool.imap_unordered(_leiden, tasks):
            name = f"{variant}_{wkind}_{a.edge_set}" if not (wkind == "sim" and a.edge_set == "all") else f"sim_all"
            tag = f"{name}_r{r:g}"
            np.save(fam / f"{tag}.npy", memb.astype(np.int32))
            st = token_stats(memb)
            row = dict(tag=tag, family=name, edge_set=a.edge_set, weights=wkind, variant=variant, resolution=r, edges=int(_G.ecount()),
                       leiden_seconds=secs, **st)
            rows.append(row)
            stamp("leiden", json.dumps(row))
    tsv = fam / f"grid_{a.edge_set}_{variant}_{'-'.join(kinds)}.tsv"
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(tsv, sep="\t", index=False, mode="a", header=not tsv.exists())
    stamp("end", f"{time.time() - t0:.0f}s rss {rss_gb()}")


def cmd_permctx(a):
    """Context scores of the candidate edges with gene order permuted within each genome (contig sizes
    kept): every neighbour becomes a random gene of the same genome. Writes edges/ctx_perm.npy."""
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    ed = WORK / "edges"
    n, gcode, cstart, cend, gstart, gn = load_universe()
    X, u, valid, cnt = load_run(EMB)
    transform(X, valid, cnt, "zca10")
    uq = np.where((u >= 0) & valid[np.maximum(u, 0)], u, -1).astype(np.int64)
    perm = np.lexsort((rng.random(n), gcode))            # slot -> gene, within genome
    assert (gcode[perm] == gcode).all()
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    uq_slot = uq[perm]
    ea = np.load(ed / "ea.npy")
    eb = np.load(ed / "eb.npy")
    sa, sb = inv[ea], inv[eb]
    o = np.argsort(sa, kind="stable")
    cp = np.zeros((len(ea), 6), dtype=np.float32)
    t1 = time.time()
    tmp = np.zeros((len(ea), 6), dtype=np.float32)
    context_kernel(sa[o].astype(np.int32), sb[o].astype(np.int32), X, uq_slot, cstart, cend, np.float32(THR), tmp)
    cp[o] = tmp
    del tmp
    np.save(ed / "ctx_perm.npy", cp)
    ctx = np.load(ed / "ctx.npy", mmap_mode="r")
    sup = np.load(ed / "support.npy", mmap_mode="r")
    man = dict(seed=a.seed, seconds_kernel=round(time.time() - t1, 1), code_sha256=code_sha(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    for v, name in enumerate(VARIANTS):
        c = cp[:, v]
        K = int(sup[:, v].sum())
        thr = float(np.partition(c, len(c) - K)[len(c) - K]) if K > 0 else None
        man[name] = dict(perm_q50_q90_q99=np.quantile(c, [0.5, 0.9, 0.99]).round(4).tolist(),
                         true_q50_q90_q99=np.quantile(ctx[:, v], [0.5, 0.9, 0.99]).round(4).tolist(),
                         true_support_threshold=None, K=K, perm_score_at_K=thr,
                         perm_edges_above_true_threshold=None)
        stamp("permctx", name, man[name])
    em = json.loads((ed / "edges.json").read_text())
    for v, name in enumerate(VARIANTS):
        t = em["support"][name].get("t")
        man[name]["true_support_threshold"] = t
        man[name]["perm_edges_above_true_threshold"] = int((cp[:, v] >= t).sum()) if t is not None else None
    (ed / "permctx.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    stamp("end", f"{time.time() - t0:.0f}s rss {rss_gb()}")


def cmd_arms(a):
    """Write token arms for chosen grid entries (tags) to $RD/engine/dpann_tokens/ctx__esmpp_small__<tag>.parquet."""
    import pandas as pd
    fam = WORK / "families"
    grid = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(fam.glob("grid_*.tsv"))])
    grid = grid.drop_duplicates("tag", keep="last").set_index("tag")
    em = json.loads((WORK / "edges/edges.json").read_text())
    for tag in a.tags.split(","):
        memb = np.load(fam / f"{tag}.npy").astype(np.int64)
        _, tok = np.unique(memb, return_inverse=True)
        arm = f"ctx__esmpp_small__{tag.replace('.', 'p')}"
        row = grid.loc[tag].to_dict()
        man = dict(arm=arm, source="context_kernel", rows=int(len(tok)), n_assigned=int(len(tok)), **row,
                   knn=str(KNN), emb=str(EMB), cap=em["cap"], thr=THR, objective="modularity", n_iterations=-1,
                   support=em["support"].get(row["variant"]), selection=em["selection"], code_sha256=code_sha(),
                   written=dt.datetime.now().isoformat(timespec="seconds"))
        write_tokens(arm, tok, man)
        with open(WORK / "INDEX.tsv", "a") as fh:
            fh.write(f"{arm}\t{row['n_tokens']}\t{row['singleton_tokens']}\t{row['token_size_max']}\t{dt.date.today()}\n")
        stamp("arm", arm, row["n_tokens"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("edges")
    p.add_argument("--cap", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(fn=cmd_edges)
    p = sub.add_parser("families")
    p.add_argument("--edge-set", choices=("all", "sup", "permtop"), required=True)
    p.add_argument("--variant", default="primary", help="primary | secondary | a variant name")
    p.add_argument("--weights", default="sim,simctx")
    p.add_argument("--res", required=True, help="comma-separated modularity resolutions")
    p.add_argument("--procs", type=int, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(fn=cmd_families)
    p = sub.add_parser("permctx")
    p.add_argument("--seed", type=int, default=18)
    p.set_defaults(fn=cmd_permctx)
    p = sub.add_parser("arms")
    p.add_argument("--tags", required=True)
    p.set_defaults(fn=cmd_arms)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
