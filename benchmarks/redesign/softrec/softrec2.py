#!/usr/bin/env python
"""softrec round 2 (workstream softrec2): statistics that keep variable pairs, and the evidence for validated families.

usage
  softrec2.py stats --threads T
      zca10 lists (genes outside the genus; depth caps 50/200/500/1000 genes; no cosine floor), observed pass and three
      gene-order nulls (seeds 1000 + r, as the engine) with an extended kernel. Per variant and instance: L, G, E exactly as
      softrec.py, plus rank-weighted lineage sums W_j = sum over other genera of w_j(m_g), m_g = min over the genus's
      supporting pairs (a', b') of max(tail rank of a' in N(a), tail rank of b' in N(b)); w_j decreasing (exp(-(t-1)/r0)
      or r0 / (r0 + t - 1)). Writes $RD/softrec/r2/stats/{obs,null0,null1,null2}.npz and stats.json.
      Checks: L, G, E equal round 1 (runs/zca10/observed_stats.npz) on the shared variants; python recount of W.
  softrec2.py cands --threads T [--names n1,n2] [--edges]
      rule selection per candidate statistic with the engine's procedure at the instance level (rules with FDR <= budget,
      top 12 by observed count, most genes covered); per candidate $RD/softrec/r2/cand/<name>/:
        sidx.npy (significant instance ids, engine order (d, left)), value.npy (support value L at the deep depth),
        cap.npy (list depth of the supporting pairs), b.npy (boundary support), arg.npy (position in sidx of the
        instance setting b; -1 none), rules.tsv, cand.json;
      with --edges (lists in memory): validated edges = every (a, a') and (b, b') of a supporting pair (a', b') of a
      significant instance (a' in N(a), b' in N(b), a' != b', within w on one contig, at the instance's depth),
      collapsed to unique sequences, undirected: edges_a.npy, edges_b.npy (int32 unique ids, a < b), edges_w.npy
      (cosine, float32); fam_cc.npy (connected components, unique level).
Candidate statistics (all at the instance level; null = within-genome gene-order shuffle, every order-dependent step after):
  cnt<K>        count rule L >= m at depth K, FDR 0.01
  pois<K>       lineage-aware Poisson P(X >= G | E), G and E over genomes outside the genus, P <= 1e-e & L >= 2|3, FDR 0.01
  tt<K>         two-tier: count at depth K (FDR 0.005) OR Poisson at depth 50 (FDR 0.005); union FDR reported
  rw<K>_<w>     rank-weighted: W_w >= t & L >= 2 at depth K, FDR 0.01
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
import softrec as SR  # noqa: E402
from embed_tokens import stamp, rss_gb  # noqa: E402

RD = SR.RD
R2 = RD / "softrec/r2"
DEPTHS = (50, 200, 500, 1000)
WEIGHTS = (("e50", 0, 50.0), ("e100", 0, 100.0), ("e200", 0, 200.0), ("e400", 0, 400.0), ("h20", 1, 20.0), ("h100", 1, 100.0))
W_T = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 260, 330, 400)
COUNT_M = SR.COUNT_M
POIS_E = SR.POIS_E
FDR = 0.01
W = 3


def all_candidates():
    c = {}
    for K in (200, 500, 1000):
        c[f"cnt{K}"] = dict(kind="count", K=K)
        c[f"tt{K}"] = dict(kind="twotier", K=K)
        for wn, _, _ in WEIGHTS:
            c[f"rw{K}_{wn}"] = dict(kind="rankw", K=K, w=wn)
    for K in (50, 200):
        c[f"pois{K}"] = dict(kind="poisson", K=K)
    return c


# ============================================================================== extended kernel
@njit(parallel=True, cache=True)
def soft_pass2(n, w, V, gene_at, cid, lstart, lend, nbs, nbg, tail, genome_gene, lin_of_genome, cap, capmax, excl,
               inst_off, nd_arr, n_genome, n_lin, ngenes_g, wtype, wpar, out_L, out_G, out_E, out_W, nchunks):
    """softrec.soft_pass (rank_mode 0, skip 0: ranked lists that already exclude the gene's genus) with the graded sum
    replaced by rank-weighted lineage sums. Per lineage the best supporting pair minimises max(tail_a, tail_b)."""
    J = len(wtype)
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
        lmin = np.zeros((V, n_lin), np.int64)
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
                        for j in range(J):
                            out_W[v, j, inst] = 0.0
                    continue
                f0 = lstart[b]
                f1 = lend[b]
                if f1 > f0 + capmax[b]:
                    f1 = f0 + capmax[b]
                ntb = 0
                for e in range(f0, f1):
                    G = nbg[e]
                    r = e - f0
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
                    mab = np.int64(tail[e0 + ra])
                    tb_ = np.int64(tail[e])
                    if tb_ > mab:
                        mab = tb_
                    for v in range(V):
                        if excl[v, inst]:
                            continue
                        if ra < cap[v, a] and r < cap[v, b]:
                            if gst[v, G] != inst:
                                gst[v, G] = inst
                                cntG[v] += 1
                            if lg != la:
                                if lst[v, lg] != inst:
                                    lst[v, lg] = inst
                                    lmin[v, lg] = mab
                                    linlist[v, cntL[v]] = lg
                                    cntL[v] += 1
                                elif mab < lmin[v, lg]:
                                    lmin[v, lg] = mab
                for v in range(V):
                    if excl[v, inst]:
                        out_L[v, inst] = -1
                        out_G[v, inst] = -1
                        out_E[v, inst] = 0.0
                        for j in range(J):
                            out_W[v, j, inst] = 0.0
                        continue
                    esum = 0.0
                    for k in range(ntb):
                        G = tch_b[k]
                        if ca[v, G] > 0 and cb[v, G] > 0:
                            x = 2.0 * w * ca[v, G] * cb[v, G] / ngenes_g[G]
                            esum += x if x < 1.0 else 1.0
                    out_L[v, inst] = cntL[v]
                    out_G[v, inst] = cntG[v]
                    out_E[v, inst] = esum
                    for j in range(J):
                        ssum = 0.0
                        for k in range(cntL[v]):
                            x = lmin[v, linlist[v, k]] - 1.0
                            if wtype[j] == 0:
                                ssum += math.exp(-x / wpar[j])
                            else:
                                ssum += wpar[j] / (wpar[j] + x)
                        out_W[v, j, inst] = ssum
                for k in range(ntb):
                    G = tch_b[k]
                    for v in range(V):
                        cb[v, G] = 0
            for k in range(nta):
                G = tch_a[k]
                for v in range(V):
                    ca[v, G] = 0


def run_pass2(u, L, gene_at, slot_of, nd, off, n_inst, threads, label):
    t0 = time.time()
    nbs = np.empty_like(L.nb)
    SR.map_slots(L.nb, slot_of.astype(np.int32), nbs)
    excl = np.zeros((L.V, n_inst), np.bool_)
    SR.exclusion(u.n, L.V, gene_at, nd, off, L.excl_mode, L.tok, L.X, L.uq, L.capcos, excl)
    wtype = np.array([x[1] for x in WEIGHTS], np.int64)
    wpar = np.array([x[2] for x in WEIGHTS], np.float64)
    oL = np.zeros((L.V, n_inst), np.int16)
    oG = np.zeros((L.V, n_inst), np.int16)
    oE = np.zeros((L.V, n_inst), np.float32)
    oW = np.zeros((L.V, len(WEIGHTS), n_inst), np.float32)
    soft_pass2(u.n, W, L.V, gene_at, u.cid, L.lstart, L.lend, nbs, L.nbg, L.tail, u.gc, u.lin_of_genome, L.cap,
               L.capmax, excl, off, nd, u.ng, u.nl, u.n_g, wtype, wpar, oL, oG, oE, oW, threads * 8)
    del nbs
    left = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    own = np.minimum(1.0, 2.0 * W / u.n_g[u.gc[gene_at[left]]]).astype(np.float32)
    stamp(f"pass {label}: {time.time() - t0:.0f}s; excluded per variant",
          {v: int((oL[i] < 0).sum()) for i, v in enumerate(L.variants)}, f"rss {rss_gb()}")
    return dict(L=oL, G=oG, E=oE, W=oW, own=own)


def brute_W(u, L, P, nd, off, rng, m=300):
    """Pure-python recount of W (and L) for random observed instances (observed order: slot == gene)."""
    n_inst = P["L"].shape[1]
    left_all = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    pick = rng.choice(n_inst, m, replace=False)
    bad, maxd = 0, 0.0
    for inst in pick:
        i = int(left_all[inst])
        a, b = i, i + int(inst - off[i] + 1)
        la = int(u.lin[a])
        for v in range(L.V):
            if P["L"][v, inst] < 0:
                continue
            Ka, Kb = int(L.cap[v, a]), int(L.cap[v, b])
            ea = [(int(L.nb[L.lstart[a] + r]), int(L.tail[L.lstart[a] + r])) for r in range(min(Ka, L.lend[a] - L.lstart[a]))]
            eb = [(int(L.nb[L.lstart[b] + r]), int(L.tail[L.lstart[b] + r])) for r in range(min(Kb, L.lend[b] - L.lstart[b]))]
            best_a = {}
            for x, t in ea:
                best_a[x] = min(best_a.get(x, 1 << 30), t)
            lins = {}
            for y, t2 in eb:
                ta = 1 << 30
                for dd in range(-W, W + 1):
                    x = y + dd
                    if dd != 0 and 0 <= x < u.n and x in best_a and u.cid[x] == u.cid[y]:
                        ta = min(ta, best_a[x])
                if ta < (1 << 30):
                    lg = int(u.lin[y])
                    if lg != la:
                        lins[lg] = min(lins.get(lg, 1 << 30), max(ta, t2))
            for j, (_, wt, r0) in enumerate(WEIGHTS):
                want = sum((math.exp(-(mm - 1) / r0) if wt == 0 else r0 / (r0 + mm - 1)) for mm in lins.values())
                got = float(P["W"][v, j, inst])
                maxd = max(maxd, abs(got - want) / max(1.0, want))
                if abs(got - want) > 1e-3 * max(1.0, want) or len(lins) != int(P["L"][v, inst]):
                    bad += 1
    return dict(instances=int(m), mismatches=bad, max_rel_diff=maxd)


class ListArgs:
    def __init__(self, depths):
        self.source = "zca10"
        self.skip = "genus"
        self.ku = 2000
        self.depths = list(depths)
        self.floors = [-2.0]


def cmd_stats(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    out = R2 / "stats"
    out.mkdir(parents=True, exist_ok=True)
    u = SR.Universe()
    nd, off, left, d = u.instances(W)
    n_inst = len(left)
    L = SR.Lists(u, ListArgs(DEPTHS))
    man = dict(code_sha256=SR.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               host=os.uname().nodename, threads=a.threads, variants=L.variants, weights=[list(x) for x in WEIGHTS],
               lists=L.info, n_instances=int(n_inst), seconds_lists=round(time.time() - t0, 1))
    ident = np.arange(u.n, dtype=np.int64)
    t1 = time.time()
    obs = run_pass2(u, L, ident, ident, nd, off, n_inst, a.threads, "observed")
    man["seconds_observed_pass"] = round(time.time() - t1, 1)
    man["brute_W_observed"] = brute_W(u, L, obs, nd, off, np.random.default_rng(17), 300)
    stamp("brute W", man["brute_W_observed"])
    r1 = np.load(RD / "softrec/runs/zca10/observed_stats.npz")
    r1v = list(r1["variants"])
    chk = {}
    for i, v in enumerate(L.variants):
        if v in r1v:
            j = r1v.index(v)
            chk[v] = dict(L_equal=bool(np.array_equal(r1["L"][j], obs["L"][i])), G_equal=bool(np.array_equal(r1["G"][j], obs["G"][i])),
                          E_maxdiff=float(np.abs(r1["E"][j] - obs["E"][i]).max()))
    man["round1_check"] = chk
    stamp("round-1 check", chk)
    np.savez(out / "obs.npz", L=obs["L"], G=obs["G"], E=obs["E"], W=obs["W"], own=obs["own"], variants=np.array(L.variants))
    for r in range(a.perms):
        t1 = time.time()
        ga, so = SR.null_order(u, r)
        p = run_pass2(u, L, ga, so, nd, off, n_inst, a.threads, f"null {r}")
        np.savez(out / f"null{r}.npz", L=p["L"], G=p["G"], E=p["E"], W=p["W"], own=p["own"], variants=np.array(L.variants))
        man[f"seconds_null_{r}"] = round(time.time() - t1, 1)
        del p
    man["seconds"] = round(time.time() - t0, 1)
    man["peak_rss_gb"] = rss_gb()
    (out / "stats.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    stamp("end stats", man["seconds"], f"rss {rss_gb()}")
    if a.then_cands:
        a.lists = L
        a.universe = u
        cmd_cands(a)


# ============================================================================== rule selection
def load_pass(fn):
    z = np.load(fn)
    return dict(L=z["L"], G=z["G"], E=z["E"], W=z["W"], own=z["own"], variants=list(z["variants"]))


def derived(P, v):
    L = P["L"][v].astype(np.int64)
    ok = L >= 0
    Lt = np.where(ok, L + 1, -1)
    G = P["G"][v].astype(np.int64) + 1
    E = P["E"][v].astype(np.float64) + P["own"]
    return ok, Lt, G, E


def rule_masks(kind, P, v, j=None, only=None):
    ok, Lt, G, E = derived(P, v)
    if kind == "count":
        for m in COUNT_M:
            name = f"L>={m}"
            if only is None or name == only:
                yield name, Lt >= m
    elif kind == "poisson":
        logP = SR.logp_poisson(G, E, ok & (Lt >= 2))
        for e in POIS_E:
            for m in (2, 3):
                name = f"P<=1e-{e}&L>={m}"
                if only is None or name == only:
                    yield name, (Lt >= m) & (logP <= -e * math.log(10))
    elif kind == "rankw":
        Wv = P["W"][v, j]
        for t in W_T:
            name = f"W>={t:g}"
            if only is None or name == only:
                yield name, (Lt >= 2) & (Wv >= t)


def covered(n, mask, left, d):
    g = np.zeros(n, bool)
    g[left[mask]] = True
    g[left[mask] + d[mask]] = True
    return g


def select(n, obs, nulls, kind, v, j, fdr, left, d):
    nc = [{name: int(m.sum()) for name, m in rule_masks(kind, p, v, j)} for p in nulls]
    rows, masks = [], {}
    for name, m in rule_masks(kind, obs, v, j):
        n_obs = int(m.sum())
        n_null = float(np.mean([c[name] for c in nc]))
        rows.append(dict(rule=name, observed=n_obs, null_mean=n_null, fdr=(n_null / n_obs if n_obs else np.nan)))
        masks[name] = m
    rules = pd.DataFrame(rows)
    cand = rules[(rules.fdr <= fdr) & (rules.observed > 0)].sort_values("observed", ascending=False).head(12)
    best, best_cov = None, -1
    for name in cand.rule:
        cov = int(covered(n, masks[name], left, d).sum())
        rules.loc[rules.rule == name, "genes_covered"] = cov
        if cov > best_cov:
            best, best_cov = name, cov
    null_best = [next(rule_masks(kind, p, v, j, only=best))[1] for p in nulls] if best else None
    return rules, best, (masks[best] if best else None), null_best


def build_candidate(name, spec, u, obs, nulls, left, d, variants):
    K = spec["K"]
    vK = variants.index(f"K{K}_fnone")
    okK, LtK, _, _ = derived(obs, vK)
    info = dict(name=name, **spec)
    if spec["kind"] in ("count", "poisson", "rankw"):
        j = [x[0] for x in WEIGHTS].index(spec["w"]) if spec["kind"] == "rankw" else None
        rules, best, sig, nullm = select(u.n, obs, nulls, spec["kind"], vK, j, FDR, left, d)
        if best is None:
            return None
        rules["tier"] = spec["kind"]
        value = LtK
        cap = np.full(len(left), K, np.int16)
        row = rules[rules.rule == best].iloc[0]
        info.update(rule=best, fdr=float(row.fdr), null_mean=float(row.null_mean))
    else:
        r1, b1, m1, n1 = select(u.n, obs, nulls, "count", vK, None, FDR / 2, left, d)
        v50 = variants.index("K50_fnone")
        r2, b2, m2, n2 = select(u.n, obs, nulls, "poisson", v50, None, FDR / 2, left, d)
        r1["tier"] = f"count@{K}"
        r2["tier"] = "poisson@50"
        rules = pd.concat([r1, r2], ignore_index=True)
        sig = m1 | m2
        null_union = float(np.mean([(a_ | b_).sum() for a_, b_ in zip(n1, n2)]))
        _, Lt50, _, _ = derived(obs, v50)
        value = np.maximum(LtK, Lt50)
        cap = np.where(m1, K, 50).astype(np.int16)
        info.update(rule=f"[{b1}]@{K} OR [{b2}]@50", rule_deep=b1, rule_shallow=b2, deep_only=int((m1 & ~m2).sum()),
                    shallow_only=int((m2 & ~m1).sum()), both=int((m1 & m2).sum()), null_mean=null_union,
                    fdr=null_union / max(int(sig.sum()), 1),
                    fdr_deep=float(r1[r1.rule == b1].fdr.iloc[0]), fdr_shallow=float(r2[r2.rule == b2].fdr.iloc[0]))
    sidx = np.flatnonzero(sig)
    sidx = sidx[np.lexsort((left[sidx], d[sidx]))]
    val = value[sidx].astype(np.float32)
    assert (val >= 2).all(), "significant instance with support < 2"
    b, arg = SR.boundary_support_argmax(left[sidx], d[sidx], val, np.arange(len(sidx), dtype=np.int64), u.n)
    genes = covered(u.n, sig, left, d)
    info.update(significant_instances=int(len(sidx)), genes_covered=int(genes.sum()), genes_covered_frac=float(genes.mean()),
                supported_boundaries=int((b > 0).sum()),
                L_q10_q50_q90=np.quantile(val, [0.1, 0.5, 0.9]).tolist())
    return dict(info=info, rules=rules, sidx=sidx, value=val.astype(np.int16), cap=cap[sidx], b=b, arg=arg)


# ============================================================================== validated edges
@njit(parallel=True, cache=True)
def validated(sel_left, sel_d, sel_cap, n, w, cid, lstart, lend, nb, valid, nchunks):
    """valid[e] = True for every list entry e (of a or b) that belongs to a supporting pair of a selected instance."""
    m = len(sel_left)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        mst = np.full(n, -1, np.int64)
        bst = np.full(n, -1, np.int64)
        for q in range(lo, hi):
            a = sel_left[q]
            b = a + sel_d[q]
            K = sel_cap[q]
            ea0 = lstart[a]
            ea1 = lend[a] if lend[a] < ea0 + K else ea0 + K
            eb0 = lstart[b]
            eb1 = lend[b] if lend[b] < eb0 + K else eb0 + K
            for e in range(ea0, ea1):
                s = nb[e]
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                for x in range(ulo, uhi):
                    if x != s and cid[x] == c:
                        mst[x] = q
            for e in range(eb0, eb1):
                s = nb[e]
                bst[s] = q
                if mst[s] == q:
                    valid[e] = True
            for e in range(ea0, ea1):
                s = nb[e]
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                for x in range(ulo, uhi):
                    if x != s and cid[x] == c and bst[x] == q:
                        valid[e] = True
                        break


@njit(cache=True)
def row_counts(valid, Kmax, n):
    c = np.zeros(n + 1, np.int64)
    for a in range(n):
        k = 0
        for r in range(Kmax):
            if valid[a * Kmax + r]:
                k += 1
        c[a + 1] = k
    return c


@njit(parallel=True, cache=True)
def fill_keys(valid, nb, uq, Kmax, n, off, U, out):
    for a in prange(n):
        k = off[a]
        ua = uq[a]
        for r in range(Kmax):
            if valid[a * Kmax + r]:
                ub = uq[nb[a * Kmax + r]]
                if ua < ub:
                    out[k] = ua * U + ub
                elif ub < ua:
                    out[k] = ub * U + ua
                else:
                    out[k] = -1
                k += 1


@njit(parallel=True, cache=True)
def edge_cos(ea, eb, X, out):
    dim = X.shape[1]
    for i in prange(len(ea)):
        s = 0.0
        x = ea[i]
        y = eb[i]
        for k in range(dim):
            s += X[x, k] * X[y, k]
        out[i] = s


@njit(cache=True)
def union_find_cc(n, ea, eb):
    parent = np.arange(n)
    for i in range(len(ea)):
        x = ea[i]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = eb[i]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x != y:
            if x < y:
                parent[y] = x
            else:
                parent[x] = y
    for i in range(n):
        x = i
        while parent[x] != x:
            x = parent[x]
        parent[i] = x
    return parent


def extract_edges(u, L, sidx, left, d, cap, threads, out):
    t0 = time.time()
    Kmax = int(L.nb.size // u.n)
    assert np.array_equal(L.lstart, np.arange(u.n, dtype=np.int64) * Kmax), "lists are not a dense n x Kmax block"
    valid = np.zeros(u.n * Kmax, np.bool_)
    validated(left[sidx].astype(np.int64), d[sidx].astype(np.int64), cap.astype(np.int64), u.n, W, u.cid, L.lstart, L.lend,
              L.nb, valid, threads * 8)
    rc = row_counts(valid, Kmax, u.n)
    off = np.cumsum(rc)
    nv = int(off[-1])
    U = int(L.uq.max()) + 1
    keys = np.empty(nv, np.int64)
    fill_keys(valid, L.nb, L.uq, Kmax, u.n, off[:-1].copy(), U, keys)
    per_gene = rc[1:]
    del valid
    t1 = time.time()
    keys = keys[keys >= 0]
    n_dir = int(len(keys))
    keys = np.unique(keys)
    ea = (keys // U).astype(np.int32)
    eb = (keys % U).astype(np.int32)
    del keys
    wgt = np.empty(len(ea), np.float32)
    edge_cos(ea, eb, L.X, wgt)
    cc = union_find_cc(U, ea.astype(np.int64), eb.astype(np.int64))
    _, cc = np.unique(cc, return_inverse=True)
    sizes = np.bincount(cc)
    deg = np.bincount(ea, minlength=U) + np.bincount(eb, minlength=U)
    np.save(out / "edges_a.npy", ea)
    np.save(out / "edges_b.npy", eb)
    np.save(out / "edges_w.npy", wgt)
    np.save(out / "fam_cc.npy", cc.astype(np.int32))
    np.save(out / "valid_per_gene.npy", per_gene.astype(np.int32))
    info = dict(valid_entries=nv, genes_with_valid_entry=int((per_gene > 0).sum()),
                valid_per_gene_q50_q90_q99_of_genes_with_any=np.quantile(per_gene[per_gene > 0], [0.5, 0.9, 0.99]).tolist() if nv else [],
                directed_unique_level_nonself=n_dir, undirected_unique_edges=int(len(ea)), unique_nodes=U,
                unique_nodes_with_edge=int((deg > 0).sum()), degree_q50_q90_q99_max=np.quantile(deg[deg > 0], [0.5, 0.9, 0.99, 1]).tolist(),
                cosine_q01_q10_q50=np.quantile(wgt, [0.01, 0.1, 0.5]).tolist() if len(wgt) else [],
                cc_components=int(len(sizes)), cc_nontrivial=int((sizes >= 2).sum()), cc_largest=int(sizes.max()),
                cc_largest_share_of_nodes_with_edge=float(sizes.max() / max(1, (deg > 0).sum())),
                cc_size_q50_q90_q99_nontrivial=np.quantile(sizes[sizes >= 2], [0.5, 0.9, 0.99]).tolist(),
                seconds_valid=round(t1 - t0, 1), seconds_total=round(time.time() - t0, 1))
    stamp("edges", json.dumps(info), f"rss {rss_gb()}")
    return info


def cmd_cands(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    u = getattr(a, "universe", None) or SR.Universe()
    nd, off, left, d = u.instances(W)
    obs = load_pass(R2 / "stats/obs.npz")
    nulls = [load_pass(R2 / f"stats/null{r}.npz") for r in range(3)]
    variants = obs["variants"]
    allc = all_candidates()
    names = [x for x in a.names.split(",") if x] if a.names else list(allc)
    edge_names = [x for x in a.edge_names.split(",") if x]
    L = getattr(a, "lists", None)
    if a.edges and L is None:
        L = SR.Lists(u, ListArgs(DEPTHS))
    for name in names:
        t1 = time.time()
        c = build_candidate(name, allc[name], u, obs, nulls, left, d, variants)
        if c is None:
            stamp(name, "no rule passes")
            continue
        out = R2 / "cand" / name
        out.mkdir(parents=True, exist_ok=True)
        for k in ("sidx", "value", "cap", "b", "arg"):
            np.save(out / f"{k}.npy", c[k])
        c["rules"].to_csv(out / "rules.tsv", sep="\t", index=False)
        info = c["info"]
        info["seconds_select"] = round(time.time() - t1, 1)
        if a.edges and (not edge_names or name in edge_names):
            info["edges"] = extract_edges(u, L, c["sidx"], left, d, c["cap"], a.threads, out)
        (out / "cand.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
        stamp("candidate", json.dumps({k: v for k, v in info.items() if k != "edges"}, default=float), f"rss {rss_gb()}")
    stamp("end cands", round(time.time() - t0, 1), f"rss {rss_gb()}")


# ============================================================================== task 4: kNN depth needed per list depth
@njit(parallel=True, cache=True)
def unique_needed(uq, lin, I, mptr, members, depths, out):
    """out[a, di] = number of unique kNN rows of a's sequence consumed (1-based, 0 = own identical copies suffice,
    -1 = not reached within the stored k) until a's list of genes outside its genus reaches depths[di]."""
    n = len(uq)
    Ku = I.shape[1]
    ND = len(depths)
    for a in prange(n):
        for di in range(ND):
            out[a, di] = -1
        U = uq[a]
        if U < 0:
            continue
        la = lin[a]
        k = 0
        for t in range(mptr[U], mptr[U + 1]):
            if lin[members[t]] != la:
                k += 1
        for di in range(ND):
            if out[a, di] < 0 and k >= depths[di]:
                out[a, di] = 0
        for j in range(Ku):
            V = I[U, j]
            if V < 0:
                break
            if V != U:
                for t in range(mptr[V], mptr[V + 1]):
                    if lin[members[t]] != la:
                        k += 1
            for di in range(ND):
                if out[a, di] < 0 and k >= depths[di]:
                    out[a, di] = j + 1
            if k >= depths[ND - 1]:
                break


def cmd_depthcost(a):
    import numba
    numba.set_num_threads(a.threads)
    u = SR.Universe()
    uq = np.load(SR.EMB / "uid_of_row.npy").astype(np.int64)
    I = np.ascontiguousarray(np.load(RD / "softrec/knn/zca10__k2000__I.npy", mmap_mode="r"))
    U = I.shape[0]
    ok = uq >= 0
    order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
    mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=U))].astype(np.int64)
    depths = np.array(DEPTHS, np.int64)
    out = np.zeros((u.n, len(depths)), np.int64)
    unique_needed(uq, u.lin.astype(np.int64), I, mptr, order.astype(np.int64), depths, out)
    res = {}
    for di, K in enumerate(DEPTHS):
        x = out[:, di]
        reached = x >= 0
        res[int(K)] = dict(genes_reached_within_2000=float(reached.mean()),
                           unique_needed_q50_q90_q99_q999=np.quantile(np.where(reached, x, 10 ** 6), [0.5, 0.9, 0.99, 0.999]).tolist(),
                           share_reached_by_k={int(kk): float((reached & (x <= kk)).mean()) for kk in (100, 200, 300, 500, 750, 1000, 1500, 2000)})
    (R2 / "results").mkdir(parents=True, exist_ok=True)
    (R2 / "results/depthcost.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    stamp("depthcost", json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "cands", "depthcost"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--names", default="")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--edge-names", default="", help="with --edges: only these candidates (default all selected)")
    ap.add_argument("--then-cands", action="store_true", help="stats: continue with cands (lists stay in memory)")
    a = ap.parse_args()
    {"stats": cmd_stats, "cands": cmd_cands, "depthcost": cmd_depthcost}[a.cmd](a)


if __name__ == "__main__":
    main()
