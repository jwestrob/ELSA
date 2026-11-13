#!/usr/bin/env python
"""softrec round 4 (ws_softrec4): orientation-aware support (lever 2) and the support edges of the instance graph (lever 1).

usage
  softrec4.py stats --modes 0,1,2 --threads T [--then-cands] [--edges NAMES]
      zca10 lists (genes outside the genus; depth caps 200 / 1000; no floor), observed pass per mode and three gene-order
      nulls (seeds 1000 + r) for every mode > 0. Mode 0 = round 2's statistic (observed pass only, checked equal to
      r2/stats/obs.npz). Writes $RD/softrec/r4/stats/m<mode>_{obs,null0,null1,null2}.npz and stats.json.
  softrec4.py cands --modes 1,2 --threads T
      instance-level rule selection (FDR 0.01, top 12 by count, most genes covered) for o<mode>rw1000_e50 (W_e50 >= t &
      L >= 2, grid W_T4) and o<mode>cnt1000 (L >= m); chance-support summary. -> $RD/softrec/r4/cand/<name>/
  softrec4.py edges --names rw1000_e50,o1rw1000_e50,o2rw1000_e50 --threads T
      instance-graph support edges: per significant instance and per other genus, the best-ranked supporting pair that
      is itself a significant instance (orientation filter of the landscape's mode); top 10 genera by rank m =
      max(tail rank a', tail rank b'). -> $RD/softrec/r4/graph/<name>/sup_{dst,m,cnt}.npy, edges.json
Orientation modes (supporting pair (a', b') of instance (a, b) with a left of b; strands travel with genes in the null):
  0  none (round 2)
  1  strand flag: strand(a') * strand(b') == strand(a) * strand(b)            (the token engine's --oriented)
  2  arrangement: in the frame where a' precedes b' (contig read reversed when b' precedes a', strands flipped) the
     strands of (a', b') equal those of (a, b): co-directional a->b, co-directional b->a, convergent, divergent differ.
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

CODE = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/softrec/code")
sys.path.insert(0, str(CODE))
import softrec as SR  # noqa: E402
import softrec2 as S2  # noqa: E402
from embed_tokens import stamp, rss_gb  # noqa: E402

RD = SR.RD
R2 = RD / "softrec/r2"
R4 = RD / "softrec/r4"
DEPTHS = (200, 1000)
WEIGHTS = S2.WEIGHTS
WNAMES = [x[0] for x in WEIGHTS]
W_T4 = (0.05, 0.1, 0.15, 0.2) + tuple(S2.W_T)
FDR = 0.01
W = 3
KMAX_EDGES = 10


def jdump(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------ lineage unit (one pluggable input)
# The lineage unit enters in exactly one place: a genome_code -> lineage id array (u.lin_of_genome, with u.lin, u.nl and
# u.genes_in_lin derived from it). Default: softrec.Universe's GTDB genus (genomes.parquet 'genus'; genomes without one
# are their own unit). It decides (1) which genes a list excludes (own lineage, expand_knn via u.lin), (2) what L and the
# rank-weighted W count (distinct lineages other than the instance's, soft_pass4) and (3) the per-lineage best support
# kept for the instance graph (support_edges). --lineage FILE replaces it: a .npy int array indexed by genome_code, or a
# parquet with columns genome, lineage; outputs then go to $RD/softrec/r4/lin_<--lin-tag>/.
def load_lineage(path, u):
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        df = pd.read_parquet(path)
        m = dict(zip(df.genome.astype(str), df.lineage))
        arr = np.array([m[g] for g in u.gm.genome.astype(str)])
    assert len(arr) == u.ng, (len(arr), u.ng)
    return pd.factorize(arr)[0].astype(np.int32)


def set_lineage(u, arr):
    u.lin_of_genome = np.asarray(arr, np.int32)
    u.nl = int(u.lin_of_genome.max()) + 1
    u.lin = u.lin_of_genome[u.gc]
    u.genes_in_lin = np.bincount(u.lin, minlength=u.nl).astype(np.int64)
    return u


def universe(lineage=""):
    u = SR.Universe()
    if lineage:
        set_lineage(u, load_lineage(lineage, u))
        stamp("lineage map", lineage, "units", u.nl)
    return u


def strand_of(u):
    g = pd.read_parquet(SR.PKG / "universe/genes.parquet", columns=["gid", "strand"])
    assert (g.gid.to_numpy() == np.arange(u.n)).all()
    s = g.strand.to_numpy().astype(np.int8)
    assert set(np.unique(s).tolist()) <= {-1, 1}
    return s


# ============================================================================== oriented kernel
@njit(parallel=True, cache=True)
def soft_pass4(n, w, V, gene_at, cid, lstart, lend, nbs, nbg, tail, genome_gene, lin_of_genome, cap, capmax, excl,
               inst_off, nd_arr, n_genome, n_lin, ngenes_g, wtype, wpar, sstr, omode, out_L, out_G, out_E, out_W, nchunks):
    """softrec2.soft_pass2 with an orientation filter on supporting pairs. sstr[slot] = strand of the gene in that slot.
    Marks of a's list are kept in two channels; a b'-list probe reads the channel that the orientation rule allows."""
    J = len(wtype)
    for ch in prange(nchunks):
        lo = ch * n // nchunks
        hi = (ch + 1) * n // nchunks
        mstamp = np.full((2, n), -1, np.int32)
        mrank = np.zeros((2, n), np.int32)
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
            sa = np.int64(sstr[i])
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
                chn = 0
                if omode == 1:
                    if sstr[s] < 0:
                        chn = 1
                elif omode == 2:
                    if np.int64(sstr[s]) == sa:
                        ulo = s + 1
                    else:
                        chn = 1
                        uhi = s
                for u in range(ulo, uhi):
                    if u != s and cid[u] == c and mstamp[chn, u] != i:
                        mstamp[chn, u] = i
                        mrank[chn, u] = r
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
                sb = np.int64(sstr[i + d])
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
                    chb = 0
                    if omode == 1:
                        if np.int64(sstr[s]) * sa * sb < 0:
                            chb = 1
                    elif omode == 2:
                        if np.int64(sstr[s]) != sb:
                            chb = 1
                    if mstamp[chb, s] != i:
                        continue
                    ra = mrank[chb, s]
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


def run_pass4(u, L, gene_at, slot_of, sstr_slot, mode, nd, off, n_inst, threads, label):
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
    soft_pass4(u.n, W, L.V, gene_at, u.cid, L.lstart, L.lend, nbs, L.nbg, L.tail, u.gc, u.lin_of_genome, L.cap,
               L.capmax, excl, off, nd, u.ng, u.nl, u.n_g, wtype, wpar, sstr_slot, int(mode), oL, oG, oE, oW, threads * 8)
    del nbs
    left = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    own = np.minimum(1.0, 2.0 * W / u.n_g[u.gc[gene_at[left]]]).astype(np.float32)
    stamp(f"pass mode {mode} {label}: {time.time() - t0:.0f}s; L>=1 per variant",
          {v: int((oL[i] >= 1).sum()) for i, v in enumerate(L.variants)}, f"rss {rss_gb()}")
    return dict(L=oL, G=oG, E=oE, W=oW, own=own)


def brute4(u, L, P, nd, off, sstr, mode, rng, m=300):
    """Pure-python recount of L and W (observed order: slot == gene) with the orientation rule written out directly."""
    n_inst = P["L"].shape[1]
    left_all = np.repeat(np.arange(u.n, dtype=np.int64), nd)
    pick = rng.choice(n_inst, m, replace=False)
    bad, maxd, nonzero = 0, 0.0, 0
    for inst in pick:
        i = int(left_all[inst])
        a, b = i, i + int(inst - off[i] + 1)
        la = int(u.lin[a])
        sa, sb = int(sstr[a]), int(sstr[b])
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
                    if dd == 0 or not (0 <= x < u.n) or x not in best_a or u.cid[x] != u.cid[y]:
                        continue
                    sx, sy = int(sstr[x]), int(sstr[y])
                    if mode == 1 and sx * sy != sa * sb:
                        continue
                    if mode == 2:
                        if y > x and not (sx == sa and sy == sb):
                            continue
                        if y < x and not (sx == -sa and sy == -sb):
                            continue
                    ta = min(ta, best_a[x])
                if ta < (1 << 30):
                    lg = int(u.lin[y])
                    if lg != la:
                        lins[lg] = min(lins.get(lg, 1 << 30), max(ta, t2))
            nonzero += int(len(lins) > 0)
            for j, (_, wt, r0) in enumerate(WEIGHTS):
                want = sum((math.exp(-(mm - 1) / r0) if wt == 0 else r0 / (r0 + mm - 1)) for mm in lins.values())
                got = float(P["W"][v, j, inst])
                maxd = max(maxd, abs(got - want) / max(1.0, want))
                if abs(got - want) > 1e-3 * max(1.0, want) or len(lins) != int(P["L"][v, inst]):
                    bad += 1
    return dict(instances=int(m), mismatches=bad, max_rel_diff=maxd, checks_with_support=nonzero)


def cmd_stats(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    out = R4 / "stats"
    out.mkdir(parents=True, exist_ok=True)
    u = universe(a.lineage)
    strand = strand_of(u)
    nd, off, left, d = u.instances(W)
    n_inst = len(left)
    L = SR.Lists(u, S2.ListArgs(DEPTHS))
    man = dict(code_sha256=SR.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               host=os.uname().nodename, threads=a.threads, variants=L.variants, weights=[list(x) for x in WEIGHTS],
               lists=L.info, n_instances=int(n_inst), seconds_lists=round(time.time() - t0, 1), modes={})
    ident = np.arange(u.n, dtype=np.int64)
    modes = [int(x) for x in a.modes.split(",") if x]
    keep = {}
    for mode in modes:
        mm = {}
        t1 = time.time()
        obs = run_pass4(u, L, ident, ident, strand, mode, nd, off, n_inst, a.threads, "observed")
        mm["seconds_observed_pass"] = round(time.time() - t1, 1)
        mm["brute_observed"] = brute4(u, L, obs, nd, off, strand, mode, np.random.default_rng(17 + mode), 300)
        stamp("brute", mode, mm["brute_observed"])
        if mode == 0 and not a.lineage:
            r2 = np.load(R2 / "stats/obs.npz")
            r2v = list(r2["variants"])
            chk = {}
            for i, v in enumerate(L.variants):
                j = r2v.index(v)
                chk[v] = dict(L_equal=bool(np.array_equal(r2["L"][j], obs["L"][i])), G_equal=bool(np.array_equal(r2["G"][j], obs["G"][i])),
                              E_maxdiff=float(np.abs(r2["E"][j] - obs["E"][i]).max()),
                              W_maxdiff=float(np.abs(r2["W"][j] - obs["W"][i]).max()))
            mm["round2_check"] = chk
            stamp("round-2 check", chk)
            del r2
        else:
            np.savez(out / f"m{mode}_obs.npz", L=obs["L"], G=obs["G"], E=obs["E"], W=obs["W"], own=obs["own"], variants=np.array(L.variants))
            for r in range(a.perms):
                t1 = time.time()
                ga, so = SR.null_order(u, r)
                p = run_pass4(u, L, ga, so, strand[ga], mode, nd, off, n_inst, a.threads, f"null {r}")
                np.savez(out / f"m{mode}_null{r}.npz", L=p["L"], G=p["G"], E=p["E"], W=p["W"], own=p["own"], variants=np.array(L.variants))
                mm[f"seconds_null_{r}"] = round(time.time() - t1, 1)
                del p
        man["modes"][str(mode)] = mm
        del obs
        jdump(out / "stats.json", man)
    man["seconds"] = round(time.time() - t0, 1)
    man["peak_rss_gb"] = rss_gb()
    jdump(out / "stats.json", man)
    stamp("end stats", man["seconds"], f"rss {rss_gb()}")
    a.lists, a.universe, a.strand = L, u, strand
    if a.then_cands:
        cmd_cands(a)
    if a.edges:
        cmd_edges(a)


# ============================================================================== rule selection
def load_pass(fn):
    z = np.load(fn)
    return dict(L=z["L"], G=z["G"], E=z["E"], W=z["W"], own=z["own"], variants=list(z["variants"]))


def rule_masks4(kind, P, v, j=None, only=None):
    ok, Lt, G, E = S2.derived(P, v)
    if kind == "count":
        for m in SR.COUNT_M:
            name = f"L>={m}"
            if only is None or name == only:
                yield name, Lt >= m
    elif kind == "rankw":
        Wv = P["W"][v, j]
        for t in W_T4:
            name = f"W>={t:g}"
            if only is None or name == only:
                yield name, (Lt >= 2) & (Wv >= t)


def select4(n, obs, nulls, kind, v, j, fdr, left, d):
    nc = [{name: int(m.sum()) for name, m in rule_masks4(kind, p, v, j)} for p in nulls]
    rows, masks = [], {}
    for name, m in rule_masks4(kind, obs, v, j):
        n_obs = int(m.sum())
        n_null = float(np.mean([c[name] for c in nc]))
        rows.append(dict(rule=name, observed=n_obs, null_mean=n_null, fdr=(n_null / n_obs if n_obs else np.nan)))
        masks[name] = m
    rules = pd.DataFrame(rows)
    cand = rules[(rules.fdr <= fdr) & (rules.observed > 0)].sort_values("observed", ascending=False).head(12)
    best, best_cov = None, -1
    for name in cand.rule:
        cov = int(S2.covered(n, masks[name], left, d).sum())
        rules.loc[rules.rule == name, "genes_covered"] = cov
        if cov > best_cov:
            best, best_cov = name, cov
    return rules, best, (masks[best] if best else None)


def build4(name, kind, K, wname, mode, u, obs, nulls, left, d, variants):
    vK = variants.index(f"K{K}_fnone")
    j = WNAMES.index(wname) if kind == "rankw" else None
    rules, best, sig = select4(u.n, obs, nulls, kind, vK, j, FDR, left, d)
    if best is None:
        return None
    _, Lt, _, _ = S2.derived(obs, vK)
    row = rules[rules.rule == best].iloc[0]
    sidx = np.flatnonzero(sig)
    sidx = sidx[np.lexsort((left[sidx], d[sidx]))]
    val = Lt[sidx].astype(np.float32)
    assert (val >= 2).all()
    b, arg = SR.boundary_support_argmax(left[sidx], d[sidx], val, np.arange(len(sidx), dtype=np.int64), u.n)
    genes = S2.covered(u.n, sig, left, d)
    info = dict(name=name, kind=kind, K=K, w=wname, mode=mode, rule=best, fdr=float(row.fdr), null_mean=float(row.null_mean),
                significant_instances=int(len(sidx)), genes_covered=int(genes.sum()), genes_covered_frac=float(genes.mean()),
                supported_boundaries=int((b > 0).sum()), L_q10_q50_q90=np.quantile(val, [0.1, 0.5, 0.9]).tolist())
    return dict(info=info, rules=rules, sidx=sidx, value=val.astype(np.int16), cap=np.full(len(sidx), K, np.int16), b=b, arg=arg)


def chance_summary(obs, nulls, variants, mode):
    out = {}
    for K in DEPTHS:
        v = variants.index(f"K{K}_fnone")
        _, Lt, _, _ = S2.derived(obs, v)
        nl = [S2.derived(p, v)[1] for p in nulls]
        je = WNAMES.index("e50")
        out[f"K{K}"] = dict(obs_L_ge2=int((Lt >= 2).sum()), null_L_ge2=float(np.mean([(x >= 2).sum() for x in nl])),
                            obs_L_ge15=int((Lt >= 15).sum()), null_L_ge15=float(np.mean([(x >= 15).sum() for x in nl])),
                            obs_e50_W_ge1=int(((Lt >= 2) & (obs["W"][v, je] >= 1)).sum()),
                            null_e50_W_ge1=float(np.mean([((x >= 2) & (p["W"][v, je] >= 1)).sum() for x, p in zip(nl, nulls)])),
                            null_W_e50_sum=float(np.mean([p["W"][v, je][x >= 2].sum() for x, p in zip(nl, nulls)])),
                            obs_W_e50_sum=float(obs["W"][v, je][Lt >= 2].sum()))
    return out


def cmd_cands(a):
    import numba
    numba.set_num_threads(a.threads)
    u = getattr(a, "universe", None) or universe(a.lineage)
    nd, off, left, d = u.instances(W)
    res_fn = R4 / "cand" / "cands.json"
    (R4 / "cand").mkdir(parents=True, exist_ok=True)
    summ = json.loads(res_fn.read_text()) if res_fn.exists() else {}
    # chance support of round 2's statistic (mode 0), from round 2's stored passes
    if "mode0_chance" not in summ and not a.lineage:
        o2 = S2.load_pass(R2 / "stats/obs.npz")
        n2 = [S2.load_pass(R2 / f"stats/null{r}.npz") for r in range(3)]
        summ["mode0_chance"] = chance_summary(o2, n2, o2["variants"], 0)
        c = build4("o0rw1000_e50", "rankw", 1000, "e50", 0, u, o2, n2, left, d, o2["variants"])
        r2s = np.load(R2 / "cand/rw1000_e50/sidx.npy")
        summ["mode0_reselect_W_T4"] = dict(rule=c["info"]["rule"], fdr=c["info"]["fdr"], sidx_equal_round2=bool(np.array_equal(c["sidx"], r2s)))
        c0 = build4("o0cnt1000", "count", 1000, None, 0, u, o2, n2, left, d, o2["variants"])
        summ["mode0_count1000"] = c0["info"]
        stamp("mode 0", summ["mode0_reselect_W_T4"], c0["info"]["rule"])
        del o2, n2
        jdump(res_fn, summ)
    for mode in [int(x) for x in a.modes.split(",") if x and (int(x) > 0 or a.lineage)]:
        obs = load_pass(R4 / f"stats/m{mode}_obs.npz")
        nulls = [load_pass(R4 / f"stats/m{mode}_null{r}.npz") for r in range(3)]
        variants = obs["variants"]
        summ[f"mode{mode}_chance"] = chance_summary(obs, nulls, variants, mode)
        for name, kind, K, wn in ((f"o{mode}rw1000_e50", "rankw", 1000, "e50"), (f"o{mode}cnt1000", "count", 1000, None),
                                  (f"o{mode}rw200_e50", "rankw", 200, "e50")):
            t1 = time.time()
            c = build4(name, kind, K, wn, mode, u, obs, nulls, left, d, variants)
            if c is None:
                stamp(name, "no rule passes")
                summ[name] = None
                continue
            out = R4 / "cand" / name
            out.mkdir(parents=True, exist_ok=True)
            for k in ("sidx", "value", "cap", "b", "arg"):
                np.save(out / f"{k}.npy", c[k])
            c["rules"].to_csv(out / "rules.tsv", sep="\t", index=False)
            c["info"]["seconds_select"] = round(time.time() - t1, 1)
            jdump(out / "cand.json", c["info"])
            summ[name] = c["info"]
            stamp("candidate", json.dumps(c["info"], default=float))
        del obs, nulls
        jdump(res_fn, summ)
    stamp("cands done")


# ============================================================================== instance-graph support edges
@njit(parallel=True, cache=True)
def support_edges(sel_left, sel_d, K, n, w, cid, lstart, lend, nb, tail, nbg, lin_of_genome, sstr, omode, inst_pos,
                  n_lin, kmax, out_dst, out_m, out_cnt, out_ng, nchunks):
    """Observed order (slot == gene). For selected instance q = (a, b): per other genus the best-ranked supporting pair
    (a', b') whose instance (min, |diff|) is selected (inst_pos >= 0); a' = the best-ranked list member of a marking b''s
    slot in the channel the orientation rule allows. Keeps the kmax genera with the smallest m (ties: genus id)."""
    m = len(sel_left)
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        mstamp = np.full((2, n), -1, np.int32)
        mrank = np.zeros((2, n), np.int32)
        msrc = np.zeros((2, n), np.int32)
        gst = np.full(n_lin, -1, np.int64)
        gm = np.zeros(n_lin, np.int64)
        gq = np.zeros(n_lin, np.int64)
        glist = np.zeros(n_lin, np.int64)
        for q in range(lo, hi):
            a = sel_left[q]
            b = a + sel_d[q]
            sa = np.int64(sstr[a])
            sb = np.int64(sstr[b])
            e0 = lstart[a]
            e1 = lend[a]
            if e1 > e0 + K:
                e1 = e0 + K
            for e in range(e0, e1):
                s = nb[e]
                r = e - e0
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                chn = 0
                if omode == 1:
                    if sstr[s] < 0:
                        chn = 1
                elif omode == 2:
                    if np.int64(sstr[s]) == sa:
                        ulo = s + 1
                    else:
                        chn = 1
                        uhi = s
                for u in range(ulo, uhi):
                    if u != s and cid[u] == c and mstamp[chn, u] != q:
                        mstamp[chn, u] = q
                        mrank[chn, u] = r
                        msrc[chn, u] = s
            ng = 0
            f0 = lstart[b]
            f1 = lend[b]
            if f1 > f0 + K:
                f1 = f0 + K
            for e in range(f0, f1):
                s = nb[e]
                chb = 0
                if omode == 1:
                    if np.int64(sstr[s]) * sa * sb < 0:
                        chb = 1
                elif omode == 2:
                    if np.int64(sstr[s]) != sb:
                        chb = 1
                if mstamp[chb, s] != q:
                    continue
                xa = msrc[chb, s]
                left = xa if xa < s else s
                dd = s - xa if s > xa else xa - s
                qq = inst_pos[left * w + dd - 1]
                if qq < 0:
                    continue
                ra = mrank[chb, s]
                mab = np.int64(tail[e0 + ra])
                tb_ = np.int64(tail[e])
                if tb_ > mab:
                    mab = tb_
                lg = lin_of_genome[nbg[e]]
                if gst[lg] != q:
                    gst[lg] = q
                    gm[lg] = mab
                    gq[lg] = qq
                    glist[ng] = lg
                    ng += 1
                elif mab < gm[lg]:
                    gm[lg] = mab
                    gq[lg] = qq
            out_ng[q] = ng
            if ng == 0:
                out_cnt[q] = 0
                continue
            key = np.empty(ng, np.int64)
            for t in range(ng):
                key[t] = gm[glist[t]] * n_lin + glist[t]
            order = np.argsort(key)
            k = ng if ng < kmax else kmax
            out_cnt[q] = k
            for t in range(k):
                lg = glist[order[t]]
                out_dst[q, t] = gq[lg]
                out_m[q, t] = gm[lg]


def cand_dir(name):
    return (R4 / "cand" / name) if name.startswith("o") else (R2 / "cand" / name)


def mode_of(name):
    return int(name[1]) if name.startswith("o") else 0


def cmd_edges(a):
    import numba
    numba.set_num_threads(a.threads)
    u = getattr(a, "universe", None) or universe(a.lineage)
    strand = getattr(a, "strand", None)
    if strand is None:
        strand = strand_of(u)
    L = getattr(a, "lists", None) or SR.Lists(u, S2.ListArgs(DEPTHS))
    nd, off, left, d = u.instances(W)
    for name in [x for x in a.edge_names.split(",") if x] if getattr(a, "edge_names", "") else [x for x in a.names.split(",") if x]:
        t0 = time.time()
        cd = cand_dir(name)
        sidx = np.load(cd / "sidx.npy")
        K = int(json.loads((cd / "cand.json").read_text())["K"])
        mode = mode_of(name)
        sl, sd = left[sidx].astype(np.int64), d[sidx].astype(np.int64)
        inst_pos = np.full(u.n * W, -1, np.int64)
        inst_pos[sl * W + sd - 1] = np.arange(len(sidx))
        M = len(sidx)
        dst = np.full((M, KMAX_EDGES), -1, np.int32)
        mm = np.zeros((M, KMAX_EDGES), np.int32)
        cnt = np.zeros(M, np.int8)
        ngen = np.zeros(M, np.int32)
        support_edges(sl, sd, K, u.n, W, u.cid, L.lstart, L.lend, L.nb, L.tail, L.nbg, u.lin_of_genome, strand, mode, inst_pos,
                      u.nl, KMAX_EDGES, dst, mm, cnt, ngen, a.threads * 8)
        out = R4 / "graph" / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "sup_dst.npy", dst)
        np.save(out / "sup_m.npy", mm)
        np.save(out / "sup_cnt.npy", cnt)
        np.save(out / "sup_ngenera.npy", ngen)
        # self check: a supporting instance is never the instance itself (lists exclude the own genus)
        selfhits = int(sum(int((dst[:, t] == np.arange(M)).sum()) for t in range(KMAX_EDGES)))
        info = dict(name=name, mode=mode, K=K, instances=M, with_any_support=int((cnt > 0).sum()),
                    genera_with_significant_support_q10_q50_q90=np.quantile(ngen, [0.1, 0.5, 0.9]).tolist(),
                    kept_q50=float(np.median(cnt)), self_hits=selfhits, m_of_kept_q50_q90=np.quantile(mm[dst >= 0], [0.5, 0.9]).tolist(),
                    seconds=round(time.time() - t0, 1))
        jdump(out / "edges.json", info)
        stamp("edges", json.dumps(info, default=float), f"rss {rss_gb()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "cands", "edges"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--modes", default="0,1,2")
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--then-cands", action="store_true")
    ap.add_argument("--edges", action="store_true", help="stats: then edges for --edge-names (lists stay in memory)")
    ap.add_argument("--edge-names", default="", help="comma-separated candidate names")
    ap.add_argument("--names", default="", help="edges: ';'-separated candidate names")
    ap.add_argument("--lineage", default="", help="genome -> lineage map (.npy by genome_code or parquet genome, lineage)")
    ap.add_argument("--lin-tag", default="", help="with --lineage: outputs under $RD/softrec/r4/lin_<tag>/")
    a = ap.parse_args()
    if a.lineage:
        global R4
        assert a.lin_tag, "--lin-tag is required with --lineage"
        R4 = RD / "softrec/r4" / f"lin_{a.lin_tag}"
        R4.mkdir(parents=True, exist_ok=True)
    {"stats": cmd_stats, "cands": cmd_cands, "edges": cmd_edges}[a.cmd](a)


if __name__ == "__main__":
    main()
