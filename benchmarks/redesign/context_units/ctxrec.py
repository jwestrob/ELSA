#!/usr/bin/env python
"""context_units (workstream ws_context): the effective number of independent genomic contexts (N_eff) in place of the
genus count L of the token recurrence engine.

Statistic (predeclared.json)
  x_g        zca10 unit vector of gene g (uid-level transform of softrec/embed_tokens.transform)
  context    instance (a, b), lo = min, hi = max on contig c: window genes [lo - W, lo - 1] and [hi + 1, hi + W] inside c
             S = sum of the window's vectors (variant gc: x_g - mu_genome), n_ctx = window size, B = S / ||S||
  per genome B_g = unit-normalised sum of B over the pair's instances in g with n_ctx >= 1 (informative genome)
  pooling    k informative genomes: c_bar = (||sum B_g||^2 - k) / (k (k - 1))   (mean pairwise cosine, O(k d))
  floor      c0 = mean B.B' over random pairs of observed instances in different genomes
  N_eff      rho = clip((c_bar - c0) / (1 - c0), 0, 1); N_eff = G / (1 + (G - 1) rho); k < 2 -> rho = 0
Variants: raw, gc (genome-centred), shuf_raw / shuf_gc (contexts permuted among instances, seed 17),
          member (pair's own vectors minus their token-family means; W-independent).
Rule family P <= 1e-e & N_eff >= t (engine selection: FDR <= 0.01, top 12 by count, most genes covered); landscape value
L := max(2 N_eff / t, 2). Readings R1 and DH (boundary/rules.py) scored on the DPANN package; logs print odd numbers only.

usage
  ctxrec.py refcheck --threads T                         genus map through this driver: reproduce stored rescue3 DH
  ctxrec.py run --variants raw,shuf_raw,member --threads T [--W 10,25,50,100] [--t 1.5,2,3] [--t-shuf 2]
The run reads no taxonomy (recurrence_lf.Universe with one lineage per genome; G is all it uses).
"""
from __future__ import annotations

import argparse
import gc as _gc
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
U = PKG / "universe"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
CU = RD / "context_units"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
LF = RD / "lineagefree"
sys.path.insert(0, str(LF / "code"))
import recurrence_lf as RL  # noqa: E402  (read-only import)
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as BR  # noqa: E402
sys.path.insert(0, str(RD / "softrec/code"))
from embed_tokens import load_run, transform  # noqa: E402

E_LIST = (2, 3, 4, 6, 8, 10, 15, 20, 30)
CALLERS = BR.CALLERS
W_ENGINE = 3
FDR = 0.01
REF_SCORES = RD / "boundary/scores/rescue3"
REF_LOCI = RD / "boundary/loci/rescue3"
REF_NAME = {"R1": "R1_conserved2", "DH": "DH_coreH_res0.3_f0.5"}
COMPARATORS = ("emb_hom_g0.75", "emb_mean_g0.55", "fam_jac_g0.2", "genome")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    path = Path(path)
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


# ============================================================================== kernels
@njit(parallel=True, cache=True)
def build_prefix(X, uid, c_start, c_end, P):
    """P[k + c] = sum of x over genes c_start[c] .. k - 1 of contig c (row c_start[c] + c is zero)."""
    d = X.shape[1]
    for c in prange(len(c_start)):
        s = c_start[c]
        base = s + c
        for j in range(d):
            P[base, j] = 0.0
        for k in range(s, c_end[c]):
            r = base + (k - s)
            q = uid[k]
            for j in range(d):
                P[r + 1, j] = P[r, j] + X[q, j]


@njit(cache=True)
def window_sum(P, c, cs, ce, lo, hi, W, buf):
    a = max(cs, lo - W)
    b = min(ce, hi + W + 1)
    n = (lo - a) + (b - hi - 1)
    if n <= 0:
        return 0
    ra, rl, rh, rb = a + c, lo + c, hi + 1 + c, b + c
    for j in range(buf.shape[0]):
        buf[j] = (P[rl, j] - P[ra, j]) + (P[rb, j] - P[rh, j])
    return n


@njit(cache=True)
def inst_vector(mode, j, w, Ws, P, c_start, c_end, slo, shi, scid, sgen, mu_g, center, X, uid, lo, hi, ta, tb,
                mu_tok, buf):
    """Unnormalised instance vector into buf; returns the number of genes behind it (0 = uninformative)."""
    if mode == 0:
        c = scid[j]
        n = window_sum(P, c, c_start[c], c_end[c], slo[j], shi[j], Ws[w], buf)
        if n > 0 and center:
            g = sgen[j]
            for q in range(buf.shape[0]):
                buf[q] -= n * mu_g[g, q]
        return n
    qa, qb = uid[lo[j]], uid[hi[j]]
    for q in range(buf.shape[0]):
        buf[q] = (X[qa, q] - mu_tok[ta[j], q]) + (X[qb, q] - mu_tok[tb[j], q])
    return 2


@njit(parallel=True, cache=True)
def pooled_cbar(ptr, gen, mode, Ws, P, c_start, c_end, slo, shi, scid, sgen, mu_g, center, X, uid, lo, hi, ta, tb,
                mu_tok, nchunks, out_cbar, out_k):
    npairs = len(ptr) - 1
    nW = len(Ws)
    d = X.shape[1]
    for ch in prange(nchunks):
        p0 = ch * npairs // nchunks
        p1 = (ch + 1) * npairs // nchunks
        m = np.zeros((nW, d))
        acc = np.zeros((nW, d))
        buf = np.empty(d)
        for p in range(p0, p1):
            m[:] = 0.0
            k = 0
            i = ptr[p]
            end = ptr[p + 1]
            while i < end:
                g = gen[i]
                acc[:] = 0.0
                inf = False
                j = i
                while j < end and gen[j] == g:
                    for w in range(nW):
                        n = inst_vector(mode, j, w, Ws, P, c_start, c_end, slo, shi, scid, sgen, mu_g, center, X, uid,
                                        lo, hi, ta, tb, mu_tok, buf)
                        if n == 0:
                            break
                        nb = 0.0
                        for q in range(d):
                            nb += buf[q] * buf[q]
                        if nb > 0:
                            nb = math.sqrt(nb)
                            for q in range(d):
                                acc[w, q] += buf[q] / nb
                            inf = True
                    j += 1
                if inf:
                    k += 1
                    for w in range(nW):
                        na = 0.0
                        for q in range(d):
                            na += acc[w, q] * acc[w, q]
                        if na > 0:
                            na = math.sqrt(na)
                            for q in range(d):
                                m[w, q] += acc[w, q] / na
                i = j
            out_k[p] = k
            for w in range(nW):
                if k >= 2:
                    s2 = 0.0
                    for q in range(d):
                        s2 += m[w, q] * m[w, q]
                    out_cbar[p, w] = (s2 - k) / (k * (k - 1.0))
                else:
                    out_cbar[p, w] = np.nan


@njit(parallel=True, cache=True)
def floor_sample(ia, ib, mode, Ws, P, c_start, c_end, slo, shi, scid, sgen, mu_g, center, X, uid, lo, hi, ta, tb,
                 mu_tok, out_dot, out_n):
    nW = len(Ws)
    d = X.shape[1]
    for s in prange(len(ia)):
        b1 = np.empty(d)
        b2 = np.empty(d)
        for w in range(nW):
            n1 = inst_vector(mode, ia[s], w, Ws, P, c_start, c_end, slo, shi, scid, sgen, mu_g, center, X, uid, lo, hi,
                             ta, tb, mu_tok, b1)
            n2 = inst_vector(mode, ib[s], w, Ws, P, c_start, c_end, slo, shi, scid, sgen, mu_g, center, X, uid, lo, hi,
                             ta, tb, mu_tok, b2)
            out_n[s, w] = min(n1, n2)
            if n1 == 0 or n2 == 0:
                out_dot[s, w] = np.nan
                continue
            x1 = 0.0
            x2 = 0.0
            x12 = 0.0
            for q in range(d):
                x1 += b1[q] * b1[q]
                x2 += b2[q] * b2[q]
                x12 += b1[q] * b2[q]
            out_dot[s, w] = x12 / math.sqrt(x1 * x2) if x1 > 0 and x2 > 0 else np.nan


@njit(parallel=True, cache=True)
def gene_pair_cos(ga, gb, X, uid, mu_g, gen, center, out):
    d = X.shape[1]
    for s in prange(len(ga)):
        qa, qb = uid[ga[s]], uid[gb[s]]
        x1 = 0.0
        x2 = 0.0
        x12 = 0.0
        for q in range(d):
            a = X[qa, q] - (mu_g[gen[ga[s]], q] if center else 0.0)
            b = X[qb, q] - (mu_g[gen[gb[s]], q] if center else 0.0)
            x1 += a * a
            x2 += b * b
            x12 += a * b
        out[s] = x12 / math.sqrt(x1 * x2) if x1 > 0 and x2 > 0 else np.nan


@njit(cache=True)
def token_means(tok, uid, X, T):
    d = X.shape[1]
    mu = np.zeros((T, d), dtype=np.float32)
    cnt = np.zeros(T, dtype=np.int64)
    for k in range(len(tok)):
        t = tok[k]
        if t < 0:
            continue
        cnt[t] += 1
        q = uid[k]
        for j in range(d):
            mu[t, j] += X[q, j]
    for t in range(T):
        if cnt[t] > 0:
            for j in range(d):
                mu[t, j] /= cnt[t]
    return mu


# ============================================================================== base data
class Base:
    """Universe (no taxonomy), tokens, observed instances and pair statistics, and three gene-order nulls."""

    def __init__(self, threads, universe=None):
        t0 = time.time()
        self.u = u = universe or RL.Universe(U, lineage_genome=True)
        assert np.array_equal(u.gid, np.arange(u.n)), "universe order != gid order"
        t = pd.read_parquet(TOK)
        tok = np.full(u.n, -1, dtype=np.int64)
        tok[t.gid.to_numpy()] = t.token.to_numpy()
        self.tok = tok
        self.cid = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start).astype(np.int64)
        self.key, self.left, self.dist, self.T = RL.colocalization(u, tok, W_ENGINE)
        self.st = RL.pair_statistics(u, tok, self.key, self.left, W_ENGINE, self.T)
        log("observed", len(self.key), "instances", len(self.st["key"]), "pairs", f"{time.time()-t0:.0f}s")
        self.nulls = []
        for r in range(3):
            rng = np.random.default_rng(1000 + r)
            tok_n = RL.shuffle_within(u.g_start, u.g_end, tok, rng)
            k_n, l_n, d_n, _ = RL.colocalization(u, tok_n, W_ENGINE)
            st_n = RL.pair_statistics(u, tok_n, k_n, l_n, W_ENGINE, self.T)
            self.nulls.append(dict(key=k_n, left=l_n, dist=d_n, st=st_n, tok=tok_n))
            log("null", r, len(k_n), "instances", f"{time.time()-t0:.0f}s")


def select_instances(st, key, left, dist, u, cid):
    """Pairs with G >= 2 and P <= 1e-2; their instances sorted by (pair, genome)."""
    selp = np.flatnonzero((st["P"] <= 1e-2) & (st["G"] >= 2))
    sk = st["key"][selp]
    pos = np.minimum(np.searchsorted(sk, key), max(len(sk) - 1, 0))
    found = sk[pos] == key
    ii = np.flatnonzero(found)
    pidx = pos[found]
    gen = u.gc[left[ii]].astype(np.int64)
    o = np.lexsort((gen, pidx))
    ii, pidx, gen = ii[o], pidx[o], gen[o]
    ptr = np.searchsorted(pidx, np.arange(len(selp) + 1)).astype(np.int64)
    return dict(selp=selp, ii=ii, ptr=ptr, gen=gen)


# ============================================================================== engine output + readings
def ribo_counts(rib):
    rec = int(rib["recovered"])
    res = int(rib["resolved"])
    parv_rec = int(str(rib["parvarchaeaceae"]).split("/")[0])
    parv_res = int(str(rib.get("parvarchaeaceae_resolved", "0")).split("/")[0])
    return rec, res, parv_rec, parv_res


def cond_even(pr, rib, rib_ref):
    rec, res, parv, _ = ribo_counts(rib)
    _, res_ref, _, _ = ribo_counts(rib_ref)
    d = pr["summed"][0]
    c = dict(summed_loss_guard=bool(d >= -0.01), summed_literal=bool(abs(d) <= 0.01),
             no_F1_loss_over_0p01=bool(all(pr[x]["d_f1"][0] >= -0.01 for x in CALLERS)),
             ribosomal=bool(rec >= 1650 and parv >= 110 and res >= 1400),
             ribosomal_3pct=bool(rec >= 1650 and parv >= 110 and abs(res - res_ref) <= 0.03 * res_ref),
             matches_F1_0p02=bool(all(pr[x]["d_f1"][0] >= -0.02 for x in CALLERS)))
    c["replaces"] = bool(c["summed_loss_guard"] and c["no_F1_loss_over_0p01"] and c["ribosomal"])
    c["matches"] = bool(c["summed_loss_guard"] and c["matches_F1_0p02"] and c["ribosomal"])
    return c


def cond_odd(pr, rib):
    rec, res, parv, _ = ribo_counts(rib)
    c1 = bool(pr["summed"][0] >= -0.01)
    c2 = bool(all(pr[x]["d_f1"][0] >= -0.01 for x in CALLERS))
    c3 = bool(rec >= 830 and parv >= 54 and res >= 705)
    c2m = bool(all(pr[x]["d_f1"][0] >= -0.02 for x in CALLERS))
    return dict(summed_loss_guard=c1, no_F1_loss_over_0p01=c2, ribosomal=c3, n_met=int(c1) + int(c2) + int(c3),
                matches=bool(c1 and c2m and c3))


class Scorer:
    def __init__(self, base):
        self.base = base
        self.gcode = B.odd_even_code()
        self.runs = B.ribo_runs()
        self.even_runs = (self.runs.genome_code % 2 == 0).to_numpy()
        self.refs = {}
        for name in ("R1", "DH"):
            d = {}
            ref = pd.read_parquet(REF_SCORES / f"{REF_NAME[name]}.parquet")
            ref["excess"] = ref.f1 - ref.control_mean_f1
            rl = pd.read_parquet(REF_LOCI / f"{REF_NAME[name]}.parquet")
            d["genus"] = (ref.set_index("instance_id"), B.ribo_score(self.runs, rl.gid_s.to_numpy(), rl.gid_e.to_numpy()))
            for c in COMPARATORS:
                f = LF / "scores" / f"tok_{c}__{name}.parquet"
                fl = LF / "loci" / f"tok_{c}__{name}.parquet"
                if f.exists():
                    x = pd.read_parquet(f)
                    if "excess" not in x:
                        x["excess"] = x.f1 - x.control_mean_f1
                    sc = B.ribo_score(self.runs, *[pd.read_parquet(fl)[k].to_numpy() for k in ("gid_s", "gid_e")]) \
                        if fl.exists() else None
                    d["lf_" + c] = (x.set_index("instance_id"), sc)
            self.refs[name] = d
        self.rib_ref = {name: B.ribo_summary(self.runs, self.refs[name]["genus"][1]) for name in ("R1", "DH")}

    def landscape(self, rdir):
        """blib.Landscape on the shared universe / instances (no reload)."""
        b_ = self.base
        u = b_.u
        L = object.__new__(B.Landscape)
        L.rdir = Path(rdir)
        L.summary = json.load(open(L.rdir / "summary.json"))
        L.w = W_ENGINE
        L.u = u
        L.tokens_path = TOK
        L.tok = b_.tok
        key, left, dist, T = b_.key, b_.left, b_.dist, b_.T
        sp = pd.read_parquet(L.rdir / "significant_pairs.parquet")
        assert (sp.key.to_numpy() == sp.lo.to_numpy() * T + sp.hi.to_numpy()).all()
        assert np.all(np.diff(sp.key.to_numpy()) > 0)
        L.sp = sp
        skey = sp.key.to_numpy()
        pidx = np.minimum(np.searchsorted(skey, key), len(skey) - 1)
        found = skey[pidx] == key
        li, di, pi = left[found], dist[found], pidx[found].astype(np.int64)
        L.inst_left, L.inst_dist, L.inst_pair = li, di, pi
        Lv = sp.L.to_numpy().astype(np.float32)
        J = sp.J.to_numpy().astype(np.float32)
        L.b, L.arg = B.R.boundary_support_argmax(li, di, Lv[pi], pi, u.n)
        saved = np.load(L.rdir / "boundary_support.npy")
        assert np.array_equal(L.b, saved), "boundary support not reproduced"
        L.bJ, _ = B.R.boundary_support_argmax(li, di, J[pi], pi, u.n)
        L.nspan = B.span_count(li, di, u.n)
        tm = pd.read_parquet(L.rdir / "token_modules.parquet")
        token_module = np.full(T, -1, dtype=np.int64)
        token_module[tm.token.to_numpy()] = tm.module.to_numpy()
        L.T = T
        L.token_module = token_module
        L.gene_module = np.where(b_.tok >= 0, token_module[np.maximum(b_.tok, 0)], -1).astype(np.int64)
        L.pair_module = sp.module.to_numpy().astype(np.int64)
        L.cs, L.ce = u.c_start, u.c_end
        L.genome_arr = u.genes.genome.to_numpy()
        L.contig_arr = u.genes.contig.to_numpy()
        L.genome_code = u.gc
        L.s2, L.e2 = B.R.runs_at_least(L.b, L.cs, L.ce, 2.0, 0)
        return L

    def score(self, rdir, tag, threads, readings=("R1", "DH"), extra_builds=None):
        L = self.landscape(rdir)
        bld = BR.Builder(L)
        builds = {"R1": ("R1", {}), "DH": ("D", dict(core="H", res=0.3, f=0.5))}
        out = {}
        for name in readings:
            t1 = time.time()
            if extra_builds and name in extra_builds:
                s, e = extra_builds[name](L, bld)
                refname = "DH"
            else:
                s, e = bld.build(*builds[name])
                refname = name
            df = L.table(s, e)
            df.to_parquet(CU / "loci" / f"{tag}__{name}.parquet", index=False)
            per, _ = B.score_table(B.PKG, df, threads)
            per.to_parquet(CU / "scores" / f"{tag}__{name}.parquet", index=False)
            pi = per.set_index("instance_id")
            g = pi.genome.map(self.gcode).to_numpy()
            summ = {sp: B.summarize_per(pi[g % 2 == (1 if sp == "odd" else 0)].reset_index()) for sp in ("odd", "even")}
            sc = B.ribo_score(self.runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
            er = self.even_runs
            item = dict(summary=summ, ribo=B.ribo_summary(self.runs, sc),
                        ribo_odd=B.ribo_summary(self.runs[~er].reset_index(drop=True), sc[~er].reset_index(drop=True)),
                        ribo_even=B.ribo_summary(self.runs[er].reset_index(drop=True), sc[er].reset_index(drop=True)),
                        loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), L.u.n), ref=refname)
            for cname, (rper, rsc) in self.refs[refname].items():
                item[f"vs_{cname}_even"] = BR.paired(pi, rper, self.gcode, "even")
                item[f"vs_{cname}_odd"] = BR.paired(pi, rper, self.gcode, "odd")
                if rsc is not None:
                    item[f"ribo_vs_{cname}"] = BR.ribo_paired(sc, rsc)
            item["conditions_even"] = cond_even(item["vs_genus_even"], item["ribo"], self.rib_ref[refname])
            item["conditions_odd"] = cond_odd(item["vs_genus_odd"], item["ribo_odd"])
            item["seconds"] = round(time.time() - t1, 1)
            out[name] = item
            od = summ["odd"]
            log(tag, name, "ODD summed", round(od["summed_DF_TXSS_excess"], 4),
                {c[:4]: round(od[c]["f1"], 3) for c in CALLERS},
                "d vs genus", [round(x, 4) for x in item["vs_genus_odd"]["summed"]],
                {c[:4]: round(item["vs_genus_odd"][c]["d_f1"][0], 4) for c in CALLERS},
                "ribo odd", item["ribo_odd"]["recovered"], item["ribo_odd"]["resolved"],
                item["ribo_odd"]["parvarchaeaceae"], "cond odd", item["conditions_odd"], f"{item['seconds']}s")
        return out


def odd_view(res):
    keep_r = {}
    for name, it in res["readings"].items():
        keep_r[name] = {k: v for k, v in it.items() if k.endswith("_odd") or k in ("loci", "seconds", "ref")}
        keep_r[name]["summary_odd"] = it["summary"]["odd"]
    o = {k: v for k, v in res.items() if k != "readings"}
    o["readings"] = keep_r
    return o


def write_rdir(rdir, base, st, mask, value, extra, summary):
    """Engine-format result directory (significant_pairs, token_modules (none), boundary_support, summary)."""
    rdir = Path(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    u = base.u
    sp = pd.DataFrame({k: st[k][mask] for k in ("key", "lo", "hi", "G", "J", "E", "P", "F_lo", "F_hi")})
    sp["L"] = value[mask].astype(np.float64)
    sp["NP"] = 0
    sp["n_other"] = 0
    sp["module"] = -1
    for k, v in (extra or {}).items():
        sp[k] = v[mask]
    sp.to_parquet(rdir / "significant_pairs.parquet", index=False)
    pd.DataFrame({"token": np.zeros(0, np.int64), "module": np.zeros(0, np.int64)}).to_parquet(
        rdir / "token_modules.parquet", index=False)
    sig = np.zeros(len(st["key"]), dtype=bool)
    sig[mask] = True
    pos = np.searchsorted(st["key"], base.key)
    inst = sig[pos]
    sig_rank = np.cumsum(mask) - 1
    li, di = base.left[inst], base.dist[inst]
    b, _ = B.R.boundary_support_argmax(li, di, value[pos[inst]].astype(np.float32), sig_rank[pos[inst]], u.n)
    np.save(rdir / "boundary_support.npy", b)
    summary = dict(summary, w=W_ENGINE, significant_pairs=int(mask.sum()))
    jdump(rdir / "summary.json", summary)


def select_rule(base, stat_obs, stat_nulls, t, rule_kind="neff"):
    """Engine selection over P <= 1e-e & X >= t (X = N_eff or L)."""
    u, st = base.u, base.st
    rows = []
    for e in E_LIST:
        mo = (st["P"] <= 10.0 ** -e) & (stat_obs >= t)
        n_obs = int(mo.sum())
        n_null = float(np.mean([((nl["st"]["P"] <= 10.0 ** -e) & (sn >= t)).sum()
                                for nl, sn in zip(base.nulls, stat_nulls)]))
        rows.append(dict(rule=f"P<=1e-{e}&{rule_kind}>={t:g}", e=e, observed=n_obs, null_mean=n_null,
                         fdr=(n_null / n_obs if n_obs else np.nan)))
    rules = pd.DataFrame(rows)
    ok = rules[(rules.fdr <= FDR) & (rules.observed > 0)]
    best, best_cov = None, -1
    for _, r in ok.sort_values("observed", ascending=False).head(12).iterrows():
        mo = (st["P"] <= 10.0 ** -r.e) & (stat_obs >= t)
        _, genes = RL.covered_genes(u, st, mo, base.key, base.left, base.dist)
        cov = int(genes.sum())
        rules.loc[rules.rule == r.rule, "genes_covered"] = cov
        if cov > best_cov:
            best, best_cov = r, cov
    if best is None:
        return None, rules
    mask = (st["P"] <= 10.0 ** -best.e) & (stat_obs >= t)
    return dict(rule=best.rule, e=int(best.e), fdr=float(best.fdr), genes_covered=int(best_cov), mask=mask), rules


# ============================================================================== commands
def cmd_refcheck(a):
    """Genus map through this driver (L = genus count; evaluation map from lineagefree): must reproduce the stored
    rescue3 significant pairs and the DH per-instance scores (even summed 0.4270)."""
    import numba
    numba.set_num_threads(a.threads)
    ug = RL.Universe(U, lineage_map=LF / "groups/genus_map.parquet")
    base = Base(a.threads, universe=ug)
    Lobs = base.st["L"].astype(np.float64)
    Lnull = [nl["st"]["L"].astype(np.float64) for nl in base.nulls]
    sel, rules = select_rule(base, Lobs, Lnull, 2, "L")
    log("genus rule", sel["rule"], sel["fdr"], int(sel["mask"].sum()))
    stored = pd.read_parquet(RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3/significant_pairs.parquet")
    st = base.st
    m = sel["mask"]
    same = dict(n=int(m.sum()) == len(stored),
                key=bool(np.array_equal(st["key"][m], stored.key.to_numpy())),
                L=bool(np.array_equal(st["L"][m], stored.L.to_numpy())),
                G=bool(np.array_equal(st["G"][m], stored.G.to_numpy())),
                P=bool(np.allclose(st["P"][m], stored.P.to_numpy(), rtol=0, atol=0)))
    log("significant pairs equal stored:", same)
    rdir = CU / "rdir" / "refcheck_genus"
    write_rdir(rdir, base, st, m, st["L"].astype(np.float64), None, dict(arm="refcheck_genus", rule=sel["rule"]))
    sc = Scorer(base)
    res = sc.score(rdir, "refcheck_genus", a.threads)
    ref = sc.refs["DH"]["genus"][0]
    per = pd.read_parquet(CU / "scores" / "refcheck_genus__DH.parquet").set_index("instance_id")
    eq = dict(f1_equal=bool(np.array_equal(per.loc[ref.index].f1.to_numpy(), ref.f1.to_numpy())),
              control_equal=bool(np.allclose(per.loc[ref.index].control_mean_f1.to_numpy(),
                                             ref.control_mean_f1.to_numpy(), rtol=0, atol=1e-12)),
              DH_even_summed=res["DH"]["summary"]["even"]["summed_DF_TXSS_excess"],
              DH_even_f1={c: res["DH"]["summary"]["even"][c]["f1"] for c in CALLERS},
              DH_ribo=res["DH"]["ribo"])
    log("refcheck (stored reference numbers):", eq)
    jdump(CU / "results" / "refcheck_genus.json", dict(significant_pairs_equal=same, scores=eq, rule=sel["rule"]))


def load_vectors(base, threads):
    t0 = time.time()
    X, uu, valid, cnt = load_run(EMB)
    try:                                   # BLAS threads for the whitening only (the job sets OMP_NUM_THREADS=1)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads):
            info = transform(X, valid, cnt, "zca10")
    except ImportError:
        info = transform(X, valid, cnt, "zca10")
    uid = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    assert len(uid) == base.u.n and (uid >= 0).all()
    info["invalid_unique"] = int((~valid).sum())
    info["genes_with_invalid_vector"] = int((~valid[uid]).sum())
    log("zca10 vectors", X.shape, info, f"{time.time()-t0:.0f}s")
    return X, uid, info


def cmd_run(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    Ws = np.array([int(x) for x in a.W.split(",")], dtype=np.int64)
    ts = [float(x) for x in a.t.split(",")]
    variants = a.variants.split(",")
    base = Base(a.threads)
    u = base.u
    X, uid, vinfo = load_vectors(base, a.threads)
    d = X.shape[1]
    nc = len(u.c_start)
    P = np.empty((u.n + nc, d), dtype=np.float32)
    build_prefix(X, uid, u.c_start, u.c_end, P)
    log("prefix sums", P.shape, f"{time.time()-t0:.0f}s")
    # genome means of x (gc variant): from the prefix sums
    mu_g = np.zeros((u.ng, d), dtype=np.float64)
    cont_sum = (P[u.c_end + np.arange(nc)].astype(np.float64) - P[u.c_start + np.arange(nc)].astype(np.float64))
    np.add.at(mu_g, u.gc[u.c_start], cont_sum)
    del cont_sum
    mu_g /= np.bincount(u.gc, minlength=u.ng)[:, None]
    need_member = any(v == "member" for v in variants)
    mu_tok = token_means(base.tok, uid, X, base.T) if need_member else np.zeros((1, d), np.float32)
    log("means", f"{time.time()-t0:.0f}s", "mu_tok", mu_tok.shape)
    desc = dict(vectors=vinfo, W=Ws.tolist(), variants=variants)

    # ---- descriptive: composition coherence (gene pairs within vs between genomes) and n_ctx distribution
    rng = np.random.default_rng(17)
    ns = 2_000_000
    ga = rng.integers(0, u.n, ns)
    gstart, gend = u.g_start[u.gc[ga]], u.g_end[u.gc[ga]]
    gb_same = gstart + (rng.random(ns) * (gend - gstart)).astype(np.int64)
    gb_diff = rng.integers(0, u.n, ns)
    okd = u.gc[gb_diff] != u.gc[ga]
    oks = gb_same != ga
    gen_all = u.gc.astype(np.int64)
    for center in (False, True):
        o1 = np.empty(ns)
        o2 = np.empty(ns)
        gene_pair_cos(ga, gb_same, X, uid, mu_g, gen_all, center, o1)
        gene_pair_cos(ga, gb_diff, X, uid, mu_g, gen_all, center, o2)
        desc[f"gene_cos_{'gc' if center else 'raw'}"] = dict(
            same_genome_mean=float(np.nanmean(o1[oks])), same_genome_sd=float(np.nanstd(o1[oks])),
            diff_genome_mean=float(np.nanmean(o2[okd])), diff_genome_sd=float(np.nanstd(o2[okd])),
            diff_q999=float(np.nanquantile(o2[okd], 0.999)))
    lo_all = base.left
    hi_all = base.left + base.dist
    c_all = base.cid[lo_all]
    nctx = {}
    for W in Ws:
        a_ = np.maximum(u.c_start[c_all], lo_all - W)
        b_ = np.minimum(u.c_end[c_all], hi_all + W + 1)
        n = (lo_all - a_) + (b_ - hi_all - 1)
        nctx[int(W)] = dict(zero=float((n == 0).mean()), q=np.quantile(n, [.1, .25, .5, .75, .9]).tolist(),
                            full=float((n == 2 * W).mean()))
    desc["n_ctx_observed_instances"] = nctx
    log("descriptive", json.dumps({k: v for k, v in desc.items() if k != "vectors"}))

    # ---- optional exterior windows: the window starts outside the pass-1 level-2 locus containing the instance
    ext = None
    if a.exterior_from:
        b1 = np.load(Path(a.exterior_from) / "boundary_support.npy")
        rs, re_ = RL.runs_at_least(b1, u.c_start, u.c_end, 2.0, 0)
        ls_, le_ = np.arange(u.n, dtype=np.int64), np.arange(u.n, dtype=np.int64)
        ln = re_ - rs + 1
        idx = np.repeat(np.arange(len(rs)), ln)
        genes_in = rs[idx] + (np.arange(len(idx)) - np.repeat(np.cumsum(ln) - ln, ln))
        ls_[genes_in] = rs[idx]
        le_[genes_in] = re_[idx]
        ext = (ls_, le_)
        desc["exterior"] = dict(source=a.exterior_from, loci=int(len(rs)), median_locus_genes=float(np.median(ln)))

    def exterior(lo, hi):
        if ext is None:
            return lo, hi
        return np.minimum(lo, ext[0][lo]), np.maximum(hi, ext[1][hi])

    # ---- per variant: noise floor (observed instances), N_eff for observed and null passes
    gen_inst_all = u.gc[lo_all].astype(np.int64)
    ni = len(lo_all)
    ia = rng.integers(0, ni, ns)
    ib = rng.integers(0, ni, ns)
    okp = gen_inst_all[ia] != gen_inst_all[ib]
    ia, ib = ia[okp], ib[okp]
    tok = base.tok
    neff = {}
    floors = {}
    for v in variants:
        mode = 1 if v == "member" else 0
        center = v in ("gc", "shuf_gc")
        Wv = np.array([0], np.int64) if mode == 1 else Ws
        # floor on observed instances (shuf: same as its base variant; contexts are random instances anyway)
        ta_all = tok[lo_all] if mode == 1 else np.zeros(1, np.int64)
        tb_all = tok[hi_all] if mode == 1 else np.zeros(1, np.int64)
        od = np.empty((len(ia), len(Wv)))
        on = np.empty((len(ia), len(Wv)), dtype=np.int64)
        flo, fhi = exterior(lo_all, hi_all)
        floor_sample(ia, ib, mode, Wv, P, u.c_start, u.c_end, flo, fhi, c_all, gen_inst_all, mu_g, center, X, uid,
                     lo_all, hi_all, ta_all, tb_all, mu_tok, od, on)
        fl = {}
        for wi, W in enumerate(Wv):
            x = od[:, wi]
            ok = np.isfinite(x)
            strata = {}
            for lo_, hi_ in ((1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 200)):
                mm = ok & (on[:, wi] >= lo_) & (on[:, wi] <= hi_)
                if mm.sum() >= 1000:
                    strata[f"{lo_}-{hi_}"] = dict(n=int(mm.sum()), mean=float(x[mm].mean()), sd=float(x[mm].std()))
            fl[int(W)] = dict(c0=float(x[ok].mean()), sd=float(x[ok].std()),
                              q=np.quantile(x[ok], [.01, .5, .99, .999]).tolist(), n=int(ok.sum()), by_min_nctx=strata)
        floors[v] = fl
        c0 = np.array([fl[int(W)]["c0"] for W in Wv])
        sd0 = np.array([fl[int(W)]["sd"] for W in Wv])
        log(v, "floor", {int(W): (round(fl[int(W)]["c0"], 4), round(fl[int(W)]["sd"], 4)) for W in Wv})

        passes = [("obs", base.st, base.key, base.left, base.dist)] + \
                 [(f"null{r}", nl["st"], nl["key"], nl["left"], nl["dist"]) for r, nl in enumerate(base.nulls)]
        out = {}
        for pname, st, key, left, dist in passes:
            sel = select_instances(st, key, left, dist, u, base.cid)
            ii = sel["ii"]
            lo = left[ii]
            hi = left[ii] + dist[ii]
            if v.startswith("shuf"):
                perm = np.random.default_rng(17).permutation(len(left))
                src = perm[ii]
                slo, shi = left[src], left[src] + dist[src]
            else:
                slo, shi = lo, hi
            slo, shi = exterior(slo, shi)
            scid = base.cid[slo]
            sgen = u.gc[slo].astype(np.int64)
            tok_p = tok if pname == "obs" else base.nulls[int(pname[-1])]["tok"]
            ta = tok_p[lo] if mode == 1 else np.zeros(1, np.int64)
            tb = tok_p[hi] if mode == 1 else np.zeros(1, np.int64)
            cbar = np.empty((len(sel["selp"]), len(Wv)))
            kk = np.zeros(len(sel["selp"]), dtype=np.int64)
            pooled_cbar(sel["ptr"], sel["gen"], mode, Wv, P, u.c_start, u.c_end, slo, shi, scid, sgen, mu_g, center, X,
                        uid, lo, hi, ta, tb, mu_tok, a.threads * 16, cbar, kk)
            G = st["G"][sel["selp"]].astype(np.float64)
            # amendment_1: subtract z null standard errors of c_bar (sd0 * sqrt(2 / (k (k - 1))))
            se = sd0[None, :] * np.sqrt(2.0 / np.maximum(kk * (kk - 1.0), 1.0))[:, None]
            rho = np.clip((cbar - c0[None, :] - a.z * se) / (1.0 - c0[None, :]), 0.0, 1.0)
            rho[kk < 2, :] = 0.0
            ne = G[:, None] / (1.0 + (G[:, None] - 1.0) * rho)
            full = np.zeros((len(Wv), len(st["key"])))
            full[:, sel["selp"]] = ne.T
            out[pname] = dict(neff=full, selp=sel["selp"], cbar=cbar, k=kk)
            log(v, pname, "pairs", len(sel["selp"]), "instances", len(ii), "k<2", int((kk < 2).sum()),
                "N_eff/G median by W", [round(float(np.median(ne[:, w] / G)), 3) for w in range(len(Wv))],
                f"{time.time()-t0:.0f}s")
        neff[v] = out
        # per-pair observed table for evaluation
        ob = out["obs"]
        tab = pd.DataFrame(dict(key=base.st["key"][ob["selp"]], G=base.st["G"][ob["selp"]], P=base.st["P"][ob["selp"]],
                                k=ob["k"]))
        for wi, W in enumerate(Wv):
            tab[f"cbar_W{W}"] = ob["cbar"][:, wi]
            tab[f"neff_W{W}"] = ob["neff"][wi, ob["selp"]]
        tab.to_parquet(CU / "ctx" / f"pairs_{v}{'_z%g' % a.z if a.z else ''}{'_ext' if a.exterior_from else ''}.parquet", index=False)
    desc["floors"] = floors
    jdump(CU / "results" / f"descriptive_{'_'.join(variants)}{'_z%g' % a.z if a.z else ''}{'_ext' if a.exterior_from else ''}.json", desc)

    # ---- free the vector arrays before scoring (forked scorer workers)
    del P, X, mu_tok
    _gc.collect()
    sc = Scorer(base)
    for v in variants:
        Wv = [0] if v == "member" else Ws.tolist()
        tlist = ts if v in ("raw", "gc", "member") else [float(x) for x in a.t_shuf.split(",")]
        for wi, W in enumerate(Wv):
            for t in tlist:
                zs = (f"_z{a.z:g}" if a.z else "") + ("_ext" if a.exterior_from else "")
                tag = f"{v}{zs}_W{W}_t{t:g}" if v != "member" else f"member{zs}_t{t:g}"
                nobs = neff[v]["obs"]["neff"][wi]
                nnull = [neff[v][f"null{r}"]["neff"][wi] for r in range(3)]
                sel, rules = select_rule(base, nobs, nnull, t)
                if sel is None:
                    log(tag, "no admissible rule")
                    continue
                m = sel["mask"]
                value = np.maximum(2.0 * nobs / t, 2.0)
                rdir = CU / "rdir" / tag
                write_rdir(rdir, base, base.st, m, value, dict(neff=nobs, G_genomes=base.st["G"]),
                           dict(arm=tag, variant=v, W=W, t=t, rule=sel["rule"], fdr=sel["fdr"]))
                rules.to_csv(rdir / "rules.tsv", sep="\t", index=False)
                log(tag, "rule", sel["rule"], "fdr", round(sel["fdr"], 4), "sig pairs", int(m.sum()),
                    "genes", sel["genes_covered"])
                res = dict(tag=tag, variant=v, z=a.z, W=W, t=t, rule=sel["rule"], fdr=sel["fdr"],
                           significant_pairs=int(m.sum()), genes_covered=sel["genes_covered"],
                           neff_q_sig=np.quantile(nobs[m], [.1, .25, .5, .75, .9, .99]).tolist(),
                           G_q_sig=np.quantile(base.st["G"][m], [.1, .25, .5, .75, .9, .99]).tolist(),
                           floor=floors[v][int(W)])
                res["readings"] = sc.score(rdir, tag, a.threads)
                jdump(CU / "results" / "full" / f"{tag}.json", res)
                jdump(CU / "results" / f"{tag}.odd.json", odd_view(res))
    log("done", f"{time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("refcheck", "run"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--variants", default="raw,shuf_raw,member")
    ap.add_argument("--W", default="10,25,50,100")
    ap.add_argument("--t", default="1.5,2,3")
    ap.add_argument("--t-shuf", default="2")
    ap.add_argument("--z", type=float, default=0.0, help="amendment_1: null standard errors subtracted from c_bar")
    ap.add_argument("--exterior-from", default="", help="pass-1 result dir: windows start outside its level-2 loci")
    a = ap.parse_args()
    for sub in ("results", "results/full", "scores", "loci", "ctx", "rdir"):
        (CU / sub).mkdir(parents=True, exist_ok=True)
    {"refcheck": cmd_refcheck, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
