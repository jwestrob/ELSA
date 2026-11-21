#!/usr/bin/env python
"""softrec round 4 readings: instance communities (L1), oriented landscapes (L2), critic extension and bridging (L3),
reading parameters (L4), scoring on odd (tune) or all genomes (test, even = verdict), label diagnostics, token controls.

usage
  readings4.py graph --land NAME --k 5 --wo 0,0.3,1 --gammas 0.3,1,3,10,30,100,1000 --threads T [--procs P]
      instance graph of a landscape (nodes = its significant instances): support edges (top-k genera per instance from
      softrec4.py edges, weight exp(-(m-1)/50)) + overlap edges (intervals [left, left+d] of two significant instances
      intersect; weight w_o; w_o = 0 is the support-only graph). Connected components and Leiden (igraph modularity,
      n_iterations -1, seed 17) per (w_o, gamma). -> $RD/softrec/r4/graph/<NAME>/ic_k<k>o<wo>_<g|cc>.npy, graph.json
  readings4.py tune --configs FILE --tag TAG --threads T     odd genomes (shadow gold package, 848 odd ribosomal runs)
  readings4.py test --configs FILE --tag TAG --threads T     all genomes; even = verdict; paired vs token R1 / DH / R2sel
  readings4.py diag --configs FILE --tag TAG --threads T     module-label diagnostics (ribosomal runs, odd gold spans)
Configs: one JSON object per line: name, land, lab, crit, f, readings (default R1,C,H,DH).
  land  <cand>        round-2 candidate (r2/cand/<cand>) or round-4 candidate (r4/cand/<cand>, names starting 'o')
        tok | tokor   token engine landscape (rescue3) | token engine --oriented landscape (control)
  lab   none | vf:<famcand>:<fam>:<res> (validated families of <famcand>, Louvain res) | rescue3:<res> (control) |
        ic:<graph tag> (instance communities of the landscape) | tokmod:<res> (token / tokor landscapes)
  crit  none | q<quantile>m<m><x|b>   critic-on = held-out critic score > quantile of the fold's null; x = extend + bridge,
        b = bridge only; added boundaries get support 2 and no label
  f     trimming factor of the H core
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
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "softrec/code"))
import readings2 as R2m  # noqa: E402
import softrec as SR  # noqa: E402
B = R2m.B
BR = R2m.BR

R2 = RD / "softrec/r2"
R4 = RD / "softrec/r4"
GR = RD / "grammar"
TOKOR = RD / "engine/results_dpann_softrec/dpann__softrec4__tok_rescue3__w3__oriented"
CALLERS = R2m.CALLERS
W = 3
log = R2m.log
jdump = R2m.jdump


# ============================================================================== landscapes
class Land4:
    def __init__(self, name, b, arg, sig_left=None, sig_d=None, kind="cand", info=None, extra=None):
        self.name, self.b, self.arg, self.kind = name, b, arg, kind
        self.sig_left, self.sig_d = sig_left, sig_d
        self.info = info or {}
        self.extra = extra or {}


def cand_dir(name):
    return (R4 / "cand" / name) if name.startswith("o") else (R2 / "cand" / name)


def land_cand(ctx, name):
    cd = cand_dir(name)
    sidx = np.load(cd / "sidx.npy")
    b = np.load(cd / "b.npy")
    arg = np.load(cd / "arg.npy")
    return Land4(name, b, arg, ctx.left[sidx], ctx.d[sidx], "cand", json.loads((cd / "cand.json").read_text()))


def land_tok(ctx):
    BL = B.Landscape(B.ARMS["rescue3"])
    return Land4("tok", BL.b, BL.arg, BL.inst_left, BL.inst_dist, "tok", dict(arm="rescue3"), dict(BL=BL, bld=BR.Builder(BL)))


def land_tokor(ctx):
    """The token engine's --oriented run on rescue3 tokens: instances keyed by (token pair, same-strand flag)."""
    u = ctx.u
    summ = json.loads((TOKOR / "summary.json").read_text())
    assert summ["oriented"] and int(summ["w"]) == 3
    sp = pd.read_parquet(TOKOR / "significant_pairs.parquet")
    tok = ctx.tok_rescue3
    strand = u.genes.strand.to_numpy(np.int64)
    key, left, dist, T = B.R.colocalization(u, tok, W, True, strand)
    skey = sp.key.to_numpy()
    assert np.all(np.diff(skey) > 0)
    assert (skey // 2 == sp.lo.to_numpy() * T + sp.hi.to_numpy()).all()
    pidx = np.minimum(np.searchsorted(skey, key), len(skey) - 1)
    found = skey[pidx] == key
    li, di, pi = left[found], dist[found], pidx[found].astype(np.int64)
    Lv = sp.L.to_numpy().astype(np.float32)
    b, arg = B.R.boundary_support_argmax(li, di, Lv[pi], pi, u.n)
    saved = np.load(TOKOR / "boundary_support.npy")
    eq = bool(np.array_equal(b, saved))
    assert eq, "oriented token boundary support not reproduced"
    info = dict(rule=summ["rule"], fdr=summ["fdr"], significant_pairs=summ["significant_pairs"], instances=int(found.sum()),
                boundary_support_equals_saved=eq, genes_covered=summ.get("genes_covered"))
    return Land4("tokor", b, arg, li, di, "tokor", info, dict(sp=sp))


def landscape(ctx, name):
    if name == "tok":
        return land_tok(ctx)
    if name == "tokor":
        return land_tokor(ctx)
    return land_cand(ctx, name)


# ============================================================================== labels
_MODCACHE = {}


def louvain_pairs(lo, hi, J, Lp, T, res, seed=17):
    import networkx as nx
    H = nx.Graph()
    wt = J * np.log2(np.maximum(Lp, 2).astype(np.float64))
    H.add_weighted_edges_from(zip(lo.tolist(), hi.tolist(), wt.tolist()))
    tm = np.full(T, -1, np.int64)
    for i, c in enumerate(nx.community.louvain_communities(H, weight="weight", resolution=res, seed=seed)):
        if len(c) >= 2:
            tm[np.fromiter(c, dtype=np.int64)] = i
    pm = tm[lo].copy()
    pm[tm[hi] != pm] = -1
    return tm, pm


def labels_for(ctx, land, lab):
    """-> (boundary labels or None, gene_module for the locus table, info)."""
    n = ctx.u.n
    if lab == "none":
        return None, np.full(n, -1, np.int64), {}
    kind, *p = lab.split(":")
    key = (land.name, lab)
    if key in _MODCACHE:
        return _MODCACHE[key]
    if kind in ("vf", "rescue3"):
        if kind == "vf":
            famcand, fam, res = p[0], p[1], float(p[2])
            tok = R2m.tokens_for(ctx, famcand, fam)
        else:
            res = float(p[0])
            tok = ctx.tok_rescue3.copy()
        Ltmp = R2m.Land(land.name, land.sig_left, land.sig_d, np.zeros(len(land.sig_left), np.float32), land.b, land.arg)
        mod = R2m.modules(ctx, Ltmp, tok, res=res)
        labs = np.where(land.arg >= 0, mod["pm"][np.maximum(land.arg, 0)], -2).astype(np.int64)
        out = (labs, mod["gene_module"], mod["info"])
    elif kind == "ic":
        comm = np.load(R4 / "graph" / land.name / f"ic_{p[0]}.npy").astype(np.int64)
        assert len(comm) == len(land.sig_left)
        sz = np.bincount(comm)
        cl = np.where(sz[comm] >= 2, comm, -1)
        labs = np.where(land.arg >= 0, cl[np.maximum(land.arg, 0)], -2).astype(np.int64)
        out = (labs, np.full(n, -1, np.int64), dict(communities=int(len(sz)), nontrivial=int((sz >= 2).sum()),
                                                    instances_in_nontrivial=float((sz[comm] >= 2).mean()),
                                                    largest=int(sz.max())))
    elif kind == "tokmod":
        res = float(p[0])
        if land.kind == "tok":
            BL, bld = land.extra["BL"], land.extra["bld"]
            tm, pm = bld.modules(res)
            labs = BL.boundary_labels(pm)
            gm = np.where(BL.tok >= 0, tm[np.maximum(BL.tok, 0)], -1).astype(np.int64)
        else:
            sp = land.extra["sp"]
            T = int(ctx.tok_rescue3.max()) + 1
            tm, pm = louvain_pairs(sp.lo.to_numpy(), sp.hi.to_numpy(), sp.J.to_numpy(), sp.L.to_numpy(), T, res)
            labs = np.where(land.arg >= 0, pm[np.maximum(land.arg, 0)], -2).astype(np.int64)
            gm = tm[ctx.tok_rescue3]
        out = (labs, gm, dict(modules=int(len(np.unique(tm[tm >= 0])))))
    else:
        raise ValueError(lab)
    _MODCACHE[key] = out
    return out


# ============================================================================== critic
class Critic:
    def __init__(self, n):
        z = np.load(GR / "eval/assembled_cv.npz")
        adj = pd.read_parquet(GR / "prep/adj1.parquet", columns=["i", "j"])
        i = adj.i.to_numpy()
        assert ((adj.j.to_numpy() - i) == 1).all()
        self.n = n
        self.score = np.full(n, np.nan, np.float32)
        self.score[i] = z["primary"]
        self.fold = np.full(n, -1, np.int8)
        self.fold[i] = z["fold_of"]
        self.null = {k: np.load(GR / f"runs/cv{k}/null_scores.npz")["primary"].astype(np.float64) for k in range(5)}
        self.cache = {}

    def on(self, q):
        if q not in self.cache:
            thr = np.array([np.quantile(self.null[k], q) for k in range(5)])
            o = np.zeros(self.n, np.bool_)
            m = self.fold >= 0
            o[m] = self.score[m] > thr[self.fold[m]]
            self.cache[q] = (o, thr)
        return self.cache[q]


@njit(cache=True)
def critic_fill(b, on, cs, ce, m, bridge_only, val):
    """Unsupported gaps (b = 0) inside each contig: a gap of <= m boundaries, all critic-on, between two supported
    boundaries is bridged; otherwise (mode x) up to m critic-on boundaries next to a supported side are added, never
    closing the gap from both sides. Returns (b2, extensions, bridges, boundaries added)."""
    b2 = b.copy()
    n_ext, n_bri, n_add = 0, 0, 0
    for c in range(len(cs)):
        s, e = cs[c], ce[c]
        k = s
        while k < e - 1:
            if b[k] > 0:
                k += 1
                continue
            k0 = k
            while k < e - 1 and b[k] == 0:
                k += 1
            k1 = k - 1
            left_sup = k0 - 1 >= s and b[k0 - 1] > 0
            right_sup = k1 + 1 < e - 1 and b[k1 + 1] > 0
            glen = k1 - k0 + 1
            if left_sup and right_sup and glen <= m:
                allon = True
                for t in range(k0, k1 + 1):
                    if not on[t]:
                        allon = False
                        break
                if allon:
                    for t in range(k0, k1 + 1):
                        b2[t] = val
                    n_bri += 1
                    n_add += glen
                    continue
            if bridge_only:
                continue
            last_left = k0 - 1
            if left_sup:
                lim = k1 - 1 if right_sup else k1
                t = k0
                cnt = 0
                while t <= lim and cnt < m and on[t]:
                    b2[t] = val
                    last_left = t
                    t += 1
                    cnt += 1
                if cnt > 0:
                    n_ext += 1
                    n_add += cnt
            if right_sup:
                t = k1
                cnt = 0
                while t >= last_left + 2 and cnt < m and on[t]:
                    b2[t] = val
                    t -= 1
                    cnt += 1
                if cnt > 0:
                    n_ext += 1
                    n_add += cnt
    return b2, n_ext, n_bri, n_add


def parse_crit(c):
    if c == "none":
        return None
    q = float(c[1:c.index("m")])
    rest = c[c.index("m") + 1:]
    return dict(q=q, m=int(rest[:-1]), bridge_only=(rest[-1] == "b"))


# ============================================================================== readings
def build_tables(ctx, b, lab, gm, f, which):
    s2, e2 = SR.runs_at_least(b, ctx.cs, ctx.ce, 2.0, 0)
    out = {}
    if "R1" in which:
        out["R1"] = (s2, e2)
    if "A" in which:
        out["A"] = B.trim_ends(b, s2, e2, f)
    if lab is not None and any(x in which for x in ("C", "H", "DH")):
        cs_, ce_, _ = B.label_segments(b, lab, s2, e2)
        if "C" in which:
            out["C"] = (cs_, ce_)
        ok = ce_ > cs_
        hs, he = B.trim_ends(b, cs_[ok], ce_[ok], f)
        if "H" in which:
            out["H"] = (hs, he)
        if "DH" in which:
            se = np.unique(np.c_[np.r_[s2, hs], np.r_[e2, he]], axis=0)
            out["DH"] = (se[:, 0], se[:, 1])
    return {k: ctx.table(s, e, gm) for k, (s, e) in out.items()}, out


def covered_mask(n, s, e):
    cov = np.zeros(n + 1, np.int32)
    keep = e > s
    np.add.at(cov, s[keep], 1)
    np.add.at(cov, e[keep] + 1, -1)
    return np.cumsum(cov)[:n] > 0


# ============================================================================== references and the odd score
class Refs:
    def __init__(self, ctx, split):
        self.split = split
        bsc = RD / "boundary/scores/rescue3"
        self.per = {}
        for k, f in (("tok_R1", "R1_conserved2"), ("tok_DH", "DH_coreH_res0.3_f0.5")):
            r = pd.read_parquet(bsc / f"{f}.parquet")
            r["excess"] = r.f1 - r.control_mean_f1
            self.per[k] = r.set_index("instance_id")
        g = self.per["tok_DH"].genome.map(ctx.gcode).to_numpy()
        par = 1 if split == "odd" else 0
        self.summary = B.summarize_per(self.per["tok_DH"][g % 2 == par].reset_index())
        loci = pd.read_parquet(RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet")
        runs = ctx.runs_odd if split == "odd" else ctx.runs_all
        self.ribo = B.ribo_summary(runs, B.ribo_score(runs, loci.gid_s.to_numpy(), loci.gid_e.to_numpy()))
        if split == "odd":
            self.bars = dict(recovered=830, parv=54, resolved=705)
        else:
            self.bars = dict(recovered=1650, parv=110, resolved=1400)

    def margins(self, summ, ribo):
        t, out = self.summary, {}
        out["summed"] = (summ["summed_DF_TXSS_excess"] - (t["summed_DF_TXSS_excess"] - 0.01)) / 0.01
        for c in CALLERS:
            out[c] = (summ[c]["f1"] - (t[c]["f1"] - 0.02)) / 0.02
        tr = self.ribo
        pv = lambda x: int(str(x).split("/")[0])
        for q, v, tv in (("recovered", ribo["recovered"], tr["recovered"]), ("parv", pv(ribo["parvarchaeaceae"]), pv(tr["parvarchaeaceae"])),
                         ("resolved", ribo["resolved"], tr["resolved"])):
            out[q] = (v - self.bars[q]) / max(tv - self.bars[q], 1e-9)
        shortfall = float(sum(max(0.0, -x) for x in out.values()))
        return dict(margins=out, shortfall=shortfall, score=[-shortfall, summ["summed_DF_TXSS_excess"]])


def r2_conditions(summ, ribo, refs):
    """round 2's odd conditions o1 (ribosomal), o2 (summed), o3 (F1) against token DH."""
    t = refs.summary
    pv = int(str(ribo["parvarchaeaceae"]).split("/")[0])
    o1 = ribo["recovered"] >= refs.bars["recovered"] and pv >= refs.bars["parv"] and ribo["resolved"] >= refs.bars["resolved"]
    o2 = summ["summed_DF_TXSS_excess"] >= t["summed_DF_TXSS_excess"] - 0.01
    o3 = all(summ[c]["f1"] >= t[c]["f1"] - 0.02 for c in CALLERS)
    return [bool(o1), bool(o2), bool(o3)]


def verdict(pr, ribo):
    pv = int(str(ribo["parvarchaeaceae"]).split("/")[0])
    ribo_ok = ribo["recovered"] >= 1650 and pv >= 110 and ribo["resolved"] >= 1400
    matches = dict(summed=pr["summed"][0] >= -0.01, **{c: pr[c]["d_f1"][0] >= -0.02 for c in CALLERS}, ribosomal=ribo_ok)
    beats = dict(summed=pr["summed"][0] >= 0.02 and pr["summed"][1] > 0, **{c: pr[c]["d_f1"][0] >= -0.01 for c in CALLERS},
                 ribosomal=ribo_ok)
    return dict(matches=all(matches.values()), beats=all(beats.values()), matches_conditions=matches, beats_conditions=beats,
                blocking_matches=[k for k, v in matches.items() if not v])


# ============================================================================== label diagnostics
def label_diag(ctx, lab, gold_spans):
    r = ctx.runs_all
    n_ = np.zeros(len(r), np.int64)
    ed = np.zeros(len(r), np.int64)
    R2m._span_labels(lab, r.gid_start.to_numpy(np.int64), r.gid_end.to_numpy(np.int64), n_, ed)
    parv = (r.family == "Parvarchaeaceae").to_numpy()
    item = dict(ribosomal=dict(runs=int(len(r)), one_label=int((n_ == 1).sum()), two_plus=int((n_ >= 2).sum()),
                               no_label=int((n_ == 0).sum()), parv_runs=int(parv.sum()), parv_one_label=int(((n_ == 1) & parv).sum()),
                               labels_per_run_q50_q90=np.quantile(n_, [0.5, 0.9]).tolist()))
    for cl, g in gold_spans.groupby("caller"):
        n2 = np.zeros(len(g), np.int64)
        e2 = np.zeros(len(g), np.int64)
        R2m._span_labels(lab, g["min"].to_numpy(np.int64), g["max"].to_numpy(np.int64), n2, e2)
        item[cl] = dict(n=int(len(g)), one_label=float((n2 == 1).mean()), two_plus=float((n2 >= 2).mean()),
                        no_label=float((n2 == 0).mean()), edge_change_both_sides=float(e2.mean()))
    item["labelled_boundaries"] = float((lab >= 0).mean())
    # cross-genome consistency of ribosomal run labels: distinct labels over all runs' boundaries
    labs_all = []
    for s, e in zip(r.gid_start.to_numpy(), r.gid_end.to_numpy()):
        x = lab[s:e]
        labs_all.append(x[x >= 0])
    allv = np.concatenate(labs_all) if labs_all else np.zeros(0, np.int64)
    if len(allv):
        vals, cnt = np.unique(allv, return_counts=True)
        item["ribosomal"]["distinct_labels_all_runs"] = int(len(vals))
        item["ribosomal"]["largest_label_share"] = float(cnt.max() / cnt.sum())
    return item


def gold_spans_odd(ctx):
    inst = pd.read_parquet(B.PKG / "gold/instances.parquet")
    mem = ctx.members.groupby("instance_id").gid.agg(["min", "max", "count"]).reset_index()
    mem = mem.merge(inst[["instance_id", "caller", "genome"]], on="instance_id")
    return mem[(mem["count"] >= 2) & (mem.genome.map(ctx.gcode) % 2 == 1)]


# ============================================================================== run configs
def read_configs(fn):
    out = []
    for line in Path(fn).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = json.loads(line)
            c.setdefault("lab", "none")
            c.setdefault("crit", "none")
            c.setdefault("f", 0.5)
            c.setdefault("readings", ["R1", "C", "H", "DH"])
            out.append(c)
    return out


def run_configs(a, split):
    import numba
    numba.set_num_threads(a.threads)
    t00 = time.time()
    ctx = R2m.Ctx()
    rdir = R4 / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    sdir = R4 / "scores" / split
    sdir.mkdir(parents=True, exist_ok=True)
    fn = rdir / f"{split}_{a.tag}.json"
    out = json.loads(fn.read_text()) if fn.exists() else {}
    refs = Refs(ctx, split)
    out["_refs"] = dict(split=split, tok_DH=refs.summary, tok_DH_ribo=refs.ribo, bars=refs.bars)
    log("token DH reference", split, round(refs.summary["summed_DF_TXSS_excess"], 4),
        {c[:4]: round(refs.summary[c]["f1"], 3) for c in CALLERS}, refs.ribo)
    crit = None
    lands = {}
    base_per = {}           # name -> {reading: per-instance frame}, configs with "keep_per": true (e.g. r2sel)
    gold_sp = None
    tokmask = None
    for c in read_configs(a.configs):
        name = c["name"]
        if name in out and not a.redo and name not in getattr(a, "always", []):
            if not c.get("keep_per"):
                log("skip", name)
                continue
        t0 = time.time()
        if c["land"] not in lands:
            lands[c["land"]] = landscape(ctx, c["land"])
        land = lands[c["land"]]
        lab, gm, labinfo = labels_for(ctx, land, c["lab"])
        b = land.b
        item = dict(config=c, land=land.info, labels=labinfo)
        cr = parse_crit(c["crit"])
        if cr is not None:
            if crit is None:
                crit = Critic(ctx.u.n)
            on, thr = crit.on(cr["q"])
            b, n_ext, n_bri, n_add = critic_fill(land.b, on, ctx.cs, ctx.ce, cr["m"], cr["bridge_only"], np.float32(2.0))
            item["critic"] = dict(thresholds=thr.tolist(), critic_on_unsupported=int((on & (land.b == 0)).sum()),
                                  critic_on_all=int(on.sum()), extensions=int(n_ext), bridges=int(n_bri), boundaries_added=int(n_add),
                                  supported_boundaries_before=int((land.b > 0).sum()), after=int((b > 0).sum()))
        tabs, se = build_tables(ctx, b, lab, gm, float(c["f"]), c["readings"])
        res, pers = R2m.evaluate(ctx, tabs, land, split, a.threads)
        item["readings"] = res
        parity = 1 if split == "odd" else 0
        item["member_support"] = {("odd" if parity else "even"): R2m.member_support(ctx, b, parity)}
        item["supported_boundaries"] = int((b > 0).sum())
        if lab is not None and c.get("diag", False):
            if gold_sp is None:
                gold_sp = gold_spans_odd(ctx)
            item["label_diag"] = label_diag(ctx, lab, gold_sp)
        # odd score / verdict on the primary reading
        prim = "DH" if "DH" in res else c["readings"][-1]
        sp_ = "odd" if split == "odd" else "even"
        summ = res[prim]["summary"][sp_]
        ribo = res[prim]["ribo"]["scored"]
        item["primary"] = prim
        item["score"] = refs.margins(summ, ribo)
        item["r2_conditions"] = r2_conditions(summ, ribo, refs)
        item["paired"] = {}
        for rn, (per, sc) in pers.items():
            pi = per.set_index("instance_id")
            for refk in ("tok_R1", "tok_DH"):
                for spx in ((sp_,) if split == "odd" else ("even", "odd")):
                    item["paired"][f"{spx}:{rn}_vs_{refk}"] = BR.paired(pi, refs.per[refk], ctx.gcode, spx)
            for bn, bp in base_per.items():
                if rn in bp and bn != name:
                    for spx in ((sp_,) if split == "odd" else ("even", "odd")):
                        item["paired"][f"{spx}:{rn}_vs_{bn}_{rn}"] = BR.paired(pi, bp[rn], ctx.gcode, spx)
        if split != "odd" and "DH" in pers:
            item["verdict_vs_tok_DH"] = verdict(item["paired"]["even:DH_vs_tok_DH"], res["DH"]["ribo"]["scored"])
        if c.get("keep_per"):
            base_per[name] = {rn: per.set_index("instance_id") for rn, (per, sc) in pers.items()}
        for rn, path in c.get("check_against", {}).items():
            if rn in pers:
                ref = pd.read_parquet(path).set_index("instance_id")
                pi = pers[rn][0].set_index("instance_id")
                ids = pi.index.intersection(ref.index)
                item.setdefault("checks", {})[rn] = dict(
                    reference=path, n_common=int(len(ids)), n_here=int(len(pi)),
                    f1_maxdiff=float(np.abs(pi.loc[ids].f1.to_numpy() - ref.loc[ids].f1.to_numpy()).max()),
                    control_maxdiff=float(np.abs(pi.loc[ids].control_mean_f1.to_numpy() - ref.loc[ids].control_mean_f1.to_numpy()).max()))
                log("check", name, rn, item["checks"][rn])
        # what the critic added (genes newly in DH loci against the same configuration without the critic)
        if cr is not None and "DH" in se:
            _, se0 = build_tables(ctx, land.b, lab, gm, float(c["f"]), ["DH"])
            n = ctx.u.n
            add = covered_mask(n, *se["DH"]) & ~covered_mask(n, *se0["DH"])
            if tokmask is None:
                ts = np.bincount(ctx.tok_rescue3)
                tokmask = ts[ctx.tok_rescue3] == 1
            par = ctx.gc % 2 == parity
            mem = ctx.members
            memg = {cl: np.zeros(n, bool) for cl in CALLERS}
            for cl, g in mem.groupby("caller"):
                memg[cl][g.gid.to_numpy()] = True
            anym = memg[CALLERS[0]] | memg[CALLERS[1]] | memg[CALLERS[2]]
            item["critic"]["added_genes"] = dict(
                all=int(add.sum()), split_genomes=int((add & par).sum()),
                singleton_family=int((add & tokmask).sum()), singleton_family_split=int((add & tokmask & par).sum()),
                gold_member_split=int((add & anym & par).sum()),
                gold_member_share_split=float((add & anym & par).sum() / max(1, (add & par).sum())),
                gold_member_share_singleton_split=float((add & anym & par & tokmask).sum() / max(1, (add & tokmask & par).sum())),
                base_rate_gold_member_split=float((anym & par).sum() / max(1, par.sum())),
                **{f"members_{cl}_split": int((add & memg[cl] & par).sum()) for cl in CALLERS})
        for rn, (per, sc) in pers.items():
            if c.get("save_per", False):
                per.to_parquet(sdir / f"{a.tag}__{name}__{rn}.parquet", index=False)
        item["seconds"] = round(time.time() - t0, 1)
        out[name] = item
        jdump(fn, out)
        sc_ = item["score"]
        log("done", name, prim, "shortfall", round(sc_["shortfall"], 3), "summed", round(summ["summed_DF_TXSS_excess"], 4),
            {k: round(v, 2) for k, v in sc_["margins"].items()}, item["r2_conditions"], f"{item['seconds']}s")
    log("wrote", fn, f"{time.time() - t00:.0f}s")


def cmd_tune(a):
    run_configs(a, "odd")


def cmd_test(a):
    run_configs(a, "all")


def cmd_diag(a):
    import numba
    numba.set_num_threads(a.threads)
    ctx = R2m.Ctx()
    gold_sp = gold_spans_odd(ctx)
    fn = R4 / "results" / f"diag_{a.tag}.json"
    (R4 / "results").mkdir(parents=True, exist_ok=True)
    out = json.loads(fn.read_text()) if fn.exists() else {}
    lands = {}
    for c in read_configs(a.configs):
        if c["name"] in out and not a.redo:
            continue
        if c["land"] not in lands:
            lands[c["land"]] = landscape(ctx, c["land"])
        land = lands[c["land"]]
        lab, gm, info = labels_for(ctx, land, c["lab"])
        item = label_diag(ctx, lab, gold_sp)
        item["labels"] = info
        out[c["name"]] = item
        jdump(fn, out)
        log("diag", c["name"], json.dumps(item["ribosomal"]), {k: round(v["one_label"], 3) for k, v in item.items()
                                                                 if isinstance(v, dict) and "one_label" in v and k != "ribosomal"})


# ============================================================================== instance graph
@njit(cache=True)
def overlap_edges(sl, sd, inst_pos, n, w):
    M = len(sl)
    cnt = 0
    for q in range(M):
        l = sl[q]
        r = l + sd[q]
        lo = l - w if l - w > 0 else 0
        for l2 in range(lo, r + 1):
            for d2 in range(1, w + 1):
                if l2 + d2 < l or l2 + d2 >= n:
                    continue
                q2 = inst_pos[l2 * w + d2 - 1]
                if q2 > q:
                    cnt += 1
    ea = np.empty(cnt, np.int32)
    eb = np.empty(cnt, np.int32)
    k = 0
    for q in range(M):
        l = sl[q]
        r = l + sd[q]
        lo = l - w if l - w > 0 else 0
        for l2 in range(lo, r + 1):
            for d2 in range(1, w + 1):
                if l2 + d2 < l or l2 + d2 >= n:
                    continue
                q2 = inst_pos[l2 * w + d2 - 1]
                if q2 > q:
                    ea[k] = q
                    eb[k] = q2
                    k += 1
    return ea, eb


_GS = {}


def _leiden(task):
    import random
    import igraph as ig
    key, gamma = task
    G, wt = _GS[key]
    t = time.time()
    random.seed(17)
    ig.set_random_number_generator(random)
    part = G.community_leiden(objective_function="modularity", weights=wt, resolution=gamma, n_iterations=-1)
    q = float(G.modularity(part.membership, weights=wt, resolution=gamma))
    return key, gamma, np.asarray(part.membership, dtype=np.int32), q, round(time.time() - t, 1)


def comm_stats(memb):
    sz = np.bincount(memb)
    nt = sz >= 2
    return dict(communities=int(len(sz)), nontrivial=int(nt.sum()), instances_in_nontrivial=float(nt[memb].mean()),
                largest=int(sz.max()), size_q50_q90_q99_nontrivial=(np.quantile(sz[nt], [0.5, 0.9, 0.99]).tolist() if nt.any() else []))


def cmd_graph(a):
    import multiprocessing as mp
    import igraph as ig
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    su = SR.Universe()
    nd, off, left, d = su.instances(W)
    name = a.land
    gdir = R4 / "graph" / name
    sidx = np.load(cand_dir(name) / "sidx.npy")
    sl, sd = left[sidx].astype(np.int64), d[sidx].astype(np.int64)
    M = len(sidx)
    inst_pos = np.full(su.n * W, -1, np.int64)
    inst_pos[sl * W + sd - 1] = np.arange(M)
    dst = np.load(gdir / "sup_dst.npy")
    mm = np.load(gdir / "sup_m.npy")
    k = a.k
    srcs, dsts, ws = [], [], []
    for t in range(k):
        ok = dst[:, t] >= 0
        srcs.append(np.flatnonzero(ok))
        dsts.append(dst[ok, t].astype(np.int64))
        ws.append(np.exp(-(mm[ok, t].astype(np.float64) - 1.0) / 50.0))
    s_, d_, w_ = np.concatenate(srcs), np.concatenate(dsts), np.concatenate(ws)
    lo, hi = np.minimum(s_, d_), np.maximum(s_, d_)
    keep = lo != hi
    key = lo[keep] * M + hi[keep]
    w_ = w_[keep]
    o = np.argsort(key, kind="stable")
    key, w_ = key[o], w_[o]
    first = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(first)
    sup_key = key[first]
    sup_w = np.maximum.reduceat(w_, starts)
    sea, seb = (sup_key // M).astype(np.int64), (sup_key % M).astype(np.int64)
    oa, ob = overlap_edges(sl, sd, inst_pos, su.n, W)
    info = dict(land=name, instances=M, k=k, support_edges_undirected=int(len(sea)), overlap_edges=int(len(oa)),
                support_weight_q10_q50=np.quantile(sup_w, [0.1, 0.5]).tolist(), seconds_edges=round(time.time() - t0, 1), graphs={})
    log("edges", json.dumps(info))
    wos = [float(x) for x in a.wo.split(",") if x]
    gammas = [float(x) for x in a.gammas.split(",") if x]
    for wo in wos:
        t1 = time.time()
        if wo > 0:
            ea = np.r_[sea, oa.astype(np.int64)]
            eb = np.r_[seb, ob.astype(np.int64)]
            wt = np.r_[sup_w, np.full(len(oa), wo)]
        else:
            ea, eb, wt = sea, seb, sup_w
        G = ig.Graph(n=M, edges=np.column_stack([ea, eb]), directed=False)
        _GS[wo] = (G, wt.tolist())
        tag = f"k{k}o{wo:g}"
        cc = np.asarray(G.connected_components().membership, dtype=np.int32)
        np.save(gdir / f"ic_{tag}_cc.npy", cc)
        info["graphs"][tag] = dict(edges=int(G.ecount()), build_seconds=round(time.time() - t1, 1), cc=comm_stats(cc), leiden={})
        log("graph", tag, G.ecount(), "edges", "cc", json.dumps(info["graphs"][tag]["cc"]))
        jdump(gdir / "graph.json", info)
    tasks = [(wo, g) for wo in wos for g in gammas]
    procs = a.procs if a.procs > 0 else a.threads
    with mp.get_context("fork").Pool(min(len(tasks), procs)) as pool:
        for wo, g, memb, q, secs in pool.imap_unordered(_leiden, tasks):
            tag = f"k{k}o{wo:g}"
            np.save(gdir / f"ic_{tag}_g{g:g}.npy", memb)
            st = comm_stats(memb)
            st.update(modularity=q, seconds=secs)
            info["graphs"][tag]["leiden"][f"{g:g}"] = st
            log("leiden", tag, g, json.dumps(st))
            jdump(gdir / "graph.json", info)
    info["seconds"] = round(time.time() - t0, 1)
    jdump(gdir / "graph.json", info)
    log("graph done", info["seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("graph", "tune", "test", "diag"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--configs", default="")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--land", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--wo", default="0,0.3,1")
    ap.add_argument("--gammas", default="0.3,1,3,10,30,100,1000")
    ap.add_argument("--procs", type=int, default=0, help="graph: Leiden processes (0 = --threads)")
    ap.add_argument("--r4base", default="", help="round-4 output root (default $RD/softrec/r4; lineage runs: r4/lin_<tag>)")
    a = ap.parse_args()
    if a.r4base:
        global R4
        R4 = Path(a.r4base)
    {"graph": cmd_graph, "tune": cmd_tune, "test": cmd_test, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
