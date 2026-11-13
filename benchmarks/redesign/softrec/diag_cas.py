#!/usr/bin/env python
"""ws_cas: why the embedding-only (softrec) method scores lower than the token engine on CRISPR-Cas loci (DPANN).

Embedding side: landscape rw1000_e50 (round 2: rank-weighted zca10 lists, depth 1,000, w = exp(-(t-1)/50), rule
W >= 1 & L >= 2), read by (i) round 4's post hoc configuration 'post' (instance-graph labels ic:k5o0.3_g1, trimming
f 0.8, DH) and (ii) round 2's selected configuration 'r2sel' (validated l2000 modules, Louvain 0.3, f 0.5, DH).
Token side: the rescue3 engine landscape and its DH reading (boundary workstream). Diagnosis only: taxonomy, MMseqs2
families and caller labels are used freely here; nothing below is a run-time step of the method.

usage (each subcommand writes $RD/softrec/cas/results/<cmd>.json)
  diag_cas.py repro  --threads T   token DH / R1 rescored from the stored loci (per-instance F1 equal to the stored
                                   scores), post and r2sel R1 / DH rebuilt and rescored (equal to the stored round-4
                                   scores), even CRISPRCasTyper F1 against the stored JSONs; loci tables -> cas/loci/
  diag_cas.py part1  --threads T   per gold CRISPRCasTyper instance: best locus in token DH vs embedding DH, class,
                                   F1 contribution, landscape vs reading; oracle landscapes (diagnostic)
  diag_cas.py pairs  --threads T   Part 2: instances (d <= 3) inside gold Cas spans, token vs embedding significance,
                                   Cas vs non-Cas at matched token L; Part 4: rule thresholds with genome-wide null FDR,
                                   AUC of stored per-instance statistics against the nulls at fixed L
  diag_cas.py lists  --threads T   Part 3: family members in the depth-1,000 lists of Cas genes vs matched non-Cas
                                   genes; per-lineage recovery of the token's support; reciprocity of supporting hits
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
CAS = RD / "softrec/cas"
sys.path.insert(0, str(RD / "softrec/r4/code"))
import readings4 as R4m  # noqa: E402
import softrec2 as S2  # noqa: E402  (on sys.path through readings4 -> softrec/code)

R2m, B, BR, SR = R4m.R2m, R4m.B, R4m.BR, R4m.SR
R2 = RD / "softrec/r2"
R4 = RD / "softrec/r4"
BD = RD / "boundary"
W = 3
CCT = "CRISPRCasTyper"
CALLERS = R2m.CALLERS
LAND = "rw1000_e50"
EMB_CFG = {
    "post": dict(lab="ic:k5o0.3_g1", f=0.8, stored="post__POST_L1L4_rw1000_e50_k5o0.3_g1_f0.8", json="all_post.json",
                 key="POST_L1L4_rw1000_e50_k5o0.3_g1_f0.8"),
    "r2sel": dict(lab="vf:rw1000_e50:l2000:0.3", f=0.5, stored="main__r2sel", json="all_main.json", key="r2sel"),
}
TOK_LOCI = {"DH": BD / "loci/rescue3/DH_coreH_res0.3_f0.5.parquet", "R1": BD / "loci/rescue3/R1_conserved2.parquet"}
TOK_SCORES = {"DH": BD / "scores/rescue3/DH_coreH_res0.3_f0.5.parquet", "R1": BD / "scores/rescue3/R1_conserved2.parquet"}
LN10 = math.log(10.0)
log = R2m.log


def jdump(path, obj):
    R2m.jdump(path, obj)


def jload(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def parity_of(ctx, genomes):
    return pd.Series(genomes).map(ctx.gcode).to_numpy() % 2


def summ_split(ctx, per):
    g = per.genome.map(ctx.gcode).to_numpy() % 2
    return {sp: B.summarize_per(per[g == (1 if sp == "odd" else 0)].reset_index(drop=True)) for sp in ("even", "odd")}


def brief(s):
    return dict(summed=round(s["summed_DF_TXSS_excess"], 4), **{c[:4]: round(s[c]["f1"], 4) for c in CALLERS})


def gold_frames(ctx):
    inst = pd.read_parquet(B.PKG / "gold/instances.parquet")
    inst["parity"] = inst.genome.map(ctx.gcode).to_numpy() % 2
    sc = inst[inst.resolved.fillna(False) & inst.order_trusted.fillna(False) & (inst.n_members_resolved >= 2)].copy()
    sc["mem"] = sc.member_gids.map(lambda x: np.unique(np.asarray(x, np.int64)))
    sc["m0"] = sc.mem.map(lambda x: int(x[0]))
    sc["m1"] = sc.mem.map(lambda x: int(x[-1]))
    return inst, sc.reset_index(drop=True)


def boot_rate_diff(gA, yA, gB, yB, draws=2000, seed=17):
    """Genome bootstrap of mean(yA) - mean(yB) (both groups resampled with the same genome draw)."""
    rng = np.random.default_rng(seed)
    allg, inv = np.unique(np.r_[gA, gB], return_inverse=True)
    ia, ib = inv[:len(gA)], inv[len(gA):]
    G = len(allg)
    sa, na = np.bincount(ia, yA, G), np.bincount(ia, minlength=G).astype(float)
    sb, nb = np.bincount(ib, yB, G), np.bincount(ib, minlength=G).astype(float)
    pick = rng.integers(0, G, size=(draws, G))
    da = sa[pick].sum(1) / np.maximum(na[pick].sum(1), 1)
    db = sb[pick].sum(1) / np.maximum(nb[pick].sum(1), 1)
    dd = da - db
    return [float(yA.mean() - yB.mean()) if len(yA) and len(yB) else None, float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]


# ============================================================================== repro
def cmp_per(per, stored):
    a_ = per.set_index("instance_id")
    b_ = stored.set_index("instance_id")
    ids = a_.index.intersection(b_.index)
    return dict(n_here=int(len(a_)), n_stored=int(len(b_)), n_common=int(len(ids)),
                f1_maxdiff=float(np.abs(a_.loc[ids].f1.to_numpy() - b_.loc[ids].f1.to_numpy()).max()),
                control_maxdiff=float(np.abs(a_.loc[ids].control_mean_f1.to_numpy() - b_.loc[ids].control_mean_f1.to_numpy()).max()))


def cmd_repro(a):
    t0 = time.time()
    ctx = R2m.Ctx()
    for sub in ("loci", "scores", "results"):
        (CAS / sub).mkdir(parents=True, exist_ok=True)
    fn = CAS / "results/repro.json"
    out = {}
    js_post = jload(R4 / "results/all_post.json")
    out["stored_json_tok_DH_even"] = brief(js_post["_refs"]["tok_DH"])
    for rn in ("DH", "R1"):
        loci = pd.read_parquet(TOK_LOCI[rn])
        per, _ = B.score_table(B.PKG, loci, a.threads)
        per.to_parquet(CAS / f"scores/tok__{rn}.parquet", index=False)
        stored = pd.read_parquet(TOK_SCORES[rn])
        s_here, s_stored = summ_split(ctx, per), summ_split(ctx, stored)
        out[f"tok_{rn}"] = dict(check=cmp_per(per, stored), even=brief(s_here["even"]), odd=brief(s_here["odd"]),
                                stored_even=brief(s_stored["even"]), CCT_even_f1=s_here["even"][CCT]["f1"],
                                CCT_even_f1_stored_scores=s_stored["even"][CCT]["f1"])
        log("token", rn, out[f"tok_{rn}"])
        jdump(fn, out)
    out["tok_DH_CCT_even_equals_json"] = bool(out["tok_DH"]["CCT_even_f1"] == js_post["_refs"]["tok_DH"][CCT]["f1"])
    land = R4m.landscape(ctx, LAND)
    for name, c in EMB_CFG.items():
        lab, gm, info = R4m.labels_for(ctx, land, c["lab"])
        tabs, se = R4m.build_tables(ctx, land.b, lab, gm, float(c["f"]), ["R1", "DH"])
        js = jload(R4 / "results" / c["json"])[c["key"]]["readings"]
        item = {}
        for rn, df in tabs.items():
            df.to_parquet(CAS / f"loci/{name}__{rn}.parquet", index=False)
            per, _ = B.score_table(B.PKG, df, a.threads)
            per.to_parquet(CAS / f"scores/{name}__{rn}.parquet", index=False)
            stored = pd.read_parquet(R4 / "scores/all" / f"{c['stored']}__{rn}.parquet")
            s_here = summ_split(ctx, per)
            item[rn] = dict(check=cmp_per(per, stored), even=brief(s_here["even"]), odd=brief(s_here["odd"]),
                            json_even=brief(js[rn]["summary"]["even"]),
                            CCT_even_f1=s_here["even"][CCT]["f1"], CCT_even_f1_json=js[rn]["summary"]["even"][CCT]["f1"],
                            CCT_even_equal_json=bool(s_here["even"][CCT]["f1"] == js[rn]["summary"]["even"][CCT]["f1"]),
                            loci=int(len(df)))
            log(name, rn, item[rn])
        out[name] = dict(labels=info, readings=item)
        jdump(fn, out)
    out["seconds"] = round(time.time() - t0, 1)
    jdump(fn, out)
    log("repro done", out["seconds"])


# ============================================================================== part 1
class LociIdx:
    def __init__(self, df):
        s = df.gid_s.to_numpy(np.int64)
        e = df.gid_e.to_numpy(np.int64)
        o = np.lexsort((e, s))
        self.s, self.e = s[o], e[o]
        self.smax = int((self.e - self.s).max()) + 1 if len(s) else 1
        self.keys = set(zip(self.s.tolist(), self.e.tolist()))

    def cand(self, lo, hi):
        i0 = np.searchsorted(self.s, lo - self.smax, "left")
        i1 = np.searchsorted(self.s, hi, "right")
        j = np.arange(i0, i1)
        return j[self.e[j] >= lo]

    def best(self, mem):
        """The scorer's rule: max F1, then max coverage, min span, min start, min end; None if no member is covered."""
        j = self.cand(int(mem[0]), int(mem[-1]))
        if len(j) == 0:
            return None
        s, e = self.s[j], self.e[j]
        ov = np.searchsorted(mem, e, "right") - np.searchsorted(mem, s, "left")
        if ov.max() <= 0:
            return None
        k = len(mem)
        span = e - s + 1
        f1 = 2.0 * ov.astype(np.float64) / (k + span)
        cov = ov.astype(np.float64) / k
        o = np.lexsort((e, s, span, -cov, -f1))[0]
        return dict(s=int(s[o]), e=int(e[o]), ov=int(ov[o]), cov=float(cov[o]), f1=float(f1[o]), span=int(span[o]))

    def union_cov(self, mem):
        j = self.cand(int(mem[0]), int(mem[-1]))
        c = np.zeros(len(mem), bool)
        for s, e in zip(self.s[j], self.e[j]):
            c |= (mem >= s) & (mem <= e)
        return int(c.sum())

    def disjoint_neighbour(self, lo, hi, m0, m1):
        """True if a locus intersecting [lo, hi] lies entirely outside the gold span [m0, m1]."""
        j = self.cand(lo, hi)
        s, e = self.s[j], self.e[j]
        return bool(((e < m0) | (s > m1)).any())


def run_of(s2, e2, s, e):
    """Index of the level-2 run (genes [s2, e2]) that contains genes [s, e], or -1."""
    i = np.searchsorted(s2, s, "right") - 1
    if i >= 0 and e2[i] >= e:
        return int(i)
    return -1


def sub_class(R, Q, Q_union, merged):
    """R = the better best locus, Q = the worse one (same gold instance)."""
    if Q is None:
        return "missed"
    if Q["ov"] < R["ov"]:
        if Q_union > Q["ov"]:
            return "fragmented"
        if (Q["span"] - Q["ov"]) > (R["span"] - R["ov"]):
            return "shifted"
        return "truncated"
    return "merged" if merged else "over-extended"


def cause_of(R, b_worse, s2w, e2w, s2r, e2r):
    """Landscape vs reading for the worse output: R = the better output's best locus; b_worse = the worse output's
    boundary support; runs (s2, e2) of both outputs' landscapes."""
    seg = b_worse[R["s"]:R["e"]]
    nmiss = int((seg <= 0).sum())
    if nmiss > 0:
        return "landscape:missing_support", nmiss
    iw = run_of(s2w, e2w, R["s"], R["e"])
    ir = run_of(s2r, e2r, R["s"], R["e"])
    if iw >= 0 and ir >= 0 and (s2w[iw] < s2r[ir] or e2w[iw] > e2r[ir]):
        return "reading:extra_support", 0
    return "reading:same_support", 0


def classify_all(ctx, cct, tokL, embL, tokR1, embR1, tok_b, emb_b, cs, ce):
    s2t, e2t = SR.runs_at_least(tok_b, cs, ce, 2.0, 0)
    s2e, e2e = SR.runs_at_least(emb_b, cs, ce, 2.0, 0)
    rows = []
    for r in cct.itertuples():
        mem = r.mem
        T = tokL.best(mem)
        X = embL.best(mem)
        ft = T["f1"] if T else 0.0
        fx = X["f1"] if X else 0.0
        row = dict(instance_id=r.instance_id, parity=r.parity, subtype=r.system_subtype, k=len(mem), gspan=r.m1 - r.m0 + 1,
                   f1_tok=ft, f1_emb=fx, d=fx - ft)
        for nm, Bx in (("tok", T), ("emb", X)):
            for kk in ("s", "e", "ov", "span"):
                row[f"{nm}_{kk}"] = Bx[kk] if Bx else -1
        row["tok_kind"] = ("R1" if (T["s"], T["e"]) in tokR1.keys else "core") if T else "none"
        row["emb_kind"] = ("R1" if (X["s"], X["e"]) in embR1.keys else "core") if X else "none"
        bnd = slice(r.m0, r.m1)
        row["span_sup_tok"] = float((tok_b[bnd] > 0).mean())
        row["span_sup_emb"] = float((emb_b[bnd] > 0).mean())
        if abs(fx - ft) < 1e-12:
            row["cls"] = "equal" if (T is None or X is None or (T["s"], T["e"]) == (X["s"], X["e"])) else "equal_f1_other_locus"
            row["cause"] = ""
            row["nmiss"] = 0
        elif fx < ft:
            merged = X is not None and tokL.disjoint_neighbour(X["s"], X["e"], r.m0, r.m1)
            row["cls"] = "loss:" + sub_class(T, X, embL.union_cov(mem) if X else 0, merged)
            row["cause"], row["nmiss"] = cause_of(T, emb_b, s2e, e2e, s2t, e2t)
        else:
            merged = T is not None and embL.disjoint_neighbour(T["s"], T["e"], r.m0, r.m1)
            row["cls"] = "gain:" + sub_class(X, T, tokL.union_cov(mem) if T else 0, merged)
            row["cause"], row["nmiss"] = cause_of(X, tok_b, s2t, e2t, s2e, e2e)
        rows.append(row)
    return pd.DataFrame(rows)


def class_tables(df):
    out = {}
    for sp, par in (("even", 0), ("odd", 1)):
        g = df[df.parity == par]
        N = len(g)
        tab = g.groupby("cls").agg(n=("d", "size"), dF1_sum=("d", "sum")).reset_index()
        tab["contribution"] = tab.dF1_sum / N
        cz = g[g.cls.str.startswith(("loss", "gain"))].groupby(["cls", "cause"]).agg(n=("d", "size"), dF1_sum=("d", "sum")).reset_index()
        cz["contribution"] = cz.dF1_sum / N
        bycause = g[g.cls.str.startswith(("loss", "gain"))].copy()
        bycause["dir"] = np.where(bycause.cls.str.startswith("loss"), "loss", "gain")
        cc = bycause.groupby(["dir", "cause"]).agg(n=("d", "size"), dF1_sum=("d", "sum")).reset_index()
        cc["contribution"] = cc.dF1_sum / N
        kinds = g[g.cls.str.startswith("loss")].groupby(["tok_kind", "emb_kind"]).agg(n=("d", "size"), dF1_sum=("d", "sum")).reset_index()
        kinds["contribution"] = kinds.dF1_sum / N
        st = g.groupby("subtype").agg(n=("d", "size"), dF1_mean=("d", "mean"), f1_tok=("f1_tok", "mean"), f1_emb=("f1_emb", "mean"),
                                      lost=("d", lambda x: int((x < -1e-12).sum())), gained=("d", lambda x: int((x > 1e-12).sum())))
        st = st.sort_values("n", ascending=False).head(12).reset_index()
        out[sp] = dict(n=int(N), f1_tok=float(g.f1_tok.mean()), f1_emb=float(g.f1_emb.mean()), dF1=float(g.d.mean()),
                       classes=tab.to_dict("records"), class_cause=cz.to_dict("records"), cause=cc.to_dict("records"),
                       loss_locus_kinds=kinds.to_dict("records"), subtypes=st.to_dict("records"),
                       span_support_tok_mean=float(g.span_sup_tok.mean()), span_support_emb_mean=float(g.span_sup_emb.mean()),
                       span_fully_supported_tok=float((g.span_sup_tok == 1).mean()), span_fully_supported_emb=float((g.span_sup_emb == 1).mean()))
    return out


def cas_near_mask(n, cct, cs_of_gene, ce_of_gene, margin=3):
    near = np.zeros(n + 1, np.int64)
    for r in cct.itertuples():
        lo = max(r.m0 - margin, int(cs_of_gene[r.m0]))
        hi = min(r.m1 + margin, int(ce_of_gene[r.m0]) - 1)
        near[lo] += 1
        near[hi + 1] -= 1
    return np.cumsum(near)[:n] > 0


def cmd_part1(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    ctx = R2m.Ctx()
    fn = CAS / "results/part1.json"
    out = jload(fn)
    inst, sc = gold_frames(ctx)
    cct = sc[sc.caller == CCT].reset_index(drop=True)
    # members as gids agree with the scorer's positions
    ct = pd.read_parquet(B.PKG / "universe/contigs.parquet", columns=["contig_code", "first_gid"]).set_index("contig_code").first_gid
    bad = sum(int((np.unique(np.asarray(p, np.int64)) + int(ct.loc[cc]) != m).any()) for p, cc, m in zip(cct.member_positions, cct.contig_code, cct.mem))
    assert bad == 0, bad
    land = R4m.landscape(ctx, LAND)
    BL = B.Landscape(B.ARMS["rescue3"])
    cs, ce = ctx.cs, ctx.ce
    tokD, tokR = LociIdx(pd.read_parquet(TOK_LOCI["DH"])), LociIdx(pd.read_parquet(TOK_LOCI["R1"]))
    per_tok = pd.read_parquet(CAS / "scores/tok__DH.parquet").set_index("instance_id")
    out["n_cct_scored"] = dict(all=int(len(cct)), even=int((cct.parity == 0).sum()), odd=int((cct.parity == 1).sum()))
    for name in EMB_CFG:
        embD = LociIdx(pd.read_parquet(CAS / f"loci/{name}__DH.parquet"))
        embR = LociIdx(pd.read_parquet(CAS / f"loci/{name}__R1.parquet"))
        df = classify_all(ctx, cct, tokD, embD, tokR, embR, BL.b, land.b, cs, ce)
        per_emb = pd.read_parquet(CAS / f"scores/{name}__DH.parquet").set_index("instance_id")
        chk = dict(tok_f1_maxdiff=float(np.abs(df.f1_tok.to_numpy() - per_tok.loc[df.instance_id].f1.to_numpy()).max()),
                   emb_f1_maxdiff=float(np.abs(df.f1_emb.to_numpy() - per_emb.loc[df.instance_id].f1.to_numpy()).max()))
        log(name, "best-locus check vs scorer", chk)
        df.to_parquet(CAS / f"results/part1_{name}.parquet", index=False)
        out[name] = dict(check=chk, tables=class_tables(df))
        # R1-level F1 for context
        r1t = pd.read_parquet(CAS / "scores/tok__R1.parquet").set_index("instance_id")
        r1e = pd.read_parquet(CAS / f"scores/{name}__R1.parquet").set_index("instance_id")
        ids = cct.instance_id
        out[name]["R1_CCT_f1"] = {sp: dict(tok=float(r1t.loc[ids[cct.parity == p]].f1.mean()), emb=float(r1e.loc[ids[cct.parity == p]].f1.mean()))
                                  for sp, p in (("even", 0), ("odd", 1))}
        jdump(fn, out)
        log(name, json.dumps({sp: {k: v for k, v in out[name]["tables"][sp].items() if k in ("n", "f1_tok", "f1_emb", "dF1")} for sp in ("even", "odd")}))
    # ---------------------------------------------------------------- oracle landscapes (post hoc reading)
    su = SR.Universe()
    nd, off, left, d = su.instances(W)
    n = su.n
    cd = R2 / "cand" / LAND
    sidx = np.load(cd / "sidx.npy")
    val = np.load(cd / "value.npy").astype(np.float32)
    b0, arg0 = SR.boundary_support_argmax(left[sidx], d[sidx].astype(np.int64), val, np.arange(len(sidx), dtype=np.int64), n)
    assert np.array_equal(b0, land.b) and np.array_equal(arg0, land.arg)
    comm = np.load(R4 / "graph" / LAND / "ic_k5o0.3_g1.npy").astype(np.int64)
    sz = np.bincount(comm)
    cl = np.where(sz[comm] >= 2, comm, -1)
    tokidx = np.unique(off[BL.inst_left] + BL.inst_dist - 1)
    tokS = np.zeros(len(left), bool)
    tokS[tokidx] = True
    tokLval = np.zeros(len(left), np.float32)
    tokLval[off[BL.inst_left] + BL.inst_dist - 1] = BL.sp.L.to_numpy().astype(np.float32)[BL.inst_pair]
    z = np.load(R2 / "stats/obs.npz")
    vK = [str(x) for x in z["variants"]].index("K1000_fnone")
    Lraw = z["L"][vK].astype(np.int32)
    del z
    Lt = np.where(Lraw >= 0, Lraw + 1, 2).astype(np.float32)
    embS = np.zeros(len(left), bool)
    embS[sidx] = True
    assert np.array_equal(su.c_start, cs) and np.array_equal(su.c_end, ce)
    gene_cs = su.c_start[su.cid]
    gene_ce = su.c_end[su.cid]
    near_gene = cas_near_mask(n, cct, gene_cs, gene_ce, 3)
    cg = np.r_[0, np.cumsum(near_gene)]
    touch = (cg[left + d + 1] - cg[left]) > 0
    add_g = tokidx[~embS[tokidx]]
    add_c = add_g[touch[add_g]]
    keep_tok = tokS[sidx]
    near_s = touch[sidx]
    oracles = {
        "O1g_add_token_sig_all": (np.ones(len(sidx), bool), add_g, False),
        "O1c_add_token_sig_near_cas": (np.ones(len(sidx), bool), add_c, False),
        "O2g_keep_token_sig_only": (keep_tok, np.zeros(0, np.int64), False),
        "O2c_drop_nontoken_near_cas": (keep_tok | ~near_s, np.zeros(0, np.int64), False),
        "O3c_token_set_near_cas": (keep_tok | ~near_s, add_c, False),
        "O4c_token_set_and_token_L_near_cas": (keep_tok | ~near_s, add_c, True),
    }
    base_per = pd.read_parquet(CAS / "scores/post__DH.parquet").set_index("instance_id")
    gm = np.full(n, -1, np.int64)
    res = out.get("oracles", {})
    for oname, (keep, add, tokval) in oracles.items():
        idx = np.r_[sidx[keep], add].astype(np.int64)
        v = np.r_[val[keep], np.maximum(Lt[add], 2.0)].astype(np.float32)
        if tokval:
            m_ = touch[idx] & tokS[idx]
            v[m_] = tokLval[idx[m_]]
        labi = np.r_[cl[keep], np.full(len(add), -1, np.int64)]
        b, arg = SR.boundary_support_argmax(left[idx], d[idx].astype(np.int64), v, np.arange(len(idx), dtype=np.int64), n)
        lab = np.where(arg >= 0, labi[np.maximum(arg, 0)], -2).astype(np.int64)
        tabs, _ = R4m.build_tables(ctx, b, lab, gm, 0.8, ["R1", "DH"])
        item = dict(instances=int(len(idx)), added=int(len(add)), dropped=int((~keep).sum()), token_L_values=bool(tokval))
        for rn, df_ in tabs.items():
            per, _ = B.score_table(B.PKG, df_, a.threads)
            s_ = summ_split(ctx, per)
            item[rn] = dict(even=brief(s_["even"]), odd=brief(s_["odd"]))
            if rn == "DH":
                pi = per.set_index("instance_id")
                pr = BR.paired(pi, base_per, ctx.gcode, "even")
                item["DH_vs_post_even"] = dict(summed=pr["summed"], **{c[:4]: pr[c]["d_f1"] for c in CALLERS})
                pr2 = BR.paired(pi, base_per, ctx.gcode, "odd")
                item["DH_vs_post_odd"] = dict(summed=pr2["summed"], **{c[:4]: pr2[c]["d_f1"] for c in CALLERS})
                sc_ = B.ribo_score(ctx.runs_all, df_.gid_s.to_numpy(), df_.gid_e.to_numpy())
                rb = B.ribo_summary(ctx.runs_all, sc_)
                item["ribo"] = dict(recovered=rb["recovered"], resolved=rb["resolved"], parv=rb["parvarchaeaceae"])
        res[oname] = item
        out["oracles"] = res
        jdump(fn, out)
        log("oracle", oname, json.dumps(item))
    out["seconds"] = round(time.time() - t0, 1)
    jdump(fn, out)
    log("part1 done", out["seconds"])


# ============================================================================== attribution (after part 1 / pairs)
def score_land(ctx, n, left, d, idx, val, labi, f, base_per, threads, which=("DH",)):
    """Post hoc reading on an instance set: boundary support / arg from (idx, val), labels from labi (per instance;
    -1 = unlabelled), DH with trimming f; scored on all genomes; paired against base_per (even and odd)."""
    b, arg = SR.boundary_support_argmax(left[idx], d[idx].astype(np.int64), val.astype(np.float32), np.arange(len(idx), dtype=np.int64), n)
    lab = np.where(arg >= 0, labi[np.maximum(arg, 0)], -2).astype(np.int64)
    tabs, _ = R4m.build_tables(ctx, b, lab, np.full(n, -1, np.int64), f, list(which))
    item = {}
    for rn, df_ in tabs.items():
        per, _ = B.score_table(B.PKG, df_, threads)
        s_ = summ_split(ctx, per)
        item[rn] = dict(even=brief(s_["even"]), odd=brief(s_["odd"]))
        pi = per.set_index("instance_id")
        for sp in ("even", "odd"):
            pr = BR.paired(pi, base_per, ctx.gcode, sp)
            item[f"{rn}_vs_base_{sp}"] = dict(summed=pr["summed"], **{c[:4]: pr[c]["d_f1"] for c in CALLERS})
        sc_ = B.ribo_score(ctx.runs_all, df_.gid_s.to_numpy(), df_.gid_e.to_numpy())
        rb = B.ribo_summary(ctx.runs_all, sc_)
        item[f"{rn}_ribo"] = dict(recovered=rb["recovered"], resolved=rb["resolved"], parv=rb["parvarchaeaceae"],
                                  parv_resolved=rb.get("parvarchaeaceae_resolved"))
    return item, b


def cmd_attrib(a):
    """Which instance-level mechanism empties the Cas loci: homolog exclusion (embedding L = -1) or the rank-weight
    floor (W < 1). Boundary-level attribution per gold CCT locus, genome-wide exclusion rates, targeted oracles and the
    method-level Poisson union rule (W >= 1 or P(G | E) <= 1e-e, with L >= 2, depth 1,000)."""
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    fn = CAS / "results/attrib.json"
    out = jload(fn)
    ctx = R2m.Ctx()
    su = SR.Universe()
    nd, off, left, d = su.instances(W)
    n, n_inst = su.n, len(left)
    inst, sc = gold_frames(ctx)
    cct = sc[sc.caller == CCT].reset_index(drop=True)
    BL = B.Landscape(B.ARMS["rescue3"])
    land = R4m.landscape(ctx, LAND)
    tokidx = np.unique(off[BL.inst_left] + BL.inst_dist - 1)
    tokS = np.zeros(n_inst, bool)
    tokS[tokidx] = True
    tokLv = np.zeros(n_inst, np.int64)
    tokLv[off[BL.inst_left] + BL.inst_dist - 1] = BL.sp.L.to_numpy()[BL.inst_pair]
    P = load_pass(R2 / "stats/obs.npz")
    Lt = P["Lt1000"]
    excl = Lt < 0
    sidx = np.load(R2 / "cand" / LAND / "sidx.npy")
    embS = np.zeros(n_inst, bool)
    embS[sidx] = True
    labs = pd.read_parquet(CAS / "cas_gene_labels.parquet")
    is_cas_gene = np.zeros(n, bool)
    is_cas_gene[labs.gid.to_numpy()] = True
    gclass = np.full(n, "", dtype=object)
    gclass[labs.gid.to_numpy()] = labs.gene.fillna("?").to_numpy()
    span_gene = np.zeros(n + 1, np.int64)
    for r in cct.itertuples():
        span_gene[r.m0] += 1
        span_gene[r.m1 + 1] -= 1
    in_span = np.cumsum(span_gene)[:n] > 0
    casish = is_cas_gene | in_span
    cas_inst = casish[left] & casish[left + d]
    noncas = ~casish[left] & ~casish[left + d]
    # ---------------- genome-wide exclusion rates
    ex = {}
    for nm, m in (("all", np.ones(n_inst, bool)), ("tokS_noncas", tokS & noncas), ("tokS_cas_both_genes", tokS & cas_inst),
                  ("cas_both_genes", cas_inst), ("noncas", noncas)):
        ex[nm] = dict(n=int(m.sum()), excluded=int((m & excl).sum()), share=float((m & excl).sum() / max(1, m.sum())),
                      lost=int((m & tokS & ~embS).sum()), lost_excluded=int((m & tokS & ~embS & excl).sum()))
    ex["share_of_lost_that_is_excluded"] = dict(cas=float(ex["tokS_cas_both_genes"]["lost_excluded"] / max(1, ex["tokS_cas_both_genes"]["lost"])),
                                                noncas=float(ex["tokS_noncas"]["lost_excluded"] / max(1, ex["tokS_noncas"]["lost"])))
    m = tokS & cas_inst & excl
    pairs = pd.Series(gclass[left[m]]).replace("", "-") + "|" + pd.Series(gclass[(left + d)[m]]).replace("", "-")
    ex["tokS_cas_excluded_class_pairs"] = {str(k): int(v) for k, v in pairs.value_counts().head(20).items()}
    out["exclusion"] = ex
    log("exclusion", json.dumps({k: v for k, v in ex.items() if k != "tokS_cas_excluded_class_pairs"}))
    jdump(fn, out)
    # ---------------- boundary-level attribution per gold CCT locus (post reading)
    p1 = pd.read_parquet(CAS / "results/part1_post.parquet").set_index("instance_id")
    pois4 = (Lt >= 2) & (P["logP1000"] <= -4 * LN10)
    # for each boundary: does a token-significant instance spanning it have status excluded / pois4 / floor ?
    def span_any(mask):
        return B.span_count(left[mask], d[mask].astype(np.int64), n) > 0
    bnd_tok = span_any(tokS)
    bnd_tok_excl = span_any(tokS & excl)
    bnd_tok_pois = span_any(tokS & ~excl & ~embS & pois4)
    bnd_tok_floor = span_any(tokS & ~excl & ~embS & ~pois4)
    bnd_pois_any = span_any(pois4 & ~embS)            # method-level union rule adds these boundaries
    emb_b = land.b
    rows = []
    for r in cct.itertuples():
        x = p1.loc[r.instance_id]
        if not (x.d < 0 and x.cause == "landscape:missing_support"):
            continue
        s_, e_ = int(x.tok_s), int(x.tok_e)
        kk = np.arange(s_, e_)
        kk = kk[emb_b[kk] <= 0]
        rows.append(dict(instance_id=r.instance_id, parity=int(x.parity), d=float(x.d), cls=x.cls, n_missing=int(len(kk)),
                         any_excl=bool(bnd_tok_excl[kk].any()), all_excl=bool(bnd_tok_excl[kk].all()),
                         any_pois=bool(bnd_tok_pois[kk].any()), all_pois=bool((bnd_tok_pois[kk] | bnd_tok_excl[kk]).all()),
                         all_by_union=bool(bnd_pois_any[kk].all()), any_by_union=bool(bnd_pois_any[kk].any()),
                         any_floor=bool(bnd_tok_floor[kk].any())))
    at = pd.DataFrame(rows)
    at.to_parquet(CAS / "results/attrib_loci_post.parquet", index=False)
    att = {}
    for sp_, par in (("even", 0), ("odd", 1)):
        g = at[at.parity == par]
        N = int((cct.parity == par).sum())
        def mech(z):
            if z.all_excl:
                return "excluded_only"
            if z.any_excl:
                return "excluded_and_floor"
            return "floor_only"
        mm_ = g.apply(mech, axis=1)
        t = g.assign(mech=mm_).groupby("mech").agg(n=("d", "size"), dF1_sum=("d", "sum")).reset_index()
        t["contribution"] = t.dF1_sum / N
        u = g.groupby("all_by_union").agg(n=("d", "size"), dF1_sum=("d", "sum")).reset_index()
        u["contribution"] = u.dF1_sum / N
        att[sp_] = dict(N=N, loci=int(len(g)), loss=float(g.d.sum() / N), by_mechanism=t.to_dict("records"),
                        all_missing_boundaries_marked_by_pois4_union=u.to_dict("records"))
    out["attribution_post"] = att
    log("attribution", json.dumps(att))
    jdump(fn, out)
    # ---------------- targeted oracles and method-level union rules (post reading; added instances unlabelled)
    comm = np.load(R4 / "graph" / LAND / "ic_k5o0.3_g1.npy").astype(np.int64)
    sz = np.bincount(comm)
    cl = np.where(sz[comm] >= 2, comm, -1)
    val = np.load(R2 / "cand" / LAND / "value.npy").astype(np.float32)
    base_per = pd.read_parquet(CAS / "scores/post__DH.parquet").set_index("instance_id")
    nulls = [load_pass(R2 / f"stats/null{r}.npz") for r in range(3)]
    gene_cs = su.c_start[su.cid]
    gene_ce = su.c_end[su.cid]
    near_gene = cas_near_mask(n, cct, gene_cs, gene_ce, 3)
    cg = np.r_[0, np.cumsum(near_gene)]
    near = (cg[left + d + 1] - cg[left]) > 0
    Ltv = np.where(Lt >= 2, Lt, 2).astype(np.float32)
    sets = {
        "A1_tokS_excluded_all": tokS & excl,
        "A2_tokS_floor_all": tokS & ~excl & ~embS,
        "A5_tokS_excluded_near_cas": tokS & excl & near,
        "A6_tokS_floor_near_cas": tokS & ~excl & ~embS & near,
    }
    for e_ in (3, 4, 6):
        sets[f"U{e_}_union_pois1e-{e_}"] = (Lt >= 2) & (P["logP1000"] <= -e_ * LN10) & ~embS
    res = out.get("oracles", {})
    for nm, msk in sets.items():
        add = np.flatnonzero(msk)
        idx = np.r_[sidx, add].astype(np.int64)
        v = np.r_[val, Ltv[add]]
        labi = np.r_[cl, np.full(len(add), -1, np.int64)]
        item, b = score_land(ctx, n, left, d, idx, v, labi, 0.8, base_per, a.threads)
        item["added"] = int(len(add))
        item["added_tokS"] = int(tokS[add].sum())
        if nm.startswith("U"):
            e_ = int(nm[1])
            nn = [int((((Q["Lt1000"] >= 2) & (Q["We50"] >= 1)) | ((Q["Lt1000"] >= 2) & (Q["logP1000"] <= -e_ * LN10))).sum()) for Q in nulls]
            obs_n = int(len(idx))
            item["rule_fdr"] = dict(observed=obs_n, null_mean=float(np.mean(nn)), fdr=float(np.mean(nn) / obs_n))
        item["member_support_even"] = R2m.member_support(ctx, b, 0)
        res[nm] = item
        out["oracles"] = res
        jdump(fn, out)
        log("oracle", nm, json.dumps(item))
    out["seconds"] = round(time.time() - t0, 1)
    jdump(fn, out)
    log("attrib done", out["seconds"])


def cmd_callers(a):
    """Part 1 classes and in-span instance counts for all three callers (post vs token DH; stored loci only)."""
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    fn = CAS / "results/callers.json"
    out = {}
    ctx = R2m.Ctx()
    su = SR.Universe()
    nd, off, left, d = su.instances(W)
    n_inst = len(left)
    inst, sc = gold_frames(ctx)
    land = R4m.landscape(ctx, LAND)
    BL = B.Landscape(B.ARMS["rescue3"])
    tokD, tokR = LociIdx(pd.read_parquet(TOK_LOCI["DH"])), LociIdx(pd.read_parquet(TOK_LOCI["R1"]))
    embD, embR = LociIdx(pd.read_parquet(CAS / "loci/post__DH.parquet")), LociIdx(pd.read_parquet(CAS / "loci/post__R1.parquet"))
    tokS = np.zeros(n_inst, bool)
    tokS[off[BL.inst_left] + BL.inst_dist - 1] = True
    tokL = np.zeros(n_inst, np.int64)
    tokL[off[BL.inst_left] + BL.inst_dist - 1] = BL.sp.L.to_numpy()[BL.inst_pair]
    sidx = np.load(R2 / "cand" / LAND / "sidx.npy")
    embS = np.zeros(n_inst, bool)
    embS[sidx] = True
    z = np.load(R2 / "stats/obs.npz")
    vK = [str(x) for x in z["variants"]].index("K1000_fnone")
    excl = z["L"][vK] < 0
    del z
    for caller in CALLERS:
        g = sc[sc.caller == caller].reset_index(drop=True)
        df = classify_all(ctx, g, tokD, embD, tokR, embR, BL.b, land.b, ctx.cs, ctx.ce)
        tabs = class_tables(df)
        idx = []
        for r in g.itertuples():
            for l_ in range(r.m0, r.m1):
                for dd in range(1, W + 1):
                    if l_ + dd > r.m1:
                        break
                    idx.append(int(off[l_] + dd - 1))
        idx = np.unique(np.array(idx, np.int64))
        ts = tokS[idx]
        out[caller] = dict(
            loci=dict(all=int(len(g)), even=int((g.parity == 0).sum())),
            span_genes_q50=float((g.m1 - g.m0 + 1).median()), members_q50=float(g.n_members_resolved.median()),
            even={k: tabs["even"][k] for k in ("n", "f1_tok", "f1_emb", "dF1", "span_support_tok_mean", "span_support_emb_mean",
                                               "span_fully_supported_tok", "span_fully_supported_emb", "classes", "cause")},
            span_instances=int(len(idx)), tokS=int(ts.sum()), embS=int(embS[idx].sum()), lost=int((ts & ~embS[idx]).sum()),
            lost_share=float((ts & ~embS[idx]).sum() / max(1, ts.sum())), lost_excluded=int((ts & ~embS[idx] & excl[idx]).sum()),
            tokS_tokL_2_7_share=float(((tokL[idx] >= 2) & (tokL[idx] <= 7))[ts].mean()) if ts.any() else None,
            tokS_per_locus=float(ts.sum() / max(1, len(g))))
        log("caller", caller, json.dumps({k: v for k, v in out[caller].items() if k != "even"}),
            json.dumps({k: out[caller]["even"][k] for k in ("n", "f1_tok", "f1_emb", "dF1")}))
        jdump(fn, out)
    out["seconds"] = round(time.time() - t0, 1)
    jdump(fn, out)


# ============================================================================== part 2 + part 4
WN = [x[0] for x in S2.WEIGHTS]


def load_pass(fn):
    z = np.load(fn)
    variants = [str(x) for x in z["variants"]]
    v = {k: variants.index(f"K{k}_fnone") for k in (50, 200, 1000)}
    Lr = z["L"]
    P = {}
    for k, vi in v.items():
        l_ = Lr[vi].astype(np.int32)
        P[f"Lt{k}"] = np.where(l_ >= 0, l_ + 1, -1)
    Wz = z["W"]
    for j, wn in enumerate(WN):
        P[f"W{wn}"] = np.ascontiguousarray(Wz[v[1000], j]).astype(np.float32)
    P["W200e50"] = np.ascontiguousarray(Wz[v[200], 0]).astype(np.float32)
    own = z["own"].astype(np.float64)
    for k in (50, 1000):
        P[f"G{k}"] = z["G"][v[k]].astype(np.int32) + 1
        P[f"E{k}"] = z["E"][v[k]].astype(np.float64) + own
        m = P[f"Lt{k}"] >= 2
        P[f"logP{k}"] = SR.logp_poisson(P[f"G{k}"], P[f"E{k}"], m)
    del z, Wz, Lr
    return P


STATS = [  # name, function of pass dict, higher = more evidence
    ("We50", lambda P: P["We50"]), ("We100", lambda P: P["We100"]), ("We200", lambda P: P["We200"]),
    ("We400", lambda P: P["We400"]), ("Wh20", lambda P: P["Wh20"]), ("Wh100", lambda P: P["Wh100"]),
    ("W200e50", lambda P: P["W200e50"]), ("Lt200", lambda P: P["Lt200"]), ("Lt50", lambda P: P["Lt50"]),
    ("G1000", lambda P: P["G1000"]), ("negE1000", lambda P: -P["E1000"]), ("G_over_E1000", lambda P: P["G1000"] / np.maximum(P["E1000"], 1e-9)),
    ("neglogP1000", lambda P: -P["logP1000"]), ("neglogP50", lambda P: -P["logP50"]),
]
STRATA = [(2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 10 ** 6)]


def rules_grid():
    R = []
    for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
        R.append((f"We50>={t:g}&L>=2@1000", lambda P, t=t: (P["Lt1000"] >= 2) & (P["We50"] >= t)))
    for m in (2, 3, 4, 5, 6, 8, 10, 15):
        R.append((f"L>={m}@1000", lambda P, m=m: P["Lt1000"] >= m))
    for m in (2, 3, 4, 5, 8):
        R.append((f"L>={m}@200", lambda P, m=m: P["Lt200"] >= m))
    for t in (0.1, 0.25, 0.5, 1.0):
        R.append((f"We50>={t:g}&L>=2@200", lambda P, t=t: (P["Lt200"] >= 2) & (P["W200e50"] >= t)))
    for e in (2, 3, 4, 6):
        R.append((f"P<=1e-{e}&L>=2@50", lambda P, e=e: (P["Lt50"] >= 2) & (P["logP50"] <= -e * LN10)))
        R.append((f"P<=1e-{e}&L>=2@1000", lambda P, e=e: (P["Lt1000"] >= 2) & (P["logP1000"] <= -e * LN10)))
    for wn in ("We100", "We200", "Wh20"):
        for t in (0.5, 1.0, 2.0):
            R.append((f"{wn}>={t:g}&L>=2@1000", lambda P, t=t, wn=wn: (P["Lt1000"] >= 2) & (P[wn] >= t)))
    return R


def auc_tpr(pos, neg):
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return dict(auc=None)
    r = rankdata(np.r_[pos, neg])
    auc = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    out = dict(auc=float(auc), n_pos=int(len(pos)), n_neg=int(len(neg)))
    for fpr in (1e-2, 1e-3):
        thr = np.quantile(neg, 1 - fpr)
        out[f"tpr_at_fpr{fpr:g}"] = float((pos > thr).mean())
    return out


def lbin(x):
    edges = [2, 3, 4, 8, 16, 64]
    names = ["2", "3", "4-7", "8-15", "16-63", "64+"]
    return np.array(names, dtype=object)[np.clip(np.searchsorted(edges, x, "right") - 1, 0, len(names) - 1)]


def sbin(x):
    edges = [1, 3, 6, 11, 31]
    names = ["1-2", "3-5", "6-10", "11-30", "31+"]
    return np.array(names, dtype=object)[np.clip(np.searchsorted(edges, x, "right") - 1, 0, len(names) - 1)]


def cmd_pairs(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    ctx = R2m.Ctx()
    su = SR.Universe()
    nd, off, left, d = su.instances(W)
    n, n_inst = su.n, len(left)
    lin = su.lin.astype(np.int64)
    fn = CAS / "results/pairs.json"
    out = {}
    inst, sc = gold_frames(ctx)
    cct = sc[sc.caller == CCT].reset_index(drop=True)
    tok = ctx.tok_rescue3.astype(np.int64)
    labs = pd.read_parquet(CAS / "cas_gene_labels.parquet")
    gclass = np.full(n, "", dtype=object)
    gclass[labs.gid.to_numpy()] = labs.gene.fillna("?").to_numpy()
    is_cas_gene = np.zeros(n, bool)
    is_cas_gene[labs.gid.to_numpy()] = True
    # ---------------- token pair statistics for every co-located pair (ta != tb)
    key, kl, kd, T = B.R.colocalization(ctx.u, tok, W)
    st = B.R.pair_statistics(ctx.u, tok, key, kl, W, T)
    st.pop("incidence", None)
    sigp = (st["P"] <= 1e-4) & (st["L"] >= 2)
    sp = pd.read_parquet(B.ARMS["rescue3"] / "significant_pairs.parquet", columns=["key", "L", "G"])
    out["token_check"] = dict(sig_pairs=int(sigp.sum()), equal_to_engine=bool(np.array_equal(st["key"][sigp], sp.key.to_numpy())
                                                                              and np.array_equal(st["L"][sigp], sp.L.to_numpy())))
    assert out["token_check"]["equal_to_engine"]
    kidx = off[kl] + kd - 1
    pidx = np.searchsorted(st["key"], key)
    tp = np.full(n_inst, -1, np.int64)
    tp[kidx] = pidx
    tokS = np.zeros(n_inst, bool)
    tokS[kidx] = sigp[pidx]
    tokL = np.where(tp >= 0, st["L"][np.maximum(tp, 0)], 0).astype(np.int64)
    tokG = np.where(tp >= 0, st["G"][np.maximum(tp, 0)], 0).astype(np.int64)
    tokE = np.where(tp >= 0, st["E"][np.maximum(tp, 0)], np.nan)
    tokP = np.where(tp >= 0, st["P"][np.maximum(tp, 0)], np.nan)
    fam_genes = np.bincount(tok)
    tl = np.unique(tok * su.nl + lin)
    fam_genera = np.bincount(tl // su.nl, minlength=len(fam_genes))
    # ---------------- embedding statistics (round 2, stored) and the landscape
    P = load_pass(R2 / "stats/obs.npz")
    embS = (P["Lt1000"] >= 2) & (P["We50"] >= 1)
    sidx = np.load(R2 / "cand" / LAND / "sidx.npy")
    out["emb_check"] = dict(significant=int(embS.sum()), sidx=int(len(sidx)), equal=bool(embS.sum() == len(sidx) and embS[sidx].all()))
    assert out["emb_check"]["equal"]
    log("checks", out["token_check"], out["emb_check"])
    # ---------------- instances inside gold Cas spans
    rows = []
    for gi, r in enumerate(cct.itertuples()):
        mset = set(r.mem.tolist())
        for l_ in range(r.m0, r.m1):
            for dd in range(1, W + 1):
                if l_ + dd > r.m1:
                    break
                assert nd[l_] >= dd
                rows.append((int(off[l_] + dd - 1), l_, dd, gi, r.m1 - r.m0 + 1, (l_ in mset) and (l_ + dd in mset)))
    ci = pd.DataFrame(rows, columns=["idx", "l", "dd", "g", "gspan", "mm"]).sort_values(["idx", "gspan"])
    mm_any = ci.groupby("idx").mm.max()
    ci = ci.drop_duplicates("idx").set_index("idx")
    ci["mm"] = mm_any
    ci = ci.reset_index()
    idx = ci.idx.to_numpy()
    aa, bb = ci.l.to_numpy(), ci.l.to_numpy() + ci.dd.to_numpy()
    ci["parity"] = su.gc[aa] % 2
    ci["genome"] = su.gc[aa]
    ci["subtype"] = cct.system_subtype.to_numpy()[ci.g.to_numpy()]
    ci["tokS"] = tokS[idx]
    ci["embS"] = embS[idx]
    ci["same_family"] = tok[aa] == tok[bb]
    ci["emb_excluded"] = P["Lt1000"][idx] < 0
    for k_, arr in (("tokL", tokL), ("tokG", tokG), ("tokE", tokE), ("tokP", tokP)):
        ci[k_] = arr[idx]
    for k_ in ("Lt1000", "Lt200", "Lt50", "We50", "W200e50", "G1000", "E1000", "logP1000", "logP50", "G50", "E50"):
        ci["emb_" + k_] = P[k_][idx]
    ci["famgenes_min"] = np.minimum(fam_genes[tok[aa]], fam_genes[tok[bb]])
    ci["famgenes_max"] = np.maximum(fam_genes[tok[aa]], fam_genes[tok[bb]])
    ci["famgenera_min"] = np.minimum(fam_genera[tok[aa]], fam_genera[tok[bb]])
    ci["famgenera_max"] = np.maximum(fam_genera[tok[aa]], fam_genera[tok[bb]])
    ci["class_a"], ci["class_b"] = gclass[aa], gclass[bb]
    ci["cas_both"] = is_cas_gene[aa] & is_cas_gene[bb]
    ci.to_parquet(CAS / "results/pairs_cas_instances.parquet", index=False)
    lost = ci[ci.tokS & ~ci.embS]
    found = ci[ci.tokS & ci.embS]
    inv = ci[~ci.tokS & ci.embS]

    def dist(s, bins_fn=None):
        s = pd.Series(s)
        if bins_fn is not None:
            s = pd.Series(bins_fn(s.to_numpy()))
        return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}

    def qs(x, q=(0.1, 0.25, 0.5, 0.75, 0.9)):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return [round(float(v), 4) for v in np.quantile(x, q)] if len(x) else []

    summ = {}
    for sp_, g in (("all", ci), ("even", ci[ci.parity == 0]), ("odd", ci[ci.parity == 1])):
        lo, fo = g[g.tokS & ~g.embS], g[g.tokS & g.embS]
        summ[sp_] = dict(
            instances=int(len(g)), member_member=int(g.mm.sum()), tokS=int(g.tokS.sum()), embS=int(g.embS.sum()),
            both=int((g.tokS & g.embS).sum()), lost=int(len(lo)), inverse=int((~g.tokS & g.embS).sum()),
            neither=int((~g.tokS & ~g.embS).sum()), token_same_family=int(g.same_family.sum()), emb_excluded=int(g.emb_excluded.sum()),
            lost_share_of_tokS=float(len(lo) / max(1, g.tokS.sum())),
            lost_mm=int(lo.mm.sum()), tokS_mm=int((g.tokS & g.mm).sum()), embS_mm=int((g.embS & g.mm).sum()))
    out["counts"] = summ
    out["lost"] = dict(
        n=int(len(lost)),
        tokL=dist(lost.tokL, lbin), tokL_found=dist(found.tokL, lbin),
        share_tokL_2_7=float(((lost.tokL >= 2) & (lost.tokL <= 7)).mean()),
        share_tokL_2_7_found=float(((found.tokL >= 2) & (found.tokL <= 7)).mean()),
        tokL_q=qs(lost.tokL), tokG_q=qs(lost.tokG), tokE_q=qs(lost.tokE), tokG_q_found=qs(found.tokG),
        emb_Lt1000=dist(lost.emb_Lt1000.clip(-1, 64), None), emb_Lt1000_q=qs(lost.emb_Lt1000), emb_We50_q=qs(lost.emb_We50),
        emb_E1000_q=qs(lost.emb_E1000), emb_Lt200_q=qs(lost.emb_Lt200), emb_Lt50_q=qs(lost.emb_Lt50),
        emb_excluded=int(lost.emb_excluded.sum()),
        emb_Lt1000_ge2=int((lost.emb_Lt1000 >= 2).sum()), emb_We50_ge_0p5=int((lost.emb_We50 >= 0.5).sum()),
        famgenes_min_q=qs(lost.famgenes_min), famgenes_max_q=qs(lost.famgenes_max),
        famgenera_min_q=qs(lost.famgenera_min), famgenera_max_q=qs(lost.famgenera_max),
        famgenera_min_q_found=qs(found.famgenera_min), famgenes_min_q_found=qs(found.famgenes_min),
        subtype=lost.subtype.value_counts().head(12).to_dict(),
        subtype_lost_share={s_: float(((ci.subtype == s_) & ci.tokS & ~ci.embS).sum() / max(1, ((ci.subtype == s_) & ci.tokS).sum()))
                            for s_ in ci.subtype.value_counts().head(8).index},
        class_pairs=(lost.class_a.replace("", "-") + "|" + lost.class_b.replace("", "-")).value_counts().head(15).to_dict(),
        class_pairs_found=(found.class_a.replace("", "-") + "|" + found.class_b.replace("", "-")).value_counts().head(15).to_dict(),
        member_member=int(lost.mm.sum()))
    out["inverse"] = dict(
        n=int(len(inv)), token_same_family=int(inv.same_family.sum()),
        tokL=dist(inv.tokL, lbin), tokP_le_1e4=int((inv.tokP <= 1e-4).sum()),
        tokP_q=qs(inv.tokP), emb_Lt1000_q=qs(inv.emb_Lt1000), emb_We50_q=qs(inv.emb_We50),
        famgenes_min_q=qs(inv.famgenes_min), member_member=int(inv.mm.sum()),
        class_pairs=(inv.class_a.replace("", "-") + "|" + inv.class_b.replace("", "-")).value_counts().head(10).to_dict())
    jdump(fn, out)
    # ---------------- Cas vs non-Cas at matched token L (and family lineage spread)
    span_gene = np.zeros(n + 1, np.int64)
    for r in cct.itertuples():
        span_gene[r.m0] += 1
        span_gene[r.m1 + 1] -= 1
    in_span = np.cumsum(span_gene)[:n] > 0
    casish = is_cas_gene | in_span
    noncas = ~casish[left] & ~casish[left + d]
    allS = np.flatnonzero(tokS)
    nonS = allS[noncas[allS]]
    casS = idx[ci.tokS.to_numpy()]
    lostflag = ~embS
    lb_all = lbin(tokL)
    spread = np.minimum(fam_genera[tok[left]], fam_genera[tok[left + d]])
    sb_all = sbin(spread)
    cmpL = {}
    for nm in ["2", "3", "4-7", "8-15", "16-63", "64+"]:
        A = casS[lb_all[casS] == nm]
        Bn = nonS[lb_all[nonS] == nm]
        if len(A) == 0:
            continue
        cmpL[nm] = dict(cas_n=int(len(A)), cas_lost=float(lostflag[A].mean()), noncas_n=int(len(Bn)), noncas_lost=float(lostflag[Bn].mean()),
                        diff=boot_rate_diff(su.gc[left[A]], lostflag[A].astype(float), su.gc[left[Bn]], lostflag[Bn].astype(float)))
    out["cas_vs_noncas_by_tokL"] = cmpL
    # standardization: expected Cas lost share from non-Cas rates in (token L bin x min family genera bin) cells
    cellB = pd.Series(lostflag[nonS].astype(float)).groupby([lb_all[nonS], sb_all[nonS]]).agg(["mean", "size"])
    exp_rate = np.array([cellB["mean"].get((x, y), np.nan) for x, y in zip(lb_all[casS], sb_all[casS])])
    ok = np.isfinite(exp_rate)
    rng = np.random.default_rng(17)
    gA = su.gc[left[casS[ok]]]
    y_obs, y_exp = lostflag[casS[ok]].astype(float), exp_rate[ok]
    ug, inv_ = np.unique(gA, return_inverse=True)
    so, se_, nn_ = np.bincount(inv_, y_obs), np.bincount(inv_, y_exp), np.bincount(inv_).astype(float)
    pick = rng.integers(0, len(ug), size=(2000, len(ug)))
    dd_ = (so[pick].sum(1) - se_[pick].sum(1)) / nn_[pick].sum(1)
    out["cas_standardized"] = dict(cas_tokS=int(len(casS)), cells_used=int(ok.sum()), observed_lost=float(y_obs.mean()),
                                   expected_lost_from_noncas_cells=float(y_exp.mean()),
                                   diff_obs_minus_exp=[float(y_obs.mean() - y_exp.mean()), float(np.quantile(dd_, .025)), float(np.quantile(dd_, .975))],
                                   noncas_tokS=int(len(nonS)), noncas_lost=float(lostflag[nonS].mean()),
                                   noncas_cells=cellB.reset_index().rename(columns={"level_0": "tokL", "level_1": "spread"}).to_dict("records"))
    # share of all tokS instances genome-wide in token L 2-7, Cas vs non-Cas
    out["tokL_distribution_tokS"] = dict(cas=dist(tokL[casS], lbin), noncas=dist(tokL[nonS], lbin))
    jdump(fn, out)
    log("part2 done", f"{time.time() - t0:.0f}s")
    # ================================================================ part 4: floor
    nulls = [load_pass(R2 / f"stats/null{r}.npz") for r in range(3)]
    log("nulls loaded", f"{time.time() - t0:.0f}s")
    lost_idx = idx[(ci.tokS & ~ci.embS).to_numpy()]
    tokcas_idx = casS
    lost_non = nonS[~embS[nonS]]
    # lost Cas boundaries: inside gold Cas spans, token support, no embedding support
    tok_cov = B.span_count(left[tokS], d[tokS].astype(np.int64), n) > 0
    emb_cov = B.span_count(left[embS], d[embS].astype(np.int64), n) > 0
    casb = np.zeros(n, bool)
    for r in cct.itertuples():
        casb[r.m0:r.m1] = True
    lostb = casb & tok_cov & ~emb_cov
    out["lost_boundaries"] = dict(cas_boundaries=int(casb.sum()), tok_supported=int((casb & tok_cov).sum()),
                                  emb_supported=int((casb & emb_cov).sum()), lost=int(lostb.sum()),
                                  emb_only=int((casb & emb_cov & ~tok_cov).sum()))
    rows = []
    for name, fnr in rules_grid():
        mo = fnr(P)
        nn = [int(fnr(Q).sum()) for Q in nulls]
        cov = B.span_count(left[mo], d[mo].astype(np.int64), n) > 0
        rows.append(dict(rule=name, observed=int(mo.sum()), null_mean=float(np.mean(nn)), fdr=float(np.mean(nn) / max(1, mo.sum())),
                         lost_cas_inst=float(mo[lost_idx].mean()), lost_cas_bound=float(cov[lostb].mean()),
                         tok_cas_inst=float(mo[tokcas_idx].mean()), lost_noncas_inst=float(mo[lost_non].mean()),
                         emb_sig_inst_kept=float(mo[sidx].mean())))
        log("rule", rows[-1])
    out["floor_rules"] = rows
    jdump(fn, out)
    # thresholds marking 50 / 80 / 90 % of lost Cas instances, per family, and their null FDR
    thr = {}
    rng = np.random.default_rng(17)
    for stat, col, extra in (("We50@1000", "We50", "Lt1000"), ("Lt@1000", "Lt1000", None), ("Lt@200", "Lt200", None), ("W200e50", "W200e50", "Lt200"),
                             ("neglogP50", None, "Lt50"), ("neglogP1000", None, "Lt1000")):
        def sv(Q):
            if stat == "neglogP50":
                return -Q["logP50"]
            if stat == "neglogP1000":
                return -Q["logP1000"]
            return Q[col].astype(np.float64)
        x = sv(P)
        base = (P[extra] >= 2) if extra else np.ones(n_inst, bool)
        xl = x[lost_idx]
        bl = base[lost_idx]
        item = {}
        for q in (0.5, 0.8, 0.9):
            # threshold t: marks a share q of all lost Cas instances (base condition and statistic >= t)
            if bl.sum() == 0:
                continue
            t = float(np.quantile(xl[bl], 1 - q / bl.mean())) if bl.mean() >= q else None
            if t is None:
                item[f"q{q:g}"] = dict(t=None, note=f"only {bl.mean():.3f} of lost Cas instances pass the base condition")
                continue
            mo = base & (x >= t)
            nn = [int(((Q[extra] >= 2) if extra else np.ones(n_inst, bool)).__and__(sv(Q) >= t).sum()) for Q in nulls]
            item[f"q{q:g}"] = dict(t=t, marks_lost=float(mo[lost_idx].mean()), observed=int(mo.sum()), null_mean=float(np.mean(nn)),
                                   fdr=float(np.mean(nn) / max(1, mo.sum())))
        item["base_pass_share"] = float(bl.mean())
        thr[stat] = item
    out["floor_thresholds"] = thr
    jdump(fn, out)
    # AUC at fixed Lt (K1000): lost Cas instances vs null instances (pooled 3 nulls, subsampled to 500k per stratum)
    roc = {}
    for lo_, hi_ in STRATA:
        sname = f"{lo_}" if lo_ == hi_ else f"{lo_}-{hi_ if hi_ < 10 ** 6 else 'inf'}"
        pos_i = lost_idx[(P["Lt1000"][lost_idx] >= lo_) & (P["Lt1000"][lost_idx] <= hi_)]
        lnc = lost_non[(P["Lt1000"][lost_non] >= lo_) & (P["Lt1000"][lost_non] <= hi_)]
        if len(lnc) > 200000:
            lnc = rng.choice(lnc, 200000, replace=False)
        negs = []
        for Q in nulls:
            m = np.flatnonzero((Q["Lt1000"] >= lo_) & (Q["Lt1000"] <= hi_))
            if len(m) > 170000:
                m = rng.choice(m, 170000, replace=False)
            negs.append(m)
        item = dict(n_lost_cas=int(len(pos_i)), n_lost_noncas=int(len(lnc)), n_null=int(sum(len(m) for m in negs)))
        for sn, f_ in STATS:
            neg = np.concatenate([np.asarray(f_(Q), np.float64)[m] for Q, m in zip(nulls, negs)])
            pv = np.asarray(f_(P), np.float64)
            item[sn] = dict(cas=auc_tpr(pv[pos_i], neg), noncas=auc_tpr(pv[lnc], neg))
        roc[sname] = item
        log("auc", sname, {k: (v["cas"].get("auc"), v["noncas"].get("auc")) for k, v in item.items() if isinstance(v, dict)})
    out["auc_fixed_L"] = roc
    jdump(fn, out)
    # ---------------- query sets for the lists step
    rng = np.random.default_rng(17)
    q = {}
    q["cas_idx"] = idx
    q["cas_tokS"] = ci.tokS.to_numpy()
    q["cas_embS"] = ci.embS.to_numpy()
    q["cas_tokL"] = tokL[idx]
    q["noncas_lost"] = rng.choice(lost_non, min(20000, len(lost_non)), replace=False)
    fnon = nonS[embS[nonS]]
    q["noncas_found"] = rng.choice(fnon, min(20000, len(fnon)), replace=False)
    q["noncas_lost_tokL"] = tokL[q["noncas_lost"]]
    q["noncas_found_tokL"] = tokL[q["noncas_found"]]
    ok_obs = np.flatnonzero(P["Lt1000"] >= 1)
    q["verify_obs"] = rng.choice(ok_obs, 2000, replace=False)
    for r, Q in enumerate(nulls):
        sel = []
        for lo_, hi_ in STRATA[:5]:
            m = np.flatnonzero((Q["Lt1000"] >= lo_) & (Q["Lt1000"] <= hi_))
            sel.append(rng.choice(m, min(4000, len(m)), replace=False))
        q[f"null{r}"] = np.concatenate(sel)
        q[f"null{r}_Lt"] = Q["Lt1000"][q[f"null{r}"]]
        okn = np.flatnonzero(Q["Lt1000"] >= 1)
        q[f"verify_null{r}"] = rng.choice(okn, 1000, replace=False)
    np.savez(CAS / "results/query_sets.npz", **q)
    out["seconds"] = round(time.time() - t0, 1)
    jdump(fn, out)
    log("pairs done", out["seconds"])


# ============================================================================== lists (part 3) + reciprocity
@njit(parallel=True, cache=True)
def sort_rows(x, out):
    for i in prange(x.shape[0]):
        out[i] = np.sort(x[i])


@njit(cache=True)
def in_row(nbsort, x, v):
    row = nbsort[x]
    j = np.searchsorted(row, v)
    return j < row.shape[0] and row[j] == v


@njit(parallel=True, cache=True)
def support_detail(qi, qd, gene_at, slot_of, cid, nb2, tail2, nbsort, lin, tok, n_lin, w, ksh, outI, outF, nchunks):
    """Per query instance (left slot qi, distance qd) in the gene order gene_at: per lineage other than the instance's,
    the best supporting pair (a' in N(a), b' in N(b), a' within w of b' on one contig, a' != b'; m = max(tail rank a',
    tail rank b'); a' = the best-ranked list member of a marking b''s slot in the channel) for channels
      0 any, 1 family-consistent (tok a' = tok a, tok b' = tok b), 2 reciprocal (a in N(a') and b in N(b')),
      3 family-consistent and reciprocal, 4 any with both ranks < ksh, 5 family-consistent with both ranks < ksh.
    outI[q] = [L_0..L_5, lineages whose channel-0 best pair is reciprocal, min m channel 0, min m channel 1]
    outF[q] = [W_e50 of channels 0..5]"""
    m = len(qi)
    n = len(gene_at)
    K = nb2.shape[1]
    BIG = 1 << 40
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        if lo >= hi:
            continue
        st = np.full((4, n), -1, np.int64)
        rk = np.zeros((4, n), np.int32)
        rec0 = np.zeros(n, np.int8)
        lst = np.full(n_lin, -1, np.int64)
        best = np.full((6, n_lin), BIG, np.int64)
        brec = np.zeros(n_lin, np.int8)
        linlist = np.zeros(n_lin, np.int64)
        for q in range(lo, hi):
            i = qi[q]
            a = gene_at[i]
            b = gene_at[i + qd[q]]
            la = lin[a]
            ta = tok[a]
            tb = tok[b]
            for r in range(K):
                x = nb2[a, r]
                if x < 0:
                    break
                famx = tok[x] == ta
                recx = in_row(nbsort, x, a)
                s = slot_of[x]
                c = cid[s]
                ulo = s - w if s - w > 0 else 0
                uhi = s + w + 1 if s + w + 1 < n else n
                for u in range(ulo, uhi):
                    if u == s or cid[u] != c:
                        continue
                    if st[0, u] != q:
                        st[0, u] = q
                        rk[0, u] = r
                        rec0[u] = 1 if recx else 0
                    if famx and st[1, u] != q:
                        st[1, u] = q
                        rk[1, u] = r
                    if recx and st[2, u] != q:
                        st[2, u] = q
                        rk[2, u] = r
                    if famx and recx and st[3, u] != q:
                        st[3, u] = q
                        rk[3, u] = r
            nl = 0
            for r in range(K):
                y = nb2[b, r]
                if y < 0:
                    break
                s = slot_of[y]
                if st[0, s] != q:
                    continue
                lg = lin[y]
                if lg == la:
                    continue
                if lst[lg] != q:
                    lst[lg] = q
                    for k in range(6):
                        best[k, lg] = BIG
                    brec[lg] = 0
                    linlist[nl] = lg
                    nl += 1
                famy = tok[y] == tb
                recy = in_row(nbsort, y, b)
                tby = np.int64(tail2[b, r])
                r0 = rk[0, s]
                mm = np.int64(tail2[a, r0])
                if tby > mm:
                    mm = tby
                if mm < best[0, lg]:
                    best[0, lg] = mm
                    brec[lg] = 1 if (recy and rec0[s] == 1) else 0
                if r0 < ksh and r < ksh and mm < best[4, lg]:
                    best[4, lg] = mm
                if famy and st[1, s] == q:
                    r1 = rk[1, s]
                    m1 = np.int64(tail2[a, r1])
                    if tby > m1:
                        m1 = tby
                    if m1 < best[1, lg]:
                        best[1, lg] = m1
                    if r1 < ksh and r < ksh and m1 < best[5, lg]:
                        best[5, lg] = m1
                if recy and st[2, s] == q:
                    r2 = rk[2, s]
                    m2 = np.int64(tail2[a, r2])
                    if tby > m2:
                        m2 = tby
                    if m2 < best[2, lg]:
                        best[2, lg] = m2
                if famy and recy and st[3, s] == q:
                    r3 = rk[3, s]
                    m3 = np.int64(tail2[a, r3])
                    if tby > m3:
                        m3 = tby
                    if m3 < best[3, lg]:
                        best[3, lg] = m3
            nbr = 0
            mn0 = BIG
            mn1 = BIG
            for k in range(6):
                outI[q, k] = 0
                outF[q, k] = 0.0
            for t in range(nl):
                lg = linlist[t]
                for k in range(6):
                    if best[k, lg] < BIG:
                        outI[q, k] += 1
                        outF[q, k] += math.exp(-(best[k, lg] - 1.0) / 50.0)
                if brec[lg] == 1:
                    nbr += 1
                if best[0, lg] < mn0:
                    mn0 = best[0, lg]
                if best[1, lg] < mn1:
                    mn1 = best[1, lg]
            outI[q, 6] = nbr
            outI[q, 7] = mn0 if mn0 < BIG else -1
            outI[q, 8] = mn1 if mn1 < BIG else -1


def run_detail(su, left, d, idx, order, nb2, tail2, nbsort, lin, tok, threads, ksh=200):
    gene_at, slot_of = order
    qi = left[idx].astype(np.int64)
    qd = d[idx].astype(np.int64)
    outI = np.zeros((len(idx), 9), np.int64)
    outF = np.zeros((len(idx), 6), np.float64)
    support_detail(qi, qd, gene_at.astype(np.int64), slot_of.astype(np.int64), su.cid, nb2, tail2, nbsort, lin, tok, int(su.nl), W, ksh,
                   outI, outF, max(1, threads * 4))
    return outI, outF


DCOLS_I = ["L_any", "L_fam", "L_rec", "L_famrec", "L_any200", "L_fam200", "L_bestrec", "mmin_any", "mmin_fam"]
DCOLS_F = ["W_any", "W_fam", "W_rec", "W_famrec", "W_any200", "W_fam200"]


def cmd_lists(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    fn = CAS / "results/lists.json"
    out = {}
    ctx = R2m.Ctx()
    su = SR.Universe()
    nd, off, left, d = su.instances(W)
    n = su.n
    Q = dict(np.load(CAS / "results/query_sets.npz"))
    Ls = SR.Lists(su, S2.ListArgs((1000,)))
    K = 1000
    llen = Ls.lend - Ls.lstart
    out["lists"] = dict(full=int((llen == K).sum()), n=int(n))
    assert (llen == K).all()
    nb2 = Ls.nb.reshape(n, K)
    tail2 = Ls.tail.reshape(n, K)
    X, uq = Ls.X, Ls.uq
    nbsort = np.empty_like(nb2)
    sort_rows(nb2, nbsort)
    log("lists ready", f"{time.time() - t0:.0f}s")
    tok = ctx.tok_rescue3.astype(np.int64)
    lin = su.lin.astype(np.int64)
    ident = np.arange(n, dtype=np.int64)
    orders = {-1: (ident, ident)}
    for r in range(3):
        orders[r] = SR.null_order(su, r)
    # ---------------- support detail for every query set; verification against the stored statistics
    obs = np.load(R2 / "stats/obs.npz")
    variants = [str(x) for x in obs["variants"]]
    v1000, v200 = variants.index("K1000_fnone"), variants.index("K200_fnone")
    det = {}
    sets = [("cas", -1, Q["cas_idx"]), ("noncas_lost", -1, Q["noncas_lost"]), ("noncas_found", -1, Q["noncas_found"]),
            ("verify_obs", -1, Q["verify_obs"])] + [(f"null{r}", r, Q[f"null{r}"]) for r in range(3)] + \
           [(f"verify_null{r}", r, Q[f"verify_null{r}"]) for r in range(3)]
    stored = {}
    for r in (-1, 0, 1, 2):
        z = obs if r < 0 else np.load(R2 / f"stats/null{r}.npz")
        Lz = z["L"]
        stored[r] = dict(L1000=Lz[v1000].astype(np.int64), L200=Lz[v200].astype(np.int64),
                         W=np.ascontiguousarray(z["W"][v1000, 0]).astype(np.float64))
        del z, Lz
    verify = {}
    for name, r, idx in sets:
        t1 = time.time()
        oI, oF = run_detail(su, left, d, idx, orders[r], nb2, tail2, nbsort, lin, tok, a.threads)
        det[name] = (idx, r, oI, oF)
        z = stored[r]
        Ls_ = z["L1000"][idx]
        L2_ = z["L200"][idx]
        Ws_ = z["W"][idx]
        ok = Ls_ >= 0
        verify[name] = dict(n=int(len(idx)), not_excluded=int(ok.sum()), L1000_equal=int((oI[ok, 0] == Ls_[ok]).sum()),
                            L200_equal=int((oI[ok, 4] == L2_[ok]).sum()),
                            W_maxreldiff=float((np.abs(oF[ok, 0] - Ws_[ok]) / np.maximum(1.0, Ws_[ok])).max()) if ok.any() else None,
                            seconds=round(time.time() - t1, 1))
        log("detail", name, verify[name])
    out["verify"] = verify
    np.savez(CAS / "results/support_detail.npz", **{f"{k}_idx": v[0] for k, v in det.items()}, **{f"{k}_I": v[2] for k, v in det.items()},
             **{f"{k}_F": v[3] for k, v in det.items()})
    jdump(fn, out)
    # ---------------- per-lineage recovery of the token's support for lost instances (Cas, non-Cas sample)
    key, kl, kd, T = B.R.colocalization(ctx.u, tok, W)
    ko = np.argsort(key, kind="stable")
    ks = key[ko]

    def token_support(a_, b_):
        ta, tb = tok[a_], tok[b_]
        kk = min(ta, tb) * T + max(ta, tb)
        j0, j1 = np.searchsorted(ks, kk, "left"), np.searchsorted(ks, kk, "right")
        sup = ko[j0:j1]
        xs, ys = kl[sup], kl[sup] + kd[sup]
        sw = tok[xs] != ta
        xs, ys = np.where(sw, ys, xs), np.where(sw, xs, ys)
        keep = lin[xs] != lin[a_]
        xs, ys = xs[keep], ys[keep]
        ra = dict(zip(nb2[a_].tolist(), tail2[a_].tolist()))
        pa = dict(zip(nb2[a_].tolist(), range(K)))
        rb = dict(zip(nb2[b_].tolist(), tail2[b_].tolist()))
        pb = dict(zip(nb2[b_].tolist(), range(K)))
        per_g = {}
        for x, y in zip(xs.tolist(), ys.tolist()):
            g = int(lin[x])
            ia, ib = x in ra, y in rb
            mm = max(ra[x], rb[y]) if (ia and ib) else 10 ** 9
            m200 = mm if (ia and ib and pa[x] < 200 and pb[y] < 200) else 10 ** 9
            cur = per_g.get(g)
            stt = 3 if (ia and ib) else (1 if ia else (2 if ib else 0))
            if cur is None:
                per_g[g] = [mm, m200, stt]
            else:
                cur[0] = min(cur[0], mm)
                cur[1] = min(cur[1], m200)
                cur[2] = max(cur[2], stt) if stt != 3 else 3
        xa, xb = X[uq[a_]], X[uq[b_]]
        cos_x = X[uq[xs]] @ xa if len(xs) else np.zeros(0)
        cos_y = X[uq[ys]] @ xb if len(ys) else np.zeros(0)
        cap_a = float(X[uq[nb2[a_, K - 1]]] @ xa)
        cap_b = float(X[uq[nb2[b_, K - 1]]] @ xb)
        vals = np.array(list(per_g.values()), np.int64).reshape(-1, 3)
        found = vals[:, 2] == 3 if len(vals) else np.zeros(0, bool)
        return dict(tok_genera=int(len(vals)), both=int(found.sum()), a_only=int((vals[:, 2] == 1).sum()) if len(vals) else 0,
                    b_only=int((vals[:, 2] == 2).sum()) if len(vals) else 0, neither=int((vals[:, 2] == 0).sum()) if len(vals) else 0,
                    both200=int((vals[:, 1] < 10 ** 9).sum()) if len(vals) else 0,
                    W=float(np.exp(-(vals[found, 0] - 1.0) / 50.0).sum()) if len(vals) else 0.0,
                    m_best=int(vals[found, 0].min()) if found.any() else -1,
                    m_med=float(np.median(vals[found, 0])) if found.any() else -1.0,
                    cos_a_q50=float(np.median(cos_x)) if len(cos_x) else np.nan, cos_b_q50=float(np.median(cos_y)) if len(cos_y) else np.nan,
                    cap_a=cap_a, cap_b=cap_b,
                    frac_x_above_cap=float((cos_x >= cap_a).mean()) if len(cos_x) else np.nan,
                    frac_y_above_cap=float((cos_y >= cap_b).mean()) if len(cos_y) else np.nan)

    ci = pd.read_parquet(CAS / "results/pairs_cas_instances.parquet")
    lost = ci[ci.tokS & ~ci.embS]
    rows = []
    for r_ in lost.itertuples():
        a_, b_ = int(r_.l), int(r_.l + r_.dd)
        rows.append(dict(idx=int(r_.idx), group="cas_lost", tokL=int(r_.tokL), parity=int(r_.parity), genome=int(su.gc[a_]), **token_support(a_, b_)))
    found_c = ci[ci.tokS & ci.embS]
    for r_ in found_c.itertuples():
        a_, b_ = int(r_.l), int(r_.l + r_.dd)
        rows.append(dict(idx=int(r_.idx), group="cas_found", tokL=int(r_.tokL), parity=int(r_.parity), genome=int(su.gc[a_]), **token_support(a_, b_)))
    rng = np.random.default_rng(17)
    for gname, key_ in (("noncas_lost", "noncas_lost"), ("noncas_found", "noncas_found")):
        sel = Q[key_][:5000]
        tl_ = Q[f"{key_}_tokL"][:5000]
        for ii, tlv in zip(sel.tolist(), tl_.tolist()):
            a_, b_ = int(left[ii]), int(left[ii] + d[ii])
            rows.append(dict(idx=int(ii), group=gname, tokL=int(tlv), parity=int(su.gc[a_] % 2), genome=int(su.gc[a_]), **token_support(a_, b_)))
    ts = pd.DataFrame(rows)
    ts.to_parquet(CAS / "results/token_support_recovery.parquet", index=False)
    log("token support recovery", len(ts), f"{time.time() - t0:.0f}s")
    # ---------------- part 3: family members in the lists of Cas genes vs matched non-Cas genes
    labs = pd.read_parquet(CAS / "cas_gene_labels.parquet")
    is_cas = np.zeros(n, bool)
    is_cas[labs.gid.to_numpy()] = True
    gclass = np.full(n, "", dtype=object)
    gclass[labs.gid.to_numpy()] = labs.gene.fillna("?").to_numpy()
    T_ = int(tok.max()) + 1
    fam_genes = np.bincount(tok, minlength=T_)
    tlk = np.unique(tok * su.nl + lin)
    fam_genera = np.bincount(tlk // su.nl, minlength=T_)
    cas_fam = np.zeros(T_, bool)
    cas_fam[tok[is_cas]] = True
    torder = np.argsort(tok, kind="stable")
    tptr = np.searchsorted(tok[torder], np.arange(T_ + 1))

    def gene_metrics(g):
        t = tok[g]
        mem = torder[tptr[t]:tptr[t + 1]]
        outm = mem[lin[mem] != lin[g]]
        Na = nb2[g]
        fm = tok[Na] == t
        n_in = int(fm.sum())
        xa = X[uq[g]]
        cos_out = X[uq[outm]] @ xa if len(outm) else np.zeros(0)
        inl = np.isin(outm, Na[fm]) if len(outm) else np.zeros(0, bool)
        pos = np.flatnonzero(fm)
        nonf = Na[~fm]
        cap = float(X[uq[Na[K - 1]]] @ xa)
        res = dict(gid=int(g), fam_genes=int(fam_genes[t]), fam_genera=int(fam_genera[t]), n_out=int(len(outm)),
                   genera_out=int(len(np.unique(lin[outm]))) if len(outm) else 0, n_in=n_in,
                   frac_in=(n_in / len(outm)) if len(outm) else np.nan, frac_in_cap=(n_in / min(len(outm), K)) if len(outm) else np.nan,
                   genera_reached=int(len(np.unique(lin[Na[fm]]))) if n_in else 0,
                   rank_med=float(np.median(tail2[g][fm])) if n_in else np.nan, rank_max=int(tail2[g][fm].max()) if n_in else -1,
                   first_fam_pos=int(pos[0]) if n_in else -1,
                   intrusions=int((~fm[:pos[-1] + 1]).sum()) if n_in else 0,
                   cos_in_med=float(np.median(cos_out[inl])) if inl.any() else np.nan,
                   cos_out_med=float(np.median(cos_out[~inl])) if (~inl).any() else np.nan,
                   cos_out_max=float(cos_out[~inl].max()) if (~inl).any() else np.nan,
                   cap_cos=cap, nonfam=int(len(nonf)), nonfam_cas=int(is_cas[nonf].sum()), nonfam_casfam=int(cas_fam[tok[nonf]].sum()),
                   nonfam_same_class=int((gclass[nonf] == gclass[g]).sum()) if gclass[g] != "" else 0,
                   top10_fam=int(fm[:10].sum()), top50_fam=int(fm[:50].sum()))
        return res

    def binv(v):
        v = np.asarray(v, np.float64)
        return np.where(v <= 10, v, 10 + np.ceil(np.log2(np.maximum(v, 1) / 10.0) * 4))

    lost_genes = np.unique(np.r_[lost.l.to_numpy(), (lost.l + lost.dd).to_numpy()])
    found_genes = np.unique(np.r_[found_c.l.to_numpy(), (found_c.l + found_c.dd).to_numpy()])
    found_genes = np.setdiff1d(found_genes, lost_genes)
    pool = np.flatnonzero(~is_cas & ~cas_fam[tok])
    pkey = binv(fam_genes[tok[pool]]) * 1000 + binv(fam_genera[tok[pool]])
    porder = np.argsort(pkey, kind="stable")
    pks = pkey[porder]

    def matched(genes, per=3):
        kk = binv(fam_genes[tok[genes]]) * 1000 + binv(fam_genera[tok[genes]])
        outg, src = [], []
        for g, k_ in zip(genes.tolist(), kk.tolist()):
            j0, j1 = np.searchsorted(pks, k_, "left"), np.searchsorted(pks, k_, "right")
            if j1 <= j0:
                continue
            pick = porder[rng.integers(j0, j1, size=per)]
            outg += pool[pick].tolist()
            src += [g] * per
        return np.array(outg, np.int64), np.array(src, np.int64)

    ctrl_lost, src_lost = matched(lost_genes)
    ctrl_found, src_found = matched(found_genes)
    gm_rows = []
    for gname, genes in (("cas_lostpair", lost_genes), ("cas_foundpair", found_genes), ("ctrl_lostpair", ctrl_lost), ("ctrl_foundpair", ctrl_found)):
        for g in genes.tolist():
            row = gene_metrics(g)
            row["group"] = gname
            row["cas_class"] = gclass[g]
            gm_rows.append(row)
    gmd = pd.DataFrame(gm_rows)
    gmd.to_parquet(CAS / "results/list_gene_metrics.parquet", index=False)
    out["matching"] = dict(lost_genes=int(len(lost_genes)), lost_genes_matched=int(len(np.unique(src_lost))),
                           found_genes=int(len(found_genes)), found_genes_matched=int(len(np.unique(src_found))), per=3)
    log("gene metrics", len(gmd), f"{time.time() - t0:.0f}s")
    out["seconds"] = round(time.time() - t0, 1)
    jdump(fn, out)
    log("lists done", out["seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("repro", "part1", "pairs", "lists", "attrib", "callers"))
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    {"repro": cmd_repro, "part1": cmd_part1, "pairs": cmd_pairs, "lists": cmd_lists, "attrib": cmd_attrib,
     "callers": cmd_callers}[a.cmd](a)


if __name__ == "__main__":
    main()
