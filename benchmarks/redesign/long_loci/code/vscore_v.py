#!/usr/bin/env python
"""Readings, scoring, typing and vocabulary statistics for ws_vocab runs.

usage (biotite, Slurm):
  vscore.py refcheck                         rescore token DH loci; must reproduce the stored reference exactly
  vscore.py odd   --arm main --tau T         readings V-top / V-nest / V-mob scored on ODD genomes only (selection)
  vscore.py even  --arm main --tau T --primary V-nest|V-mob     even verdict of all readings + typing + vocabulary stats
Every command writes JSON to $RD/vocab/results/. The odd command never prints or writes an even number.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402  (read-only import)
import rules as RU  # noqa: E402

CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
DH_LOCI = RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet"
DH_SCORES = RD / "boundary/scores/rescue3/DH_coreH_res0.3_f0.5.parquet"
R1_SCORES = RD / "boundary/scores/rescue3/R1_conserved2.parquet"
R1_LOCI = RD / "boundary/loci/rescue3/R1_conserved2.parquet"
REF_EVEN = dict(summed=0.4270, f1={"DefenseFinder": 0.744, "TXSScan": 0.548, "CRISPRCasTyper": 0.479},
                recovered=1686, resolved=1512, parv_resolved=113)
ODD_BARS = dict(recovered=830, resolved=705, parv=54)      # 1650 / 1400 / 110 scaled to the 848 odd runs (softrec2)
ALL_BARS = dict(recovered=1650, resolved=1400, parv=110)
log = B.log


def jdump(obj, fn):
    with open(fn, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log("wrote", fn)


# --------------------------------------------------------------------------- run outputs
class Run:
    def __init__(self, arm, tau, which="obs"):
        d = VD / "runs" / arm / f"tau{tau:g}" / which
        self.d = d
        self.words = pd.read_parquet(d / "words.parquet")
        z = np.load(d / "inst.npz")
        self.inst = {k: z[k] for k in z.files}
        z = np.load(d / "live.npz")
        self.live = {k: z[k] for k in z.files}
        self.T = json.load(open(d / "run.json"))["T"]
        top = ~self.live["term"]
        self.top_inst = self.live["ninst"][top]
        self.top_s, self.top_e, self.top_sym = self.live["ns"][top], self.live["ne"][top], self.live["sym"][top]


@njit(cache=True)
def mobile_cores(top_inst, ia, ib, isym, mobile):
    """Maximal sub-instances (below the top) whose type is mobile. Returns instance ids."""
    out = np.empty(len(ia), dtype=np.int64)
    n = 0
    stack = np.empty(len(ia) + 2, dtype=np.int64)
    for t in range(len(top_inst)):
        sp = 0
        r = top_inst[t]
        if ia[r] >= 0:
            stack[sp] = ia[r]
            sp += 1
        if ib[r] >= 0:
            stack[sp] = ib[r]
            sp += 1
        while sp > 0:
            sp -= 1
            x = stack[sp]
            if mobile[isym[x]]:
                out[n] = x
                n += 1
            else:
                if ia[x] >= 0:
                    stack[sp] = ia[x]
                    sp += 1
                if ib[x] >= 0:
                    stack[sp] = ib[x]
                    sp += 1
    return out[:n]


def readings(run, grp_of_genome, gc):
    I = run.inst
    rd = {}
    rd["V-top"] = (run.top_s, run.top_e, run.top_sym)
    ch = np.r_[I["a"][run.top_inst], I["b"][run.top_inst]]
    ch = ch[ch >= 0]
    rd["V-nest"] = (np.r_[run.top_s, I["s"][ch]], np.r_[run.top_e, I["e"][ch]], np.r_[run.top_sym, I["sym"][ch]])
    # mobile: types that are top-level instances in >= 2 groups
    S = int(max(I["sym"].max() if len(I["sym"]) else 0, run.T)) + 1
    g = grp_of_genome[gc[run.top_s]]
    ng = int(grp_of_genome.max()) + 1
    tg = np.unique(run.top_sym * ng + g)
    cnt = np.bincount(tg // ng, minlength=S)
    mobile = cnt >= 2
    mc = mobile_cores(run.top_inst.astype(np.int64), I["a"], I["b"], I["sym"], mobile)
    rd["V-mob"] = (np.r_[run.top_s, I["s"][mc]], np.r_[run.top_e, I["e"][mc]], np.r_[run.top_sym, I["sym"][mc]])
    info = dict(mobile_types=int(mobile.sum()), mobile_cores=int(len(mc)), nest_children=int(len(ch)))
    out = {}
    for k, (s, e, m) in rd.items():
        se = pd.DataFrame(dict(s=s, e=e, m=m)).drop_duplicates(["s", "e"]).sort_values(["s", "e"])
        out[k] = se
    return out, info


class Ctx:
    def __init__(self):
        self.genes = pd.read_parquet(B.PKG / "universe/genes.parquet", columns=["gid", "genome", "genome_code", "contig",
                                                                                "position_index"])
        self.genome_arr = self.genes.genome.to_numpy()
        self.contig_arr = self.genes.contig.to_numpy()
        self.pos = self.genes.position_index.to_numpy()
        self.gc = self.genes.genome_code.to_numpy(np.int64)
        self.gcode = B.odd_even_code()
        self.runs = B.ribo_runs()
        self.odd_runs = (self.runs.genome_code % 2 == 1).to_numpy()

    def table(self, se):
        s, e = se.s.to_numpy(np.int64), se.e.to_numpy(np.int64)
        keep = e > s
        s, e, m = s[keep], e[keep], se.m.to_numpy(np.int64)[keep]
        return pd.DataFrame(dict(genome=self.genome_arr[s], contig=self.contig_arr[s], start=self.pos[s], end=self.pos[e],
                                 module_id=m, gid_s=s, gid_e=e))


def score(ctx, df, threads):
    per, _ = B.score_table(B.PKG, df, threads)
    g = per.genome.map(ctx.gcode).to_numpy()
    summ = {sp: B.summarize_per(per[(g % 2 == (1 if sp == "odd" else 0))].reset_index(drop=True)) for sp in ("odd", "even")}
    sc = B.ribo_score(ctx.runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
    rib = dict(all=B.ribo_summary(ctx.runs, sc),
               odd=B.ribo_summary(ctx.runs[ctx.odd_runs].reset_index(drop=True), sc[ctx.odd_runs].reset_index(drop=True)),
               even=B.ribo_summary(ctx.runs[~ctx.odd_runs].reset_index(drop=True), sc[~ctx.odd_runs].reset_index(drop=True)))
    loc = B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), len(ctx.gc))
    return per, summ, sc, rib, loc


def parv(x):
    return int(str(x).split("/")[0])


def odd_conditions(summ_odd, rib_odd, ref_odd):
    c_sum = summ_odd["summed_DF_TXSS_excess"] >= ref_odd["summed_DF_TXSS_excess"] - 0.01
    c_f1 = all(summ_odd[c]["f1"] >= ref_odd[c]["f1"] - 0.02 for c in CALLERS)
    c_rib = (rib_odd["recovered"] >= ODD_BARS["recovered"] and rib_odd["resolved"] >= ODD_BARS["resolved"]
             and parv(rib_odd["parvarchaeaceae"]) >= ODD_BARS["parv"])
    return [bool(c_sum), bool(c_f1), bool(c_rib)]


def verdict(pr, rib_all, ref_rib):
    """matches / beats against the reference (pr = rules.paired(A, ref) on even)."""
    d_sum = pr["summed"]
    m_sum = d_sum[0] >= -0.01
    m_f1 = {c: pr[c]["d_f1"][0] >= -0.02 for c in CALLERS}
    m_rib = (rib_all["recovered"] >= ALL_BARS["recovered"] and rib_all["resolved"] >= ALL_BARS["resolved"]
             and parv(rib_all["parvarchaeaceae"]) >= ALL_BARS["parv"])
    matches = bool(m_sum and all(m_f1.values()) and m_rib)
    beats = bool(matches and d_sum[1] > 0)
    return dict(matches=matches, beats=beats, summed_ok=bool(m_sum), f1_ok={c: bool(v) for c, v in m_f1.items()},
                ribo_ok=bool(m_rib), label=("beats" if beats else "matches" if matches else "neither"))


# --------------------------------------------------------------------------- commands
def cmd_refcheck(a):
    ctx = Ctx()
    out = {}
    for name, lp, sp in (("DH", DH_LOCI, DH_SCORES), ("R1", R1_LOCI, R1_SCORES)):
        df = pd.read_parquet(lp)
        per, summ, sc, rib, loc = score(ctx, df, a.threads)
        stored = pd.read_parquet(sp).set_index("instance_id")
        p = per.set_index("instance_id").loc[stored.index]
        same = {c: bool(np.array_equal(p[c].to_numpy(), stored[c].to_numpy())) for c in ("f1", "control_mean_f1", "coverage")}
        out[name] = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo_all=rib["all"], ribo_odd=rib["odd"],
                         loci=loc, per_instance_equal_stored=same)
        log(name, "summed even", round(summ["even"]["summed_DF_TXSS_excess"], 4),
            {c: round(summ["even"][c]["f1"], 3) for c in CALLERS}, "ribo", rib["all"]["recovered"], rib["all"]["resolved"],
            rib["all"]["parvarchaeaceae"], rib["all"]["parvarchaeaceae_resolved"], "equal stored", same)
    dh = out["DH"]
    ok = (round(dh["summary_even"]["summed_DF_TXSS_excess"], 4) == REF_EVEN["summed"]
          and all(round(dh["summary_even"][c]["f1"], 3) == REF_EVEN["f1"][c] for c in CALLERS)
          and dh["ribo_all"]["recovered"] == REF_EVEN["recovered"] and dh["ribo_all"]["resolved"] == REF_EVEN["resolved"]
          and parv(dh["ribo_all"]["parvarchaeaceae_resolved"]) == REF_EVEN["parv_resolved"]
          and all(dh["per_instance_equal_stored"].values()))
    out["reproduced"] = bool(ok)
    log("DH reproduced exactly:", ok)
    jdump(out, VD / "results" / "refcheck.json")


def cmd_odd(a):
    ctx = Ctx()
    grp = grp_map(a.arm, ctx)
    run = Run(a.arm, a.tau)
    rds, info = readings(run, grp, ctx.gc)
    ref = json.load(open(VD / "results" / "refcheck.json"))
    assert ref["reproduced"]
    dh_odd, dh_rib_odd = ref["DH"]["summary_odd"], ref["DH"]["ribo_odd"]
    dh_per = pd.read_parquet(DH_SCORES).set_index("instance_id")
    res = dict(arm=a.arm, tau=a.tau, info=info, DH_odd=dict(summary=dh_odd, ribo=dh_rib_odd), readings={})
    for name, se in rds.items():
        df = ctx.table(se)
        per, summ, sc, rib, loc = score(ctx, df, a.threads)
        pr = RU.paired(per.set_index("instance_id"), dh_per, ctx.gcode, "odd")
        cond = odd_conditions(summ["odd"], rib["odd"], dh_odd)
        res["readings"][name] = dict(summary_odd=summ["odd"], ribo_odd=rib["odd"], loci=loc, vs_DH_odd=pr,
                                     conditions_odd=cond, score=[sum(cond), summ["odd"]["summed_DF_TXSS_excess"]])
        log(name, "odd summed", round(summ["odd"]["summed_DF_TXSS_excess"], 4), "vs DH", [round(x, 4) for x in pr["summed"]],
            {c: round(summ["odd"][c]["f1"], 3) for c in CALLERS}, "ribo odd", rib["odd"]["recovered"],
            rib["odd"]["resolved"], rib["odd"]["parvarchaeaceae"], "cond", cond, "loci", loc)
    cands = {k: v for k, v in res["readings"].items() if k in ("V-nest", "V-mob")}
    best = max(cands, key=lambda k: tuple(cands[k]["score"]))
    res["selected_primary"] = best
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log("selected primary nested reading (odd):", best)
    jdump(res, VD / "results" / f"odd_{a.arm}_tau{a.tau:g}{('_' + a.tag) if a.tag else ''}.json")


def grp_map(arm, ctx):
    import vocab as V
    return V.load_groups(arm, int(ctx.gc.max()) + 1)


def cmd_even(a):
    ctx = Ctx()
    grp = grp_map(a.arm, ctx)
    run = Run(a.arm, a.tau)
    rds, info = readings(run, grp, ctx.gc)
    ref = json.load(open(VD / "results" / "refcheck.json"))
    assert ref["reproduced"]
    dh_per = pd.read_parquet(DH_SCORES).set_index("instance_id")
    r1_per = pd.read_parquet(R1_SCORES).set_index("instance_id")
    dh_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(DH_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    res = dict(arm=a.arm, tau=a.tau, primary=a.primary, info=info, readings={})
    (VD / "loci").mkdir(exist_ok=True)
    (VD / "scores").mkdir(exist_ok=True)
    pers = {}
    for name, se in rds.items():
        df = ctx.table(se)
        tag = f"{a.arm}_tau{a.tau:g}_{name}"
        df.to_parquet(VD / "loci" / f"{tag}.parquet", index=False)
        per, summ, sc, rib, loc = score(ctx, df, a.threads)
        per.to_parquet(VD / "scores" / f"{tag}.parquet", index=False)
        pers[name] = per
        P = per.set_index("instance_id")
        pr_dh = RU.paired(P, dh_per, ctx.gcode, "even")
        pr_r1 = RU.paired(P, r1_per, ctx.gcode, "even")
        item = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo=rib, loci=loc, vs_DH_even=pr_dh,
                    vs_R1_even=pr_r1, ribo_resolved_vs_DH=RU.ribo_paired(sc, dh_sc),
                    verdict_vs_DH=verdict(pr_dh, rib["all"], ref["DH"]["ribo_all"]))
        res["readings"][name] = item
        log(name, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "vs DH", [round(x, 4) for x in pr_dh["summed"]],
            "dF1", {c[:4]: [round(x, 3) for x in pr_dh[c]["d_f1"]] for c in CALLERS}, "ribo", rib["all"]["recovered"],
            rib["all"]["resolved"], rib["all"]["parvarchaeaceae"], rib["all"]["parvarchaeaceae_resolved"],
            "verdict", item["verdict_vs_DH"]["label"], "vs R1", [round(x, 4) for x in pr_r1["summed"]])
    res["typing"] = typing(ctx, run, pers, a.threads)
    res["vocabulary"] = vocab_stats(ctx, run, grp)
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, VD / "results" / f"even_{a.arm}_tau{a.tau:g}{('_' + a.tag) if a.tag else ''}.json")


# --------------------------------------------------------------------------- typing
def label_scores(y_true, y_pred):
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    return dict(n=int(len(y_true)), n_true=int(len(np.unique(y_true))), n_pred=int(len(np.unique(y_pred))),
                ami=float(adjusted_mutual_info_score(y_true, y_pred)), homogeneity=float(h), completeness=float(c),
                v_measure=float(v))


def typing(ctx, run, pers, threads):
    """Word identity vs gold system type/subtype on even gold instances, against engine module labels on the same
    instances. Labels: plurality over member genes (top-level word type; engine token module at Louvain res 10 and at
    res 0.3), requiring the plurality unit to hold >= 50% of members; and the scorer's best-locus module."""
    inst = pd.read_parquet(B.PKG / "gold/instances.parquet",
                           columns=["instance_id", "caller", "system_type", "system_subtype", "genome"])
    mem = pd.read_parquet(B.PKG / "gold/members.parquet")
    n = len(ctx.gc)
    # gene -> top-level word type (-1 background or terminal)
    gene_word = np.full(n, -1, dtype=np.int64)
    gene_topid = np.full(n, -1, dtype=np.int64)
    lens = run.top_e - run.top_s + 1
    idx = np.repeat(np.arange(len(run.top_s)), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    genes = run.top_s[idx] + (np.arange(len(idx)) - off[idx])
    gene_word[genes] = run.top_sym[idx]
    gene_topid[genes] = idx
    L = B.Landscape(B.ARMS["rescue3"], verify=False)
    bld = RU.Builder(L)
    tm03, _ = bld.modules(0.3)
    mod10 = L.gene_module
    mod03 = np.where(L.tok >= 0, tm03[np.maximum(L.tok, 0)], -1)

    def plurality(lbl, unit=None):
        m = mem.copy()
        m["lab"] = lbl[m.gid.to_numpy()]
        m["unit"] = (unit if unit is not None else lbl)[m.gid.to_numpy()]
        tot = m.groupby("instance_id").size()
        mm = m[m.unit >= 0]
        cnt = mm.groupby(["instance_id", "unit", "lab"]).size().rename("k").reset_index()
        cnt = cnt.sort_values(["instance_id", "k", "lab"], ascending=[True, False, True]).drop_duplicates("instance_id")
        cnt["frac"] = cnt.k.to_numpy() / tot.loc[cnt.instance_id].to_numpy()
        cnt = cnt[cnt.frac >= 0.5]
        return cnt.set_index("instance_id").lab

    labs = dict(word=plurality(gene_word, gene_topid), module_res10=plurality(mod10), module_res03=plurality(mod03))
    inst = inst.set_index("instance_id")
    even = inst.genome.map(ctx.gcode).to_numpy() % 2 == 0
    mcount = mem.groupby("instance_id").size()
    inst["n_mem"] = mcount.reindex(inst.index).fillna(0).to_numpy()
    out = {}
    for caller in CALLERS:
        sel = inst[(inst.caller == caller) & even & (inst.n_mem >= 2)]
        r = dict(n_gold=int(len(sel)), covered={k: int(sel.index.isin(v.index).sum()) for k, v in labs.items()})
        for eng in ("module_res10", "module_res03"):
            both = sel.index[sel.index.isin(labs["word"].index) & sel.index.isin(labs[eng].index)]
            for tcol in ("system_type", "system_subtype"):
                yt = sel.loc[both, tcol].fillna("NA").astype(str).to_numpy()
                r[f"{eng}|{tcol}"] = dict(word=label_scores(yt, labs["word"].loc[both].to_numpy()),
                                          engine=label_scores(yt, labs[eng].loc[both].to_numpy()))
        # scorer best-locus labels (V-top vs token DH best_module), instances with covered50 in both
        dh = pd.read_parquet(DH_SCORES).set_index("instance_id")
        vt = pers["V-top"].set_index("instance_id").copy()
        dh["best_module"] = pd.to_numeric(dh.best_module, errors="coerce").fillna(-1).astype(np.int64)
        vt["best_module"] = pd.to_numeric(vt.best_module, errors="coerce").fillna(-1).astype(np.int64)
        ids = sel.index.intersection(dh.index).intersection(vt.index)
        ids = ids[(dh.loc[ids].covered50.to_numpy() > 0) & (vt.loc[ids].covered50.to_numpy() > 0)
                  & (dh.loc[ids].best_module.to_numpy() >= 0) & (vt.loc[ids].best_module.to_numpy() >= 0)]
        for tcol in ("system_type", "system_subtype"):
            yt = sel.loc[ids, tcol].fillna("NA").astype(str).to_numpy()
            r[f"bestlocus|{tcol}"] = dict(word=label_scores(yt, vt.loc[ids].best_module.to_numpy()),
                                          engine=label_scores(yt, dh.loc[ids].best_module.to_numpy()))
        out[caller] = r
        log("typing", caller, {k: (round(v["word"]["ami"], 3), round(v["engine"]["ami"], 3)) for k, v in r.items()
                               if "|" in k})
    return out


# --------------------------------------------------------------------------- vocabulary statistics
@njit(cache=True)
def type_size_depth(T, wsym, wa, wb, S):
    size = np.zeros(S, dtype=np.int64)
    depth = np.zeros(S, dtype=np.int64)
    size[:T] = 1
    for i in range(len(wsym)):          # words are created in increasing symbol order
        x, y = wa[i], wb[i]
        size[wsym[i]] = size[x] + size[y]
        depth[wsym[i]] = 1 + max(depth[x], depth[y])
    return size, depth


def vocab_stats(ctx, run, grp):
    w = run.words.sort_values("sym")
    S = int(w.sym.max()) + 1 if len(w) else run.T
    size, depth = type_size_depth(run.T, w.sym.to_numpy(np.int64), w.a.to_numpy(np.int64), w.b.to_numpy(np.int64), S)
    ws, wd = size[w.sym.to_numpy()], depth[w.sym.to_numpy()]
    n = len(ctx.gc)
    top_len = run.top_e - run.top_s + 1
    top_slots = size[run.top_sym]
    out = dict(word_types=int(len(w)), instances=int(len(run.inst["sym"])), top_instances=int(len(run.top_s)),
               top_types=int(len(np.unique(run.top_sym))),
               types_by_slots={str(k): int(v) for k, v in zip(*np.unique(np.minimum(ws, 50), return_counts=True))},
               types_by_depth={str(k): int(v) for k, v in zip(*np.unique(wd, return_counts=True))},
               top_types_by_slots={str(k): int(v) for k, v in zip(*np.unique(np.minimum(size[np.unique(run.top_sym)], 50),
                                                                              return_counts=True))},
               top_instance_span_quantiles=np.quantile(top_len, [.1, .25, .5, .75, .9, .99, 1]).tolist(),
               genes_in_top=int(top_len.sum()), genes_in_top_frac=float(top_len.sum() / n),
               slot_genes=int(top_slots.sum()), inserted_genes=int(top_len.sum() - top_slots.sum()),
               max_depth=int(wd.max()) if len(wd) else 0,
               lam_quantiles=np.quantile(w.lam, [.01, .1, .5, .9, .99]).tolist() if len(w) else [],
               n_groups_quantiles=np.quantile(w.n, [.1, .5, .9, .99]).tolist() if len(w) else [])
    # ribosomal runs: top-level instances overlapping each run, best single-instance coverage, its type
    runs = ctx.runs
    a = np.argsort(run.top_s)
    ts, te, tsym = run.top_s[a], run.top_e[a], run.top_sym[a]
    rows = []
    for rs, re_, fam in zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy(), runs.family.to_numpy()):
        j0 = np.searchsorted(te, rs)
        best, bsym, k, blen = 0, -1, 0, 0
        j = j0
        while j < len(ts) and ts[j] <= re_:
            ov = min(te[j], re_) - max(ts[j], rs) + 1
            if ov > 0:
                k += 1
                if ov > best:
                    best, bsym, blen = ov, tsym[j], te[j] - ts[j] + 1
            j += 1
        rows.append((k, best / (re_ - rs + 1), bsym, blen, fam == "Parvarchaeaceae", re_ - rs + 1))
    rr = pd.DataFrame(rows, columns=["n_top", "best_frac", "best_type", "best_len", "parv", "run_len"])
    one = rr.best_frac >= 0.8
    vc = rr[one].best_type.value_counts()
    out["ribosomal"] = dict(runs=int(len(rr)), median_top_instances=float(rr.n_top.median()),
                            runs_80pct_in_one_instance=int(one.sum()),
                            parv_80pct_in_one_instance=f"{int((one & rr.parv).sum())}/{int(rr.parv.sum())}",
                            distinct_types_covering_80pct=int(len(vc)), top_type_share=float(vc.iloc[0] / one.sum()) if len(vc) else None,
                            top3_types=[[int(k), int(v), int(size[k]), int(depth[k])] for k, v in vc.head(3).items()],
                            median_best_len_over_run_len=float((rr.best_len / rr.run_len)[one].median()) if one.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refcheck", "odd", "even"])
    ap.add_argument("--arm", default="main")
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--primary", default="V-nest")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    sys.path.insert(0, str(VD / "code"))
    {"refcheck": cmd_refcheck, "odd": cmd_odd, "even": cmd_even}[a.cmd](a)


if __name__ == "__main__":
    main()
