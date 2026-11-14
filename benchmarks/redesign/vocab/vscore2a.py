#!/usr/bin/env python
"""Round-2A readings and scoring (predeclared2a.json).

Readings of a runs2a vocabulary:
  V-top  top-level word instances (label = class label of the live node, else its symbol)
  V-mob  V-top + maximal sub-instances whose label is also a top-level label in >= 2 units (groups for group-mode
         runs, genomes for genome / N_eff runs)
  V-phr  V-top + phrases: maximal runs of consecutive top-level words (<= 2 genes apart) whose gaps are covered by the
         run's own round-0 significant adjacencies (b0.npy)
usage (biotite, Slurm):
  vscore2a.py odd  --src runs2a/<tag> --tau T [--tag X]
  vscore2a.py even --src runs2a/<tag> --tau T --primary V-mob|V-phr [--compare <score prefix>] [--tag X]
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
sys.path.insert(0, str(VD / "code"))
import vscore as S1  # noqa: E402
import vscore2 as S2  # noqa: E402
import vocab as V  # noqa: E402

B, RU = S1.B, S1.RU
KAPPA = RD / "context_units/ctx/kappa_var_tokens.npy"      # ws_context token cohesion, read-only
COH_THETAS = (0.02, 0.05, 0.1, 0.2)


@njit(cache=True)
def phrases_coh(ts, te, tc, kappa, theta):
    """Consecutive top words (sorted by start, same contig, <= 2 genes apart) joined when every boundary from the end of
    one word to the start of the next has kappa >= theta or NaN."""
    n = len(ts)
    out_s = np.empty(n, dtype=np.int64)
    out_e = np.empty(n, dtype=np.int64)
    m = 0
    i = 0
    while i < n:
        s, e = ts[i], te[i]
        j = i + 1
        while j < n and tc[j] == tc[i] and ts[j] - e - 1 <= 2:
            ok = True
            for k in range(e, ts[j]):
                if kappa[k] < theta:
                    ok = False
                    break
            if not ok:
                break
            e = te[j]
            j += 1
        out_s[m] = s
        out_e[m] = e
        m += 1
        i = j
    return out_s[:m], out_e[:m]
log = B.log
CALLERS = S1.CALLERS


class Run2A:
    def __init__(self, src, tau):
        d = VD / src / f"tau{tau:g}" / "obs"
        self.d = d
        self.meta = json.load(open(d / "run.json"))
        self.inst = dict(np.load(d / "inst.npz"))
        self.live = dict(np.load(d / "live.npz"))
        self.b0 = np.load(d / "b0.npy") if (d / "b0.npy").exists() else None
        self.T = self.meta["T"]
        top = ~self.live["term"]
        self.top_inst = self.live["ninst"][top]
        self.top_s, self.top_e = self.live["ns"][top], self.live["ne"][top]
        self.top_sym = self.live["sym"][top]                       # class label (or exact symbol)
        if "label" not in self.inst:
            self.inst["label"] = self.inst["sym"].copy()
        self.spec = self.meta.get("spec", "group:none:0:none:0")
        self.mode = self.spec.split(":")[0]


def readings(run, unit_of_genome, gc, cc):
    I = run.inst
    rd = {"V-top": pd.DataFrame(dict(s=run.top_s, e=run.top_e, m=run.top_sym))}
    lab = I["label"]
    S = int(max(lab.max() if len(lab) else 0, run.top_sym.max() if len(run.top_sym) else 0)) + 1
    nu = int(unit_of_genome.max()) + 1
    tg = np.unique(run.top_sym * nu + unit_of_genome[gc[run.top_s]])
    mobile = np.bincount(tg // nu, minlength=S) >= 2
    mc = S1.mobile_cores(run.top_inst.astype(np.int64), I["a"], I["b"], lab, mobile)
    rd["V-mob"] = pd.DataFrame(dict(s=np.r_[run.top_s, I["s"][mc]], e=np.r_[run.top_e, I["e"][mc]],
                                    m=np.r_[run.top_sym, lab[mc]]))
    o = np.argsort(run.top_s)
    ps, pe = S2.phrases(run.top_s[o], run.top_e[o], cc[run.top_s[o]], run.b0)
    rd["V-phr"] = pd.DataFrame(dict(s=np.r_[run.top_s, ps], e=np.r_[run.top_e, pe],
                                    m=np.r_[run.top_sym, np.full(len(ps), -1, dtype=np.int64)]))
    words = set(zip(run.top_s.tolist(), run.top_e.tolist()))
    multi = np.array([(x, y) not in words for x, y in zip(ps.tolist(), pe.tolist())])
    kappa = np.load(KAPPA).astype(np.float64)
    coh = {}
    for th in COH_THETAS:
        cs_, ce_ = phrases_coh(run.top_s[o], run.top_e[o], cc[run.top_s[o]], kappa, th)
        rd[f"V-coh{th:g}"] = pd.DataFrame(dict(s=np.r_[run.top_s, cs_], e=np.r_[run.top_e, ce_],
                                               m=np.r_[run.top_sym, np.full(len(cs_), -1, dtype=np.int64)]))
        mm = np.array([(x, y) not in words for x, y in zip(cs_.tolist(), ce_.tolist())])
        coh[f"{th:g}"] = dict(multi=int(mm.sum()), genes_in_multi=int((ce_ - cs_ + 1)[mm].sum()) if mm.any() else 0)
    info = dict(top=int(len(run.top_s)), top_labels=int(len(np.unique(run.top_sym))), mobile_labels=int(mobile.sum()),
                mobile_cores=int(len(mc)), phrases_multi=int(multi.sum()),
                genes_in_multi_phrases=int((pe - ps + 1)[multi].sum()) if multi.any() else 0, coh_phrases=coh)
    return {k: v.drop_duplicates(["s", "e"]).sort_values(["s", "e"]) for k, v in rd.items()}, info


def setup(a):
    ctx = S1.Ctx()
    cc, gc, _ = V.load_universe()
    run = Run2A(a.src, a.tau)
    ng = int(gc.max()) + 1
    unit = V.load_groups("main", ng) if run.mode == "group" else np.arange(ng, dtype=np.int64)
    if run.b0 is None:          # stage-1 vocabulary: round-0 significant adjacencies of its own statistic
        tok = V.load_tokens(len(cc))
        run.b0, _ = S2.significant_boundaries(tok[:, None], cc, gc, unit, a.tau)
    rds, info = readings(run, unit, gc, cc)
    info["spec"] = run.spec
    info["phases"] = run.meta.get("phases", [])
    return ctx, cc, gc, run, rds, info


def cmd_odd(a):
    ctx, cc, gc, run, rds, info = setup(a)
    ref = json.load(open(VD / "results" / "refcheck.json"))
    assert ref["reproduced"]
    dh_odd = ref["DH"]["summary_odd"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    res = dict(src=a.src, tau=a.tau, info=info, readings={})
    for name, se in rds.items():
        df = ctx.table(se)
        per, summ, sc, rib, loc = S1.score(ctx, df, a.threads)
        pr = RU.paired(per.set_index("instance_id"), dh_per, ctx.gcode, "odd")
        cond = S1.odd_conditions(summ["odd"], rib["odd"], dh_odd)
        res["readings"][name] = dict(summary_odd=summ["odd"], ribo_odd=rib["odd"], loci=loc, vs_DH_odd=pr,
                                     conditions_odd=cond, score=[sum(cond), summ["odd"]["summed_DF_TXSS_excess"]])
        log(a.src, name, "odd summed", round(summ["odd"]["summed_DF_TXSS_excess"], 4), "vs DH",
            [round(x, 4) for x in pr["summed"]], {c[:4]: round(summ["odd"][c]["f1"], 3) for c in CALLERS},
            "ribo odd", rib["odd"]["recovered"], rib["odd"]["resolved"], rib["odd"]["parvarchaeaceae"], "cond", cond,
            "genes", loc["genes_covered"])
    cands = {k: v for k, v in res["readings"].items() if k != "V-top"}
    res["selected_reading"] = max(cands, key=lambda k: tuple(cands[k]["score"]))
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log(a.src, "best odd reading:", res["selected_reading"], cands[res["selected_reading"]]["score"], "| info", info)
    S1.jdump(res, VD / "results" / f"odd2a_{a.src.replace('/', '_')}_tau{a.tau:g}{('_' + a.tag) if a.tag else ''}.json")


def cmd_even(a):
    ctx, cc, gc, run, rds, info = setup(a)
    ref = json.load(open(VD / "results" / "refcheck.json"))
    assert ref["reproduced"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    dh_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(S1.DH_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    tagb = f"{a.src.replace('/', '_')}_tau{a.tau:g}{('_' + a.tag) if a.tag else ''}"
    res = dict(src=a.src, tau=a.tau, primary=a.primary, info=info, readings={})
    pers = {}
    for name, se in rds.items():
        df = ctx.table(se)
        df.to_parquet(VD / "loci" / f"{tagb}_{name}.parquet", index=False)
        per, summ, sc, rib, loc = S1.score(ctx, df, a.threads)
        per.to_parquet(VD / "scores" / f"{tagb}_{name}.parquet", index=False)
        pers[name] = per
        P = per.set_index("instance_id")
        pr_dh = RU.paired(P, dh_per, ctx.gcode, "even")
        item = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo=rib, loci=loc, vs_DH_even=pr_dh,
                    ribo_resolved_vs_DH=RU.ribo_paired(sc, dh_sc), verdict_vs_DH=S1.verdict(pr_dh, rib["all"], ref["DH"]["ribo_all"]))
        for cmp in [c for c in a.compare.split(",") if c]:
            fn = VD / "scores" / f"{cmp}_{name}.parquet"
            if fn.exists():
                item[f"vs_{cmp}"] = RU.paired(P, pd.read_parquet(fn).set_index("instance_id"), ctx.gcode, "even")
        res["readings"][name] = item
        log(a.src, name, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "vs DH",
            [round(x, 4) for x in pr_dh["summed"]], "F1", {c[:4]: round(summ["even"][c]["f1"], 3) for c in CALLERS},
            "dF1", {c[:4]: [round(x, 3) for x in pr_dh[c]["d_f1"]] for c in CALLERS}, "ribo", rib["all"]["recovered"],
            rib["all"]["resolved"], rib["all"]["parvarchaeaceae"], rib["all"]["parvarchaeaceae_resolved"], "verdict",
            item["verdict_vs_DH"]["label"],
            {k: [round(x, 4) for x in v["summed"]] for k, v in item.items() if k.startswith("vs_") and k != "vs_DH_even"})
    mem = pd.read_parquet(B.PKG / "gold/members.parquet")
    res["typing"] = S1.typing(ctx, run, pers, a.threads)
    inword = np.zeros(len(gc), dtype=bool)
    lens = run.top_e - run.top_s + 1
    idx = np.repeat(np.arange(len(run.top_s)), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    inword[run.top_s[idx] + (np.arange(len(idx)) - off[idx])] = True
    k = mem.groupby("instance_id").gid.transform("size")
    mm = mem[k >= 2]
    res["members_outside_words"] = {c: float(1 - inword[g.gid.to_numpy()].mean()) for c, g in mm.groupby("caller")}
    log("members outside words", res["members_outside_words"])
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    S1.jdump(res, VD / "results" / f"even2a_{tagb}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["odd", "even"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--primary", default="V-mob")
    ap.add_argument("--compare", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"odd": cmd_odd, "even": cmd_even}[a.cmd](a)


if __name__ == "__main__":
    main()
