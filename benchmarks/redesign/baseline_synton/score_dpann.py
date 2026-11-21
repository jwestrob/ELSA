#!/usr/bin/env python
"""DPANN scoring of synton loci exactly as vocab/vscore.py scores word readings (workstream baseline_synton).

Scorer: boundary/code/blib.score_table (ws_dpann package scorer, 200 controls, seed 17, min_members 2), summaries on
even genome_code (test split), ribosomal check over all runs (blib.ribo_score / ribo_summary), paired genome bootstrap
rules.paired (2,000 draws, seed 17) and rules.ribo_paired. All imports read-only.

usage
  score_dpann.py refcheck --threads T          token DH loci must reproduce the stored reference exactly
  score_dpann.py score --loci P --label L [--loci P --label L ...] --tag NAME --threads T
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
BS = RD / "baseline_synton"
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as RU  # noqa: E402

CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
DH_LOCI = RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet"
DH_SCORES = RD / "boundary/scores/rescue3/DH_coreH_res0.3_f0.5.parquet"
WORD_LOCI = RD / "vocab/loci/runs_main_tau10_s1coh_V-coh0.1.parquet"
WORD_SCORES = RD / "vocab/scores/runs_main_tau10_s1coh_V-coh0.1.parquet"
REF_EVEN = dict(summed=0.4270, f1={"DefenseFinder": 0.744, "TXSScan": 0.548, "CRISPRCasTyper": 0.479},
                recovered=1686, resolved=1512, parv_resolved=113)
log = B.log


def jdump(obj, fn):
    with open(fn, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log("wrote", fn)


class Ctx:
    def __init__(self):
        self.genes = pd.read_parquet(B.PKG / "universe/genes.parquet",
                                     columns=["gid", "genome", "genome_code", "contig", "contig_code", "position_index"])
        assert np.array_equal(self.genes.gid.to_numpy(), np.arange(len(self.genes)))
        self.gcode = B.odd_even_code()
        self.runs = B.ribo_runs()
        self.odd_runs = (self.runs.genome_code % 2 == 1).to_numpy()
        first = self.genes.groupby("contig_code").gid.min()
        assert np.array_equal(first.loc[self.genes.contig_code].to_numpy() + self.genes.position_index.to_numpy(),
                              self.genes.gid.to_numpy()), "gid = first gid of contig + position_index"
        self.first = first


def score(ctx, df, threads):
    per, _ = B.score_table(B.PKG, df, threads)
    g = per.genome.map(ctx.gcode).to_numpy()
    summ = {sp: B.summarize_per(per[(g % 2 == (1 if sp == "odd" else 0))].reset_index(drop=True)) for sp in ("odd", "even")}
    sc = B.ribo_score(ctx.runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
    rib = dict(all=B.ribo_summary(ctx.runs, sc),
               even=B.ribo_summary(ctx.runs[~ctx.odd_runs].reset_index(drop=True), sc[~ctx.odd_runs].reset_index(drop=True)))
    loc = B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), len(ctx.genes))
    return per, summ, sc, rib, loc


def card(summ_even, rib_all):
    return dict(summed=summ_even["summed_DF_TXSS_excess"], f1={c: summ_even[c]["f1"] for c in CALLERS},
                control={c: summ_even[c]["control"] for c in CALLERS},
                bprec={c: summ_even[c]["median_bprec_covered"] for c in CALLERS},
                ribo_recovered=rib_all["recovered"], ribo_resolved=rib_all["resolved"],
                parvarchaeaceae=rib_all["parvarchaeaceae"], parvarchaeaceae_resolved=rib_all["parvarchaeaceae_resolved"])


def cmd_refcheck(a):
    ctx = Ctx()
    df = pd.read_parquet(DH_LOCI)
    per, summ, sc, rib, loc = score(ctx, df, a.threads)
    stored = pd.read_parquet(DH_SCORES).set_index("instance_id")
    p = per.set_index("instance_id").loc[stored.index]
    same = {c: bool(np.array_equal(p[c].to_numpy(), stored[c].to_numpy())) for c in ("f1", "control_mean_f1", "coverage")}
    se = summ["even"]
    ok = (round(se["summed_DF_TXSS_excess"], 4) == REF_EVEN["summed"]
          and all(round(se[c]["f1"], 3) == REF_EVEN["f1"][c] for c in CALLERS)
          and rib["all"]["recovered"] == REF_EVEN["recovered"] and rib["all"]["resolved"] == REF_EVEN["resolved"]
          and int(str(rib["all"]["parvarchaeaceae_resolved"]).split("/")[0]) == REF_EVEN["parv_resolved"]
          and all(same.values()))
    out = dict(DH=dict(card=card(se, rib["all"]), summary_even=se, ribo_all=rib["all"], loci=loc,
                       per_instance_equal_stored=same), reproduced=bool(ok))
    # the word method's adopted output, rescored the same way (reference row; its stored even summed is .6082)
    wdf = pd.read_parquet(WORD_LOCI)
    wper, wsumm, wsc, wrib, wloc = score(ctx, wdf, a.threads)
    wstored = pd.read_parquet(WORD_SCORES).set_index("instance_id")
    wp = wper.set_index("instance_id").loc[wstored.index]
    out["word"] = dict(card=card(wsumm["even"], wrib["all"]), loci=wloc,
                       per_instance_equal_stored={c: bool(np.array_equal(wp[c].to_numpy(), wstored[c].to_numpy()))
                                                  for c in ("f1", "control_mean_f1", "coverage")})
    log("DH", out["DH"]["card"], "equal stored", same, "REPRODUCED", ok)
    log("word V-coh0.1", out["word"]["card"], "equal stored", out["word"]["per_instance_equal_stored"])
    jdump(out, BS / "results/dpann_refcheck.json")


def to_table(ctx, loci):
    s = ctx.first.loc[loci.contig_code].to_numpy() + loci.start.to_numpy()
    e = ctx.first.loc[loci.contig_code].to_numpy() + loci.end.to_numpy()
    mod = loci["family"].to_numpy(np.int64) if "family" in loci else np.full(len(loci), -1, dtype=np.int64)
    keep = e > s
    g = ctx.genes
    return pd.DataFrame(dict(genome=g.genome.to_numpy()[s[keep]], contig=g.contig.to_numpy()[s[keep]],
                             start=g.position_index.to_numpy()[s[keep]], end=g.position_index.to_numpy()[e[keep]],
                             module_id=mod[keep], gid_s=s[keep], gid_e=e[keep]))


def cmd_score(a):
    ref = json.load(open(BS / "results/dpann_refcheck.json"))
    assert ref["reproduced"], "DH reference not reproduced"
    ctx = Ctx()
    dh_per = pd.read_parquet(DH_SCORES).set_index("instance_id")
    w_per = pd.read_parquet(WORD_SCORES).set_index("instance_id")
    dh_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(DH_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    w_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(WORD_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    res = dict(references=dict(DH=ref["DH"]["card"], word=ref["word"]["card"]), outputs={})
    pers = {}
    for path, label in zip(a.loci, a.label):
        df = to_table(ctx, pd.read_parquet(path))
        per, summ, sc, rib, loc = score(ctx, df, a.threads)
        (BS / "scores").mkdir(exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in label)
        per.to_parquet(BS / "scores" / f"dpann_{a.tag}_{safe}.parquet", index=False)
        P = per.set_index("instance_id")
        pers[label] = (P, sc)
        item = dict(card=card(summ["even"], rib["all"]), summary_even=summ["even"], ribo=rib, loci=loc,
                    vs_DH_even=RU.paired(P, dh_per, ctx.gcode, "even"),
                    vs_word_even=RU.paired(P, w_per, ctx.gcode, "even"),
                    ribo_resolved_vs_DH=RU.ribo_paired(sc, dh_sc), ribo_resolved_vs_word=RU.ribo_paired(sc, w_sc))
        res["outputs"][label] = item
        log(label, "card", item["card"], "loci", loc)
        log(label, "vs DH summed", [round(x, 4) for x in item["vs_DH_even"]["summed"]],
            "vs word summed", [round(x, 4) for x in item["vs_word_even"]["summed"]],
            "dF1 vs DH", {c[:4]: [round(x, 3) for x in item["vs_DH_even"][c]["d_f1"]] for c in CALLERS})
    labs = list(pers)
    res["between"] = {f"{x} - {y}": dict(summed=RU.paired(pers[x][0], pers[y][0], ctx.gcode, "even")["summed"],
                                         ribo_resolved=RU.ribo_paired(pers[x][1], pers[y][1]))
                      for i, x in enumerate(labs) for y in labs[:i]}
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, BS / "results" / f"dpann_{a.tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refcheck", "score"])
    ap.add_argument("--loci", action="append")
    ap.add_argument("--label", action="append")
    ap.add_argument("--tag", default="scores")
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"refcheck": cmd_refcheck, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
