#!/usr/bin/env python
"""Compact report tables and the predeclared verdict from summarize.py outputs.

Reads $RD/ws_metric/<tag>/results/{retrieval.tsv,pair_metrics.tsv,retrieval_diff_ci.tsv,
pair_auroc_diff_ci.tsv[,retrieval_tm.tsv]} and writes REPORT_TABLES.md beside them.
Small inputs; runs in seconds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import LEARNED, LEARNED_R2, MODELS, UNTRAINED, W

R1 = [k for k in LEARNED if k not in LEARNED_R2]
R2 = list(LEARNED_R2)
PAIR_ONLY = ["csls_q", "z_q", "rank_q"]


def f(x, nd=3):
    return "-" if pd.isna(x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    R = W / a.tag / "results"
    ret = pd.read_csv(R / "retrieval.tsv", sep="\t")
    met = pd.read_csv(R / "pair_metrics.tsv", sep="\t")
    ci = pd.read_csv(R / "retrieval_diff_ci.tsv", sep="\t")
    tm = pd.read_csv(R / "retrieval_tm.tsv", sep="\t") if (R / "retrieval_tm.tsv").exists() else None
    L = ["# ws_metric report tables", ""]
    verdict = []
    for m in MODELS:
        r = ret[ret.model == m]
        if not len(r):
            continue
        ra, ro = r[r.search == "all"].set_index("arm"), r[r.search == "other"].set_index("arm")

        def pm(pop, sub, arm):
            x = met[(met["pop"] == pop) & (met.model == m) & (met.subset == sub) & (met.arm == arm)]
            return x.auroc.iloc[0] if len(x) else np.nan

        def tmv(arm, search):
            if tm is None:
                return np.nan
            x = tm[(tm.model == m) & (tm.arm == arm) & (tm.search == search)]
            return x.inv_tm05_frac.iloc[0] if len(x) else np.nan

        L += [f"## {m}", "",
              "Retrieval: exact top-60 per held-out query. vis = fraction sequence-visible (E <= 1e-3); "
              "qany = queries with >= 1 visible hit; strepinv = share of retrieved pairs to the Streptomyces "
              "reference genome that are invisible; inv TM = fraction TM>=0.5 among a hash sample of invisible "
              "retrieved pairs. AUROCs on held-out pairs; TM = sequence-invisible in-block pairs, label TM>=0.5.", "",
              "| arm | round | vis all | vis other | qany other | ident vis all | other-phylum share (all) | strepinv | SD recall all | inv TM all | inv TM other | TM AUROC in | TM AUROC other | popA in | popA other | xphy other |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in UNTRAINED + PAIR_ONLY + R1 + R2:
            rnd = "untrained" if arm in UNTRAINED + PAIR_ONLY else ("r1" if arm in R1 else "r2")
            g = lambda df, c: df.loc[arm, c] if arm in df.index else np.nan
            vals = [g(ra, "vis_frac"), g(ro, "vis_frac"), g(ro, "q_any_vis"), g(ra, "med_ident_vis"), g(ra, "share_3"),
                    g(ra, "strep_invis_share"), g(ra, "sd_recall"), tmv(arm, "all"), tmv(arm, "other"),
                    pm("popB", "in_phylum", arm), pm("popB", "other_phylum", arm),
                    pm("popA", "in_phylum", arm), pm("popA", "other_phylum", arm), pm("xphy", "other_phylum", arm)]
            if all(pd.isna(v) for v in vals):
                continue
            nd = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3]
            L.append(f"| {arm} | {rnd} | " + " | ".join(f(v, n) for v, n in zip(vals, nd)) + " |")
        L.append("")
        # predeclared verdict
        un_r = [x for x in UNTRAINED if x in ra.index]
        best = {}
        for key, df, col in (("vis_all", ra, "vis_frac"), ("vis_other", ro, "vis_frac")):
            bu = df.loc[un_r, col].idxmax()
            for rnd, pool in (("r1", R1), ("r1+r2", R1 + R2)):
                pool = [x for x in pool if x in df.index]
                if not pool:
                    continue
                bl = df.loc[pool, col].idxmax()
                best[(key, rnd)] = (bl, bu, df.loc[bl, col] - df.loc[bu, col])
        for sub in ("in_phylum", "other_phylum"):
            mm = met[(met["pop"] == "popB") & (met.model == m) & (met.subset == sub)].set_index("arm")
            if not len(mm):
                continue
            bu = mm.loc[[x for x in UNTRAINED + PAIR_ONLY if x in mm.index], "auroc"].idxmax()
            for rnd, pool in (("r1", R1), ("r1+r2", R1 + R2)):
                pool = [x for x in pool if x in mm.index]
                if not pool:
                    continue
                bl = mm.loc[pool, "auroc"].idxmax()
                best[("tm_" + sub, rnd)] = (bl, bu, mm.loc[bl, "auroc"] - mm.loc[bu, "auroc"])
        L += ["Predeclared comparison: best learned minus best untrained (each picked on that metric).", "",
              "| metric | rounds | best learned | best untrained | difference | margin | met |", "|---|---|---|---|---|---|---|"]
        for (key, rnd), (bl, bu, dlt) in best.items():
            margin = 0.05 if key.startswith("vis") else 0.03
            L.append(f"| {key} | {rnd} | {bl} | {bu} | {dlt:+.4f} | {margin} | {'yes' if dlt >= margin else 'no'} |")
            verdict.append(dict(model=m, metric=key, rounds=rnd, best_learned=bl, best_untrained=bu, diff=dlt, met=dlt >= margin))
        L.append("")
        for rnd in ("r1", "r1+r2"):
            g_ = lambda k: best.get((k, rnd), (None, None, -1))[2]
            genomes = g_("vis_all") >= 0.05 or g_("tm_in_phylum") >= 0.03
            phyla = g_("vis_other") >= 0.05 or g_("tm_other_phylum") >= 0.03
            L.append(f"- {rnd}: held-out genomes criterion {'met' if genomes else 'not met'}; held-out phyla criterion "
                     f"{'met' if phyla else 'not met'}; gain = {'YES' if genomes and phyla else 'NO'}.")
        L.append("")
    pd.DataFrame(verdict).to_csv(R / "verdict.tsv", sep="\t", index=False)
    (R / "REPORT_TABLES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
