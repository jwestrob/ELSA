#!/usr/bin/env python
"""Render results/ablation_table.tsv and per-model summaries into results/ABLATION.md."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, W

LABEL = {"context60": "V0 sel", "seeds": "V1 sel", "ev_p50": "V2-p50 sel", "ev_p80": "V2-p80 sel",
         "ev_p95": "V2-p95 sel"}
CHAIN = {"lf": "length-first", "sf0": "V3 score-first, no gap cost", "sfc": "V3 score-first, concave gap"}
STRATA = ["0_genus_Streptomyces", "1_same_class_other_genus", "2_same_phylum_other_class", "3_other_phylum",
          "9_no_gtdb"]


def short(v):
    sel, ch = v.split("__")
    base = {"context60": "V0", "seeds": "V1", "ev_p50": "V2p50", "ev_p80": "V2p80", "ev_p95": "V2p95"}[sel]
    return base + {"lf": "", "sf0": "+V3g0", "sfc": "+V3gc"}[ch]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    t = pd.read_csv(W / "results/ablation_table.tsv", sep="\t")
    t["id"] = t.variant.map(short)
    L = ["# ws_chain ablation tables", "",
         "Variant ids: V0 = published selection + length-first; V1 = seeds only; V2pXX = evidence-gated selection "
         "at P >= 0.XX; +V3g0 / +V3gc = score-first chaining without / with concave gap cost. Rules: raw = all "
         "blocks; cos5 = per-size mean-cosine cut at 5% (permuted_panels); q5 = V4 summed log-odds q-value <= 0.05; "
         "T4 / T0 = built-in chain cut. ratio = permuted/observed for the accepted set. vis = sequence-visible "
         "fraction of accepted anchor pairs (E <= 1e-3). dEx = F1 excess minus calibrated V0's, paired 95% "
         "region bootstrap.", ""]
    for m in MODELS:
        s = t[t.model == m].copy()
        if s.empty:
            continue
        summ = json.load(open(W / "results" / m / "summary.json"))
        runs = json.load(open(W / "runs" / m / "run_summary.json"))
        L += [f"## {m}", "", "### Raw output and chance ratio by size", ""]
        rows = []
        for v, e in summ["variants"].items():
            r = dict(id=short(v), observed=f"{e['observed']:,}", permuted_mean=f"{e['permuted_mean']:,.0f}",
                     ratio=f"{e['ratio']:.3f}")
            for k, bs in e["by_size"].items():
                r[f"r{k}{'+' if k == '6' else ''}"] = f"{bs['ratio']:.3f}"
            ob = runs["runs"][v]["observed"]
            r["sel_s"] = f"{ob['selection_seconds']:.0f}"
            r["chain_s"] = f"{ob['chain_seconds']:.0f}"
            rows.append(r)
        L += [md(pd.DataFrame(rows)), "", "### Accepted sets", ""]
        cols = dict(id="id", rule="rule", blocks="blocks", anchor_pairs="pairs", achieved_ratio="ratio",
                    vis_frac="vis", vis_frac_heldout="vis_held", f1="F1", f1_random="F1_rand", f1_excess="excess",
                    d_excess="dEx", gain="gain")
        x = s[[c for c in cols]].rename(columns=cols).copy()
        x["blocks"] = x.blocks.map("{:,}".format)
        x["pairs"] = x.pairs.map("{:,}".format)
        for c in ("ratio", "vis", "vis_held", "F1", "F1_rand"):
            x[c] = x[c].map("{:.3f}".format)
        x["excess"] = [f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" for e, lo, hi in
                       zip(s.f1_excess, s.f1_excess_lo, s.f1_excess_hi)]
        x["dEx"] = [f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" for e, lo, hi in zip(s.d_excess, s.d_excess_lo, s.d_excess_hi)]
        L += [md(x), "", "### Accepted blocks by target stratum (cos5 and q5)", ""]
        st = s[s.rule.isin(["cos5", "q5", "T4"])]
        rows = []
        for _, r in st.iterrows():
            d = dict(id=r.id, rule=r.rule)
            for k in STRATA:
                b = r.get(f"blocks_{k}")
                vv = r.get(f"vis_{k}")
                d[k.split("_", 1)[1]] = "-" if pd.isna(b) else f"{int(b):,} ({vv:.2f})"
            rows.append(d)
        L += [md(pd.DataFrame(rows)), "", "(cell = accepted blocks (sequence-visible pair fraction))", ""]
        curves = []
        for v, e in summ["variants"].items():
            if "T_curve" in e:
                ok = [c for c in e["T_curve"] if c["ratio"] <= 0.05]
                t4 = next(c for c in e["T_curve"] if c["T"] == 4.0)
                curves.append(dict(id=short(v), T_min_ratio05=(ok[0]["T"] if ok else "-"),
                                   pairs_at_T=(f"{ok[0]['pairs']:,}" if ok else "-"),
                                   ratio_T4=f"{t4['ratio']:.3f}", pairs_T4=f"{t4['pairs']:,}"))
        if curves:
            L += ["### Score-first: smallest chain cut T with raw ratio <= 0.05", "", md(pd.DataFrame(curves)), ""]
    (W / "results/ABLATION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
