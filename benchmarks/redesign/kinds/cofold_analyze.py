#!/usr/bin/env python3
"""Co-fold test F (predeclared.json): per-fold criteria 1-3, class rates with exact intervals, Fisher tests,
length strata against the stored screen-1 rates, Mantel-Haenszel, matched-pair and swap-margin views.
Runs on the Mac (pandas only). Inputs: work/manifest.tsv, work/folds*/metrics.jsonl, ../cofold/folds.tsv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact

HERE = Path(__file__).resolve().parent
W = HERE / "work"
BINS = [0, 200, 300, 451]
LAB = ["<200", "200-300", "300-450"]


def cp(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def passes(m):
    return (m.iptm >= 0.75) & (m.iface_pae <= 5) & (m.contacts >= 10) & (m.interface_plddt >= 70)


def mh_or(tabs):
    num = sum(a * d / (a + b + c + d) for a, b, c, d in tabs if a + b + c + d)
    den = sum(b * c / (a + b + c + d) for a, b, c, d in tabs if a + b + c + d)
    return num / den if den else float("inf")


def main():
    man = pd.read_csv(W / "manifest.tsv", sep="\t")
    mets = []
    for d in sorted(W.glob("folds*")):
        p = d / "metrics.jsonl"
        if p.exists():
            mets += [json.loads(l) for l in open(p) if l.strip()]
    met = pd.DataFrame(mets).drop_duplicates("fold_id", keep="last")
    f = man.merge(met[["fold_id", "iptm", "ptm", "iface_pae", "contacts", "interface_plddt", "chain_plddt"]],
                  on="fold_id", how="left")
    f["folded"] = f.iptm.notna()
    f["pass_c123"] = passes(f) & f.folded
    f["lbin"] = pd.cut(f.len_sum, BINS, right=False, labels=LAB)
    f.to_csv(W / "folds_kinds.tsv", sep="\t", index=False)
    out = dict(items=len(f), folded=int(f.folded.sum()))
    rows = []
    for c in ("a_high", "b_low", "c_swap", "d_critic", "d_critic_swap"):
        g = f[(f.cls == c) & f.folded]
        k, n = int(g.pass_c123.sum()), len(g)
        lo, hi = cp(k, n) if n else (np.nan, np.nan)
        r = dict(cls=c, n=n, passes=k, rate=k / n if n else np.nan, lo=lo, hi=hi,
                 iptm_med=float(g.iptm.median()) if n else np.nan, iptm_q90=float(g.iptm.quantile(.9)) if n else np.nan,
                 iptm_ge75=int((g.iptm >= 0.75).sum()))
        for b in LAB:
            gb = g[g.lbin == b]
            r[f"bin_{b}"] = f"{int(gb.pass_c123.sum())}/{len(gb)}"
        rows.append(r)
    ct = pd.DataFrame(rows)
    # stored screen-1 rates by the same bins
    s1 = pd.read_csv(HERE.parent / "cofold/folds.tsv", sep="\t")
    s1 = s1[s1.cls.isin(["cognate", "uncalled", "positive", "noncognate"])].copy()
    L = s1.lengths.str.split(";", expand=True).astype(float)
    s1["len_sum"] = L[0] + L[1]
    s1["p"] = passes(s1)
    assert int(s1[s1.cls == "cognate"].p.sum()) == 27, "screen-1 reference not reproduced"
    s1["lbin"] = pd.cut(s1.len_sum, BINS, right=False, labels=LAB)
    a = f[(f.cls == "a_high") & f.folded]
    wa = a.lbin.value_counts(normalize=True).reindex(LAB).fillna(0)
    stored = []
    for c in ("cognate", "uncalled", "positive", "noncognate"):
        g = s1[(s1.cls == c) & (s1.len_sum < 451)]
        r = dict(cls=f"screen1_{c}", n=len(g), passes=int(g.p.sum()), rate=g.p.mean())
        std = 0.0
        for b in LAB:
            gb = g[g.lbin == b]
            r[f"bin_{b}"] = f"{int(gb.p.sum())}/{len(gb)}"
            std += wa[b] * (gb.p.mean() if len(gb) else np.nan)
        r["rate_std_to_class_a_lengths"] = std
        stored.append(r)
    for c in ("a_high", "b_low", "c_swap"):
        g = f[(f.cls == c) & f.folded]
        ct.loc[ct.cls == c, "rate_std_to_class_a_lengths"] = sum(
            wa[b] * (g[g.lbin == b].pass_c123.mean() if (g.lbin == b).any() else np.nan) for b in LAB)
    tab = pd.concat([ct, pd.DataFrame(stored)], ignore_index=True)
    tab.to_csv(W / "cofold_table.tsv", sep="\t", index=False)
    print(tab.round(4).to_string(index=False))
    # tests
    b = f[(f.cls == "b_low") & f.folded]
    c = f[(f.cls == "c_swap") & f.folded]
    ka, na, kb, nb, kc, nc = int(a.pass_c123.sum()), len(a), int(b.pass_c123.sum()), len(b), int(c.pass_c123.sum()), len(c)
    out["a_vs_b_fisher_one_sided"] = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="greater")
    out["a_vs_c_fisher_one_sided"] = fisher_exact([[ka, na - ka], [kc, nc - kc]], alternative="greater")
    out["ratio_a_over_b"] = (ka / na) / (kb / nb) if kb else float("inf")
    tabs = []
    for bn in LAB:
        x, y = a[a.lbin == bn], b[b.lbin == bn]
        tabs.append((int(x.pass_c123.sum()), len(x) - int(x.pass_c123.sum()), int(y.pass_c123.sum()),
                     len(y) - int(y.pass_c123.sum())))
    out["mantel_haenszel_or_a_vs_b"] = mh_or(tabs)
    out["mh_tables"] = tabs
    # matched pairs (a rank r vs b matched_to r)
    mb = b.set_index("rank").pass_c123
    ma = a.set_index("rank").pass_c123
    common = ma.index.intersection(mb.index)
    out["matched_pairs"] = dict(n=int(len(common)), a_only=int((ma[common] & ~mb[common]).sum()),
                                b_only=int((~ma[common] & mb[common]).sum()), both=int((ma[common] & mb[common]).sum()))
    # swap margin (criterion 4) for class-a pairs that have a swap
    sw = c.set_index("rank").iptm
    ai = a.set_index("rank").iptm
    cm = ai.index.intersection(sw.index)
    marg = ai[cm] - sw[cm]
    out["swap_margin"] = dict(n=int(len(cm)), a_pass_with_swap=int(a.set_index("rank").pass_c123[cm].sum()),
                              a_pass_and_margin_ge_0_3=int((a.set_index("rank").pass_c123[cm] & (marg >= 0.3)).sum()),
                              margin_median=float(marg.median()))
    lo_a, hi_a = cp(ka, na)
    lo_b, hi_b = cp(kb, nb)
    rc = kc / nc if nc else np.nan
    sup = (ka / na >= 3 * kb / max(nb, 1)) and hi_b < lo_a and out["a_vs_b_fisher_one_sided"][1] < 0.01 and rc <= 0.02
    refuted = (ka / na <= 1.5 * kb / max(nb, 1)) or out["a_vs_b_fisher_one_sided"][1] >= 0.05 or (rc >= 0.5 * ka / na)
    out["verdict"] = "supported" if sup else ("refuted" if refuted else "partly supported")
    json.dump(out, open(W / "cofold_results.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
