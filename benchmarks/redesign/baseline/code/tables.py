#!/usr/bin/env python
"""Comparison tables: legacy variants against the recurrence engine on the DPANN gold loci.

Per score pass (score_min2, score_min2_span30), per caller, per split (all genomes; even genome_code =
test): mean best-locus F1, mean random-placement control F1, excess = F1 - control, each with a 95%
genome-bootstrap interval (2,000 draws, seed 17, genomes resampled as in dpann_compare.py); paired
differences legacy - engine on shared instances; summed DefenseFinder + TXSScan excess.

usage: tables.py --out DIR --refs REF ... --sets NAME=RESULT_DIR ... (every non-reference set is paired with every reference)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def load(d, score):
    t = pd.read_parquet(Path(d) / score / "instance_scores.parquet")
    t["excess"] = t.f1 - t.control_mean_f1
    return t.set_index("instance_id")


def boot(values, groups, draws=2000, seed=17):
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(codes))
    n = np.bincount(inv, minlength=len(codes)).astype(float)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return float(values.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def boot_sum(d1, g1, d2, g2, draws=2000, seed=17):
    rng = np.random.default_rng(seed)
    allg = np.unique(np.concatenate([g1, g2]))
    i1, i2 = np.searchsorted(allg, g1), np.searchsorted(allg, g2)
    s1 = np.bincount(i1, weights=d1, minlength=len(allg)); n1 = np.bincount(i1, minlength=len(allg))
    s2 = np.bincount(i2, weights=d2, minlength=len(allg)); n2 = np.bincount(i2, minlength=len(allg))
    pick = rng.integers(0, len(allg), size=(draws, len(allg)))
    bs = s1[pick].sum(1) / np.maximum(n1[pick].sum(1), 1) + s2[pick].sum(1) / np.maximum(n2[pick].sum(1), 1)
    return float(d1.mean() + d2.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", nargs="+", required=True, help="names (among the sets) of the engine references")
    ap.add_argument("--sets", nargs="+", required=True)
    a = ap.parse_args()
    REFS = a.refs
    sets = dict(s.split("=", 1) for s in a.sets)
    gen = pd.read_parquet(RD / "ws_dpann/package/universe/genomes.parquet", columns=["genome", "genome_code"])
    code = dict(zip(gen.genome, gen.genome_code))
    rows, prows, srows = [], [], []
    for score in ("score_min2", "score_min2_span30"):
        T = {n: load(d, score) for n, d in sets.items()}
        for name, t in T.items():
            g = t.genome.map(code).to_numpy()
            for split, m in (("all", np.ones(len(t), bool)), ("even", g % 2 == 0)):
                ex = {}
                for c in CALLERS:
                    k = m & (t.caller.to_numpy() == c)
                    f1 = boot(t.f1.to_numpy()[k], g[k])
                    ctl = boot(t.control_mean_f1.to_numpy()[k], g[k])
                    e = boot(t.excess.to_numpy()[k], g[k])
                    ex[c] = (t.excess.to_numpy()[k], g[k])
                    rows.append(dict(score=score, set=name, split=split, caller=c, n=int(k.sum()), f1=round(f1[0], 4),
                                     f1_ci=f"[{f1[1]:.3f}, {f1[2]:.3f}]", control=round(ctl[0], 4),
                                     excess=round(e[0], 4), excess_ci=f"[{e[1]:+.3f}, {e[2]:+.3f}]"))
                s = boot_sum(*ex["DefenseFinder"], *ex["TXSScan"])
                srows.append(dict(score=score, set=name, split=split, summed_excess=round(s[0], 4),
                                  ci=f"[{s[1]:+.3f}, {s[2]:+.3f}]"))
        for name in T:
            if name in REFS:
                continue
            for ref in REFS:
                if ref not in T:
                    continue
                x, y = T[name], T[ref]
                ids = x.index.intersection(y.index)
                x, y = x.loc[ids], y.loc[ids]
                g = x.genome.map(code).to_numpy()
                for split, m in (("all", np.ones(len(ids), bool)), ("even", g % 2 == 0)):
                    dd = {}
                    for c in CALLERS:
                        k = m & (x.caller.to_numpy() == c)
                        df1 = boot((x.f1.to_numpy() - y.f1.to_numpy())[k], g[k])
                        dex = boot((x.excess.to_numpy() - y.excess.to_numpy())[k], g[k])
                        dd[c] = ((x.excess.to_numpy() - y.excess.to_numpy())[k], g[k])
                        prows.append(dict(score=score, a=name, b=ref, split=split, caller=c, n=int(k.sum()),
                                          d_f1=fmt(df1), d_excess=fmt(dex)))
                    s = boot_sum(*dd["DefenseFinder"], *dd["TXSScan"])
                    prows.append(dict(score=score, a=name, b=ref, split=split, caller="DF+TXSS summed excess",
                                      n=int((m & np.isin(x.caller.to_numpy(), ["DefenseFinder", "TXSScan"])).sum()),
                                      d_f1="", d_excess=fmt(s)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R, P, S = pd.DataFrame(rows), pd.DataFrame(prows), pd.DataFrame(srows)
    R.to_csv(out / "per_set.tsv", sep="\t", index=False)
    P.to_csv(out / "paired.tsv", sep="\t", index=False)
    S.to_csv(out / "summed.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.max_columns", 20):
        print(R.to_string(index=False))
        print(S.to_string(index=False))
        print(P.to_string(index=False))


if __name__ == "__main__":
    main()
