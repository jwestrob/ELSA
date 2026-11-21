#!/usr/bin/env python
"""Paired comparison of engine runs on the DPANN gold loci, with a fixed genome split.

usage: dpann_compare.py [--score score_min2] <results dir A> <results dir B> [more B ...]
Reads <dir>/<score>/instance_scores.parquet of each run. Per caller and per split it prints the
mean best-locus F1 and excess (F1 minus the mean of 200 control placements) of A and B and the
paired differences A - B with 95% intervals from a genome-level bootstrap (genomes resampled,
2,000 draws, seed 17). Splits: all genomes; tune = odd genome_code; test = even genome_code.
The last block sums DefenseFinder and TXSScan (the predeclared DPANN criterion).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")


def load(d, score):
    t = pd.read_parquet(Path(d) / score / "instance_scores.parquet")
    t["excess"] = t.f1 - t.control_mean_f1
    return t.set_index("instance_id")


def boot_ci(values, groups, rng, draws=2000):
    """Mean of `values` with a 95% interval, resampling whole groups (genomes)."""
    codes, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(codes))
    n = np.bincount(inv, minlength=len(codes)).astype(float)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return values.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="score_min2")
    ap.add_argument("a")
    ap.add_argument("b", nargs="+")
    a = ap.parse_args()
    gen = pd.read_parquet(RD / "ws_dpann/package/universe/genomes.parquet", columns=["genome", "genome_code"])
    code = dict(zip(gen.genome, gen.genome_code))
    A = load(a.a, a.score)
    rng = np.random.default_rng(17)
    for bdir in a.b:
        B = load(bdir, a.score)
        ids = A.index.intersection(B.index)
        x, y = A.loc[ids], B.loc[ids]
        gcode = x.genome.map(code).to_numpy()
        print(f"A = {Path(a.a).name}\nB = {Path(bdir).name}   ({len(ids):,} shared instances, {a.score})")
        for split, m in (("all", np.ones(len(ids), bool)), ("tune (odd)", gcode % 2 == 1), ("test (even)", gcode % 2 == 0)):
            tot = {}
            for caller in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
                k = m & (x.caller.to_numpy() == caller)
                g = gcode[k]
                df1 = boot_ci((x.f1.to_numpy() - y.f1.to_numpy())[k], g, rng)
                dex = boot_ci((x.excess.to_numpy() - y.excess.to_numpy())[k], g, rng)
                tot[caller] = (x.excess.to_numpy()[k].mean(), y.excess.to_numpy()[k].mean(), (x.excess.to_numpy() - y.excess.to_numpy())[k], g)
                print(f"  {split:<11} {caller:<15} n={int(k.sum()):>5,}  F1 A {x.f1.to_numpy()[k].mean():.3f} B {y.f1.to_numpy()[k].mean():.3f} "
                      f"A-B {df1[0]:+.3f} [{df1[1]:+.3f}, {df1[2]:+.3f}]   excess A {tot[caller][0]:+.3f} B {tot[caller][1]:+.3f} "
                      f"A-B {dex[0]:+.3f} [{dex[1]:+.3f}, {dex[2]:+.3f}]")
            # predeclared criterion: sum of the DefenseFinder and TXSScan mean excess; bootstrap over genomes of both
            d1, g1 = tot["DefenseFinder"][2], tot["DefenseFinder"][3]
            d2, g2 = tot["TXSScan"][2], tot["TXSScan"][3]
            allg = np.unique(np.concatenate([g1, g2]))
            idx = {c: i for i, c in enumerate(allg)}
            s1 = np.bincount([idx[c] for c in g1], weights=d1, minlength=len(allg)); n1 = np.bincount([idx[c] for c in g1], minlength=len(allg))
            s2 = np.bincount([idx[c] for c in g2], weights=d2, minlength=len(allg)); n2 = np.bincount([idx[c] for c in g2], minlength=len(allg))
            pick = rng.integers(0, len(allg), size=(2000, len(allg)))
            bs = s1[pick].sum(1) / np.maximum(n1[pick].sum(1), 1) + s2[pick].sum(1) / np.maximum(n2[pick].sum(1), 1)
            print(f"  {split:<11} DefenseFinder+TXSScan summed excess: A {tot['DefenseFinder'][0] + tot['TXSScan'][0]:+.3f} "
                  f"B {tot['DefenseFinder'][1] + tot['TXSScan'][1]:+.3f}  A-B {d1.mean() + d2.mean():+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")


if __name__ == "__main__":
    main()
