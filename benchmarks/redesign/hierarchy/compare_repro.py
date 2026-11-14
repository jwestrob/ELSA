#!/usr/bin/env python
"""Exact comparison of two engine result directories (v1.5 against recurrence_ml.py with one level).

usage: compare_repro.py <dir A> <dir B>; exit status 1 on any difference. Compares summary.json values (timing,
engine name, arm and the ML-only keys excluded), rules.tsv, significant_pairs.parquet (columns of A),
token_modules.parquet, boundary_support.npy and loci_all_genomes.parquet."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SKIP = {"seconds", "stats_seconds", "engine", "arm", "levels", "level_of_significant_pairs",
        "level_of_significant_instances"}


def main():
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    bad = []
    sa, sb = json.load(open(A / "summary.json")), json.load(open(B / "summary.json"))
    for k in sorted(set(sa) | set(sb)):
        if k in SKIP:
            continue
        if sa.get(k) != sb.get(k):
            bad.append(f"summary[{k}]: {sa.get(k)} != {sb.get(k)}")
    ra, rb = pd.read_csv(A / "rules.tsv", sep="\t"), pd.read_csv(B / "rules.tsv", sep="\t")
    if not ra.equals(rb[ra.columns]):
        bad.append("rules.tsv differs")
    for f in ("significant_pairs.parquet", "token_modules.parquet", "loci_all_genomes.parquet"):
        x, y = pd.read_parquet(A / f), pd.read_parquet(B / f)
        same = len(x) == len(y) and x.reset_index(drop=True).equals(y[x.columns].reset_index(drop=True))
        print(f"{f}: rows {len(x):,} vs {len(y):,} identical={same}")
        if not same:
            bad.append(f"{f} differs")
    ba, bb = np.load(A / "boundary_support.npy"), np.load(B / "boundary_support.npy")
    same = np.array_equal(ba, bb)
    print(f"boundary_support.npy: {len(ba):,} boundaries identical={same}")
    if not same:
        bad.append("boundary_support differs")
    for sub in ("score_min2",):
        fa, fb = A / sub / "instance_scores.parquet", B / sub / "instance_scores.parquet"
        if fa.exists() and fb.exists():
            x, y = pd.read_parquet(fa), pd.read_parquet(fb)
            same = x.equals(y)
            print(f"{sub}/instance_scores: identical={same}")
            if not same:
                bad.append(f"{sub} differs")
    print("summary keys compared:", len([k for k in set(sa) | set(sb) if k not in SKIP]))
    if bad:
        print("DIFFERENCES:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"IDENTICAL: {A.name} == {B.name}")


if __name__ == "__main__":
    main()
