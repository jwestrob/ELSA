#!/usr/bin/env python
"""ws_showcase2: do the two replicates agree in arrangement? For every interface passing both replicates:
(1) interface correspondence through sequence: replicate chains aligned globally (BLOSUM62, -10/-0.5); the share of
replicate-1 interface residues (contacts CB-CB <= 8 A with the partner chain) whose aligned replicate-2 residue is
also an interface residue of the same pair, and the reverse; the mean of the two is reported (expected under a
random placement of the same number of interface residues: the product of interface fractions, also reported);
(2) US-align of the two replicate complexes (-mm 1 -ter 0; TM-scores normalised by each complex) from the post job
(post/<dir>/us/*.txt), when present.
Usage: python sc2_agree.py FOLD_DIR [...] [--us US_DIR]; writes analysis/agreement.tsv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

HERE = Path(__file__).resolve().parent.parent
A = HERE / "analysis"
W = HERE / "work"
AB = "ABCDEFGH"
al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-10, extend_gap_score=-0.5)


def amap(a, b):
    aln = al.align(a, b)[0]
    m = {}
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        for k in range(e1 - s1):
            m[s1 + k] = s2 + k
    return m


def us_scores(us_dir):
    out = {}
    if us_dir is None or not Path(us_dir).exists():
        return out
    for p in Path(us_dir).glob("*.txt"):
        for line in open(p):
            if line.startswith("#") or not line.strip():
                continue
            x = line.split("\t")
            if len(x) > 4:
                try:
                    out[p.stem] = (float(x[2]), float(x[3]), float(x[4]) if len(x) > 4 else np.nan)
                except ValueError:
                    pass
    return out


def main(args):
    us_dir = None
    if "--us" in args:
        us_dir = args[args.index("--us") + 1]
        args = [a for a in args if a not in ("--us", us_dir)]
    met = {}
    for d in map(Path, args):
        for line in open(d / "metrics.jsonl"):
            m = json.loads(line)
            met[m["fold_id"]] = m
    seqs = {}
    for p in W.glob("jobs*.jsonl"):
        for line in open(p):
            j = json.loads(line)
            seqs[j["fold_id"]] = [c["seq"] for c in j["chains"]]
    us = us_scores(us_dir)
    calls = pd.read_csv(A / "calls.tsv", sep="\t")
    rows = []
    for _, r in calls[calls.pass_both].iterrows():
        rank, a, b = int(r["rank"]), int(r.i), int(r.j)
        f1, f2 = f"sc2c_{rank:03d}_r1", f"sc2c_{rank:03d}_r2"
        m1, m2 = met[f1], met[f2]
        p1, p2 = m1["pairs"][f"{AB[a]}{AB[b]}"], m2["pairs"][f"{AB[a]}{AB[b]}"]
        out = dict(rank=rank, i=a, j=b)
        sh, ex = [], []
        for lab, ch, r1, r2 in (("i", a, p1["res_i"], p2["res_i"]), ("j", b, p1["res_j"], p2["res_j"])):
            s1, s2 = seqs[f1][ch], seqs[f2][ch]
            mp = amap(s1, s2)
            rv = {v: k for k, v in mp.items()}
            set2, set1 = set(r2), set(r1)
            f12 = np.mean([mp.get(x) in set2 for x in r1]) if r1 else np.nan
            f21 = np.mean([rv.get(x) in set1 for x in r2]) if r2 else np.nan
            out[f"iface_corr_{lab}"] = round(float(np.nanmean([f12, f21])), 3)
            out[f"iface_expected_{lab}"] = round(len(r2) / len(s2), 3)
            sh.append(out[f"iface_corr_{lab}"])
            ex.append(out[f"iface_expected_{lab}"])
        out["iface_corr_mean"] = round(float(np.mean(sh)), 3)
        out["iface_expected_mean"] = round(float(np.mean(ex)), 3)
        key = f"{f1}__{f2}"
        if key in us:
            out["usalign_tm_r1"], out["usalign_tm_r2"], out["usalign_rmsd"] = us[key][0], us[key][1], us[key][2]
        rows.append(out)
    z = pd.DataFrame(rows)
    z.to_csv(A / "agreement.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(z.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
