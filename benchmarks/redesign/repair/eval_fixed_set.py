#!/usr/bin/env python3
"""Gain retained (predeclared amendment_1.gain_retained): each arm scored on rescue3's touched-instance set.

usage: eval_fixed_set.py <repair dir> <engine results dir> <base results dir> <reference arm> <arm> [...]
Touched set = gold instances (score_min2) with >= 1 member gene changed in the reference arm. Paired genome
bootstrap (2,000 draws, seed 17) per split; ratio = arm mean F1 gain / reference arm mean F1 gain on the same set.
Writes results/fixed_set.tsv.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

R, ENG, BASE, REF = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
ARMS = [REF] + [a for a in sys.argv[5:] if a != REF]
P = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/package")
code = dict(pd.read_parquet(P / "universe/genomes.parquet", columns=["genome", "genome_code"]).values)
B = pd.read_parquet(BASE / "score_min2/instance_scores.parquet").set_index("instance_id")
B["excess"] = B.f1 - B.control_mean_f1
g = B.genome.map(code).to_numpy()
mem = pd.read_parquet(P / "gold/members.parquet", columns=["instance_id", "gid"])
ch = set(pd.read_parquet(R / f"results/changed_{REF}__dpann.parquet").gid.tolist())
touched = B.index.isin(mem.instance_id[mem.gid.isin(ch)].unique())

def boot(v, gg):
    codes, inv = np.unique(gg, return_inverse=True)
    pick = np.random.default_rng(17).integers(0, len(codes), size=(2000, len(codes)))
    s = np.bincount(inv, weights=v, minlength=len(codes)); n = np.bincount(inv, minlength=len(codes)).astype(float)
    m = s[pick].sum(1) / n[pick].sum(1)
    return v.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)

rows, ref = [], {}
for arm in ARMS:
    A = pd.read_parquet(ENG / f"dpann__{arm}__w3/score_min2/instance_scores.parquet").set_index("instance_id").loc[B.index]
    A["excess"] = A.f1 - A.control_mean_f1
    for split, sm in (("all", np.ones(len(B), bool)), ("odd", g % 2 == 1), ("even", g % 2 == 0)):
        k = touched & sm
        f = boot((A.f1 - B.f1).to_numpy()[k], g[k]); e = boot((A.excess - B.excess).to_numpy()[k], g[k])
        c = boot((A.control_mean_f1 - B.control_mean_f1).to_numpy()[k], g[k])
        if arm == REF: ref[split] = f[0]
        rows.append({"arm": arm, "split": split, "n": int(k.sum()), "f1_diff": f[0], "f1_lo": f[1], "f1_hi": f[2],
                     "ctrl_diff": c[0], "excess_diff": e[0], "excess_lo": e[1], "excess_hi": e[2], "ratio_to_ref": f[0] / ref[split]})
T = pd.DataFrame(rows)
T.to_csv(R / "results/fixed_set.tsv", sep="\t", index=False, float_format="%.4f")
print(T.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
