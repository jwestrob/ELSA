#!/usr/bin/env python
"""Paired genome-bootstrap intervals for AMI differences between partitions (workstream ws_locusspace).

Bootstrap AMI levels are biased upward by duplicated items (a resampled genome contributes identical copies), so
levels are reported as point estimates and intervals are given for paired differences (same draws for both
partitions), where the duplication bias largely cancels.
  lspace_post.py gold   -> out/gold/ami_paired.json   (part-1 partitions against MOD and J, even instances)
  lspace_post.py map    -> out/map/ami_paired_MEAN.json (locus-space clusters against engine module labels)
Single process, one thread.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import homogeneity_completeness_v_measure as hcv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lspace import OUT, dump, log  # noqa: E402

DRAWS = 2000
SEED = 17


def paired(lab, parts, gcode, idx, pairs, draws=DRAWS):
    """Bootstrap (with replacement; duplicated items bias AMI upward) and half-sampling (50% of genomes without
    replacement; interval = point + centred 2.5 / 97.5% quantiles of the half-sample estimates, which for a mean
    reproduces the full-sample variance) for levels and paired differences."""
    out = _paired(lab, parts, gcode, idx, pairs, draws, half=False)
    out["half_sampling"] = _paired(lab, parts, gcode, idx, pairs, draws, half=True)
    return out


def _paired(lab, parts, gcode, idx, pairs, draws, half):
    rng = np.random.default_rng(SEED)
    g = gcode[idx]
    codes, inv = np.unique(g, return_inverse=True)
    members = [idx[inv == k] for k in range(len(codes))]
    point = {}
    for k, cl in parts.items():
        h, c, _ = hcv(lab[idx], cl[idx])
        point[k] = dict(ami=float(ami(lab[idx], cl[idx])), homogeneity=float(h), completeness=float(c))
    diffs = {f"{x}-{y}": np.empty((draws, 3)) for x, y in pairs}
    levels = {k: np.empty((draws, 3)) for k in {z for p in pairs for z in p}}
    for d in range(draws):
        if half:
            pick = rng.choice(len(codes), len(codes) // 2, replace=False)
        else:
            pick = rng.integers(0, len(codes), len(codes))
        sel = np.concatenate([members[k] for k in pick])
        a = lab[sel]
        vals = {}
        for k in {z for p in pairs for z in p}:
            h, c, _ = hcv(a, parts[k][sel])
            vals[k] = (ami(a, parts[k][sel]), h, c)
            levels[k][d] = vals[k]
        for x, y in pairs:
            diffs[f"{x}-{y}"][d] = np.subtract(vals[x], vals[y])
    M = ("ami", "homogeneity", "completeness")

    def iv(pt, D, j):
        if half:
            c = D[:, j] - D[:, j].mean()
            return [pt, float(pt + np.quantile(c, .025)), float(pt + np.quantile(c, .975))]
        return [pt, float(np.quantile(D[:, j], .025)), float(np.quantile(D[:, j], .975))]
    out = dict(point=point, paired={}, levels={})
    for key, D in diffs.items():
        x, y = key.split("-", 1)
        out["paired"][key] = {m: iv(point[x][m] - point[y][m], D, j) for j, m in enumerate(M)}
    for k, D in levels.items():
        out["levels"][k] = {m: iv(point[k][m], D, j) for j, m in enumerate(M)}
    return out


def gold():
    I = pd.read_parquet(OUT / "gold/gold_instances.parquet")
    P = pd.read_parquet(OUT / "gold/gold_clusters.parquet")
    lab = pd.factorize(I.key)[0]
    gcode = I.genome_code.to_numpy()
    even = np.flatnonzero(I.split.eq("even").to_numpy())
    parts = {c: P[c].to_numpy() for c in P.columns}
    reps = [c for c in P.columns if c not in ("MOD", "J")]
    pairs = [(r, "MOD") for r in reps] + [(r, "J") for r in reps] + [("J", "MOD")]
    res = paired(lab, parts, gcode, even, pairs)
    # per caller (DefenseFinder / CRISPRCasTyper even instances)
    for caller in ("DefenseFinder", "CRISPRCasTyper"):
        idx = np.flatnonzero((I.split.eq("even") & I.caller.eq(caller)).to_numpy())
        res[caller] = paired(lab, parts, gcode, idx, [(r, "MOD") for r in reps] + [(r, "J") for r in reps], draws=500)
    dump(res, OUT / "gold/ami_paired.json")
    log("gold paired AMI done")


def mapp(rep="MEAN"):
    C = pd.read_parquet(OUT / f"map/cores_clusters_{rep}.parquet", columns=["cluster", "module_id"])
    B = pd.read_parquet(OUT / f"map/gold_cores_{rep}.parquet")
    lab = pd.factorize(B.key)[0]
    hil = pd.factorize(B.hikey)[0]
    rows = B.core_row.to_numpy()
    parts = {"locus_space": C.cluster.to_numpy()[rows], "modules": C.module_id.to_numpy()[rows]}
    gcode = B.genome_code.to_numpy()
    even = np.flatnonzero(B.split.eq("even").to_numpy())
    res = dict(label=paired(lab, parts, gcode, even, [("locus_space", "modules")]),
               higher=paired(hil, parts, gcode, even, [("locus_space", "modules")]))
    dump(res, OUT / f"map/ami_paired_{rep}.json")
    log("map paired AMI done")


if __name__ == "__main__":
    {"gold": gold, "map": mapp}[sys.argv[1]](*sys.argv[2:])
