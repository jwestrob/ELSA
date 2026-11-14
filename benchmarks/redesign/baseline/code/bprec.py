#!/usr/bin/env python
"""Boundary precision of the loci that cover gold instances (workstream ws_baseline).

For each scored gold instance (resolved, order-trusted, >= 2 members; the scorer's --min-members 2
set) and each locus set (distinct per-genome intervals):
  covering loci   = loci holding >= 50% of the instance's members
  P_contain       = members / span of the smallest locus that contains every member
  P_cover_median  = median over covering loci of members-in-locus / locus span
  P_cover_max     = the most precise covering locus
  P_best          = boundary precision at the scorer's best-F1 locus (instance_scores.parquet,
                    instances whose best locus has coverage >= 0.5)
Medians over instances, per caller, on all genomes and on even genome_code (test split).

usage: bprec.py --out DIR NAME=LOCI_TABLE[:SCORE_DIR] ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def per_instance(inst, loci):
    loci = loci.drop_duplicates(["contig", "start", "end"])
    by = {c: (g.start.to_numpy(np.int64), g.end.to_numpy(np.int64)) for c, g in loci.groupby("contig")}
    rows = []
    for c, gi in inst.groupby("contig"):
        s, e = by.get(c, (np.empty(0, np.int64), np.empty(0, np.int64)))
        span = e - s + 1
        for iid, mem in zip(gi.instance_id.to_numpy(), gi.member_positions.to_numpy()):
            P = np.unique(np.asarray(mem, dtype=np.int64))
            k = len(P)
            if len(s) == 0:
                rows.append((iid, 0, 0, np.nan, np.nan, np.nan, np.nan))
                continue
            ov = np.searchsorted(P, e, side="right") - np.searchsorted(P, s, side="left")
            contain = (s <= P[0]) & (e >= P[-1])
            cov = ov >= 0.5 * k
            pc = k / span[contain].min() if contain.any() else np.nan
            prec = ov[cov] / span[cov]
            rows.append((iid, int(contain.sum()), int(cov.sum()), pc,
                         float(np.median(prec)) if len(prec) else np.nan,
                         float(prec.max()) if len(prec) else np.nan,
                         float(np.median(span[cov])) if len(prec) else np.nan))
    return pd.DataFrame(rows, columns=["instance_id", "n_containing", "n_covering", "p_contain", "p_cover_median",
                                       "p_cover_max", "covering_span_median"])


def summarize(d):
    out = dict(instances=int(len(d)))
    for c in ("p_contain", "p_cover_median", "p_cover_max", "p_best", "covering_span_median"):
        if c in d:
            v = d[c].dropna()
            out[c] = dict(n=int(len(v)), median=float(v.median()) if len(v) else None,
                          q25=float(v.quantile(.25)) if len(v) else None, q75=float(v.quantile(.75)) if len(v) else None)
    out["median_covering_loci"] = float(d.n_covering.median())
    out["frac_with_containing"] = float((d.n_containing > 0).mean())
    out["frac_with_covering"] = float((d.n_covering > 0).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("sets", nargs="+")
    a = ap.parse_args()
    inst = pd.read_parquet(PKG / "gold/instances.parquet")
    inst = inst[inst.resolved.fillna(False) & inst.order_trusted.fillna(False) & (inst.n_members_resolved >= 2)]
    gen = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    inst = inst.merge(gen, on="genome")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for spec in a.sets:
        name, rest = spec.split("=", 1)
        loci_path, _, score_dir = rest.partition(":")
        t0 = time.time()
        l = pd.read_parquet(loci_path, columns=["genome", "contig", "start", "end"])
        d = inst[["instance_id", "caller", "genome", "genome_code", "contig", "member_positions"]].merge(
            per_instance(inst, l), on="instance_id")
        if score_dir:
            sc = pd.read_parquet(Path(score_dir) / "instance_scores.parquet",
                                 columns=["instance_id", "coverage", "boundary_precision"])
            sc["p_best"] = np.where(sc.coverage >= 0.5, sc.boundary_precision, np.nan)
            d = d.merge(sc[["instance_id", "p_best"]], on="instance_id", how="left")
        d.drop(columns=["member_positions"]).to_parquet(out / f"bprec__{name}.parquet", index=False)
        rep = {}
        for split, m in (("all", np.ones(len(d), bool)), ("even", (d.genome_code % 2 == 0).to_numpy())):
            rep[split] = {c: summarize(d[m & (d.caller == c).to_numpy()]) for c in sorted(d.caller.unique())}
        report[name] = rep
        log(name, f"{time.time() - t0:.1f}s", json.dumps(rep["all"]))
    (out / "bprec_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["set\tsplit\tcaller\tinstances\tP_contain(n, median)\tP_cover_median(n, median)\tP_cover_max(median)\t"
             "P_best(n, median)\tcovering_loci_median\tcovering_span_median"]
    for name, rep in report.items():
        for split, cs in rep.items():
            for c, v in cs.items():
                pb = v.get("p_best", {"n": 0, "median": None})
                lines.append(f"{name}\t{split}\t{c}\t{v['instances']}\t{v['p_contain']['n']}, {v['p_contain']['median']}\t"
                             f"{v['p_cover_median']['n']}, {v['p_cover_median']['median']}\t{v['p_cover_max']['median']}\t"
                             f"{pb['n']}, {pb['median']}\t{v['median_covering_loci']}\t{v['covering_span_median']['median']}")
    (out / "bprec_summary.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
