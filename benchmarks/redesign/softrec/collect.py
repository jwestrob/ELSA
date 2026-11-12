#!/usr/bin/env python
"""Collect softrec result directories into one table (tune split by default; --reveal adds the test split).

usage: collect.py [--reveal] [--out results/engine_results.tsv] [--analyses ARM ...]
Columns per result: source, variant, statistic, rule, instance FDR, significant instances, genes covered,
loci per kind, median span, ribosomal recovered / Parvarchaeaceae / admissible, and from dpann_compare
(A = softrec, B = base or default): F1 and excess per caller, summed DF + TXSS excess difference with its
95% genome-bootstrap interval.
--analyses: for the named result dirs, even-genome complementarity against base and default (instances with
token F1 < 0.5 and softrec F1 >= 0.5, and the reverse), median boundary precision at best locus, and gold
member genes touching a supported boundary.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
RES = RD / "engine/results_dpann_softrec"
PKG = RD / "ws_dpann/package"
BASE = RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3"
DEF = RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"
NUM = r"([+-]?\d+\.\d+)"
LINE = re.compile(rf"^\s+(all|tune \(odd\)|test \(even\))\s+(DefenseFinder|TXSScan|CRISPRCasTyper)\s+n=\s*([\d,]+)\s+F1 A {NUM} B {NUM} A-B {NUM} "
                  rf"\[{NUM}, {NUM}\]\s+excess A {NUM} B {NUM} A-B {NUM} \[{NUM}, {NUM}\]")
SUM = re.compile(rf"^\s+(all|tune \(odd\)|test \(even\))\s+DefenseFinder\+TXSScan summed excess: A {NUM} B {NUM}\s+A-B {NUM} \[{NUM}, {NUM}\]")
SPLIT = {"all": "all", "tune (odd)": "tune", "test (even)": "test"}
CALL = {"DefenseFinder": "DF", "TXSScan": "TXSS", "CRISPRCasTyper": "CCT"}


def parse_compare(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = LINE.match(line)
        if m:
            s, c = SPLIT[m.group(1)], CALL[m.group(2)]
            v = [float(x) for x in m.groups()[3:]]
            out[(s, c)] = dict(n=int(m.group(3).replace(",", "")), f1A=v[0], f1B=v[1], df1=v[2], df1_lo=v[3], df1_hi=v[4],
                               exA=v[5], exB=v[6], dex=v[7], dex_lo=v[8], dex_hi=v[9])
            continue
        m = SUM.match(line)
        if m:
            s = SPLIT[m.group(1)]
            v = [float(x) for x in m.groups()[1:]]
            out[(s, "SUM")] = dict(A=v[0], B=v[1], d=v[2], lo=v[3], hi=v[4])
    return out


def row_for(d, splits):
    r = dict(result=d.name)
    s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    for k in ("source", "variant", "statistic", "rule", "fdr", "significant_instances", "genes_covered", "genes_covered_frac"):
        r[k] = s.get(k)
    lk = s.get("loci_all_genomes", {})
    r["loci_conserved"] = lk.get("conserved")
    r["loci_unit"] = lk.get("unit")
    r["span_med_conserved"] = s.get("locus_span_median", {}).get("conserved")
    rb = d / "ribosomal_check.json"
    if rb.exists():
        p = json.loads(rb.read_text())["primary"]
        r["ribo_recovered"] = p["recovered"]
        r["ribo_resolved"] = p["resolved"]
        r["parv"] = p["parvarchaeaceae"]
        parv = int(str(p["parvarchaeaceae"]).split("/")[0])
        r["admissible"] = bool(p["recovered"] >= 1650 and parv >= 110)
    for tag, f in (("base", "compare_vs_base.txt"), ("def", "compare_vs_default.txt")):
        c = parse_compare(d / f)
        for sp in splits:
            if (sp, "SUM") in c:
                x = c[(sp, "SUM")]
                r[f"{sp}_sum_{tag}"] = x["d"]
                r[f"{sp}_sum_{tag}_ci"] = f"[{x['lo']:+.3f}, {x['hi']:+.3f}]"
                if tag == "base":
                    r[f"{sp}_sumA"] = x["A"]
            for cl in ("DF", "TXSS", "CCT"):
                if (sp, cl) in c:
                    x = c[(sp, cl)]
                    if tag == "base":
                        r[f"{sp}_{cl}_F1"] = x["f1A"]
                        r[f"{sp}_{cl}_ex"] = x["exA"]
                    r[f"{sp}_{cl}_dF1_{tag}"] = x["df1"]
    sc = d / "score_min2/summary.json"
    if sc.exists():
        bc = json.loads(sc.read_text())["by_caller"]
        for cl, k in CALL.items():
            if cl in bc:
                r[f"all_{k}_bprec_med"] = bc[cl].get("median_boundary_precision_at_best")
                r[f"all_{k}_ctrlF1"] = bc[cl].get("control_mean_best_f1")
    return r


def analyses(d):
    gen = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    code = dict(zip(gen.genome, gen.genome_code))
    A = pd.read_parquet(d / "score_min2/instance_scores.parquet").set_index("instance_id")
    res = {}
    for tag, bdir in (("base", BASE), ("default", DEF)):
        B = pd.read_parquet(bdir / "score_min2/instance_scores.parquet").set_index("instance_id")
        ids = A.index.intersection(B.index)
        x, y = A.loc[ids], B.loc[ids]
        even = x.genome.map(code).to_numpy() % 2 == 0
        out = {}
        for cl in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
            m = even & (x.caller.to_numpy() == cl)
            fa, fb = x.f1.to_numpy()[m], y.f1.to_numpy()[m]
            out[cl] = dict(n=int(m.sum()), token_poor=int((fb < 0.5).sum()), token_poor_soft_good=int(((fb < 0.5) & (fa >= 0.5)).sum()),
                           soft_poor=int((fa < 0.5).sum()), soft_poor_token_good=int(((fa < 0.5) & (fb >= 0.5)).sum()),
                           median_bprec_soft=float(np.median(x.boundary_precision.to_numpy()[m])),
                           median_bprec_token=float(np.median(y.boundary_precision.to_numpy()[m])),
                           median_span_soft=float(np.median(x.best_span.to_numpy()[m])),
                           median_span_token=float(np.median(y.best_span.to_numpy()[m])))
        res[tag] = out
    # gold member genes touching a supported boundary (soft vs base vs default), even genomes
    mem = pd.read_parquet(PKG / "gold/members.parquet", columns=["instance_id", "caller", "gid"])
    gc = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code"]).genome_code.to_numpy()
    cov = {}
    for tag, dd in (("soft", d), ("base", BASE), ("default", DEF)):
        b = np.load(dd / "boundary_support.npy")
        touch = (b > 0) | (np.r_[0, b[:-1]] > 0)
        m = mem[gc[mem.gid.to_numpy()] % 2 == 0]
        cov[tag] = {cl: float(touch[g.gid.to_numpy()].mean()) for cl, g in m.groupby("caller")}
    res["gold_genes_touching_support_even"] = cov
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal", action="store_true", help="include the test (even) split")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyses", nargs="*", default=[])
    a = ap.parse_args()
    splits = ("tune", "test") if a.reveal else ("tune",)
    rows = [row_for(d, splits) for d in sorted(RES.iterdir()) if d.is_dir()]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
    cols = [c for c in ["result", "rule", "fdr", "significant_instances", "genes_covered", "ribo_recovered", "parv", "admissible",
                        "tune_sumA", "tune_sum_base", "tune_sum_base_ci", "tune_sum_def", "tune_DF_F1", "tune_TXSS_F1", "tune_CCT_F1",
                        "test_sumA", "test_sum_base", "test_sum_base_ci", "test_sum_def", "test_sum_def_ci", "test_DF_F1",
                        "test_TXSS_F1", "test_CCT_F1"] if c in df]
    print(df[cols].to_string(index=False))
    if a.out:
        df.to_csv(a.out, sep="\t", index=False)
    for name in a.analyses:
        print(name, json.dumps(analyses(RES / name), indent=1))


if __name__ == "__main__":
    main()
