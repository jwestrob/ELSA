#!/usr/bin/env python
"""Gather every small result of the baseline workstream into one text digest (stdout).

usage: collect.py [--arms raw zca] [--tag full]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
B = RD / "baseline"


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["raw", "zca"])
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    for arm in a.arms:
        run = B / "runs" / arm
        print(f"=== RUN {arm}")
        c = j(run / "chain_summary.json")
        if c:
            print(json.dumps({k: c[k] for k in ("anchors", "blocks", "anchors_in_blocks", "blocks_by_size", "seconds",
                                                "maxrss_gb", "threads")}))
        for s in ("identity", "17", "18", "19"):
            p = j(run / f"perm_{s}_summary.json")
            if p:
                print("perm", s, json.dumps({k: p.get(k) for k in ("blocks", "blocks_by_size", "seconds", "maxrss_gb",
                                                                   "replay")}))
        cl = j(run / "cluster_summary.json")
        if cl:
            print("cluster", json.dumps(cl))
        cal = j(run / "calibration.json")
        if cal:
            print("calibration", json.dumps({k: cal[k] for k in ("observed", "permuted", "ratio_all", "accepted",
                                                                 "accepted_ratio")}))
            print(pd.DataFrame(cal["by_size"]).to_string(index=False))
    st = j(B / "results/locus_stats.json") or {}
    print("=== LOCUS STATS")
    for k, v in st.items():
        print(k, json.dumps(v))
    print("=== RIBOSOMAL")
    for d in sorted((B / "results").glob("*/ribosomal_check.txt")):
        print(d.read_text().strip())
    print("=== BOUNDARY PRECISION")
    p = B / f"results/bprec_{a.tag}/bprec_summary.tsv"
    if p.exists():
        print(p.read_text())
    print("=== TABLES")
    for f in ("summed.tsv", "per_set.tsv", "paired.tsv"):
        p = B / f"results/tables_{a.tag}/{f}"
        if p.exists():
            print(f"--- {f}")
            print(pd.read_csv(p, sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()
