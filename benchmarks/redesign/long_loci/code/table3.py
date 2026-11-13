#!/usr/bin/env python
"""Part-3 source207 table. usage: table3.py RESULTS_DIR"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
REF = ("p1L4b", "p1L1", "CC2", "engine_ref", "spacedust", "P2_Vcoh0.1")


def f(x, n=3, s=False):
    if x is None:
        return "-"
    t = f"{x:+.{n}f}" if s else f"{x:.{n}f}"
    return t.replace("0.", ".", 1)


def iv(v):
    return f"{f(v[0], 3, True)} [{f(v[1], 3, True)},{f(v[2], 3, True)}]"


for arm in ("Z", "Zs"):
    fn = R / f"s207_sd_{arm}.json"
    if not fn.exists():
        continue
    d = json.load(open(fn))
    print(f"SOURCE207 sd_{arm}")
    print(f"{'output':14s} {'F1':>5s} {'ctrl':>5s} {'excess [95%]':>22s} " + " ".join(f"{'d ex vs ' + r:>24s}" for r in REF)
          + "  union/best (>=11)  qgenes")
    for nm, o in d["outputs"].items():
        s, fr = o["score"], o["fragmentation"]
        print(f"{nm:14s} {f(s['f1_observed']):>5s} {f(s['f1_random']):>5s} {iv([s['f1_excess'], s['ci_lo'], s['ci_hi']]):>22s} "
              + " ".join(f"{iv(o['vs_' + r]['paired']['excess']):>24s}" for r in REF)
              + f"  {f(fr['union_cov_q50'], 2)}({f(fr['union_cov_q50_ge11'], 2)})/{f(fr['best_cov_q50'], 2)}({f(fr['best_cov_q50_ge11'], 2)})"
              f"  {f(o['query_genes_in_loci_frac'])}")
        print(" " * 15 + "d F1: " + " ".join(f"{r} {iv(o['vs_' + r]['paired']['f1'])}" for r in REF)
              + " | verdicts " + " ".join(f"{r}:{o['vs_' + r]['verdict']}" for r in REF))
        print(" " * 15 + "by size excess " + " ".join(f"{k} {f(v['excess'], 3, True)}" for k, v in s["by_size"].items())
              + f"; bprec {f(s['median_boundary_precision'], 2)} cov {f(s['median_coverage'], 2)} loci {o['loci']}")
