#!/usr/bin/env python
"""Print the REPORT tables from the result JSONs copied into baseline_synton/results/ (Mac-side, small files)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def f(x, n=3):
    return "-" if x is None else (f"{x:+.{n}f}" if isinstance(x, float) and x < 0 else f"{x:.{n}f}")


def ci(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


def spacedust():
    g = json.load(open(R / "spacedust_gate.json"))
    rows = [("engine reference (stored, best of 42 readings)", g["engine"]), ("native Spacedust (all blocks)", g["native"])]
    for fn in sorted(R.glob("spacedust_final*.json")) or sorted(R.glob("spacedust_sd_*.json")):
        d = json.load(open(fn))
        for k, v in d.items():
            if isinstance(v, dict) and "f1_observed" in v:
                rows.append((k, v))
    print("SPACEDUST source207 (207 regions; excess = F1 - mean of 200 translated placements; 95% region bootstrap)")
    print(f"  {'output':48s} {'intervals':>9s} {'F1':>6s} {'random':>6s} {'excess [95%]':>24s} {'loc':>5s} "
          f"{'bprec*':>6s} {'cov*':>5s} {'span*':>5s} {'overlap':>7s} {'qcov':>5s}")
    for name, v in rows:
        print(f"  {name:48s} {v['n_intervals']:9d} {v['f1_observed']:.3f} {v['f1_random']:.3f} "
              f"{v['f1_excess']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] {v['localized_observed']:.3f} "
              f"{f(v.get('median_boundary_precision_overlapping', v.get('median_boundary_precision')), 2):>6s} "
              f"{f(v.get('median_coverage_overlapping', v.get('median_coverage')), 2):>5s} "
              f"{f(v.get('median_best_span_overlapping'), 0):>5s} {v.get('regions_with_any_overlap', 0):7d} "
              f"{v['query_gene_coverage']:.3f}")
    print("  (* medians over regions with an overlapping interval for synton rows; loc = localized fraction;"
          " qcov = share of query genes inside intervals)")
    for name, v in rows[2:]:
        if "by_size_class" in v:
            print(f"  {name}: F1 / control by region size " + "; ".join(
                f"{c} ({s['regions']}) {s['f1']:.3f} / {s['control']:.3f}" for c, s in v["by_size_class"].items()))
    print("  paired differences A - B (10,000 region bootstrap, seed 17):")
    for name, v in rows[2:]:
        for p in v.get("paired", []):
            print(f"    {p['a']:28s} - {p['b']:32s} dF1 {ci(p['d_f1'])}  dExcess {ci(p['d_excess'])}  "
                  f"regions A>B {p['regions_a_better_f1']} / B>A {p['regions_b_better_f1']}")


def dpann():
    fn = sorted(R.glob("dpann_final*.json")) or sorted(R.glob("dpann_dp_*.json"))
    ref = json.load(open(R / "dpann_refcheck.json"))
    print("DPANN (even genome_code; ribosomal check over all 1,686 runs)")
    print(f"  {'output':34s} {'summed':>7s} {'F1 DF':>6s} {'TXSS':>6s} {'CCT':>6s} {'ctrl DF/TXSS/CCT':>17s} "
          f"{'ribo rec':>8s} {'res':>5s} {'Parv rec':>8s} {'Parv res':>8s}")
    rows = [("token DH (reference)", ref["DH"]["card"]), ("word method V-coh0.1", ref["word"]["card"])]
    outs = {}
    for p in fn:
        d = json.load(open(p))
        for k, v in d.get("outputs", {}).items():
            rows.append((k, v["card"]))
            outs[k] = v
    for name, c in rows:
        print(f"  {name:34s} {c['summed']:.4f} {c['f1']['DefenseFinder']:.3f} {c['f1']['TXSScan']:.3f} "
              f"{c['f1']['CRISPRCasTyper']:.3f} {c['control']['DefenseFinder']:.3f}/{c['control']['TXSScan']:.3f}/"
              f"{c['control']['CRISPRCasTyper']:.3f} {c['ribo_recovered']:8d} {c['ribo_resolved']:5d} "
              f"{str(c['parvarchaeaceae']):>8s} {str(c['parvarchaeaceae_resolved']):>8s}")
    for k, v in outs.items():
        for ref_name, key in (("DH", "vs_DH_even"), ("word", "vs_word_even")):
            pr = v[key]
            print(f"  {k} - {ref_name}: summed {ci(pr['summed'])}; dF1 " + ", ".join(
                f"{c[:4]} {ci(pr[c]['d_f1'])}" for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper")))
        print(f"  {k}: loci {v['loci']}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "spacedust"):
        spacedust()
    if which in ("all", "dpann"):
        dpann()
