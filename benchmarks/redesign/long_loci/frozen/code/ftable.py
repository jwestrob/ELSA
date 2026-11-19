#!/usr/bin/env python
"""Text tables for REPORT_frozen.txt from results/score_lut.json and the per-step time files (no scoring here)."""
import json
import re
import sys
from pathlib import Path

FZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
R = json.load(open(FZ / "results/score_lut.json"))
O = R["outputs"]
ORDER = ["PRIMARY_WA_L4b_t0.2", "WA_L1", "SECONDARY_Z_L4bp_t0.02", "Z_L1", "engine_DH", "synton_faithful", "synton_matched"]
ORDER = [x for x in ORDER if x in O]


def f(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def ci(v):
    return "[-, -]" if not v or v[0] is None else f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def lut_row(lab, s):
    return (f"{lab:24s} {s['n']:4d}  {f(s['f1'])} {ci(s['f1_ci']).replace('+', '')}  {f(s['control'])}  {s['excess']:+.3f} {ci(s['excess_ci'])}"
            f"  cov {f(s['coverage'])} prec {f(s['precision'])} c50 {f(s['covered50'])} none {s['no_overlap']}")


out = []
for split in ("all", "even"):
    out.append(f"LUT LOCI (lut_homology_main), {split} genomes: n, F1 [95%], control F1, excess [95%], mean coverage / precision of the best locus, covered50, loci with no overlapping locus")
    for k in ORDER:
        out.append(lut_row(k, O[k]["lut_main"][split]))
    out.append("")
out.append("RIBOSOMAL RUNS (g2, >= 16 marked): recovered / resolved (standing) / resolved, any fitting locus counts | even half (431 runs)")
for k in ORDER:
    a, e = O[k]["ribosomal"]["all"], O[k]["ribosomal"]["even"]
    out.append(f"{k:24s} {a['recovered']:4d} / {a['resolved']:4d} / {a['resolved_any_fit']:4d} of {a['runs']}  median precision (recovered) {f(a['median_precision_recovered'])}"
               f"  | even {e['recovered']} / {e['resolved']} / {e['resolved_any_fit']}")
out.append("")
out.append("COMPLETE CLUSTERS (80, layer a1): held by one locus >= 80% coverage; precision of that locus median [q10, q90], >= .8; transporter module (A B C E F G) matched by a locus (cov >= .8, prec >= .8); matched by a locus nested in the holding locus")
for k in ORDER:
    c = O[k]["complete"]
    q = c["held_precision_q10_q90"] or [None, None]
    out.append(f"{k:24s} held {c['held']:2d}/80  precision {f(c['held_precision_median'])} [{f(q[0])}, {f(q[1])}], >= .8: {c['held_precision_ge_0_8']:2d}  module matched {c['module_matched']:2d}  nested {c['module_matched_nested']:2d}")
out.append("")
out.append("BY ARCHITECTURE CLASS (main, all genomes): F1 / excess (n)")
classes = sorted({c for k in ORDER for c in O[k]["lut_main"]["by_class"]})
for c in classes:
    row = [f"{c[:38]:38s}"]
    for k in ORDER:
        s = O[k]["lut_main"]["by_class"].get(c)
        row.append(f"{s['f1']:.3f}/{s['excess']:+.3f}" if s else "-")
    n = O[ORDER[0]]["lut_main"]["by_class"][c]["n"]
    out.append(" ".join(row) + f" ({n})")
out.append("columns: " + ", ".join(ORDER))
out.append("")
out.append("BY GENOME LAYER (main, all genomes): F1 / excess (n)")
layers = sorted({c for k in ORDER for c in O[k]["lut_main"]["by_layer"]})
for c in layers:
    row = [f"{c[:26]:26s}"]
    for k in ORDER:
        s = O[k]["lut_main"]["by_layer"].get(c)
        row.append(f"{s['f1']:.3f}/{s['excess']:+.3f}" if s else "-")
    n = O[ORDER[0]]["lut_main"]["by_layer"][c]["n"]
    out.append(" ".join(row) + f" ({n})")
out.append("")
out.append("STRICT INSTANCES (121), all genomes: F1 / control / excess [95%]")
for k in ORDER:
    s = O[k]["lut_strict"]["all"]
    out.append(f"{k:24s} {f(s['f1'])} / {f(s['control'])} / {s['excess']:+.3f} {ci(s['excess_ci'])}")
out.append("")
out.append("PAIRED (bootstrap over truth loci / runs, 10,000 draws): d F1 [95%], d excess [95%] verdict, better/worse loci; d resolved runs [95%]")
for k, v in R["pairs"].items():
    a, e = v["lut_main_all"], v["lut_main_even"]
    out.append(f"{k:58s} dF1 {a['f1'][0]:+.3f} [{a['f1'][1]:+.3f}, {a['f1'][2]:+.3f}]  dExc {a['excess'][0]:+.3f} [{a['excess'][1]:+.3f}, {a['excess'][2]:+.3f}] {a['verdict']}"
               f"  {a['better_worse_f1'][0]}/{a['better_worse_f1'][1]}  dRes {v['ribo_resolved_all'][0]:+d} [{v['ribo_resolved_all'][1]:+.0f}, {v['ribo_resolved_all'][2]:+.0f}]"
               f" | even dExc {e['excess'][0]:+.3f} [{e['excess'][1]:+.3f}, {e['excess'][2]:+.3f}] dRes {v['ribo_resolved_even'][0]:+d}")
out.append("")
out.append("FALSE-POSITIVE SIDE (b1 + b2 genomes, no lut locus): isolated RB homologs in any locus / in a locus >= 3 genes; background b1/b2 genes; loci holding RB homologs of >= 2 / >= 3 components")
for k in ORDER:
    p = O[k]["false_positive_side"]
    out.append(f"{k:24s} isolated (n={p['isolated_homologs']}) {f(p['isolated_in_any_locus'])} / {f(p['isolated_in_locus_ge3'])}  smallest span {f(p['isolated_smallest_locus_span_median'], 1)}"
               f"  background {f(p['background_b1b2_in_any_locus'])} / {f(p['background_b1b2_in_locus_ge3'])}  loci >=2 comp {p['loci_b1b2_ge2_components']}, >=3 comp {p['loci_b1b2_ge3_components']}")
out.append("")
out.append("COVERAGE: loci, share of genes in loci, span q50 / q90 / q99 / max")
for k in ORDER:
    c = O[k]["coverage"]
    out.append(f"{k:24s} {c['loci']:9,d}  {c['genes_in_loci_frac']:.3f}  {c['span_q50_q90_q99_max']}")
out.append("")
out.append("COSTS (per step: wall clock, peak RSS GB)")
for p in sorted((FZ / "logs").glob("time_*.txt")):
    t = p.read_text()
    w = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    out.append(f"  {p.stem[5:]:22s} {w.group(1) if w else '-':>10s}  {int(m.group(1)) / 1024 / 1024:6.1f}" if m else f"  {p.stem}")
print("\n".join(out))
