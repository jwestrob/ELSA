#!/usr/bin/env python
"""Text tables for REPORT_frozen2.txt from results/score_srvp.json (no scoring here)."""
import json
import sys
from pathlib import Path

FZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
R = json.load(open(FZ / "results/score_srvp.json"))
O = R["outputs"]
ORDER = [k for k in ("PRIMARY_Z_L4bp_t0.02", "Z_L1", "SECONDARY_WA_L4b_t0.2", "WA_L1", "engine_DH", "synton_faithful",
                     "synton_matched") if k in O]


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def ci(v):
    return "[-, -]" if not v or v[0] is None else f"[{v[0]:+.3f}, {v[1]:+.3f}]"


out = []
for sub in ("all", "even", "Bacteria", "Archaea", "multi_contig", "single_contig"):
    out.append(f"CURATED, {sub}: DefenseFinder F1 / control / excess [95%] (n) | TXSScan as called F1 / excess | TXSScan span <= 30 F1 / excess | summed DF+TXSS excess (span30; without G03c08f50ba11988e)")
    for k in ORDER:
        c = O[k]["curated"][sub]
        d, t, t3 = c["DefenseFinder"], c["TXSScan"], c["TXSScan_span30"]
        if not d.get("n"):
            continue
        out.append(f"{k:24s} DF {f(d['f1'])} / {f(d['control'])} / {d['excess']:+.3f} {ci(d['excess_ci'])} ({d['n']}) | TX {f(t.get('f1'))} / {t.get('excess', 0):+.3f} ({t.get('n')}) | "
                   f"TX30 {f(t3.get('f1'))} / {t3.get('excess', 0):+.3f} ({t3.get('n')}) | summed {c['summed']:+.3f} ({c['summed_span30']:+.3f}; {c['summed_noG03']:+.3f})")
    out.append("")
for sub in ("all", "even", "Bacteria", "Archaea", "multi_contig", "single_contig"):
    out.append(f"RIBOSOMAL, {sub}: recovered / resolved / any-fit / accurate strict / loose of runs")
    for k in ORDER:
        r = O[k]["ribosomal"][sub]
        out.append(f"{k:24s} {r['recovered']:4d} / {r['resolved']:4d} / {r['any_fit']:4d} / {r.get('accurate_strict', '-')} / {r.get('accurate_loose', '-')} of {r['runs']}  median precision {f(r['median_precision_recovered'])}")
    out.append("")
out.append("PAIRED (genome bootstrap 10,000): d F1 / d excess per caller [95%]; summed (as called / span30 / no G03) with verdict; d resolved runs [95%]")
for k, v in R["pairs"].items():
    s1, s2, s3 = v["summed"], v["summed_span30"], v["summed_noG03"]
    df, tx = v["DefenseFinder"], v["TXSScan"]
    out.append(f"{k:58s} DF dF1 {df['f1'][0]:+.3f} [{df['f1'][1]:+.3f}, {df['f1'][2]:+.3f}] dExc {df['excess'][0]:+.3f} [{df['excess'][1]:+.3f}, {df['excess'][2]:+.3f}] | "
               f"TX dF1 {tx['f1'][0]:+.3f} dExc {tx['excess'][0]:+.3f} [{tx['excess'][1]:+.3f}, {tx['excess'][2]:+.3f}] | summed {s1['d'][0]:+.3f} [{s1['d'][1]:+.3f}, {s1['d'][2]:+.3f}] {s1['verdict']} / "
               f"{s2['d'][0]:+.3f} {s2['verdict']} / {s3['d'][0]:+.3f} {s3['verdict']} | dRes {v['ribo_resolved'][0]:+d} [{v['ribo_resolved'][1]:+.0f}, {v['ribo_resolved'][2]:+.0f}]"
               + (f" | d accurate strict {v['accurate_strict']:+d}" if "accurate_strict" in v else ""))
out.append("")
out.append("COVERAGE: loci, share of genes in loci (all / Bacteria / Archaea / multi-contig / single-contig), span q50 / q90 / q99 / max")
for k in ORDER:
    c = O[k]["coverage"]
    out.append(f"{k:24s} {c['loci']:9,d}  {c['genes_in_loci_frac']:.3f} / {c['Bacteria']:.3f} / {c['Archaea']:.3f} / {c['multi_contig']:.3f} / {c['single_contig']:.3f}  {c['span_q50_q90_q99_max']}")
print("\n".join(out))
