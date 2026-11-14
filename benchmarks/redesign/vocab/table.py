#!/usr/bin/env python
"""Tabulate even-genome results of every scored output (absolute F1, control, excess, covered50, boundary precision,
ribosomal) from results/even_*.json and results/even2_*.json. usage: table.py > results/table_even.txt"""
import json
from pathlib import Path

VD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/vocab")
C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")

ref = json.load(open(VD / "results/refcheck.json"))
print("output".ljust(34), "reading", "summed", "dDH[95%]".ljust(22), "F1 DF/TXSS/CCT".ljust(18), "ctrl DF/TXSS/CCT".ljust(18),
      "cov50 DF/TXSS/CCT".ljust(18), "bprec DF/TXSS/CCT".ljust(18), "ribo rec/res parv", "loci genes", sep=" | ")
for name in ("DH", "R1"):
    s, r = ref[name]["summary_even"], ref[name]["ribo_all"]
    print(f"token {name}".ljust(34), "-".ljust(7), f"{s['summed_DF_TXSS_excess']:.4f}", "".ljust(22),
          "/".join(f"{s[c]['f1']:.3f}" for c in C).ljust(18), "/".join(f"{s[c]['control']:.3f}" for c in C).ljust(18),
          "/".join(f"{s[c]['covered50']:.3f}" for c in C).ljust(18),
          "/".join(f"{(s[c]['median_bprec_covered'] or 0):.3f}" for c in C).ljust(18),
          f"{r['recovered']}/{r['resolved']} {r['parvarchaeaceae']}/{r['parvarchaeaceae_resolved'].split('/')[0]}",
          ref[name]["loci"]["genes_covered"], sep=" | ")
for fn in sorted((VD / "results").glob("even*_*.json")):
    d = json.load(open(fn))
    for rd, it in d["readings"].items():
        s, r = it["summary_even"], it["ribo"]["all"]
        dd = it["vs_DH_even"]["summed"]
        print(fn.stem[:34].ljust(34), rd.ljust(7), f"{s['summed_DF_TXSS_excess']:.4f}",
              f"{dd[0]:+.3f} [{dd[1]:+.3f},{dd[2]:+.3f}]".ljust(22),
              "/".join(f"{s[c]['f1']:.3f}" for c in C).ljust(18), "/".join(f"{s[c]['control']:.3f}" for c in C).ljust(18),
              "/".join(f"{s[c]['covered50']:.3f}" for c in C).ljust(18),
              "/".join(f"{(s[c]['median_bprec_covered'] or 0):.3f}" for c in C).ljust(18),
              f"{r['recovered']}/{r['resolved']} {r['parvarchaeaceae']}/{r['parvarchaeaceae_resolved'].split('/')[0]}",
              it["loci"]["genes_covered"], sep=" | ")
