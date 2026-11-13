#!/usr/bin/env python
"""Part-5 odd selection among Z-WA L1 / L4b_t* / L4bp_t*: max [conditions, odd resolved, odd summed]. usage: sel5.py RESULTS"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
d = json.load(open(R / "odd_dpann_ZWA.json"))["outputs"]
best = max(d, key=lambda k: tuple(d[k]["score"]))
for k, v in sorted(d.items(), key=lambda kv: tuple(kv[1]["score"]), reverse=True):
    s, r = v["summary_odd"], v["ribo_odd"]
    print(f"{k:12s} {v['score'][0]} {v['score'][1]:4d} {v['score'][2]:.4f}  " + "/".join(f"{s[c]['f1']:.3f}" for c in C)
          + f"  rec {r['recovered']} parv {r['parvarchaeaceae']} genes {v['loci']['genes_covered']}")
print("SELECTED", best, d[best]["score"])
(R / "selection5.json").write_text(json.dumps(dict(selected=best, score=d[best]["score"],
                                                   candidates={k: v["score"] for k, v in d.items()}), indent=1) + "\n")
