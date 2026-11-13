#!/usr/bin/env python
"""Part-4 odd selection among W3 / W3b / W5 / W8 / WA x {L1, L4b_t*}: max [conditions, odd resolved, odd summed].
usage: sel4.py RESULTS_DIR"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
SRC = {"W3": ["odd_dpann_group.json", "odd_dpann_group_L4b.json"], "W3b": ["odd_dpann_W3b.json"],
       "W5": ["odd_dpann_W5.json"], "W8": ["odd_dpann_W8.json"], "WA": ["odd_dpann_WA.json"]}
C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
cand = {}
for arm, fns in SRC.items():
    for fn in fns:
        if not (R / fn).exists():
            continue
        for k, v in json.load(open(R / fn))["outputs"].items():
            if k == "L1" or k.startswith("L4b_t"):
                cand[f"{arm}:{k}"] = v
best = max(cand, key=lambda k: tuple(cand[k]["score"]))
print(f"{'candidate':16s} cond res  summed  F1 DF/TXSS/CCT   rec  parv  genes")
for k, v in sorted(cand.items(), key=lambda kv: tuple(kv[1]["score"]), reverse=True):
    s, r = v["summary_odd"], v["ribo_odd"]
    print(f"{k:16s} {v['score'][0]}    {v['score'][1]:4d} {v['score'][2]:.4f}  "
          + "/".join(f"{s[c]['f1']:.3f}" for c in C) + f"  {r['recovered']}  {r['parvarchaeaceae']}  {v['loci']['genes_covered']}")
print("SELECTED", best, cand[best]["score"])
(R / "selection4.json").write_text(json.dumps(dict(selected=best, score=cand[best]["score"],
                                                   candidates={k: v["score"] for k, v in cand.items()}), indent=1) + "\n")
