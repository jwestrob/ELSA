#!/usr/bin/env python
"""Part-2 odd selection per arm (predeclared2.json): among L1 / L4b_t* / L4bp_t* maximise [odd conditions met, odd
resolved, odd summed]. usage: sel2.py RESULTS_DIR"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
SRC = {"F": ["odd_dpann_group.json", "odd_dpann_group_L4b.json", "odd_dpann_group_L4bp.json"],
       "Z": ["odd_dpann_Z.json"], "A": ["odd_dpann_A.json"], "Zs": ["odd_dpann_Zs.json"], "As": ["odd_dpann_As.json"],
       "FZ": ["odd_dpann_FZ.json"]}
out = {}
for arm, fns in SRC.items():
    cand = {}
    for fn in fns:
        if not (R / fn).exists():
            continue
        for k, v in json.load(open(R / fn))["outputs"].items():
            if k == "L1" or k.startswith("L4b_t") or k.startswith("L4bp_t"):
                cand[k] = v
    if not cand:
        continue
    best = max(cand, key=lambda k: tuple(cand[k]["score"]))
    out[arm] = dict(selected=best, score=cand[best]["score"],
                    candidates={k: v["score"] for k, v in sorted(cand.items(), key=lambda kv: tuple(kv[1]["score"]), reverse=True)})
    print(arm, "->", best, cand[best]["score"])
    for k, v in out[arm]["candidates"].items():
        print("   ", k, [v[0], v[1], round(v[2], 4)])
(R / "selection2.json").write_text(json.dumps(out, indent=1) + "\n")
