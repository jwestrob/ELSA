#!/usr/bin/env python
"""softrec3 stage-2 selection (predeclared score: odd conditions met, then odd summed excess) over the DH reading of
every stage-2 configuration; prints the ranking and the selected configuration."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect3 as C  # noqa: E402

tags = sys.argv[1:]
rows = [r for r in C.odd_rows(tags) if r["reading"] == "DH"]
rows.sort(key=lambda r: r["score"], reverse=True)
for r in rows:
    print(C.fmt(r))
best = rows[0]
r1 = [r for r in C.odd_rows(tags) if r["reading"] == "R1" and r["key"] == best["key"]][0]
print("SELECTED", best["key"], "score", best["score"], "cond", best["cond"], "R1 cond", r1["cond"],
      "contingency (DH fails o1 while R1 passes o1):", (not best["cond"][0]) and bool(r1["cond"][0]))
