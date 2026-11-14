#!/usr/bin/env python
"""Paired region-bootstrap differences between any two scored outputs (per-region tables results/pr_<name>.parquet).
usage: pairs.py OUT.json A~B [A~B ...]     names as in pr_<name>.parquet, e.g. P2__CC2~P2__V-coh0.1"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s207  # noqa: E402

R = s207.TS / "results"
out = {}
for spec in sys.argv[2:]:
    a, b = spec.split("~")
    A, B = pd.read_parquet(R / f"pr_{a}.parquet"), pd.read_parquet(R / f"pr_{b}.parquet")
    p = s207.paired(A, B)
    p["verdict_excess"] = s207.verdict(p)
    p["A_f1_excess"] = [float(A.f1.mean()), float(A.excess.mean())]
    p["B_f1_excess"] = [float(B.f1.mean()), float(B.excess.mean())]
    out[f"{a} - {b}"] = p
    print(f"{a} - {b}: dF1 {p['f1'][0]:+.3f} [{p['f1'][1]:+.3f}, {p['f1'][2]:+.3f}]  d excess {p['excess'][0]:+.3f} "
          f"[{p['excess'][1]:+.3f}, {p['excess'][2]:+.3f}]  {p['verdict_excess']}  w/t/l {p['win_tie_loss_f1']}")
old = json.load(open(R / sys.argv[1])) if (R / sys.argv[1]).exists() else {}
old.update(out)
(R / sys.argv[1]).write_text(json.dumps(old, indent=1) + "\n")
