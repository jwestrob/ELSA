#!/usr/bin/env python3
"""Biohub item ledger for ws_kinds (budget 900 items). Counts every task recorded in work/folds*/state.json.
usage: fold_ledger.py [--check N]   exits 1 if adding N items would exceed the budget."""
import json
import sys
from pathlib import Path

BUDGET = 900
W = Path(__file__).resolve().parent / "work"


def used():
    n, per = 0, {}
    for d in sorted(W.glob("folds*")):
        p = d / "state.json"
        if p.exists():
            k = len(json.load(open(p))["tasks"])
            per[d.name] = k
            n += k
    return n, per


if __name__ == "__main__":
    n, per = used()
    add = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--check" else 0
    print(json.dumps(dict(used=n, per_dir=per, adding=add, budget=BUDGET, remaining_after=BUDGET - n - add)))
    sys.exit(1 if n + add > BUDGET else 0)
