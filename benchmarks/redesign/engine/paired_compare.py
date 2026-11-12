#!/usr/bin/env python
"""Paired comparison of two engine runs on source207: per-region observed F1 from summary.json.

usage: paired_compare.py <results dir A> <results dir B> [more B ...]
Prints, per selection, mean F1 of A and B, the mean paired difference A - B with a 95% bootstrap
interval over regions (10,000 resamples, seed 17) and the win / tie / loss counts.
When both runs carry per-region random-placement F1 (engine >= v1.2), the paired difference of
the excess is reported as well.
"""
import json
import sys
from pathlib import Path

import numpy as np

SEL = ("conserved loci, gaps 0-2", "conserved (gaps 0-2) + profile units", "all locus kinds")


def load(d):
    s = json.load(open(Path(d) / "summary.json"))
    return s, {x["selection"]: x for x in s["source207"] if x.get("n_intervals")}


def main():
    a_dir, b_dirs = sys.argv[1], sys.argv[2:]
    sa, A = load(a_dir)
    rng = np.random.default_rng(17)
    for b_dir in b_dirs:
        sb, B = load(b_dir)
        print(f"A = {sa['arm']} w{sa['w']}   B = {sb['arm']} w{sb['w']}")
        for sel in SEL:
            if sel not in A or sel not in B:
                continue
            ids = sorted(A[sel]["per_region_f1"])
            x = np.array([A[sel]["per_region_f1"][i] for i in ids])
            y = np.array([B[sel]["per_region_f1"].get(i, 0.0) for i in ids])
            d = x - y
            boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
            line = (f"  {sel:<38} A {x.mean():.4f}  B {y.mean():.4f}  A-B {d.mean():+.4f} "
                    f"[{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}]  "
                    f"win/tie/loss {int((d > 1e-9).sum())}/{int((abs(d) <= 1e-9).sum())}/{int((d < -1e-9).sum())}")
            if "per_region_random_f1" in A[sel] and "per_region_random_f1" in B[sel]:
                ex = d - np.array([A[sel]["per_region_random_f1"][i] - B[sel]["per_region_random_f1"][i] for i in ids])
                bx = rng.choice(ex, size=(10000, len(ex)), replace=True).mean(1)
                line += f"  excess A-B {ex.mean():+.4f} [{np.quantile(bx, .025):+.4f}, {np.quantile(bx, .975):+.4f}]"
            print(line)


if __name__ == "__main__":
    main()
