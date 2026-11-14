#!/usr/bin/env python
"""Paired comparisons between stored per-instance score files (even genomes; rules.paired, 2,000 draws, seed 17).
usage: collect.py A_prefix_reading B_prefix_reading [...pairs]   e.g. main_tau10_V-mob genome_tau8_V-mob
Writes results/paired_<A>__<B>.json and prints summed and per-caller F1 differences."""
import json
import sys
from pathlib import Path

import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as RU  # noqa: E402


def main():
    args = sys.argv[1:]
    gcode = B.odd_even_code()
    for A, Bn in zip(args[::2], args[1::2]):
        a = pd.read_parquet(VD / "scores" / f"{A}.parquet").set_index("instance_id")
        b = pd.read_parquet(VD / "scores" / f"{Bn}.parquet").set_index("instance_id")
        pr = RU.paired(a, b, gcode, "even")
        with open(VD / "results" / f"paired_{A}__{Bn}.json", "w") as fh:
            json.dump(pr, fh, indent=1)
        print(A, "-", Bn, "summed", [round(x, 4) for x in pr["summed"]],
              {c[:4]: [round(x, 3) for x in pr[c]["d_f1"]] for c in RU.CALLERS})


if __name__ == "__main__":
    main()
