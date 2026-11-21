#!/usr/bin/env python
"""Paired genome bootstrap (2,000 draws, seed 17; boundary/rules.paired) of stored per-instance score files against the
references, even or odd genomes; ribosomal counts from the matching locus tables. Single-threaded, seconds.

usage: compare.py --split even --a FILE[,FILE...] --refs NAME=FILE,...   (FILE = per-instance scores parquet; the locus
       table is looked up at the same name under loci/ when present)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as BR  # noqa: E402

CALLERS = BR.CALLERS


def load(f):
    x = pd.read_parquet(f)
    if "excess" not in x:
        x["excess"] = x.f1 - x.control_mean_f1
    return x.set_index("instance_id")


def loci_for(f):
    f = Path(f)
    cand = [f.parent.parent / "loci" / f.name]
    for c in cand:
        if c.exists():
            return pd.read_parquet(c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="even")
    ap.add_argument("--a", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    gcode = B.odd_even_code()
    runs = B.ribo_runs()
    refs = dict(x.split("=", 1) for x in a.refs.split(","))
    out = {}
    for fa in a.a.split(","):
        A = load(fa)
        g = A.genome.map(gcode).to_numpy()
        summ = B.summarize_per(A[g % 2 == (0 if a.split == "even" else 1)].reset_index())
        la = loci_for(fa)
        rib = B.ribo_summary(runs, B.ribo_score(runs, la.gid_s.to_numpy(), la.gid_e.to_numpy())) if la is not None else None
        item = dict(summed=summ["summed_DF_TXSS_excess"], f1={c: summ[c]["f1"] for c in CALLERS}, ribo=rib, vs={})
        for name, fr in refs.items():
            pr = BR.paired(A, load(fr), gcode, a.split)
            item["vs"][name] = dict(summed=pr["summed"], d_f1={c: pr[c]["d_f1"] for c in CALLERS})
        out[fa] = item
        print(Path(fa).name, a.split, "summed %.4f" % item["summed"], "F1", "/".join("%.3f" % item["f1"][c] for c in CALLERS),
              "ribo", (rib["recovered"], rib["resolved"], rib["parvarchaeaceae"], rib["parvarchaeaceae_resolved"]) if rib else None)
        for name, v in item["vs"].items():
            print("   vs %-22s %+.4f [%+.3f,%+.3f]" % ((name,) + tuple(v["summed"])),
                  "dF1", " / ".join("%+.3f [%+.3f,%+.3f]" % tuple(v["d_f1"][c]) for c in CALLERS))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
