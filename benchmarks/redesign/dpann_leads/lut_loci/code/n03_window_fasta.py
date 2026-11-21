#!/usr/bin/env python
"""lut_loci n03 (biotite login node): protein sequences of the lut window genes and of extra gid lists.

usage: n03_window_fasta.py OUT.faa GIDFILE [GIDFILE ...]   (GIDFILE: one gid per line, or a parquet with column gid)
Reads the package FASTA once (headers are gids) and writes the requested sequences with the gid as header.
"""
import sys

import pandas as pd

FA = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/testset/package_k8/proteins_gid.faa"
out, files = sys.argv[1], sys.argv[2:]
want = set()
for f in files:
    if f.endswith(".parquet"):
        want |= set(pd.read_parquet(f, columns=["gid"]).gid.astype(int).tolist())
    else:
        want |= {int(x) for x in open(f) if x.strip()}
n = 0
with open(FA) as fh, open(out, "w") as fo:
    keep = False
    for line in fh:
        if line[0] == ">":
            keep = int(line[1:].split()[0]) in want
            if keep:
                n += 1
        if keep:
            fo.write(line)
print("wrote", n, "of", len(want))
