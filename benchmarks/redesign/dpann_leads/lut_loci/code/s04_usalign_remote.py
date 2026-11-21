#!/usr/bin/env python
"""lut_loci s04 (biotite login node, single-threaded serial loop): US-align of apo models against experimental structures
(first chain of each crystal). TM1 = normalized by the model, TM2 = by the crystal. With MATRIX=1 for a pair, the crystal
is superposed onto the model (crystal = structure 1) and the rotation matrix is saved to mats/<model>__<crystal>.txt.
usage: s04_usalign_remote.py PAIRS.tsv OUT.tsv      (PAIRS columns: model, crystal, matrix)
"""
import os
import subprocess
import sys

import pandas as pd

US = "/shared/software/bin/USalign"
P = pd.read_csv(sys.argv[1], sep="\t")
os.makedirs("mats", exist_ok=True)
rows = []
for r in P.itertuples(index=False):
    a, b = f"models/{r.model}.pdb", f"crystals/{r.crystal}.pdb"
    res = subprocess.run([US, a, b, "-outfmt", "2"], capture_output=True, text=True)
    lines = [x for x in res.stdout.splitlines() if x.strip() and not x.startswith("#") and not x.startswith("Warning")]
    rec = dict(model=r.model, crystal=r.crystal)
    if res.returncode == 0 and lines:
        f = lines[-1].split("\t")
        rec.update(TM1=float(f[2]), TM2=float(f[3]), RMSD=float(f[4]), IDali=float(f[7]), L1=int(f[8]), L2=int(f[9]),
                   Lali=int(f[10]))
    if int(r.matrix):
        subprocess.run([US, b, a, "-m", f"mats/{r.model}__{r.crystal}.txt"], capture_output=True, text=True)
    rows.append(rec)
pd.DataFrame(rows).to_csv(sys.argv[2], sep="\t", index=False)
print(len(rows), "pairs")
