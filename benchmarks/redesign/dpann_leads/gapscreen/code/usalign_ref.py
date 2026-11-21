#!/usr/bin/env python
"""Rung 4 (biotite login node, single-threaded serial loop): US-align each predicted model (chain 1) onto the experimental
reference chain (chain 2). Records TM1 (normalized by the model), TM2 (by the reference), RMSD, Lali and the aligned
sequences with US-align's distance markers (':' = pair within 5 A).  usage: usalign_ref.py PAIRS.tsv OUT.tsv"""
import subprocess, sys
import pandas as pd
US = "/shared/software/bin/USalign"
P = pd.read_csv(sys.argv[1], sep="\t")
rows = []
for r in P.itertuples(index=False):
    out = subprocess.run([US, r.model, r.ref], capture_output=True, text=True).stdout.splitlines()
    rec = r._asdict()
    for l in out:
        if l.startswith("Aligned length="):
            f = [x.strip() for x in l.replace("Aligned length=", "").split(",")]
            rec["Lali"] = int(f[0]); rec["RMSD"] = float(f[1].split("=")[1])
        elif l.startswith("TM-score=") and "Structure_1" in l:
            rec["TM1"] = float(l.split()[1])
        elif l.startswith("TM-score=") and "Structure_2" in l:
            rec["TM2"] = float(l.split()[1])
    k = [i for i, l in enumerate(out) if l.startswith('(":" denotes')]
    if k:
        rec["aln_model"], rec["aln_marks"], rec["aln_ref"] = out[k[0] + 1], out[k[0] + 2], out[k[0] + 3]
    rows.append(rec)
pd.DataFrame(rows).to_csv(sys.argv[2], sep="\t", index=False)
print(len(rows), "pairs")
