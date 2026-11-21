#!/usr/bin/env python
"""Biotite login node, serial US-align. Monomers vs experimental RpiB chains (2VVP_A RPIB_MYCTU, 1NN4_A RPIB_ECOLI) and vs
the canonical DPANN RpiB model; dimers vs the experimental dimers (-mm 1). Saves TM-scores and the residue alignment.
usage: 10_usalign_remote.py pairs.tsv out.tsv   (pairs.tsv: kind, a, b, mode)"""
import subprocess, sys, pandas as pd
US = "/shared/software/bin/USalign"
P = pd.read_csv(sys.argv[1], sep="\t"); rows = []
for r in P.itertuples(index=False):
    cmd = [US, r.a, r.b] + (["-mm", "1", "-ter", "0"] if r.mode == "mm" else [])
    o = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
    rec = dict(kind=r.kind, a=r.a, b=r.b, mode=r.mode)
    tms = [l for l in o if l.startswith("TM-score=")]
    try:
        rec["TM_a"] = float(tms[0].split()[1]); rec["TM_b"] = float(tms[1].split()[1])
        al = [l for l in o if l.startswith("Aligned length=")][0]
        rec["Lali"] = int(al.split("Aligned length=")[1].split(",")[0]); rec["RMSD"] = float(al.split("RMSD=")[1].split(",")[0])
        rec["IDali"] = float(al.split("n_identical/n_aligned=")[1].split()[0]) if "n_identical/n_aligned=" in al else None
        i = [k for k, l in enumerate(o) if l.startswith('(":"')][0]
        rec["aln_a"], rec["aln_m"], rec["aln_b"] = o[i + 1], o[i + 2], o[i + 3]
    except Exception as e:
        rec["status"] = f"parse_error {e}"
    rows.append(rec)
pd.DataFrame(rows).to_csv(sys.argv[2], sep="\t", index=False)
print(len(rows), "pairs done")
