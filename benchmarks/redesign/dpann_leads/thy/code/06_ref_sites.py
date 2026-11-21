#!/usr/bin/env python3
"""Active-site composition of the experimental T. maritima ThyX homotetramer (PDB 1O26, FAD + dUMP, 1.6 A): for each
dUMP (UMP) and FAD copy, the protein residues within 4.0 A, labeled by chain, and for the core residues of
05_halves_msa.py (H53 R78 H79 R80 S88 R90 R147 R174, plus R165) which chain supplies them relative to the chain that
carries the bulk of the site. Writes data/ref_sites.tsv.
usage: python code/06_ref_sites.py"""
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
atoms = []
for l in open("data/ref/1O26.pdb"):
    if l.startswith(("ATOM", "HETATM")):
        atoms.append(dict(rec=l[:6].strip(), name=l[12:16].strip(), res=l[17:20].strip(), chain=l[21], num=int(l[22:26]),
                          x=float(l[30:38]), y=float(l[38:46]), z=float(l[46:54])))
A = pd.DataFrame(atoms)
P = A[A.rec == "ATOM"]; X = P[["x", "y", "z"]].to_numpy()
CORE = [53, 78, 79, 80, 88, 90, 147, 165, 174]
rows = []
for (lig, ch, num), g in A[A.res.isin(["UMP", "FAD"])].groupby(["res", "chain", "num"]):
    G = g[["x", "y", "z"]].to_numpy()
    d = np.sqrt(((X[:, None, :] - G[None, :, :]) ** 2).sum(-1)).min(1)
    near = P[d <= 4.0].assign(d=d[d <= 4.0]).groupby(["chain", "num", "res"]).d.min().reset_index()
    cnt = near.chain.value_counts().to_dict()
    for r in near.itertuples():
        rows.append(dict(ligand=lig, lig_chain=ch, lig_num=num, chain=r.chain, resnum=r.num, res=r.res, dmin=round(r.d, 2), core=r.num in CORE))
    print(f"{lig} {ch}{num}: residues within 4 A by chain {cnt}; core residues:",
          sorted([(r.chain, r.res, r.num) for r in near.itertuples() if r.num in CORE], key=lambda t: t[2]))
pd.DataFrame(rows).to_csv("data/ref_sites.tsv", sep="\t", index=False)
