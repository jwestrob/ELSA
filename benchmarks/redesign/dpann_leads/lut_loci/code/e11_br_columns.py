#!/usr/bin/env python
"""lut_loci e11 (Mac): conserved carboxylate (D/E) and carbonyl-donor (N/Q side chain; G as a backbone-carbonyl turn marker;
S/T hydroxyl) columns, overall and by frozen genome group, for (a) the family, rescue3 token 275376 (one member per genome,
msa/br_275376_aln.faa from e09), and (b) the 29 handed-over motif proteins (mafft --auto). Columns with gap fraction <= 0.20; conserved = >= 0.80 of non-gap residues in the class (the T_seq
cutoff). Numbering of the first Bradyrhizobium representative (sel/selection_br.tsv). Writes data/br_columns.tsv.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
M = pd.read_csv(B + "/data/br_members.tsv", sep="\t")
SETNAME = sys.argv[1] if len(sys.argv) > 1 else "family"
q = M[M.in_handover].copy()
seqs = {int(r.id): str(r.seq) for r in SeqIO.parse(B + "/data/br_members.faa", "fasta")}
with open(B + "/msa/br29.faa", "w") as f:
    for g in q.gid:
        f.write(f">{g}\n{seqs[g]}\n")
with open(B + "/msa/br29_aln.faa", "w") as fo:
    subprocess.run(["mafft", "--auto", "--thread", "4", "--quiet", B + "/msa/br29.faa"], stdout=fo, check=True)
al = {int(r.id): str(r.seq) for r in SeqIO.parse(B + ("/msa/br29_aln.faa" if SETNAME == "29" else "/msa/br_275376_aln.faa"), "fasta")}
if SETNAME != "29":
    q = M[M.gid.isin(list(al))].copy()
S = pd.read_csv(B + "/sel/selection_br.tsv", sep="\t")
ref = int(S[(S.role == "br") & (S.genus == "Bradyrhizobium")].gid.iloc[0])
grp = q.set_index("gid").group
gsize = grp.value_counts()
big = gsize[gsize >= 3].index.tolist()
CL = {"DE": "DE", "NQ": "NQ", "G": "G", "ST": "ST"}
rows, pos = [], 0
ids = list(al)
A = np.array([list(al[i]) for i in ids])
for j, c in enumerate(al[ref]):
    if c != "-":
        pos += 1
    col = A[:, j]
    ng = col[col != "-"]
    gap = 1 - len(ng) / len(col)
    r = dict(col=j + 1, ref_pos=pos if c != "-" else None, ref_res=c, gap=round(gap, 2))
    for k, aa in CL.items():
        r[f"all_{k}"] = round(float(np.isin(ng, list(aa)).mean()) if len(ng) else 0, 2)
    for g in big:
        sub = A[[grp[i] == g for i in ids], j]
        sng = sub[sub != "-"]
        r[f"g{g}_DE"] = round(float(np.isin(sng, list("DE")).mean()) if len(sng) else 0, 2)
        r[f"g{g}_NQ"] = round(float(np.isin(sng, list("NQ")).mean()) if len(sng) else 0, 2)
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(B + f"/data/br_columns_{SETNAME}.tsv", sep="\t", index=False)
ok = D[D.gap <= 0.20]
print(SETNAME, len(q), "proteins, genome groups:", gsize.to_dict(), "| reference", ref, len(seqs[ref]), "aa")
for k in CL:
    sel = ok[ok[f"all_{k}"] >= 0.80]
    print(f"conserved {k} (>=.80, all): {len(sel)} columns:", ";".join(f"{r.ref_res}{r.ref_pos}" for r in sel.itertuples()))
for g in big:
    for k in ["DE", "NQ"]:
        sel = ok[ok[f"g{g}_{k}"] >= 0.80]
        print(f"group {g} (n={gsize[g]}) conserved {k}: {len(sel)}:", ";".join(f"{r.ref_res}{r.ref_pos}" for r in sel.itertuples()))
print("reference sequence:", seqs[ref])
