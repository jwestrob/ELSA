#!/usr/bin/env python
"""lut_loci s01 (Mac): family alignments for every tested family (tests_rule.json T_seq).

Family members: sel/fam_<family>_gids.txt (one member per genome, <= 150; code/n07_select.py) plus the family's representatives
(sel/selection.tsv). mafft --auto --thread 4. Output msa/<family>_aln.faa (FASTA, header = gid).
"""
import os
import subprocess

import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
os.makedirs(B + "/msa", exist_ok=True)
S = pd.read_csv(B + "/sel/selection.tsv", sep="\t")
seqs = {int(r.id): str(r.seq).rstrip("*") for r in SeqIO.parse(B + "/sel/all.faa", "fasta")}
for fam, g in S.groupby("family"):
    out = f"{B}/msa/{fam}_aln.faa"
    if os.path.exists(out):
        continue
    mem = [int(x) for x in open(f"{B}/sel/fam_{fam}_gids.txt") if x.strip()]
    ids = list(dict.fromkeys(mem + g.gid.astype(int).tolist()))
    fa = f"{B}/msa/{fam}.faa"
    with open(fa, "w") as f:
        for i in ids:
            f.write(f">{i}\n{seqs[i]}\n")
    with open(out, "w") as fo:
        subprocess.run(["mafft", "--auto", "--thread", "4", "--quiet", fa], stdout=fo, check=True)
    print(fam, len(ids))
