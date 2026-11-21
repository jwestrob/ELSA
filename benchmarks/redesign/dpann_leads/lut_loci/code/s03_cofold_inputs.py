#!/usr/bin/env python
"""lut_loci s03 (Mac): co-fold set by tests_rule.json T_cofold 'who', from data/sites_apo.tsv.

top 2 representatives per candidate family by score_cons then score_struct then mean pLDDT; 2 per collection-positive family
(lanM, LanD) by the same order; both public references; every negative A; the 10 negatives B with the highest score_struct
(then score_cons, mean pLDDT). xoxF excluded. Writes folds_in/cofold.faa.
"""
import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
D = pd.read_csv(B + "/data/sites_apo.tsv", sep="\t")
key = ["score_cons", "score_struct", "mean_plddt"]
pick = []
for fam, g in D[D.role == "candidate"].groupby("family"):
    pick += g.sort_values(key, ascending=False).head(2).name.tolist()
for fam in ["lanM", "lutD_LanD"]:
    pick += D[D.family == fam].sort_values(key, ascending=False).head(2).name.tolist()
pick += D[D.role == "posref"].name.tolist()
pick += D[D.role == "negA"].name.tolist()
pick += D[D.role == "negB"].sort_values(["score_struct", "score_cons", "mean_plddt"], ascending=False).head(10).name.tolist()
seqs = {r.id: str(r.seq) for r in SeqIO.parse(B + "/folds_in/apo.faa", "fasta")}
with open(B + "/folds_in/cofold.faa", "w") as f:
    for p in pick:
        f.write(f">{p}\n{seqs[p]}\n")
print(len(pick))
