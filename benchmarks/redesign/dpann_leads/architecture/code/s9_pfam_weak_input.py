#!/usr/bin/env python3
"""Architecture screen, step 9 input: up to 3 members per conserved group (distinct genera first, longest-label-region
first) for a weak full-Pfam search with Aksha (E <= 10 per domain), to annotate the extra region."""
import pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
S = pd.read_csv("groups_conserved.tsv", sep="\t")
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
A = A[A.rep.isin(S.rep)].copy()
A["genus"] = A.taxonomy.str.split(";").str[5]
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
pick = A.sort_values(["rep", "tier"]).groupby(["rep", "genus"]).head(1).groupby("rep").head(3)
with open("data/pfam_weak_input.faa", "w") as f:
    for pid in pick.protein_id: f.write(f">{pid}\n{seqs[pid].rstrip('*')}\n")
pick[["rep", "protein_id"]].to_csv("data/pfam_weak_input.tsv", sep="\t", index=False)
print(len(pick), "sequences,", pick.rep.nunique(), "groups, residues", int(pick.len.sum()))
