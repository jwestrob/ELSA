#!/usr/bin/env python3
"""Counts of pathway neighbors within +-3 genes of each lead-1 gene (from data/neighbors_lead1.tsv): tmk, RNR (incl.
NrdR), dUMP-producing enzymes, thioredoxin. Also start codon / partial flags of the long genes.
usage: python code/16_neighbor_counts.py"""
import pandas as pd, duckdb
N = pd.read_csv("data/neighbors_lead1.tsv", sep="\t")
N = N[N.rel.abs() <= 3]
lab = N.lab.fillna("") + " " + N.cdd.fillna("") + " " + N.koacc.fillna("")
pats = {"tmk": r"Thymidylate_kin|K00943|Tmk|TMPK", "RNR": r"Ribonuc_red|NRDD|NrdR|K00525|K21636|K00526|ATP-cone",
        "dUMP_src": r"dUTPase|DCD|dCMP_cyt_deam|MafB19|K01520|K01494|K01493|K09887", "thioredoxin": r"Thioredoxin"}
L = pd.read_csv("data/lead_genes.tsv", sep="\t").query("lab=='Thy1'")
for k, p in pats.items():
    hit = set(N[lab.str.contains(p, regex=True)].focal)
    print(f"{k}: {len(hit)}/{L.protein_id.nunique()} lead-1 genes have one within +-3")
anyp = set(N[lab.str.contains("|".join(pats.values()), regex=True)].focal)
print("any of these:", len(anyp), "/", L.protein_id.nunique())
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
x = con.execute(f"SELECT protein_id, sequence_length, partial, left(sequence,1) aa1 FROM d.proteins WHERE protein_id IN {tuple(L.protein_id)}").df()
x = x[x.sequence_length >= 420]
print("long genes:", len(x), "start with M:", int((x.aa1 == 'M').sum()), "partial flags:", x.partial.fillna("None (NCBI protein)").value_counts().to_dict())
