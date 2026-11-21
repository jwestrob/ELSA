#!/usr/bin/env python3
"""Domain map of the query set from the Aksha Pfam scan (E <= 10): per-domain hits with HMM coverage (model length read
from the Pfam-A.hmm LENG lines) and sequence coordinates; summary per set."""
import os, re, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 600)
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
h = pd.read_csv(f"{W}/aksha_pfam_E10/user_hmms_hits_df.tsv", sep="\t")
s = pd.read_csv(f"{W}/query_arch_sets.tsv", sep="\t")
need = set(h.hmm_name)
leng, nm = {}, None
with open(os.path.expanduser("~/.config/Astra/PFAM/Pfam-A.hmm")) as fh:
    for line in fh:
        if line.startswith("NAME "):
            nm = line.split()[1]
        elif line.startswith("LENG ") and nm in need:
            leng[nm] = int(line.split()[1])
h["hmm_len"] = h.hmm_name.map(leng)
h["hmm_cov"] = ((h.hmm_to - h.hmm_from + 1) / h.hmm_len).round(2)
h = h.merge(s, left_on="sequence_id", right_on="protein_id")
L = {l[1:].strip(): None for l in open(f"{W}/query_arch.faa") if l.startswith(">")}
seq = {}
cur = None
for l in open(f"{W}/query_arch.faa"):
    if l.startswith(">"):
        cur = l[1:].strip()
    else:
        seq[cur] = l.strip()
h["qlen"] = h.sequence_id.map(lambda p: len(seq[p]))
k = h[h.i_evalue <= 1].sort_values(["set", "sequence_id", "ali_from"])
cols = ["set", "sequence_id", "qlen", "hmm_name", "evalue", "i_evalue", "dom_bitscore", "ali_from", "ali_to", "hmm_from", "hmm_to", "hmm_len", "hmm_cov"]
print(k[cols].to_string(index=False))
k[cols].to_csv(f"{W}/arch_domains_iE1.tsv", sep="\t", index=False)
print("\n# models hit (i_evalue <= 1) per set: number of proteins")
print(k.groupby(["hmm_name", "set"]).sequence_id.nunique().unstack(fill_value=0).to_string())
