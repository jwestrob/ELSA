#!/usr/bin/env python3
"""Step 11: final screen table. Joins known-form, extra-region annotation and gene-call controls; counts independent
sequences per group (single-linkage clusters at >= 95% identity over the MMseqs2 all-vs-all); applies the last controls:
  independence: >= 3 sequence clusters at 95% identity in >= 2 genera
  split control (step 13, refined label domain, extension-sized partners only): short-form genomes with an adjacent
                 extra-region homolog <= 10% and members holding a separate adjacent copy <= 10%
Writes candidates_ranked.tsv (every group that passes the known-form control, with pass/fail flags) and prints the
counts at each step."""
import pandas as pd, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
P = pd.read_parquet("data/extra_annot.parquet")
C = pd.read_parquet("data/split_check.parquet")
C2 = pd.read_csv("data/split_refined.tsv", sep="\t")
C = C.drop(columns=["short_genomes", "short_genomes_with_adjacent_extra", "members_with_separate_adjacent_extra", "side"]).merge(
    C2.rename(columns={"short_genomes_split": "short_genomes_with_adjacent_extra", "members_with_separate_copy": "members_with_separate_adjacent_extra"})[
    ["rep", "side", "short_genomes", "short_genomes_with_adjacent_extra", "members_with_separate_adjacent_extra", "x_from_med", "x_to_med"]], on="rep")
C["x_len_med"] = C.x_to_med - C.x_from_med + 1
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
A = A[A.rep.isin(P.rep)]
ava = pd.read_csv("data/cand_ava.m8.gz", sep="\t", header=None, names="query target fident alnlen qstart qend qlen tstart tend tlen evalue bits".split())
ava = ava[(ava.fident >= 0.95) & ava["query"].isin(A.protein_id) & ava.target.isin(A.protein_id)]
par = {p: p for p in A.protein_id}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
rep_of = dict(zip(A.protein_id, A.rep))
for q, t in zip(ava["query"], ava.target):
    if rep_of[q] == rep_of[t]: par[find(q)] = find(t)
A["c95"] = [find(p) for p in A.protein_id]
ind = A.groupby("rep").c95.nunique().rename("seq_clusters95")
P = P.merge(ind, left_on="rep", right_index=True).merge(C.rename(columns={"members": "members_n"}), on="rep", how="left")
P["independent"] = (P.seq_clusters95 >= 3) & (P.genera >= 2)
P["split_frac_short"] = P.short_genomes_with_adjacent_extra / P.short_genomes.clip(lower=1)
P["split_frac_members"] = P.members_with_separate_adjacent_extra / P.members_n
P["split_ok"] = (P.split_frac_short <= 0.10) & (P.split_frac_members <= 0.10)
P["pass_all"] = P.independent & P.split_ok
n = len(P)
print("groups passing known-form", n, "| independent", int(P.independent.sum()), "| + split control", int(P.pass_all.sum()),
      "| + extra region annotated (>= 50% of members)", int((P.pass_all & P.extra_annotated).sum()))
cols = ["rep", "label", "genes", "genomes", "genera", "seq_clusters95", "top_order", "orders", "phyla", "len_med", "len_min", "len_max", "car_med", "ncar_all",
        "L_from_med", "L_to_med", "side", "x_len_med", "extra_pfam", "extra_cdd", "label_region_pfam", "carriers_fl", "arch_annotated_in_carriers",
        "short_genomes", "short_genomes_with_adjacent_extra", "members_with_separate_adjacent_extra", "fident_q50", "t1", "support_med", "emb_only", "hyd_med",
        "independent", "split_ok", "extra_annotated", "pass_all", "members"]
P = P.sort_values(["pass_all", "extra_annotated", "genomes"], ascending=False)
P[cols].to_csv("candidates_ranked.tsv", sep="\t", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 50); pd.set_option("display.max_rows", 200)
print(P[P.pass_all][["rep", "label", "genes", "genera", "seq_clusters95", "len_med", "car_med", "side", "x_len_med", "extra_pfam", "carriers_fl", "short_genomes_with_adjacent_extra", "short_genomes"]].to_string(index=False))
print("\nFAILED independence or split control:")
print(P[~P.pass_all][["rep", "label", "genes", "genera", "seq_clusters95", "split_frac_short", "split_frac_members", "extra_pfam"]].to_string(index=False))
