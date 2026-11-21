#!/usr/bin/env python3
"""Compact DPANN-wide answer to Q4: among genomes (completeness >= 70) that carry a lipid-specific step (GGGPS K17104,
DGGGPS K17105 or CarS K19664), the fraction carrying each upstream step, per phylum and for Micrarchaeota families with
>= 20 genomes; the HMGS step counted as canonical OR the ~560-aa family; blank-step genomes with a slot call for that
step; and a completeness-90 check for the HMGCR gap."""
import pandas as pd
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
G = pd.read_parquet(f"{W}/genome_steps.parquet")
G["HMGS_any"] = G["2_HMGCS_canonical"] | G["2b_HMGS_like_560fam"]
S = pd.read_csv(f"{W}/slot_calls_isoprenoid_steps.tsv", sep="\t")
steps = ["1_thiolase_AACT", "HMGS_any", "2_HMGCS_canonical", "2b_HMGS_like_560fam", "3_HMGCR", "MVK_or_alt", "decarb_any", "6_IPK", "7_IDI", "8_GGPS"]
def tab(df, by):
    lp = df[df.lipid_specific_any]
    t = pd.DataFrame({"genomes_c70": df.groupby(by).size(), "lipid_genomes": lp.groupby(by).size()})
    for c in steps:
        t[c] = (lp.groupby(by)[c].mean() * 100).round(0)
    return t.fillna(0)
hq = G[G.completeness >= 70]
print("# percent of lipid-step genomes (completeness >= 70) carrying each upstream step, by phylum")
print(tab(hq, "phylum").to_string())
mic = hq[hq.phylum == "p__Micrarchaeota"]
big = mic.fam.value_counts(); big = big[big >= 20].index
print("\n# Micrarchaeota families with >= 20 genomes")
print(tab(mic[mic.fam.isin(big)], "fam").to_string())
print("\n# genomes where HMGS is blank by both groups / HMGCR blank, among lipid-step genomes (completeness >= 70)")
lp = hq[hq.lipid_specific_any]
print(lp.groupby("phylum").agg(n=("HMGS_any", "size"), HMGS_blank=("HMGS_any", lambda s: (~s).sum()), HMGCR_blank=("3_HMGCR", lambda s: (~s).sum())).to_string())
print("\n# completeness >= 90: HMGCR in genus Micrarchaeum_A and in 560-family genomes of Micrarchaeaceae")
h9 = G[G.completeness >= 90]
print("Micrarchaeum_A:", int(h9[h9.genus == "g__Micrarchaeum_A"]["3_HMGCR"].sum()), "/", int((h9.genus == "g__Micrarchaeum_A").sum()),
      "| M3K route:", int(h9[h9.genus == "g__Micrarchaeum_A"]["4alt_M3K_route"].sum()))
print("\n# slot calls whose label maps to a step, in genomes where that step is otherwise blank (all completeness)")
print(S.groupby("step").agg(genes=("protein_id", "size"), genomes=("bin_id", "nunique"), in_blank_genomes=("step_blank_in_genome", lambda s: (s == True).sum()),
      T1=("tier", lambda s: (s == "T1").sum()), len_med=("len", "median")).to_string())
