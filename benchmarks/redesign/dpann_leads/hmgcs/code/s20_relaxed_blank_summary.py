#!/usr/bin/env python3
"""Summarize the relaxed HMGS/HMGCR search over blank-step lipid genomes: per phylum, genomes lacking HMGS (or HMGCR) by
database annotation that carry a sub-threshold hit (full-sequence E <= 1e-10 to any HMGS profile, or to any HMGCR
profile), with the hit protein lengths. Hits to GHMP kinases by the K00054 profile are excluded by requiring a
HMG-CoA_red or K00021 hit for HMGCR."""
import pandas as pd, duckdb
pd.set_option("display.width", 300)
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
h = pd.read_csv(f"{W}/aksha_relaxed_blank/user_hmms_hits_df.tsv", sep="\t")
g = pd.read_csv(f"{W}/relaxed_blank_genomes.tsv", sep="\t", index_col=0)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
info = con.execute("SELECT protein_id sequence_id, bin_id, sequence_length len FROM d.proteins WHERE protein_id IN (SELECT unnest(?))", [h.sequence_id.unique().tolist()]).df()
h = h.merge(info, on="sequence_id")
h = h.sort_values("evalue").drop_duplicates(["sequence_id", "hmm_name"])
hmgs = h[h.hmm_name.isin(["HMG_CoA_synt_N", "HMG_CoA_synt_C", "K01641"]) & (h.evalue <= 1e-10)]
hmgcr = h[h.hmm_name.isin(["HMG-CoA_red", "K00021"]) & (h.evalue <= 1e-10)]
rows = []
for ph, sub in g.groupby("phylum"):
    a = sub[~sub.HMGS_any]; b = sub[~sub["3_HMGCR"]]
    ha = hmgs[hmgs.bin_id.isin(a.index)]; hb = hmgcr[hmgcr.bin_id.isin(b.index)]
    rows.append(dict(phylum=ph, HMGS_blank=len(a), HMGS_blank_with_subthreshold_hit=ha.bin_id.nunique(),
                     hit_len_med=ha.len.median(), HMGCR_blank=len(b), HMGCR_blank_with_subthreshold_hit=hb.bin_id.nunique()))
print(pd.DataFrame(rows).to_string(index=False))
print("\nbest HMGS-profile hit per HMGS-blank genome: E-value distribution")
best = hmgs[hmgs.bin_id.isin(g.index[~g.HMGS_any])].groupby("bin_id").evalue.min()
print(best.describe())
# which profile carries the hits and how they sit against the curated thresholds (KOfam K01641 full-sequence 282.27 bits)
x = hmgs[hmgs.bin_id.isin(g.index[~g.HMGS_any])]
print("\nper profile: proteins, genomes, median bitscore, max bitscore, median length")
print(x.groupby("hmm_name").agg(prot=("sequence_id", "nunique"), genomes=("bin_id", "nunique"), bits_med=("bitscore", "median"), bits_max=("bitscore", "max"), len_med=("len", "median")).to_string())
pf = con.execute("SELECT DISTINCT a.protein_id, a.name FROM d.annotations a WHERE a.protein_id IN (SELECT unnest(?)) AND a.source='pfam'", [x.sequence_id.unique().tolist()]).df()
print("Pfam (GA) names already carried by these hit proteins (top 8):"); print(pf.name.value_counts().head(8).to_string())
