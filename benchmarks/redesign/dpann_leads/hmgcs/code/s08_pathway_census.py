#!/usr/bin/env python3
"""Isoprenoid-precursor and archaeal-lipid steps per DPANN genome from every annotation source in the database
(kofam, kegg, pfam at GA), with the KEGG module M00849/M00365 step definitions plus the archaeal lipid steps; the two
HMGS-like groups are split: canonical (HMG_CoA_synt Pfam or K01641, outside the ~560-aa Micrarchaeota family) and the
~560-aa family (permissive family fam:5ad133d455:DAIDBP010000029.1_20 + fam:5ad133d455:DAHWNI010000033.1_6).
Outputs: genome_steps.parquet (genome x step booleans), census tables by phylum and by Micrarchaeota family."""
import duckdb, pandas as pd, numpy as np
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
FAM560 = ["fam:5ad133d455:DAIDBP010000029.1_20", "fam:5ad133d455:DAHWNI010000033.1_6"]
STEPS = {  # step: (KOs, Pfam names)
    "1_thiolase_AACT": (["K00626"], []),
    "2_HMGCS_canonical": (["K01641"], ["HMG_CoA_synt_N", "HMG_CoA_synt_C"]),
    "3_HMGCR": (["K00021", "K00054"], ["HMG-CoA_red"]),
    "4_MVK": (["K00869"], []),
    "4alt_M3K_route": (["K18689", "K18690", "K22813"], []),
    "5_PMDC": (["K17942"], []),
    "5alt_dehydratase_route": (["K25517", "K09128", "K25518"], []),
    "5alt_PMK_MVD": (["K00938", "K13273", "K01597"], []),
    "6_IPK": (["K06981"], []),
    "7_IDI": (["K01823"], []),
    "8_GGPS": (["K13787", "K13789", "K00804", "K00795"], []),
    "9_G1PDH": (["K00096"], []),
    "10_GGGPS": (["K17104"], []),
    "11_DGGGPS": (["K17105"], []),
    "12_CarS": (["K19664"], []),
    "13_GGR": (["K17830"], []),
    "MEP_any": (["K01662", "K00099", "K00991", "K12506", "K00919", "K01770", "K03526", "K03527"], []),
    "DUF35_scaffold": ([], ["OB_ChsH2_C", "zf-ChsH2"]),
}
bins = con.execute("""SELECT bin_id, completeness, contamination, split_part(taxonomy,';',2) phylum, split_part(taxonomy,';',3) cls,
                      split_part(taxonomy,';',4) ord, split_part(taxonomy,';',5) fam, split_part(taxonomy,';',6) genus, split_part(taxonomy,';',7) sp FROM d.bins""").df().set_index("bin_id")
fam560 = con.execute("SELECT DISTINCT protein_id FROM d.protein_families WHERE family_id IN (SELECT unnest(?))", [FAM560]).df().protein_id
con.execute("CREATE TEMP TABLE f560 AS SELECT unnest(?) protein_id", [fam560.tolist()])
G = pd.DataFrame(index=bins.index)
for st, (kos, pfs) in STEPS.items():
    q = []
    if kos:
        q.append(con.execute("SELECT DISTINCT pr.bin_id FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source IN ('kofam','kegg') AND a.accession IN (SELECT unnest(?))" + (" AND a.protein_id NOT IN (SELECT protein_id FROM f560)" if st.startswith("2_") else ""), [kos]).df().bin_id)
    if pfs:
        q.append(con.execute("SELECT DISTINCT pr.bin_id FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam' AND a.name IN (SELECT unnest(?))" + (" AND a.protein_id NOT IN (SELECT protein_id FROM f560)" if st.startswith("2_") else ""), [pfs]).df().bin_id)
    s = set().union(*[set(x) for x in q])
    G[st] = G.index.isin(s)
G["2b_HMGS_like_560fam"] = G.index.isin(set(con.execute("SELECT DISTINCT pr.bin_id FROM f560 JOIN d.proteins pr USING (protein_id)").df().bin_id))
G["2b_560fam_GA_annotated"] = G.index.isin(set(con.execute("SELECT DISTINCT pr.bin_id FROM f560 JOIN d.proteins pr USING (protein_id) JOIN d.annotations a USING (protein_id) WHERE a.source='pfam' AND a.name LIKE 'HMG_CoA_synt%'").df().bin_id))
G["lipid_specific_any"] = G[["10_GGGPS", "11_DGGGPS", "12_CarS"]].any(axis=1)
G["MVK_or_alt"] = G[["4_MVK", "4alt_M3K_route"]].any(axis=1)
G["decarb_any"] = G[["5_PMDC", "5alt_dehydratase_route", "5alt_PMK_MVD"]].any(axis=1)
G = G.join(bins)
G.to_parquet(f"{W}/genome_steps.parquet")
stepcols = [c for c in G.columns if c[0].isdigit() or c in ("MEP_any", "DUF35_scaffold", "lipid_specific_any", "MVK_or_alt", "decarb_any")]
def census(df, by, label):
    t = df.groupby(by).agg(genomes=("completeness", "size"), comp_med=("completeness", "median"))
    for c in stepcols:
        t[c] = df.groupby(by)[c].sum()
    print(f"\n## {label}: genomes with each step (counts)")
    print(t.to_string())
    return t
hq = G[G.completeness >= 70]
t1 = census(G, "phylum", "all DPANN genomes by phylum")
t2 = census(hq, "phylum", "DPANN genomes with completeness >= 70% by phylum")
t1.to_csv(f"{W}/census_phylum_all.tsv", sep="\t"); t2.to_csv(f"{W}/census_phylum_comp70.tsv", sep="\t")
mic = G[G.phylum == "p__Micrarchaeota"]
t3 = census(mic, "fam", "Micrarchaeota by family (all)")
t3.to_csv(f"{W}/census_micrarchaeota_family.tsv", sep="\t")
# genomes that carry the lipid-specific steps: which early steps are blank (completeness >= 70)
print("\n## genomes (completeness >= 70) carrying any lipid-specific step (GGGPS/DGGGPS/CarS): early-step presence")
lp = hq[hq.lipid_specific_any]
early = ["1_thiolase_AACT", "2_HMGCS_canonical", "2b_HMGS_like_560fam", "3_HMGCR", "MVK_or_alt", "decarb_any", "6_IPK", "7_IDI", "8_GGPS"]
t4 = lp.groupby("phylum").agg(lipid_genomes=("completeness", "size"))
t4["phylum_genomes_comp70"] = hq.groupby("phylum").size()
for c in early:
    t4[c] = lp.groupby("phylum")[c].sum()
t4["upper_MVA_all_blank(1-3)"] = lp.groupby("phylum").apply(lambda x: (~x[["2_HMGCS_canonical", "2b_HMGS_like_560fam", "3_HMGCR"]].any(axis=1)).sum())
print(t4.to_string())
t4.to_csv(f"{W}/census_lipid_genomes_early_steps.tsv", sep="\t")
# co-occurrence of the 560 family with canonical HMGCS and HMGCR inside Micrarchaeota
print("\n## Micrarchaeota (all completeness): 560-family x canonical HMGCS x HMGCR")
print(pd.crosstab([mic["2b_HMGS_like_560fam"], mic["2_HMGCS_canonical"]], mic["3_HMGCR"], margins=True))
m70 = mic[mic.completeness >= 70]
print("\n## Micrarchaeota completeness >= 70")
print(pd.crosstab([m70["2b_HMGS_like_560fam"], m70["2_HMGCS_canonical"]], m70["3_HMGCR"], margins=True))
