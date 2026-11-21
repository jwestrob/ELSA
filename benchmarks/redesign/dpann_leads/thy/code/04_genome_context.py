#!/usr/bin/env python3
"""Question 1 / rung 5: per-genome presence of every dTMP-forming enzyme family and of the pathway steps around it,
by every annotation source in the Sharur DPANN database (pfam, kofam, kegg rows in annotations; CDD best hits in
protein_resemblance), excluding the lead gene's own rows. Rates in genomes carrying a lead gene versus the rest of the
same clade (phylum and order from bins.taxonomy), stratified by bins.completeness.
Writes data/genome_functions.tsv (one row per DPANN genome) and data/context_rates.tsv; prints the tables.
usage: python code/04_genome_context.py"""
import duckdb, pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
L = pd.read_csv("data/lead_genes.tsv", sep="\t")
con.register("leads", L[["protein_id", "lab", "tier"]])

# function -> (Pfam names, KOs [kofam or kegg], CDD accessions)
F = {
 "ThyX":      (["Thy1"], ["K03465"], ["cd20175", "COG1351", "pfam02511", "TIGR02170"]),
 "ThyA":      (["Thymidylat_synt"], ["K00560", "K13998", "K22882"],
               ["pfam00303", "PRK00956", "PRK01827", "PRK13821", "TIGR03283", "TIGR03284", "cd00351", "COG0207", "PTZ00164"]),
 "DUF166":    (["DUF166"], [], ["pfam02593"]),
 "RNR":       (["Ribonuc_red_lgC", "Ribonuc_red_lgN", "NRDD"], ["K00525", "K00526", "K10807", "K10808", "K21636", "K00527"],
               ["PRK08665", "PRK08270"]),
 "dUMP_src":  (["dUTPase", "dUTPase_2", "DCD"], ["K01520", "K01494", "K09887", "K01493"],
               ["PRK00601", "TIGR00576", "COG0756", "cd07557", "TIGR02274", "cd01286", "COG2131", "pfam00692", "pfam08761"]),
 "tmk":       (["Thymidylate_kin"], ["K00943"], ["COG0125", "cd01672", "TIGR00041", "PRK07933", "pfam02223", "PLN02924"]),
 "tdk":       (["TK"], ["K00857"], ["COG1435", "PRK04296", "pfam00265"]),
 "DHFR":      (["DHFR_1"], ["K00287", "K13998"], ["cd00209"]),
 "glyA":      (["SHMT"], ["K00600"], []),
}
rows = []
for fn, (pf, ko, cdd) in F.items():
    q = f"""
    SELECT DISTINCT pr.bin_id, '{fn}' fn, x.src FROM (
      SELECT protein_id, 'pfam' src FROM d.annotations WHERE source='pfam' AND "name" IN (SELECT unnest(?::VARCHAR[]))
      UNION ALL SELECT protein_id, 'ko' FROM d.annotations WHERE source IN ('kofam','kegg') AND accession IN (SELECT unnest(?::VARCHAR[]))
      UNION ALL SELECT protein_id, 'cdd' FROM d.protein_resemblance WHERE accession IN (SELECT unnest(?::VARCHAR[]))
    ) x JOIN d.proteins pr USING (protein_id) WHERE x.protein_id NOT IN (SELECT protein_id FROM leads)"""
    rows.append(con.execute(q, [pf or ["__none__"], ko or ["__none__"], cdd or ["__none__"]]).df())
H = pd.concat(rows)
B = con.execute("SELECT bin_id, completeness, contamination, taxonomy FROM d.bins").df()
B["phylum"] = B.taxonomy.str.split(";").str[1]
B["order"] = B.taxonomy.str.split(";").str[3]
W = H.assign(v=1).pivot_table(index="bin_id", columns="fn", values="v", aggfunc="max", fill_value=0)
for src in ["pfam", "ko", "cdd"]:
    for fn in ["ThyX", "ThyA"]:
        s = set(H[(H.fn == fn) & (H.src == src)].bin_id)
        B[f"{fn}_{src}"] = B.bin_id.isin(s).astype(int)
B = B.merge(W, left_on="bin_id", right_index=True, how="left").fillna({k: 0 for k in F})
for k in F:
    B[k] = B[k].astype(int)
B["TS_any"] = ((B.ThyX + B.ThyA + B.DUF166) > 0).astype(int)
B["lead1"] = B.bin_id.isin(set(L[L.lab == "Thy1"].bin_id)).astype(int)
B["lead1_T1"] = B.bin_id.isin(set(L[(L.lab == "Thy1") & (L.tier == "T1")].bin_id)).astype(int)
B["lead2"] = B.bin_id.isin(set(L[L.lab == "Thymidylat_synt"].bin_id)).astype(int)
B["comp_bin"] = pd.cut(B.completeness, [0, 70, 80, 90, 101], right=False, labels=["<70", "70-80", "80-90", ">=90"])
B.drop(columns="taxonomy").to_csv("data/genome_functions.tsv", sep="\t", index=False)

cols = ["TS_any", "ThyX", "ThyA", "DUF166", "RNR", "dUMP_src", "tmk", "tdk", "DHFR", "glyA"]
out = []
def summ(df, name):
    r = dict(group=name, n=len(df), comp_median=round(df.completeness.median(), 1))
    for c in cols:
        r[c] = f"{int(df[c].sum())}/{len(df)}"
    r["none_TS_RNR_tmk"] = f"{int(((df.TS_any == 0) & (df.RNR == 1) & (df.tmk == 1)).sum())}/{len(df)}"
    out.append(r)
for lead, flag in [("lead1", "lead1"), ("lead1_T1", "lead1_T1"), ("lead2", "lead2")]:
    g = B[B[flag] == 1]
    summ(g, f"{lead} genomes (all)")
    summ(g[g.completeness >= 80], f"{lead} genomes comp>=80")
    for ph, od in g[["phylum", "order"]].value_counts().index:
        pass
# clade baselines: the orders that hold the lead genes
for od in ["o__Pacearchaeales", "o__Woesearchaeales", "o__SCGC-AAA011-G17"]:
    c = B[(B.order == od)]
    for flag in ["lead1", "lead2"]:
        summ(c[c[flag] == 1], f"{od} with {flag}")
    rest = c[(c.lead1 == 0) & (c.lead2 == 0)]
    summ(rest, f"{od} rest (no lead gene)")
    summ(rest[rest.completeness >= 80], f"{od} rest comp>=80")
    for cb in ["70-80", "80-90", ">=90"]:
        summ(rest[rest.comp_bin == cb], f"{od} rest comp {cb}")
nb = B[B.phylum == "p__Nanobdellota"]
summ(nb[(nb.lead1 == 0) & (nb.lead2 == 0)], "Nanobdellota rest")
summ(B, "all DPANN")
R = pd.DataFrame(out)
R.to_csv("data/context_rates.tsv", sep="\t", index=False)
print(R.to_string(index=False))

# per lead-1 genome detail
D = B[B.lead1 == 1].merge(L[L.lab == "Thy1"][["bin_id", "protein_id", "len", "tier"]], on="bin_id")
print(D[["bin_id", "protein_id", "len", "tier", "completeness", "order", "TS_any", "ThyX_pfam", "ThyX_ko", "ThyX_cdd", "ThyA_pfam", "ThyA_ko", "ThyA_cdd", "DUF166",
         "lead2", "RNR", "dUMP_src", "tmk", "tdk", "DHFR", "glyA"]].sort_values(["TS_any", "order"]).to_string(index=False))
D2 = B[B.lead2 == 1]
print("lead2 genomes:", len(D2), "with ThyX any source", int(D2.ThyX.sum()), "with ThyA (other gene) any source", int(D2.ThyA.sum()),
      "with lead1 gene", int(D2.lead1.sum()), "TS_any", int(D2.TS_any.sum()), "RNR", int(D2.RNR.sum()), "tmk", int(D2.tmk.sum()), "DHFR", int(D2.DHFR.sum()))
