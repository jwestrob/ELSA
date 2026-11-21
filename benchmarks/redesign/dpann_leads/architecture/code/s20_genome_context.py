#!/usr/bin/env python3
"""Step 20 (rung 5/6): genome context for L1 and L2 members.
L2 (Tmk-NTP_transf_5): does the genome hold any other gene for dTMP kinase by any annotation source (Pfam Thymidylate_kin
  at GA, KOfam K00943, KEGG table, CDD best hits TMPK / Tmk), and the dTTP-pathway neighbors (thymidylate synthase ThyA
  K00560 / ThyX K03465, Pfam Thy1 / Thymidylat_synt; dUTPase / dCTP deaminase). Completeness vs the order's genomes.
L1 (NTP_transf_3-APH kinase): other genes in the genome with Pfam NTP_transf_3 / NTP_transferase / Choline_kinase / APH
  at GA (paralogs that could carry the same reactions), and the clade denominators.
Writes data/context_L1.tsv, data/context_L2.tsv and prints the summary."""
import duckdb, pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
bins = con.execute("SELECT bin_id, completeness, contamination, split_part(taxonomy,';',2) phylum, split_part(taxonomy,';',4) ord, split_part(taxonomy,';',5) fam FROM d.bins").df()
def genome_hits(bin_ids, pf_names=(), kos=(), cdd_names=(), exclude=()):
    con.register("b", pd.DataFrame(dict(bin_id=list(bin_ids))))
    q = f"""SELECT p.bin_id, p.protein_id, a.source, a.accession, a."name" FROM d.proteins p JOIN b USING (bin_id) JOIN d.annotations a USING (protein_id)
            WHERE (a.source='pfam' AND a."name" IN (SELECT unnest(?))) OR (a.source IN ('kofam','kegg') AND a.accession IN (SELECT unnest(?)))"""
    x = con.execute(q, [list(pf_names), list(kos)]).df()
    y = con.execute("""SELECT p.bin_id, p.protein_id, 'cdd' source, r.accession, r."name" FROM d.proteins p JOIN b USING (bin_id) JOIN d.protein_resemblance r USING (protein_id)
            WHERE r."name" IN (SELECT unnest(?))""", [list(cdd_names)]).df()
    z = pd.concat([x, y]); return z[~z.protein_id.isin(exclude)]
# ---- L2
g2 = A[A.rep.isin(["MBN2421503.1", "HIG96325.1"])][["protein_id", "bin_id", "len"]].merge(bins, on="bin_id")
tmk = genome_hits(g2.bin_id, ["Thymidylate_kin"], ["K00943"], ["TMPK", "Tmk", "PRK00698", "PRK13973", "PRK13974", "PRK13975", "PRK13976"], exclude=set(A.protein_id))
thy = genome_hits(g2.bin_id, ["Thy1", "Thymidylat_synt"], ["K00560", "K03465"], ["thyA", "ThyA", "Thy1", "thyX", "TS_Pyrimidine_HMase"])
g2["other_tmk_genes"] = g2.bin_id.map(tmk.groupby("bin_id").protein_id.nunique()).fillna(0).astype(int)
g2["thymidylate_synthase_genes"] = g2.bin_id.map(thy.groupby("bin_id").protein_id.nunique()).fillna(0).astype(int)
g2.to_csv("data/context_L2.tsv", sep="\t", index=False)
print("L2 member genomes", g2.bin_id.nunique(), "| with another dTMP-kinase gene (any source)", int((g2.other_tmk_genes > 0).sum()),
      "| with thymidylate synthase (ThyA or ThyX)", int((g2.thymidylate_synthase_genes > 0).sum()), "| completeness median", g2.completeness.median())
print(g2[["protein_id", "bin_id", "ord", "completeness", "other_tmk_genes", "thymidylate_synthase_genes"]].to_string(index=False))
print("sources of the other dTMP-kinase genes:", tmk.groupby(["source", "name"]).protein_id.nunique().to_dict())
# order baseline: share of genomes with any dTMP-kinase gene, by order and completeness >= 90
for o in g2.ord.value_counts().index[:3]:
    ob = bins[bins.ord == o]
    t = genome_hits(ob.bin_id, ["Thymidylate_kin"], ["K00943"], ["TMPK", "Tmk"], exclude=set())
    has = ob.bin_id.isin(t.bin_id)
    hi = ob.completeness >= 90
    print(f"  order {o}: genomes {len(ob)}, with any Tmk-family gene (incl. fused) {int(has.sum())} ({has.mean():.3f}); complete>=90%: {int(hi.sum())}, with Tmk {int((has & hi).sum())}")
# ---- L1
g1 = A[A.rep == "HLC90278.1"][["protein_id", "bin_id", "len"]].merge(bins, on="bin_id")
par = genome_hits(g1.bin_id, ["NTP_transf_3", "NTP_transferase", "Choline_kinase", "APH", "EcKL"], [], [], exclude=set(g1.protein_id))
for nm in ["NTP_transf_3", "NTP_transferase", "Choline_kinase", "APH"]:
    g1[nm] = g1.bin_id.map(par[par["name"] == nm].groupby("bin_id").protein_id.nunique()).fillna(0).astype(int)
g1.to_csv("data/context_L1.tsv", sep="\t", index=False)
print("\nL1 member genomes", g1.bin_id.nunique(), "| completeness median", g1.completeness.median())
print(g1[["protein_id", "bin_id", "ord", "fam", "completeness", "NTP_transf_3", "NTP_transferase", "Choline_kinase", "APH"]].to_string(index=False))
for o in g1.ord.value_counts().index:
    print(f"  order {o}: genomes {int((bins.ord == o).sum())}, with the fusion {g1[g1.ord == o].bin_id.nunique()}")
for f in g1.fam.value_counts().index:
    print(f"  family {f}: genomes {int((bins.fam == f).sum())}, with the fusion {g1[g1.fam == f].bin_id.nunique()}")
for o in g2.ord.value_counts().index[:4]:
    print(f"  L2 order {o}: genomes {int((bins.ord == o).sum())}, with the fusion {g2[g2.ord == o].bin_id.nunique()}")
