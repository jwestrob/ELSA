#!/usr/bin/env python3
"""Families holding lead-1 genes (mmseqs 30% family level and the permissive e3c50s75 level): members, genomes, members
with a Pfam Thy1 row (and with two), slot-called members, lengths. Also Fisher tests for question 1.
Writes data/lead1_families.tsv. usage: python code/10_families.py"""
import duckdb, pandas as pd
from scipy.stats import fisher_exact
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
SR = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/slot_rescue"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
L = pd.read_csv("data/lead_genes.tsv", sep="\t").query("lab=='Thy1'")
out = []
for run, col in [("pfr-8a31f5747e-family", "fam30"), ("permissive-e3c50s75", "perm")]:
    fams = tuple(L[col].dropna().unique())
    df = con.execute(f"""
      WITH m AS (SELECT pf.family_id, pf.protein_id, pr.bin_id, pr.sequence_length len FROM d.protein_families pf JOIN d.proteins pr USING (protein_id)
                 WHERE pf.clustering_run=? AND pf.family_id IN {fams}),
      t AS (SELECT protein_id, count(*) nd FROM d.annotations WHERE source='pfam' AND "name"='Thy1' GROUP BY 1),
      k AS (SELECT DISTINCT protein_id FROM d.annotations WHERE source IN ('kofam','kegg') AND accession='K03465'),
      s AS (SELECT protein_id, proposed_label_name FROM read_parquet('{SR}/pass/dpann_slot_rescue_pass.parquet'))
      SELECT m.family_id, count(*) members, count(DISTINCT m.bin_id) genomes, median(m.len) med_len,
             sum(CASE WHEN m.len >= 450 THEN 1 ELSE 0 END) n_ge450,
             sum(CASE WHEN t.nd >= 1 THEN 1 ELSE 0 END) pfam_thy1, sum(CASE WHEN t.nd = 2 THEN 1 ELSE 0 END) pfam_thy1_x2,
             sum(CASE WHEN k.protein_id IS NOT NULL THEN 1 ELSE 0 END) k03465,
             sum(CASE WHEN s.proposed_label_name='Thy1' THEN 1 ELSE 0 END) slot_thy1,
             sum(CASE WHEN t.nd IS NULL AND s.protein_id IS NULL THEN 1 ELSE 0 END) unlabeled
      FROM m LEFT JOIN t USING (protein_id) LEFT JOIN k USING (protein_id) LEFT JOIN s USING (protein_id) GROUP BY 1 ORDER BY members DESC""", [run]).df()
    df.insert(0, "level", col)
    df["lead_genes"] = df.family_id.map(L.groupby(col).size())
    out.append(df)
F = pd.concat(out)
F.to_csv("data/lead1_families.tsv", sep="\t", index=False)
print(F.to_string(index=False))

G = pd.read_csv("data/genome_functions.tsv", sep="\t")
pace = G[G.order == "o__Pacearchaeales"]
a = pace[pace.lead1 == 1]; b = pace[(pace.lead1 == 0) & (pace.lead2 == 0)]
t = [[int((a.TS_any == 0).sum()), int((a.TS_any == 1).sum())], [int((b.TS_any == 0).sum()), int((b.TS_any == 1).sum())]]
print("Pacearchaeales, lack any TS: lead1 genomes", t[0], "rest", t[1], "Fisher", fisher_exact(t))
# completeness-matched expectation: rest-of-order rate per completeness bin applied to the lead genomes
rate = b.groupby("comp_bin").apply(lambda x: (x.TS_any == 0).mean())
exp = a.comp_bin.map(rate).sum()
print("expected TS-lacking among lead1 Pacearchaeales genomes from completeness bins:", round(exp, 2), "observed", t[0][0], "of", len(a), "| bin rates", rate.round(3).to_dict())
a1 = G[G.lead1 == 1]
nb = G[(G.phylum == "p__Nanobdellota") & (G.lead1 == 0) & (G.lead2 == 0)]
rate2 = nb.groupby(["order", "comp_bin"]).apply(lambda x: (x.TS_any == 0).mean())
e2 = sum(rate2.get((o, c), (nb.TS_any == 0).mean()) for o, c in zip(a1.order, a1.comp_bin))
print("all lead1 genomes: observed TS-lacking", int((a1.TS_any == 0).sum()), "/", len(a1), "expected from order x completeness rates", round(e2, 2))
a2 = G[G.lead2 == 1]
e3 = sum(rate2.get((o, c), (nb.TS_any == 0).mean()) for o, c in zip(a2.order, a2.comp_bin))
print("lead2 genomes: observed TS-lacking", int((a2.TS_any == 0).sum()), "/", len(a2), "expected", round(e3, 2))
t2 = [[int((a2[a2.order=='o__Pacearchaeales'].TS_any == 0).sum()), int((a2[a2.order=='o__Pacearchaeales'].TS_any == 1).sum())], t[1]]
print("Pacearchaeales lead2 vs rest", t2, fisher_exact(t2))
