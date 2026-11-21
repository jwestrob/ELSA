#!/usr/bin/env python3
"""Who else belongs to the protein families of the 17 lead genes (default family run and permissive runs), with their
taxonomy, length, Pfam/KO annotations; and the DPANN-wide set of genes annotated HMG_CoA_synt_N/C at the gathering threshold."""
import duckdb, pandas as pd
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 80); pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 800)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{R}/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
ids = [l[1:].strip() for l in open(f"{W}/lead17.faa") if l.startswith(">")]
print(con.execute("SELECT clustering_run, level, is_default, families, left(parameters,120) par FROM d.protein_family_runs").df().to_string())
fams = con.execute("SELECT DISTINCT family_id, clustering_run FROM d.protein_families WHERE protein_id IN (SELECT unnest(?))", [ids]).df()
for r in fams.itertuples():
    m = con.execute("""SELECT f.protein_id, pr.sequence_length len, split_part(b.taxonomy,';',2) ph, split_part(b.taxonomy,';',5) fam, split_part(b.taxonomy,';',7) sp,
          (SELECT string_agg(DISTINCT a.name, ',') FROM d.annotations a WHERE a.protein_id=f.protein_id AND a.source='pfam') pfam,
          (SELECT string_agg(DISTINCT a.accession, ',') FROM d.annotations a WHERE a.protein_id=f.protein_id AND a.source IN ('kofam','kegg')) ko,
          (SELECT arg_min(r.name, r.evalue) FROM d.protein_resemblance r WHERE r.protein_id=f.protein_id) cdd,
          (SELECT proposed_label_name FROM pass p WHERE p.protein_id=f.protein_id) slot
          FROM d.protein_families f JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id WHERE f.family_id=?""", [r.family_id]).df()
    print(f"\n## {r.family_id} ({r.clustering_run}): {len(m)} members, {m.ph.nunique()} phyla")
    if len(m) > 60:
        print(m.groupby(["ph", "pfam"], dropna=False).agg(n=("protein_id", "size"), len_med=("len", "median")).to_string())
    else:
        print(m.to_string())
# DPANN-wide genes with HMG_CoA_synt Pfam at GA
g = con.execute("""SELECT a.protein_id, a.name pf, a.start_aa, a.end_aa, a.evalue, pr.sequence_length len, pr.bin_id, split_part(b.taxonomy,';',2) ph, split_part(b.taxonomy,';',4) ord
                   FROM d.annotations a JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)
                   WHERE a.source='pfam' AND a.name IN ('HMG_CoA_synt_C','HMG_CoA_synt_N')""").df()
print("\n## DPANN genes with HMG_CoA_synt Pfam (GA):", g.protein_id.nunique(), "genes,", g.bin_id.nunique(), "genomes")
print(g.groupby(["ph", "pf"]).agg(genes=("protein_id", "nunique"), genomes=("bin_id", "nunique"), len_med=("len", "median"), len_min=("len", "min"), len_max=("len", "max")).to_string())
g.to_csv(f"{W}/dpann_hmgcs_pfam_GA_genes.tsv", sep="\t", index=False)
