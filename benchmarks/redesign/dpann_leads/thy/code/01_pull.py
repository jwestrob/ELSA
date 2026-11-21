#!/usr/bin/env python3
"""thy workstream step 1: pull the two leads (slot-call labels Thy1 and Thymidylat_synt) with sequences, every stored
annotation of each lead gene (pfam, kofam, kegg, CDD rows with coordinates), their mmseqs family ids, and the
DPANN-wide sets of annotated family members (Pfam rows, KO rows from kofam and kegg, CDD best hits by name).
Writes data/lead_genes.tsv, data/lead1.faa, data/lead2.faa, data/lead_annotations.tsv, data/annotated_members.tsv,
data/annotated_thy1.faa, data/annotated_thya.faa.
usage (from the thy folder): python code/01_pull.py
"""
import duckdb, pandas as pd, os
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 60)
SR = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/slot_rescue"
os.makedirs("data", exist_ok=True)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{SR}/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{SR}/verify/verification_per_protein.parquet')")
con.execute(f"CREATE VIEW cka AS SELECT * FROM read_parquet('{SR}/verify/A/checkA_per_protein.parquet')")
print(con.execute("select column_name from duckdb_columns() where table_name='cka'").df().column_name.tolist())
print(con.execute("select column_name from duckdb_columns() where table_name='ver'").df().column_name.tolist())

L = con.execute("""
SELECT p.proposed_label_name lab, p.protein_id, pr.bin_id, pr.contig_id, pr.gene_index, pr.strand, pr.start, pr.end_coord, pr.partial,
       pr.sequence_length len, v.tier, v.vfrac support, v.proposed_evalue, v.best_other_label, v.best_other_evalue,
       p.votes_for, p.voters_annotated, p.vote_share, p.supporting_genomes, p.supporting_groups, p.runner_up_name, p.runner_up_votes,
       p.alignment_label, p.alignment_lists_same_label, p.own_ko, p.own_ko_agrees, p.confidence_bin, p.locus_id, p.locus_size, p.n_bonds,
       b.completeness, b.contamination, b.taxonomy, pr.sequence
FROM pass p JOIN ver v USING (protein_id) JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)
WHERE p.proposed_label_name IN ('Thy1','Thymidylat_synt') ORDER BY lab, len""").df()
fam = con.execute("""SELECT protein_id, clustering_run, family_id FROM d.protein_families
                     WHERE protein_id IN (SELECT protein_id FROM pass WHERE proposed_label_name IN ('Thy1','Thymidylat_synt'))""").df()
for run, col in [("pfr-8a31f5747e-family", "fam30"), ("pfr-8a31f5747e-subfamily", "sub50"), ("permissive-e3c50s75", "perm")]:
    L = L.merge(fam[fam.clustering_run == run][["protein_id", "family_id"]].rename(columns={"family_id": col}), on="protein_id", how="left")
L.drop(columns="sequence").to_csv("data/lead_genes.tsv", sep="\t", index=False)
for lab, fn in [("Thy1", "data/lead1.faa"), ("Thymidylat_synt", "data/lead2.faa")]:
    with open(fn, "w") as f:
        for r in L[L.lab == lab].itertuples():
            f.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
print(L.groupby("lab").agg(n=("protein_id", "size"), genomes=("bin_id", "nunique"), med_len=("len", "median")))
print(L[["lab", "protein_id", "len", "tier", "partial", "fam30", "perm"]].to_string(index=False))

ids = tuple(L.protein_id)
A = con.execute(f"""SELECT protein_id, source, accession, "name" nm, description, evalue, score, start_aa, end_aa FROM d.annotations WHERE protein_id IN {ids}
    UNION ALL SELECT protein_id, 'cdd' source, accession, "name" nm, description, evalue, score, start_aa, end_aa FROM d.protein_resemblance WHERE protein_id IN {ids}""").df()
A.to_csv("data/lead_annotations.tsv", sep="\t", index=False)
A = A.merge(L[["protein_id", "lab", "len"]], on="protein_id")
print(A.sort_values(["lab", "len", "protein_id", "start_aa"])[["lab", "protein_id", "len", "source", "accession", "nm", "evalue", "start_aa", "end_aa"]].to_string(index=False))

# DPANN-wide annotated members of both families, any source
M = con.execute("""
WITH hits AS (
  SELECT protein_id, source, accession, "name" nm, evalue, start_aa, end_aa FROM d.annotations
   WHERE (source='pfam' AND "name" IN ('Thy1','Thymidylat_synt')) OR (source IN ('kofam','kegg') AND accession IN ('K03465','K00560','K13998','K21636'))
  UNION ALL
  SELECT protein_id, 'cdd', accession, "name", evalue, start_aa, end_aa FROM d.protein_resemblance
   WHERE lower("name") SIMILAR TO '.*(thyx|thy1|thya|thy_syn|thymidylat|ts_pyrimidine|tdhfr|dhfr-ts).*' OR lower(description) LIKE '%thymidylate synthase%')
SELECT h.*, pr.bin_id, pr.sequence_length len, b.completeness, b.taxonomy FROM hits h JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)""").df()
M.to_csv("data/annotated_members.tsv", sep="\t", index=False)
print(M.groupby(["source", "nm"]).agg(rows=("protein_id", "size"), prots=("protein_id", "nunique"), genomes=("bin_id", "nunique"), med_len=("len", "median")).to_string())
for nm, fn in [("Thy1", "data/annotated_thy1.faa"), ("Thymidylat_synt", "data/annotated_thya.faa")]:
    s = con.execute("""SELECT a.protein_id, pr.sequence FROM d.annotations a JOIN d.proteins pr USING (protein_id)
                       WHERE a.source='pfam' AND a."name"=? GROUP BY ALL""", [nm]).df()
    with open(fn, "w") as f:
        for r in s.itertuples():
            f.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
    print(nm, "annotated Pfam proteins:", len(s))
