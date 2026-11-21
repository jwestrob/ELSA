#!/usr/bin/env python3
"""Lead overview for the 17 HMG_CoA_synt_C slot calls: pass rows, verification tier, every annotation of each protein
(pfam, kofam, kegg, CDD with coordinates), family-level HHsearch hits, and the 17 sequences written to lead17.faa."""
import duckdb, pandas as pd
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 80); pd.set_option("display.max_colwidth", 90)
pd.set_option("display.max_rows", 500)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{R}/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{R}/slot_rescue/verify/verification_per_protein.parquet')")
con.execute(f"CREATE VIEW ca AS SELECT * FROM read_parquet('{R}/slot_rescue/verify/A/checkA_per_protein.parquet')")
L = con.execute("""SELECT p.*, v.tier, v.vfrac, v.amp, v.annotated, v.proposed_evalue, v.best_other_label, v.best_other_evalue,
                   pr.sequence, pr.bin_id, pr.gene_index gi, b.completeness, b.taxonomy
                   FROM pass p JOIN ver v USING (protein_id) JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)
                   WHERE p.proposed_label_name='HMG_CoA_synt_C' ORDER BY p.locus_id, p.protein_id""").df()
print(len(L), "lead rows")
cols = ["protein_id", "genome", "contig", "position_index", "strand", "length", "own_ko", "proposed_label_positional_homology",
        "votes_for", "voters_annotated", "vote_share", "supporting_genomes", "supporting_groups", "runner_up_name", "runner_up_votes",
        "locus_id", "locus_size", "n_bonds", "max_bond_lambda", "alignment_label", "alignment_lists_same_label", "confidence_bin",
        "own_ko_agrees", "basis", "tier", "vfrac", "amp", "annotated", "proposed_evalue", "best_other_label", "best_other_evalue"]
print(L[cols].to_string())
print("\ntaxonomy:\n", L.taxonomy.value_counts().to_string())
with open(f"{W}/lead17.faa", "w") as fh:
    for r in L.itertuples():
        fh.write(f">{r.protein_id}\n{r.sequence}\n")
ids = L.protein_id.tolist()
con.execute("CREATE TEMP TABLE ids AS SELECT unnest(?) protein_id", [ids])
print("\n# annotations table rows for the 17")
print(con.execute("SELECT a.protein_id, source, accession, a.name, evalue, score, start_aa, end_aa FROM d.annotations a JOIN ids USING (protein_id) ORDER BY 1, start_aa").df().to_string())
print("\n# CDD (protein_resemblance) hits for the 17")
print(con.execute("SELECT r.protein_id, accession, r.name, left(description,70) descr, evalue, score, start_aa, end_aa FROM d.protein_resemblance r JOIN ids USING (protein_id) ORDER BY 1, start_aa").df().to_string())
print("\n# checkA weak-hit summary")
print(con.execute("SELECT c.* FROM ca c JOIN ids USING (protein_id)").df().to_string())
print("\n# protein families and family-level HHsearch hits")
fam = con.execute("SELECT f.protein_id, f.family_id, f.clustering_run, f.assignment FROM d.protein_families f JOIN ids USING (protein_id)").df()
print(fam.to_string())
fids = fam.family_id.unique().tolist()
if fids:
    print(con.execute("""SELECT family_id, source, db_name, hit_rank, hit_accession, hit_name, probability, evalue, q_start, q_end, q_length, t_start, t_end, t_length
                         FROM d.family_annotations WHERE family_id IN (SELECT unnest(?)) ORDER BY family_id, source, hit_rank""", [fids]).df().to_string())
print("\n# predicates")
print(con.execute("SELECT p.protein_id, predicates FROM d.protein_predicates p JOIN ids USING (protein_id)").df().to_string())
