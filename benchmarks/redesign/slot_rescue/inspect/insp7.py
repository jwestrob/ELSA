#!/usr/bin/env python3
"""Neighborhood make-up of two over-extended labels, and check-A rows for the example genes."""
import duckdb, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('pass/dpann_slot_rescue_pass.parquet')")
for lab in ["GM_CSF", "Transcrip_act", "Mis14"]:
    con.execute("""CREATE OR REPLACE TEMP TABLE nb AS
      SELECT g.protein_id focal, n.protein_id nbr, n.gene_index - f.gene_index d_off
      FROM pass g JOIN d.proteins f USING (protein_id)
      JOIN d.proteins n ON n.contig_id=f.contig_id AND n.gene_index BETWEEN f.gene_index-3 AND f.gene_index+3 AND n.protein_id<>f.protein_id
      WHERE g.proposed_label_name=?""", [lab])
    n = con.execute("select count(distinct focal) from nb").fetchone()[0]
    tot = con.execute("select count(*) from pass where proposed_label_name=?", [lab]).fetchone()[0]
    print(f"\n### {lab}: {tot} labeled genes ({n} with a neighbor within 3 genes)")
    print(" Pfam domains on neighbors (genes with the domain within 3):")
    print(con.execute("""SELECT a.name, count(distinct nb.focal) n FROM nb JOIN d.annotations a ON a.protein_id=nb.nbr AND a.source='pfam'
        GROUP BY 1 ORDER BY n DESC LIMIT 8""").df().to_string(index=False))
    print(" proposed labels on neighbors:")
    print(con.execute("""SELECT p.proposed_label_name lab, count(distinct nb.focal) n, count(distinct CASE WHEN abs(nb.d_off)=1 THEN nb.focal END) adjacent
        FROM nb JOIN pass p ON p.protein_id=nb.nbr GROUP BY 1 ORDER BY n DESC LIMIT 6""").df().to_string(index=False))
    print(" phyla of the labeled genes:")
    print(con.execute("""SELECT split_part(b.taxonomy,';',2) phylum, count(*) n, count(distinct pr.bin_id) genomes FROM pass g JOIN d.proteins pr USING (protein_id)
        JOIN d.bins b USING (bin_id) WHERE g.proposed_label_name=? GROUP BY 1 ORDER BY n DESC LIMIT 8""", [lab]).df().to_string(index=False))
ids = ["MEM2032468.1","MEW6721824.1","MBS3076775.1","JBFYLN010000177.1_6","DAHXEF010000003.1_67","MEI6731623.1","MBI4147890.1","DAICMH010000013.1_19","MEM4071999.1","MBI4018456.1","MCX6707026.1"]
print("\n### check A rows")
print(con.execute("SELECT * FROM read_parquet('verify/verification_per_protein.parquet') WHERE protein_id IN (SELECT unnest(?))", [ids]).df().to_string(index=False))
print(con.execute("select count(distinct bin_id) from d.proteins").fetchone())
