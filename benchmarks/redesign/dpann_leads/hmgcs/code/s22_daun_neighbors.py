#!/usr/bin/env python3
"""Neighbors (+-6) of the one lead gene outside the main permissive family (DAUN01000032.1_9), for the 17-gene neighbor table."""
import duckdb, pandas as pd
pd.set_option("display.width", 250)
con = duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
print(con.execute("""SELECT n.gene_index - m.gene_index rel, n.protein_id, n.strand, n.sequence_length len,
   (SELECT string_agg(DISTINCT a.name, ',') FROM d.annotations a WHERE a.protein_id=n.protein_id AND a.source='pfam') pfam,
   (SELECT arg_min(r.name, r.evalue) FROM d.protein_resemblance r WHERE r.protein_id=n.protein_id) cdd,
   (SELECT f.family_id FROM d.protein_families f WHERE f.protein_id=n.protein_id AND f.clustering_run='permissive-e3c50s75-rescue3c') fid
   FROM d.proteins m JOIN d.proteins n ON n.contig_id=m.contig_id AND abs(n.gene_index-m.gene_index)<=6 WHERE m.protein_id='DAUN01000032.1_9' ORDER BY rel""").df().to_string(index=False))
