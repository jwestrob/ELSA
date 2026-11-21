#!/usr/bin/env python3
"""Step 22: locus co-occurrence around lead members (+-5 genes): for each Pfam (GA), CDD best hit and slot label, the
number of member loci (of N) that carry it. usage: s22_locus_cooccurrence.py REP"""
import sys, duckdb, pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('../../slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
for rep in sys.argv[1:]:
    g = A[A.rep == rep][["protein_id", "contig_id", "gene_index"]]
    con.register("g", g)
    w = con.execute("""SELECT g.protein_id focal, p.protein_id FROM g JOIN d.proteins p ON p.contig_id=g.contig_id
                       AND p.gene_index BETWEEN g.gene_index-5 AND g.gene_index+5 AND p.protein_id<>g.protein_id""").df()
    con.register("w", w)
    x = con.execute("""SELECT 'pfam:'||a."name" f, count(DISTINCT w.focal) n FROM w JOIN d.annotations a USING (protein_id) WHERE a.source='pfam' GROUP BY 1
        UNION ALL SELECT 'cdd:'||r."name", count(DISTINCT w.focal) FROM w JOIN d.protein_resemblance r USING (protein_id) GROUP BY 1
        UNION ALL SELECT 'slot:'||s.proposed_label_name, count(DISTINCT w.focal) FROM w JOIN pass s USING (protein_id) GROUP BY 1""").df()
    x = x.sort_values("n", ascending=False)
    print(f"#### {rep}: {len(g)} loci; features in >= 40% of loci")
    print(x[x.n >= 0.4 * len(g)].to_string(index=False))
    x.to_csv(f"data/locus_cooc_{rep}.tsv", sep="\t", index=False)
