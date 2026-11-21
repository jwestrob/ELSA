#!/usr/bin/env python3
"""gapscreen step 2b (Mac, read-only): label-level function (fallback where partners carry no KO): for each Pfam label of
the pass, the KO distribution among all DPANN proteins with a stored row for that Pfam (accession form or name form).
Writes data/label_ko.tsv: label, n_label_proteins, n_with_ko, top_ko, top_n, share (top_n / n_with_ko), second_ko, second_n."""
from pathlib import Path
import duckdb, pandas as pd
H = Path(__file__).resolve().parents[1]
L = pd.read_parquet(H / "data/labeled.parquet", columns=["proposed_label_positional_homology", "proposed_label_name", "proposed_label_kind"])
labs = L[L.proposed_label_kind == "pfam"].drop_duplicates("proposed_label_positional_homology")
con = duckdb.connect(); con.execute("PRAGMA threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE TEMP TABLE lab AS SELECT * FROM labs")
r = con.execute("""
  WITH lp AS (SELECT DISTINCT l.proposed_label_positional_homology AS label, a.protein_id FROM d.annotations a JOIN lab l
              ON a.source = 'pfam' AND (split_part(a.accession, '.', 1) = l.proposed_label_positional_homology OR a."name" = l.proposed_label_name)),
       n AS (SELECT label, count(*) AS n_label_proteins FROM lp GROUP BY 1),
       k AS (SELECT lp.label, a.accession AS ko, count(DISTINCT lp.protein_id) AS c FROM lp JOIN d.annotations a USING (protein_id)
             WHERE a.source IN ('kofam', 'kegg') GROUP BY 1, 2),
       w AS (SELECT lp.label, count(DISTINCT lp.protein_id) AS n_with_ko FROM lp JOIN d.annotations a USING (protein_id)
             WHERE a.source IN ('kofam', 'kegg') GROUP BY 1),
       r AS (SELECT *, row_number() OVER (PARTITION BY label ORDER BY c DESC, ko) AS rk FROM k)
  SELECT n.label, n.n_label_proteins, coalesce(w.n_with_ko, 0) AS n_with_ko, r1.ko AS top_ko, coalesce(r1.c, 0) AS top_n,
         r2.ko AS second_ko, coalesce(r2.c, 0) AS second_n
  FROM n LEFT JOIN w USING (label) LEFT JOIN r r1 ON r1.label = n.label AND r1.rk = 1 LEFT JOIN r r2 ON r2.label = n.label AND r2.rk = 2""").df()
r["share"] = (r.top_n / r.n_with_ko.where(r.n_with_ko > 0)).round(4)
r.to_csv(H / "data/label_ko.tsv", sep="\t", index=False)
print(len(r), "labels;", int((r.n_with_ko >= 5).sum()), "with >= 5 KO-annotated carriers;", int(((r.n_with_ko >= 5) & (r.share >= 0.8)).sum()), "strong")
print(r[r.label.isin(["PF02502", "PF08540", "PF02511"])].to_string())
