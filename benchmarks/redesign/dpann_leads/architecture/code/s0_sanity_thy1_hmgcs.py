#!/usr/bin/env python3
"""Sanity check for the two cases assigned elsewhere: carrier length and domain copy number of Thy1 (PF02511) and
HMG_CoA_synt_C (PF08540) in DPANN, and the labeled genes' ratio to the carrier median."""
import duckdb, pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
con = duckdb.connect(); con.execute("SET threads=2")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
G = pd.read_parquet("data/genes2.parquet")
for acc, nm in [("PF02511", "Thy1"), ("PF08540", "HMG_CoA_synt_C")]:
    df = con.execute(f"""SELECT a.protein_id, any_value(p.sequence_length) len, count(*) ndom, min(a.start_aa) s, max(a.end_aa) e
        FROM d.annotations a JOIN d.proteins p USING (protein_id) WHERE a.source='pfam' AND split_part(a.accession,'.',1)='{acc}' GROUP BY 1""").df()
    g = G[G.acc == acc]
    print(nm, "carriers", len(df), "median len", df.len.median(), "| domains per carrier", df.ndom.value_counts().sort_index().to_dict(),
          "| carriers >= 450 aa", int((df.len >= 450).sum()))
    print("  labeled", len(g), "median len", g.len.median(), "ratio to carrier median", round(g.len.median() / df.len.median(), 2),
          "outliers", int(g.outlier.sum()), "label copies>=2 (weak hits)", int((g.lab_ndom >= 2).sum()))
