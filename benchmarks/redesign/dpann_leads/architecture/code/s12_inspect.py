#!/usr/bin/env python3
"""Step 12: inspect one group: members (length, genome, taxonomy, completeness, tier, support, alignment-list agreement),
weak full-Pfam domains (i-E <= 0.01) and CDD regions with coordinates, and the locus (+-3 genes) of up to 3 members.
usage: s12_inspect.py REP [REP ...]"""
import sys, duckdb, pandas as pd, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 48); pd.set_option("display.max_rows", 300)
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
H = pd.read_csv("data/pfam_weak_hits_all.tsv.gz", sep="\t")
R = pd.read_parquet("data/regions.parquet")
con = duckdb.connect(); con.execute("SET threads=2")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('../../slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
def locus(pid, w=3):
    c, gi = con.execute("select contig_id, gene_index from d.proteins where protein_id=?", [pid]).fetchone()
    q = """WITH win AS (SELECT protein_id, gene_index, strand, sequence_length FROM d.proteins WHERE contig_id=? AND gene_index BETWEEN ? AND ?),
    pf AS (SELECT protein_id, string_agg(DISTINCT "name", ',') pfam FROM d.annotations WHERE source='pfam' AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    ko AS (SELECT protein_id, string_agg(DISTINCT accession || ' ' || coalesce(description, "name"), '; ') ko FROM d.annotations WHERE source='kofam' AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    cd AS (SELECT protein_id, arg_min("name", evalue) cdd FROM d.protein_resemblance WHERE protein_id IN (SELECT protein_id FROM win) GROUP BY 1)
    SELECT w.gene_index - ? AS pos, w.protein_id, w.strand s, w.sequence_length len, left(pf.pfam,40) pfam, left(ko.ko, 50) ko, left(cd.cdd,25) cdd, p.proposed_label_name slot
    FROM win w LEFT JOIN pf USING (protein_id) LEFT JOIN ko USING (protein_id) LEFT JOIN cd USING (protein_id) LEFT JOIN pass p USING (protein_id) ORDER BY w.gene_index"""
    print(con.execute(q, [c, gi - w, gi + w, gi]).df().to_string(index=False))
for rep in sys.argv[1:]:
    g = A[A.rep == rep].copy()
    g["tax"] = g.taxonomy.str.split(";").str[3:6].str.join(";")
    print(f"\n######## group {rep}: {len(g)} genes, label {g.lname.iloc[0]} ({g.acc.iloc[0]}), carrier median {g.car_med_len.iloc[0]}")
    print(g[["protein_id", "bin_id", "len", "L_from", "L_to", "tier", "support", "lab_hmm_cov", "alignment_lists_same_label", "completeness", "hyd", "tax"]].sort_values("len").to_string(index=False))
    pids = list(g.protein_id[:4])
    h = H[H.sequence_id.isin(pids) & (H.i_evalue <= 1e-2)][["sequence_id", "hmm_name", "i_evalue", "env_from", "env_to", "hmm_from", "hmm_to"]]
    print(h.sort_values(["sequence_id", "env_from"]).to_string(index=False))
    r = R[R.protein_id.isin(pids) & (R.source == "cdd")][["protein_id", "fam", "evalue", "start_aa", "end_aa"]]
    print(r.sort_values(["protein_id", "start_aa"]).to_string(index=False))
    for pid in pids[:3]:
        print("  locus of", pid); locus(pid)
