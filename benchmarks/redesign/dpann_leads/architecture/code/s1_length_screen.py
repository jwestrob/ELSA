#!/usr/bin/env python3
"""Architecture screen, step 1: carrier length per label and the length-outlier flag for every slot-labeled gene.

Carriers = DPANN proteins annotated with the label at the gathering threshold (Sharur annotations, source pfam by
accession; source kofam by KO for KO labels), labeled genes excluded. Writes data/genes.parquet (one row per labeled
gene with sequence-derived composition) and data/carriers.parquet (per-label carrier length stats)."""
import duckdb, pandas as pd, numpy as np, re, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SR = "../../slot_rescue"
os.makedirs("data", exist_ok=True)
con = duckdb.connect()
con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{SR}/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{SR}/verify/verification_per_protein.parquet')")
con.execute("CREATE TEMP TABLE labs AS SELECT DISTINCT proposed_label_positional_homology acc, proposed_label_kind kind, proposed_label_name lname FROM pass")
# carriers: Pfam by accession (version stripped), KO by accession
con.execute("""CREATE TEMP TABLE car AS
  SELECT l.acc, a.protein_id, pr.bin_id, pr.sequence_length len, a.start_aa, a.end_aa, a.score, pr.partial
  FROM labs l JOIN d.annotations a ON a.source='pfam' AND split_part(a.accession,'.',1)=l.acc AND l.kind='pfam'
  JOIN d.proteins pr USING (protein_id)
  UNION ALL
  SELECT l.acc, a.protein_id, pr.bin_id, pr.sequence_length len, a.start_aa, a.end_aa, a.score, pr.partial
  FROM labs l JOIN d.annotations a ON a.source='kofam' AND a.accession=l.acc AND l.kind='ko'
  JOIN d.proteins pr USING (protein_id)""")
con.execute("DELETE FROM car WHERE protein_id IN (SELECT protein_id FROM pass)")
C = con.execute("""SELECT acc, count(DISTINCT protein_id) n_car, count(DISTINCT bin_id) car_genomes,
    median(len) car_med_len, quantile_cont(len,0.9) car_q90_len, quantile_cont(len,0.95) car_q95_len, max(len) car_max_len,
    median(end_aa-start_aa+1) car_med_span
  FROM (SELECT acc, protein_id, any_value(bin_id) bin_id, any_value(len) len, min(start_aa) start_aa, max(end_aa) end_aa FROM car GROUP BY 1,2)
  GROUP BY 1""").df()
C.to_parquet("data/carriers.parquet", index=False)
con.execute("CREATE TEMP TABLE carstat AS SELECT * FROM C")
G = con.execute("""SELECT p.protein_id, p.proposed_label_positional_homology acc, p.proposed_label_name lname, p.proposed_label_kind kind,
    pr.bin_id, pr.contig_id, pr.gene_index, pr.strand, pr.start, pr.end_coord, pr.partial, pr.sequence_length len, pr.sequence seq,
    p.votes_for, p.supporting_genomes, p.vote_share, p.alignment_label, p.alignment_lists_same_label, p.own_ko, p.own_ko_agrees,
    v.tier, v.vfrac support, v.amp, v.proposed_evalue, v.best_other_label, v.best_other_evalue,
    b.completeness, b.contamination, b.taxonomy, ct.length contig_len,
    c.n_car, c.car_genomes, c.car_med_len, c.car_q90_len, c.car_q95_len, c.car_max_len, c.car_med_span
  FROM pass p JOIN ver v USING (protein_id) JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)
  LEFT JOIN d.contigs ct ON ct.contig_id=pr.contig_id
  LEFT JOIN carstat c ON c.acc=p.proposed_label_positional_homology""").df()
s = G.seq.str.rstrip("*")
G["hyd"] = s.map(lambda x: sum(ch in "ILVFMAWC" for ch in x) / max(len(x), 1))
G["ratio"] = G.len / G.car_med_len
G["extra"] = G.len - G.car_med_len
G["outlier"] = (G.ratio >= 1.4) & (G.extra >= 100)
G.to_parquet("data/genes.parquet", index=False)
n = len(G)
print("labeled genes", n, "| labels", G.acc.nunique())
print("genes whose label has >= 1 DPANN carrier", int(G.n_car.notna().sum()), "; labels without carrier", G[G.n_car.isna()].acc.nunique())
print("genes whose label has >= 5 carriers", int((G.n_car >= 5).sum()))
o = G[G.outlier]
print("length outliers (>= 1.4x carrier median and >= 100 aa extra):", len(o), "genes,", o.acc.nunique(), "labels,", o.bin_id.nunique(), "genomes")
print("  of which label has >= 5 carriers:", int((o.n_car >= 5).sum()))
print("  T1:", int((o.tier == "T1").sum()), " hyd<0.45:", int((o.hyd < 0.45).sum()))
print(o.groupby("lname").agg(n=("protein_id", "size"), genomes=("bin_id", "nunique"), len=("len", "median"), car=("car_med_len", "first"),
      ncar=("n_car", "first"), t1=("tier", lambda x: (x == "T1").sum())).sort_values("n", ascending=False).head(60).to_string())
