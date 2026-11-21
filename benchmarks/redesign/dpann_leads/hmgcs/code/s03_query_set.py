#!/usr/bin/env python3
"""Query set for the weak-profile architecture scan: the 17 leads, GA-annotated members of the same ~560-aa
Micrarchaeota family (one per species), and canonical ~350-aa HMG_CoA_synt_N genes from Micrarchaeota and other DPANN."""
import duckdb, pandas as pd
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
lead = [l[1:].strip() for l in open(f"{W}/lead17.faa") if l.startswith(">")]
# members of the permissive family that carry HMG_CoA_synt Pfam at GA, one per species, longest first
fam = con.execute("""
  WITH m AS (SELECT f.protein_id, pr.sequence_length len, split_part(b.taxonomy,';',7) sp, split_part(b.taxonomy,';',5) fa,
             (SELECT string_agg(DISTINCT a.name, ',') FROM d.annotations a WHERE a.protein_id=f.protein_id AND a.source='pfam') pf
             FROM d.protein_families f JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id
             WHERE f.family_id='fam:5ad133d455:DAIDBP010000029.1_20')
  SELECT * FROM m WHERE pf LIKE '%HMG_CoA%' AND len BETWEEN 540 AND 620
  QUALIFY row_number() OVER (PARTITION BY sp ORDER BY len DESC)=1""").df()
print(len(fam), "GA-annotated family members (one per species)")
print(fam.to_string())
can = con.execute("""
  SELECT a.protein_id, pr.sequence_length len, split_part(b.taxonomy,';',2) ph, split_part(b.taxonomy,';',7) sp
  FROM d.annotations a JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)
  WHERE a.source='pfam' AND a.name='HMG_CoA_synt_N' AND pr.sequence_length BETWEEN 330 AND 380
  QUALIFY row_number() OVER (PARTITION BY split_part(b.taxonomy,';',2) ORDER BY b.completeness DESC)<=2""").df()
print(can.to_string())
sel = lead + fam.protein_id.tolist()[:12] + can.protein_id.tolist()
seqs = con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))", [sel]).df().set_index("protein_id").sequence
with open(f"{W}/query_arch.faa", "w") as fh:
    for p in sel:
        fh.write(f">{p}\n{seqs[p].rstrip('*')}\n")
pd.DataFrame({"protein_id": sel, "set": ["lead"] * len(lead) + ["family_GA"] * len(fam.protein_id.tolist()[:12]) + ["canonical_N"] * len(can)}).to_csv(f"{W}/query_arch_sets.tsv", sep="\t", index=False)
print(len(sel), "queries written")
