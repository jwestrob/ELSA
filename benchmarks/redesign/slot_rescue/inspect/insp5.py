#!/usr/bin/env python3
"""Candidate examples: structure-check pairs joined to tier, support and the pass table."""
import duckdb, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
con = duckdb.connect()
B = pd.read_csv("verify/B/checkB_pairs.tsv", sep="\t", dtype={"kind": str}).fillna({"kind": "null"})
B["kind"] = B.kind.str.lower()
con.register("B", B)
q = """
SELECT b.kind, b.protein_id, b.partner, b.label_name, round(b.TM1,2) tm, b.RMSD, round(b.IDali,2) idali, b.length len, b.partner_length plen,
       b.plddt_protein pl1, b.plddt_partner pl2, v.tier, round(v.vfrac,3) support, round(v.amp,1) amp, v.proposed_evalue ev,
       p.votes_for vf, p.supporting_genomes sg, p.supporting_groups grp, p.locus_size ls, p.genome_code
FROM B b JOIN read_parquet('verify/verification_per_protein.parquet') v USING (protein_id)
JOIN read_parquet('pass/dpann_slot_rescue_pass.parquet') p USING (protein_id)
WHERE b.kind='true' AND b.status='ok'
"""
T = con.execute(q).df()
print(T.tier.value_counts())
print("\n== T2 (no weak hit, >=50% support), confident structures, best TM")
print(T[(T.tier=="T2")&(T.pl1>=70)&(T.pl2>=70)].sort_values("tm",ascending=False).head(25).to_string())
print("\n== rest/T2b with low sequence identity but high TM")
print(T[(T.tier.isin(["rest","T2b"]))&(T.pl1>=70)&(T.pl2>=70)&(T.tm>=0.6)].sort_values("idali").head(25).to_string())
print("\n== T1 lowest identity, high TM")
print(T[(T.tier=="T1")&(T.pl1>=70)&(T.pl2>=70)&(T.tm>=0.7)].sort_values("idali").head(15).to_string())
print("\n== GM_CSF / LIM / L39 / Mis14 / Transcrip_act")
print(T[T.label_name.isin(["GM_CSF","LIM","Ribosomal_L39","Mis14","Transcrip_act","UBA_3","Med9","BcsB"])].sort_values(["label_name","tm"]).to_string())
print("\n== contradicted")
print(T[T.tier=="contradicted"].sort_values("tm",ascending=False).head(15).to_string())
T.to_csv("inspect/examples_candidates.tsv", sep="\t", index=False)
