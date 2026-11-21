#!/usr/bin/env python3
"""Build the residue-mapping alignment (mafft L-INS-i, 4 threads): 11 curated Swiss-Prot PF02502 members (UniProt JSON
in data/ref/), the 287 labeled long-form genes, the 165 same-slot PF02502 carriers (annotated long form), and the other
Micrarchaeota PF02502 carriers (canonical short form, other slots). Writes data/msa_input.faa, data/msa_aln.faa, data/msa_meta.tsv."""
import json, subprocess, duckdb, pandas as pd
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
REFS="P37351 P9WKD7 P0CL19 Q9ZB26 A0QXE5 J0MXJ0 P39156 Q92IL8 P23494 P23495 B9JN19".split()
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t"); car=pd.read_csv(D+"/carriers_class.tsv",sep="\t")
meta=[]; recs=[]
for a in REFS:
    j=json.load(open(f"{D}/ref/{a}.json")); recs.append((a,j["sequence"]["value"])); meta.append((a,"ref_"+j["uniProtkbId"]))
ids=list(lab.protein_id)+list(car.protein_id)
seq=con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))",[ids]).df().set_index("protein_id").sequence
for p in lab.protein_id: recs.append((p,seq[p].rstrip("*"))); meta.append((p,"labeled_"+lab.set_index("protein_id").tier[p]))
for r in car.itertuples():
    recs.append((r.protein_id,seq[r.protein_id].rstrip("*"))); meta.append((r.protein_id,"carrier_"+r.slot))
with open(D+"/msa_input.faa","w") as f:
    for i,s in recs: f.write(f">{i}\n{s}\n")
pd.DataFrame(meta,columns=["id","group"]).to_csv(D+"/msa_meta.tsv",sep="\t",index=False)
out=subprocess.run(["mafft","--localpair","--maxiterate","1000","--thread","4","--quiet",D+"/msa_input.faa"],capture_output=True,text=True,check=True).stdout
open(D+"/msa_aln.faa","w").write(out)
print(len(recs),"sequences aligned")
