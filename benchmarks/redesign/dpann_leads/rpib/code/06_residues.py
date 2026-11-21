#!/usr/bin/env python3
"""Rung 3: residues of the long-form genes at the curated positions of RPIB_ECOLI (P37351) and RPIB_MYCTU (P9WKD7)
(active and binding sites from the UniProt records in data/ref/). Reads data/msa_aln.faa. Writes data/residues_per_seq.tsv
and data/residue_table.tsv; prints the per-group residue frequencies."""
import json, pandas as pd, numpy as np
from collections import Counter
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
def readfa(p):
    s={};k=None
    for l in open(p):
        l=l.strip()
        if l.startswith(">"): k=l[1:].split()[0]; s[k]=""
        elif k: s[k]+=l
    return s
A=readfa(D+"/msa_aln.faa"); meta=pd.read_csv(D+"/msa_meta.tsv",sep="\t").set_index("id").group
def col_of(seqid,pos):  # alignment column of residue number pos (1-based)
    n=0
    for i,c in enumerate(A[seqid]):
        if c!="-":
            n+=1
            if n==pos: return i
def feats(acc):
    j=json.load(open(f"{D}/ref/{acc}.json")); out={}
    for f in j["features"]:
        if f["type"] in ("Active site","Binding site"):
            b=f["location"]["start"]["value"]; e=f["location"]["end"]["value"]
            for p in range(b,e+1):
                out.setdefault(p,set()).add("ACT:"+f.get("description","") if f["type"]=="Active site" else "BIND")
    return out
cols={}
for acc,tag in [("P37351","Ec"),("P9WKD7","Mt")]:
    for p,k in feats(acc).items(): cols.setdefault(col_of(acc,p),[]).append(f"{tag}{A[acc].replace('-','')[p-1]}{p}({'/'.join(sorted(k))})")
# also Coccidioides catalytic Cys76 (3SGW)
cols.setdefault(col_of("P0CL19",76),[]).append("CiC76(ACT:Proton acceptor)")
cols=dict(sorted(cols.items()))
groups={"labeled":[i for i in A if str(meta.get(i,"")).startswith("labeled")],
        "same_slot_PF02502":[i for i in A if meta.get(i)=="carrier_same_slot_as_labeled"],
        "short_other_slot":[i for i in A if str(meta.get(i,"")).startswith("carrier_other_slot")]}
rows=[]
for c,lbl in cols.items():
    rec={"col":c,"reference_residues":"; ".join(lbl)}
    for r in ["P37351","P9WKD7","P0CL19","Q9ZB26","J0MXJ0","P23494","P23495","Q92IL8"]: rec[r]=A[r][c]
    for g,ids in groups.items():
        cnt=Counter(A[i][c] for i in ids); n=len(ids)
        rec[g]=", ".join(f"{a}:{v/n:.2f}" for a,v in cnt.most_common(3))
    rows.append(rec)
T=pd.DataFrame(rows); T.to_csv(D+"/residue_table.tsv",sep="\t",index=False)
pd.set_option("display.width",300); pd.set_option("display.max_colwidth",70); pd.set_option("display.max_rows",100)
print(T.to_string(index=False))
ps=pd.DataFrame({i:{f"c{c}":A[i][c] for c in cols} for i in A}).T; ps["group"]=ps.index.map(meta); ps.to_csv(D+"/residues_per_seq.tsv",sep="\t")
