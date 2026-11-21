#!/usr/bin/env python3
"""Anomaly audit: the most conserved alignment columns across the 452 long-form sequences (287 labeled + 165 same-slot
annotated), with the RPIB_ECOLI residue at the same column, and the C-terminal motif frequency. Writes data/lf_conserved_columns.tsv."""
import pandas as pd, re
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
lf=[i for i in A if str(meta[i]).startswith("labeled") or meta[i]=="carrier_same_slot_as_labeled"]
ec=A["P37351"]; ecn=[]; n=0
for c in ec:
    if c!="-": n+=1
    ecn.append(n if c!="-" else None)
rows=[]
for c in range(len(ec)):
    col=[A[i][c] for i in lf]; cnt=Counter(col); aa,v=cnt.most_common(1)[0]
    if aa=="-": continue
    rows.append(dict(col=c,top=aa,frac=round(v/len(lf),3),gap=round(cnt["-"]/len(lf),3),ecoli=(ec[c]+str(ecn[c])) if ecn[c] else "-",mt=A["P9WKD7"][c]))
T=pd.DataFrame(rows); T.to_csv(D+"/lf_conserved_columns.tsv",sep="\t",index=False)
print(T[T.frac>=0.7].to_string(index=False))
tails=[A[i].replace("-","")[-15:] for i in lf]
m=sum(bool(re.search(r"G[ILVMF].G..K...G[ILVM]",t)) for t in tails); print("C-terminal motif G[ILVMF]xGxxKxxxG[ILVM] in last 15 aa:", m, "of", len(lf))
print(Counter(t[-12:] for t in tails).most_common(8))
