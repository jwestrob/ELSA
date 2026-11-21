#!/usr/bin/env python3
"""Per-sequence calls of the catalytic signatures (from data/msa_aln.faa, columns located through the curated positions of
P37351 / P9WKD7), with a +-2 column window to tolerate local alignment shifts; and where the extra length sits relative
to RPIB_ECOLI (N-terminal / core / C-terminal). Writes data/residue_calls.tsv."""
import pandas as pd, numpy as np, re
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
def readfa(p):
    s={};k=None
    for l in open(p):
        l=l.strip()
        if l.startswith(">"): k=l[1:].split()[0]; s[k]=""
        elif k: s[k]+=l
    return s
A=readfa(D+"/msa_aln.faa"); meta=pd.read_csv(D+"/msa_meta.tsv",sep="\t").set_index("id").group
def col(seqid,pos):
    n=0
    for i,c in enumerate(A[seqid]):
        if c!="-":
            n+=1
            if n==pos: return i
E="P37351"; M="P9WKD7"
sites={"D9":col(E,9),"H10":col(E,10),"C66":col(E,66),"G69":col(E,69),"E75mt":col(M,75),"H99":col(E,99),"N100":col(E,100),"R110":col(E,110),"R133":col(E,133),"R137":col(E,137)}
want={"D9":"D","H10":"H","C66":"C","G69":"G","E75mt":"E","H99":"H","N100":"N","R110":"R","R133":"R","R137":"R"}
ec_first=col(E,1); ec_last=col(E,149)
rows=[]
for i,s in A.items():
    g=str(meta.get(i,""))
    r={"id":i,"group":g}
    for k,c in sites.items():
        r[k]=s[c]; r[k+"_win2"]=want[k] in s[max(0,c-2):c+3]
    seq=s.replace("-","")
    r["len"]=len(seq); r["n_before_ec1"]=len(s[:ec_first].replace("-","")); r["n_after_ec149"]=len(s[ec_last+1:].replace("-",""))
    r["n_core"]=r["len"]-r["n_before_ec1"]-r["n_after_ec149"]
    tail=seq[-60:]; r["tail60_KEQRDP"]=sum(tail.count(a) for a in "KEQRDP")/max(1,len(tail))
    r["ct_motif"]=bool(re.search(r"G[ILVMF][FMVLIW]G[KSARNQT][LIVMF][KR][DENQ]..G[ILVM]",seq[-25:]))
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv(D+"/residue_calls.tsv",sep="\t",index=False)
R["grp"]=np.where(R.group.str.startswith("labeled"),"labeled_287",np.where(R.group=="carrier_same_slot_as_labeled","same_slot_PF02502_165",np.where(R.group.str.startswith("carrier_other"),"other_slot_PF02502_246","ref")))
pd.set_option("display.width",250)
R["Ec_type(C66+H99)"]=(R.C66=="C")&(R.H99=="H"); R["Mt_type(E75+H102)"]=(R.E75mt=="E")&(R.H99=="H")
R["any_base_win2"]=R.C66_win2|R.E75mt_win2; R["acid_H_win2"]=R.H99_win2
R["P_at_G69"]=R.G69=="P"
agg=R.groupby("grp").agg(n=("id","size"),EcType=("Ec_type(C66+H99)","mean"),MtType=("Mt_type(E75+H102)","mean"),C66=("C66",lambda x:(x=="C").mean()),E75=("E75mt",lambda x:(x=="E").mean()),
    H99=("H99",lambda x:(x=="H").mean()),H10=("H10",lambda x:(x=="H").mean()),D9=("D9",lambda x:(x=="D").mean()),R110=("R110",lambda x:(x=="R").mean()),R133=("R133",lambda x:(x=="R").mean()),R137=("R137",lambda x:(x=="R").mean()),
    base_win2=("any_base_win2","mean"),H99_win2=("acid_H_win2","mean"),P69=("P_at_G69","mean"),len=("len","median"),core=("n_core","median"),cterm=("n_after_ec149","median"),nterm=("n_before_ec1","median"),
    tailKEQRDP=("tail60_KEQRDP","median"),ctmotif=("ct_motif","mean"))
print(agg.round(2).T.to_string())
print(R[R.grp=="ref"][["id","D9","H10","C66","G69","E75mt","H99","N100","R110","R133","R137","len","n_after_ec149"]].to_string(index=False))
# other-slot carriers split by length (long-form members that were not bonded)
o=R[R.grp=="other_slot_PF02502_246"]; print("other-slot carriers with ct_motif:", int(o.ct_motif.sum()), "of", len(o), "| their median len", o[o.ct_motif].len.median(), "| without motif median len", o[~o.ct_motif].len.median(), "| EcType among without motif", round(o[~o.ct_motif]["Ec_type(C66+H99)"].mean(),2))
# where the extra length sits: residues before / after the column of RPIB_ECOLI residue 118 (end of the helix after the
# H99-N100 loop; the C-terminal 31 residues of P37351 carry R133 / R137 and are dispersed by mafft over the long tails)
for anc in (100,118):
    c=col(E,anc)
    R[f"before_ec{anc}"]=[len(A[i][:c+1].replace("-","")) for i in R.id]; R[f"after_ec{anc}"]=R.len-R[f"before_ec{anc}"]
print(R.groupby("grp")[["len","before_ec100","after_ec100","before_ec118","after_ec118"]].median().to_string())
print("E. coli residue 118 =", A[E].replace("-","")[117], "; P37351 residues 119-149:", A[E].replace("-","")[118:])
R.to_csv(D+"/residue_calls.tsv",sep="\t",index=False)
