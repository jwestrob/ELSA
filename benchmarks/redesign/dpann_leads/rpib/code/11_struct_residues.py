#!/usr/bin/env python3
"""Rung 3/4 by structure: residues of each folded monomer at the positions that US-align superposes on the 2VVP chain A
active-site residues (RPIB_MYCTU numbering, curated in P9WKD7) and on the E. coli catalytic Cys66 (1NN4 chain A, P37351
numbering). Then the co-folds: residues within 4.0 A of each R5P copy, by chain, translated to RPIB_MYCTU numbering through
the monomer alignment. Writes data/struct_site_residues.tsv and data/cofold_contacts.tsv."""
import pandas as pd, numpy as np, re, sys
sys.path.insert(0,"/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/code")
from importlib import import_module
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"
site=import_module("08_ref_site")
AA1=dict(ALA="A",ARG="R",ASN="N",ASP="D",CYS="C",GLN="Q",GLU="E",GLY="G",HIS="H",ILE="I",LEU="L",LYS="K",MET="M",PHE="F",PRO="P",SER="S",THR="T",TRP="W",TYR="Y",VAL="V")
def resnums(pdb,ch="A"):
    a=site.atoms(pdb); a=a[(a.rec=="ATOM")&(a.ch==ch)]
    return a.drop_duplicates("num")[["num","res"]].values.tolist()
U=pd.read_csv(D+"/usalign_out.tsv",sep="\t"); fs=pd.read_csv(D+"/fold_set.tsv",sep="\t")
ref_res={"2VVP_A":resnums(D+"/pdb/2VVP_A.pdb"),"1NN4_A":resnums(D+"/pdb/1NN4_A.pdb")}
# 1NN4 numbering vs P37351: find offset by matching the sequence
s1nn4="".join(AA1.get(r,"X") for _,r in ref_res["1NN4_A"])
import json; ec=json.load(open(D+"/ref/P37351.json"))["sequence"]["value"]
off=ec.find(s1nn4[:15])-s1nn4.find(s1nn4[:15]); first=ref_res["1NN4_A"][0][0]
ec_num=lambda pdbnum: None
MT={"D11":11,"H12":12,"Y46":46,"G70":70,"S71":71,"G72":72,"G74":74,"E75":75,"H102":102,"N103":103,"R113":113,"R137":137,"H138":138,"R141":141}
def mapping(row,refkey):
    """query residue index (1-based in model) aligned to each reference residue number, with ':' (d<5A) flag"""
    qa,mm,ra=row.aln_a,row.aln_m,row.aln_b; qi=ri=0; out={}
    rr=ref_res[refkey]
    for x,m,y in zip(qa,mm,ra):
        if x!="-": qi+=1
        if y!="-": ri+=1
        if x!="-" and y!="-": out[rr[ri-1][0]]=(qi,x,m)
    return out
rows=[]
for r in U[U.kind=="mono_vs_2VVP_A"].itertuples():
    pid=r.a.split("/")[-1][:-4]; mp=mapping(r,"2VVP_A"); rec={"protein_id":pid,"role":fs.set_index("protein_id").role.get(pid,"")}
    for k,n in MT.items():
        q=mp.get(n); rec[k]=(q[1]+str(q[0])+("" if q[2]==":" else "~")) if q else "-"
    rows.append(rec)
S=pd.DataFrame(rows)
# E. coli Cys66 through 1NN4
for r in U[U.kind=="mono_vs_1NN4_A"].itertuples():
    pid=r.a.split("/")[-1][:-4]; mp=mapping(r,"1NN4_A")
    pdbnum=[66]  # 1NN4 numbering = P37351 numbering (DBREF 1-149)
    q=mp.get(pdbnum[0]) if pdbnum else None
    S.loc[S.protein_id==pid,"EcC66"]=(q[1]+str(q[0])+("" if q[2]==":" else "~")) if q else "-"
    pdbnum=[99]; q=mp.get(pdbnum[0])
    S.loc[S.protein_id==pid,"EcH99"]=(q[1]+str(q[0])+("" if q[2]==":" else "~")) if q else "-"
S.to_csv(D+"/struct_site_residues.tsv",sep="\t",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print("1NN4 check: residue 66 =", [res for n,res in ref_res["1NN4_A"] if n==66], "; 99 =", [res for n,res in ref_res["1NN4_A"] if n==99])
print(S.to_string(index=False))
# catalytic signature by structure
lfS=S[S.role.str.startswith("labeled")|S.role.str.startswith("pos_same")]
def has(col,aa): return lfS[col].str[0].eq(aa)
print("long-form models (n=%d): E75->E %d, H102->H %d, EcC66->C %d, EcH99->H %d, D11->D %d, H12->H %d, R113->R %d, R137->R %d, R141->R %d" % (len(lfS),has("E75","E").sum(),has("H102","H").sum(),has("EcC66","C").sum(),has("EcH99","H").sum(),has("D11","D").sum(),has("H12","H").sum(),has("R113","R").sum(),has("R137","R").sum(),has("R141","R").sum()))
# co-fold ligand contacts
C=[]
for r in U[U.kind=="dimer_vs_2VVP_AB"].itertuples():
    pdb=R+"/folds/dimers/"+r.a.split("/")[-1]; pid=r.a.split("/")[-1].replace("__dimer_R5P.pdb","")
    a=site.atoms(pdb); ligch=sorted(a[a.res=="R5P"].ch.unique())
    # monomer mapping query residue -> RPIB_MYCTU number, from this protein's monomer alignment to 2VVP_A (MtRpiB: identity)
    if pid=="P9WKD7": q2mt=lambda i:i
    else:
        mr=U[(U.kind=="mono_vs_2VVP_A")&(U.a.str.contains(re.escape(pid)+r"\.pdb"))].iloc[0]; mp=mapping(mr,"2VVP_A"); inv={v[0]:k for k,v in mp.items()}
        q2mt=lambda i,inv=inv: inv.get(i)
    for lc in ligch:
        L=a[(a.res=="R5P")&(a.ch==lc)]; P=a[(a.rec=="ATOM")&a.ch.isin(["A","B"])&~a.res.eq("R5P")]
        d=np.sqrt(((P[["x","y","z"]].to_numpy()[:,None,:]-L[["x","y","z"]].to_numpy()[None,:,:])**2).sum(-1)).min(1)
        near=P[d<=4.0].assign(dist=d[d<=4.0]).groupby(["ch","num","res"]).dist.min().reset_index()
        plddt=float(L.shape[0])
        C.append(dict(protein_id=pid,ligand_chain=lc,n_contacts=len(near),chains=",".join(f"{c}:{n}" for c,n in near.groupby("ch").size().items()),
            residues=" ".join(f"{x.ch}:{AA1.get(x.res,'X')}{x.num}(Mt{q2mt(x.num)})" for x in near.itertuples())))
C=pd.DataFrame(C); C.to_csv(D+"/cofold_contacts.tsv",sep="\t",index=False)
pd.set_option("display.max_colwidth",400); print(C.to_string(index=False))
