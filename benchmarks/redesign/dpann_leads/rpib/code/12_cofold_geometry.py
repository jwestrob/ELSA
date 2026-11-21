#!/usr/bin/env python3
"""Co-fold geometry: ligand confidence (mean B-factor = pLDDT x 100 of R5P atoms), distance from each R5P centroid to the
centroid of the RpiB site positions (chain X: Mt11, Mt70, Mt75, Mt113 equivalents; partner chain: Mt102, Mt137, Mt141
equivalents, mapped through the monomer US-align alignment), and the interface check: CA distance between chain A Mt70
equivalent and chain B Mt102 equivalent, against the same distance in 2VVP (A G70 / B H102). Writes data/cofold_geometry.tsv."""
import pandas as pd, numpy as np, re, sys
sys.path.insert(0,"/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/code")
from importlib import import_module
site=import_module("08_ref_site"); sr=import_module("11_struct_residues") if False else None
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"
U=pd.read_csv(D+"/usalign_out.tsv",sep="\t")
ref=site.atoms(D+"/pdb/2VVP_A.pdb"); rr=ref[(ref.rec=="ATOM")].drop_duplicates("num").num.tolist()
def q2mt_map(pid):
    if pid=="P9WKD7": return None
    r=U[(U.kind=="mono_vs_2VVP_A")&(U.a.str.contains(re.escape(pid)+r"\.pdb"))].iloc[0]
    qi=ri=0; m={}
    for x,y in zip(r.aln_a,r.aln_b):
        if x!="-": qi+=1
        if y!="-": ri+=1
        if x!="-" and y!="-": m[rr[ri-1]]=qi
    return m
def ca(a,ch,num):
    x=a[(a.rec=="ATOM")&(a.ch==ch)&(a.num==num)&(a.name=="CA")]
    return x[["x","y","z"]].to_numpy()[0] if len(x) else None
rows=[]
r2=site.atoms(D+"/pdb/2VVP_AB.pdb"); d_ref=np.linalg.norm(ca(r2,"A",70)-ca(r2,"B",102))
for f in sorted(U[U.kind=="dimer_vs_2VVP_AB"].a):
    pid=f.split("/")[-1].replace("__dimer_R5P.pdb",""); a=site.atoms(R+"/folds/dimers/"+f.split("/")[-1])
    m=q2mt_map(pid); mt=(lambda n:n) if m is None else (lambda n:m.get(n))
    own=[mt(n) for n in (11,70,75,113)]; par=[mt(n) for n in (102,137,141)]
    for lc in sorted(a[a.res=="R5P"].ch.unique()):
        L=a[(a.res=="R5P")&(a.ch==lc)]; cen=L[["x","y","z"]].to_numpy().mean(0); lp=L.x.size and float(np.mean([0]))
        best=None
        for X,Y in (("A","B"),("B","A")):
            pts=[ca(a,X,n) for n in own if n]+[ca(a,Y,n) for n in par if n]; pts=[p for p in pts if p is not None]
            if not pts: continue
            dd=np.linalg.norm(cen-np.mean(pts,0))
            if best is None or dd<best[0]: best=(dd,X,len(pts))
        rows.append(dict(protein_id=pid,ligand=lc,ligand_atoms=len(L),dist_to_site_centroid=round(best[0],1) if best else None,site_chain=best[1] if best else None,site_points=best[2] if best else 0))
    d70_102=[np.linalg.norm(ca(a,"A",mt(70))-ca(a,"B",mt(102))) if mt(70) and mt(102) and ca(a,"A",mt(70)) is not None and ca(a,"B",mt(102)) is not None else None]
    for r in rows:
        if r["protein_id"]==pid: r["CA_A70_B102"]=None if d70_102[0] is None else round(float(d70_102[0]),1); r["CA_A70_B102_in_2VVP"]=round(float(d_ref),1)
# ligand pLDDT from B-factor column
def lig_b(path):
    v={}
    for l in open(path):
        if l.startswith("HETATM") and l[17:20]=="R5P": v.setdefault(l[21],[]).append(float(l[60:66]))
    return {k:round(np.mean(x),1) for k,x in v.items()}
G=pd.DataFrame(rows)
G["ligand_pLDDTx100"]=[lig_b(R+"/folds/dimers/"+p+"__dimer_R5P.pdb").get(l) for p,l in zip(G.protein_id,G.ligand)]
G.to_csv(D+"/cofold_geometry.tsv",sep="\t",index=False); pd.set_option("display.width",250); print(G.to_string(index=False))
