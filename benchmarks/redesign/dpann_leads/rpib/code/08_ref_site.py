#!/usr/bin/env python3
"""Active site of the experimental structure 2VVP (RPIB_MYCTU with ribose 5-phosphate R5P and ribulose 5-phosphate 5RP):
residues within 4.0 A of each ligand copy, by chain, for the dimer A/B (biological unit 1). Shows whether the site is
built from one chain or two. Writes data/ref_site_2VVP.tsv."""
import numpy as np, pandas as pd, sys
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
def atoms(p):
    out=[]
    for l in open(p):
        if l.startswith(("ATOM","HETATM")):
            out.append(dict(rec=l[:6].strip(),name=l[12:16].strip(),res=l[17:20].strip(),ch=l[21],num=int(l[22:26]),x=float(l[30:38]),y=float(l[38:46]),z=float(l[46:54])))
    return pd.DataFrame(out)
def site(pdb, lig_res, chains, cut=4.0):
    a=atoms(pdb); P=a[(a.rec=="ATOM")&a.ch.isin(chains)]; rows=[]
    for (lch,lnum),L in a[(a.res==lig_res)&a.ch.isin(chains)].groupby(["ch","num"]):
        d=np.sqrt(((P[["x","y","z"]].to_numpy()[:,None,:]-L[["x","y","z"]].to_numpy()[None,:,:])**2).sum(-1)).min(1)
        near=P[d<=cut].assign(dist=d[d<=cut]).groupby(["ch","num","res"]).dist.min().reset_index()
        for r in near.itertuples(): rows.append(dict(pdb=pdb.split("/")[-1],ligand=f"{lig_res}_{lch}{lnum}",chain=r.ch,resnum=r.num,res=r.res,min_dist=round(r.dist,2)))
    return pd.DataFrame(rows)
if __name__=="__main__":
    T=pd.concat([site(D+"/pdb/2VVP.pdb","R5P",["A","B"]),site(D+"/pdb/2VVP.pdb","5RP",["A","B"])])
    T.to_csv(D+"/ref_site_2VVP.tsv",sep="\t",index=False)
    for lg,g in T.groupby("ligand"):
        print(lg, "chains contributing:", g.groupby("chain").size().to_dict(), "|", " ".join(f"{r.chain}:{r.res}{r.resnum}({r.min_dist})" for r in g.itertuples()))
