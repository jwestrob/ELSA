#!/usr/bin/env python3
"""Transplant the crystallographic R5P (2VVP, ligand A200, which contacts chain A D11/H12/G70-E75/R113 and chain B
H102/N103/R137/R141) into each model frame with the US-align matrix (2VVP chain A -> model chain A), then list the model
residues within 4.5 A of the transplanted ligand: what occupies the substrate site in each model.
Writes data/transplant_site.tsv."""
import numpy as np, pandas as pd, glob, os, sys
sys.path.insert(0,"/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/code")
from importlib import import_module
site=import_module("08_ref_site")
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"
AA1=dict(ALA="A",ARG="R",ASN="N",ASP="D",CYS="C",GLN="Q",GLU="E",GLY="G",HIS="H",ILE="I",LEU="L",LYS="K",MET="M",PHE="F",PRO="P",SER="S",THR="T",TRP="W",TYR="Y",VAL="V")
ref=site.atoms(D+"/pdb/2VVP.pdb"); L=ref[(ref.res=="R5P")&(ref.ch=="A")&(ref.num==200)][["x","y","z"]].to_numpy()
def mat(p):
    l=[x for x in open(p).read().splitlines() if x.strip()[:1] in "012" and len(x.split())==5]
    t=np.array([float(x.split()[1]) for x in l]); U=np.array([[float(v) for v in x.split()[2:]] for x in l]); return t,U
fs=pd.read_csv(D+"/fold_set.tsv",sep="\t").set_index("protein_id").role
rows=[]
for m in sorted(glob.glob(D+"/mats/*.txt")):
    b=os.path.basename(m)[:-4]; kind,pid=b.split("_",1)
    t,U=mat(m); X=L@U.T+t
    pdb=f"{R}/folds/{'monomers' if kind=='mono' else 'dimers'}/{pid}.pdb"; a=site.atoms(pdb); P=a[(a.rec=="ATOM")]
    d=np.sqrt(((P[["x","y","z"]].to_numpy()[:,None,:]-X[None])**2).sum(-1)).min(1)
    near=P[d<=4.5].assign(dist=d[d<=4.5]).groupby(["ch","num","res"]).dist.min().reset_index()
    polar=near[near.res.isin(["HIS","CYS","GLU","ASP","ARG","LYS"])]
    rows.append(dict(model=b,role=fs.get(pid.replace("__dimer_R5P",""),""),n_res=len(near),chains=",".join(f"{c}:{n}" for c,n in near.groupby("ch").size().items()),
                     HCEDRK=" ".join(f"{x.ch}:{AA1[x.res]}{x.num}" for x in polar.itertuples()),residues=" ".join(f"{x.ch}:{AA1[x.res]}{x.num}" for x in near.itertuples())))
T=pd.DataFrame(rows); T.to_csv(D+"/transplant_site.tsv",sep="\t",index=False)
pd.set_option("display.width",300); pd.set_option("display.max_colwidth",160); print(T.to_string(index=False))
