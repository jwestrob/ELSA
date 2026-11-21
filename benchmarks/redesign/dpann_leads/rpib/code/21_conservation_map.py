#!/usr/bin/env python3
"""Frame-critic test: is the old substrate pocket conserved in LF? Column conservation (1 - normalized Shannon entropy over
the 452 LF sequences, gaps counted as a symbol) mapped onto each LF monomer model; pocket = residues within 6 A of the
transplanted crystal R5P (data/mats); background = 2,000 random surface patches of the same size (k nearest residues by
CA around a random residue). Empirical p = share of patches with mean conservation >= the pocket. Also: distance from the
conserved C-terminal motif Lys of each chain to the transplanted R5P in the LF dimer models. And the side check: canonical
isomerase presence among LF+ / LF- genomes that carry transketolase. Writes data/conservation_pocket.tsv."""
import numpy as np, pandas as pd, sys, re, glob, os
from collections import Counter
sys.path.insert(0,"/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/code")
from importlib import import_module
site=import_module("08_ref_site"); tp=import_module("17_transplant") if False else None
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"
def readfa(p):
    s={};k=None
    for l in open(p):
        l=l.strip()
        if l.startswith(">"): k=l[1:].split()[0]; s[k]=""
        elif k: s[k]+=l
    return s
A=readfa(D+"/msa_aln.faa"); meta=pd.read_csv(D+"/msa_meta.tsv",sep="\t").set_index("id").group
lf=[i for i in A if str(meta[i]).startswith("labeled") or meta[i]=="carrier_same_slot_as_labeled"]
L=len(A[lf[0]]); cons=np.zeros(L)
for c in range(L):
    cnt=np.array(list(Counter(A[i][c] for i in lf).values()),float); p=cnt/cnt.sum()
    cons[c]=1-(-(p*np.log(p)).sum()/np.log(21))
def res_cons(pid):  # conservation per residue index (1-based) of sequence pid
    out={}; n=0
    for c,ch in enumerate(A[pid]):
        if ch!="-": n+=1; out[n]=cons[c]
    return out
ref=site.atoms(D+"/pdb/2VVP.pdb"); LIG=ref[(ref.res=="R5P")&(ref.ch=="A")&(ref.num==200)][["x","y","z"]].to_numpy()
def mat(p):
    l=[x for x in open(p).read().splitlines() if x.strip()[:1] in "012" and len(x.split())==5]
    return np.array([float(x.split()[1]) for x in l]), np.array([[float(v) for v in x.split()[2:]] for x in l])
fs=pd.read_csv(D+"/fold_set.tsv",sep="\t"); rng=np.random.default_rng(17); rows=[]
for pid in fs[fs.role.str.startswith("labeled")|fs.role.str.startswith("pos_same")].protein_id:
    a=site.atoms(f"{R}/folds/monomers/{pid}.pdb"); ca=a[(a.rec=="ATOM")&(a.name=="CA")].reset_index(drop=True)
    t,U=mat(f"{D}/mats/mono_{pid}.txt"); X=LIG@U.T+t
    P=a[a.rec=="ATOM"]; d=np.sqrt(((P[["x","y","z"]].to_numpy()[:,None,:]-X[None])**2).sum(-1)).min(1)
    pocket=sorted(P[d<=6.0].num.unique()); rc=res_cons(pid); k=len(pocket)
    xyz=ca[["x","y","z"]].to_numpy(); nums=ca.num.to_numpy()
    # surface proxy: CA neighbor count within 10 A below the median (exposed); core region only (exclude the C-terminal tail past the last pocket-adjacent residue + 0)
    nb=(np.sqrt(((xyz[:,None]-xyz[None])**2).sum(-1))<10).sum(1); surf=np.flatnonzero(nb<=np.median(nb))
    pm=np.mean([rc[n] for n in pocket]); bg=[]
    for _ in range(2000):
        c0=rng.choice(surf); dd=np.sqrt(((xyz-xyz[c0])**2).sum(1)); patch=nums[np.argsort(dd)[:k]]; bg.append(np.mean([rc[n] for n in patch]))
    bg=np.array(bg); rows.append(dict(protein_id=pid,pocket_size=k,pocket_cons=round(pm,3),bg_median=round(np.median(bg),3),bg_q90=round(np.quantile(bg,.9),3),p_emp=round((bg>=pm).mean(),3),
        top_patch_pctile=round((bg<pm).mean(),3),seq_mean=round(np.mean(list(rc.values())),3)))
T=pd.DataFrame(rows); T.to_csv(D+"/conservation_pocket.tsv",sep="\t",index=False); pd.set_option("display.width",200); print(T.to_string(index=False))
print("pocket in top decile of surface patches:", int((T.p_emp<=0.10).sum()), "of", len(T), "| median p_emp", T.p_emp.median())
# conserved tail Lys vs transplanted pocket in the LF dimers
for f in sorted(glob.glob(f"{R}/folds/dimers/*__dimer_R5P.pdb")):
    pid=os.path.basename(f).replace("__dimer_R5P.pdb","")
    if pid in ("P9WKD7","JAJYYB010000017.1_66"): continue
    s=A[pid].replace("-",""); m=list(re.finditer(r"K...G[ILVMF]",s[-16:]))
    kpos=(len(s)-16+m[-1].start()+1) if m else None   # motif Lys: K-x3-G[ILVMF] in the last 16 residues
    if kpos is None:  # fall back: the column of the 96%-conserved motif Lys (alignment column 490)
        n=0
        for c,ch in enumerate(A[pid]):
            if ch!="-": n+=1
            if c==490: kpos=n if ch=="K" else None; break
    a=site.atoms(f); t,U=mat(f"{D}/mats/dimer_{pid}__dimer_R5P.txt"); X=LIG@U.T+t
    for ch in "AB":
        z=a[(a.rec=="ATOM")&(a.ch==ch)&(a.num==kpos)&(a.name=="NZ")] if kpos else pd.DataFrame()
        dist=None if z.empty else round(float(np.sqrt(((X-z[["x","y","z"]].to_numpy()[0])**2).sum(1)).min()),1)
        print(f"  {pid} chain {ch}: motif Lys{kpos} NZ to transplanted R5P {dist} A")
# side check
g=pd.read_csv(D+"/gap_test.tsv",sep="\t"); x=g[(g.comp>=70)&g.K00615]
for lf_,y in x.groupby("LF_any"): print(f"  transketolase genomes (comp>=70), LF {'+' if lf_ else '-'}: any R5P isomerase {int(y.any_R5P_isomerase.sum())}/{len(y)}")
x=g[(g.comp>=70)&g.K00615&g.K01783]
for lf_,y in x.groupby("LF_any"): print(f"  transketolase+RPE genomes (comp>=70), LF {'+' if lf_ else '-'}: any R5P isomerase {int(y.any_R5P_isomerase.sum())}/{len(y)}")
