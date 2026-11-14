"""Conserved Nanobdellota clusters with cell-surface sequence signatures.

DPANN are episymbionts and their surface machinery is where the interesting
biology sits, but almost none of it carries a domain call. Sequence features
that are computable without annotation stand in: a signal-peptide-like
N-terminus, transmembrane segments, and length.

Everything here is OBSERVED. A signal-peptide-like N-terminus is a hydrophobic
stretch in the first 30 residues, not a SignalP call; a transmembrane segment is
a hydrophobic window, not a topology. No system is named.
"""
import duckdb, numpy as np, pandas as pd, collections, re
from pathlib import Path
SH=Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
RES=SH/"synteny/results_v6"
OUT=Path("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/2d5d32b3-64c0-4fff-a070-114ecb9df832/scratchpad")

KD=dict(zip("AVLIPFMWGSTCYNQDEKRH",
            [1.8,4.2,3.8,4.5,-1.6,2.8,1.9,-0.9,-0.4,-0.8,-0.7,2.5,-1.3,-3.5,-3.5,-3.5,-3.5,-3.9,-4.5,-3.2]))

def hydro(seq):
    return np.array([KD.get(c,0.0) for c in seq], dtype=np.float32)

def tm_segments(h, win=19, cut=1.6):
    """Count non-overlapping windows whose mean hydrophobicity clears `cut`."""
    if len(h)<win: return 0
    k=np.convolve(h, np.ones(win)/win, mode="valid")
    n=0; i=0
    while i < len(k):
        if k[i]>=cut: n+=1; i+=win
        else: i+=1
    return n

def signal_like(seq, h):
    """Hydrophobic h-region inside the first 30 residues, after a short n-region."""
    if len(seq)<40: return False
    win=np.convolve(h[:30], np.ones(9)/9, mode="valid") if len(h)>=9 else np.array([])
    return bool(win.size and win.max()>=2.0)

asg=pd.read_parquet(RES/"schema/cluster_slot_assignments.parquet",
                    columns=["cluster_id","locus_id","gene_id","genome_id"])
con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)
tax=dict(con.execute("SELECT bin_id,taxonomy FROM bins").fetchall())
pf=set(x[0] for x in con.execute("SELECT DISTINCT protein_id FROM annotations WHERE source='pfam'").fetchall())
anyann=set(x[0] for x in con.execute("SELECT DISTINCT protein_id FROM annotations").fetchall())

def phylum(g):
    m=re.search(r"p__([^;]*)",str(tax.get(g,""))); return m.group(1) if m and m.group(1) else None
asg["phylum"]=asg.genome_id.map({g:phylum(g) for g in set(asg.genome_id)})

nb=asg[asg.phylum=="Nanobdellota"]
size=collections.Counter(p for g,p in ((g,phylum(g)) for g in tax) if p)
g=asg.groupby("cluster_id").agg(genomes=("genome_id","nunique"), genes=("gene_id","size")).reset_index()
nbg=nb.groupby("cluster_id").genome_id.nunique().reset_index(name="nb_genomes")
g=g.merge(nbg,on="cluster_id",how="inner")
g["nb_frac_of_cluster"]=g.nb_genomes/g.genomes
g["nb_prevalence"]=g.nb_genomes/size["Nanobdellota"]
cand=g[(g.nb_genomes>=150)&(g.nb_frac_of_cluster>=0.7)].copy()
print(f"Nanobdellota-dominated clusters (>=150 NB genomes, >=70% of cluster): {len(cand):,}", flush=True)

ids=list(asg.gene_id[asg.cluster_id.isin(set(cand.cluster_id))].unique())
print(f"scoring {len(ids):,} proteins...", flush=True)
seqs={}
for i in range(0,len(ids),4000):
    q=",".join("'"+x.replace("'","''")+"'" for x in ids[i:i+4000])
    seqs.update(dict(con.execute(
        f"SELECT protein_id,sequence FROM proteins WHERE protein_id IN ({q})").fetchall()))
con.close()

feat={}
for pid,sq in seqs.items():
    if not sq: continue
    s=sq.rstrip("*")
    h=hydro(s)
    feat[pid]=(len(s), tm_segments(h), signal_like(s,h), float(h.mean()))

rows=[]
for cid,sub in asg[asg.cluster_id.isin(set(cand.cluster_id))].groupby("cluster_id"):
    ids2=[x for x in sub.gene_id.unique() if x in feat]
    if len(ids2)<30: continue
    L=np.array([feat[x][0] for x in ids2]); TM=np.array([feat[x][1] for x in ids2])
    SP=np.array([feat[x][2] for x in ids2]); KDm=np.array([feat[x][3] for x in ids2])
    r=cand[cand.cluster_id==cid].iloc[0]
    rows.append({"cluster":cid,"nb_genomes":int(r.nb_genomes),
                 "nb_prev":round(float(r.nb_prevalence),3),
                 "n_prot":len(ids2),"med_len":int(np.median(L)),
                 "frac_signal":round(float(SP.mean()),3),
                 "med_tm":int(np.median(TM)),"frac_multitm":round(float((TM>=3).mean()),3),
                 "med_kd":round(float(np.median(KDm)),2),
                 "frac_pfam":round(float(np.mean([x in pf for x in ids2])),3),
                 "frac_anyann":round(float(np.mean([x in anyann for x in ids2])),3)})
d=pd.DataFrame(rows)
d.to_csv(OUT/"nanobdellota_surface_scan.tsv",sep="\t",index=False)

print("\n=== SECRETED/SURFACE-LIKE: signal-peptide-like N-terminus, few TM, dark ===")
sec=d[(d.frac_signal>=0.5)&(d.med_tm<=1)&(d.frac_pfam<=0.25)].sort_values("nb_prev",ascending=False)
print(sec.head(12).to_string(index=False))
print("\n=== POLYTOPIC MEMBRANE: many TM segments, dark ===")
mem=d[(d.frac_multitm>=0.5)&(d.frac_pfam<=0.25)].sort_values("nb_prev",ascending=False)
print(mem.head(12).to_string(index=False))
