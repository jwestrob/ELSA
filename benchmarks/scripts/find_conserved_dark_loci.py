#!/usr/bin/env python3
"""Gene neighbourhoods conserved across DPANN that carry no annotation.

This is the case ELSA can reach and an HMM scan cannot: a locus held in the
same arrangement across hundreds of genomes whose genes have no Pfam hit. A
domain scan sees nothing; synteny sees a conserved cassette.

Everything reported here is OBSERVED — conservation, breadth, taxonomy, and the
presence or absence of a curated system call. Nothing is named. The curated
callers actually populated in this database are DefenseFinder, TXSScan and
HydDB; bgc_loci and locus_proteins are empty, so no BGC call is available.
"""
import duckdb, numpy as np, pandas as pd
from pathlib import Path

SH=Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
RES=SH/"synteny/results_v6"
OUT=Path("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/2d5d32b3-64c0-4fff-a070-114ecb9df832/scratchpad")

asg=pd.read_parquet(RES/"schema/cluster_slot_assignments.parquet",
                    columns=["cluster_id","slot_idx","gene_id","genome_id"])
clus=pd.read_csv(RES/"micro_chain_clusters.csv",usecols=["cluster_id","size","genome_support"])
print(f"{len(asg):,} slot assignments across {asg.cluster_id.nunique():,} clusters", flush=True)

con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)
annotated=set(x[0] for x in con.execute(
    "SELECT DISTINCT protein_id FROM annotations WHERE source='pfam'").fetchall())
# curated system callers, taken from what is actually populated
sysprot=set(x[0] for x in con.execute("SELECT DISTINCT protein_id FROM system_proteins").fetchall())
called=set(x[0] for x in con.execute(
    "SELECT DISTINCT protein_id FROM annotations "
    "WHERE source IN ('defensefinder_system','txsscan_system','hyddb')").fetchall())
tax=dict(con.execute("SELECT bin_id, taxonomy FROM bins").fetchall())
con.close()

asg["ann"]=asg.gene_id.isin(annotated)
asg["curated"]=asg.gene_id.isin(sysprot|called)
g=asg.groupby("cluster_id").agg(
    n_genes=("gene_id","size"), n_slots=("slot_idx","nunique"),
    n_genomes=("genome_id","nunique"),
    frac_annotated=("ann","mean"), n_curated=("curated","sum")).reset_index()
g=g.merge(clus,on="cluster_id",how="left")
g["frac_dark"]=1-g.frac_annotated

# conserved: present in many genomes; dark: almost nothing carries a Pfam hit
cand=g[(g.n_genomes>=50)&(g.frac_dark>=0.9)&(g.n_slots>=3)].copy()
cand=cand.sort_values(["n_genomes","n_slots"],ascending=False)
print(f"\nclusters with >=50 genomes, >=3 slots, >=90% dark: {len(cand):,}")
print(f"  of these, carrying any curated system call: {int((cand.n_curated>0).sum()):,}")

top=cand.head(15).copy()
def clade(cid, rank="p"):
    import re
    gs=asg.genome_id[asg.cluster_id==cid].unique()
    v=[re.search(rf"{rank}__([^;]*)",str(tax.get(x,""))) for x in gs]
    v=[m.group(1) for m in v if m and m.group(1)]
    return pd.Series(v).value_counts()
rows=[]
for r in top.itertuples():
    vc=clade(r.cluster_id)
    rows.append({"cluster_id":r.cluster_id,"genomes":r.n_genomes,"slots":r.n_slots,
                 "genes":r.n_genes,"dark":round(r.frac_dark,3),"curated":int(r.n_curated),
                 "phyla":len(vc),"dominant":f"{vc.index[0]} {100*vc.iloc[0]/vc.sum():.0f}%" if len(vc) else "?"})
res=pd.DataFrame(rows)
print("\n=== CONSERVED, UNANNOTATED NEIGHBOURHOODS ===")
print(res.to_string(index=False))
res.to_csv(OUT/"dark_conserved.tsv",sep="\t",index=False)
g.to_csv(OUT/"cluster_annotation_profile.tsv",sep="\t",index=False)
