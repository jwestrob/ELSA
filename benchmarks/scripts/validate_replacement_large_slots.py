"""Do large slots genuinely hold non-homologous occupants?

The q10 cut-point was calibrated on slots of 4-25 occupants. Firing rises to 93%
above 100 occupants, which is either real heterogeneity across a wider taxonomic
span or a statistic that drifts with n. Subsampling each large slot to a fixed
20 occupants makes it comparable to the calibrated regime.
"""
import duckdb, numpy as np, pandas as pd
from pathlib import Path
SH=Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
SCHEMA=SH/"synteny/results_v3/schema_v4"
OUT=Path("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/2d5d32b3-64c0-4fff-a070-114ecb9df832/scratchpad/large")
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(17)

slots=pd.read_parquet(SCHEMA/"cluster_slots.parquet",
                      columns=["cluster_id","slot_idx","n_occupants","has_alternate"])
big=slots[slots.n_occupants>100]
take=big.iloc[np.sort(rng.choice(len(big),min(400,len(big)),replace=False))]
print(f"{len(big):,} large slots; sampling {len(take):,}", flush=True)

asg=pd.read_parquet(SH/"synteny/results_v3/schema_v4/cluster_slot_assignments.parquet",
                    columns=["cluster_id","slot_idx","gene_id"])
key=set(zip(take.cluster_id,take.slot_idx))
sub=asg[[(c,s) in key for c,s in zip(asg.cluster_id,asg.slot_idx)]]
rows=[]
for (cid,sidx),grp in sub.groupby(["cluster_id","slot_idx"],sort=False):
    ids=list(grp.gene_id)
    if len(ids)>20:
        ids=[ids[i] for i in rng.choice(len(ids),20,replace=False)]
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            rows.append((ids[i],ids[j],cid,sidx))
pairs=pd.DataFrame(rows,columns=["a","b","cluster_id","slot_idx"])
pairs.to_csv(OUT/"pairs.tsv",sep="\t",index=False)
need=sorted(set(pairs.a)|set(pairs.b))
print(f"{len(need):,} occupants subsampled, {len(pairs):,} pairs", flush=True)

con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)
seqs={}
for i in range(0,len(need),4000):
    ch=need[i:i+4000]; qs=",".join("'"+p.replace("'","''")+"'" for p in ch)
    seqs.update(dict(con.execute(
        f"SELECT protein_id, sequence FROM proteins WHERE protein_id IN ({qs})").fetchall()))
con.close()
with (OUT/"slots.faa").open("w") as fh:
    for pid,s in seqs.items():
        if s: fh.write(f">{pid}\n{s}\n")
take.to_csv(OUT/"slots.tsv",sep="\t",index=False)
print(f"wrote {len(seqs):,} sequences", flush=True)
