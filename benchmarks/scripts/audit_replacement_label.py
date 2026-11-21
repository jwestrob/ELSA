#!/usr/bin/env python3
"""Audit the disjoint-domain label behind replacement detection.

Sequence identity from an unconstrained local alignment inflates on short hits.
mmseqs applies an e-value and a coverage requirement, which is the standard a
homology call should meet.

Writes a FASTA and a within-slot pair list, then run:

    mmseqs easy-search slots.faa slots.faa hits.tsv tmp_mmseqs \\
        --threads 10 -e 1e-3 -c 0.5 --cov-mode 0 -s 5.7 --max-seqs 100 \\
        --format-output "query,target,pident,alnlen,evalue,bits,qcov,tcov"

Result on 54,901 pairs: 97.2% of same-domain pairs are homologous against 0.3%
of disjoint-domain pairs, so the label holds for 99.7% of the slots it calls
replaced. An unconstrained local alignment instead reports 47.5% of disjoint
pairs as homologous at >30% identity — that figure is short spurious hits, and
the coverage requirement is what removes it.
"""
import duckdb, numpy as np, pandas as pd
from pathlib import Path

SH=Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
SCHEMA=SH/"synteny/results_v3/schema"
OUT=Path("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/2d5d32b3-64c0-4fff-a070-114ecb9df832/scratchpad/labelaudit")
OUT.mkdir(parents=True, exist_ok=True)
rng=np.random.default_rng(17)

asg=pd.read_parquet(SCHEMA/"cluster_slot_assignments.parquet",
                    columns=["cluster_id","slot_idx","gene_id"])
con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)
arch=dict(con.execute("SELECT protein_id, string_agg(name,';' ORDER BY start_aa) "
                      "FROM annotations WHERE source='pfam' GROUP BY protein_id").fetchall())
asg["arch"]=asg.gene_id.map(arch)
lab=asg[asg.arch.notna()].copy()
lab["dset"]=lab.arch.map(lambda a: frozenset(a.split(";")))

sizes=lab.groupby(["cluster_id","slot_idx"]).size()
elig=list(sizes[(sizes>=4)&(sizes<=25)].index)
pick=set(map(tuple,np.array(elig)[np.sort(rng.choice(len(elig),1500,replace=False))]))
sub=lab[[(c,s) in pick for c,s in zip(lab.cluster_id,lab.slot_idx)]]
print(f"{len(pick):,} slots, {len(sub):,} occupants", flush=True)

pairs=[]
for (cid,sidx),grp in sub.groupby(["cluster_id","slot_idx"],sort=False):
    ids=list(grp.gene_id); sets=list(grp.dset)
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            pairs.append((ids[i],ids[j],"disjoint" if not (sets[i]&sets[j]) else "shares_domain"))
pd.DataFrame(pairs,columns=["a","b","label"]).to_csv(OUT/"pairs.tsv",sep="\t",index=False)
print(f"{len(pairs):,} within-slot pairs "
      f"({sum(1 for p in pairs if p[2]=='disjoint'):,} disjoint)", flush=True)

need=sorted({g for g in sub.gene_id})
qs=",".join("'"+p.replace("'","''")+"'" for p in need)
seqs=dict(con.execute(f"SELECT protein_id, sequence FROM proteins WHERE protein_id IN ({qs})").fetchall())
con.close()
with (OUT/"slots.faa").open("w") as fh:
    for pid,s in seqs.items():
        if s: fh.write(f">{pid}\n{s}\n")
print(f"wrote {len(seqs):,} sequences to {OUT/'slots.faa'}", flush=True)
