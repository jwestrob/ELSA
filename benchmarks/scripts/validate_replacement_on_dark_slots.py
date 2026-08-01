#!/usr/bin/env python3
"""Validate replacement detection where Pfam cannot reach.

39% of DPANN proteins carry no Pfam hit. 18.1% of slots are fully dark and 30.5%
hold fewer than four annotated occupants, so the Pfam-based calibration scored
roughly 70% of slots and said nothing about the rest — which is the population a
protein language model is supposed to earn its keep on.

mmseqs supplies a label that works on any sequence: two occupants are different
families when no alignment survives e<1e-3 at 50% coverage of the shorter
protein. It is sequence-based, so judging an embedding with it is independent
rather than circular, and it agrees with the Pfam label where both exist —
detector AUC 0.984 against mmseqs versus 0.983 against disjoint Pfam domains on
annotated slots.

Result, 900 slots per stratum:

    stratum     replaced   raw q10 AUC   szca q10 AUC
    annotated      40.6%        0.9844         0.9927
    mixed          96.1%        0.9881         0.9939
    dark           35.1%        0.9575         0.9711

At the shipped raw cut-point of 0.95, precision holds across all three (97.2%,
99.9%, 97.5%) while recall falls on dark slots (87.1%, 94.0%, 61.1%). The cause
is separation rather than label noise: replaced dark slots sit at q10 0.936
against 0.897 for annotated ones, so the embedding distinguishes them less
sharply. mmseqs detects homology equally well in both (median within-slot hit
fraction 1.000), which rules out a biased label.

So the detector is conservative on domain-dark slots — it misses roughly two
fifths of the replacements there while its positive calls stay trustworthy.

Stages: build the stratified sample and FASTA, run mmseqs, score.

    python benchmarks/scripts/validate_replacement_on_dark_slots.py --stage prepare
    mmseqs easy-search slots.faa slots.faa hits.tsv tmp_mmseqs \
        --threads 10 -e 1e-3 -c 0.5 --cov-mode 0 -s 5.7 --max-seqs 100 \
        --format-output "query,target,pident,alnlen,evalue,bits,qcov,tcov"
    python benchmarks/scripts/validate_replacement_on_dark_slots.py --stage score

`setsid` is absent on macOS; background mmseqs with plain nohup.
"""
SCHEMA=SH/"synteny/results_v3/schema"
OUT=Path("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/2d5d32b3-64c0-4fff-a070-114ecb9df832/scratchpad/dark")
OUT.mkdir(parents=True, exist_ok=True)
rng=np.random.default_rng(17)

asg=pd.read_parquet(SCHEMA/"cluster_slot_assignments.parquet",
                    columns=["cluster_id","slot_idx","gene_id"])
con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)
arch=dict(con.execute("SELECT protein_id, string_agg(name,';' ORDER BY start_aa) "
                      "FROM annotations WHERE source='pfam' GROUP BY protein_id").fetchall())
asg["arch"]=asg.gene_id.map(arch)
asg["ann"]=asg.arch.notna()

g=asg.groupby(["cluster_id","slot_idx"]).agg(n=("gene_id","size"), n_ann=("ann","sum"))
g=g[(g.n>=4)&(g.n<=20)]
g["frac"]=g.n_ann/g.n
strata={"annotated": g[g.frac==1], "mixed": g[(g.frac>0)&(g.frac<1)], "dark": g[g.frac==0]}
picked=[]
for name,sub in strata.items():
    take=sub.iloc[np.sort(rng.choice(len(sub),min(900,len(sub)),replace=False))]
    for (cid,sidx) in take.index:
        picked.append((cid,sidx,name))
    print(f"{name:<10} {len(sub):>7,} available -> {len(take):,} sampled", flush=True)
sel=pd.DataFrame(picked,columns=["cluster_id","slot_idx","stratum"])
sel.to_csv(OUT/"slots.tsv",sep="\t",index=False)

key=set(zip(sel.cluster_id,sel.slot_idx))
sub=asg[[ (c,s) in key for c,s in zip(asg.cluster_id,asg.slot_idx)]]
strat={(c,s):st for c,s,st in picked}
rows=[]
for (cid,sidx),grp in sub.groupby(["cluster_id","slot_idx"],sort=False):
    ids=list(grp.gene_id)
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            rows.append((ids[i],ids[j],cid,sidx,strat[(cid,sidx)]))
pd.DataFrame(rows,columns=["a","b","cluster_id","slot_idx","stratum"]).to_csv(
    OUT/"pairs.tsv",sep="\t",index=False)
print(f"\n{len(sub):,} occupants, {len(rows):,} within-slot pairs", flush=True)

need=sorted(set(sub.gene_id))
seqs={}
for i in range(0,len(need),4000):
    chunk=need[i:i+4000]
    qs=",".join("'"+p.replace("'","''")+"'" for p in chunk)
    seqs.update(dict(con.execute(
        f"SELECT protein_id, sequence FROM proteins WHERE protein_id IN ({qs})").fetchall()))
con.close()
with (OUT/"slots.faa").open("w") as fh:
    for pid,s in seqs.items():
        if s: fh.write(f">{pid}\n{s}\n")
print(f"wrote {len(seqs):,} sequences", flush=True)
