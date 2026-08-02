"""What sits beside the cluster:1043 cassette, more often than chance?

Guilt by association needs a null. Abundant housekeeping genes turn up beside
everything, so raw counts in the flanks say little. The comparison here is
against windows of the same size drawn at random positions on the same contigs,
which controls for both genome composition and contig gene density.
"""
import duckdb, numpy as np, pandas as pd, collections
from pathlib import Path

SH=Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
RES=SH/"synteny/results_v6"
FLANK=10
rng=np.random.default_rng(17)

asg=pd.read_parquet(RES/"schema/cluster_slot_assignments.parquet",
                    columns=["cluster_id","slot_idx","gene_id","genome_id","contig_id"])
s=asg[asg.cluster_id==1043].drop_duplicates("gene_id")
con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)

ids=list(s.gene_id)
gi={}
for i in range(0,len(ids),4000):
    q=",".join("'"+x.replace("'","''")+"'" for x in ids[i:i+4000])
    for pid,ci,g in con.execute(
        f"SELECT protein_id, contig_id, gene_index FROM proteins WHERE protein_id IN ({q})").fetchall():
        gi[pid]=(ci,g)
s["contig"]=[gi.get(x,(None,None))[0] for x in s.gene_id]
s["gidx"]=[gi.get(x,(None,None))[1] for x in s.gene_id]
s=s.dropna(subset=["gidx"])

loci=[]
for (g,c),grp in s.groupby(["genome_id","contig"]):
    if grp.slot_idx.nunique()<3: continue
    loci.append((g,c,int(grp.gidx.min()),int(grp.gidx.max())))
print(f"complete loci: {len(loci):,}", flush=True)

contigs=sorted({c for _,c,_,_ in loci})
maxidx={}
for i in range(0,len(contigs),2000):
    q=",".join("'"+x.replace("'","''")+"'" for x in contigs[i:i+2000])
    for ci,mx in con.execute(
        f"SELECT contig_id, max(gene_index) FROM proteins WHERE contig_id IN ({q}) GROUP BY contig_id").fetchall():
        maxidx[ci]=mx

def window_ids(c, lo, hi):
    return (c, max(0, lo-FLANK), hi+FLANK)

obs_w=[window_ids(c,lo,hi) for _,c,lo,hi in loci]
# null: same contig, same window width, random placement, avoiding the real locus
null_w=[]
for (_,c,lo,hi) in loci:
    w=(hi+FLANK)-(lo-FLANK)
    mx=maxidx.get(c,hi+FLANK)
    if mx<=w: continue
    for _ in range(3):
        st=int(rng.integers(0, max(1,mx-w)))
        if abs(st-(lo-FLANK))>w:
            null_w.append((c,st,st+w)); break

def annot_counts(windows, label):
    per_source=collections.Counter(); loci_with=collections.defaultdict(set)
    B=400
    for i in range(0,len(windows),B):
        chunk=windows[i:i+B]
        clause=" OR ".join([f"(p.contig_id='{c}' AND p.gene_index BETWEEN {a} AND {b})"
                            for c,a,b in chunk])
        rows=con.execute(
            f"SELECT p.contig_id, p.gene_index, a.source, a.name FROM proteins p "
            f"JOIN annotations a ON a.protein_id=p.protein_id WHERE {clause}").fetchall()
        for ci,g,src,nm in rows:
            for j,(c,a,b) in enumerate(chunk):
                if c==ci and a<=g<=b:
                    loci_with[(src,nm)].add(i+j); break
        print(f"  {label}: {min(i+B,len(windows))}/{len(windows)}", end="\r", flush=True)
    print()
    return {k:len(v) for k,v in loci_with.items()}

obs=annot_counts(obs_w,"observed")
nul=annot_counts(null_w,"null")
con.close()

no, nn = len(obs_w), len(null_w)
rows=[]
for k,v in obs.items():
    if v<30: continue
    e=nul.get(k,0)
    fo, fe = v/no, max(e/nn, 0.5/nn)
    rows.append({"source":k[0],"name":k[1],"loci":v,"pct_loci":round(100*fo,1),
                 "null_pct":round(100*e/nn,2),"enrichment":round(fo/fe,1)})
d=pd.DataFrame(rows).sort_values("enrichment",ascending=False)
print(f"\nobserved windows {no:,} | null windows {nn:,} | flank +/-{FLANK} genes\n")
print("=== ENRICHED BESIDE THE CASSETTE ===")
print(d.head(22).to_string(index=False))
d.to_csv("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/2d5d32b3-64c0-4fff-a070-114ecb9df832/scratchpad/neighbourhood.tsv",sep="\t",index=False)
