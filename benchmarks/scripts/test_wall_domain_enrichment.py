"""Is cell-wall-active domain content enriched in Nanobdellota secreted clusters?

SLT hits appear across the large secreted candidates, which is suggestive for an
episymbiont whose host has a wall it does not. Suggestive is not evidence, so
this asks whether such domains are enriched against the rest of DPANN, and what
share of the clusters they actually cover.

Domains are OBSERVED Pfam hits. No system is named: a lytic-transglycosylase
domain is a domain, and no curated caller has asserted a mechanism here.
"""
import duckdb, numpy as np, pandas as pd, collections, re
from pathlib import Path
SH=Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
RES=SH/"synteny/results_v6"

# Pfam families that act on, or bind, a bacterial cell envelope.
WALL = {"SLT","SLT_2","Transglycosylase","Glucosaminidase","CHAP","Amidase_2",
        "Amidase_3","Peptidase_M23","NLPC_P60","PG_binding_1","PG_binding_2",
        "LysM","Lysozyme","Glyco_hydro_25","Muramidase","Hydrolase_2","Peptidase_M15"}

asg=pd.read_parquet(RES/"schema/cluster_slot_assignments.parquet",
                    columns=["cluster_id","gene_id","genome_id"])
con=duckdb.connect(str(SH/"sharur.duckdb"),read_only=True)
tax=dict(con.execute("SELECT bin_id,taxonomy FROM bins").fetchall())
def ph(g):
    m=re.search(r"p__([^;]*)",str(tax.get(g,""))); return m.group(1) if m and m.group(1) else None
asg["phylum"]=asg.genome_id.map({g:ph(g) for g in set(asg.genome_id)})

wall_ids=set()
qs=",".join("'"+d+"'" for d in WALL)
for pid, in con.execute(f"SELECT DISTINCT protein_id FROM annotations "
                        f"WHERE source='pfam' AND name IN ({qs})").fetchall():
    wall_ids.add(pid)
total_prot=con.execute("SELECT count(*) FROM proteins").fetchone()[0]
print(f"proteins carrying a wall-active Pfam domain: {len(wall_ids):,} of {total_prot:,} "
      f"({100*len(wall_ids)/total_prot:.3f}%)")

# per-phylum rate
byph=collections.Counter(); tot=collections.Counter()
for g,p in ((g,ph(g)) for g in tax):
    if p: tot[p]+=0
rows=con.execute("SELECT p.protein_id, p.bin_id FROM proteins p").fetchall()
for pid,binid in rows:
    p=ph(binid)
    if not p: continue
    tot[p]+=1
    if pid in wall_ids: byph[p]+=1
print("\nwall-active domain rate by phylum:")
for p,_ in sorted(tot.items(), key=lambda x:-x[1])[:6]:
    if tot[p]: print(f"  {p:<20} {byph[p]:>5} / {tot[p]:>9,}  ({100*byph[p]/tot[p]:.3f}%)")

CANDS=[3913,5379,207,1096,1258,1043]
base=len(wall_ids)/total_prot
print(f"\n=== enrichment in the candidate clusters (background {100*base:.3f}%) ===")
print(f"{'cluster':>9} {'proteins':>9} {'wall-domain':>12} {'rate':>8} {'enrichment':>11} {'genomes w/ >=1':>15}")
for cid in CANDS:
    s=asg[asg.cluster_id==cid].drop_duplicates("gene_id")
    ids=set(s.gene_id); hit=ids & wall_ids
    genomes_with=s.genome_id[s.gene_id.isin(hit)].nunique()
    rate=len(hit)/len(ids)
    print(f"{cid:>9} {len(ids):>9,} {len(hit):>12,} {100*rate:>7.2f}% "
          f"{rate/base:>10.1f}x {genomes_with:>8,}/{s.genome_id.nunique():,}")
con.close()
