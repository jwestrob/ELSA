#!/usr/bin/env python3
"""Q3 gap test. Per Micrarchaeales genome: any annotated R5P isomerase (K01807, K01808, Pfam Rib_5-P_isom_A, canonical
short PF02502), Ru5P sources (K00033 6PGD; RPE K01783 with transketolase K00615), R5P sinks (K00948 PRPS) and sources
(K01515 nudF, K00852 rbsK), split by long-form carriage, completeness and family. Writes data/gap_test.tsv."""
import duckdb, pandas as pd, numpy as np
from math import comb
pd.set_option("display.width",250)
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
B=pd.read_csv(D+"/genome_groups.tsv",sep="\t"); mi=B[B.ord=="o__Micrarchaeales"].copy()
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
K=["K01807","K01808","K00033","K01783","K00615","K00616","K00948","K01515","K00852","K01601","K18931","K18237","K00036"]
GK=con.execute("""SELECT DISTINCT pr.bin_id, a.accession ko FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source IN ('kofam','kegg') AND a.accession IN (SELECT unnest(?))""",[K]).df()
GP=con.execute("""SELECT DISTINCT pr.bin_id FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam' AND a.name='Rib_5-P_isom_A'""").df()
for k in K: mi[k]=mi.bin_id.isin(set(GK[GK.ko==k].bin_id))
mi["RpiA_pfam"]=mi.bin_id.isin(set(GP.bin_id))
mi["any_R5P_isomerase"]=mi.K01807|mi.K01808|mi.RpiA_pfam|mi.canonical_PF02502
mi["Ru5P_source"]=mi.K00033|(mi.K01783&mi.K00615)
mi["R5P_sink_PRPS"]=mi.K00948
mi["R5P_source"]=mi.K01515|mi.K00852
mi.to_csv(D+"/gap_test.tsv",sep="\t",index=False)
def fisher(a,b,c,d):  # two-sided Fisher exact on [[a,b],[c,d]]
    n=a+b+c+d; r1=a+b; c1=a+c
    p=lambda x: comb(r1,x)*comb(n-r1,c1-x)/comb(n,c1)
    p0=p(a); return min(1.0,sum(p(x) for x in range(max(0,c1-(n-r1)),min(r1,c1)+1) if p(x)<=p0*(1+1e-9)))
cols=["any_R5P_isomerase","K01808","canonical_PF02502","Ru5P_source","K00033","K01783","K00615","K00616","R5P_sink_PRPS","R5P_source","K01601","K18931"]
for lo in (70,90):
    x=mi[mi.comp>=lo]; a=x[x.LF_any]; b=x[~x.LF_any]
    print(f"\n== Micrarchaeales completeness >= {lo}: LF+ {len(a)} | LF- {len(b)}")
    for c in cols:
        p=fisher(int(a[c].sum()),len(a)-int(a[c].sum()),int(b[c].sum()),len(b)-int(b[c].sum()))
        print(f"  {c:20s} LF+ {int(a[c].sum()):4d}/{len(a)} ({a[c].mean():.3f})   LF- {int(b[c].sum()):4d}/{len(b)} ({b[c].mean():.3f})   Fisher p={p:.2g}")
print("\n== within family, completeness >= 70")
for fam,x in mi[mi.comp>=70].groupby("fam"):
    if len(x)<30: continue
    a=x[x.LF_any]; b=x[~x.LF_any]
    print(f" {fam}: LF+ {len(a)} LF- {len(b)} | " + " | ".join(f"{c} {a[c].mean():.2f}/{b[c].mean():.2f}" for c in ["any_R5P_isomerase","K00615","K01783","K00033","R5P_sink_PRPS"]))
lab=mi[mi.LF_labeled]
print(f"\n== 287 labeled genomes: any other R5P isomerase {int(lab.any_R5P_isomerase.sum())}; none {int((~lab.any_R5P_isomerase).sum())} "
      f"(comp>=90: none {int((~lab.any_R5P_isomerase&(lab.comp>=90)).sum())} of {int((lab.comp>=90).sum())})")
g=lab[~lab.any_R5P_isomerase]
print(f"   in the {len(g)} with no other isomerase: Ru5P source {int(g.Ru5P_source.sum())}, R5P sink PRPS {int(g.R5P_sink_PRPS.sum())}, both {int((g.Ru5P_source&g.R5P_sink_PRPS).sum())}, "
      f"R5P source (nudF/rbsK) {int(g.R5P_source.sum())}, 6PGD {int(g.K00033.sum())}, RPE+TKT {int((g.K01783&g.K00615).sum())}, G6PDH {int(g.K00036.sum())}")
