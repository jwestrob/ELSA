#!/usr/bin/env python3
"""Q2 ortholog vs paralog. Annotate the supporters (the partner genes in other genomes at the matching slot) of the 287
labeled genes and of the 411 annotated PF02502 carriers in Micrarchaeota. Writes data/supporters_annot.tsv,
data/carriers_class.tsv and prints the summaries used in REPORT.txt."""
import duckdb, pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_rows",300); pd.set_option("display.max_colwidth",90); pd.set_option("display.max_columns",30)
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
con=duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t")
S=pd.read_csv(D+"/rpib_labeled_supporters.tsv",sep="\t"); C=pd.read_csv(D+"/rpib_carriers_supporters.tsv",sep="\t")
carl=pd.read_csv(D+"/carriers_micr_pf02502.txt",header=None)[0]
allp=pd.DataFrame({"protein_id":pd.unique(pd.concat([S.partner_protein_id,C.partner_protein_id,C["query"],lab.protein_id,carl]))})
con.register("allp",allp)
A=con.execute("""SELECT p.protein_id, pr.bin_id, pr.contig_id, pr.gene_index, pr.sequence_length len, split_part(b.taxonomy,';',2) phylum, split_part(b.taxonomy,';',4) ord, b.completeness comp,
  (SELECT string_agg(DISTINCT a.name, ',') FROM d.annotations a WHERE a.protein_id=p.protein_id AND a.source='pfam') pfam,
  (SELECT min(a.evalue) FROM d.annotations a WHERE a.protein_id=p.protein_id AND a.source='pfam' AND a.name='LacAB_rpiB') pf02502_e,
  (SELECT string_agg(DISTINCT a.accession, ',') FROM d.annotations a WHERE a.protein_id=p.protein_id AND a.source IN ('kofam','kegg')) ko,
  (SELECT arg_min(r."name", r.evalue) FROM d.protein_resemblance r WHERE r.protein_id=p.protein_id) cdd
  FROM allp p JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id""").df()
A["is_labeled"]=A.protein_id.isin(set(lab.protein_id))
A["has_pf02502"]=A.pf02502_e.notna()
S2=S.merge(A,left_on="partner_protein_id",right_on="protein_id",how="left")
S2.to_csv(D+"/supporters_annot.tsv",sep="\t",index=False)
print("== supporters of the 287 labeled genes:", len(S2), "rows;", S2.partner_protein_id.nunique(), "distinct partner genes in", S2.bin_id.nunique(), "genomes")
u=S2.drop_duplicates("partner_protein_id")
print(" distinct partners by phylum:", u.phylum.value_counts().to_dict())
print(" distinct partners by order (Micrarchaeota):", u[u.phylum=="p__Micrarchaeota"].ord.value_counts().to_dict())
def cls(r):
    if r.has_pf02502: return "PF02502_GA"
    if r.is_labeled: return "labeled_dark(287)"
    if isinstance(r.pfam,str): return "other_pfam:"+r.pfam
    if isinstance(r.ko,str): return "ko_only:"+r.ko
    return "unannotated_other"
u=u.assign(c=u.apply(cls,axis=1))
print(" distinct partner classes:", u.c.value_counts().head(15).to_dict())
print(" partner length by class:", u.groupby("c").len.median().head(10).to_dict())
print(" PF02502 partners KO:", u[u.has_pf02502].ko.fillna("none").value_counts().head(8).to_dict())
print(" PF02502 partners CDD:", u[u.has_pf02502].cdd.fillna("none").value_counts().head(8).to_dict())
print(" PF02502 partners length quantiles:", u[u.has_pf02502].len.quantile([.05,.25,.5,.75,.95]).tolist())
print(" PF02502 partners e-value quantiles:", u[u.has_pf02502].pf02502_e.quantile([.05,.5,.95]).tolist())
# per query: share of supporters in Micrarchaeales, share PF02502, share labeled
S2["c"]=S2.apply(cls,axis=1)
pq=S2.groupby("query").agg(n=("rank","size"), micr=("phylum",lambda x:(x=="p__Micrarchaeota").mean()), pf=("has_pf02502","mean"), labd=("is_labeled","mean"),
    other=("c",lambda x: x.str.startswith(("other_pfam","ko_only")).mean()))
print(" per-query medians: supporters", pq.n.median(), "| Micrarchaeota share", pq.micr.median(), "| PF02502 share", pq.pf.median(), "| labeled-dark share", pq.labd.median(), "| other annotated share", pq.other.median())
print(" queries where >=90% of supporters are Micrarchaeota:", int((pq.micr>=0.9).sum()), "of", len(pq))
# carriers: is a Micrarchaeota PF02502 carrier at the same slot as the labeled genes (supporters include labeled genes) ?
C2=C.merge(A[["protein_id","is_labeled","has_pf02502","phylum","ord"]],left_on="partner_protein_id",right_on="protein_id",how="left")
cs=C2.groupby("query").agg(n=("rank","size"), n_lab=("is_labeled","sum"), n_pf=("has_pf02502","sum")).reset_index()
car=A[A.protein_id.isin(set(C["query"])|set(pd.read_csv(D+"/carriers_micr_pf02502.txt",header=None)[0]))][["protein_id","bin_id","contig_id","gene_index","len","ord","ko","cdd","pf02502_e","comp"]]
car=car.merge(cs,left_on="protein_id",right_on="query",how="left").drop(columns="query").fillna({"n":0,"n_lab":0,"n_pf":0})
# positional ortholog of the labeled slot = carrier whose supporters include >= 3 labeled genes, or a carrier that is itself a supporter of labeled genes
supp_of_lab=S2[S2.has_pf02502].groupby("partner_protein_id").size()
car["supports_n_labeled"]=car.protein_id.map(supp_of_lab).fillna(0).astype(int)
car["slot"]=np.where((car.n_lab>=3)|(car.supports_n_labeled>=3),"same_slot_as_labeled",np.where(car.n>0,"other_slot_bonded","other_slot_unbonded"))
car.to_csv(D+"/carriers_class.tsv",sep="\t",index=False)
print("== 411 Micrarchaeota PF02502 carriers: slot class", car.slot.value_counts().to_dict())
print(car.groupby("slot").agg(n=("protein_id","size"), genomes=("bin_id","nunique"), med_len=("len","median"), q10=("len",lambda x:x.quantile(.1)), q90=("len",lambda x:x.quantile(.9)),
      k01808=("ko",lambda x:x.fillna("").str.contains("K01808").sum()), micrarchaeales=("ord",lambda x:(x=="o__Micrarchaeales").sum()), med_e=("pf02502_e","median")).to_string())
print(" carriers by order x slot:"); print(pd.crosstab(car.ord, car.slot).to_string())
print(" CDD of carriers by slot:"); print(car.groupby("slot").cdd.apply(lambda x:x.fillna("none").value_counts().head(5).to_dict()).to_string())
