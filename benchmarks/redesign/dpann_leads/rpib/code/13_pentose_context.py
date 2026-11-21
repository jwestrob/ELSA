#!/usr/bin/env python3
"""Q3 / rung 5-6: which routes to and from pentose phosphates are annotated in Micrarchaeota. The KO set is built from
KEGG (downloaded by accession into data/kegg/): every reaction linked to R5P C00117, Ru5P C00199, X5P C00231, PRPP C00119,
R1P C00620, R15P C01151, RuBP C01182, dR5P C00673, dR1P C00672, ribose C00121, 6-phosphogluconate C00345, F6P C00085,
L-Ru5P C01101, hexulose-6P C06019, E4P C00279; every KO linked to those reactions. Genome presence from the Sharur
annotations (sources kofam and kegg), split by long-form (LF) carriage, with completeness strata.
Writes data/genome_groups.tsv, data/pentose_ko_presence.tsv."""
import duckdb, pandas as pd, numpy as np, glob, os
pd.set_option("display.width",260); pd.set_option("display.max_rows",400); pd.set_option("display.max_colwidth",60)
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"; K=D+"/kegg"
CN={"C00117":"R5P","C00199":"Ru5P","C00231":"X5P","C00119":"PRPP","C00620":"R1P","C01151":"R15P","C01182":"RuBP","C00673":"dR5P","C00672":"dR1P","C00121":"ribose","C00345":"6PG","C00085":"F6P","C01101":"L-Ru5P","C06019":"Hu6P","C00279":"E4P"}
c2r=pd.concat([pd.read_csv(f,sep="\t",header=None,names=["c","rn"]).assign(c=os.path.basename(f)[4:10]) for f in glob.glob(K+"/rxn_C*.txt") if os.path.getsize(f)>0])
r2k=pd.read_csv(K+"/rn_ko.txt",sep="\t",header=None,names=["rn","ko"]); r2k["ko"]=r2k.ko.str.replace("ko:",""); 
kon=pd.read_csv(K+"/ko_names.txt",sep="\t",header=None,names=["ko","nm"]); kon["ko"]=kon.ko.str.replace("ko:","")
m=c2r.merge(r2k,on="rn"); m["cn"]=m.c.map(CN)
kc=m.groupby("ko").agg(compounds=("cn",lambda x:",".join(sorted(set(x)))),reactions=("rn",lambda x:",".join(sorted(set(x.str.replace("rn:",""))))) ).reset_index().merge(kon,on="ko",how="left")
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
B=con.execute("SELECT bin_id, completeness comp, contamination cont, split_part(taxonomy,';',2) phylum, split_part(taxonomy,';',4) ord, split_part(taxonomy,';',5) fam FROM d.bins WHERE taxonomy LIKE '%p__Micrarchaeota%'").df()
# LF carriage per genome
rc=pd.read_csv(D+"/residue_calls.tsv",sep="\t"); rc=rc[~rc.group.str.startswith("ref")]
rc["LF"]=(rc.len>=175)&~(rc.H99_win2&(rc.C66=="C"))
lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t")
gid=con.execute("SELECT protein_id id, bin_id FROM d.proteins WHERE protein_id IN (SELECT unnest(?))",[rc.id.tolist()]).df()
rc=rc.merge(gid,on="id")
B["LF_labeled"]=B.bin_id.isin(set(lab.bin_id))
B["LF_annotated"]=B.bin_id.isin(set(rc[rc.LF&~rc.group.str.startswith("labeled")].bin_id))
B["LF_any"]=B.LF_labeled|B.LF_annotated
B["canonical_PF02502"]=B.bin_id.isin(set(rc[~rc.LF&~rc.group.str.startswith("labeled")].bin_id))
B.to_csv(D+"/genome_groups.tsv",sep="\t",index=False)
GK=con.execute("""SELECT DISTINCT pr.bin_id, a.accession ko FROM d.annotations a JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id
   WHERE a.source IN ('kofam','kegg') AND b.taxonomy LIKE '%p__Micrarchaeota%' AND a.accession IN (SELECT unnest(?))""",[kc.ko.tolist()]).df()
GP=con.execute("""SELECT DISTINCT pr.bin_id, a.name pf FROM d.annotations a JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id
   WHERE a.source='pfam' AND b.taxonomy LIKE '%p__Micrarchaeota%'""").df()
print("Micrarchaeota genomes", len(B), "| by order", B.ord.value_counts().head(4).to_dict())
mi=B[B.ord=="o__Micrarchaeales"]
print("Micrarchaeales", len(mi), "| LF labeled", int(mi.LF_labeled.sum()), "| LF annotated", int(mi.LF_annotated.sum()), "| LF any", int(mi.LF_any.sum()), "| canonical PF02502", int(mi.canonical_PF02502.sum()))
print("LF any outside Micrarchaeales:", int(B[B.ord!="o__Micrarchaeales"].LF_any.sum()), "of", int((B.ord!="o__Micrarchaeales").sum()))
for lo in (0,70,90):
    x=mi[mi.comp>=lo]; print(f"  Micrarchaeales comp>={lo}: n={len(x)} LF_any={x.LF_any.mean():.3f} ({int(x.LF_any.sum())})  canonical={x.canonical_PF02502.mean():.3f}")
print("  completeness median LF_any vs not:", mi[mi.LF_any].comp.median(), mi[~mi.LF_any].comp.median())
present=GK.groupby("ko").bin_id.apply(set).to_dict()
rows=[]
grp={"Micrarchaeota_all":B,"Micrarchaeales_all":mi,"Micrarchaeales_LF+_comp70":mi[mi.LF_any&(mi.comp>=70)],"Micrarchaeales_LF-_comp70":mi[~mi.LF_any&(mi.comp>=70)],
     "labeled287":B[B.LF_labeled],"Anstonellales":B[B.ord=="o__Anstonellales"],"Norongarragalinales":B[B.ord=="o__Norongarragalinales"]}
for r in kc.itertuples():
    s=present.get(r.ko,set())
    if not s: continue
    rec=dict(ko=r.ko,name=str(r.nm)[:55],compounds=r.compounds)
    for g,x in grp.items(): rec[g]=int(x.bin_id.isin(s).sum())
    rows.append(rec)
T=pd.DataFrame(rows).sort_values("Micrarchaeota_all",ascending=False)
den={g:len(x) for g,x in grp.items()}
print("denominators:", den)
T.to_csv(D+"/pentose_ko_presence.tsv",sep="\t",index=False)
print(T.to_string(index=False))
absent=kc[~kc.ko.isin(T.ko)]
print("\nKEGG pentose-linked KOs with zero Micrarchaeota genomes:", len(absent), "of", len(kc))
for k in ["K01807","K01808","K01783","K00615","K00616","K13810","K08093","K08094","K13812","K13831","K00948","K00852","K01839","K18931","K18237","K01601","K00036","K00033","K01057","K03783","K00772","K00757","K00758","K01619","K22199","K01241","K01515","K00759","K00760","K00761","K00762","K00764","K00765","K00766","K00767","K00763","K14455"]:
    if k in set(absent.ko): print("   absent:",k,absent.set_index("ko").nm.get(k,"")[:70])
# non-homologous / Pfam-level checks
for pf in ["Rib_5-P_isom_A","LacAB_rpiB","Ribul_P_3_epim","Transketolase_N","Transketolase_C","Transketolase_C_2","TAL_FSA","Pribosyltran_N","Pribosyl_synth","PNP_UDP_1","Glycos_trans_3N","Glycos_transf_3","RuBisCO_large","RuBisCO_large_N","IF-2B","PfkB","Metalloenzyme","DeoC","OMPdecase","SIS","Pribosyltran","PRTase_2","UPRTase","G6PD_N","6PGD"]:
    s=set(GP[GP.pf==pf].bin_id)
    print(f"   Pfam {pf:18s} Micrarchaeota {len(s & set(B.bin_id)):4d}/{len(B)} | Micrarchaeales LF+ comp70 {len(s & set(grp['Micrarchaeales_LF+_comp70'].bin_id)):3d}/{den['Micrarchaeales_LF+_comp70']} | LF- comp70 {len(s & set(grp['Micrarchaeales_LF-_comp70'].bin_id)):3d}/{den['Micrarchaeales_LF-_comp70']} | labeled287 {len(s & set(grp['labeled287'].bin_id)):3d}")
