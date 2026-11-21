#!/usr/bin/env python3
"""Triage of four enzyme-family leads from the slot-rescue pass: candidates, tiers, lengths, and whether the genome
already carries the function through another annotated gene (same family or a non-homologous isofunctional enzyme)."""
import duckdb, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_rows",300); pd.set_option("display.max_colwidth",70)
SR = "../slot_rescue"
con=duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{SR}/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{SR}/verify/verification_per_protein.parquet')")
con.execute("""CREATE TEMP TABLE gk AS SELECT DISTINCT pr.bin_id, a.accession ko FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='kofam'""")
con.execute("""CREATE TEMP TABLE gp AS SELECT DISTINCT pr.bin_id, a.name pf FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam'""")
FAM = {
 "LacAB_rpiB": dict(pf=["LacAB_rpiB","Rib_5-P_isom_A"], ko=["K01808","K01807","K01819","K00615","K00616","K00948","K08093","K08094","K13812","K13831","K18931","K18237","K01601","K00852"]),
 "Thy1": dict(pf=["Thy1","Thymidylat_synt"], ko=["K03465","K00560","K00287","K00943","K00525","K00526","K21636"]),
 "Thymidylat_synt": dict(pf=["Thy1","Thymidylat_synt"], ko=["K03465","K00560","K00287","K00943"]),
 "HMG_CoA_synt_C": dict(pf=["HMG_CoA_synt_C","HMG_CoA_synt_N","HMG-CoA_red","Thiolase_N","GHMP_kinases_N","AA_kinase"], ko=["K01641","K00021","K00054","K00626","K00869","K06981","K01823","K13787","K17104","K17105","K19664"]),
}
out=[]
for lab,spec in FAM.items():
    c = con.execute("""SELECT p.protein_id, pr.bin_id, pr.contig_id, pr.gene_index, pr.sequence_length len, v.tier, round(v.vfrac,3) support, v.proposed_evalue ev, p.votes_for vf, p.supporting_genomes sg, p.vote_share,
        b.completeness comp, b.contamination cont, split_part(b.taxonomy,';',2) phylum, split_part(b.taxonomy,';',4) ord,
        (SELECT arg_min("name", evalue) FROM d.protein_resemblance r WHERE r.protein_id=p.protein_id) cdd,
        (SELECT min(evalue) FROM d.protein_resemblance r WHERE r.protein_id=p.protein_id) cdd_e,
        len(regexp_replace(pr.sequence,'[^ILVFMAWC]','','g'))::double/pr.sequence_length hyd
        FROM pass p JOIN ver v USING (protein_id) JOIN d.proteins pr USING (protein_id) JOIN d.bins b USING (bin_id)
        WHERE p.proposed_label_name=?""",[lab]).df()
    c["label"]=lab
    for k in spec["ko"]:
        s=set(con.execute("select bin_id from gk where ko=?",[k]).df().bin_id); c[k]=c.bin_id.isin(s)
    for f in spec["pf"]:
        s=set(con.execute("select bin_id from gp where pf=?",[f]).df().bin_id); c["pf_"+f]=c.bin_id.isin(s)
    print(f"\n######## {lab}: {len(c)} labeled genes in {c.bin_id.nunique()} genomes")
    print(c.tier.value_counts().to_dict(), "| length quantiles", c.len.quantile([.1,.5,.9]).round().tolist(), "| completeness median", c.comp.median())
    print(" phyla:", c.phylum.value_counts().head(6).to_dict())
    print(" CDD best hit:", c.cdd.fillna("(none)").value_counts().head(8).to_dict())
    t1=c[c.tier=="T1"]
    print(f" T1: {len(t1)} genes; genome-level presence of other annotations among T1 genes' genomes:")
    print("  ", {k:int(t1[k].sum()) for k in spec['ko']}, {f:int(t1['pf_'+f].sum()) for f in spec['pf']})
    print(" all tiers:", {k:int(c[k].sum()) for k in spec['ko']}, {f:int(c['pf_'+f].sum()) for f in spec['pf']})
    out.append(c)
    # DPANN-wide baseline for the same annotations
    nb=con.execute("select count(*) from d.bins").fetchone()[0]
    print(" DPANN-wide genomes with:", {k:con.execute("select count(*) from gk where ko=?",[k]).fetchone()[0] for k in spec['ko']}, {f:con.execute("select count(*) from gp where pf=?",[f]).fetchone()[0] for f in spec['pf']}, "of", nb)
pd.concat(out).to_csv("candidates_all.tsv",sep="\t",index=False)
