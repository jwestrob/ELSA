#!/usr/bin/env python3
"""Rung 5, remaining annotation sources: CDD best hits (protein_resemblance) and family-level HHsearch Pfam hits
(family_annotations via protein_families) naming a ribose-5-phosphate isomerase (either type) in the 229 labeled genomes
that carry no KO / Pfam R5P isomerase, excluding the labeled long-form gene itself."""
import duckdb, pandas as pd
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
g=pd.read_csv(D+"/gap_test.tsv",sep="\t"); lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t")
gap=g[g.LF_labeled&~g.any_R5P_isomerase].bin_id.tolist(); print("gap genomes", len(gap))
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
pat="(?i)(rpia|rpib|ribose.5.phosphate isomerase|ribose-5-P isomerase|LacAB|Rib_5-P_isom|PRK05571|PRK00702|PRK08621|PRK08622|PRK12613|PRK12615|COG0120|COG0698|TIGR00021|TIGR00689|TIGR01118|TIGR01119|TIGR01120|PTZ00215|RPI_A|RpiB_LacA_LacB)"
c=con.execute(f"""SELECT pr.bin_id, r.protein_id, r."name", r.description, r.evalue FROM d.protein_resemblance r JOIN d.proteins pr USING (protein_id)
   WHERE pr.bin_id IN (SELECT unnest(?)) AND (regexp_matches(r."name", '{pat}') OR regexp_matches(r.description, '{pat}'))""",[gap]).df()
c=c[~c.protein_id.isin(set(lab.protein_id))]
print("CDD R5P-isomerase-like hits on other genes in gap genomes:", len(c), "proteins in", c.bin_id.nunique(), "genomes"); print(c.head(10).to_string())
f=con.execute(f"""SELECT pf.bin_id, pf.protein_id, fa.hit_name, fa.probability FROM d.protein_families pf JOIN d.family_annotations fa USING (family_id)
   WHERE pf.bin_id IN (SELECT unnest(?)) AND fa.probability>=90 AND (fa.hit_name IN ('LacAB_rpiB','Rib_5-P_isom_A'))""",[gap]).df()
own=f[f.protein_id.isin(set(lab.protein_id))]; f=f[~f.protein_id.isin(set(lab.protein_id))]
print("family-level HHsearch (prob>=90) LacAB_rpiB / Rib_5-P_isom_A on other genes in gap genomes:", f.protein_id.nunique(), "proteins in", f.bin_id.nunique(), "genomes;", "on the labeled genes themselves:", own.protein_id.nunique())
print(f.hit_name.value_counts().to_dict())
