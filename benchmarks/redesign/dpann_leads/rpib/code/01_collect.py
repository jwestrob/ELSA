#!/usr/bin/env python3
"""Collect the 287 LacAB_rpiB-labeled genes: pass columns, verification tier, taxonomy, completeness, sequences.
Writes data/rpib_genes.tsv and data/rpib_genes.faa."""
import duckdb, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_rows",300); pd.set_option("display.max_colwidth",80); pd.set_option("display.max_columns",40)
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
OUT=R+"/dpann_leads/rpib/data"
con=duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{R}/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{R}/slot_rescue/verify/verification_per_protein.parquet')")
con.execute(f"CREATE VIEW ca AS SELECT * FROM read_parquet('{R}/slot_rescue/verify/A/checkA_per_protein.parquet')")
print(con.execute("describe ver").df()[["column_name","column_type"]].T.to_string())
print(con.execute("describe ca").df()[["column_name","column_type"]].T.to_string())
df=con.execute("""SELECT p.*, v.* EXCLUDE (protein_id), pr.sequence, pr.sequence_length, pr.bin_id, pr.contig_id, pr.gene_index,
  b.completeness, b.contamination, b.taxonomy
  FROM pass p JOIN ver v USING (protein_id) JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id
  WHERE p.proposed_label_name='LacAB_rpiB'""").df()
ca=con.execute("SELECT * FROM ca WHERE protein_id IN (SELECT protein_id FROM pass WHERE proposed_label_name='LacAB_rpiB')").df()
df=df.merge(ca,on="protein_id",how="left",suffixes=("","_ca"))
df["support"]=df.votes_for/df.supporting_genomes
for i,k in enumerate(["domain","phylum","class","order","family","genus","species"]):
    df[k]=df.taxonomy.str.split(";").str[i]
print(len(df), df.bin_id.nunique(), df.tier.value_counts().to_dict())
print(df.phylum.value_counts().to_dict()); print(df.order.value_counts().to_dict())
print(df.proposed_label_positional_homology.value_counts().to_dict())
print(df[["votes_for","voters_annotated","supporting_genomes","vote_share","support","sequence_length","completeness"]].describe().round(3).to_string())
print(df.alignment_lists_same_label.value_counts(dropna=False).to_dict(), df.alignment_label.value_counts(dropna=False).head(8).to_dict())
print(df.columns.tolist())
df.drop(columns=["sequence"]).to_csv(OUT+"/rpib_genes.tsv",sep="\t",index=False)
with open(OUT+"/rpib_genes.faa","w") as f:
    for r in df.itertuples(): f.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
