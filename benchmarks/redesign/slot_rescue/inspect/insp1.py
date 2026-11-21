import duckdb, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_rows",200); pd.set_option("display.max_colwidth",60)
con=duckdb.connect()
con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS dpann (READ_ONLY)")
con.execute("CREATE TEMP TABLE srt AS SELECT * FROM read_parquet('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
con.execute("""CREATE TEMP TABLE lab_annot AS
 SELECT split_part(accession,'.',1) acc, count(DISTINCT protein_id) annotated FROM dpann.annotations WHERE source='pfam' GROUP BY 1
 UNION ALL SELECT accession acc, count(DISTINCT protein_id) annotated FROM dpann.annotations WHERE source='kofam' GROUP BY 1""")
con.execute("""CREATE TEMP TABLE lx AS WITH l AS (SELECT proposed_label_positional_homology lab, any_value(proposed_label_name) nm, count(*) n_called,
   median(votes_for::double/supporting_genomes) med_voter_frac, avg((vote_share>0.8)::int) frac_hi FROM srt GROUP BY 1)
 SELECT l.*, coalesce(a.annotated,0) annotated, l.n_called::double/greatest(coalesce(a.annotated,0),1) amp FROM l LEFT JOIN lab_annot a ON a.acc=l.lab""")
print(con.execute("""SELECT CASE WHEN amp>=10 THEN 'a: n_called >= 10x annotated' WHEN amp>=1 THEN 'b: 1x-10x' WHEN amp>=0.2 THEN 'c: 0.2x-1x' ELSE 'd: < 0.2x' END amp_bin,
 count(*) labels, sum(n_called) rows_called, round(100.0*sum(n_called)/(SELECT count(*) FROM srt),1) pct, sum((annotated=0)::int) labels_0_annot FROM lx GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
print()
print(con.execute("SELECT nm, lab, n_called, annotated, round(amp,1) amp, round(med_voter_frac,3) med_voter_frac, round(frac_hi,2) frac_hi FROM lx WHERE n_called>=150 ORDER BY amp DESC LIMIT 45").df().to_string(index=False))
print()
# voter fraction x label amplification
print(con.execute("""SELECT CASE WHEN s.votes_for::double/s.supporting_genomes<0.05 THEN '<5%' WHEN s.votes_for::double/s.supporting_genomes<0.2 THEN '5-20%' WHEN s.votes_for::double/s.supporting_genomes<0.5 THEN '20-50%' ELSE '>=50%' END voter_frac,
 CASE WHEN lx.amp>=1 THEN 'label amp>=1' ELSE 'label amp<1' END a, count(*) n FROM srt s JOIN lx ON lx.lab=s.proposed_label_positional_homology GROUP BY 1,2 ORDER BY 1,2""").df().to_string(index=False))
con.execute("COPY (SELECT * FROM lx) TO 'label_stats.parquet' (FORMAT parquet)")
