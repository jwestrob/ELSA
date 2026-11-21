import duckdb, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_rows",200); pd.set_option("display.max_colwidth",70)
con=duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS dpann (READ_ONLY)")
con.execute("CREATE TEMP TABLE srt AS SELECT * FROM read_parquet('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
con.execute("CREATE TEMP TABLE lx AS SELECT * FROM 'label_stats.parquet'")
con.execute("CREATE TEMP TABLE cdd AS SELECT r.protein_id, r.accession, r.name, r.evalue FROM dpann.protein_resemblance r WHERE r.protein_id IN (SELECT protein_id FROM srt)")
con.execute("""CREATE TEMP TABLE s2 AS SELECT s.*, lx.amp, s.votes_for::double/s.supporting_genomes vfrac,
  CASE WHEN s.votes_for::double/s.supporting_genomes<0.05 THEN '1 <5%' WHEN s.votes_for::double/s.supporting_genomes<0.2 THEN '2 5-20%' WHEN s.votes_for::double/s.supporting_genomes<0.5 THEN '3 20-50%' ELSE '4 >=50%' END vbin,
  CASE WHEN lx.amp>=10 THEN '1 >=10x' WHEN lx.amp>=1 THEN '2 1-10x' ELSE '3 <1x' END abin
  FROM srt s JOIN lx ON lx.lab=s.proposed_label_positional_homology""")
q="""WITH p AS (SELECT s.protein_id, any_value(s.{b}) b, any_value(s.proposed_label_positional_homology) lab,
        bool_or(replace(lower(c.accession),'pfam','PF')=s.proposed_label_positional_homology) same
      FROM s2 s JOIN cdd c USING(protein_id) WHERE c.accession ILIKE 'pfam%' AND s.proposed_label_positional_homology LIKE 'PF%' GROUP BY 1),
     t AS (SELECT {b} b, count(*) n_rows, count(DISTINCT CASE WHEN c.protein_id IS NOT NULL THEN s.protein_id END) n_cdd FROM s2 s LEFT JOIN (SELECT DISTINCT protein_id FROM cdd) c USING(protein_id) GROUP BY 1)
     SELECT t.b, t.n_rows, round(100.0*t.n_cdd/t.n_rows,1) pct_any_cdd, count(p.protein_id) n_cdd_pfam, round(100.0*avg(p.same::int),1) pct_same_pfam
     FROM t LEFT JOIN p ON p.b=t.b GROUP BY 1,2,3 ORDER BY 1"""
for b in ("vbin","abin"):
    print(b); print(con.execute(q.format(b=b)).df().to_string(index=False)); print()
# source annotations behind the most amplified labels
print(con.execute("""SELECT split_part(a.accession,'.',1) acc, a.name, a.protein_id, p.sequence_length len, a.start_aa, a.end_aa, a.score, a.evalue
  FROM dpann.annotations a JOIN dpann.proteins p USING(protein_id) WHERE a.source='pfam' AND split_part(a.accession,'.',1) IN ('PF01109','PF04949','PF09288','PF07544') ORDER BY 1, a.score DESC""").df().to_string(index=False))
# what the CDD says for genes labeled with those
print(con.execute("""SELECT s.proposed_label_name lab, c.name cdd, count(*) n FROM srt s JOIN cdd c USING(protein_id)
  WHERE s.proposed_label_positional_homology IN ('PF01109','PF04949','PF09288','PF07544','PF08641','PF03170') GROUP BY 1,2 QUALIFY row_number() OVER (PARTITION BY s.proposed_label_name ORDER BY count(*) DESC)<=4 ORDER BY 1, n DESC""").df().to_string(index=False))
print(con.execute("""SELECT proposed_label_name lab, count(*) n, round(median(length)) med_len, round(median(supporting_genomes)) med_supporters, round(median(votes_for)) med_votes,
   round(100.0*avg((alignment_lists_same_label='yes')::int),1) pct_aln_same, count(DISTINCT locus_id) loci, count(DISTINCT genome) genomes
  FROM srt WHERE proposed_label_positional_homology IN ('PF01109','PF04949','PF09288','PF07544','PF08641','PF03170','PF00412') GROUP BY 1 ORDER BY n DESC""").df().to_string(index=False))
print(con.execute("SELECT alignment_lists_same_label v, count(*) n FROM srt GROUP BY 1 ORDER BY n DESC").df().to_string(index=False))
