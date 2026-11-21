#!/usr/bin/env python3
"""Screen for labeled genes that are studyable and carry an inferable function: own weak profile hit to the proposed
family (tier T1), no KO of their own, at least 150 aa, soluble by composition; per label, how often the genome has no
other gene annotated with that family (the label fills a gap)."""
import duckdb, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_rows",400); pd.set_option("display.max_colwidth",60)
con=duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('pass/dpann_slot_rescue_pass.parquet')")
con.execute("CREATE VIEW ver AS SELECT * FROM read_parquet('verify/verification_per_protein.parquet')")
con.execute("""CREATE TEMP TABLE c AS
  SELECT p.protein_id, p.proposed_label_positional_homology acc, p.proposed_label_name lab, p.proposed_label_kind kind, pr.bin_id, pr.sequence_length len,
         v.vfrac, v.proposed_evalue ev, v.amp,
         len(regexp_replace(pr.sequence,'[^ILVFMAWC]','','g'))::double/pr.sequence_length hyd
  FROM pass p JOIN ver v USING (protein_id) JOIN d.proteins pr USING (protein_id)
  WHERE v.tier='T1' AND p.own_ko_agrees='no_ko'""")
con.execute("""CREATE TEMP TABLE has AS SELECT DISTINCT pr.bin_id, a.name FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam'""")
print(con.execute("select count(*), sum((len>=150 and hyd<0.45)::int) from c").fetchall())
df = con.execute("""SELECT c.lab, count(*) n, round(median(len)) len, round(median(hyd),2) hyd, round(median(vfrac),2) support, median(ev) med_e,
     sum((h.name IS NULL)::int) gap_genomes
  FROM c LEFT JOIN has h ON h.bin_id=c.bin_id AND h.name=c.lab
  WHERE c.kind='pfam' AND len>=150 AND hyd<0.45 GROUP BY 1 HAVING count(*)>=15 ORDER BY gap_genomes DESC LIMIT 160""").df()
print(df.to_string(index=False))
