"""Per-protein composition for all 3,585,303 DPANN proteins from the Sharur DB (read-only, 1 thread):
hydrophobic fraction (ILVFMAWC / length), X count, sequence md5 (to cross-check the biotite FASTA)."""
import duckdb
from pathlib import Path
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
OUT = Path(__file__).resolve().parent / 'work' / 'data' / 'composition.parquet'
con = duckdb.connect(DB, read_only=True)
con.execute("SET threads=1")
con.execute(f"""
COPY (
  SELECT protein_id,
         length(rtrim(sequence,'*')) AS aa_len,
         (length(rtrim(sequence,'*')) - length(regexp_replace(rtrim(sequence,'*'), '[ILVFMAWC]', '', 'g')))::DOUBLE
             / greatest(length(rtrim(sequence,'*')),1) AS hyd,
         length(rtrim(sequence,'*')) - length(replace(rtrim(sequence,'*'), 'X', '')) AS n_x,
         md5(rtrim(sequence,'*')) AS seq_md5
  FROM proteins
) TO '{OUT}' (FORMAT parquet)
""")
print(con.execute(f"SELECT count(*), avg(hyd), quantile_cont(hyd, [0.1,0.5,0.9]), sum((n_x>0)::int) FROM '{OUT}'").fetchall())
