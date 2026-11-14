"""Per-protein annotation evidence from the Sharur DPANN database that the package's
annotation_flags.parquet does not carry: CRISPRCasTyper profile assignments (system_proteins),
derived `_locus` / `_validation` atoms, and the Prodigal `partial` flag. Read-only."""
import duckdb
from pathlib import Path
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
OUT = Path(__file__).resolve().parent / 'work' / 'data' / 'sharur_extra_flags.parquet'
con = duckdb.connect(DB, read_only=True)
con.execute("SET threads=1")
con.execute(f"""
COPY (
  SELECT p.protein_id,
         p.partial,
         p.protein_id IN (SELECT protein_id FROM system_proteins) AS in_system_proteins,
         p.protein_id IN (SELECT protein_id FROM system_proteins WHERE system_source='cctyper') AS src_cctyper,
         p.protein_id IN (SELECT protein_id FROM semantic_atoms WHERE source_db IN ('_locus','_validation')) AS src_derived_locus_validation,
         p.protein_id IN (SELECT protein_id FROM annotations) AS in_annotations
  FROM proteins p
) TO '{OUT}' (FORMAT parquet)
""")
print(con.execute(f"SELECT count(*), sum(src_cctyper::int), sum(src_derived_locus_validation::int), sum(in_annotations::int), sum(in_system_proteins::int) FROM '{OUT}'").fetchall())
print(con.execute(f"SELECT partial, count(*) FROM '{OUT}' GROUP BY ALL ORDER BY 2 DESC").fetchall())
