"""Screen 3 / census, step 0: export small tables from the live Sharur DPANN database (republished
12 November 2026, dataset_id f652ef47..., opened READ_ONLY with one DuckDB thread).

Writes work/s3/data/:
  prot_flags.parquet   per protein: curated annotation evidence (every `annotations` source except 'cdd',
                       system_proteins incl. cctyper, derived _locus/_validation atoms), per-source flags, CDD
                       presence, Prodigal partial flag, sequence md5 (to check the protein set is unchanged)
  prot_fams.parquet    per protein: family_id in permissive-e3c50s75 (searched run) and in
                       permissive-e3c50s75-rescue3 (the engine token run), representative flags
  fam_stats.parquet    family_stats rows of both runs
  tier_status.parquet  family_annotation_status (searched families)
  tier_hits.parquet    family_annotations rows of the searched run with hit_rank <= 10, or meeting the rule
  cdd_prot.parquet     annotations rows with source = 'cdd' (best model per region, E <= 0.01)
  cdd_models.parquet   distinct CDD accession, name, description, unknown-function flag (Sharur regex)
  export_meta.json     counts, dataset_id, generated_at, file size and mtime of the database
"""
import json
import os
import re
from pathlib import Path

import duckdb

DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
SEAL = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/dataset.seal.json"
OUT = Path(__file__).resolve().parent / "work" / "s3" / "data"
SEARCHED = "permissive-e3c50s75"
TOKENRUN = "permissive-e3c50s75-rescue3"
# Sharur feature/remote-homology-dpann sharur/remote_homology/hhr.py UNKNOWN_FUNCTION_PATTERN
UF = r"DUF\d|UPF\d|unknown function|uncharacteri|hypothetical|function unknown|unknown protein"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET threads=1")
    con.execute(f"""COPY (
      WITH a AS (SELECT protein_id,
                        bool_or(source <> 'cdd') AS cur_annot,
                        bool_or(source = 'pfam') AS src_pfam,
                        bool_or(source IN ('kofam','kegg')) AS src_ko,
                        bool_or(source IN ('defensefinder','defensefinder_system')) AS src_df,
                        bool_or(source IN ('txsscan','txsscan_system')) AS src_txss,
                        bool_or(source IN ('hyddb','hyddb_subgroup')) AS src_hyddb,
                        bool_or(source = 'cdd') AS cdd_any
                 FROM annotations GROUP BY 1),
           s AS (SELECT protein_id, true AS in_sys, bool_or(system_source='cctyper') AS cctyper
                 FROM system_proteins GROUP BY 1),
           d AS (SELECT DISTINCT protein_id, true AS derived FROM semantic_atoms
                 WHERE source_db IN ('_locus','_validation'))
      SELECT p.protein_id, p.partial, md5(rtrim(p.sequence,'*')) AS seq_md5,
             coalesce(a.cur_annot,false) AS cur_annot, coalesce(a.src_pfam,false) AS src_pfam,
             coalesce(a.src_ko,false) AS src_ko, coalesce(a.src_df,false) AS src_df,
             coalesce(a.src_txss,false) AS src_txss, coalesce(a.src_hyddb,false) AS src_hyddb,
             coalesce(s.in_sys,false) AS in_sys, coalesce(s.cctyper,false) AS cctyper,
             coalesce(d.derived,false) AS derived, coalesce(a.cdd_any,false) AS cdd_any
      FROM proteins p LEFT JOIN a USING (protein_id) LEFT JOIN s USING (protein_id) LEFT JOIN d USING (protein_id)
    ) TO '{OUT}/prot_flags.parquet' (FORMAT parquet)""")
    con.execute(f"""COPY (
      SELECT e.protein_id, e.family_id AS fam_e3, e.is_representative AS rep_e3,
             r.family_id AS fam_r3, r.is_representative AS rep_r3
      FROM (SELECT * FROM protein_families WHERE clustering_run='{SEARCHED}') e
      FULL JOIN (SELECT * FROM protein_families WHERE clustering_run='{TOKENRUN}') r USING (protein_id)
    ) TO '{OUT}/prot_fams.parquet' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT * FROM family_stats WHERE clustering_run IN ('{SEARCHED}','{TOKENRUN}'))
                    TO '{OUT}/fam_stats.parquet' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT search_run, clustering_run, family_id, status, n_members, n_genomes, n_unique,
                                 n_sequences, rep_protein_id, rep_length, n_hits, verified
                          FROM family_annotation_status) TO '{OUT}/tier_status.parquet' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT family_id, clustering_run, search_run, source, db_name, db_version, msa_kind,
                                 hit_rank, hit_accession, hit_name, left(hit_description, 200) AS hit_description,
                                 probability, evalue, aligned_cols, q_start, q_end, q_length, t_start, t_end,
                                 t_length, unknown_function
                          FROM family_annotations
                          WHERE clustering_run='{SEARCHED}'
                            AND (hit_rank <= 10 OR (probability >= 90 AND aligned_cols >= 30)))
                    TO '{OUT}/tier_hits.parquet' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT protein_id, accession, evalue, score, start_aa, end_aa FROM annotations
                          WHERE source='cdd') TO '{OUT}/cdd_prot.parquet' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT accession, any_value(name) AS name, left(any_value(description), 300) AS description,
                                 regexp_matches(coalesce(any_value(name),'') || ' ' || coalesce(any_value(description),''),
                                                '(?i)({UF})') AS unknown_function
                          FROM annotations WHERE source='cdd' GROUP BY 1)
                    TO '{OUT}/cdd_models.parquet' (FORMAT parquet)""")
    seal = json.load(open(SEAL))
    st = os.stat(DB)
    meta = dict(db=DB, db_bytes=st.st_size, db_mtime=st.st_mtime, dataset_id=seal.get("dataset_id"),
                seal_generated_at=seal.get("generated_at"), unknown_function_pattern=UF)
    for f in sorted(OUT.glob("*.parquet")):
        meta[f.name] = con.execute(f"SELECT count(*) FROM '{f}'").fetchone()[0]
    meta["flags"] = dict(zip(["proteins", "cur_annot", "cdd_any", "in_sys", "cctyper", "derived", "dark_curated"],
                             con.execute(f"""SELECT count(*), sum(cur_annot::int), sum(cdd_any::int), sum(in_sys::int),
                                             sum(cctyper::int), sum(derived::int),
                                             sum((NOT (cur_annot OR in_sys OR derived))::int)
                                             FROM '{OUT}/prot_flags.parquet'""").fetchone()))
    meta["cdd_models_unknown"] = con.execute(f"SELECT sum(unknown_function::int), count(*) FROM '{OUT}/cdd_models.parquet'").fetchone()
    json.dump(meta, open(OUT / "export_meta.json", "w"), indent=1, default=str)
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
