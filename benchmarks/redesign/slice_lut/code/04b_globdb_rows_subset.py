#!/usr/bin/env python3
"""GlobDB r232 Pfam + KOfam rows for a genome subset (workstream slice_lut).

usage: 04b_globdb_rows_subset.py PARQUET_DIR GLOBDB_TAX SELECT OUT_PARQUET THREADS
  SELECT  'rank:<field>:<name>' (field 1-7 of the GTDB taxonomy string, e.g. rank:3:c__Alphaproteobacteria)
          or 'list:<file>' (one GlobDB genome ID per line)
"""
import json, os, sys, time
import duckdb

PQ, GTAX, SEL, OUTP, TH = sys.argv[1:6]
t0 = time.time()
con = duckdb.connect()
tmpd = os.path.join(os.path.dirname(OUTP), "duck_tmp_" + os.path.basename(OUTP))
con.execute(f"PRAGMA threads={TH}; SET preserve_insertion_order=false; PRAGMA temp_directory='{tmpd}'")
if SEL.startswith("rank:"):
    _, f, name = SEL.split(":", 2)
    con.execute(f"""CREATE TABLE ids AS SELECT column0 AS genome_id FROM read_csv('{GTAX}', delim='\t', header=false, all_varchar=true)
                    WHERE split_part(column1, ';', {int(f)}) = '{name}'""")
else:
    con.execute(f"CREATE TABLE ids AS SELECT DISTINCT column0 AS genome_id FROM read_csv('{SEL[5:]}', header=false, all_varchar=true)")
n_ids = con.execute("SELECT count(*) FROM ids").fetchone()[0]
con.execute(f"""COPY (SELECT genome_id, gene_idx, gene_raw, source, accession, function, e_value
    FROM read_parquet('{PQ}/*.parquet') WHERE source IN ('Pfam', 'KOfam') AND genome_id IN (SELECT genome_id FROM ids)
    ORDER BY genome_id, gene_idx) TO '{OUTP}.tmp' (FORMAT parquet, COMPRESSION zstd)""")
os.replace(OUTP + ".tmp", OUTP)
n = con.execute(f"SELECT count(*), count(DISTINCT genome_id) FROM '{OUTP}'").fetchone()
print(json.dumps(dict(select=SEL, ids=n_ids, rows=n[0], genomes=n[1], seconds=round(time.time() - t0, 1))), flush=True)
