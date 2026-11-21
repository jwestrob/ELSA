#!/usr/bin/env python
"""Read-only census of the SR-VP bacteria (PacBio) Sharur database: tables, row counts, columns,
annotation sources, structured callers, taxonomy and coordinate/strand presence. One DuckDB thread.
Writes a JSON receipt; touches nothing in the source tree."""
import json
import os
import sys
import time

import duckdb

DB = sys.argv[1]
OUT = sys.argv[2]
st0 = os.stat(DB)
con = duckdb.connect(DB, read_only=True, config={"threads": 1})
q = lambda s: con.execute(s).fetchall()
res = dict(db=DB, size=st0.st_size, mtime_epoch=st0.st_mtime, mtime=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st0.st_mtime)),
           duckdb_version=duckdb.__version__, opened_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
tabs = q("SELECT table_name FROM duckdb_tables() WHERE schema_name='main' ORDER BY 1")
views = q("SELECT view_name, sql FROM duckdb_views() WHERE NOT internal AND schema_name='main' ORDER BY 1")
res["tables"] = {}
for (t,) in tabs:
    cols = q(f"SELECT column_name, data_type FROM duckdb_columns() WHERE table_name='{t}' AND schema_name='main' ORDER BY column_index")
    n = q(f'SELECT count(*) FROM "{t}"')[0][0]
    res["tables"][t] = dict(rows=n, columns=cols)
res["views"] = {}
for v, sql in views:
    try:
        n = q(f'SELECT count(*) FROM "{v}"')[0][0]
    except Exception as e:  # noqa
        n = f"error: {e}"
    res["views"][v] = dict(rows=n, sql=sql[:400])
for t, s in [
    ("annotation_sources", "SELECT source, count(*), count(DISTINCT protein_id) FROM annotations GROUP BY 1 ORDER BY 2 DESC"),
    ("annotation_source_versions", "SELECT * FROM annotation_source_versions"),
    ("dataset_meta", "SELECT * FROM dataset_meta"),
    ("system_proteins_sources", "SELECT system_source, count(*), count(DISTINCT system_id) FROM system_proteins GROUP BY 1 ORDER BY 1"),
    ("crispr_cas_callers", "SELECT caller, status, count(*) FROM crispr_cas_systems GROUP BY ALL ORDER BY ALL"),
    ("crispr_cas_predictions", "SELECT prediction, count(*) FROM crispr_cas_systems GROUP BY 1 ORDER BY 2 DESC"),
    ("crispr_cas_genes_count", "SELECT genes_count, count(*) FROM crispr_cas_systems GROUP BY 1 ORDER BY 1"),
    ("loci_types", "SELECT locus_type, count(*) FROM loci GROUP BY 1 ORDER BY 2 DESC"),
    ("crispr_array_callers", "SELECT caller, count(*) FROM crispr_array_types GROUP BY 1"),
    ("hydrogenase_outcomes", "SELECT outcome, count(*) FROM hydrogenase_classifications GROUP BY 1 ORDER BY 2 DESC"),
    ("defense_types_n", "SELECT count(*), count(DISTINCT system_type), sum((genes_count>=2)::INT) FROM defense_systems"),
    ("secretion_types", "SELECT system_type, count(*), sum((genes_count>=2)::INT) FROM secretion_systems GROUP BY 1 ORDER BY 2 DESC"),
    ("strand_values", "SELECT strand, count(*) FROM proteins GROUP BY 1 ORDER BY 2 DESC"),
    ("partial_values", "SELECT partial, count(*) FROM proteins GROUP BY 1 ORDER BY 2 DESC LIMIT 20"),
    ("protein_nulls", """SELECT count(*), sum((start IS NULL)::INT), sum((end_coord IS NULL)::INT), sum((strand IS NULL)::INT),
                          sum((gene_index IS NULL)::INT), sum((sequence IS NULL OR length(sequence)=0)::INT),
                          sum((bin_id IS NULL)::INT), sum((contig_id IS NULL)::INT), count(DISTINCT protein_id),
                          count(DISTINCT bin_id), count(DISTINCT contig_id) FROM proteins"""),
    ("bins_sample", "SELECT * FROM bins LIMIT 3"),
    ("contigs_sample", "SELECT * EXCLUDE (sequence) FROM contigs LIMIT 3" ),
]:
    try:
        res[t] = q(s)
    except Exception as e:  # noqa
        res[t] = f"error: {e}"
try:
    res["proteins_sample"] = q("SELECT * EXCLUDE (sequence) FROM proteins LIMIT 3")
except Exception as e:  # noqa
    res["proteins_sample"] = f"error: {e}"
st1 = os.stat(DB)
res["mtime_after"] = st1.st_mtime
res["size_after"] = st1.st_size
res["stable_during_census"] = (st0.st_mtime == st1.st_mtime and st0.st_size == st1.st_size)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "tables"}, indent=1, default=str)[:20000])
for t, d in res["tables"].items():
    print(f"{t:32s} {d['rows']:>12,}  " + ", ".join(c for c, _ in d["columns"]))
