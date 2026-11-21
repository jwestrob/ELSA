#!/usr/bin/env python3
"""convertalis TSV (query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen) -> parquet, typed, self hits dropped.
Copy of repair/code/tsv2parquet.py with positions and lengths widened from USMALLINT to UINTEGER: this data set
holds a 78,141-aa protein (DPANN maximum 24,483)."""
import os, sys, json, duckdb
src, dst = sys.argv[1], sys.argv[2]
selfdrop = "WHERE q <> t" if (len(sys.argv) < 4 or sys.argv[3] != "keep_all") else ""   # keep_all: query ids are not uids
con = duckdb.connect()
con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}; SET preserve_insertion_order=false")
cols = ("{'q':'UINTEGER','t':'UINTEGER','evalue':'DOUBLE','bits':'FLOAT','fident':'FLOAT','alnlen':'UINTEGER','qstart':'UINTEGER',"
        "'qend':'UINTEGER','qlen':'UINTEGER','tstart':'UINTEGER','tend':'UINTEGER','tlen':'UINTEGER'}")
con.execute(f"COPY (SELECT * FROM read_csv('{src}', delim='\t', header=false, columns={cols}) {selfdrop}) "
            f"TO '{dst}.tmp' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
os.replace(f"{dst}.tmp", dst)
n = con.execute(f"SELECT count(*), count(DISTINCT q) FROM '{dst}'").fetchone()
print(json.dumps({"file": dst, "rows": n[0], "queries_with_hits": n[1]}))
