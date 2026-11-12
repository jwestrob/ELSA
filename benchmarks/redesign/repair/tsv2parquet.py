#!/usr/bin/env python3
"""convertalis TSV (query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen) -> parquet, typed, self hits dropped."""
import os, sys, json, duckdb
src, dst = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}; SET preserve_insertion_order=false")
cols = ("{'q':'UINTEGER','t':'UINTEGER','evalue':'DOUBLE','bits':'FLOAT','fident':'FLOAT','alnlen':'USMALLINT','qstart':'USMALLINT',"
        "'qend':'USMALLINT','qlen':'USMALLINT','tstart':'USMALLINT','tend':'USMALLINT','tlen':'USMALLINT'}")
con.execute(f"COPY (SELECT * FROM read_csv('{src}', delim='\t', header=false, columns={cols}) WHERE q <> t) "
            f"TO '{dst}.tmp' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
os.replace(f"{dst}.tmp", dst)
n = con.execute(f"SELECT count(*), count(DISTINCT q) FROM '{dst}'").fetchone()
print(json.dumps({"file": dst, "rows": n[0], "queries_with_hits": n[1]}))
