#!/usr/bin/env python3
"""convertalis TSV (query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen) -> typed parquet."""
import os, sys, json, duckdb
src, dst = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '4')}; SET preserve_insertion_order=false")
cols = ("{'q':'UBIGINT','t':'UBIGINT','evalue':'DOUBLE','bits':'FLOAT','fident':'FLOAT','alnlen':'INTEGER','qstart':'INTEGER',"
        "'qend':'INTEGER','qlen':'INTEGER','tstart':'INTEGER','tend':'INTEGER','tlen':'INTEGER'}")
con.execute(f"COPY (SELECT * FROM read_csv('{src}', delim='\t', header=false, columns={cols})) TO '{dst}.tmp' "
            f"(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
os.replace(f"{dst}.tmp", dst)
n = con.execute(f"SELECT count(*), count(DISTINCT q), sum((q=t)::int) FROM '{dst}'").fetchone()
print(json.dumps({"file": dst, "rows": n[0], "queries_with_hits": n[1], "self_rows": n[2]}), flush=True)
