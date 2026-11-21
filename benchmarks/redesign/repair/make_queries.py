#!/usr/bin/env python3
"""Query sets for the genome-wide gene-call repairs (workstream repair).

usage: make_queries.py <repair dir> <package dir> <subgenic dir> <base token parquet> <n shards>
Writes in <repair dir>:
  lists/Q.uid, lists/Q_<k>.uid    union of the rescue and split queries (uid order, round-robin shards)
  pairs.parquet                    co-oriented distance-1 pairs with gap <= 50 nt: ga, gb (ga = lower gid), gap_nt,
                                   len_a, len_b, uid_a, uid_b, short_is_a, genome_code, strand
  genes.parquet                    gid, genome_code, contig_code, pos, strand, len (FASTA), uid, token, tsize
  queries.json                     counts
"""
import json, os, sys, time
import numpy as np, duckdb

R, P, SG, TOK, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
PD = json.load(open(f"{R}/code/predeclared.json"))
t0 = time.time()
os.makedirs(f"{R}/lists", exist_ok=True)
con = duckdb.connect()
con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}")
con.execute(f"""CREATE TABLE g AS SELECT g.gid, g.genome_code, g.contig_code, g.position_index pos, g.strand, g.start, g."end" stop,
     u.length len, m.uid, t.token
  FROM '{P}/universe/genes.parquet' g JOIN '{SG}/gid2uid.parquet' m USING (gid) JOIN '{SG}/uniq.parquet' u USING (uid)
  JOIN '{TOK}' t USING (gid) ORDER BY gid""")
con.execute("CREATE TABLE ts AS SELECT token, count(*) tsize FROM g GROUP BY token")
con.execute("CREATE TABLE g2 AS SELECT g.*, ts.tsize FROM g JOIN ts USING (token) ORDER BY gid")
n = con.execute("SELECT count(*), max(gid) FROM g2").fetchone()
assert n[0] == 3585303 and n[1] == 3585302
con.execute(f"COPY (SELECT gid, genome_code, contig_code, pos, strand, start, stop, len, uid, token, tsize FROM g2 ORDER BY gid) TO '{R}/genes.parquet' (FORMAT parquet)")
con.execute("""CREATE TABLE pr AS SELECT a.gid ga, b.gid gb, a.genome_code, a.strand,
     CASE WHEN a.start <= b.start THEN b.start - a.stop - 1 ELSE a.start - b.stop - 1 END gap_nt,
     a.len len_a, b.len len_b, a.uid uid_a, b.uid uid_b
  FROM g2 a JOIN g2 b ON b.gid = a.gid + 1 AND a.contig_code = b.contig_code AND a.strand = b.strand""")
con.execute("DELETE FROM pr WHERE gap_nt > 50")
con.execute("ALTER TABLE pr ADD COLUMN short_is_a BOOLEAN; UPDATE pr SET short_is_a = len_a <= len_b")
con.execute(f"COPY (SELECT * FROM pr ORDER BY ga) TO '{R}/pairs.parquet' (FORMAT parquet)")
resc = con.execute("SELECT DISTINCT uid FROM g2 WHERE tsize < 3").fetchnumpy()["uid"]
splitq = con.execute("SELECT DISTINCT CASE WHEN short_is_a THEN uid_a ELSE uid_b END uid FROM pr").fetchnumpy()["uid"]
Q = np.union1d(resc, splitq)
np.savetxt(f"{R}/lists/Q.uid", Q, fmt="%d")
for k in range(N):
    np.savetxt(f"{R}/lists/Q_{k}.uid", Q[k::N], fmt="%d")
ulen = con.execute(f"SELECT length FROM '{SG}/uniq.parquet' ORDER BY uid").fetchnumpy()["length"]
res = {"genes": int(n[0]), "pairs_gap_le_50": int(con.execute("SELECT count(*) FROM pr").fetchone()[0]),
       "pairs_overlapping": int(con.execute("SELECT count(*) FROM pr WHERE gap_nt < 0").fetchone()[0]),
       "rescue_query_uids": int(len(resc)), "rescue_query_residues": int(ulen[resc].sum()),
       "split_query_uids": int(len(splitq)), "split_query_residues": int(ulen[splitq].sum()),
       "union_uids": int(len(Q)), "union_residues": int(ulen[Q].sum()),
       "shards": N, "shard_sizes": [int(len(Q[k::N])) for k in range(N)],
       "shard_residues": [int(ulen[Q[k::N]].sum()) for k in range(N)],
       "token_size_counts_1_2": con.execute("SELECT sum((tsize=1)::int), sum((tsize=2)::int) FROM g2").fetchone(),
       "seconds": round(time.time() - t0, 1)}
json.dump(res, open(f"{R}/queries.json", "w"), indent=1, default=int)
print(json.dumps(res, indent=1, default=int))
