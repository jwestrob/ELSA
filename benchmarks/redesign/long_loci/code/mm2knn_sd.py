#!/usr/bin/env python
"""convertalis TSV (query, target, bits; uids) -> kNN-layout arrays: I[uid, r] = r-th best target uid by bit score
(ties: target uid; self hits dropped; -1 padded), B[uid, r] = its bits. usage: mm2knn_sd.py all.tsv OUTDIR K N  (copy of vocab_lists mm2knn.py with N as an argument)"""
import json
import os
import sys
import time

import duckdb
import numpy as np

src, out, K, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

t0 = time.time()
con = duckdb.connect()
con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}; SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{out}/tmp'")
con.execute(f"CREATE TABLE h AS SELECT q, t, bits FROM read_csv('{src}', delim='\t', header=false, "
            "columns={'q':'UINTEGER','t':'UINTEGER','bits':'FLOAT'}) WHERE q <> t")
rows = con.execute("SELECT count(*), count(DISTINCT q) FROM h").fetchone()
print("loaded", rows, f"{time.time() - t0:.0f}s", flush=True)
I = np.full((N, K), -1, np.int32)
B = np.zeros((N, K), np.float32)
nh = np.zeros(N, np.int64)
step = 200_000
for a in range(0, N, step):
    b = min(N, a + step)
    d = con.execute(f"SELECT q, t, bits, row_number() OVER (PARTITION BY q ORDER BY bits DESC, t) - 1 AS r FROM h "
                    f"WHERE q >= {a} AND q < {b}").fetchnumpy()
    q, t, bits, r = (np.asarray(d[c]) for c in ("q", "t", "bits", "r"))
    np.add.at(nh, q.astype(np.int64), 1)
    k = r < K
    I[q[k], r[k]] = t[k]
    B[q[k], r[k]] = bits[k]
    print("chunk", a, len(q), f"{time.time() - t0:.0f}s", flush=True)
np.save(f"{out}/aln__k{K}__I.npy", I)
np.save(f"{out}/aln__k{K}__B.npy", B)
info = dict(rows=int(rows[0]), queries_with_hits=int(rows[1]), hits_per_query_q=np.quantile(nh, [.1, .25, .5, .75, .9, .99, 1]).tolist(),
            queries_without_hit=int((nh == 0).sum()), K=K, seconds=round(time.time() - t0, 1))
json.dump(info, open(f"{out}/aln__k{K}.json", "w"), indent=1)
print(info, flush=True)
