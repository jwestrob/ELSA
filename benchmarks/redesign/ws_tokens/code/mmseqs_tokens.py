#!/usr/bin/env python
"""Turn the MMseqs2 cluster TSV (representative gid, member gid) into the token file.

token = gid of the cluster representative. Every gid must appear exactly once as a
member, and the database lookup must map key == gid for every row.
"""
import argparse
import json
import time

import duckdb
import numpy as np

from common import NGENES, write_tokens, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--seconds-cluster", type=float, required=True)
    ap.add_argument("--peak-rss-kb", type=float, default=0)
    ap.add_argument("--mmseqs", required=True)
    ap.add_argument("--mmseqs-version", required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    lk = con.execute(f"""SELECT count(*), sum((key = CAST(name AS BIGINT))::INT), min(key), max(key)
                         FROM read_csv('{a.work}/seqdb.lookup', delim='\t', header=false,
                         columns={{'key':'BIGINT','name':'VARCHAR','f':'BIGINT'}})""").fetchone()
    assert lk == (NGENES, NGENES, 0, NGENES - 1), f"lookup key != gid: {lk}"
    c = con.execute(f"""SELECT rep, mem FROM read_csv('{a.work}/clu.tsv', delim='\t', header=false,
                        columns={{'rep':'BIGINT','mem':'BIGINT'}})""").fetchnumpy()
    rep, mem = c["rep"], c["mem"]
    assert len(mem) == NGENES, f"{len(mem)} member rows"
    tok = np.full(NGENES, -2, dtype=np.int64)
    tok[mem] = rep
    assert (tok >= 0).all(), "some gid is not a cluster member"
    assert np.bincount(mem, minlength=NGENES).max() == 1, "a gid appears twice"
    assert (tok[np.unique(rep)] == np.unique(rep)).all(), "a representative is not in its own cluster"
    man = dict(source="mmseqs", model="seq", transform="none", detail="id30c80",
               rule="mmseqs cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0 (defaults otherwise: -s 4, "
                    "cascaded, cluster-mode 0); same rule as easy-cluster in the 2026-11-02 DPANN pass",
               token_definition="gid of the cluster representative",
               mmseqs=a.mmseqs, mmseqs_version=a.mmseqs_version,
               inputs=dict(fasta=f"{a.work}/all.faa", seqdb=f"{a.work}/seqdb",
                           query_db="native_run_20260910/query/querydb",
                           reference_db="native_run_20260910/reference/keggclusterdb"),
               seconds_cluster=a.seconds_cluster, peak_rss_gb_cluster=round(a.peak_rss_kb / 1024 / 1024, 2),
               verification="lookup key == gid for all rows; each gid exactly once as member; "
                            "sequence lengths equal universe length for all rows (make_fasta.py)")
    write_tokens("mmseqs__seq__id30c80", tok, man, a.seconds_cluster + time.time() - t0)


if __name__ == "__main__":
    main()
