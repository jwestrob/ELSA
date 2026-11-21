#!/usr/bin/env python
"""Convert an MMseqs2 cluster TSV (representative<TAB>member, headers = gid) into a token arm.

Output: <out>.parquet with columns gid BIGINT, token BIGINT (one row per universe gid, token = -1 for a
protein without a cluster), and <out>.json with parameters, inputs, counts, wall time and code hash.
Tokens are dense integers ordered by the representative's gid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True, help="path without extension")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--params", required=True, help="JSON string of clustering parameters")
    ap.add_argument("--mmseqs-seconds", type=float, required=True)
    a = ap.parse_args()
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"""CREATE TABLE cl AS SELECT rep::BIGINT AS rep, mem::BIGINT AS gid FROM read_csv('{a.clusters}', delim='\t',
                    header=false, columns={{'rep':'VARCHAR','mem':'VARCHAR'}})""")
    con.execute(f"CREATE TABLE u AS SELECT gid FROM read_parquet('{a.universe}')")
    n_u = con.execute("SELECT count(*) FROM u").fetchone()[0]
    dup = con.execute("SELECT count(*) - count(DISTINCT gid) FROM cl").fetchone()[0]
    assert dup == 0, f"{dup} proteins in more than one cluster"
    con.execute("CREATE TABLE tk AS SELECT rep, row_number() OVER (ORDER BY rep) - 1 AS token FROM (SELECT DISTINCT rep FROM cl)")
    con.execute("""CREATE TABLE arm AS SELECT u.gid::BIGINT AS gid, coalesce(tk.token, -1)::BIGINT AS token
                   FROM u LEFT JOIN cl USING (gid) LEFT JOIN tk USING (rep) ORDER BY u.gid""")
    n, n_tok, n_assigned = con.execute("SELECT count(*), count(DISTINCT token) FILTER (WHERE token >= 0), sum((token >= 0)::INT) FROM arm").fetchone()
    assert n == n_u
    stray = con.execute("SELECT count(*) FROM cl WHERE gid NOT IN (SELECT gid FROM u)").fetchone()[0]
    assert stray == 0, f"{stray} clustered ids absent from the universe"
    sizes = con.execute("""SELECT quantile_cont(k, [0.5, 0.9, 0.99]), max(k), sum((k = 1)::INT), sum((k >= 2)::INT)
                           FROM (SELECT token, count(*) k FROM arm WHERE token >= 0 GROUP BY 1)""").fetchone()
    genome_span = con.execute(f"""SELECT sum((g >= 2)::INT), sum((g >= 5)::INT), sum((g >= 50)::INT) FROM (
        SELECT a.token, count(DISTINCT u.genome) g FROM arm a JOIN read_parquet('{a.universe}') u USING (gid)
        WHERE a.token >= 0 GROUP BY 1)""").fetchone()
    out = Path(a.out)
    tmp = out.with_name("." + out.name + ".tmp.parquet")
    con.execute(f"COPY arm TO '{tmp}' (FORMAT parquet)")
    tmp.rename(out.with_suffix(".parquet"))
    meta = dict(arm=out.name, universe=a.universe, universe_rows=n_u, rows=n, n_tokens=n_tok, n_assigned=int(n_assigned),
                singleton_tokens=int(sizes[2]), tokens_ge2_members=int(sizes[3]),
                token_size_q50_q90_q99=sizes[0], token_size_max=int(sizes[1]),
                tokens_in_ge2_genomes=int(genome_span[0]), tokens_in_ge5_genomes=int(genome_span[1]),
                tokens_in_ge50_genomes=int(genome_span[2]),
                params=json.loads(a.params), clusters_tsv=a.clusters, clusters_sha256=sha(a.clusters),
                seconds_mmseqs=a.mmseqs_seconds, seconds_convert=round(time.time() - t0, 1),
                code_sha256=sha(__file__), date=time.strftime("%Y-%m-%d"),
                parquet_sha256=sha(out.with_suffix(".parquet")))
    tmpj = out.with_name("." + out.name + ".tmp.json")
    tmpj.write_text(json.dumps(meta, indent=2) + "\n")
    tmpj.rename(out.with_suffix(".json"))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
