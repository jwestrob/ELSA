#!/usr/bin/env python
"""Convert an MMseqs2 cluster TSV (representative gid, member gid) into a token arm."""
import argparse
import json
import time
from pathlib import Path

import duckdb

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
N = 4257531


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"""CREATE TABLE c AS SELECT * FROM read_csv('{a.tsv}', delim='\t', header=false,
                    columns={{'rep':'BIGINT','gid':'BIGINT'}})""")
    n, nd, lo, hi = con.execute("SELECT count(*), count(DISTINCT gid), min(gid), max(gid) FROM c").fetchone()
    assert n == N and nd == N and lo == 0 and hi == N - 1, (n, nd, lo, hi)
    tmp = RD / "tokens" / f".{a.arm}.parquet.tmp"
    con.execute(f"""COPY (SELECT gid, dense_rank() OVER (ORDER BY rep) - 1 AS token FROM c ORDER BY gid)
                    TO '{tmp}' (FORMAT parquet)""")
    n_tok, n_single, biggest = con.execute("""SELECT count(*), sum((s = 1)::INT), max(s)
        FROM (SELECT rep, count(*) AS s FROM c GROUP BY rep)""").fetchone()
    tmp.rename(RD / "tokens" / f"{a.arm}.parquet")
    manifest = dict(arm=a.arm, rows=n, tokens=int(n_tok), singletons=int(n_single), largest=int(biggest),
                    params=a.params, seconds=a.seconds, source="coordinator, engine workstream",
                    seqdb=str(RD / "ws_tokens/work/mmseqs/seqdb"))
    json.dump(manifest, open(RD / "tokens" / f"{a.arm}.json", "w"), indent=1)
    with open(RD / "tokens/INDEX.tsv", "a") as fh:
        fh.write(f"{a.arm}\t{n_tok}\t{n}\t{a.seconds}\t{time.strftime('%Y-%m-%d')}\n")
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
