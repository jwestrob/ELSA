#!/usr/bin/env python
"""Structural check of sequence-invisible retrieved pairs (TM-align on a hash sample).

prep       for every (model, arm, search) pick a deterministic hash sample of retrieved
           pairs that are sequence-invisible and have AFDB structures on both sides
           (target ~TARGET pairs per group), write the fake prefilter list
summarize  join TM-scores; per group: fraction TM >= 0.5 (max of the two normalizations),
           and the implied structurally-supported share of all retrieved pairs
           (vis + invisible x TM>=0.5 fraction)
"""
import argparse
import json
import time

import duckdb
import numpy as np
import pandas as pd

from common import MODELS, S, U, W, log

TARGET = 1500
H = "hash(CAST(qkey AS BIGINT) * 5000000 + CAST(tkey AS BIGINT)) % 100000"
SEQ_COLS = ("names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'], "
            "types={'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}")


def connect(threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    return con


def load_hits(con, base):
    con.execute(f"""CREATE TABLE G AS SELECT gid, mmseqs_key AS key, has_struct
                    FROM read_parquet('{U}/genes.parquet')""")
    con.execute(f"""CREATE TABLE SEQ AS SELECT qkey, tkey, evalue
                    FROM read_csv('{base}/align/aln.tsv', delim='\t', header=false, {SEQ_COLS})""")
    parts = []
    for m in MODELS:
        hd = base / "eval" / m / "hits"
        if hd.exists():
            parts.append(f"SELECT '{m}' AS model, regexp_extract(filename, '([^/]+)\\.parquet$', 1) AS arm, * "
                         f"FROM read_parquet('{hd}/*.parquet', filename=true)")
    con.execute(f"""
        CREATE TABLE H AS
        SELECT h.model, h.arm, h.search, qg.key AS qkey, tg.key AS tkey, (s.evalue <= 1e-3) AS vis,
               (qg.has_struct AND tg.has_struct) AS has_struct
        FROM ({' UNION ALL '.join(parts)}) h JOIN G qg ON qg.gid = h.qgid JOIN G tg ON tg.gid = h.tgid
        JOIN SEQ s ON s.qkey = qg.key AND s.tkey = tg.key""")


def prep(a):
    base = W / a.tag
    T = base / "tm"
    T.mkdir(parents=True, exist_ok=True)
    con = connect(a.threads)
    load_hits(con, base)
    g = con.execute("""SELECT model, arm, search, count(*) AS n_inv_struct FROM H
                       WHERE NOT vis AND has_struct GROUP BY ALL""").df()
    g["thr"] = np.minimum(100000, np.ceil(100000 * TARGET / g.n_inv_struct.clip(lower=1))).astype(int)
    con.register("GT", g)
    con.execute(f"""CREATE TABLE PICK AS SELECT DISTINCT H.model, H.arm, H.search, H.qkey, H.tkey FROM H
                    JOIN GT USING (model, arm, search) WHERE NOT H.vis AND H.has_struct AND {H.replace('qkey', 'H.qkey').replace('tkey', 'H.tkey')} < GT.thr""")
    con.execute(f"COPY PICK TO '{T}/pick.parquet' (FORMAT parquet)")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT DISTINCT qkey, tkey, 0 AS s, 0 AS d FROM PICK ORDER BY qkey, tkey)
                    TO '{T}/pref_tm.tsv' (DELIMITER '\t', HEADER false)""")
    n = con.execute("SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM PICK)").fetchone()[0]
    g.to_csv(T / "groups.tsv", sep="\t", index=False)
    json.dump(dict(unique_pairs=n, groups=len(g), target_per_group=TARGET), open(T / "prep.json", "w"), indent=1)
    log("tm pairs", n, "groups", len(g))


def summarize(a):
    base = W / a.tag
    T = base / "tm"
    con = connect(a.threads)
    load_hits(con, base)
    con.execute(f"CREATE TABLE qx AS SELECT * FROM read_parquet('{S}/work/qx.parquet')")
    con.execute(f"""
        CREATE TABLE TM AS
        WITH j AS (
            SELECT qx.qkey, CAST(substr(c.t, 2) AS BIGINT) AS tkey, c.qtm, c.ttm
            FROM read_csv('{T}/aln_tm.tsv', delim='\t', header=false,
                names=['q','t','qtm','ttm','alntm','lddt','fident','qcov','tcov','alnlen','qlen','tlen','evalue','bits'],
                types={{'q':'VARCHAR','t':'VARCHAR','qtm':'DOUBLE','ttm':'DOUBLE'}}) c
            JOIN qx ON qx.native_id = c.q)
        SELECT qkey, tkey, greatest(qtm, ttm) AS tm_max FROM j
        QUALIFY row_number() OVER (PARTITION BY qkey, tkey ORDER BY greatest(qtm, ttm) DESC) = 1""")
    con.execute(f"CREATE TABLE PICK AS SELECT * FROM read_parquet('{T}/pick.parquet')")
    r = con.execute("""
        WITH tot AS (SELECT model, arm, search, count(*) AS n, avg(vis::INT) AS vis_frac,
                            avg((NOT vis AND has_struct)::INT) AS inv_struct_frac, avg((NOT vis)::INT) AS inv_frac
                     FROM H GROUP BY ALL),
             s AS (SELECT p.model, p.arm, p.search, count(*) AS n_picked, count(tm.tm_max) AS n_tm,
                          avg((tm.tm_max >= 0.5)::INT) AS inv_tm05_frac, median(tm.tm_max) AS inv_tm_median
                   FROM PICK p LEFT JOIN TM tm USING (qkey, tkey) GROUP BY ALL)
        SELECT tot.*, s.n_picked, s.n_tm, s.inv_tm05_frac, s.inv_tm_median,
               tot.vis_frac + tot.inv_frac * s.inv_tm05_frac AS seq_or_struct_frac_est
        FROM tot JOIN s USING (model, arm, search) ORDER BY model, search, arm""").df()
    out = base / "results"
    out.mkdir(exist_ok=True)
    r.to_csv(out / "retrieval_tm.tsv", sep="\t", index=False)
    log(r.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "summarize"])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    (prep if a.mode == "prep" else summarize)(a)
