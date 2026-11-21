#!/usr/bin/env python
"""Pair lists for MMseqs2 alignment, and the shared visibility label table.

pool     unique (q, c) pairs of the three models' pool samples, plus 20,000 random pairs of
         tonight's aln_seq.tsv as a recipe-consistency control -> align/pool/pref.tsv
missing  every anchor pair of every observed variant block, minus pairs already labelled
         (aln_seq.tsv, align/pool/aln.tsv) -> align/missing/pref.tsv
check    compare align/pool/aln.tsv with aln_seq.tsv on the control pairs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, N, RD, S, W, log, write_json

ALN_COLS = "names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'], " \
           "types={'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}"


def connect(threads):
    import duckdb
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    con.execute(f"""CREATE TABLE ukey AS SELECT side, row_index, mmseqs_key FROM read_parquet('{RD}/universe/genes.parquet')""")
    return con


def label_sources():
    srcs = [str(S / "work/aln_seq.tsv")]
    for p in sorted((W / "align").glob("*/aln.tsv")):
        srcs.append(str(p))
    return srcs


def load_labels(con):
    """Table lab(qkey, tkey, evalue, seqid): minimum E-value over every alignment source."""
    srcs = label_sources()
    con.execute(f"""CREATE OR REPLACE TABLE lab AS
        SELECT qkey, tkey, min(evalue) AS evalue, max(seqid) AS seqid
        FROM read_csv({srcs}, delim='\t', header=false, {ALN_COLS}) GROUP BY ALL""")
    return srcs


def write_pref(con, table, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM {table} ORDER BY qkey, tkey)
                    TO '{out}' (DELIMITER '\t', HEADER false)""")


def mode_pool(a):
    con = connect(a.threads)
    # key mapping checks
    bad = con.execute(f"""
        WITH x AS (SELECT key AS qkey, gene_id FROM read_parquet('{N}/query/native_lookup_crosswalk.parquet')),
        q AS (SELECT row_index, gene_id FROM read_parquet('{RD}/universe/genes.parquet') WHERE side='query')
        SELECT count(*) FILTER (WHERE x.qkey IS DISTINCT FROM u.mmseqs_key), count(*)
        FROM q JOIN ukey u ON u.side='query' AND u.row_index=q.row_index LEFT JOIN x USING (gene_id)""").fetchone()
    assert bad[0] == 0 and bad[1] == 67148, bad
    bad = con.execute(f"""SELECT count(*) FILTER (WHERE mmseqs_key <> CAST(split_part(gene_id, ':', 2) AS BIGINT)), count(*)
        FROM read_parquet('{RD}/universe/genes.parquet') WHERE side='ref'""").fetchone()
    assert bad[0] == 0 and bad[1] == 4190383, bad
    files = [str(W / "work" / m / "pool.parquet") for m in MODELS]
    con.execute(f"""CREATE TABLE P AS
        SELECT DISTINCT q.mmseqs_key AS qkey, t.mmseqs_key AS tkey
        FROM read_parquet({files}) p
        JOIN ukey q ON q.side='query' AND q.row_index=p.query_row_index
        JOIN ukey t ON t.side='ref' AND t.row_index=p.target_row_index
        WHERE p.kind='pair'""")
    n_pool = con.execute("SELECT count(*) FROM P").fetchone()[0]
    con.execute(f"""CREATE TABLE C AS SELECT qkey, tkey FROM
        read_csv('{S}/work/aln_seq.tsv', delim='\t', header=false, {ALN_COLS}) USING SAMPLE reservoir(20000 ROWS) REPEATABLE (17)""")
    con.execute("CREATE TABLE ALLP AS SELECT * FROM P UNION SELECT * FROM C")
    n_all = con.execute("SELECT count(*) FROM ALLP").fetchone()[0]
    out = W / "align/pool"
    write_pref(con, "ALLP", out / "pref.tsv")
    con.execute(f"COPY C TO '{out}/control_pairs.parquet' (FORMAT parquet)")
    write_json(out / "pairs.json", dict(pool_pairs=n_pool, control_pairs=20000, total=n_all, models=MODELS))
    log("pool pairs", n_pool, "total", n_all)


def mode_missing(a):
    con = connect(a.threads)
    load_labels(con)
    files = [str(f) for m in MODELS for d in ([m] + ([f"{m}_{a.tag}"] if a.tag else []))
             for f in sorted((W / "runs" / d).glob("anchors__*.parquet"))]
    log("anchor files", len(files))
    con.execute(f"""CREATE TABLE A AS
        SELECT DISTINCT q.mmseqs_key AS qkey, t.mmseqs_key AS tkey
        FROM read_parquet({files}) p
        JOIN ukey q ON q.side='query' AND q.row_index=p.query_row_index
        JOIN ukey t ON t.side='ref' AND t.row_index=p.target_row_index""")
    con.execute("CREATE TABLE M AS SELECT * FROM A ANTI JOIN lab USING (qkey, tkey)")
    n_a, n_m = (con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("A", "M"))
    out = W / "align" / ("missing" + (f"_{a.tag}" if a.tag else ""))
    write_pref(con, "M", out / "pref.tsv")
    write_json(out / "pairs.json", dict(anchor_pairs=n_a, missing=n_m, files=len(files)))
    log("anchor pairs", n_a, "missing", n_m)


def mode_check(a):
    con = connect(a.threads)
    out = W / "align/pool"
    r = con.execute(f"""
        WITH c AS (SELECT * FROM read_parquet('{out}/control_pairs.parquet')),
        old AS (SELECT * FROM read_csv('{S}/work/aln_seq.tsv', delim='\t', header=false, {ALN_COLS})),
        new AS (SELECT * FROM read_csv('{out}/aln.tsv', delim='\t', header=false, {ALN_COLS}))
        SELECT count(*) AS n, count(new.qkey) AS n_new,
               sum((old.evalue = new.evalue)::INT) AS same_evalue,
               sum(((old.evalue <= 1e-3) = (new.evalue <= 1e-3))::INT) AS same_label,
               max(abs(old.bits - new.bits)) AS max_bits_diff
        FROM c JOIN old USING (qkey, tkey) LEFT JOIN new USING (qkey, tkey)""").df()
    log(r.to_dict("records"))
    write_json(out / "control_check.json", r.to_dict("records")[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "missing", "check"])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dict(pool=mode_pool, missing=mode_missing, check=mode_check)[a.mode](a)
