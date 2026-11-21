#!/usr/bin/env python
"""Labelled gene-pair sets in gid space for token pair recall.

Sets (all query x reference, so all cross-genome):
  sd_vis / sd_invis       native Spacedust cluster-hit pairs, split by pairwise SW E <= 1e-3
  blk650_vis / _invis     anchor pairs in calibrated (accept05) ESM2-650M blocks, split the same way
  tm_hi / tm_lo           sequence-invisible pairs in calibrated blocks of any model with both
                          AFDB structures (key_subset 'accepted_invisible'), TM >= 0.5 / < 0.5
                          normalized by the shorter chain (tm_max); written only once tm_key exists
  random                  300,000 random query x reference pairs (both with structures)

Keys are mapped through the universe table (trap 1). Block pairs come from key_subset's
block_pairs.parquet, which reverses the target list of orientation -1 blocks (trap 2);
every ESM2-650M pair is checked against that model's anchors.parquet here.
"""
import argparse
import json
import os
import time

import duckdb

from common import S, SCALE, UNIVERSE, WS, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = WS / "labels"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{WS}/tmp_duck'")
    con.execute(f"""CREATE TABLE uq AS SELECT mmseqs_key AS qkey, gid AS g1, row_index AS qrow
                    FROM read_parquet('{UNIVERSE}') WHERE side = 'query'""")
    con.execute(f"""CREATE TABLE ut AS SELECT mmseqs_key AS tkey, gid AS g2, row_index AS trow
                    FROM read_parquet('{UNIVERSE}') WHERE side = 'ref'""")
    con.execute(f"""
        CREATE TABLE seq AS
        SELECT qkey, tkey, min(evalue) AS evalue FROM read_csv('{S}/work/aln_seq.tsv', delim='\t', header=false,
            names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'],
            types={{'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}})
        GROUP BY ALL""")
    stats = {"aln_seq_pairs": con.execute("SELECT count(*) FROM seq").fetchone()[0]}
    con.execute("CREATE TABLE L (lset VARCHAR, qkey BIGINT, tkey BIGINT)")
    # native Spacedust
    con.execute(f"""
        INSERT INTO L
        SELECT CASE WHEN COALESCE(s.evalue <= 1e-3, FALSE) THEN 'sd_vis' ELSE 'sd_invis' END, p.qkey, p.tkey
        FROM (SELECT DISTINCT qkey, tkey FROM read_parquet('{S}/work/sd_pairs.parquet')) p
        LEFT JOIN seq s USING (qkey, tkey)""")
    stats["sd_pairs_without_alignment_row"] = con.execute(f"""
        SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM read_parquet('{S}/work/sd_pairs.parquet')) p
        ANTI JOIN seq s USING (qkey, tkey)""").fetchone()[0]
    # calibrated ESM2-650M blocks
    con.execute(f"""
        CREATE TABLE b650 AS SELECT DISTINCT qkey, tkey, vis FROM read_parquet('{S}/work_key/block_pairs.parquet')
        WHERE model = 'esm2_650m' AND accept05""")
    con.execute("""INSERT INTO L SELECT CASE WHEN vis THEN 'blk650_vis' ELSE 'blk650_invis' END, qkey, tkey FROM b650""")
    anchors = SCALE / "results/esm2_650m/analysis/full_workflow/anchors.parquet"
    n650, found = con.execute(f"""
        WITH a AS (SELECT DISTINCT query_row_index AS qrow, target_row_index AS trow FROM read_parquet('{anchors}'))
        SELECT count(*), count(a.qrow) FROM b650 JOIN uq USING (qkey) JOIN ut USING (tkey)
        LEFT JOIN a ON a.qrow = uq.qrow AND a.trow = ut.trow""").fetchone()
    stats["blk650_pairs"] = n650
    stats["blk650_pairs_found_in_anchors"] = found
    assert found == n650, f"ESM2-650M block pairs absent from anchors.parquet: {n650 - found} of {n650}"
    # TM sets
    tm = S / "work_key/tm_key.parquet"
    stats["tm_key_present"] = tm.exists()
    if tm.exists():
        con.execute(f"""
            INSERT INTO L
            SELECT CASE WHEN t.tm_max >= 0.5 THEN 'tm_hi' ELSE 'tm_lo' END, k.qkey, k.tkey
            FROM (SELECT DISTINCT qkey, tkey FROM read_parquet('{S}/work_key/keyset.parquet')
                  WHERE kset = 'accepted_invisible') k
            JOIN read_parquet('{tm}') t USING (qkey, tkey)""")
        stats["accepted_invisible_keyset"] = con.execute(f"""SELECT count(DISTINCT (qkey, tkey))
            FROM read_parquet('{S}/work_key/keyset.parquet') WHERE kset = 'accepted_invisible'""").fetchone()[0]
    # random
    con.execute(f"""INSERT INTO L SELECT DISTINCT 'random', qkey, tkey FROM read_parquet('{S}/work/membership.parquet')
                    WHERE cls = 'random'""")
    con.execute("""CREATE TABLE LG AS SELECT L.lset, uq.g1, ut.g2 FROM L JOIN uq USING (qkey) JOIN ut USING (tkey)""")
    n_l, n_lg = con.execute("SELECT (SELECT count(*) FROM L), (SELECT count(*) FROM LG)").fetchone()
    assert n_l == n_lg, f"unmapped label pairs: {n_l - n_lg}"
    stats["sets"] = dict(con.execute("SELECT lset, count(*) FROM LG GROUP BY ALL ORDER BY 1").fetchall())
    tmp = out / f".label_pairs.parquet.tmp{os.getpid()}"
    con.execute(f"COPY (SELECT * FROM LG ORDER BY lset, g1, g2) TO '{tmp}' (FORMAT parquet)")
    os.replace(tmp, out / "label_pairs.parquet")
    stats["seconds"] = round(time.time() - t0, 1)
    with open(out / "label_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    stamp(json.dumps(stats))


if __name__ == "__main__":
    main()
