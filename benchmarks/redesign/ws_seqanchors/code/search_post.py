#!/usr/bin/env python
"""Turn the MMseqs2 query x reference search into keyed sequence-anchor tables.

Reads hits by key (prefixid --tsv; trap 6), maps keys to ELSA row indices through the
shared gene universe, ranks hits per query by bit score and writes:

  work/hits.parquet        qrow, trow, qkey, tkey, bits, seqid, evalue, qcov, tcov,
                           rank_all (1 = best by bits), cov50, rank_cov50 (rank among cov50 hits)
  work/bits_top60_sorted.npy   sorted bit scores of S-top60 (defines the bit-score percentile weight)
  results/search_stats.json
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
W = RD / "ws_seqanchors"
U = RD / "universe/genes.parquet"


def main():
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--max-seqs", type=int, required=True)
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=W)
    a = ap.parse_args()
    W = a.base
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    con.execute(f"""
        CREATE TABLE raw AS SELECT * FROM read_csv('{W}/work/hits.tsv', delim='\t', header=false,
            names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'],
            types={{'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE',
                    'qstart':'BIGINT','qend':'BIGINT','qlen':'BIGINT','tstart':'BIGINT','tend':'BIGINT','tlen':'BIGINT'}})""")
    n_raw = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    assert con.execute("SELECT max(evalue) <= 1e-3 FROM raw").fetchone()[0]
    con.execute(f"CREATE TABLE uq AS SELECT mmseqs_key, row_index, genome FROM read_parquet('{U}') WHERE side='query'")
    con.execute(f"CREATE TABLE ur AS SELECT mmseqs_key, row_index, genome FROM read_parquet('{U}') WHERE side='ref'")
    assert con.execute("SELECT count(*) = count(DISTINCT mmseqs_key) FROM uq").fetchone()[0]
    assert con.execute("SELECT count(*) = count(DISTINCT mmseqs_key) FROM ur").fetchone()[0]
    con.execute("""
        CREATE TABLE h AS
        SELECT uq.row_index AS qrow, ur.row_index AS trow, r.qkey, r.tkey, r.bits, r.seqid, r.evalue,
               (r.qend - r.qstart + 1)::DOUBLE / r.qlen AS qcov, (r.tend - r.tstart + 1)::DOUBLE / r.tlen AS tcov,
               uq.genome AS qgenome, ur.genome AS tgenome
        FROM raw r JOIN uq ON uq.mmseqs_key = r.qkey JOIN ur ON ur.mmseqs_key = r.tkey""")
    n_h = con.execute("SELECT count(*) FROM h").fetchone()[0]
    assert n_h == n_raw, (n_h, n_raw)
    same = con.execute("SELECT count(*) FROM h WHERE qgenome = tgenome").fetchone()[0]
    assert same == 0, same
    dup = con.execute("SELECT count(*) - count(DISTINCT (qrow, trow)) FROM h").fetchone()[0]
    assert dup == 0, dup
    con.execute("""
        CREATE TABLE hr AS
        SELECT qrow, trow, qkey, tkey, bits, seqid, evalue, qcov, tcov,
               row_number() OVER (PARTITION BY qrow ORDER BY bits DESC, evalue ASC, trow ASC) AS rank_all,
               (qcov >= 0.5 AND tcov >= 0.5) AS cov50
        FROM h""")
    con.execute("""
        CREATE TABLE hits AS
        SELECT hr.*, CASE WHEN cov50 THEN row_number() OVER (PARTITION BY qrow, cov50
                     ORDER BY bits DESC, evalue ASC, trow ASC) END AS rank_cov50
        FROM hr ORDER BY qrow, rank_all""")
    con.execute(f"COPY hits TO '{W}/work/hits.parquet' (FORMAT parquet, ROW_GROUP_SIZE 1000000)")
    bits = con.execute("SELECT bits FROM hits WHERE rank_all <= 60 ORDER BY bits").fetchnumpy()["bits"]
    np.save(W / "work/bits_top60_sorted.npy", bits.astype(np.float64))

    con.execute(f"""CREATE TABLE pc AS SELECT * FROM read_csv('{W}/work/pref_count.tsv', delim='\t', header=false,
                    names=['qkey','n_pref'], types={{'qkey':'BIGINT','n_pref':'BIGINT'}})""")
    per_q = con.execute("""
        WITH q AS (SELECT qkey, count(*) AS n_hits, sum(cov50::INT) AS n_cov50,
                          max(bits) FILTER (WHERE rank_all = 1) AS best,
                          max(bits) FILTER (WHERE rank_all = 60) AS b60,
                          min(bits) AS worst FROM hits GROUP BY qkey)
        SELECT pc.qkey, pc.n_pref, COALESCE(q.n_hits, 0) AS n_hits, COALESCE(q.n_cov50, 0) AS n_cov50,
               q.best, q.b60, q.worst
        FROM pc LEFT JOIN q USING (qkey)""").df()
    sat = per_q[per_q.n_pref >= a.max_seqs]
    stats = dict(
        hits_e1e3=int(n_raw), queries_total=67148, queries_in_pref_table=int(len(per_q)),
        queries_with_hit=int((per_q.n_hits > 0).sum()),
        hits_per_query_quantiles={str(q): float(per_q.n_hits.quantile(q)) for q in (.1, .25, .5, .75, .9, .99, 1.0)},
        s_all=int(n_raw),
        s_top60=int(con.execute("SELECT count(*) FROM hits WHERE rank_all <= 60").fetchone()[0]),
        s_top60_cov50=int(con.execute("SELECT count(*) FROM hits WHERE rank_cov50 <= 60").fetchone()[0]),
        cov50_hits=int(con.execute("SELECT count(*) FROM hits WHERE cov50").fetchone()[0]),
        queries_ge60_hits=int((per_q.n_hits >= 60).sum()),
        queries_ge60_cov50=int((per_q.n_cov50 >= 60).sum()),
        prefilter_max_seqs=a.max_seqs,
        prefilter_entries=int(per_q.n_pref.sum()),
        queries_prefilter_saturated=int(len(sat)),
        saturated_queries_hits_quantiles={str(q): float(sat.n_hits.quantile(q)) for q in (.1, .5, .9, 1.0)} if len(sat) else None,
        saturated_queries_with_ge_maxseqs_hits=int((sat.n_hits >= a.max_seqs).sum()),
        saturated_rank60_over_best_median=float((sat.b60 / sat.best).median()) if len(sat) else None,
        saturated_worst_over_rank60_median=float((sat.worst / sat.b60).median()) if len(sat) else None,
        median_seqid_top60=float(con.execute("SELECT median(seqid) FROM hits WHERE rank_all <= 60").fetchone()[0]),
        median_seqid_all=float(con.execute("SELECT median(seqid) FROM hits").fetchone()[0]),
        bits_top60_quantiles={str(q): float(np.quantile(bits, q)) for q in (0, .1, .5, .9, 1)},
        post_seconds=round(time.time() - t0, 1),
    )
    stats["timing"] = json.loads(a.timing.read_text())
    per_q.to_parquet(W / "work/per_query_hits.parquet", index=False)
    (W / "results/search_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
