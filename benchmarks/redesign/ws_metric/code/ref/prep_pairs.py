#!/usr/bin/env python
"""Build pair sets for the sequence-invisible / structure-confirmed analysis.

Reads saved ELSA blocks/anchors/seeds for each model and the native Spacedust
blocks, maps every protein to its MMseqs database key, and writes:

  work/membership.parquet   (model, cls, qkey, tkey, similarity, is_original,
                             n_anchors_max, n_blocks)
  work/pairs.parquet        unique (qkey, tkey) with has_struct
  work/sd_pairs.parquet     native Spacedust pairs
  work/qx.parquet           query crosswalk
  work/pref_all.tsv         fake prefilter input for mmseqs align
  work/pref_struct.tsv      fake prefilter input for foldseek tmalign
  work/tref_h.tsv           header entries "K<key>" for the aliased target DB
  work/prep_stats.json
"""
import argparse
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd

R = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust"
N = f"{R}/native_run_20260910"
MODELS = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m",
          "esmc300m", "esmc600m", "esmc6b"]
RNG_SEED = 17
HASH = "hash(CAST({q} AS BIGINT) * 5000000 + CAST({t} AS BIGINT)) % 1000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit blocks per source (0 = all)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--control-target", type=int, default=400_000)
    ap.add_argument("--random-pairs", type=int, default=300_000)
    a = ap.parse_args()

    models = [m for m in a.models.split(",") if m]
    W = f"{a.work}/work"
    os.makedirs(W, exist_ok=True)
    os.makedirs(f"{a.work}/tmp", exist_ok=True)
    t0 = time.time()
    stats = {"smoke": a.smoke, "models": {}, "threads": a.threads}

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{a.work}/tmp'")
    con.execute("SET preserve_insertion_order=false")

    # --- identity tables -------------------------------------------------
    con.execute(f"""
        CREATE TABLE qx AS
        SELECT key AS qkey, native_id, name AS qname, gene_id, row_index,
               sample_id, contig_id, position_index, sequence_length
        FROM read_parquet('{N}/query/native_lookup_crosswalk.parquet')""")
    con.execute(f"""
        CREATE TABLE tl AS SELECT * FROM read_csv(
            '{N}/reference/keggclusterdb.lookup', delim='\t', header=false,
            columns={{'tkey':'BIGINT','tname':'VARCHAR','tset':'BIGINT'}})""")
    con.execute(f"""
        CREATE TABLE ts AS SELECT * FROM read_csv(
            '{N}/reference/keggclusterdb.source', delim='\t', header=false,
            columns={{'tset':'BIGINT','tgenome':'VARCHAR'}})""")
    # ELSA reference rows are in their own order; source_sequence_db_key is the
    # MMseqs database key (and the number inside gene_id).
    con.execute(f"""
        CREATE TABLE tx AS
        SELECT row_index AS trow, source_sequence_db_key AS tkey, gene_id AS tgene,
               sample_id AS tgenome, contig_id AS tcontig, position_index AS tpos,
               sequence_length AS tlen
        FROM read_parquet('{R}/results/esm2_t6/shard_000/export/metadata.parquet')""")
    assert con.execute("""
        SELECT count(*) = 4190383 AND count(DISTINCT trow) = 4190383
           AND count(DISTINCT tkey) = 4190383
           AND bool_and(tkey = CAST(split_part(tgene, ':', 2) AS BIGINT))
        FROM tx""").fetchone()[0], "reference metadata row/key mapping is not one-to-one"
    idx_cols = "columns={'key':'BIGINT','off':'BIGINT','len':'BIGINT'}"
    con.execute(f"""
        CREATE TABLE qfs AS SELECT a.key FROM
          read_csv('{N}/query/querydb_foldseek.index', delim='\t', header=false, {idx_cols}) a
          JOIN read_csv('{N}/query/querydb_foldseek_ca.index', delim='\t', header=false, {idx_cols}) b
          USING (key)""")
    con.execute(f"""
        CREATE TABLE tfs AS SELECT a.key FROM
          read_csv('{N}/reference/keggclusterdb_foldseek.index', delim='\t', header=false, {idx_cols}) a
          JOIN read_csv('{N}/reference/keggclusterdb_foldseek_ca.index', delim='\t', header=false, {idx_cols}) b
          USING (key)""")
    n = lambda q: con.execute(q).fetchone()[0]
    stats["n_query"] = n("SELECT count(*) FROM qx")
    stats["n_reference"] = n("SELECT count(*) FROM tl")
    stats["n_query_struct"] = n("SELECT count(*) FROM qfs")
    stats["n_reference_struct"] = n("SELECT count(*) FROM tfs")
    assert stats["n_query"] == 67148 and stats["n_reference"] == 4190383
    assert n("SELECT count(DISTINCT gene_id) FROM qx") == 67148
    assert n("SELECT count(DISTINCT row_index) FROM qx") == 67148
    print("identity tables", stats, f"{time.time()-t0:.1f}s", flush=True)

    con.execute("""
        CREATE TABLE membership (
            model VARCHAR, cls VARCHAR, qkey BIGINT, tkey BIGINT,
            similarity DOUBLE, is_original BOOLEAN,
            n_anchors_max BIGINT, n_blocks BIGINT)""")

    lim = f"LIMIT {a.smoke}" if a.smoke else ""
    target = a.control_target if not a.smoke else 2000

    # --- ELSA models -----------------------------------------------------
    for m in models:
        base = f"{R}/model_scaling_20260913/results/{m}/analysis/full_workflow"
        ms = {}
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE blk AS
            SELECT block_id, n_anchors,
                   from_json(anchor_query_gene_ids, '["VARCHAR"]') AS q,
                   -- Saved blocks list each side in ascending gene order, so the
                   -- target list of a reverse-orientation block pairs back to front
                   -- (verified against anchors.parquet: 4,225/4,225 and 4,067/4,067).
                   CASE WHEN orientation = -1
                        THEN list_reverse(from_json(anchor_target_gene_ids, '["VARCHAR"]'))
                        ELSE from_json(anchor_target_gene_ids, '["VARCHAR"]') END AS t
            FROM (SELECT * FROM read_parquet('{base}/all_blocks.parquet') {lim})""")
        ms["n_blocks"], ms["sum_n_anchors"], bad = con.execute(
            "SELECT count(*), sum(n_anchors), "
            "sum(CASE WHEN len(q)<>n_anchors OR len(t)<>n_anchors THEN 1 ELSE 0 END) FROM blk"
        ).fetchone()
        assert bad == 0, f"{m}: anchor list length mismatch in {bad} blocks"
        con.execute("""
            CREATE OR REPLACE TEMP TABLE chained AS
            WITH e AS (SELECT block_id, n_anchors, unnest(q) AS qg, unnest(t) AS tg FROM blk)
            SELECT qx.qkey, qx.row_index AS qrow,
                   CAST(split_part(e.tg, ':', 2) AS BIGINT) AS tkey,
                   max(e.n_anchors) AS n_anchors_max, count(*) AS n_blocks
            FROM e JOIN qx ON qx.gene_id = e.qg
            GROUP BY ALL""")
        ms["chained_pairs"] = n("SELECT count(*) FROM chained")
        ms["chained_anchor_rows"] = n("SELECT sum(n_blocks) FROM chained")
        assert ms["chained_anchor_rows"] == ms["sum_n_anchors"], (m, ms)

        qf = "WHERE s.query_row_index IN (SELECT DISTINCT qrow FROM chained)" if a.smoke else ""
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE anch AS
            SELECT s.query_row_index AS qrow, tx.tkey AS tkey,
                   max(s.similarity) AS similarity, bool_or(s.is_original) AS is_original
            FROM read_parquet('{base}/anchors.parquet') s
            JOIN tx ON tx.trow = s.target_row_index {qf} GROUP BY ALL""")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE seed AS
            SELECT s.query_row_index AS qrow, tx.tkey AS tkey,
                   max(s.similarity) AS similarity
            FROM read_parquet('{base}/seeds.parquet') s
            JOIN tx ON tx.trow = s.target_row_index {qf} GROUP BY ALL""")
        ms["anchors"] = n("SELECT count(*) FROM anch")
        ms["seeds"] = n("SELECT count(*) FROM seed")
        matched = n("SELECT count(*) FROM chained c JOIN anch a USING (qrow, tkey)")
        ms["chained_found_in_anchors"] = matched
        frac = matched / max(ms["chained_pairs"], 1)
        assert frac > 0.999, f"{m}: chained pairs absent from retained anchors; matched {frac:.4f}"

        n_ua = n("""SELECT count(*) FROM anch a WHERE NOT EXISTS
                    (SELECT 1 FROM chained c WHERE c.qrow=a.qrow AND c.tkey=a.tkey)""")
        n_us = n("""SELECT count(*) FROM seed s WHERE NOT EXISTS
                    (SELECT 1 FROM chained c WHERE c.qrow=s.qrow AND c.tkey=s.tkey)""")
        thr_a = min(1000, int(np.ceil(1000 * target / max(n_ua, 1))))
        thr_s = min(1000, int(np.ceil(1000 * target / max(n_us, 1))))
        ms.update(anchors_unchained=n_ua, seeds_unchained=n_us,
                  hash_threshold_anchor=thr_a, hash_threshold_seed=thr_s)

        con.execute(f"""
            INSERT INTO membership
            SELECT '{m}', 'chained', c.qkey, c.tkey, a.similarity, a.is_original,
                   c.n_anchors_max, c.n_blocks
            FROM chained c LEFT JOIN anch a USING (qrow, tkey)""")
        con.execute(f"""
            INSERT INTO membership
            SELECT '{m}', 'anchor_unchained', qx.qkey, a.tkey, a.similarity,
                   a.is_original, NULL, NULL
            FROM anch a JOIN qx ON qx.row_index = a.qrow
            WHERE NOT EXISTS (SELECT 1 FROM chained c WHERE c.qrow=a.qrow AND c.tkey=a.tkey)
              AND {HASH.format(q='qx.qkey', t='a.tkey')} < {thr_a}""")
        con.execute(f"""
            INSERT INTO membership
            SELECT '{m}', 'seed_unchained', qx.qkey, s.tkey, s.similarity,
                   TRUE, NULL, NULL
            FROM seed s JOIN qx ON qx.row_index = s.qrow
            WHERE NOT EXISTS (SELECT 1 FROM chained c WHERE c.qrow=s.qrow AND c.tkey=s.tkey)
              AND {HASH.format(q='qx.qkey', t='s.tkey')} < {thr_s}""")
        stats["models"][m] = ms
        print(m, ms, f"{time.time()-t0:.1f}s", flush=True)

    # --- native Spacedust pairs -------------------------------------------
    con.execute(f"""
        CREATE TEMP TABLE sdb AS
        SELECT block_id, target_accession, n_anchors, cluster_pvalue,
               from_json(query_gene_ids_json, '["VARCHAR"]') AS q,
               from_json(target_gene_ids_json, '["VARCHAR"]') AS t
        FROM (SELECT * FROM read_parquet('{N}/results/all_blocks.parquet') {lim})""")
    sd_rows = n("SELECT sum(len(q)) FROM sdb")
    con.execute("""
        CREATE TABLE sd_pairs AS
        WITH e AS (SELECT block_id, target_accession, n_anchors, cluster_pvalue,
                          unnest(q) AS qn, unnest(t) AS tn FROM sdb)
        SELECT qx.qkey, tl.tkey, max(e.n_anchors) AS sd_n_anchors_max,
               count(*) AS sd_n, min(e.cluster_pvalue) AS sd_min_cluster_p
        FROM e JOIN qx ON qx.qname = e.qn
               JOIN ts ON ts.tgenome = e.target_accession
               JOIN tl ON tl.tset = ts.tset AND tl.tname = e.tn
        GROUP BY ALL""")
    sd_joined = n("SELECT sum(sd_n) FROM sd_pairs")
    stats["spacedust"] = {"hit_rows": sd_rows, "hit_rows_mapped": sd_joined,
                          "unique_pairs": n("SELECT count(*) FROM sd_pairs")}
    assert sd_joined >= 0.999 * sd_rows, stats["spacedust"]
    con.execute("""
        INSERT INTO membership
        SELECT 'spacedust', 'spacedust', qkey, tkey, NULL, NULL, sd_n_anchors_max, sd_n
        FROM sd_pairs""")
    print("spacedust", stats["spacedust"], f"{time.time()-t0:.1f}s", flush=True)

    # --- random pairs -----------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    nr = a.random_pairs if not a.smoke else 500
    qk = con.execute("SELECT key FROM qfs ORDER BY key").fetchnumpy()["key"]
    tk = con.execute("SELECT key FROM tfs ORDER BY key").fetchnumpy()["key"]
    rnd = pd.DataFrame({"qkey": rng.choice(qk, nr), "tkey": rng.choice(tk, nr)}).drop_duplicates()
    con.register("rnd", rnd)
    con.execute("""INSERT INTO membership
                   SELECT 'none', 'random', qkey, tkey, NULL, NULL, NULL, NULL FROM rnd""")
    stats["random_pairs"] = len(rnd)

    # --- unique pairs and tool inputs ------------------------------------
    con.execute("""
        CREATE TABLE pairs AS
        SELECT p.qkey, p.tkey,
               (p.qkey IN (SELECT key FROM qfs) AND p.tkey IN (SELECT key FROM tfs)) AS has_struct
        FROM (SELECT DISTINCT qkey, tkey FROM membership) p""")
    stats["unique_pairs"] = n("SELECT count(*) FROM pairs")
    stats["unique_pairs_with_struct"] = n("SELECT count(*) FROM pairs WHERE has_struct")
    stats["membership_rows"] = n("SELECT count(*) FROM membership")

    con.execute(f"COPY membership TO '{W}/membership.parquet' (FORMAT parquet)")
    con.execute(f"COPY pairs TO '{W}/pairs.parquet' (FORMAT parquet)")
    con.execute(f"COPY sd_pairs TO '{W}/sd_pairs.parquet' (FORMAT parquet)")
    con.execute(f"COPY qx TO '{W}/qx.parquet' (FORMAT parquet)")
    con.execute(f"COPY tx TO '{W}/tx.parquet' (FORMAT parquet)")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM pairs ORDER BY qkey, tkey)
                    TO '{W}/pref_all.tsv' (DELIMITER '\t', HEADER false)""")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM pairs WHERE has_struct
                          ORDER BY qkey, tkey)
                    TO '{W}/pref_struct.tsv' (DELIMITER '\t', HEADER false)""")
    con.execute(f"""COPY (SELECT key, 'K' || CAST(key AS VARCHAR) FROM tfs ORDER BY key)
                    TO '{W}/tref_h.tsv' (DELIMITER '\t', HEADER false)""")
    stats["seconds"] = round(time.time() - t0, 1)
    with open(f"{W}/prep_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "models"}), flush=True)


if __name__ == "__main__":
    main()
