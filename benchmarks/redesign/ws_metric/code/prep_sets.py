#!/usr/bin/env python
"""Evaluation populations and training positives, keyed by universe gid.

Writes to $RD/ws_metric/sets/:
  heldout.json            held-out and training query genomes
  qset.parquet            every query protein: gid, genome, heldout flag, finite flag per model
  popA_<m>.parquet        uniform hash sample of model m's original ANN60 seeds (all queries),
                          sequence label from $S/work/aln_seq.tsv
  xphy.parquet            scaffold_k5 calibrated in-block pairs of held-out queries (all strata),
                          sequence label from $S/work_retrieval/aln_seq.tsv
  popB.parquet            sequence-invisible in-block pairs (union over 7 models) of held-out
                          queries with a TM-score in $S/work_key/tm_key.parquet
  pos_<m>.parquet         anchor pairs of model m's blocks (>= 6 anchors or accept05), with
                          sequence label, query/target genome codes and strata
  sd_heldout.parquet      native Spacedust pairs of held-out queries
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from common import (ACTINO_STRATA, MODELS, N_REF, S, SCALE, SETS, U, ROOT, load_vectors, log)

ALL7 = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esmc300m", "esmc600m", "esmc6b"]
SEQ_COLS = ("names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'], "
            "types={'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--only", default="", help="comma list of outputs to build (default all)")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    want = lambda k: not only or k in only
    SETS.mkdir(parents=True, exist_ok=True)
    tmp = SETS.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{tmp}'")
    n = lambda q: con.execute(q).fetchone()[0]

    con.execute(f"CREATE TABLE GN AS SELECT * FROM read_parquet('{U}/genomes.parquet')")
    con.execute(f"""CREATE TABLE QG AS SELECT gid AS qgid, row_index AS qrow, mmseqs_key AS qkey,
                    gene_id AS qgene, genome AS qgenome, genome_code AS qgc
                    FROM read_parquet('{U}/genes.parquet') WHERE side = 'query'""")
    con.execute(f"""CREATE TABLE TG AS SELECT g.gid AS tgid, g.row_index AS trow, g.mmseqs_key AS tkey,
                    g.gene_id AS tgene, g.genome_code AS tgc, GN.stratum AS tstratum, GN.phylum AS tphylum
                    FROM read_parquet('{U}/genes.parquet') g JOIN GN USING (genome_code) WHERE g.side = 'ref'""")
    # crosswalk checks against tonight's tables
    assert n(f"""SELECT count(*) FROM read_parquet('{S}/work/qx.parquet') x JOIN QG
                 ON QG.qrow = x.row_index AND QG.qkey = x.qkey AND QG.qgene = x.gene_id""") == 67148
    assert n(f"""SELECT count(*) FROM read_parquet('{S}/work/tx.parquet') x JOIN TG
                 ON TG.trow = x.trow AND TG.tkey = x.tkey AND TG.tgene = x.tgene""") == N_REF
    assert n("SELECT count(DISTINCT qkey) FROM QG") == 67148 and n("SELECT count(DISTINCT tkey) FROM TG") == N_REF

    names = sorted(r[0] for r in con.execute("SELECT DISTINCT qgenome FROM QG").fetchall())
    held = names[::3]
    train = [x for x in names if x not in held]
    codes = dict(con.execute("SELECT DISTINCT qgenome, qgc FROM QG").fetchall())
    info = dict(heldout=held, training=train, heldout_codes=[codes[x] for x in held],
                training_codes=[codes[x] for x in train], actino_strata=list(ACTINO_STRATA))
    json.dump(info, open(SETS / "heldout.json", "w"), indent=1)
    con.execute("CREATE TABLE HG (g VARCHAR)")
    con.executemany("INSERT INTO HG VALUES (?)", [(x,) for x in held])
    log("held-out query genomes", held)
    stats = {"heldout": held, "training": train}

    # ---- query set with per-model finite flags --------------------------------------
    if want("qset"):
        q = con.execute("SELECT qgid, qrow, qkey, qgenome, qgc, qgenome IN (SELECT g FROM HG) AS heldout FROM QG ORDER BY qrow").df()
        for m in MODELS:
            qv, _ = load_vectors(m)
            q[f"fin_{m}"] = np.isfinite(np.asarray(qv)).all(axis=1)
        q["fin_all"] = q[[f"fin_{m}" for m in MODELS]].all(axis=1)
        q.to_parquet(SETS / "qset.parquet", index=False)
        stats["qset"] = dict(n=len(q), heldout=int(q.heldout.sum()), heldout_fin_all=int((q.heldout & q.fin_all).sum()),
                             **{f"nonfinite_{m}": int((~q[f"fin_{m}"]).sum()) for m in MODELS})
        log("qset", stats["qset"])

    con.execute(f"""CREATE TABLE SEQ AS SELECT qkey, tkey, evalue, seqid
                    FROM read_csv('{S}/work/aln_seq.tsv', delim='\t', header=false, {SEQ_COLS})""")
    log("seq table", n("SELECT count(*) FROM SEQ"), f"{time.time()-t0:.0f}s")
    prep = json.load(open(S / "work/prep_stats.json"))

    # ---- popA: uniform hash sample of original seeds ------------------------------
    if want("popA"):
        for m in MODELS:
            base = SCALE / "results" / m / "analysis/full_workflow"
            thr = prep["models"][m]["hash_threshold_seed"]
            df = con.execute(f"""
                WITH s AS (SELECT query_row_index AS qrow, target_row_index AS trow, similarity AS sim,
                                  row_number() OVER (PARTITION BY query_row_index ORDER BY similarity DESC) AS rank_q
                           FROM read_parquet('{base}/seeds.parquet'))
                SELECT QG.qgid, TG.tgid, s.sim, s.rank_q, QG.qgc, TG.tgc, TG.tstratum,
                       QG.qgenome IN (SELECT g FROM HG) AS heldout,
                       seq.evalue, seq.seqid, seq.qkey IS NOT NULL AS scored
                FROM s JOIN QG ON QG.qrow = s.qrow JOIN TG ON TG.trow = s.trow
                LEFT JOIN SEQ seq ON seq.qkey = QG.qkey AND seq.tkey = TG.tkey
                WHERE hash(CAST(QG.qkey AS BIGINT) * 5000000 + CAST(TG.tkey AS BIGINT)) % 1000 < {thr}""").df()
            frac = df.scored.mean()
            assert frac > 0.999, (m, frac)
            df = df[df.scored].drop(columns="scored")
            df["vis"] = df.evalue <= 1e-3
            df.to_parquet(SETS / f"popA_{m}.parquet", index=False)
            stats[f"popA_{m}"] = dict(n=len(df), scored_frac=float(frac), thr=thr, heldout=int(df.heldout.sum()),
                                      vis_rate=float(df.vis.mean()), heldout_vis_rate=float(df[df.heldout].vis.mean()),
                                      heldout_other_phylum=int((df.heldout & (df.tstratum == "3_other_phylum")).sum()))
            log("popA", m, stats[f"popA_{m}"])

    # ---- xphy: scaffold_k5 calibrated in-block pairs ----------------------------------
    if want("xphy"):
        K = S / "work_retrieval"
        con.execute(f"""CREATE TABLE SEQR AS SELECT qkey, tkey, evalue, seqid
                        FROM read_csv('{K}/aln_seq.tsv', delim='\t', header=false, {SEQ_COLS})""")
        df = con.execute(f"""
            SELECT QG.qgid, TG.tgid, QG.qgc, TG.tgc, TG.tstratum, max(BP.n_anchors) AS n_anchors_max,
                   any_value(r.evalue) AS evalue, any_value(r.seqid) AS seqid, bool_and(r.qkey IS NOT NULL) AS scored
            FROM read_parquet('{K}/block_pairs.parquet') BP
            JOIN QG ON QG.qkey = BP.qkey JOIN TG ON TG.tkey = BP.tkey
            LEFT JOIN SEQR r ON r.qkey = BP.qkey AND r.tkey = BP.tkey
            WHERE BP.arm = 'scaffold_k5' AND QG.qgenome IN (SELECT g FROM HG)
            GROUP BY ALL""").df()
        frac = df.scored.mean()
        assert frac > 0.999, frac
        df = df[df.scored].drop(columns="scored")
        df["vis"] = df.evalue <= 1e-3
        df.to_parquet(SETS / "xphy.parquet", index=False)
        stats["xphy"] = dict(n=len(df), scored_frac=float(frac),
                             by_stratum=df.groupby("tstratum").agg(n=("vis", "size"), vis=("vis", "mean")).reset_index().to_dict("records"))
        log("xphy", stats["xphy"])

    # ---- popB: invisible in-block pairs with TM ------------------------------------
    if want("popB"):
        tmp_ = S / "work_key/tm_key.parquet"
        if tmp_.exists():
            df = con.execute(f"""
                SELECT QG.qgid, TG.tgid, QG.qgc, TG.tgc, TG.tstratum, any_value(tm.tm_max) AS tm_max,
                       any_value(tm.tm_min) AS tm_min, any_value(tm.lddt) AS lddt,
                       bool_or(B.accept05) AS any_accept05, max(B.n_anchors) AS n_anchors_max,
                       count(DISTINCT B.model) AS n_models
                FROM read_parquet('{S}/work_key/block_pairs.parquet') B
                JOIN read_parquet('{tmp_}') tm USING (qkey, tkey)
                JOIN QG ON QG.qkey = B.qkey JOIN TG ON TG.tkey = B.tkey
                WHERE NOT B.vis AND QG.qgenome IN (SELECT g FROM HG)
                GROUP BY ALL""").df()
            df["tm05"] = df.tm_max >= 0.5
            df.to_parquet(SETS / "popB.parquet", index=False)
            stats["popB"] = dict(n=len(df), tm05_rate=float(df.tm05.mean()),
                                 by_stratum=df.groupby("tstratum").agg(n=("tm05", "size"), tm05=("tm05", "mean")).reset_index().to_dict("records"))
            log("popB", stats["popB"])
        else:
            log("popB skipped: tm_key.parquet absent")

    # ---- training positives ------------------------------------------------------------
    if want("pos"):
        for m in MODELS:
            base = SCALE / "results" / m / "analysis/full_workflow"
            calls = S / "results/permuted_panels" / f"{m}_block_calls.parquet"
            con.execute(f"""
                CREATE OR REPLACE TABLE blk AS
                SELECT A.block_id, A.n_anchors, C.accept05,
                       from_json(A.anchor_query_gene_ids, '["VARCHAR"]') AS q,
                       CASE WHEN A.orientation = -1
                            THEN list_reverse(from_json(A.anchor_target_gene_ids, '["VARCHAR"]'))
                            ELSE from_json(A.anchor_target_gene_ids, '["VARCHAR"]') END AS t
                FROM read_parquet('{base}/all_blocks.parquet') A JOIN read_parquet('{calls}') C USING (block_id)
                WHERE A.n_anchors >= 6 OR C.accept05""")
            n_blk, n_anc = con.execute("SELECT count(*), sum(n_anchors) FROM blk").fetchone()
            assert n("SELECT count(*) FROM blk WHERE len(q) <> n_anchors OR len(t) <> n_anchors") == 0
            con.execute("""
                CREATE OR REPLACE TABLE e AS
                SELECT block_id, n_anchors, accept05, unnest(q) AS qg, unnest(t) AS tg FROM blk""")
            con.execute("""
                CREATE OR REPLACE TABLE P AS
                SELECT QG.qgid, TG.tgid, QG.qrow, TG.trow, QG.qkey, TG.tkey, QG.qgc, TG.tgc, TG.tstratum,
                       QG.qgenome IN (SELECT g FROM HG) AS q_heldout,
                       max(e.n_anchors) AS n_anchors_max, bool_or(e.n_anchors >= 6) AS src_ge6,
                       bool_or(e.accept05) AS src_acc05, count(*) AS n_rows
                FROM e JOIN QG ON QG.qgene = e.qg JOIN TG ON TG.tgene = e.tg
                GROUP BY ALL""")
            assert n("SELECT sum(n_rows) FROM P") == n_anc, "anchor rows lost in the gene-id join"
            # trap 2: every derived pair must be a retained anchor
            missing = n(f"""SELECT count(*) FROM P WHERE NOT EXISTS (
                             SELECT 1 FROM read_parquet('{base}/anchors.parquet') a
                             WHERE a.query_row_index = P.qrow AND a.target_row_index = P.trow)""")
            assert missing == 0, (m, "pairs absent from anchors.parquet", missing)
            df = con.execute("""SELECT P.*, s.evalue, s.seqid, s.qkey IS NOT NULL AS scored
                                FROM P LEFT JOIN SEQ s USING (qkey, tkey)""").df()
            frac = df.scored.mean()
            assert frac > 0.999, (m, frac)
            df["vis"] = df.evalue <= 1e-3
            df = df.drop(columns=["scored", "n_rows"])
            df.to_parquet(SETS / f"pos_{m}.parquet", index=False)
            tr = df[~df.q_heldout & df.tstratum.isin(ACTINO_STRATA)]
            stats[f"pos_{m}"] = dict(blocks=int(n_blk), anchor_rows=int(n_anc), pairs=len(df),
                                     train_ge6=int(tr.src_ge6.sum()), train_acc05=int(tr.src_acc05.sum()),
                                     train_vis=int((tr.src_ge6 & tr.vis).sum()),
                                     train_ge6_vis_rate=float(tr[tr.src_ge6].vis.mean()),
                                     train_acc05_vis_rate=float(tr[tr.src_acc05].vis.mean()),
                                     excluded_other_phylum_ge6=int((~df.q_heldout & ~df.tstratum.isin(ACTINO_STRATA) & df.src_ge6).sum()))
            log("pos", m, stats[f"pos_{m}"])

    # ---- native Spacedust pairs of held-out queries ------------------------------------
    if want("sd"):
        df = con.execute(f"""
            SELECT QG.qgid, TG.tgid, TG.tstratum FROM read_parquet('{S}/work/sd_pairs.parquet') sd
            JOIN QG ON QG.qkey = sd.qkey JOIN TG ON TG.tkey = sd.tkey
            WHERE QG.qgenome IN (SELECT g FROM HG)""").df()
        df.to_parquet(SETS / "sd_heldout.parquet", index=False)
        stats["sd_heldout"] = dict(n=len(df), queries=int(df.qgid.nunique()),
                                   other_phylum=int((df.tstratum == "3_other_phylum").sum()))
        log("sd", stats["sd_heldout"])

    stats["seconds"] = round(time.time() - t0, 1)
    out = SETS / ("prep_stats.json" if not only else f"prep_stats_{'_'.join(sorted(only))}.json")
    json.dump(stats, open(out, "w"), indent=1, default=str)
    log("done", stats["seconds"])


if __name__ == "__main__":
    main()
