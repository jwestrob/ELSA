#!/usr/bin/env python
"""Fast structural answer on the decisive subset of pairs.

prep:      choose pairs and write the fake prefilter input
summarize: join TM-scores and report confirmation by model, call and block size

Sets (all restricted to pairs with AFDB structures on both sides):
  accepted_invisible   every sequence-invisible pair in a calibrated (accept05) block
  rejected_invisible   hash sample of sequence-invisible pairs in rejected blocks
  accepted_visible     hash sample of sequence-visible pairs in calibrated blocks
  unchained_invisible  hash sample of sequence-invisible retained anchors outside blocks
  random               hash sample of the random pairs
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
MODELS = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esmc300m", "esmc600m", "esmc6b"]
H = "hash(CAST(qkey AS BIGINT) * 5000000 + CAST(tkey AS BIGINT)) % 1000"


def connect(run, threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{run}/tmp'")
    con.execute("SET preserve_insertion_order=false")
    return con


def prep(a):
    W, K = a.run / "work", a.run / "work_key"
    K.mkdir(parents=True, exist_ok=True)
    con = connect(a.run, a.threads)
    for t in ("qx", "pairs", "membership"):
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{W}/{t}.parquet')")
    con.execute(f"""
        CREATE TABLE seq AS
        SELECT qkey, tkey, evalue, seqid FROM read_csv('{W}/aln_seq.tsv', delim='\t', header=false,
            names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'],
            types={{'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}})""")
    con.execute("""CREATE TABLE BP (model VARCHAR, block_id BIGINT, n_anchors BIGINT, m DOUBLE,
                   accept05 BOOLEAN, accept01 BOOLEAN, qkey BIGINT, tkey BIGINT)""")
    for m in MODELS:
        base = SCALE / "results" / m / "analysis/full_workflow"
        calls = a.run / "results/permuted_panels" / f"{m}_block_calls.parquet"
        con.execute(f"""
            INSERT INTO BP
            WITH b AS (
                SELECT block_id, from_json(anchor_query_gene_ids, '["VARCHAR"]') AS q,
                       CASE WHEN orientation = -1
                            THEN list_reverse(from_json(anchor_target_gene_ids, '["VARCHAR"]'))
                            ELSE from_json(anchor_target_gene_ids, '["VARCHAR"]') END AS t
                FROM read_parquet('{base}/all_blocks.parquet')),
            e AS (SELECT block_id, unnest(q) AS qg, unnest(t) AS tg FROM b)
            SELECT '{m}', e.block_id, c.n_anchors, c.m, c.accept05, c.accept01, qx.qkey,
                   CAST(split_part(e.tg, ':', 2) AS BIGINT)
            FROM e JOIN qx ON qx.gene_id = e.qg JOIN read_parquet('{calls}') c USING (block_id)""")
    con.execute("""
        CREATE TABLE B AS
        SELECT BP.*, COALESCE(seq.evalue <= 1e-3, FALSE) AS vis, seq.seqid, seq.evalue AS seq_evalue,
               pairs.has_struct
        FROM BP JOIN pairs USING (qkey, tkey) LEFT JOIN seq USING (qkey, tkey)""")
    stats = {}
    stats["block_pairs"] = con.execute("SELECT count(*) FROM B").fetchone()[0]
    n_rej = con.execute("SELECT count(DISTINCT (qkey, tkey)) FROM B WHERE NOT accept05 AND NOT vis AND has_struct").fetchone()[0]
    n_accvis = con.execute("SELECT count(DISTINCT (qkey, tkey)) FROM B WHERE accept05 AND vis AND has_struct").fetchone()[0]
    thr_rej = min(1000, int(np.ceil(1000 * 150_000 / max(n_rej, 1))))
    thr_av = min(1000, int(np.ceil(1000 * 60_000 / max(n_accvis, 1))))
    con.execute(f"""
        CREATE TABLE keyset AS
        SELECT DISTINCT qkey, tkey, 'accepted_invisible' AS kset FROM B WHERE accept05 AND NOT vis AND has_struct
        UNION ALL
        SELECT DISTINCT qkey, tkey, 'rejected_invisible' FROM B
            WHERE NOT accept05 AND NOT vis AND has_struct AND {H} < {thr_rej}
        UNION ALL
        SELECT DISTINCT qkey, tkey, 'accepted_visible' FROM B
            WHERE accept05 AND vis AND has_struct AND {H} < {thr_av}
        UNION ALL
        SELECT DISTINCT m.qkey, m.tkey, 'unchained_invisible'
            FROM membership m JOIN pairs p USING (qkey, tkey) LEFT JOIN seq s USING (qkey, tkey)
            WHERE m.cls = 'anchor_unchained' AND p.has_struct AND NOT COALESCE(s.evalue <= 1e-3, FALSE)
              AND hash(CAST(m.qkey AS BIGINT) * 5000000 + CAST(m.tkey AS BIGINT)) % 1000 < 4
        UNION ALL
        SELECT DISTINCT m.qkey, m.tkey, 'random' FROM membership m
            WHERE m.cls = 'random' AND hash(CAST(m.qkey AS BIGINT) * 5000000 + CAST(m.tkey AS BIGINT)) % 1000 < 80""")
    stats["sets"] = dict(con.execute("SELECT kset, count(*) FROM keyset GROUP BY ALL").fetchall())
    stats.update(rejected_invisible_struct_total=n_rej, accepted_visible_struct_total=n_accvis,
                 hash_threshold_rejected=thr_rej, hash_threshold_accepted_visible=thr_av)
    con.execute(f"COPY B TO '{K}/block_pairs.parquet' (FORMAT parquet)")
    con.execute(f"COPY keyset TO '{K}/keyset.parquet' (FORMAT parquet)")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM (SELECT DISTINCT qkey, tkey FROM keyset)
                          ORDER BY qkey, tkey)
                    TO '{K}/pref_key.tsv' (DELIMITER '\t', HEADER false)""")
    stats["unique_pairs"] = con.execute("SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM keyset)").fetchone()[0]
    with open(K / "prep_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats), flush=True)


def summarize(a):
    W, K, OUT = a.run / "work", a.run / "work_key", a.run / "results/key_subset"
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect(a.run, a.threads)
    con.execute(f"CREATE TABLE qx AS SELECT * FROM read_parquet('{W}/qx.parquet')")
    con.execute(f"CREATE TABLE B AS SELECT * FROM read_parquet('{K}/block_pairs.parquet')")
    con.execute(f"CREATE TABLE keyset AS SELECT * FROM read_parquet('{K}/keyset.parquet')")
    con.execute(f"""
        CREATE TABLE tm AS
        WITH j AS (
            SELECT qx.qkey, CAST(substr(c.t, 2) AS BIGINT) AS tkey, c.qtm, c.ttm, c.lddt
            FROM read_csv('{K}/aln_tm.tsv', delim='\t', header=false,
                names=['q','t','qtm','ttm','alntm','lddt','fident','qcov','tcov','alnlen','qlen','tlen','evalue','bits'],
                types={{'q':'VARCHAR','t':'VARCHAR','qtm':'DOUBLE','ttm':'DOUBLE','alntm':'DOUBLE','lddt':'DOUBLE'}}) c
            JOIN qx ON qx.native_id = c.q)
        SELECT qkey, tkey, greatest(qtm, ttm) AS tm_max, least(qtm, ttm) AS tm_min, lddt FROM j
        QUALIFY row_number() OVER (PARTITION BY qkey, tkey ORDER BY greatest(qtm, ttm) DESC) = 1""")
    n_key, n_tm = con.execute("""SELECT (SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM keyset)),
                                        (SELECT count(*) FROM tm)""").fetchone()
    con.execute(f"COPY tm TO '{K}/tm_key.parquet' (FORMAT parquet)")
    agg = """count(*) AS n, sum((tm_max >= 0.5)::INT) AS n_conf, sum((tm_min >= 0.5)::INT) AS n_conf_strict,
             median(tm_max) AS med_tm, quantile_cont(tm_max, 0.25) AS q25_tm, quantile_cont(tm_max, 0.75) AS q75_tm"""

    def derive(df):
        df = df.copy()
        df["conf_frac"] = df.n_conf / df.n
        df["conf_strict_frac"] = df.n_conf_strict / df.n
        return df

    sets = derive(con.execute(f"""
        SELECT k.kset, {agg} FROM keyset k JOIN tm USING (qkey, tkey) GROUP BY ALL ORDER BY k.kset""").df())
    size = "CASE WHEN n_anchors >= 10 THEN '10+' WHEN n_anchors >= 6 THEN '6-9' ELSE CAST(n_anchors AS VARCHAR) END"
    # per model: invisible pairs with structures, by call (rejected ones are a hash sample)
    bym = derive(con.execute(f"""
        SELECT B.model, B.accept05, {agg}
        FROM B JOIN tm USING (qkey, tkey) WHERE NOT B.vis AND B.has_struct
        GROUP BY ALL ORDER BY B.model, B.accept05""").df())
    bysize = derive(con.execute(f"""
        SELECT B.model, B.accept05, {size} AS size_bin, {agg}
        FROM B JOIN tm USING (qkey, tkey) WHERE NOT B.vis AND B.has_struct
        GROUP BY ALL ORDER BY B.model, B.accept05, size_bin""").df())
    tot = con.execute("""
        SELECT model, accept05, count(*) AS pairs, sum((NOT vis)::INT) AS invis,
               sum((NOT vis AND has_struct)::INT) AS invis_struct
        FROM B GROUP BY ALL ORDER BY model, accept05""").df()
    bym = bym.merge(tot, on=["model", "accept05"])
    bym["invis_frac"] = bym.invis / bym.pairs
    bym["est_confirmed_invisible_pairs"] = (bym.invis_struct * bym.conf_frac).round().astype(int)
    for name, df in (("sets", sets), ("by_model_call", bym), ("by_model_call_size", bysize)):
        df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)

    def md(df, cols, fmt):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if c in fmt and pd.notna(v):
                    cells.append(fmt[c].format(v))
                elif isinstance(v, (float, np.floating)) and pd.notna(v) and float(v).is_integer():
                    cells.append(f"{int(v):,}")
                elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                    cells.append(f"{int(v):,}")
                else:
                    cells.append(str(v))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    pct, f3 = "{:.2%}", "{:.3f}"
    fm = {"conf_frac": pct, "conf_strict_frac": pct, "med_tm": f3, "q25_tm": f3, "q75_tm": f3, "invis_frac": pct}
    L = ["# Key subset: structural confirmation of sequence-invisible pairs", "",
         f"Pairs requested {n_key:,}; TM-scores returned {n_tm:,}.", "",
         "## Sets", "", md(sets, ["kset", "n", "conf_frac", "conf_strict_frac", "q25_tm", "med_tm", "q75_tm"], fm), "",
         "## Sequence-invisible pairs by model and call (rejected blocks are a hash sample)", "",
         md(bym, ["model", "accept05", "pairs", "invis_frac", "invis_struct", "n", "conf_frac", "conf_strict_frac",
                  "med_tm", "est_confirmed_invisible_pairs"], fm), "",
         "## By block size", "",
         md(bysize, ["model", "accept05", "size_bin", "n", "conf_frac", "conf_strict_frac", "med_tm"], fm), ""]
    with open(OUT / "KEY_SUBSET.md", "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "summarize"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()
    (prep if args.mode == "prep" else summarize)(args)
