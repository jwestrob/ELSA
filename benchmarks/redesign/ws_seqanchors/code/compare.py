#!/usr/bin/env python
"""Arm tables and the decisive embedding-versus-sequence comparison.

prep       all-arm table; per model, accepted pairs/blocks unique to each source and shared,
           covered query genes, search status of embedding-only pairs; structure sample
           (TM-scores looked up in tonight's tables where present, the rest written for foldseek)
summarize  join TM-scores, confirmation shares with Wilson intervals, write the reports
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
S = ROOT / "seq_invisible_20261111"
W = RD / "ws_seqanchors"
NREF = 4190383
MODELS = ["esm2_8m", "esm2_650m", "esmc600m"]
TM_MODELS = ["esm2_650m", "esmc600m"]
SEQ_ARMS = ["S-all/unit", "S-all/bitrank", "S-top60/unit", "S-top60/bitrank", "S-top60cov50/unit", "S-top60cov50/bitrank"]
SREF = "S-top60/bitrank"
N_EONLY, N_SHARED, N_SONLY, N_HSEED = 10000, 2500, 2500, 8000
H = "hash(CAST(qkey AS BIGINT) * 5000000 + CAST(tkey AS BIGINT))"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tag(arm):
    return arm.replace("/", "_").replace("@", "__")


def arm_list():
    out = list(SEQ_ARMS)
    for m in MODELS:
        out += [f"E-ctx60/cos@{m}", f"E-ctx60/rank@{m}", f"E-seed60/cos@{m}", f"E-seed60/rank@{m}",
                f"H-ctx60@{m}", f"H-seed60@{m}"]
    return out


def connect(threads):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    return con


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and pd.notna(v) and not isinstance(v, str):
                cells.append(fmt[c].format(v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)) and pd.notna(v) and float(v).is_integer() and abs(v) >= 1:
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def arm_table():
    rows, strata = [], []
    for arm in arm_list():
        p = W / "results/arms" / f"{tag(arm)}.json"
        if not p.exists():
            log("missing arm", arm)
            continue
        r = json.load(open(p))
        s = {x["selection"]: x for x in r["source207"]}
        row = dict(arm=arm, anchors_in=r["anchors_in"]["observed"], observed_blocks=r["observed_blocks"],
                   perm_over_obs=r["permuted_over_observed"])
        for size in ("2", "3", "4", "5", "6+"):
            row[f"ratio_{size}"] = r["sizes"][size]["ratio"]
        row.update(accepted05_blocks=r["accepted05_blocks"], accepted05_pairs=r["accepted05_anchor_pairs"],
                   accepted05_perm_expected=r["accepted05_permuted_expected"],
                   accepted01_blocks=r["accepted01_blocks"],
                   covered_genes05=r["accepted05_covered_query_genes"],
                   f1_cal5=s["calibrated_5pct"]["f1_observed"], f1_rand5=s["calibrated_5pct"]["f1_random"],
                   excess5=s["calibrated_5pct"]["f1_excess"],
                   excess5_lo=s["calibrated_5pct"]["f1_excess_ci95"][0], excess5_hi=s["calibrated_5pct"]["f1_excess_ci95"][1],
                   f1_all=s["all"]["f1_observed"], excess_all=s["all"]["f1_excess"],
                   xphylum05=r["accepted05_cross_phylum_blocks"], xphylum05_ge10=r["accepted05_cross_phylum_blocks_ge10"],
                   pair_in_search05=r.get("accepted05_pair_in_search"),
                   chain_s_obs=r["chain_seconds"]["observed"], chain_s_total=sum(r["chain_seconds"].values()),
                   arm_s=r["arm_seconds"])
        rows.append(row)
        for st, v in r["accepted05_by_stratum"].items():
            strata.append(dict(arm=arm, stratum=st, **v))
    return pd.DataFrame(rows), pd.DataFrame(strata)


def prep(a):
    out = W / "results"
    tm = W / "work/tm"
    tm.mkdir(parents=True, exist_ok=True)
    arms, strata = arm_table()
    arms.to_csv(out / "arms.tsv", sep="\t", index=False)
    strata.to_csv(out / "arms_by_stratum.tsv", sep="\t", index=False)
    con = connect(a.threads)
    con.execute(f"""CREATE TABLE u AS SELECT gid, side, row_index, mmseqs_key, length, has_struct
                    FROM read_parquet('{RD}/universe/genes.parquet')""")
    con.execute(f"""CREATE TABLE qlen AS SELECT row_index, model_input_length AS mlen
                    FROM read_parquet('{ROOT}/results/bgc_query_embeddings/sequence_identity.parquet')""")
    con.execute(f"""CREATE TABLE rlen AS SELECT row_index, model_input_length AS mlen
                    FROM read_parquet('{ROOT}/results/esm2_t6/shard_000/embedding/sequence_identity.parquet')""")
    con.execute(f"CREATE TABLE hits AS SELECT qrow, trow, rank_all, evalue, seqid FROM read_parquet('{W}/work/hits.parquet')")

    def load_pairs(arm, name):
        p = W / "work/arms" / tag(arm) / "pairs.parquet"
        con.execute(f"""CREATE OR REPLACE TABLE {name} AS
            SELECT DISTINCT qrow, trow FROM read_parquet('{p}') WHERE accept05""")
        b = W / "work/arms" / tag(arm) / "block_calls.parquet"
        con.execute(f"""CREATE OR REPLACE TABLE {name}_b AS
            SELECT block_id, query_genome, query_contig, target_genome, target_contig, query_start, query_end,
                   target_start, target_end, n_anchors FROM read_parquet('{b}') WHERE accept05""")
        return np.load(W / "work/arms" / tag(arm) / "covered_query_rows_05.npy")

    cov_s = load_pairs(SREF, "S")
    # union of accepted pairs over every sequence arm (secondary reference)
    con.execute("CREATE TABLE SU (qrow BIGINT, trow BIGINT)")
    cov_su = np.zeros_like(cov_s)
    for arm in SEQ_ARMS:
        p = W / "work/arms" / tag(arm) / "pairs.parquet"
        if p.exists():
            con.execute(f"INSERT INTO SU SELECT DISTINCT qrow, trow FROM read_parquet('{p}') WHERE accept05")
            cov_su |= np.load(W / "work/arms" / tag(arm) / "covered_query_rows_05.npy")
    con.execute("CREATE TABLE SUd AS SELECT DISTINCT qrow, trow FROM SU")
    stop60 = W / "work/arms" / tag(SREF) / "pairs.parquet"
    con.execute(f"""CREATE TABLE stop60 AS SELECT qrow, trow FROM hits WHERE rank_all <= 60""")

    rows = []
    con.execute("""CREATE TABLE sample (model VARCHAR, comparison VARCHAR, pset VARCHAR, qrow BIGINT, trow BIGINT,
                   in_search BOOLEAN, in_stop60 BOOLEAN, long_protein BOOLEAN)""")
    for m in MODELS:
        comps = [(f"E-ctx60/rank@{m}", "S", cov_s, SREF), (f"H-ctx60@{m}", "S", cov_s, SREF),
                 (f"E-seed60/rank@{m}", "S", cov_s, SREF), (f"H-seed60@{m}", "S", cov_s, SREF),
                 (f"E-ctx60/cos@{m}", "S", cov_s, SREF),
                 (f"E-ctx60/rank@{m}", "SUd", cov_su, "union of sequence arms"),
                 (f"H-ctx60@{m}", "SUd", cov_su, "union of sequence arms"),
                 (f"E-seed60/rank@{m}", "SUd", cov_su, "union of sequence arms"),
                 (f"H-seed60@{m}", "SUd", cov_su, "union of sequence arms")]
        for arm, ref, cov_ref, ref_name in comps:
            if not (W / "work/arms" / tag(arm) / "pairs.parquet").exists():
                continue
            cov_e = load_pairs(arm, "E")
            r = dict(model=m, arm=arm, reference=ref_name)
            r["pairs_arm"], r["pairs_ref"] = con.execute(f"SELECT (SELECT count(*) FROM E), (SELECT count(*) FROM {ref})").fetchone()
            r["pairs_shared"] = con.execute(f"SELECT count(*) FROM E JOIN {ref} USING (qrow, trow)").fetchone()[0]
            r["pairs_arm_only"] = r["pairs_arm"] - r["pairs_shared"]
            r["pairs_ref_only"] = r["pairs_ref"] - r["pairs_shared"]
            r["arm_only_over_ref"] = r["pairs_arm_only"] / max(r["pairs_ref"], 1)
            # status of arm-only pairs under the sequence search
            q = con.execute(f"""
                WITH x AS (SELECT * FROM E ANTI JOIN {ref} USING (qrow, trow))
                SELECT count(*), sum((h.qrow IS NOT NULL)::INT), sum(COALESCE(h.rank_all <= 60, FALSE)::INT)
                FROM x LEFT JOIN hits h USING (qrow, trow)""").fetchone()
            r["arm_only_in_search"] = int(q[1] or 0)
            r["arm_only_in_stop60"] = int(q[2] or 0)
            r["arm_only_seq_invisible"] = int(q[0] - (q[1] or 0))
            q = con.execute(f"""
                WITH x AS (SELECT * FROM {ref} ANTI JOIN E USING (qrow, trow))
                SELECT count(*), sum((COALESCE(ql.mlen, 0) > 1022 OR COALESCE(rl.mlen, 0) > 1022)::INT)
                FROM x LEFT JOIN qlen ql ON ql.row_index = x.qrow LEFT JOIN rlen rl ON rl.row_index = x.trow""").fetchone()
            r["ref_only_long_protein"] = int(q[1] or 0)
            # blocks: shared if an accepted block of the other side overlaps on both axes in the same contig pair
            ov = """SELECT count(DISTINCT a.block_id) FROM {A}_b a JOIN {B}_b b
                    ON a.query_genome = b.query_genome AND a.query_contig = b.query_contig
                   AND a.target_genome = b.target_genome AND a.target_contig = b.target_contig
                   AND a.query_start <= b.query_end AND a.query_end >= b.query_start
                   AND a.target_start <= b.target_end AND a.target_end >= b.target_start"""
            ref_b = "S" if ref == "S" else None
            r["blocks_arm"] = con.execute("SELECT count(*) FROM E_b").fetchone()[0]
            if ref_b:
                r["blocks_ref"] = con.execute("SELECT count(*) FROM S_b").fetchone()[0]
                r["blocks_arm_overlapping_ref"] = con.execute(ov.format(A="E", B="S")).fetchone()[0]
                r["blocks_ref_overlapping_arm"] = con.execute(ov.format(A="S", B="E")).fetchone()[0]
                r["blocks_arm_only"] = r["blocks_arm"] - r["blocks_arm_overlapping_ref"]
                r["blocks_ref_only"] = r["blocks_ref"] - r["blocks_ref_overlapping_arm"]
                xp = """SELECT count(*) FROM {A}_b a WHERE a.target_genome IN (SELECT genome FROM g3)
                        AND NOT EXISTS (SELECT 1 FROM {B}_b b WHERE a.query_genome = b.query_genome
                        AND a.query_contig = b.query_contig AND a.target_genome = b.target_genome
                        AND a.target_contig = b.target_contig AND a.query_start <= b.query_end
                        AND a.query_end >= b.query_start AND a.target_start <= b.target_end
                        AND a.target_end >= b.target_start)"""
                con.execute(f"""CREATE OR REPLACE TABLE g3 AS SELECT genome FROM read_parquet('{RD}/universe/genomes.parquet')
                                WHERE stratum = '3_other_phylum'""")
                r["xphylum_blocks_arm_only"] = con.execute(xp.format(A="E", B="S")).fetchone()[0]
                r["xphylum_blocks_ref_only"] = con.execute(xp.format(A="S", B="E")).fetchone()[0]
            r["covered_ref"] = int(cov_ref.sum())
            r["covered_arm"] = int(cov_e.sum())
            r["covered_union"] = int((cov_ref | cov_e).sum())
            r["covered_added_by_arm"] = r["covered_union"] - r["covered_ref"]
            r["covered_added_by_arm_frac"] = r["covered_added_by_arm"] / max(r["covered_ref"], 1)
            r["covered_added_by_ref"] = r["covered_union"] - r["covered_arm"]
            if arm.startswith("H-"):
                # hybrid-only pairs that came from the embedding source (absent from the S-top60 anchors)
                r["arm_only_embedding_sourced"] = con.execute(f"""
                    SELECT count(*) FROM (SELECT * FROM E ANTI JOIN {ref} USING (qrow, trow)) x
                    ANTI JOIN stop60 USING (qrow, trow)""").fetchone()[0]
            rows.append(r)
            log(r)
            psets = []
            if m in TM_MODELS and ref == "S" and arm == f"E-ctx60/rank@{m}":
                psets = [("arm_only", f"SELECT * FROM E ANTI JOIN {ref} USING (qrow, trow)", N_EONLY),
                         ("shared", f"SELECT * FROM E JOIN {ref} USING (qrow, trow)", N_SHARED),
                         ("ref_only", f"SELECT * FROM {ref} ANTI JOIN E USING (qrow, trow)", N_SONLY)]
            elif m in TM_MODELS and ref == "S" and arm == f"H-seed60@{m}":
                # accepted hybrid pairs absent from the sequence arm and from the S-top60 anchors
                psets = [("hseed_emb_added", f"""SELECT * FROM (SELECT * FROM E ANTI JOIN {ref} USING (qrow, trow)) y
                                                 ANTI JOIN stop60 USING (qrow, trow)""", N_HSEED)]
            for pset, sql, n in psets:
                if True:
                    con.execute(f"""
                        INSERT INTO sample
                        WITH x AS ({sql}),
                        k AS (SELECT x.qrow, x.trow, uq.mmseqs_key AS qkey, ur.mmseqs_key AS tkey,
                                     uq.has_struct AND ur.has_struct AS hs
                              FROM x JOIN u uq ON uq.gid = {NREF} + x.qrow JOIN u ur ON ur.gid = x.trow)
                        SELECT '{m}', '{arm} vs {ref_name}', '{pset}', k.qrow, k.trow,
                               h.qrow IS NOT NULL, COALESCE(h.rank_all <= 60, FALSE),
                               COALESCE(ql.mlen, 0) > 1022 OR COALESCE(rl.mlen, 0) > 1022
                        FROM k LEFT JOIN hits h USING (qrow, trow)
                        LEFT JOIN qlen ql ON ql.row_index = k.qrow LEFT JOIN rlen rl ON rl.row_index = k.trow
                        WHERE k.hs ORDER BY {H.replace('qkey', 'k.qkey').replace('tkey', 'k.tkey')} LIMIT {n}""")
                    hs = con.execute(f"""
                        WITH x AS ({sql}) SELECT count(*), sum((uq.has_struct AND ur.has_struct)::INT)
                        FROM x JOIN u uq ON uq.gid = {NREF} + x.qrow JOIN u ur ON ur.gid = x.trow""").fetchone()
                    r[f"{pset}_with_structures"] = int(hs[1] or 0)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "decisive_counts.tsv", sep="\t", index=False)
    con.execute(f"""CREATE TABLE sk AS SELECT s.*, uq.mmseqs_key AS qkey, ur.mmseqs_key AS tkey FROM sample s
                    JOIN u uq ON uq.gid = {NREF} + s.qrow JOIN u ur ON ur.gid = s.trow""")
    con.execute(f"COPY sk TO '{tm}/sample.parquet' (FORMAT parquet)")
    # look up pairs already scored tonight
    con.execute("CREATE TABLE known (qkey BIGINT, tkey BIGINT, tm_max DOUBLE, tm_min DOUBLE, src VARCHAR)")
    sources = {}
    tk = S / "work_key/tm_key.parquet"
    if tk.exists():
        con.execute(f"INSERT INTO known SELECT qkey, tkey, tm_max, tm_min, 'tm_key' FROM read_parquet('{tk}')")
        sources["tm_key"] = str(tk)
    pt = S / "work/pair_table.parquet"
    if pt.exists():
        cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pt}')").fetchall()]
        sources["pair_table_columns"] = cols
        if {"qtm", "ttm"} <= set(cols):
            con.execute(f"""INSERT INTO known SELECT qkey, tkey, greatest(qtm, ttm), least(qtm, ttm), 'pair_table'
                            FROM read_parquet('{pt}') WHERE qtm IS NOT NULL""")
        elif {"tm_max", "tm_min"} <= set(cols):
            con.execute(f"""INSERT INTO known SELECT qkey, tkey, tm_max, tm_min, 'pair_table'
                            FROM read_parquet('{pt}') WHERE tm_max IS NOT NULL""")
        sources["pair_table"] = str(pt)
    con.execute(f"""COPY (SELECT DISTINCT ON (qkey, tkey) k.* FROM known k
                          SEMI JOIN (SELECT DISTINCT qkey, tkey FROM sk) s USING (qkey, tkey))
                    TO '{tm}/lookup.parquet' (FORMAT parquet)""")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM
                          (SELECT DISTINCT qkey, tkey FROM sk ANTI JOIN known USING (qkey, tkey)) ORDER BY qkey, tkey)
                    TO '{tm}/pref.tsv' (DELIMITER '\t', HEADER false)""")
    n_sample = con.execute("SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM sk)").fetchone()[0]
    n_known = con.execute(f"SELECT count(*) FROM read_parquet('{tm}/lookup.parquet')").fetchone()[0]
    assert n_sample - n_known <= 50000, (n_sample, n_known)
    # header DBs carrying K<key> for both sides
    con.execute(f"""COPY (SELECT mmseqs_key, 'K' || CAST(mmseqs_key AS VARCHAR) FROM u WHERE side = 'query' AND has_struct
                          ORDER BY mmseqs_key) TO '{tm}/q_h.tsv' (DELIMITER '\t', HEADER false)""")
    con.execute(f"""COPY (SELECT mmseqs_key, 'K' || CAST(mmseqs_key AS VARCHAR) FROM u WHERE side = 'ref' AND has_struct
                          ORDER BY mmseqs_key) TO '{tm}/t_h.tsv' (DELIMITER '\t', HEADER false)""")
    stats = dict(sample_pairs=n_sample, looked_up=n_known, to_align=n_sample - n_known, sources=sources,
                 sets=con.execute("SELECT model, pset, count(*) FROM sk GROUP BY ALL ORDER BY ALL").fetchall())
    (tm / "prep_stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
    log(stats)


def summarize(a):
    out = W / "results"
    tm = W / "work/tm"
    con = connect(a.threads)
    con.execute(f"CREATE TABLE sk AS SELECT * FROM read_parquet('{tm}/sample.parquet')")
    con.execute(f"CREATE TABLE known AS SELECT qkey, tkey, tm_max, tm_min, src FROM read_parquet('{tm}/lookup.parquet')")
    p = tm / "aln_tm.tsv"
    if p.exists():
        con.execute(f"""
            INSERT INTO known
            SELECT CAST(substr(q, 2) AS BIGINT), CAST(substr(t, 2) AS BIGINT), greatest(qtm, ttm), least(qtm, ttm), 'ws_seqanchors'
            FROM read_csv('{p}', delim='\t', header=false,
                names=['q','t','qtm','ttm','alntm','lddt','fident','qcov','tcov','alnlen','qlen','tlen','evalue','bits'],
                types={{'q':'VARCHAR','t':'VARCHAR','qtm':'DOUBLE','ttm':'DOUBLE'}})""")
    con.execute("""CREATE TABLE tmk AS SELECT qkey, tkey, max(tm_max) AS tm_max, max(tm_min) AS tm_min, any_value(src) AS src
                   FROM known GROUP BY ALL""")
    j = con.execute("""SELECT sk.*, tmk.tm_max, tmk.tm_min, tmk.src FROM sk LEFT JOIN tmk USING (qkey, tkey)""").df()
    j.to_parquet(tm / "sample_scored.parquet", index=False)
    rows = []
    for (m, pset), g in j.groupby(["model", "pset"]):
        for label, sub in (("all", g), ("in_search", g[g.in_search]), ("seq_invisible", g[~g.in_search]),
                           ("long_protein", g[g.long_protein])):
            sc = sub[sub.tm_max.notna()]
            k = int((sc.tm_max >= 0.5).sum())
            lo, hi = wilson(k, len(sc))
            rows.append(dict(model=m, pset=pset, subset=label, sampled=len(sub), scored=len(sc), confirmed=k,
                             confirmed_frac=k / len(sc) if len(sc) else np.nan, ci_lo=lo, ci_hi=hi,
                             strict_frac=float((sc.tm_min >= 0.5).mean()) if len(sc) else np.nan,
                             median_tm=float(sc.tm_max.median()) if len(sc) else np.nan))
    conf = pd.DataFrame(rows)
    conf.to_csv(out / "structure_confirmation.tsv", sep="\t", index=False)
    write_report(conf)


def write_report(conf=None):
    out = W / "results"
    arms = pd.read_csv(out / "arms.tsv", sep="\t")
    strata = pd.read_csv(out / "arms_by_stratum.tsv", sep="\t")
    comp = pd.read_csv(out / "decisive_counts.tsv", sep="\t")
    search = json.load(open(out / "search_stats.json"))
    f = {"perm_over_obs": "{:.3f}", "ratio_2": "{:.3f}", "ratio_3": "{:.3f}", "ratio_4": "{:.3f}", "ratio_5": "{:.3f}",
         "ratio_6+": "{:.4f}", "f1_cal5": "{:.3f}", "f1_rand5": "{:.3f}", "excess5": "{:+.3f}", "excess5_lo": "{:+.3f}",
         "excess5_hi": "{:+.3f}", "f1_all": "{:.3f}", "excess_all": "{:+.3f}", "pair_in_search05": "{:.1%}",
         "accepted05_perm_expected": "{:,.0f}", "chain_s_obs": "{:.0f}", "chain_s_total": "{:.0f}", "arm_s": "{:.0f}",
         "arm_only_over_ref": "{:.1%}", "covered_added_by_arm_frac": "{:.1%}", "confirmed_frac": "{:.1%}",
         "ci_lo": "{:.1%}", "ci_hi": "{:.1%}", "strict_frac": "{:.1%}", "median_tm": "{:.3f}"}
    a1 = arms[["arm", "anchors_in", "observed_blocks", "perm_over_obs", "ratio_2", "ratio_3", "ratio_4", "ratio_5", "ratio_6+",
               "accepted05_blocks", "accepted05_pairs", "accepted05_perm_expected", "covered_genes05"]]
    a2 = arms[["arm", "f1_cal5", "f1_rand5", "excess5", "excess5_lo", "excess5_hi", "f1_all", "excess_all",
               "xphylum05", "xphylum05_ge10", "pair_in_search05", "chain_s_obs", "chain_s_total", "arm_s"]]
    st = strata.pivot_table(index="arm", columns="stratum", values="blocks", fill_value=0).reset_index()
    L = ["# ws_seqanchors results", "",
         f"Search: {search['hits_e1e3']:,} hits at E <= 1e-3 for {search['queries_with_hit']:,} of 67,148 queries; "
         f"S-top60 {search['s_top60']:,}, S-top60-cov50 {search['s_top60_cov50']:,}; "
         f"{search['queries_prefilter_saturated']:,} queries at the 4,000-candidate prefilter cap; "
         f"prefilter {search['timing']['prefilter_seconds']:,} s + align {search['timing']['align_seconds']:,} s on "
         f"{search['timing']['threads']} CPUs.", "",
         "## Arms: blocks, null and calibration (5% permuted/observed per size)", "", md(a1, f), "",
         "## Arms: source207 and taxonomy", "", md(a2, f), "",
         "## Accepted blocks (5%) by target stratum", "", md(st), "",
         "## Decisive comparison (5% level)", "", md(comp, f), ""]
    if conf is not None:
        L += ["## Structure confirmation (TM-score >= 0.5 normalized by the shorter chain)", "", md(conf, f), ""]
    (out / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L), flush=True)


def paired(a):
    """Paired region bootstrap of the difference in source207 F1 excess between arms (5% calls)."""
    import sys
    sys.path.insert(0, str(W / "code/scale_code"))
    from evaluate_spacedust_queries import evaluate
    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    genes = pd.read_parquet(ROOT / "bgc_queries/metadata.parquet")
    span = ["query_genome", "query_contig", "query_start", "query_end"]
    arms = ["S-all/bitrank", "S-top60/bitrank"]
    for m in MODELS:
        arms += [f"E-ctx60/cos@{m}", f"E-ctx60/rank@{m}", f"E-seed60/rank@{m}", f"H-ctx60@{m}", f"H-seed60@{m}"]
    per = {}
    for arm in arms:
        calls = pd.read_parquet(W / "work/arms" / tag(arm) / "block_calls.parquet", columns=span + ["accept05"])
        spans = calls[calls.accept05][span].drop_duplicates()
        obs, ctrl, _ = evaluate(regions, genes, spans, threads=a.threads, controls=200, seed=17)
        per[arm] = (obs.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).sort_index()
        log(arm, f"{per[arm].mean():+.4f}")
    rows = []
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(regions), size=(10000, len(regions)))
    for ref in ("S-top60/bitrank", "S-all/bitrank"):
        for arm in arms:
            if arm == ref:
                continue
            d = (per[arm] - per[ref]).to_numpy()
            boot = d[idx].mean(axis=1)
            rows.append(dict(arm=arm, reference=ref, excess_arm=float(per[arm].mean()), excess_ref=float(per[ref].mean()),
                             difference=float(d.mean()), ci_lo=float(np.quantile(boot, .025)),
                             ci_hi=float(np.quantile(boot, .975)), regions_arm_better=int((d > 0).sum()),
                             regions_ref_better=int((d < 0).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(W / "results/paired_f1_excess.tsv", sep="\t", index=False)
    pd.DataFrame(per).to_csv(W / "results/per_region_excess.tsv", sep="\t")
    print(md(df, {"excess_arm": "{:+.4f}", "excess_ref": "{:+.4f}", "difference": "{:+.4f}",
                  "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "summarize", "report", "paired"])
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    {"prep": prep, "summarize": summarize, "report": lambda a: write_report(), "paired": paired}[a.mode](a)
