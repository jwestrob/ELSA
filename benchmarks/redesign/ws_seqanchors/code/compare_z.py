#!/usr/bin/env python
"""Whitened / learned embedding seed arms: tables, decisive counts, structure sample, TM summary, paired F1.

prep       arm table + strata for the new arms; decisive counts vs S-top60/bitrank and vs the union of
           sequence arms; structure sample of HZ/HL additions (<= 20,000 pairs), TM looked up in
           pair_table, tm_key and this workstream's earlier sample; the rest written for foldseek
summarize  join TM, confirmation shares (Wilson), stratified estimates, predeclared verdict
paired     paired region bootstrap of F1-excess differences against S-top60, S-all, E-seed60, H-seed60, R-seed60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402

W, RD, ROOT, S, NREF = C.W, C.RD, C.ROOT, C.S, C.NREF
NEW = ["Z-seed60@esm2_650m", "HZ-seed60@esm2_650m", "R-seed60@esm2_650m",
       "Z-seed60@esmc600m", "HZ-seed60@esmc600m", "R-seed60@esmc600m",
       "L-seed60@esmc600m", "HL-seed60@esmc600m"]
TM_ARMS = ["HZ-seed60@esm2_650m", "HZ-seed60@esmc600m", "HL-seed60@esmc600m"]
N_NOHIT, N_INS_U, N_INS_S = 4500, 1000, 1000
UNION_PAIRS, UNION_GENES = 9823998, 58065
TMZ = W / "work/tmz"
log, tag, wilson, md = C.log, C.tag, C.wilson, C.md


def arm_rows(arms):
    rows, strata = [], []
    for arm in arms:
        r = json.load(open(W / "results/arms" / f"{tag(arm)}.json"))
        s = {x["selection"]: x for x in r["source207"]}
        row = dict(arm=arm, anchors_in=r["anchors_in"]["observed"], observed_blocks=r["observed_blocks"],
                   perm_over_obs=r["permuted_over_observed"])
        for size in ("2", "3", "4", "5", "6+"):
            row[f"ratio_{size}"] = r["sizes"][size]["ratio"]
        row.update(accepted05_blocks=r["accepted05_blocks"], accepted05_pairs=r["accepted05_anchor_pairs"],
                   accepted05_perm_expected=r["accepted05_permuted_expected"],
                   covered_genes05=r["accepted05_covered_query_genes"],
                   f1_cal5=s["calibrated_5pct"]["f1_observed"], f1_rand5=s["calibrated_5pct"]["f1_random"],
                   excess5=s["calibrated_5pct"]["f1_excess"], excess5_lo=s["calibrated_5pct"]["f1_excess_ci95"][0],
                   excess5_hi=s["calibrated_5pct"]["f1_excess_ci95"][1], f1_all=s["all"]["f1_observed"],
                   xphylum05=r["accepted05_cross_phylum_blocks"], xphylum05_ge10=r["accepted05_cross_phylum_blocks_ge10"],
                   pair_in_search05=r.get("accepted05_pair_in_search"),
                   chain_s_obs=r["chain_seconds"]["observed"], chain_s_total=sum(r["chain_seconds"].values()))
        rows.append(row)
        for st, v in r["accepted05_by_stratum"].items():
            strata.append(dict(arm=arm, stratum=st, **v))
    return pd.DataFrame(rows), pd.DataFrame(strata)


def prep(a):
    out = W / "results"
    TMZ.mkdir(parents=True, exist_ok=True)
    arms, strata = arm_rows(NEW)
    arms.to_csv(out / "zarms.tsv", sep="\t", index=False)
    strata.to_csv(out / "zarms_by_stratum.tsv", sep="\t", index=False)
    con = C.connect(a.threads)
    con.execute(f"""CREATE TABLE u AS SELECT gid, side, row_index, mmseqs_key, has_struct
                    FROM read_parquet('{RD}/universe/genes.parquet')""")
    con.execute(f"CREATE TABLE hits AS SELECT qrow, trow, rank_all FROM read_parquet('{W}/work/hits.parquet')")
    con.execute("CREATE TABLE stop60 AS SELECT qrow, trow FROM hits WHERE rank_all <= 60")
    con.execute(f"""CREATE TABLE g3 AS SELECT genome FROM read_parquet('{RD}/universe/genomes.parquet')
                    WHERE stratum = '3_other_phylum'""")

    def load(arm, name):
        p = W / "work/arms" / tag(arm)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT DISTINCT qrow, trow FROM read_parquet('{p}/pairs.parquet') WHERE accept05")
        con.execute(f"""CREATE OR REPLACE TABLE {name}_b AS SELECT block_id, query_genome, query_contig, target_genome,
                        target_contig, query_start, query_end, target_start, target_end, n_anchors
                        FROM read_parquet('{p}/block_calls.parquet') WHERE accept05""")
        return np.load(p / "covered_query_rows_05.npy")

    cov_s = load(C.SREF, "S")
    con.execute("CREATE TABLE SU (qrow BIGINT, trow BIGINT)")
    cov_su = np.zeros_like(cov_s)
    for arm in C.SEQ_ARMS:
        p = W / "work/arms" / tag(arm)
        con.execute(f"INSERT INTO SU SELECT DISTINCT qrow, trow FROM read_parquet('{p}/pairs.parquet') WHERE accept05")
        cov_su |= np.load(p / "covered_query_rows_05.npy")
    con.execute("CREATE TABLE SUd AS SELECT DISTINCT qrow, trow FROM SU")
    n_su = con.execute("SELECT count(*) FROM SUd").fetchone()[0]
    assert n_su == UNION_PAIRS and int(cov_su.sum()) == UNION_GENES, (n_su, int(cov_su.sum()))
    ov = """SELECT count(DISTINCT a.block_id) FROM {A}_b a JOIN {B}_b b
            ON a.query_genome = b.query_genome AND a.query_contig = b.query_contig
           AND a.target_genome = b.target_genome AND a.target_contig = b.target_contig
           AND a.query_start <= b.query_end AND a.query_end >= b.query_start
           AND a.target_start <= b.target_end AND a.target_end >= b.target_start"""
    xp = """SELECT count(*) FROM {A}_b a WHERE a.target_genome IN (SELECT genome FROM g3)
            AND NOT EXISTS (SELECT 1 FROM {B}_b b WHERE a.query_genome = b.query_genome
            AND a.query_contig = b.query_contig AND a.target_genome = b.target_genome
            AND a.target_contig = b.target_contig AND a.query_start <= b.query_end
            AND a.query_end >= b.query_start AND a.target_start <= b.target_end AND a.target_end >= b.target_start)"""
    rows = []
    con.execute("""CREATE TABLE sample (arm VARCHAR, pset VARCHAR, qrow BIGINT, trow BIGINT, in_search BOOLEAN)""")
    for arm in [x for x in NEW if not x.startswith("R-")] + [x for x in NEW if x.startswith("R-")]:
        cov_e = load(arm, "E")
        for ref, cov_ref, ref_name in (("S", cov_s, C.SREF), ("SUd", cov_su, "union of sequence arms")):
            r = dict(model=arm.split("@")[1], arm=arm, reference=ref_name)
            r["pairs_arm"], r["pairs_ref"] = con.execute(f"SELECT (SELECT count(*) FROM E), (SELECT count(*) FROM {ref})").fetchone()
            r["pairs_shared"] = con.execute(f"SELECT count(*) FROM E JOIN {ref} USING (qrow, trow)").fetchone()[0]
            r["pairs_arm_only"] = r["pairs_arm"] - r["pairs_shared"]
            r["pairs_ref_only"] = r["pairs_ref"] - r["pairs_shared"]
            r["arm_only_over_ref"] = r["pairs_arm_only"] / max(r["pairs_ref"], 1)
            q = con.execute(f"""WITH x AS (SELECT * FROM E ANTI JOIN {ref} USING (qrow, trow))
                                SELECT count(*), sum((h.qrow IS NOT NULL)::INT) FROM x LEFT JOIN hits h USING (qrow, trow)""").fetchone()
            r["arm_only_in_search"] = int(q[1] or 0)
            r["arm_only_no_hit"] = int(q[0] - (q[1] or 0))
            r["blocks_arm"] = con.execute("SELECT count(*) FROM E_b").fetchone()[0]
            if ref == "S":
                r["blocks_ref"] = con.execute("SELECT count(*) FROM S_b").fetchone()[0]
                r["blocks_arm_only"] = r["blocks_arm"] - con.execute(ov.format(A="E", B="S")).fetchone()[0]
                r["blocks_ref_only"] = r["blocks_ref"] - con.execute(ov.format(A="S", B="E")).fetchone()[0]
                r["xphylum_blocks_arm_only"] = con.execute(xp.format(A="E", B="S")).fetchone()[0]
                r["xphylum_blocks_ref_only"] = con.execute(xp.format(A="S", B="E")).fetchone()[0]
            r["covered_ref"], r["covered_arm"] = int(cov_ref.sum()), int(cov_e.sum())
            r["covered_union"] = int((cov_ref | cov_e).sum())
            r["covered_added_by_arm"] = r["covered_union"] - r["covered_ref"]
            r["covered_added_by_arm_frac"] = r["covered_added_by_arm"] / max(r["covered_ref"], 1)
            r["covered_added_by_ref"] = r["covered_union"] - r["covered_arm"]
            if arm.startswith("H"):
                r["arm_only_embedding_sourced"] = con.execute(f"""SELECT count(*) FROM
                    (SELECT * FROM E ANTI JOIN {ref} USING (qrow, trow)) x ANTI JOIN stop60 USING (qrow, trow)""").fetchone()[0]
            rows.append(r)
            log({k: r[k] for k in ("arm", "reference", "pairs_arm_only", "arm_only_over_ref", "arm_only_no_hit",
                                   "covered_added_by_arm", "covered_added_by_arm_frac")})
        if arm in TM_ARMS:
            hs = f"""k AS (SELECT x.qrow, x.trow, uq.mmseqs_key AS qkey, ur.mmseqs_key AS tkey,
                           uq.has_struct AND ur.has_struct AS hs
                    FROM x JOIN u uq ON uq.gid = {NREF} + x.qrow JOIN u ur ON ur.gid = x.trow)"""
            order = C.H.replace("qkey", "k.qkey").replace("tkey", "k.tkey")
            for pset, sql, n in (
                    ("nohit_union", "SELECT * FROM (SELECT * FROM E ANTI JOIN SUd USING (qrow, trow)) y ANTI JOIN hits USING (qrow, trow)", N_NOHIT),
                    ("insearch_union", "SELECT * FROM (SELECT * FROM E ANTI JOIN SUd USING (qrow, trow)) y SEMI JOIN hits USING (qrow, trow)", N_INS_U),
                    ("insearch_top60", "SELECT * FROM (SELECT * FROM E ANTI JOIN S USING (qrow, trow)) y SEMI JOIN hits USING (qrow, trow)", N_INS_S)):
                con.execute(f"""INSERT INTO sample WITH x AS ({sql}), {hs}
                                SELECT '{arm}', '{pset}', k.qrow, k.trow, {'FALSE' if pset.startswith('nohit') else 'TRUE'}
                                FROM k WHERE k.hs ORDER BY {order} LIMIT {n}""")
                tot = con.execute(f"WITH x AS ({sql}), {hs} SELECT count(*), sum(hs::INT) FROM k").fetchone()
                rows[-1][f"{pset}_total"] = int(tot[0])
                rows[-1][f"{pset}_with_structures"] = int(tot[1] or 0)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "zdecisive_counts.tsv", sep="\t", index=False)
    con.execute(f"""CREATE TABLE sk AS SELECT s.*, uq.mmseqs_key AS qkey, ur.mmseqs_key AS tkey FROM sample s
                    JOIN u uq ON uq.gid = {NREF} + s.qrow JOIN u ur ON ur.gid = s.trow""")
    n_unique = con.execute("SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM sk)").fetchone()[0]
    assert n_unique <= 20000, n_unique
    con.execute(f"COPY sk TO '{TMZ}/sample.parquet' (FORMAT parquet)")
    con.execute("CREATE TABLE known (qkey BIGINT, tkey BIGINT, tm_max DOUBLE, tm_min DOUBLE, src VARCHAR)")
    con.execute("CREATE TABLE skd AS SELECT DISTINCT qkey, tkey FROM sk")
    con.execute(f"""INSERT INTO known SELECT p.qkey, p.tkey, greatest(p.qtm, p.ttm), least(p.qtm, p.ttm), 'pair_table'
                    FROM read_parquet('{S}/work/pair_table.parquet') p SEMI JOIN skd USING (qkey, tkey)
                    WHERE p.qtm IS NOT NULL AND p.ttm IS NOT NULL""")
    con.execute(f"""INSERT INTO known SELECT p.qkey, p.tkey, p.tm_max, p.tm_min, 'tm_key'
                    FROM read_parquet('{S}/work_key/tm_key.parquet') p SEMI JOIN skd USING (qkey, tkey)""")
    con.execute(f"""INSERT INTO known SELECT DISTINCT p.qkey, p.tkey, p.tm_max, p.tm_min, 'ws_seqanchors_round1'
                    FROM read_parquet('{W}/work/tm/sample_scored.parquet') p SEMI JOIN skd USING (qkey, tkey)
                    WHERE p.tm_max IS NOT NULL""")
    con.execute(f"""COPY (SELECT DISTINCT ON (qkey, tkey) * FROM known ORDER BY qkey, tkey, src)
                    TO '{TMZ}/lookup.parquet' (FORMAT parquet)""")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM
                          (SELECT DISTINCT qkey, tkey FROM sk ANTI JOIN known USING (qkey, tkey)) ORDER BY qkey, tkey)
                    TO '{TMZ}/pref.tsv' (DELIMITER '\t', HEADER false)""")
    n_known = con.execute(f"SELECT count(*) FROM read_parquet('{TMZ}/lookup.parquet')").fetchone()[0]
    stats = dict(sample_rows=con.execute("SELECT count(*) FROM sk").fetchone()[0], unique_pairs=n_unique,
                 looked_up=n_known, to_align=n_unique - n_known,
                 looked_up_by_source=con.execute(f"SELECT src, count(*) FROM read_parquet('{TMZ}/lookup.parquet') GROUP BY ALL").fetchall(),
                 sets=con.execute("SELECT arm, pset, count(*) FROM sk GROUP BY ALL ORDER BY ALL").fetchall())
    (TMZ / "prep_stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
    log(stats)


def summarize(a):
    out = W / "results"
    con = C.connect(a.threads)
    con.execute(f"CREATE TABLE sk AS SELECT * FROM read_parquet('{TMZ}/sample.parquet')")
    con.execute(f"CREATE TABLE known AS SELECT qkey, tkey, tm_max, tm_min, src FROM read_parquet('{TMZ}/lookup.parquet')")
    p = TMZ / "aln_tm.tsv"
    if p.exists() and p.stat().st_size > 0:
        con.execute(f"""INSERT INTO known
            SELECT CAST(substr(q, 2) AS BIGINT), CAST(substr(t, 2) AS BIGINT), greatest(qtm, ttm), least(qtm, ttm), 'ws_seqanchors_round2'
            FROM read_csv('{p}', delim='\t', header=false,
                names=['q','t','qtm','ttm','alntm','lddt','fident','qcov','tcov','alnlen','qlen','tlen','evalue','bits'],
                types={{'q':'VARCHAR','t':'VARCHAR','qtm':'DOUBLE','ttm':'DOUBLE'}})""")
    con.execute("""CREATE TABLE tmk AS SELECT qkey, tkey, max(tm_max) AS tm_max, max(tm_min) AS tm_min, min(src) AS src
                   FROM known GROUP BY ALL""")
    j = con.execute("SELECT sk.*, tmk.tm_max, tmk.tm_min, tmk.src FROM sk LEFT JOIN tmk USING (qkey, tkey)").df()
    j.to_parquet(TMZ / "sample_scored.parquet", index=False)
    rows = []
    for (arm, pset), g in j.groupby(["arm", "pset"]):
        sc = g[g.tm_max.notna()]
        k = int((sc.tm_max >= 0.5).sum())
        lo, hi = wilson(k, len(sc))
        rows.append(dict(arm=arm, pset=pset, sampled=len(g), scored=len(sc), confirmed=k,
                         confirmed_frac=k / len(sc) if len(sc) else np.nan, ci_lo=lo, ci_hi=hi,
                         strict_frac=float((sc.tm_min >= 0.5).mean()) if len(sc) else np.nan,
                         median_tm=float(sc.tm_max.median()) if len(sc) else np.nan,
                         sources=dict(sc.src.value_counts())))
    conf = pd.DataFrame(rows)
    conf.to_csv(out / "zstructure_confirmation.tsv", sep="\t", index=False)
    comp = pd.read_csv(out / "zdecisive_counts.tsv", sep="\t")
    verdict = []
    for arm in TM_ARMS:
        u = comp[(comp.arm == arm) & (comp.reference == "union of sequence arms")].iloc[0]
        c = conf.set_index(["arm", "pset"])
        f_nohit = c.loc[(arm, "nohit_union"), "confirmed_frac"]
        f_ins = c.loc[(arm, "insearch_union"), "confirmed_frac"]
        est = (u.arm_only_no_hit * f_nohit + u.arm_only_in_search * f_ins) / max(u.pairs_arm_only, 1)
        material = (u.pairs_arm_only >= 0.02 * UNION_PAIRS) or (u.covered_added_by_arm >= 0.05 * UNION_GENES)
        verdict.append(dict(arm=arm, additions=int(u.pairs_arm_only), additions_frac=u.arm_only_over_ref,
                            covered_added=int(u.covered_added_by_arm), covered_added_frac=u.covered_added_by_arm_frac,
                            no_hit_additions=int(u.arm_only_no_hit), no_hit_confirmed=f_nohit,
                            no_hit_ci_lo=c.loc[(arm, "nohit_union"), "ci_lo"], no_hit_ci_hi=c.loc[(arm, "nohit_union"), "ci_hi"],
                            in_search_confirmed=f_ins, est_confirmed_frac=est,
                            est_confirmed_pairs=int(round(est * u.pairs_arm_only)),
                            est_confirmed_no_hit_pairs=int(round(f_nohit * u.arm_only_no_hit)),
                            material=bool(material), quality=bool(f_nohit >= 0.5),
                            changes_verdict=bool(material and f_nohit >= 0.5)))
    v = pd.DataFrame(verdict)
    v.to_csv(out / "zverdict.tsv", sep="\t", index=False)
    print(md(conf.drop(columns=["sources"])), "\n\n", md(v), flush=True)


def paired(a):
    sys.path.insert(0, str(W / "code/scale_code"))
    from evaluate_spacedust_queries import evaluate
    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    genes = pd.read_parquet(ROOT / "bgc_queries/metadata.parquet")
    span = ["query_genome", "query_contig", "query_start", "query_end"]
    old = pd.read_csv(W / "results/per_region_excess.tsv", sep="\t", index_col=0)
    per = {c: old[c].sort_index() for c in old.columns}
    for arm in NEW:
        calls = pd.read_parquet(W / "work/arms" / tag(arm) / "block_calls.parquet", columns=span + ["accept05"])
        obs, ctrl, _ = evaluate(regions, genes, calls[calls.accept05][span].drop_duplicates(), threads=a.threads,
                                controls=200, seed=17)
        per[arm] = (obs.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).sort_index()
        log(arm, f"{per[arm].mean():+.4f}")
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(regions), size=(10000, len(regions)))
    rows = []
    for arm in NEW:
        m = arm.split("@")[1]
        refs = ["S-top60/bitrank", "S-all/bitrank", f"E-seed60/rank@{m}", f"H-seed60@{m}"]
        if arm.startswith(("Z-", "L-")):
            refs.append(f"R-seed60@{m}")
        if arm.startswith("HL-"):
            refs.append(f"HZ-seed60@{m}")
        if arm.startswith("L-"):
            refs.append(f"Z-seed60@{m}")
        for ref in refs:
            if arm.startswith("R-") and ref.startswith("R-"):
                continue
            d = (per[arm] - per[ref]).to_numpy()
            boot = d[idx].mean(axis=1)
            rows.append(dict(arm=arm, reference=ref, excess_arm=float(per[arm].mean()), excess_ref=float(per[ref].mean()),
                             difference=float(d.mean()), ci_lo=float(np.quantile(boot, .025)),
                             ci_hi=float(np.quantile(boot, .975))))
    df = pd.DataFrame(rows)
    df.to_csv(W / "results/zpaired_f1_excess.tsv", sep="\t", index=False)
    pd.DataFrame({k: per[k] for k in NEW}).to_csv(W / "results/zper_region_excess.tsv", sep="\t")
    print(md(df, {c: "{:+.4f}" for c in ("excess_arm", "excess_ref", "difference", "ci_lo", "ci_hi")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "summarize", "paired"])
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    {"prep": prep, "summarize": summarize, "paired": paired}[a.mode](a)
