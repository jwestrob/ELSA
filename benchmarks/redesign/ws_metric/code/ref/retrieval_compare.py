#!/usr/bin/env python
"""Does the retrieval budget limit cross-phylum discovery?

Compares saved ESM2-8M (t6) runs that differ only in how candidates are chosen,
each with its own whole-pipeline order-permutation panels:

  context60    global ANN60 seeds + one-hop context selection, 60 targets per query
  scaffold_k5  exhaustive top-5 per reference scaffold (519 M candidates)
  global60     exact global top-60, no context selection

For each arm: per-size cosine cut holding permuted/observed at 5%, then the
calibrated blocks are stratified by the GTDB rank of the target genome.

prep       calibrate, write block calls and the pair list for mmseqs align
summarize  join sequence visibility, stratify, score the 207 regions
"""
import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
FULL = ROOT / "full_reference_interventions_20260910/results"
G = "/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232"
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
SEEDS = ("order_seed17", "order_seed18", "order_seed19")
SIZES = (2, 3, 4, 5, 6, 7, 8, 9, 10)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cut_for(obs_m, perm_m, n_perm, target):
    if len(obs_m) == 0:
        return None
    obs_sorted, perm_sorted = np.sort(obs_m), np.sort(perm_m)
    for t in np.unique(np.quantile(obs_m, np.linspace(0, 1, 401))):
        o = len(obs_sorted) - np.searchsorted(obs_sorted, t, side="left")
        p = (len(perm_sorted) - np.searchsorted(perm_sorted, t, side="left")) / n_perm
        if o > 0 and p / o <= target:
            return float(t)
    return None


def arms(run):
    out = {"context60": dict(
        observed=SCALE / "results/esm2_8m/analysis/full_workflow/all_blocks.parquet",
        permuted=[run / "results/permuted_panels/esm2_8m_permuted_blocks.parquet"], n_perm=3)}
    for name, sub in (("scaffold_k5", "scaffold_k5/alternatives"), ("global60", "global60/alternatives")):
        obs = FULL / sub / "observed/all_blocks.parquet"
        perms = [FULL / sub / s / "all_blocks.parquet" for s in SEEDS]
        if obs.exists() and all(p.exists() for p in perms):
            out[name] = dict(observed=obs, permuted=perms, n_perm=3)
    return out


def prep(a):
    K = a.run / "work_retrieval"
    K.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"CREATE TABLE qx AS SELECT * FROM read_parquet('{a.run}/work/qx.parquet')")
    stats = {}
    con.execute("""CREATE TABLE BP (arm VARCHAR, block_id BIGINT, n_anchors BIGINT, m DOUBLE,
                   target_genome VARCHAR, qkey BIGINT, tkey BIGINT)""")
    for arm, spec in arms(a.run).items():
        obs = pd.read_parquet(spec["observed"], columns=["block_id", "n_anchors", "chain_score"] + SPAN + ["target_genome"])
        obs["m"] = obs.chain_score / obs.n_anchors
        perm = pd.concat([pd.read_parquet(p, columns=["n_anchors", "chain_score"]) for p in spec["permuted"]])
        perm["m"] = perm.chain_score / perm.n_anchors
        accept = np.zeros(len(obs), bool)
        sizes = {}
        for size in SIZES:
            last = size == SIZES[-1]
            so = obs[obs.n_anchors >= size] if last else obs[obs.n_anchors == size]
            sp = perm[perm.n_anchors >= size] if last else perm[perm.n_anchors == size]
            t = cut_for(so.m.to_numpy(), sp.m.to_numpy(), spec["n_perm"], 0.05)
            n_acc = 0
            if t is not None:
                acc = so[so.m >= t]
                accept[acc.index.to_numpy()] = True
                n_acc = len(acc)
            sizes[f"{size}{'+' if last else ''}"] = dict(
                observed=len(so), permuted_mean=len(sp) / spec["n_perm"],
                ratio=(len(sp) / spec["n_perm"]) / max(len(so), 1), cut05=t, accepted=n_acc)
        obs["accept05"] = accept
        calls = K / f"{arm}_block_calls.parquet"
        obs.to_parquet(calls, index=False)
        stats[arm] = dict(observed_blocks=len(obs), permuted_mean=len(perm) / spec["n_perm"],
                          accepted=int(accept.sum()), sizes=sizes)
        log(arm, "observed", len(obs), "permuted mean", len(perm) / spec["n_perm"], "accepted", int(accept.sum()))
        con.execute(f"""
            INSERT INTO BP
            WITH c AS (SELECT block_id, n_anchors, m, target_genome FROM read_parquet('{calls}') WHERE accept05),
            b AS (
                SELECT block_id, from_json(anchor_query_gene_ids, '["VARCHAR"]') AS q,
                       CASE WHEN orientation = -1
                            THEN list_reverse(from_json(anchor_target_gene_ids, '["VARCHAR"]'))
                            ELSE from_json(anchor_target_gene_ids, '["VARCHAR"]') END AS t
                FROM read_parquet('{spec["observed"]}') WHERE block_id IN (SELECT block_id FROM c)),
            e AS (SELECT block_id, unnest(q) AS qg, unnest(t) AS tg FROM b)
            SELECT '{arm}', e.block_id, c.n_anchors, c.m, c.target_genome, qx.qkey,
                   CAST(split_part(e.tg, ':', 2) AS BIGINT)
            FROM e JOIN qx ON qx.gene_id = e.qg JOIN c USING (block_id)""")
        got, want = con.execute(f"""SELECT (SELECT count(*) FROM BP WHERE arm = '{arm}'),
            (SELECT sum(n_anchors) FROM read_parquet('{calls}') WHERE accept05)""").fetchone()
        assert got == want, (arm, got, want)
    con.execute(f"COPY BP TO '{K}/block_pairs.parquet' (FORMAT parquet)")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM (SELECT DISTINCT qkey, tkey FROM BP)
                          ORDER BY qkey, tkey) TO '{K}/pref.tsv' (DELIMITER '\t', HEADER false)""")
    stats["unique_pairs"] = con.execute("SELECT count(*) FROM (SELECT DISTINCT qkey, tkey FROM BP)").fetchone()[0]
    with open(K / "prep_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    log("pairs to align", stats["unique_pairs"])


def summarize(a):
    K, OUT = a.run / "work_retrieval", a.run / "results/retrieval_compare"
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCALE / "code"))
    from evaluate_spacedust_queries import evaluate
    import run_spacedust_scaling as rss
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{a.run}/tmp'")
    con.execute(f"CREATE TABLE tx AS SELECT * FROM read_parquet('{a.run}/work/tx.parquet')")
    con.execute(f"CREATE TABLE BP AS SELECT * FROM read_parquet('{K}/block_pairs.parquet')")
    con.execute(f"""
        CREATE TABLE seq AS
        SELECT qkey, tkey, evalue, seqid FROM read_csv('{K}/aln_seq.tsv', delim='\t', header=false,
            names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'],
            types={{'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}})""")
    con.execute(f"""
        CREATE TABLE gtdb AS
        SELECT regexp_extract(acc, 'GC[AF]_([0-9]{{9}})', 1) AS id9, any_value(tax) AS tax
        FROM read_csv(['{G}/bac120_taxonomy_r232.tsv', '{G}/ar53_taxonomy_r232.tsv'],
                      delim='\t', header=false, columns={{'acc':'VARCHAR','tax':'VARCHAR'}}) GROUP BY ALL""")
    con.execute("""
        CREATE TABLE genome AS
        WITH g AS (SELECT DISTINCT tgenome, regexp_extract(tgenome, 'GC[AF]_([0-9]{9})', 1) AS id9 FROM tx)
        SELECT g.tgenome, gtdb.tax, replace(split_part(gtdb.tax, ';', 2), 'p__', '') AS phylum,
               replace(split_part(gtdb.tax, ';', 3), 'c__', '') AS class,
               replace(split_part(gtdb.tax, ';', 6), 'g__', '') AS genus
        FROM g LEFT JOIN gtdb USING (id9)""")
    ref_phylum, ref_class = con.execute("""SELECT phylum, class FROM genome WHERE genus = 'Streptomyces'
                                           GROUP BY ALL ORDER BY count(*) DESC LIMIT 1""").fetchone()
    con.execute(f"""
        ALTER TABLE genome ADD COLUMN stratum VARCHAR;
        UPDATE genome SET stratum = CASE
            WHEN tax IS NULL THEN '9_no_gtdb'
            WHEN genus = 'Streptomyces' THEN '0_genus_Streptomyces'
            WHEN class = '{ref_class}' AND phylum = '{ref_phylum}' THEN '1_same_class_other_genus'
            WHEN phylum = '{ref_phylum}' THEN '2_same_phylum_other_class'
            ELSE '3_other_phylum' END""")
    con.execute("""
        CREATE TABLE J AS
        SELECT BP.*, COALESCE(seq.evalue <= 1e-3, FALSE) AS vis, seq.seqid, genome.stratum, genome.phylum
        FROM BP LEFT JOIN seq USING (qkey, tkey) JOIN genome ON genome.tgenome = BP.target_genome""")
    assert con.execute("SELECT (SELECT count(*) FROM BP) = (SELECT count(*) FROM J)").fetchone()[0]
    agg = """count(DISTINCT block_id) AS blocks, count(*) AS pairs, avg(vis::INT) AS pair_vis,
             median(seqid) FILTER (WHERE vis) AS med_ident_vis,
             count(DISTINCT target_genome) AS target_genomes, count(DISTINCT qkey) AS query_genes"""
    overall = con.execute(f"SELECT arm, {agg} FROM J GROUP BY ALL ORDER BY arm").df()
    bystr = con.execute(f"SELECT arm, stratum, {agg} FROM J GROUP BY ALL ORDER BY arm, stratum").df()
    bystr["block_share"] = bystr.blocks / bystr.groupby("arm").blocks.transform("sum")
    size = "CASE WHEN n_anchors >= 10 THEN '10+' WHEN n_anchors >= 5 THEN '5-9' ELSE CAST(n_anchors AS VARCHAR) END"
    bysize = con.execute(f"""SELECT arm, stratum = '3_other_phylum' AS other_phylum, {size} AS size_bin, {agg}
                             FROM J GROUP BY ALL ORDER BY arm, other_phylum, size_bin""").df()
    byphy = con.execute("""SELECT arm, COALESCE(phylum, 'no_gtdb') AS phylum, count(DISTINCT block_id) AS blocks,
                                  count(*) AS pairs, avg(vis::INT) AS pair_vis, count(DISTINCT target_genome) AS target_genomes
                           FROM J GROUP BY ALL ORDER BY arm, blocks DESC""").df()
    byphy = byphy.groupby("arm").head(12)

    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
    rng = np.random.default_rng(17)
    f1 = []
    for arm in overall.arm:
        calls = pd.read_parquet(K / f"{arm}_block_calls.parquet")
        for label, mask in (("all", np.ones(len(calls), bool)), ("calibrated_5pct", calls.accept05.to_numpy())):
            spans = calls[mask][SPAN].drop_duplicates()
            observed, ctrl, report = evaluate(regions, qmeta, spans, threads=a.threads, controls=200, seed=17)
            per = (observed.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).to_numpy()
            boot = rng.choice(per, size=(10000, len(per)), replace=True).mean(axis=1)
            f1.append(dict(arm=arm, selection=label, n_blocks=int(mask.sum()),
                           f1_observed=report["observed"]["mean_best_single_block_f1"],
                           f1_random=report["controls"]["mean_best_single_block_f1"], f1_excess=float(per.mean()),
                           ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975))))
            log(f1[-1])
    f1 = pd.DataFrame(f1)
    for name, df in (("overall", overall), ("by_stratum", bystr), ("by_size", bysize), ("by_phylum", byphy), ("source207", f1)):
        df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
    prep_stats = json.load(open(K / "prep_stats.json"))

    def md(df, fmt=None):
        fmt = fmt or {}
        cols = list(df.columns)
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

    fm = {"pair_vis": "{:.2%}", "med_ident_vis": "{:.3f}", "block_share": "{:.2%}", "f1_observed": "{:.4f}",
          "f1_random": "{:.4f}", "f1_excess": "{:+.4f}", "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}"}
    L = ["# Retrieval budget and cross-phylum discovery (ESM2-8M)", ""]
    for arm, s in prep_stats.items():
        if not isinstance(s, dict):
            continue
        L += [f"## {arm}: {s['observed_blocks']:,} observed, {s['permuted_mean']:,.0f} permuted mean, {s['accepted']:,} calibrated", "",
              "| size | observed | permuted/observed | cosine cut | accepted |", "|---|---|---|---|---|"]
        for size, e in s["sizes"].items():
            cut = "-" if e["cut05"] is None else f"{e['cut05']:.4f}"
            L.append(f"| {size} | {e['observed']:,} | {e['ratio']:.3f} | {cut} | {e['accepted']:,} |")
        L.append("")
    L += ["## Calibrated blocks, overall", "", md(overall, fm), "",
          "## Calibrated blocks by target stratum", "", md(bystr, fm), "",
          "## Calibrated blocks by size, other phylum vs same phylum", "", md(bysize, fm), "",
          "## Calibrated blocks by target phylum (top 12)", "", md(byphy, fm), "",
          "## source207", "", md(f1, fm), ""]
    with open(OUT / "RETRIEVAL_COMPARE.md", "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "summarize"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()
    (prep if args.mode == "prep" else summarize)(args)
