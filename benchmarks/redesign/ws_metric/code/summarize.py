#!/usr/bin/env python
"""Collect retrieved pairs for alignment, then score every arm.

collect    union of retrieved (qgid, tgid) over all models and arms -> MMseqs pair list
summarize  retrieval metrics (sequence visibility, identity, strata, lineage leakage,
           Spacedust-pair recall) and pair-population AUROC/AP, with query-cluster
           bootstrap intervals for the key comparisons; writes TSV + Markdown
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import MODELS, N_REF, SETS, U, UNTRAINED, LEARNED, LEARNED_R2, W, log

R1 = [k for k in LEARNED if k not in LEARNED_R2]

SEQ_COLS = ("names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'], "
            "types={'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}")
STRATA = ["0_genus_Streptomyces", "1_same_class_other_genus", "2_same_phylum_other_class", "3_other_phylum", "9_no_gtdb"]
ACT = STRATA[:3]


def collect(a):
    base = W / a.tag
    A = base / "align"
    A.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    con.execute("SET preserve_insertion_order=true")
    globs = [str(base / "eval" / m / "hits" / "*.parquet") for m in MODELS if (base / "eval" / m / "hits").exists()]
    log("hit globs", globs)
    con.execute(f"""
        CREATE TABLE P AS
        SELECT DISTINCT q.mmseqs_key AS qkey, t.mmseqs_key AS tkey
        FROM read_parquet({globs}) h
        JOIN read_parquet('{U}/genes.parquet') q ON q.gid = h.qgid
        JOIN read_parquet('{U}/genes.parquet') t ON t.gid = h.tgid""")
    n = con.execute("SELECT count(*) FROM P").fetchone()[0]
    con.execute(f"COPY (SELECT qkey, tkey, 0 AS s, 0 AS d FROM P ORDER BY qkey, tkey) TO '{A}/pref.tsv' (DELIMITER '\t', HEADER false)")
    json.dump(dict(unique_pairs=n, globs=globs), open(A / "collect.json", "w"), indent=1)
    log("unique retrieved pairs", n)


def auc_ap(y, s, w=None):
    y = np.asarray(y, bool)
    ok = np.isfinite(s)
    if w is not None:
        ok &= w > 0
    y, s = y[ok], s[ok]
    if len(y) < 50 or y.all() or (~y).all():
        return np.nan, np.nan
    ww = None if w is None else w[ok]
    return roc_auc_score(y, s, sample_weight=ww), average_precision_score(y, s, sample_weight=ww)


def cluster_boot_diff(y, s1, s2, clusters, reps=200, seed=17):
    """Paired query-cluster bootstrap of AUROC(s1) - AUROC(s2)."""
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(clusters, return_inverse=True)
    out = []
    for _ in range(reps):
        cnt = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        w = cnt[inv].astype(float)
        a1, _ = auc_ap(y, s1, w)
        a2, _ = auc_ap(y, s2, w)
        out.append(a1 - a2)
    out = np.array(out)
    return float(np.nanquantile(out, 0.025)), float(np.nanquantile(out, 0.975))


def summarize(a):
    base = W / a.tag
    A = base / "align"
    OUT = base / "results"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{W}/tmp'")
    con.execute(f"""CREATE TABLE G AS SELECT g.gid, g.mmseqs_key AS key, g.genome, gn.stratum
                    FROM read_parquet('{U}/genes.parquet') g JOIN read_parquet('{U}/genomes.parquet') gn USING (genome_code)""")
    con.execute(f"""CREATE TABLE SEQ AS SELECT qkey, tkey, evalue, seqid
                    FROM read_csv('{A}/aln.tsv', delim='\t', header=false, {SEQ_COLS})""")
    con.execute(f"CREATE TABLE SD AS SELECT qgid, tgid, tstratum FROM read_parquet('{SETS}/sd_heldout.parquet')")
    rows, perq = [], []
    for m in MODELS:
        hd = base / "eval" / m / "hits"
        if not hd.exists():
            continue
        for f in sorted(hd.glob("*.parquet")):
            arm = f.stem
            con.execute(f"""
                CREATE OR REPLACE TABLE H AS
                SELECT h.search, h.qgid, h.rank, h.tgid, h.score, qg.genome AS qgenome, tg.stratum,
                       s.evalue, s.seqid, (s.evalue <= 1e-3) AS vis, s.qkey IS NOT NULL AS scored
                FROM read_parquet('{f}') h JOIN G qg ON qg.gid = h.qgid JOIN G tg ON tg.gid = h.tgid
                LEFT JOIN SEQ s ON s.qkey = qg.key AND s.tkey = tg.key""")
            miss = con.execute("SELECT count(*) FROM H WHERE NOT scored").fetchone()[0]
            assert miss == 0, (m, arm, miss)
            for search in ("all", "other"):
                r = con.execute(f"""
                    SELECT count(*) AS pairs, count(DISTINCT qgid) AS queries, avg(vis::INT) AS vis_frac,
                           avg(vis::INT) FILTER (WHERE rank <= 10) AS vis_frac_top10,
                           median(seqid) FILTER (WHERE vis) AS med_ident_vis,
                           avg((stratum = '0_genus_Streptomyces' AND NOT vis)::INT) AS strep_invis_share,
                           avg((stratum = '0_genus_Streptomyces')::INT) AS strep_share,
                           avg((NOT vis)::INT) FILTER (WHERE stratum = '0_genus_Streptomyces') AS strep_invis_frac,
                           {', '.join(f"avg((stratum = '{s}')::INT) AS share_{s.split('_')[0]}" for s in STRATA)}
                    FROM H WHERE search = '{search}'""").df().iloc[0].to_dict()
                q = con.execute(f"""SELECT qgid, qgenome, avg(vis::INT) AS v, max(vis::INT) AS anyv FROM H
                                    WHERE search = '{search}' GROUP BY ALL""").df()
                r["q_any_vis"] = float(q.anyv.mean())
                for g, gg in q.groupby("qgenome"):
                    r[f"vis_{g.split()[0][:4]}_{g.split()[1][:5]}"] = float(gg.v.mean())
                sdw = "" if search == "all" else "WHERE SD.tstratum = '3_other_phylum'"
                sd = con.execute(f"""
                    WITH qs AS (SELECT DISTINCT qgid FROM H WHERE search = '{search}'),
                         sd AS (SELECT SD.* FROM SD JOIN qs USING (qgid) {sdw})
                    SELECT count(*), count(h.qgid) FROM sd LEFT JOIN (SELECT qgid, tgid FROM H WHERE search = '{search}') h
                    USING (qgid, tgid)""").fetchone()
                r.update(model=m, arm=arm, search=search, sd_pairs=sd[0], sd_recall=sd[1] / max(sd[0], 1))
                rows.append(r)
                q = q.assign(model=m, arm=arm, search=search)
                perq.append(q[["model", "arm", "search", "qgid", "v"]])
            log(m, arm, f"{time.time()-t0:.0f}s")
    ret = pd.DataFrame(rows)
    ret.to_csv(OUT / "retrieval.tsv", sep="\t", index=False)
    perq = pd.concat(perq)
    perq.to_parquet(OUT / "retrieval_per_query.parquet", index=False)

    # bootstrap CI (query clusters) for vis-fraction differences vs raw and vs the best untrained arm
    rng = np.random.default_rng(17)
    ci_rows = []
    for (m, s), g in perq.groupby(["model", "search"]):
        piv = g.pivot(index="qgid", columns="arm", values="v")
        base_arms = [x for x in UNTRAINED if x in piv.columns]
        best_u = ret[(ret.model == m) & (ret.search == s) & ret.arm.isin(base_arms)].sort_values("vis_frac").arm.iloc[-1]
        idx = rng.integers(0, len(piv), size=(1000, len(piv)))
        for arm in piv.columns:
            for ref in ("raw", best_u):
                if ref == arm or ref not in piv:
                    continue
                dlt = (piv[arm] - piv[ref]).to_numpy()
                bs = dlt[idx].mean(axis=1)
                ci_rows.append(dict(model=m, search=s, arm=arm, ref=ref, diff=float(dlt.mean()),
                                    lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975))))
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "retrieval_diff_ci.tsv", sep="\t", index=False)

    # ---- pair populations ------------------------------------------------------------
    strat = con.execute("SELECT gid, stratum FROM G").df().set_index("gid").stratum
    met = []
    pair_tabs = {}
    for pop in ("popA", "xphy", "popB"):
        for m in MODELS:
            p = base / "eval" / m / f"pairs_{pop}.parquet"
            if p.exists():
                pair_tabs[(pop, m)] = pd.read_parquet(p)
    labels = {}
    for pop, path, lab in (("xphy", SETS / "xphy.parquet", "vis"), ("popB", SETS / "popB.parquet", "tm05")):
        if path.exists():
            labels[pop] = pd.read_parquet(path)[["qgid", "tgid", lab, "tstratum"] + (["any_accept05"] if pop == "popB" else [])].rename(columns={lab: "y"})
    boot = []
    for (pop, m), sc in pair_tabs.items():
        if pop == "popA":
            lab = pd.read_parquet(SETS / f"popA_{m}.parquet", columns=["qgid", "tgid", "vis", "tstratum"]).rename(columns={"vis": "y"})
            lab = lab.drop_duplicates(["qgid", "tgid"])
            df = sc.merge(lab, on=["qgid", "tgid"], how="inner")
        else:
            # identical pairs across models: keep pairs scored in every model
            common = None
            for mm in MODELS:
                if (pop, mm) in pair_tabs:
                    k = pair_tabs[(pop, mm)][["qgid", "tgid"]]
                    common = k if common is None else common.merge(k, on=["qgid", "tgid"])
            df = sc.merge(common, on=["qgid", "tgid"]).merge(labels[pop], on=["qgid", "tgid"], how="inner")
        arms = [c for c in sc.columns if c not in ("qgid", "tgid")]
        subsets = {"all": np.ones(len(df), bool), "in_phylum": df.tstratum.isin(ACT).to_numpy(),
                   "other_phylum": (df.tstratum == "3_other_phylum").to_numpy()}
        if pop == "popB":
            subsets["accepted_other"] = subsets["other_phylum"] & df.any_accept05.to_numpy()
            subsets["accepted_in"] = subsets["in_phylum"] & df.any_accept05.to_numpy()
        for sub, mask in subsets.items():
            y = df.y.to_numpy()[mask]
            for arm in arms:
                au, apv = auc_ap(y, df[arm].to_numpy()[mask])
                met.append(dict(pop=pop, model=m, subset=sub, arm=arm, n=int(mask.sum()), pos_rate=float(y.mean()) if len(y) else np.nan,
                                auroc=au, ap=apv))
        # bootstrap: best learned vs best untrained (by AUROC on the subset) for the decisive subsets
        for sub in ("in_phylum", "other_phylum"):
            mask = subsets[sub]
            if mask.sum() < 200:
                continue
            mm_ = pd.DataFrame([r for r in met if r["pop"] == pop and r["model"] == m and r["subset"] == sub])
            un = mm_[mm_.arm.isin(UNTRAINED + ["csls_q", "z_q", "rank_q"])].sort_values("auroc")
            for rnd, pool in (("r1", R1), ("r1+r2", list(LEARNED))):
                le = mm_[mm_.arm.isin(pool)].sort_values("auroc")
                if not len(un) or not len(le):
                    continue
                bu, bl = un.arm.iloc[-1], le.arm.iloc[-1]
                if rnd == "r1+r2" and boot and boot[-1]["rounds"] == "r1" and boot[-1]["best_learned"] == bl \
                        and boot[-1]["subset"] == sub and boot[-1]["pop"] == pop and boot[-1]["model"] == m:
                    boot.append(dict(boot[-1], rounds=rnd))
                    continue
                sub_df = df[mask]
                n_max = 300_000
                if len(sub_df) > n_max:
                    sub_df = sub_df.sample(n_max, random_state=17)
                lo, hi = cluster_boot_diff(sub_df.y.to_numpy(), sub_df[bl].to_numpy(), sub_df[bu].to_numpy(), sub_df.qgid.to_numpy())
                boot.append(dict(pop=pop, model=m, subset=sub, rounds=rnd, best_learned=bl, best_untrained=bu,
                                 diff=float(le.auroc.iloc[-1] - un.auroc.iloc[-1]), lo=lo, hi=hi))
        log("pairs", pop, m, len(df), f"{time.time()-t0:.0f}s")
    met = pd.DataFrame(met)
    met.to_csv(OUT / "pair_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(boot).to_csv(OUT / "pair_auroc_diff_ci.tsv", sep="\t", index=False)
    write_md(ret, met, ci, pd.DataFrame(boot), OUT)
    log("done", f"{time.time()-t0:.0f}s")


def write_md(ret, met, ci, boot, OUT):
    L = ["# ws_metric ablation", ""]
    for m in MODELS:
        r = ret[ret.model == m]
        if not len(r):
            continue
        L += [f"## {m}", ""]
        ra = r[r.search == "all"].set_index("arm")
        ro = r[r.search == "other"].set_index("arm")

        def pm(pop, sub, arm, col="auroc"):
            x = met[(met["pop"] == pop) & (met.model == m) & (met.subset == sub) & (met.arm == arm)]
            return x[col].iloc[0] if len(x) else np.nan
        cols = ["arm", "vis_all", "vis_other", "qany_other", "ident_vis_all", "strep_invis", "share_other_in_all",
                "sd_recall", "popA_auc_in", "popA_auc_oth", "xphy_auc_oth", "tm_auc_in", "tm_auc_oth"]
        L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        arms = [x for x in UNTRAINED + ["csls_q", "z_q", "rank_q"] + list(LEARNED)]
        for arm in arms:
            def g(df, c):
                return df.loc[arm, c] if arm in df.index else np.nan
            vals = [g(ra, "vis_frac"), g(ro, "vis_frac"), g(ro, "q_any_vis"), g(ra, "med_ident_vis"),
                    g(ra, "strep_invis_share"), g(ra, "share_3"), g(ra, "sd_recall"),
                    pm("popA", "in_phylum", arm), pm("popA", "other_phylum", arm), pm("xphy", "other_phylum", arm),
                    pm("popB", "in_phylum", arm), pm("popB", "other_phylum", arm)]
            if all(pd.isna(v) for v in vals):
                continue
            cells = [arm] + ["-" if pd.isna(v) else f"{v:.4f}" for v in vals]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Retrieval vis-fraction differences (query bootstrap, 1,000 reps)", "",
          "| model | search | arm | ref | diff | 95% CI |", "|---|---|---|---|---|---|"]
    for _, x in ci[ci.arm.isin(list(LEARNED)) | (ci.ref == "raw")].iterrows():
        L.append(f"| {x.model} | {x.search} | {x.arm} | {x.ref} | {x['diff']:+.4f} | [{x.lo:+.4f}, {x.hi:+.4f}] |")
    L += ["", "## Pair AUROC: best learned minus best untrained (query-cluster bootstrap, 200 reps)", "",
          "| pop | model | subset | rounds | best learned | best untrained | diff | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    for _, x in boot.iterrows():
        L.append(f"| {x['pop']} | {x.model} | {x.subset} | {x.rounds} | {x.best_learned} | {x.best_untrained} | {x['diff']:+.4f} | [{x.lo:+.4f}, {x.hi:+.4f}] |")
    (OUT / "ABLATION.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "summarize"])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    (collect if a.mode == "collect" else summarize)(a)
