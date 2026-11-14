#!/usr/bin/env python
"""Stage F: metrics for every variant x acceptance rule of one model (or combine models).

Rules: raw (all blocks); cos5 (per-size mean-cosine cut, cut_for at 5%, as permuted_panels);
q5 (V4: sum of anchor log-odds, empirical q-value within size strata, q <= 0.05);
T0 / T4 (score-first variants only: built-in chain cut on the chain objective).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HELDOUT, MODELS, PERM_SEEDS, RD, ROOT, SPAN, W, cut_for, load_meta, log, qvalues,
                    universe_keys, write_json)

SIZES = (2, 3, 4, 5, 6)
REF = ("context60__lf", "cos5")


def sizes_of(n):
    return np.minimum(np.asarray(n), 6)


def ref_strata():
    u = pd.read_parquet(RD / "universe/genes.parquet", columns=["side", "row_index", "genome_code"])
    g = pd.read_parquet(RD / "universe/genomes.parquet", columns=["genome_code", "stratum"])
    r = u[u.side == "ref"].sort_values("row_index").merge(g, on="genome_code", how="left")
    return r.stratum.to_numpy()


def rule_masks(b, pm, kind):
    """Return {rule: (obs mask, perm mask)} for blocks b and permuted blocks pm of one variant."""
    out = {"raw": (np.ones(len(b), bool), np.ones(len(pm), bool))}
    bs, ps = sizes_of(b.n_anchors), sizes_of(pm.n_anchors)
    ob, pb = np.zeros(len(b), bool), np.zeros(len(pm), bool)
    cuts = {}
    for k in SIZES:
        so, sp = bs == k, ps == k
        t = cut_for(b.m.to_numpy()[so], pm.m.to_numpy()[sp], len(PERM_SEEDS), 0.05)
        cuts[k] = t
        if t is not None:
            ob |= so & (b.m.to_numpy() >= t)
            pb |= sp & (pm.m.to_numpy() >= t)
    out["cos5"] = (ob, pb)
    ob, pb = np.zeros(len(b), bool), np.zeros(len(pm), bool)
    qv = np.full(len(b), np.nan)
    for k in SIZES:
        so, sp = bs == k, ps == k
        q = qvalues(b.sum_w.to_numpy()[so], pm.sum_w.to_numpy()[sp], len(PERM_SEEDS))
        qv[so] = q
        acc = q <= 0.05
        if acc.any():
            t = b.sum_w.to_numpy()[so][acc].min()
            ob |= so & (qv <= 0.05)
            pb |= sp & (pm.sum_w.to_numpy() >= t)
    out["q5"] = (ob, pb)
    if kind == "score_first":
        for T in (0.0, 4.0):
            out[f"T{int(T)}"] = (b.chain_score.to_numpy() >= T, pm.chain_score.to_numpy() >= T)
    return out, cuts, qv


def per_model(a):
    sys.path.insert(0, str(W / "code/scale"))
    from evaluate_spacedust_queries import evaluate
    import pairs as pr
    m = a.model
    runs, out = W / "runs" / m, W / "results" / m
    if a.smoke:
        runs, out = W / "runs" / f"{m}_smoke", W / "results" / f"{m}_smoke"
    if a.tag:
        runs, out = W / "runs" / f"{m}_{a.tag}", W / "results" / f"{m}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    con = pr.connect(a.threads)
    srcs = pr.load_labels(con)
    qk, tk = universe_keys()
    strat = ref_strata()
    qmeta, _ = load_meta()
    held_q = qmeta.sample_id.isin(HELDOUT).to_numpy()
    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    perm = pd.read_parquet(runs / "perm_blocks.parquet")
    variants = sorted(p.name[len("blocks__"):-len(".parquet")] for p in runs.glob("blocks__*.parquet"))
    rng = np.random.default_rng(17)
    boot_idx = rng.integers(0, len(regions), size=(10000, len(regions)))
    res = dict(model=m, label_sources=srcs, variants={})
    rows, per_region = [], {}
    for v in variants:
        kind = "length_first" if v.endswith("__lf") else "score_first"
        b = pd.read_parquet(runs / f"blocks__{v}.parquet")
        an = pd.read_parquet(runs / f"anchors__{v}.parquet")
        an["qkey"] = qk[an.query_row_index.to_numpy()]
        an["tkey"] = tk[an.target_row_index.to_numpy()]
        con.register("an", an[["block_id", "qkey", "tkey"]])
        vis = con.execute("""SELECT an.block_id, count(*) AS n_pairs,
                                    sum(COALESCE(lab.evalue <= 1e-3, FALSE)::INT) AS n_vis,
                                    sum((lab.qkey IS NULL)::INT) AS n_unscored
                             FROM an LEFT JOIN lab USING (qkey, tkey) GROUP BY ALL""").df()
        con.unregister("an")
        b = b.merge(vis, on="block_id", how="left", validate="one_to_one")
        assert (b.n_pairs == b.n_anchors).all(), v
        first = an.groupby("block_id").target_row_index.first()
        b["stratum"] = strat[first.reindex(b.block_id).to_numpy()]
        b["held"] = b.query_genome.isin(HELDOUT).to_numpy()
        b["m"] = (b.chain_score / b.n_anchors) if kind == "length_first" else (b.sum_cos / b.n_anchors)
        pm = perm[perm.variant == v].reset_index(drop=True)
        pm["m"] = (pm.chain_score / pm.n_anchors) if kind == "length_first" else (pm.sum_cos / pm.n_anchors)
        masks, cuts, qv = rule_masks(b, pm, kind)
        b["q"] = qv
        bs, ps = sizes_of(b.n_anchors), sizes_of(pm.n_anchors)
        entry = dict(observed=len(b), permuted_mean=len(pm) / 3, ratio=len(pm) / 3 / max(len(b), 1),
                     cos5_cuts=cuts, by_size={str(k): dict(observed=int((bs == k).sum()),
                                                           permuted_mean=float((ps == k).sum() / 3),
                                                           ratio=float((ps == k).sum() / 3 / max((bs == k).sum(), 1)))
                                              for k in SIZES},
                     unscored_pairs=int(b.n_unscored.sum()), rules={})
        if kind == "score_first":
            curve = []
            for T in np.arange(0, 16.5, 0.5):
                o, p = int((b.chain_score >= T).sum()), float((pm.chain_score >= T).sum() / 3)
                curve.append(dict(T=float(T), observed=o, permuted_mean=p, ratio=p / max(o, 1),
                                  pairs=int(b.n_anchors[b.chain_score >= T].sum())))
            entry["T_curve"] = curve
        for rule, (ob, pb) in masks.items():
            acc = b[ob]
            spans = acc[SPAN].drop_duplicates()
            observed, ctrl, report = evaluate(regions, qmeta, spans, threads=a.threads, controls=200, seed=17)
            obs_f1 = observed.set_index("region_id").f1.reindex(regions.region_id)
            ctl_f1 = ctrl.groupby("region_id").f1.mean().reindex(regions.region_id)
            excess = (obs_f1 - ctl_f1).to_numpy()
            per_region[(v, rule)] = (obs_f1.to_numpy(), excess)
            boot = excess[boot_idx].mean(axis=1)
            heldreg = regions.sample_id.isin(HELDOUT).to_numpy()
            r = dict(variant=v, rule=rule, blocks=int(ob.sum()), anchor_pairs=int(acc.n_anchors.sum()),
                     permuted_mean=float(pb.sum() / 3), achieved_ratio=float(pb.sum() / 3 / max(ob.sum(), 1)),
                     vis_frac=float(acc.n_vis.sum() / max(acc.n_pairs.sum(), 1)),
                     vis_frac_heldout=float(acc.n_vis[acc.held].sum() / max(acc.n_pairs[acc.held].sum(), 1)),
                     pairs_heldout=int(acc.n_pairs[acc.held].sum()),
                     f1=report["observed"]["mean_best_single_block_f1"],
                     f1_random=report["controls"]["mean_best_single_block_f1"],
                     f1_excess=float(np.mean(excess)),
                     f1_excess_lo=float(np.quantile(boot, .025)), f1_excess_hi=float(np.quantile(boot, .975)),
                     f1_excess_heldout=float(np.mean(excess[heldreg])),
                     localized=report["observed"]["localized_fraction"],
                     localized_random=report["controls"]["localized_fraction"],
                     n_query_intervals=len(spans))
            for k in SIZES:
                r[f"blocks_s{k}"] = int((ob & (bs == k)).sum())
            st = acc.groupby("stratum").agg(blocks=("block_id", "size"), pairs=("n_pairs", "sum"), vis=("n_vis", "sum"))
            for s_name, srow in st.iterrows():
                r[f"blocks_{s_name}"] = int(srow.blocks)
                r[f"vis_{s_name}"] = float(srow.vis / max(srow.pairs, 1))
            rows.append(r)
            entry["rules"][rule] = r
            log(m, v, rule, {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in
                             ("blocks", "anchor_pairs", "achieved_ratio", "vis_frac", "f1", "f1_random", "f1_excess")})
        res["variants"][v] = entry
        b[["block_id", "n_anchors", "chain_score", "sum_cos", "sum_w", "m", "q", "n_vis", "stratum"]].to_parquet(
            out / f"calls__{v}.parquet", index=False)
    table = pd.DataFrame(rows)
    # paired comparison with calibrated V0
    ref_f1, ref_ex = per_region[REF]
    diffs = []
    for (v, rule), (f1v, exv) in per_region.items():
        d = exv - ref_ex
        df1 = f1v - ref_f1
        bd = d[boot_idx].mean(axis=1)
        bf = df1[boot_idx].mean(axis=1)
        diffs.append(dict(variant=v, rule=rule, d_excess=float(d.mean()), d_excess_lo=float(np.quantile(bd, .025)),
                          d_excess_hi=float(np.quantile(bd, .975)), d_f1=float(df1.mean()),
                          d_f1_lo=float(np.quantile(bf, .025)), d_f1_hi=float(np.quantile(bf, .975))))
    table = table.merge(pd.DataFrame(diffs), on=["variant", "rule"])
    refrow = table[(table.variant == REF[0]) & (table.rule == REF[1])].iloc[0]
    table["gain"] = ((table.achieved_ratio <= 0.05) & (table.anchor_pairs > refrow.anchor_pairs) &
                     (table.vis_frac >= refrow.vis_frac) & (table.f1 >= refrow.f1) &
                     (table.f1_excess >= refrow.f1_excess))
    table.insert(0, "model", m)
    table.to_csv(out / "table.tsv", sep="\t", index=False)
    pr_rows = [dict(variant=v, rule=r, region_id=rid, f1=f, excess=e)
               for (v, r), (f1v, exv) in per_region.items() for rid, f, e in zip(regions.region_id, f1v, exv)]
    pd.DataFrame(pr_rows).to_parquet(out / "per_region.parquet", index=False)
    write_json(out / "summary.json", res)
    log("done", m)


def combine(a):
    suffix = f"_{a.tag}" if a.tag else ""
    tabs = [pd.read_csv(W / "results" / f"{m}{suffix}" / "table.tsv", sep="\t") for m in MODELS
            if (W / "results" / f"{m}{suffix}" / "table.tsv").exists()]
    t = pd.concat(tabs, ignore_index=True)
    t.to_csv(W / "results" / f"ablation_table{suffix}.tsv", sep="\t", index=False)
    log("combined", len(t))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    combine(a) if a.combine else per_model(a)
