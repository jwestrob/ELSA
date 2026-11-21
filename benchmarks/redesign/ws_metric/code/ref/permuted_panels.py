#!/usr/bin/env python
"""Whole-pipeline order-permutation panels for every model, and calibrated scoring.

For one model:
  1. Replay the observed run from saved seeds (extension -> chaining) and require
     the saved block count.
  2. Permute complete query gene order (the September `ordered_metadata` rule,
     seeds 17/18/19) BEFORE neighbourhood selection, then rerun extension and
     chaining. These are valid chance panels for the whole order-dependent pipeline.
  3. Per block size, find the mean-anchor-cosine cut that holds the cumulative
     permuted/observed ratio at 5% and at 1%.
  4. Check accepted and rejected blocks against pairwise sequence visibility.
  5. Score the 207 annotated regions on each block selection, with 200 random
     placements per region.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
N = ROOT / "native_run_20260910"
MODELS = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esmc300m", "esmc600m", "esmc6b"]
SEEDS = (17, 18, 19)
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
SAVED_PERMUTED_TOTALS = {"esm2_8m": [495161, 495380, 495342], "esmc6b": [535642, 532999, 534641]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ordered_metadata(metadata, seed):
    """Identical to run_spacedust_seed_extension.ordered_metadata (September)."""
    result = metadata.copy()
    for (sample, contig), frame in metadata.groupby(["sample_id", "contig_id"], observed=True, sort=True):
        rng = np.random.default_rng(int.from_bytes(
            hashlib.sha256(f"{seed}:{sample}:{contig}".encode()).digest()[:8], "big"))
        result.loc[frame.index, "position_index"] = rng.permutation(len(frame))[
            frame.position_index.to_numpy(dtype=np.int64)]
    return result


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workers", type=int, required=True)
    a = ap.parse_args()
    model = MODELS[a.task]
    out = a.run / "results/permuted_panels"
    cache = a.run / "tmp" / f"panels_{model}"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    import numba
    numba.set_num_threads(a.workers)
    sys.path.insert(0, str(SCALE / "code"))
    import duckdb
    import run_spacedust_scaling as rss
    from evaluate_spacedust_queries import evaluate
    from spacedust_prepared_extension_executor import ARRAY_NAMES, extend_prepared_queries

    t0 = time.time()
    base = SCALE / "results" / model / "analysis/full_workflow"
    proto = json.load(open(base / "protocol.json"))
    chain, group_type = rss.load_published(SCALE / "production")
    qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
    rmeta = rss.complete_metadata(
        pd.read_parquet(ROOT / "results/esm2_t6/shard_000/export/metadata.parquet"), rss.REFERENCE_ROWS)
    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    paths = {name: base / "arrays" / f"{name}.npy" for name in ARRAY_NAMES if not name.endswith("_vectors")}
    paths["query_vectors"] = Path(proto["vectors"]["query"]["path"])
    paths["reference_vectors"] = Path(proto["vectors"]["reference"]["path"])
    for p in paths.values():
        assert p.exists(), p
    rows = np.arange(len(qmeta), dtype=np.int64)
    result = dict(model=model, workers=a.workers)

    def run_pipeline(meta, tag):
        p = dict(paths)
        if tag != "observed":
            nb = rss.neighbors(meta)
            np.save(cache / f"query_neighbors_{tag}.npy", nb)
            p["query_neighbors"] = cache / f"query_neighbors_{tag}.npy"
        ext = extend_prepared_queries(p, rows, workers=a.workers)
        numeric = ext.pop("anchors")
        blocks = rss.chain_numeric(numeric, meta, rmeta, chain, group_type)
        log(model, tag, "anchors", len(numeric), "blocks", len(blocks), f"{time.time()-t0:.0f}s")
        return blocks

    # ---- 1. observed replay --------------------------------------------------
    saved = pd.read_parquet(base / "all_blocks.parquet")
    replay = run_pipeline(qmeta, "observed")
    result["replay"] = dict(blocks=len(replay), saved_blocks=len(saved),
                            anchors=int(replay.n_anchors.sum()), saved_anchors=int(saved.n_anchors.sum()))
    assert len(replay) == len(saved) and int(replay.n_anchors.sum()) == int(saved.n_anchors.sum()), result["replay"]
    del replay

    # ---- 2. permuted panels --------------------------------------------------
    perms = []
    for seed in SEEDS:
        b = run_pipeline(ordered_metadata(qmeta, seed), f"order_seed{seed}")
        perms.append(b[["n_anchors", "chain_score"]].assign(seed=seed))
    perm = pd.concat(perms, ignore_index=True)
    perm["m"] = perm.chain_score / perm.n_anchors
    perm.to_parquet(out / f"{model}_permuted_blocks.parquet", index=False)
    totals = [int((perm.seed == s).sum()) for s in SEEDS]
    result["permuted_totals"] = totals
    if model in SAVED_PERMUTED_TOTALS:
        result["saved_permuted_totals"] = SAVED_PERMUTED_TOTALS[model]
        result["permuted_totals_match_saved"] = totals == SAVED_PERMUTED_TOTALS[model]
        log(model, "permuted totals", totals, "saved", SAVED_PERMUTED_TOTALS[model])

    # ---- 3/4. calibration and sequence visibility --------------------------------
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.workers}")
    con.execute(f"PRAGMA temp_directory='{a.run}/tmp'")
    vis = con.execute(f"""
        WITH qx AS (SELECT key AS qkey, gene_id FROM read_parquet('{N}/query/native_lookup_crosswalk.parquet')),
        seq AS (SELECT qkey, tkey, evalue FROM read_csv('{a.run}/work/aln_seq.tsv', delim='\t', header=false,
            names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'],
            types={{'qkey':'BIGINT','tkey':'BIGINT','bits':'DOUBLE','seqid':'DOUBLE','evalue':'DOUBLE'}})),
        b AS (SELECT block_id, from_json(anchor_query_gene_ids, '["VARCHAR"]') AS q,
                     CASE WHEN orientation = -1
                          THEN list_reverse(from_json(anchor_target_gene_ids, '["VARCHAR"]'))
                          ELSE from_json(anchor_target_gene_ids, '["VARCHAR"]') END AS t
              FROM read_parquet('{base}/all_blocks.parquet')),
        e AS (SELECT block_id, unnest(q) AS qg, unnest(t) AS tg FROM b)
        SELECT e.block_id, count(*) AS n_pairs,
               sum(COALESCE(seq.evalue <= 1e-3, FALSE)::INT) AS n_vis,
               sum((seq.qkey IS NULL)::INT) AS n_unscored
        FROM e JOIN qx ON qx.gene_id = e.qg
        LEFT JOIN seq ON seq.qkey = qx.qkey AND seq.tkey = CAST(split_part(e.tg, ':', 2) AS BIGINT)
        GROUP BY ALL""").df()
    obs = saved[["block_id", "n_anchors", "chain_score"] + SPAN].merge(vis, on="block_id", validate="one_to_one")
    assert len(obs) == len(saved) and (obs.n_pairs == obs.n_anchors).all() and int(obs.n_unscored.sum()) == 0
    obs["m"] = obs.chain_score / obs.n_anchors
    obs["all_vis"] = obs.n_vis == obs.n_pairs
    accept = {0.05: np.zeros(len(obs), bool), 0.01: np.zeros(len(obs), bool)}
    result["sizes"] = {}
    for size in (2, 3, 4, 5, 6):
        so = obs[obs.n_anchors == size] if size < 6 else obs[obs.n_anchors >= 6]
        sp = perm[perm.n_anchors == size] if size < 6 else perm[perm.n_anchors >= 6]
        om, pm = so.m.to_numpy(), sp.m.to_numpy()
        entry = dict(observed=len(so), permuted_mean=len(sp) / len(SEEDS),
                     ratio=(len(sp) / len(SEEDS)) / max(len(so), 1),
                     pair_vis_frac=float(so.n_vis.sum() / max(so.n_pairs.sum(), 1)),
                     blocks_all_vis=float(so.all_vis.mean()) if len(so) else None)
        for target in (0.05, 0.01):
            t = cut_for(om, pm, len(SEEDS), target)
            key = f"fdr{int(target * 100):02d}"
            if t is None:
                entry[key] = dict(threshold=None, accepted=0)
                continue
            acc, rej = so[so.m >= t], so[so.m < t]
            accept[target][acc.index.to_numpy()] = True
            entry[key] = dict(threshold=t, accepted=len(acc), accepted_frac=len(acc) / max(len(so), 1),
                              permuted_above=float((pm >= t).sum()) / len(SEEDS),
                              accepted_pair_vis=float(acc.n_vis.sum() / max(acc.n_pairs.sum(), 1)),
                              accepted_all_vis=float(acc.all_vis.mean()) if len(acc) else None,
                              rejected_pair_vis=float(rej.n_vis.sum() / max(rej.n_pairs.sum(), 1)) if len(rej) else None,
                              rejected_all_vis=float(rej.all_vis.mean()) if len(rej) else None)
        result["sizes"][str(size) if size < 6 else "6+"] = entry
        log(model, "size", size, entry["observed"], f"ratio {entry['ratio']:.3f}", entry["fdr05"])

    # ---- 5. source207 --------------------------------------------------------------
    rng = np.random.default_rng(17)

    def score(mask, label):
        b = obs[mask]
        spans = b[SPAN].drop_duplicates()
        observed, ctrl, report = evaluate(regions, qmeta, spans, threads=a.workers, controls=200, seed=17)
        per = (observed.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).to_numpy()
        boot = rng.choice(per, size=(10000, len(per)), replace=True).mean(axis=1)
        r = dict(selection=label, n_blocks=int(mask.sum()), n_query_intervals=len(spans),
                 pair_vis_frac=float(b.n_vis.sum() / max(b.n_pairs.sum(), 1)),
                 f1_observed=report["observed"]["mean_best_single_block_f1"],
                 f1_random=report["controls"]["mean_best_single_block_f1"],
                 f1_excess=float(per.mean()),
                 f1_excess_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                 localized_observed=report["observed"]["localized_fraction"],
                 localized_random=report["controls"]["localized_fraction"],
                 per_region_f1=observed.set_index("region_id").f1.round(6).to_dict())
        log(model, label, {k: r[k] for k in ("n_blocks", "pair_vis_frac", "f1_observed", "f1_random", "f1_excess")})
        return r

    n_arr = obs.n_anchors.to_numpy()
    result["source207"] = [
        score(np.ones(len(obs), bool), "all"),
        score(n_arr >= 5, "size>=5"),
        score(n_arr >= 4, "size>=4"),
        score(accept[0.05], "calibrated_5pct"),
        score(accept[0.01], "calibrated_1pct"),
        score(obs.all_vis.to_numpy(), "oracle_all_anchors_visible"),
    ]
    obs.assign(accept05=accept[0.05], accept01=accept[0.01])[
        ["block_id", "n_anchors", "m", "n_vis", "accept05", "accept01"]].to_parquet(
        out / f"{model}_block_calls.parquet", index=False)
    result["seconds"] = round(time.time() - t0, 1)
    with open(out / f"{model}.json", "w") as fh:
        json.dump(result, fh, indent=1)
    log("done", model, result["seconds"])


if __name__ == "__main__":
    main()
