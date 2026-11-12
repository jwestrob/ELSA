#!/usr/bin/env python
"""Calibration of legacy blocks and per-genome locus tables (workstream ws_baseline).

usage: post.py prepare [--arms raw zca]
  1. calibration (Spacedust audit rule): per block size 2, 3, 4, 5, 6+, the smallest cut on mean
     anchor cosine (chain_score / n_anchors) such that mean permuted blocks above the cut / observed
     blocks above the cut <= 0.05 (cut_for, identical to ws_chain/common.py and permuted_panels.py),
     over the within-contig order permutations seeds 17, 18, 19.
  2. locus tables for <arm> (as shipped) and <arm>_cal: each block contributes its query-side and
     target-side interval; identical (genome, contig, start, end) intervals are written once (the
     scorer de-duplicates them anyway). Columns genome, contig, start, end, module_id (-1).
  3. statistics for every variant and for the two engine references.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
B = RD / "baseline"
PKG = RD / "ws_dpann/package"
ENGINE = {"engine_base": RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3",
          "engine_rescue3": RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"}
SEEDS = (17, 18, 19)
SIZES = (2, 3, 4, 5, 6)
SIDES = ("query", "target")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


def cut_for(obs_m, perm_m, n_perm, target):
    """Identical to permuted_panels.cut_for / ws_chain common.cut_for."""
    if len(obs_m) == 0:
        return None
    obs_sorted, perm_sorted = np.sort(obs_m), np.sort(perm_m)
    for t in np.unique(np.quantile(obs_m, np.linspace(0, 1, 401))):
        o = len(obs_sorted) - np.searchsorted(obs_sorted, t, side="left")
        p = (len(perm_sorted) - np.searchsorted(perm_sorted, t, side="left")) / n_perm
        if o > 0 and p / o <= target:
            return float(t)
    return None


def calibrate(arm):
    run = B / "runs" / arm
    cols = ["query_genome", "target_genome", "query_contig", "target_contig", "query_start", "query_end",
            "target_start", "target_end", "n_anchors", "chain_score"]
    obs = pd.read_parquet(run / "blocks_checkpoint.parquet", columns=cols)
    perms = [pd.read_parquet(run / f"perm_{s}_blocks.parquet", columns=["n_anchors", "chain_score"]) for s in SEEDS]
    m_obs = obs.chain_score.to_numpy() / obs.n_anchors.to_numpy()
    s_obs = np.minimum(obs.n_anchors.to_numpy(), 6)
    m_p = np.concatenate([p.chain_score.to_numpy() / p.n_anchors.to_numpy() for p in perms])
    s_p = np.concatenate([np.minimum(p.n_anchors.to_numpy(), 6) for p in perms])
    acc = np.zeros(len(obs), bool)
    acc_p = np.zeros(len(m_p), bool)
    rows = []
    for k in SIZES:
        so, sp = s_obs == k, s_p == k
        t = cut_for(m_obs[so], m_p[sp], len(SEEDS), 0.05)
        if t is not None:
            acc |= so & (m_obs >= t)
            acc_p |= sp & (m_p >= t)
        n_o, n_p = int(so.sum()), sp.sum() / len(SEEDS)
        a_o, a_p = int((so & acc).sum()), (sp & acc_p).sum() / len(SEEDS)
        rows.append(dict(size=f"{k}+" if k == 6 else str(k), observed=n_o, permuted_mean=round(float(n_p), 1),
                         ratio=round(float(n_p / n_o), 4) if n_o else None, cut=t, accepted=a_o,
                         accepted_permuted_mean=round(float(a_p), 1),
                         accepted_ratio=round(float(a_p / a_o), 4) if a_o else None,
                         mean_cos_q50_obs=float(np.median(m_obs[so])) if n_o else None,
                         mean_cos_q50_perm=float(np.median(m_p[sp])) if sp.any() else None))
    perm_totals = [int(len(p)) for p in perms]
    res = dict(arm=arm, rule="per-size cut on mean anchor cosine, permuted/observed <= 0.05, 3 permutations",
               observed=int(len(obs)), permuted=perm_totals,
               ratio_all=round(float(np.mean(perm_totals) / len(obs)), 4),
               accepted=int(acc.sum()), accepted_ratio=round(float(acc_p.sum() / len(SEEDS) / max(acc.sum(), 1)), 4),
               by_size=rows)
    write_json(run / "calibration.json", res)
    log(json.dumps(res))
    return obs, acc


def interval_table(blocks):
    parts = []
    for s in SIDES:
        parts.append(pd.DataFrame({"genome": blocks[f"{s}_genome"].astype(str).to_numpy(),
                                   "contig": blocks[f"{s}_contig"].astype(str).to_numpy(),
                                   "start": blocks[f"{s}_start"].to_numpy(np.int64),
                                   "end": blocks[f"{s}_end"].to_numpy(np.int64)}))
    return pd.concat(parts, ignore_index=True)


def union_reading(iv, contigs, n_genes):
    """Single, non-overlapping reading of a set of block intervals: maximal runs of gene boundaries
    spanned by at least one interval (the engine's run rule at level 1, gap 0). Blocks that only abut
    stay separate loci. Returns genome, contig, start, end."""
    c = contigs.sort_values("first_gid").reset_index(drop=True)
    first = pd.Series(c.first_gid.to_numpy(), index=c.contig.to_numpy())
    f = first.reindex(iv.contig.to_numpy()).to_numpy().astype(np.int64)
    a = f + iv.start.to_numpy(np.int64)
    b = f + iv.end.to_numpy(np.int64)
    assert (b > a).all()
    d = np.bincount(a, minlength=n_genes + 1) - np.bincount(b, minlength=n_genes + 1)
    spanned = np.cumsum(d)[:n_genes] > 0                    # boundary k = between gid k and k + 1
    k = np.flatnonzero(spanned)
    if len(k) == 0:
        return pd.DataFrame(columns=["genome", "contig", "start", "end"])
    new = np.r_[True, np.diff(k) != 1]
    k0 = k[new]
    k1 = k[np.r_[new[1:], True]]
    gs, ge = k0, k1 + 1
    ci = np.searchsorted(c.first_gid.to_numpy(), gs, side="right") - 1
    assert (np.searchsorted(c.first_gid.to_numpy(), ge, side="right") - 1 == ci).all(), "run crosses a contig"
    return pd.DataFrame({"genome": c.genome.to_numpy()[ci], "contig": c.contig.to_numpy()[ci],
                         "start": gs - c.first_gid.to_numpy()[ci], "end": ge - c.first_gid.to_numpy()[ci]})


def write_loci(out, ivd):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "loci.tmp.parquet"
    ivd.to_parquet(tmp, index=False)
    os.replace(tmp, out / "loci.parquet")


def coverage_stats(iv, contigs, n_genes):
    """iv: genome, contig, start, end (possibly duplicated). Returns union-coverage numbers."""
    first = pd.Series(contigs.first_gid.to_numpy(), index=contigs.contig.to_numpy())
    f = first.reindex(iv.contig.to_numpy()).to_numpy()
    assert not np.isnan(f).any()
    a = f.astype(np.int64) + iv.start.to_numpy(np.int64)
    b = f.astype(np.int64) + iv.end.to_numpy(np.int64)
    d = np.bincount(a, minlength=n_genes + 1) - np.bincount(b + 1, minlength=n_genes + 1)
    cov = np.cumsum(d)[:n_genes] > 0
    return int(cov.sum()), cov


def locus_stats(iv_all, iv_distinct, contigs, n_genes, gold_contigs):
    span_all = (iv_all.end - iv_all.start + 1).to_numpy()
    span_d = (iv_distinct.end - iv_distinct.start + 1).to_numpy()
    covered, _ = coverage_stats(iv_distinct, contigs, n_genes)
    per_contig = iv_distinct.groupby("contig").size()
    gc = per_contig.reindex(gold_contigs).fillna(0)
    per_genome = iv_distinct.groupby("genome").size()
    return dict(intervals=int(len(iv_all)), distinct_intervals=int(len(iv_distinct)),
                median_span_intervals=float(np.median(span_all)), median_span_distinct=float(np.median(span_d)),
                span_quantiles_intervals={str(q): float(np.quantile(span_all, q)) for q in (0.1, 0.25, 0.75, 0.9, 0.99)},
                max_span=int(span_all.max()), genes_covered=covered, genes_covered_frac=round(covered / n_genes, 4),
                genomes_with_locus=int(iv_distinct.genome.nunique()),
                distinct_per_genome_median=float(per_genome.median()),
                distinct_per_gold_contig_max=int(gc.max()), distinct_per_gold_contig_q99=float(gc.quantile(0.99)),
                distinct_per_gold_contig_median=float(gc.median()))


def block_stats(blocks):
    na = blocks.n_anchors.to_numpy()
    sz = np.minimum(na, 6)
    return dict(blocks=int(len(blocks)), anchors_in_blocks=int(na.sum()),
                frac_blocks_2_3_anchors=round(float((na <= 3).mean()), 4) if len(na) else None,
                blocks_by_size={("6+" if k == 6 else str(k)): int((sz == k).sum()) for k in SIZES},
                median_anchors=float(np.median(na)) if len(na) else None,
                cross_genome_pairs=int(blocks.groupby(["query_genome", "target_genome"]).ngroups) if len(na) else 0)


def cmd_prepare(a):
    t0 = time.time()
    contigs = pd.read_parquet(PKG / "universe/contigs.parquet")
    n_genes = int((contigs.first_gid + contigs.n_genes).max())
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["contig", "resolved", "order_trusted",
                                                                    "n_members_resolved"])
    inst = inst[inst.resolved.fillna(False) & inst.order_trusted.fillna(False) & (inst.n_members_resolved >= 2)]
    gold_contigs = inst.contig.unique()
    stats = {}
    for arm in a.arms:
        obs, acc = calibrate(arm)
        for variant, blk in ((arm, obs), (f"{arm}_cal", obs[acc])):
            out = B / "results" / variant
            iv = interval_table(blk)
            ivd = iv.drop_duplicates(["genome", "contig", "start", "end"]).sort_values(
                ["genome", "contig", "start", "end"]).reset_index(drop=True)
            ivd["module_id"] = -1
            write_loci(out, ivd)
            st = dict(variant=variant, reading="all block intervals (both sides of every block)",
                      **block_stats(blk), **locus_stats(iv, ivd, contigs, n_genes, gold_contigs))
            stats[variant] = st
            write_json(out / "stats.json", st)
            log(variant, json.dumps(st))
            # single reading: union of the block intervals (maximal runs of spanned boundaries)
            un = union_reading(ivd, contigs, n_genes)
            un["module_id"] = -1
            write_loci(B / "results" / f"{variant}_union", un)
            st = dict(variant=f"{variant}_union", reading="union of block intervals (runs of spanned boundaries)",
                      **locus_stats(un, un, contigs, n_genes, gold_contigs))
            stats[f"{variant}_union"] = st
            write_json(B / "results" / f"{variant}_union" / "stats.json", st)
            log(f"{variant}_union", json.dumps(st))
    for name, d in ENGINE.items():
        l = pd.read_parquet(d / "loci_all_genomes.parquet")
        arm = name.split("_")[1]
        H = pd.read_parquet(RD / f"boundary/loci/{arm}/H_res0.3_f0.5.parquet")
        readings = (("full", l), ("c2", l[(l.kind == "conserved") & (l.level == 2.0)]), ("H", H))
        for label, sub in readings:
            ivd = sub[["genome", "contig", "start", "end", "module_id"]].drop_duplicates(
                ["genome", "contig", "start", "end"])
            st = dict(variant=f"{name}_{label}", rows=int(len(sub)),
                      **locus_stats(sub[["genome", "contig", "start", "end"]], ivd, contigs, n_genes, gold_contigs))
            if label != "full":
                write_loci(B / "results" / f"{name}_{label}", sub[["genome", "contig", "start", "end", "module_id"]]
                           .reset_index(drop=True))
                write_json(B / "results" / f"{name}_{label}" / "stats.json", st)
            stats[f"{name}_{label}"] = st
            log(name, label, json.dumps(st))
    prev = B / "results/locus_stats.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        old.update(stats)
        stats = old
    write_json(prev, stats)
    log("done", f"{time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--arms", nargs="+", default=["raw", "zca"])
    a = ap.parse_args()
    cmd_prepare(a)


if __name__ == "__main__":
    main()
