#!/usr/bin/env python
"""Spacedust source207: single-reading scores next to the best-of-many figures (workstream ws_baseline).

The engine's source207 numbers in docs/redesign/NOTEBOOK.md ("conserved loci, gaps 0-2") take, per region,
the best single locus over 42 readings (kinds conserved / conserved_g1 / conserved_g2 x 14 support
levels). The legacy calibrated-block numbers take the best of all accepted block intervals on the
query genome (one per reference partner). This scores, with the same `evaluate` (controls 200, seed 17):
  engine:  conserved g0-2 (replay of the notebook figure), conserved gap 0 all levels, conserved@2 (gap 0,
           level 2: one reading), conserved_g2@2
  legacy:  calibrated blocks, all intervals (replay of the notebook figure); union of the calibrated
           block intervals per query contig (maximal runs of spanned gene boundaries: one reading)
usage: spacedust_single.py --out DIR --threads N
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
S = ROOT / "seq_invisible_20261111"
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
MODELS = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esmc300m", "esmc600m", "esmc6b"]
ENGINE = ["results_v3/mmseqs__seq__e3c50s75__w5", "results_v3/mmseqs__seq__e3c50s75__w8",
          "results_v3/hybrid__esm2_650m__genomecentred__e3c50cpmr0p05__w5",
          "results_v3/hybrid__esm2_650m__genomecentred__e3c50cpmr0p05__w8"]
sys.path.insert(0, str(SCALE / "code"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def union_spans(spans):
    """Maximal runs of gene boundaries spanned by >= 1 interval, per (query_genome, query_contig)."""
    out = []
    for (g, c), d in spans.groupby(["query_genome", "query_contig"], sort=False):
        s = d.query_start.to_numpy(np.int64)
        e = d.query_end.to_numpy(np.int64)
        n = int(e.max()) + 2
        cnt = np.cumsum(np.bincount(s, minlength=n) - np.bincount(e, minlength=n))[:n] > 0
        k = np.flatnonzero(cnt)
        if len(k) == 0:
            continue
        new = np.r_[True, np.diff(k) != 1]
        k0, k1 = k[new], k[np.r_[new[1:], True]]
        out.append(pd.DataFrame({"query_genome": g, "query_contig": c, "query_start": k0, "query_end": k1 + 1}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    from evaluate_spacedust_queries import evaluate
    import run_spacedust_scaling as rss
    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
    rng = np.random.default_rng(17)
    rows = []
    per_region = {}

    def score(spans, source, reading):
        spans = spans[SPAN].drop_duplicates()
        observed, ctrl, report = evaluate(regions, qmeta, spans, threads=a.threads, controls=200, seed=17)
        o = observed.set_index("region_id").f1
        cm = ctrl.groupby("region_id").f1.mean().reindex(o.index)
        per_region[(source, reading)] = pd.DataFrame({"f1": o, "control": cm})
        per = (o - cm).to_numpy()
        boot = rng.choice(per, size=(10000, len(per)), replace=True).mean(axis=1)
        r = dict(source=source, reading=reading, n_intervals=int(len(spans)),
                 median_span=float(np.median(spans.query_end - spans.query_start + 1)),
                 f1=round(report["observed"]["mean_best_single_block_f1"], 4),
                 random=round(report["controls"]["mean_best_single_block_f1"], 4),
                 excess=round(float(per.mean()), 4),
                 ci=f"[{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]")
        log(r)
        rows.append(r)

    for d in ENGINE:
        l = pd.read_parquet(RD / "engine" / d / "query_loci.parquet")
        k = l.kind
        name = Path(d).name
        score(l[k.str.startswith("conserved")], name, "conserved g0-2, 14 levels (best of 42; notebook figure)")
        score(l[k == "conserved"], name, "conserved gap 0, 14 levels (best of 14)")
        score(l[(k == "conserved") & (l.level == 2)], name, "conserved@2 (single reading)")
        score(l[(k == "conserved_g2") & (l.level == 2)], name, "conserved_g2@2 (single reading)")
    for m in MODELS:
        b = pd.read_parquet(SCALE / "results" / m / "analysis/full_workflow/all_blocks.parquet",
                            columns=["block_id"] + SPAN)
        calls = pd.read_parquet(S / "results/permuted_panels" / f"{m}_block_calls.parquet", columns=["block_id", "accept05"])
        acc = b.merge(calls, on="block_id", validate="one_to_one")
        acc = acc[acc.accept05]
        score(acc, f"legacy {m}", "calibrated blocks, all intervals (notebook figure)")
        score(union_spans(acc[SPAN].drop_duplicates()), f"legacy {m}", "union of calibrated blocks (single reading)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "spacedust_single.tsv", sep="\t", index=False)
    pr = pd.concat({f"{k[0]} | {k[1]}": v for k, v in per_region.items()}, names=["set", "region_id"]).reset_index()
    pr.to_parquet(out / "per_region.parquet", index=False)
    # paired region bootstrap (10,000, seed 17): engine reading minus legacy reading, like with like
    prng = np.random.default_rng(17)
    pairs = []
    for e in ("mmseqs__seq__e3c50s75__w5", "mmseqs__seq__e3c50s75__w8",
              "hybrid__esm2_650m__genomecentred__e3c50cpmr0p05__w5"):
        for m in MODELS:
            for er, lr, label in (("conserved@2 (single reading)", "union of calibrated blocks (single reading)", "single vs single"),
                                  ("conserved g0-2, 14 levels (best of 42; notebook figure)",
                                   "calibrated blocks, all intervals (notebook figure)", "many vs many")):
                x, y = per_region[(e, er)], per_region[(f"legacy {m}", lr)].reindex(per_region[(e, er)].index)
                dex = ((x.f1 - x.control) - (y.f1 - y.control)).to_numpy()
                df1 = (x.f1 - y.f1).to_numpy()
                idx = prng.integers(0, len(dex), size=(10000, len(dex)))
                be, bf = dex[idx].mean(1), df1[idx].mean(1)
                pairs.append(dict(engine=e, legacy=m, comparison=label, d_f1=round(float(df1.mean()), 4),
                                  d_f1_ci=f"[{np.quantile(bf, .025):+.3f}, {np.quantile(bf, .975):+.3f}]",
                                  d_excess=round(float(dex.mean()), 4),
                                  d_excess_ci=f"[{np.quantile(be, .025):+.3f}, {np.quantile(be, .975):+.3f}]"))
    P = pd.DataFrame(pairs)
    P.to_csv(out / "spacedust_paired.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500):
        print(t.to_string(index=False))
        print(P.to_string(index=False))


if __name__ == "__main__":
    main()
