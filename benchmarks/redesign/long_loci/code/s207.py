#!/usr/bin/env python
"""source207 scoring under the matched protocol (workstream transfer_spacedust).

Uses model_scaling_20260913/code/evaluate_spacedust_queries.evaluate unchanged (best single interval per region, 200 rigid
translations per region seeded by region id). Summary numbers as recurrence.py's score(): excess = mean over regions of
(F1 - mean control F1), 95% interval from 10,000 bootstrap draws over regions with a caller-supplied rng.
Paired differences: per-region differences of F1 and of excess, 10,000 draws, rng seed 17.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
TS = RD / "transfer_spacedust"
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
sys.path.insert(0, str(SCALE / "code"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Bench:
    def __init__(self, threads):
        from evaluate_spacedust_queries import evaluate
        import run_spacedust_scaling as rss
        self.evaluate = evaluate
        self.threads = threads
        self.regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
        self.qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
        self.size = self.regions.member_position_indices_json.map(lambda s: len(json.loads(s))).to_numpy()
        self.region_ids = self.regions.region_id.to_numpy()
        self.qsizes = self.qmeta.groupby(["sample_id", "contig_id"]).size()

    def score(self, spans, label, rng=None):
        """spans: DataFrame with SPAN columns (query-genome intervals, inclusive position indices)."""
        spans = spans[SPAN].drop_duplicates()
        rng = rng if rng is not None else np.random.default_rng(17)
        if spans.empty:
            return dict(selection=label, n_intervals=0), None
        observed, ctrl, report = self.evaluate(self.regions, self.qmeta, spans, threads=self.threads, controls=200, seed=17)
        ob = observed.set_index("region_id").reindex(self.region_ids)
        cm = ctrl.groupby("region_id").f1.mean().reindex(self.region_ids)
        per = (ob.f1 - cm).to_numpy()
        # bootstrap over the same array (and order) as recurrence.py / block_fdr.py, so stored intervals reproduce
        per_ref = (observed.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).to_numpy()
        boot = rng.choice(per_ref, size=(10000, len(per_ref)), replace=True).mean(axis=1)
        per_region = pd.DataFrame(dict(region_id=self.region_ids, size=self.size, f1=ob.f1.to_numpy(),
                                       control_f1=cm.to_numpy(), excess=per, coverage=ob.coverage.to_numpy(),
                                       precision=ob.boundary_precision.to_numpy(), block_span=ob.block_span.to_numpy(),
                                       localized=ob.localized.to_numpy(), union_coverage=ob.union_coverage.to_numpy()))
        cov = query_gene_coverage(spans, self.qsizes)
        r = dict(selection=label, n_intervals=int(len(spans)),
                 f1_observed=report["observed"]["mean_best_single_block_f1"],
                 f1_random=report["controls"]["mean_best_single_block_f1"], f1_excess=float(per_ref.mean()),
                 ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975)),
                 localized_observed=report["observed"]["localized_fraction"],
                 localized_random=report["controls"]["localized_fraction"],
                 median_boundary_precision=report["observed"]["median_boundary_precision_at_best_f1"],
                 median_coverage=report["observed"]["median_coverage_at_best_f1"],
                 median_union_coverage=report["observed"]["median_union_coverage"],
                 regions_better_than_random=int((per > 0).sum()), query_gene_coverage=cov,
                 interval_span_q=np.quantile(spans.query_end - spans.query_start + 1, [.1, .5, .9, .99, 1]).tolist(),
                 by_size=by_size(per_region))
        return r, per_region


def by_size(pr):
    out = {}
    bins = [(1, 5), (6, 10), (11, 20), (21, 1000)]
    for lo, hi in bins:
        m = (pr["size"] >= lo) & (pr["size"] <= hi)
        s = pr[m]
        out[f"{lo}-{hi if hi < 1000 else '+'}"] = dict(
            regions=int(m.sum()), f1=float(s.f1.mean()), control=float(s.control_f1.mean()), excess=float(s.excess.mean()),
            coverage_q50=float(s.coverage.median()), precision_q50=float(s.precision.median()),
            best_span_q50=float(s.block_span.median()), size_q50=float(s["size"].median()))
    return out


def query_gene_coverage(spans, qsizes):
    covered = 0
    for key, b in spans.groupby(["query_genome", "query_contig"], sort=False):
        n = int(qsizes[key])
        d = np.zeros(n + 1, dtype=np.int64)
        np.add.at(d, b.query_start.to_numpy(dtype=np.int64), 1)
        np.add.at(d, np.minimum(b.query_end.to_numpy(dtype=np.int64) + 1, n), -1)
        covered += int((np.cumsum(d)[:n] > 0).sum())
    return covered / int(qsizes.sum())


def paired(A, B, seed=17, draws=10000):
    """A, B: per-region frames (same region order). Returns d F1 and d excess with 95% intervals, win/tie/loss."""
    assert (A.region_id.to_numpy() == B.region_id.to_numpy()).all()
    rng = np.random.default_rng(seed)
    out = {}
    idx = rng.integers(0, len(A), size=(draws, len(A)))
    for k in ("f1", "excess"):
        d = A[k].to_numpy() - B[k].to_numpy()
        bm = d[idx].mean(1)
        out[k] = [float(d.mean()), float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]
    d = A.f1.to_numpy() - B.f1.to_numpy()
    out["win_tie_loss_f1"] = [int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())]
    return out


def verdict(p):
    d, lo, hi = p["excess"]
    if lo > 0:
        return "BEATS"
    if d >= -0.02:
        return "MATCHES"
    return "BELOW"


def gene_spans(univ_genes, s, e):
    """gid intervals [s, e] on query genes -> SPAN frame (query genes only; both ends on one contig)."""
    g = univ_genes
    q = g.side.to_numpy()[s] == "query"
    s, e = s[q], e[q]
    return pd.DataFrame(dict(query_genome=g.genome.to_numpy()[s], query_contig=g.contig.to_numpy()[s],
                             query_start=g.position_index.to_numpy()[s], query_end=g.position_index.to_numpy()[e]))


def load_universe_genes():
    return pd.read_parquet(RD / "universe/genes.parquet", columns=["gid", "side", "genome", "contig", "position_index",
                                                                   "genome_code", "contig_code"])
