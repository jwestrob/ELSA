#!/usr/bin/env python
"""source207 scoring of synton loci on the Spacedust collection, with the reference gate (workstream baseline_synton).

Scorer: model_scaling_20260913/code/evaluate_spacedust_queries.evaluate, unchanged (read-only import): per region the
single query interval with the best gene-level F1; control = 200 rigid translations of the region's membership pattern
inside its contig (per-region seeds from sha256('17:'+region_id)); excess = mean over regions of (F1 - mean control F1).
Per-region vector exactly as engine/recurrence.py and block_fdr.py build it:
    per = (observed.set_index('region_id').f1 - ctrl.groupby('region_id').f1.mean()).to_numpy()
95% interval: 10,000 bootstrap draws over regions; rng = default_rng(17), fresh for each output (the engine replay in
the gate burns three draws first, because its 'gaps 0-2' selection was the 4th output of one generator).

usage
  score_spacedust.py gate  --threads T                     engine reference + native Spacedust; writes gate JSON
  score_spacedust.py score --loci P [--loci P ...] --label L [--label L ...] --threads T
                           [--compare NAME=per_region.json ...]
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
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
BS = RD / "baseline_synton"
ENGINE = RD / "engine/results_v2/hybrid__esm2_650m__genomecentred__mm30cpmr0p05__w8"
NATIVE = ROOT / "native_run_20260910/results/all_blocks.parquet"
NATIVE_JSON = ROOT / "seq_invisible_20261111/results/block_fdr/spacedust_native.json"
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
sys.path.insert(0, str(SCALE / "code"))
sys.path.insert(0, str(SCALE / "production"))
from evaluate_spacedust_queries import evaluate  # noqa: E402
import run_spacedust_scaling as rss  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load():
    regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
    qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
    return regions, qmeta


def query_coverage(spans, qmeta):
    sizes = qmeta.groupby(["sample_id", "contig_id"]).size()
    covered = 0
    for key, b in spans.groupby(["query_genome", "query_contig"], sort=False):
        nn = int(sizes[key])
        d = np.zeros(nn + 1, dtype=np.int64)
        np.add.at(d, b.query_start.to_numpy(dtype=np.int64), 1)
        np.add.at(d, b.query_end.to_numpy(dtype=np.int64) + 1, -1)
        covered += int((np.cumsum(d)[:nn] > 0).sum())
    return covered / int(sizes.sum())


def size_class(n):
    return np.select([n <= 5, n <= 10, n <= 20], ["1-5", "6-10", "11-20"], "21+")


def score(spans, regions, qmeta, threads, rng, label):
    spans = spans[SPAN].drop_duplicates()
    observed, ctrl, report = evaluate(regions, qmeta, spans, threads=threads, controls=200, seed=17)
    per = (observed.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean())
    v = per.to_numpy()
    boot = rng.choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    obs = observed.set_index("region_id")
    cm = ctrl.groupby("region_id").f1.mean()
    ncds = obs.annotated_cds.to_numpy()
    cls = pd.Series(size_class(ncds), index=obs.index)
    by_size = {}
    for c in ("1-5", "6-10", "11-20", "21+"):
        m = (cls == c).to_numpy()
        by_size[c] = dict(regions=int(m.sum()), f1=float(obs.f1[m].mean()), control=float(cm.reindex(obs.index)[m].mean()),
                          median_best_span=float(obs.block_span[m].median()))
    r = dict(selection=label, n_intervals=int(len(spans)), query_gene_coverage=query_coverage(spans, qmeta),
             f1_observed=report["observed"]["mean_best_single_block_f1"],
             f1_random=report["controls"]["mean_best_single_block_f1"], f1_excess=float(v.mean()),
             ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975)),
             localized_observed=report["observed"]["localized_fraction"],
             localized_random=report["controls"]["localized_fraction"],
             median_boundary_precision=report["observed"]["median_boundary_precision_at_best_f1"],
             median_coverage=report["observed"]["median_coverage_at_best_f1"],
             median_union_coverage=report["observed"]["median_union_coverage"],
             regions_with_any_overlap=int((obs.f1 > 0).sum()),
             median_boundary_precision_overlapping=float(obs.boundary_precision[obs.f1 > 0].median()) if (obs.f1 > 0).any() else None,
             median_coverage_overlapping=float(obs.coverage[obs.f1 > 0].median()) if (obs.f1 > 0).any() else None,
             median_best_span_overlapping=float(obs.block_span[obs.f1 > 0].median()) if (obs.f1 > 0).any() else None,
             fully_contained_fraction=report["observed"]["fully_contained_fraction"],
             regions_better_than_random=int((v > 0).sum()), by_size_class=by_size,
             per_region_f1=obs.f1.round(6).to_dict(), per_region_random_f1=cm.round(6).to_dict(),
             per_region_coverage=obs.coverage.round(6).to_dict(),
             per_region_boundary_precision=obs.boundary_precision.round(6).to_dict(),
             per_region_best_span=obs.block_span.astype(int).to_dict())
    log(label, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()
                if not k.startswith("per_region") and k != "by_size_class"})
    return r


def paired(a, b, name_a, name_b):
    """A - B on per-region F1 and excess; 10,000 region bootstrap draws, fresh default_rng(17)."""
    ids = sorted(set(a["per_region_f1"]) & set(b["per_region_f1"]))
    fa = np.array([a["per_region_f1"][i] for i in ids])
    fb = np.array([b["per_region_f1"][i] for i in ids])
    ea = fa - np.array([a["per_region_random_f1"][i] for i in ids])
    eb = fb - np.array([b["per_region_random_f1"][i] for i in ids])
    out = dict(a=name_a, b=name_b, regions=len(ids))
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    for k, d in (("d_f1", fa - fb), ("d_excess", ea - eb)):
        bm = d[idx].mean(axis=1)
        out[k] = [float(d.mean()), float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]
    out["regions_a_better_f1"] = int((fa > fb).sum())
    out["regions_b_better_f1"] = int((fb > fa).sum())
    return out


def cmd_gate(a):
    regions, qmeta = load()
    stored = json.load(open(ENGINE / "summary.json"))
    ref = [s for s in stored["source207"] if s["selection"] == "conserved loci, gaps 0-2"][0]
    loci = pd.read_parquet(ENGINE / "query_loci.parquet")
    rng = np.random.default_rng(17)
    for _ in range(3):                                   # replay: three earlier selections drew (10000 x 207) each
        rng.choice(np.zeros(len(regions)), size=(10000, len(regions)), replace=True)
    eng = score(loci[loci.kind.str.startswith("conserved")], regions, qmeta, a.threads, rng, "engine reference")
    keys = ("n_intervals", "f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi", "localized_observed", "localized_random")
    eq = {k: bool(np.isclose(eng[k], ref[k], rtol=0, atol=1e-12)) for k in keys}
    per_keys = [k for k in ("per_region_f1", "per_region_random_f1") if k in ref]     # v1.1 summaries store F1 only
    per_eq = all(abs(eng[k][i] - ref[k][i]) < 1e-9 for k in per_keys for i in ref[k])
    log("per-region fields compared with the stored summary:", per_keys)
    nat_stored = json.load(open(NATIVE_JSON))["thresholds"][0]
    blocks = pd.read_parquet(NATIVE, columns=SPAN + ["n_anchors"])
    nat = score(blocks[blocks.n_anchors >= 2], regions, qmeta, a.threads, np.random.default_rng(17), "native Spacedust")
    nkeys = dict(f1_observed="f1_observed", f1_random="f1_random", f1_excess="f1_excess")
    neq = {k: bool(np.isclose(nat[k], nat_stored[s], rtol=0, atol=1e-12)) for k, s in nkeys.items()}
    neq["ci"] = bool(np.allclose([nat["ci_lo"], nat["ci_hi"]], nat_stored["f1_excess_ci95"], rtol=0, atol=1e-12))
    neq["n_query_intervals"] = nat["n_intervals"] == nat_stored["n_query_intervals"]
    out = dict(engine=eng, engine_equal_stored=eq, engine_per_region_equal=bool(per_eq), native=nat,
               native_equal_stored=neq, native_stored=dict((k, nat_stored[k]) for k in
                                                          ("f1_observed", "f1_random", "f1_excess", "f1_excess_ci95",
                                                           "n_query_intervals", "query_gene_coverage")),
               reproduced=bool(all(eq.values()) and per_eq and all(neq.values())),
               written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log("engine equal", eq, "per-region", per_eq, "native equal", neq, "REPRODUCED", out["reproduced"])
    (BS / "results").mkdir(exist_ok=True)
    json.dump(out, open(BS / "results/spacedust_gate.json", "w"), indent=1)


def cmd_score(a):
    gate = json.load(open(BS / "results/spacedust_gate.json"))
    assert gate["reproduced"], "reference gate not passed"
    regions, qmeta = load()
    compares = {"engine reference": gate["engine"], "native Spacedust": gate["native"]}
    for c in a.compare or []:
        name, path = c.split("=", 1)
        if path.endswith(".parquet"):           # transfer_spacedust per-region frame (region_id, f1, control_f1, ...)
            d = pd.read_parquet(path).set_index("region_id")
            compares[name] = dict(per_region_f1=d.f1.to_dict(), per_region_random_f1=d.control_f1.to_dict())
        else:
            compares[name] = json.load(open(path))
    res = {}
    for path, label in zip(a.loci, a.label):
        loci = pd.read_parquet(path)
        q = loci[loci.side == "query"].rename(columns=dict(genome="query_genome", contig="query_contig",
                                                           start="query_start", end="query_end"))
        r = score(q, regions, qmeta, a.threads, np.random.default_rng(17), label)
        r["paired"] = [paired(r, b, label, name) for name, b in compares.items()]
        for p in r["paired"]:
            log(label, "-", p["b"], "dF1", [round(x, 4) for x in p["d_f1"]], "dExcess", [round(x, 4) for x in p["d_excess"]])
        res[label] = r
        compares[label] = r
    # pairs among the scored outputs (e.g. matched variant - faithful)
    labs = list(res)
    res["_between"] = [paired(res[x], res[y], x, y) for i, x in enumerate(labs) for y in labs[:i]]
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = BS / "results" / f"spacedust_{a.tag}.json"
    json.dump(res, open(out, "w"), indent=1)
    log("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "score"])
    ap.add_argument("--loci", action="append")
    ap.add_argument("--label", action="append")
    ap.add_argument("--compare", action="append")
    ap.add_argument("--tag", default="scores")
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    {"gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
