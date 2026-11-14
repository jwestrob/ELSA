#!/usr/bin/env python
"""source207 scoring of long_loci outputs (transfer_spacedust's s207.py copied; its per-region frames read-only).

usage (Slurm): sscore.py --tag sd_group --names L1,L2_t0.05,... --threads T [--label DEVELOPMENT]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s207  # noqa: E402

RD = s207.RD
TS = s207.TS
LL = RD / "long_loci"
log = s207.log
REFS = {"P2_Vcoh0.1": "pr_P2__V-coh0.1", "P2_Vmob": "pr_P2__V-mob", "CC1": "pr_P2__CC1", "CC2": "pr_P2__CC2",
        "engine_ref": "pr_engine_ref", "engine_r3_w3_single": "pr_engine_r3_w3_single", "spacedust": "pr_spacedust"}


def frag(pr):
    big = pr["size"] >= 11
    bins = {"1-5": (1, 5), "6-10": (6, 10), "11-20": (11, 20), "21+": (21, 10 ** 6)}
    by = {k: dict(union_q50=float(pr.union_coverage[(pr["size"] >= lo) & (pr["size"] <= hi)].median()),
                  best_q50=float(pr.coverage[(pr["size"] >= lo) & (pr["size"] <= hi)].median()))
          for k, (lo, hi) in bins.items()}
    return dict(by_size=by, union_cov_q50=float(pr.union_coverage.median()), union_cov_q50_ge11=float(pr.union_coverage[big].median()),
                best_cov_q50=float(pr.coverage.median()), best_cov_q50_ge11=float(pr.coverage[big].median()),
                best_span_over_size_q50=float(np.median(pr.block_span.to_numpy() / pr["size"].to_numpy())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--extra-refs", dest="extra_refs", default="")
    ap.add_argument("--out-suffix", dest="out_suffix", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    g = s207.load_universe_genes()
    n = len(g)
    qmask = (g.side == "query").to_numpy()
    B = s207.Bench(a.threads)
    refs = {k: pd.read_parquet(TS / "results" / f"{v}.parquet") for k, v in REFS.items()}
    for c in [x for x in a.extra_refs.split(",") if x]:        # label=tag/pr_name (this workstream's per-region frames)
        lab, path = c.split("=")
        refs[lab] = pd.read_parquet(LL / "scores" / f"{path}.parquet")
    res = dict(tag=a.tag, label=a.label, references={k: frag(v) | dict(f1=float(v.f1.mean()), excess=float(v.excess.mean()))
                                                       for k, v in refs.items()}, outputs={})
    od = LL / "scores" / a.tag
    od.mkdir(parents=True, exist_ok=True)
    for nm in [x for x in a.names.split(",") if x]:
        df = pd.read_parquet(LL / "loci" / a.tag / f"{nm}.parquet")
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        spans = s207.gene_spans(g, s, e)
        r, pr = B.score(spans, f"{a.tag} {nm}", np.random.default_rng(17))
        pr.to_parquet(od / f"pr_{nm}.parquet", index=False)
        cov = np.zeros(n + 1, np.int32)
        np.add.at(cov, s, 1)
        np.add.at(cov, e + 1, -1)
        inl = np.cumsum(cov)[:n] > 0
        item = dict(score=r, fragmentation=frag(pr), genes_in_loci_frac=float(inl.mean()),
                    query_genes_in_loci_frac=float(inl[qmask].mean()), loci=int(len(df)),
                    query_span_q50_q90_q99_max=np.quantile((e - s + 1)[qmask[s]], [.5, .9, .99, 1]).tolist())
        for k, v in refs.items():
            item[f"vs_{k}"] = dict(paired=s207.paired(pr, v), verdict=s207.verdict(s207.paired(pr, v)))
        res["outputs"][nm] = item
        log(a.tag, nm, a.label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                 if k in ("n_intervals", "f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi",
                                          "median_boundary_precision", "median_coverage", "query_gene_coverage")},
            "frag", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in item["fragmentation"].items()},
            {k: [round(x, 4) for x in item[f"vs_{k}"]["paired"]["excess"]] for k in refs})
    res["seconds"] = round(time.time() - t0, 1)
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fn = LL / "results" / f"s207_{a.tag}{a.out_suffix}.json"
    fn.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
