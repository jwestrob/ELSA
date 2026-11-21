#!/usr/bin/env python
"""Reference reproduction and reference per-region tables (workstream transfer_spacedust).

rescore   (i) stored engine v1.1 hybrid w8 query loci rescored with the engine's own selection sequence (8 score calls,
          one rng seeded 17): every summary number and per-region F1 must equal the stored summary.json;
          (ii) native Spacedust, all blocks, as block_fdr.py (rng seeded 17, first threshold): must equal the stored JSON;
          (iii) engine single reading (stored loci, kind 'conserved' at level 2).
          Writes results/refcheck.json and per-region tables results/pr_<name>.parquet.
compare   compare a rerun engine directory's summary.json with the stored one (rule, pairs, source207 numbers).
engine    per-region table for an engine result directory (selection 'conserved loci, gaps 0-2').
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s207  # noqa: E402

RD, TS, ROOT = s207.RD, s207.TS, s207.ROOT
STORED = RD / "engine/results_v2/hybrid__esm2_650m__genomecentred__mm30cpmr0p05__w8"
NATIVE = ROOT / "native_run_20260910/results/all_blocks.parquet"
NATIVE_JSON = ROOT / "seq_invisible_20261111/results/block_fdr/spacedust_native.json"
log = s207.log
SEL = ["conserved loci, all levels", "conserved loci, gap 1", "conserved loci, gap 2", "conserved loci, gaps 0-2",
       "islands", "profile units, all tau", "conserved (gaps 0-2) + profile units", "all locus kinds"]


def engine_selections(loci):
    k = loci.kind
    return [loci[k == "conserved"], loci[k == "conserved_g1"], loci[k == "conserved_g2"], loci[k.str.startswith("conserved")],
            loci[k == "island"], loci[k == "unit"], loci[k.str.startswith("conserved") | (k == "unit")], loci]


def cmd_rescore(a):
    t0 = time.time()
    B = s207.Bench(a.threads)
    out = {}
    # (i) engine stored
    st = json.load(open(STORED / "summary.json"))
    loci = pd.read_parquet(STORED / "query_loci.parquet")
    rng = np.random.default_rng(17)
    checks = []
    for name, df in zip(SEL, engine_selections(loci)):
        r, pr = B.score(df, name, rng)
        s = [x for x in st["source207"] if x["selection"] == name][0]
        eq = {k: (r[k] == s[k]) for k in ("n_intervals", "f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi",
                                         "localized_observed", "localized_random")}
        per_ok = all(abs(pr.set_index("region_id").f1[k] - v) <= 5e-7 for k, v in s["per_region_f1"].items())
        checks.append(dict(selection=name, equal=eq, per_region_f1_equal=per_ok, all_equal=all(eq.values()) and per_ok,
                           new={k: r[k] for k in eq}, stored={k: s[k] for k in eq}))
        log("engine", name, "equal" if checks[-1]["all_equal"] else "DIFFERS", {k: r[k] for k in eq})
        if name == "conserved loci, gaps 0-2":
            pr.to_parquet(TS / "results/pr_engine_ref.parquet", index=False)
            out["engine_reference"] = {k: v for k, v in r.items()}
    out["engine_checks"] = checks
    # (iii) engine single reading: conserved, level 2
    r, pr = B.score(loci[(loci.kind == "conserved") & (loci.level == 2)], "engine conserved level 2 (single reading)")
    pr.to_parquet(TS / "results/pr_engine_single.parquet", index=False)
    out["engine_single_reading"] = r
    log("engine single reading", {k: r[k] for k in ("n_intervals", "f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi")})
    # (ii) native Spacedust, all blocks (n_anchors >= 2 = every block)
    blocks = pd.read_parquet(NATIVE, columns=s207.SPAN + ["n_anchors"])
    sj = json.load(open(NATIVE_JSON))["thresholds"][0]
    r, pr = B.score(blocks[blocks.n_anchors >= 2], "native Spacedust, all blocks", np.random.default_rng(17))
    eq = dict(n_query_intervals=r["n_intervals"] == sj["n_query_intervals"], f1_observed=r["f1_observed"] == sj["f1_observed"],
              f1_random=r["f1_random"] == sj["f1_random"], f1_excess=r["f1_excess"] == sj["f1_excess"],
              ci=[r["ci_lo"], r["ci_hi"]] == sj["f1_excess_ci95"],
              median_boundary_precision=r["median_boundary_precision"] == sj["median_boundary_precision"],
              query_gene_coverage=abs(r["query_gene_coverage"] - sj["query_gene_coverage"]) < 1e-12)
    out["spacedust_check"] = dict(equal=eq, all_equal=all(eq.values()), new=r, stored=sj)
    out["spacedust"] = r
    pr.to_parquet(TS / "results/pr_spacedust.parquet", index=False)
    log("spacedust", "equal" if all(eq.values()) else "DIFFERS", eq, {k: r[k] for k in ("f1_observed", "f1_random", "f1_excess")})
    out["reproduced"] = all(c["all_equal"] for c in checks) and out["spacedust_check"]["all_equal"]
    out["seconds"] = round(time.time() - t0, 1)
    (TS / "results/refcheck.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    log("refcheck reproduced:", out["reproduced"])


def cmd_compare(a):
    new = json.load(open(Path(a.rdir) / "summary.json"))
    st = json.load(open(STORED / "summary.json"))
    keys = ("rule", "significant_pairs", "fdr", "genes_covered", "tokens", "instances", "token_pairs")
    eq = {k: new.get(k) == st.get(k) for k in keys}
    src = {}
    for s in st["source207"]:
        n = [x for x in new["source207"] if x["selection"] == s["selection"]][0]
        src[s["selection"]] = all(n[k] == s[k] for k in ("n_intervals", "f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi")) \
            and n["per_region_f1"] == s["per_region_f1"]
    res = dict(rdir=a.rdir, equal=eq, source207_equal=src, all_equal=all(eq.values()) and all(src.values()),
               new={k: new.get(k) for k in keys + ("seconds", "engine")}, stored={k: st.get(k) for k in keys})
    (TS / "results/refcheck_rerun.json").write_text(json.dumps(res, indent=1) + "\n")
    log("engine rerun equal to stored:", res["all_equal"], eq, src)


def cmd_engine(a):
    B = s207.Bench(a.threads)
    loci = pd.read_parquet(Path(a.rdir) / "query_loci.parquet")
    r, pr = B.score(loci[loci.kind.str.startswith("conserved")], a.name)
    pr.to_parquet(TS / f"results/pr_{a.name}.parquet", index=False)
    r2, pr2 = B.score(loci[(loci.kind == "conserved") & (loci.level == 2)], a.name + "_single")
    pr2.to_parquet(TS / f"results/pr_{a.name}_single.parquet", index=False)
    (TS / f"results/ref_{a.name}.json").write_text(json.dumps(dict(union=r, single=r2), indent=1, default=float) + "\n")
    log(a.name, {k: r[k] for k in ("f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rescore", "compare", "engine"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rdir", default="")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    {"rescore": cmd_rescore, "compare": cmd_compare, "engine": cmd_engine}[a.cmd](a)


if __name__ == "__main__":
    main()
