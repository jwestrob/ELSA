#!/usr/bin/env python
"""lineagefree: readings R1 and DH on a recurrence_lf result directory, scored against the genus-based token engine.

usage
  readings_lf.py score --rdir RESULT_DIR --tag TAG --threads T [--tokens TOK.parquet] [--refcheck]
      R1 = conserved@2; DH = level-2 locus + rule-H core (Louvain res 0.3 seed 17 on J x log2 max(L, 2) of the run's
      own significant pairs, end trimming f 0.5), built with the boundary workstream's code (read-only import of
      $RD/boundary/code/{blib,rules}.py). Scored on the full DPANN package (200 controls, seed 17); summaries on odd and
      even genomes; paired genome bootstrap (2,000, seed 17) against the genus-based R1 / DH scores
      ($RD/boundary/scores/rescue3); ribosomal check on all 1,686 runs and the paired resolved difference.
      --refcheck: also assert that the loci equal the stored genus-based loci and the per-instance scores equal the
      stored scores (for the genus map run).
      Writes $RD/lineagefree/results/tok_<TAG>.json and $RD/lineagefree/scores/tok_<TAG>__{R1,DH}.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as BR  # noqa: E402

LF = RD / "lineagefree"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
REF = {"R1": "R1_conserved2", "DH": "DH_coreH_res0.3_f0.5"}
CALLERS = BR.CALLERS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


def ribo_counts(rib):
    rec = int(rib["recovered"])
    res = int(rib["resolved"])
    parv_rec = int(str(rib["parvarchaeaceae"]).split("/")[0])
    parv_res = int(str(rib.get("parvarchaeaceae_resolved", "0")).split("/")[0])
    return rec, res, parv_rec, parv_res


def conditions(pr, rib, rib_ref):
    """Predeclared replacement conditions (predeclared.json) for one reading vs the genus-based reading."""
    rec, res, parv_rec, _ = ribo_counts(rib)
    _, res_ref, _, _ = ribo_counts(rib_ref)
    c_sum = abs(pr["summed"][0]) <= 0.01
    c_f1 = all(pr[c]["d_f1"][0] >= -0.01 for c in CALLERS)
    c_rib = rec >= 1650 and parv_rec >= 110 and abs(res - res_ref) <= 0.03 * res_ref
    c_sum_loss = pr["summed"][0] >= -0.01
    return dict(summed_within_0p01=bool(c_sum), summed_not_below_m0p01=bool(c_sum_loss), no_F1_loss_over_0p01=bool(c_f1),
                ribosomal=bool(c_rib), replaces=bool(c_sum_loss and c_f1 and c_rib))


def conditions_odd(pr, rib, rib_ref):
    """The same conditions on odd genomes and the 848 odd ribosomal runs (bars scaled: 830 recovered, 54 Parv)."""
    rec, res, parv_rec, _ = ribo_counts(rib)
    _, res_ref, _, _ = ribo_counts(rib_ref)
    c_sum_loss = pr["summed"][0] >= -0.01
    c_f1 = all(pr[c]["d_f1"][0] >= -0.01 for c in CALLERS)
    c_rib = rec >= 830 and parv_rec >= 54 and abs(res - res_ref) <= 0.03 * res_ref
    return dict(summed_not_below_m0p01=bool(c_sum_loss), no_F1_loss_over_0p01=bool(c_f1), ribosomal=bool(c_rib),
                n_met=int(c_sum_loss) + int(c_f1) + int(c_rib))


def cmd_score(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    L = B.Landscape(a.rdir, tokens=a.tokens)
    bld = BR.Builder(L)
    gcode = B.odd_even_code()
    runs = B.ribo_runs()
    even_runs = (runs.genome_code % 2 == 0).to_numpy()
    (LF / "scores").mkdir(parents=True, exist_ok=True)
    (LF / "results").mkdir(parents=True, exist_ok=True)
    (LF / "loci").mkdir(parents=True, exist_ok=True)
    summ_engine = json.load(open(Path(a.rdir) / "summary.json"))
    out = dict(tag=a.tag, rdir=str(a.rdir), engine={k: summ_engine.get(k) for k in (
        "lineage_source", "lineages", "effective_genomes", "rule", "fdr", "significant_pairs", "genes_covered",
        "genes_covered_frac", "token_pairs_L2", "L_quantiles", "tokens_in_significant_pairs", "modules", "seconds")},
        readings={})
    rules_tsv = pd.read_csv(Path(a.rdir) / "rules.tsv", sep="\t")
    out["engine"]["rules_at_fdr_0.01"] = rules_tsv[rules_tsv.fdr <= 0.01].sort_values("observed", ascending=False).head(12)[
        ["rule", "observed", "null_mean", "fdr"] + (["genes_covered"] if "genes_covered" in rules_tsv else [])].to_dict("records")
    builds = {"R1": ("R1", {}), "DH": ("D", dict(core="H", res=0.3, f=0.5))}
    for name, (fam, p) in builds.items():
        t1 = time.time()
        s, e = bld.build(fam, p)
        df = L.table(s, e)
        df.to_parquet(LF / "loci" / f"tok_{a.tag}__{name}.parquet", index=False)
        per, _ = B.score_table(B.PKG, df, a.threads)
        per.to_parquet(LF / "scores" / f"tok_{a.tag}__{name}.parquet", index=False)
        pi = per.set_index("instance_id")
        g = pi.genome.map(gcode).to_numpy()
        summ = {sp: B.summarize_per(pi[g % 2 == (1 if sp == "odd" else 0)].reset_index()) for sp in ("odd", "even")}
        sc = B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
        rib = B.ribo_summary(runs, sc)
        rib_even = B.ribo_summary(runs[even_runs].reset_index(drop=True), sc[even_runs].reset_index(drop=True))
        rib_odd = B.ribo_summary(runs[~even_runs].reset_index(drop=True), sc[~even_runs].reset_index(drop=True))
        ref = pd.read_parquet(RD / "boundary/scores/rescue3" / f"{REF[name]}.parquet")
        ref["excess"] = ref.f1 - ref.control_mean_f1
        ref = ref.set_index("instance_id")
        ref_loci = pd.read_parquet(RD / "boundary/loci/rescue3" / f"{REF[name]}.parquet")
        sc_ref = B.ribo_score(runs, ref_loci.gid_s.to_numpy(), ref_loci.gid_e.to_numpy())
        rib_ref = B.ribo_summary(runs, sc_ref)
        rib_ref_odd = B.ribo_summary(runs[~even_runs].reset_index(drop=True), sc_ref[~even_runs].reset_index(drop=True))
        item = dict(summary=summ, ribo=rib, ribo_even=rib_even, ribo_odd=rib_odd, ribo_genus=rib_ref, ribo_genus_odd=rib_ref_odd,
                    ribo_resolved_vs_genus=BR.ribo_paired(sc, sc_ref),
                    loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), L.u.n),
                    vs_genus_even=BR.paired(pi, ref, gcode, "even"), vs_genus_odd=BR.paired(pi, ref, gcode, "odd"))
        item["conditions_vs_genus_even"] = conditions(item["vs_genus_even"], rib, rib_ref)
        item["conditions_vs_genus_odd"] = conditions_odd(item["vs_genus_odd"], rib_odd, rib_ref_odd)
        if a.refcheck:
            item["refcheck"] = dict(
                loci_equal=bool(len(df) == len(ref_loci) and (df[["gid_s", "gid_e", "module_id"]].to_numpy() ==
                                                              ref_loci[["gid_s", "gid_e", "module_id"]].to_numpy()).all()),
                f1_equal=bool(np.array_equal(pi.loc[ref.index].f1.to_numpy(), ref.f1.to_numpy())),
                control_equal=bool(np.allclose(pi.loc[ref.index].control_mean_f1.to_numpy(), ref.control_mean_f1.to_numpy(),
                                               rtol=0, atol=1e-12)))
        item["seconds"] = round(time.time() - t1, 1)
        out["readings"][name] = item
        od = summ["odd"]
        # the log shows odd-genome numbers only (selection is made on odd genomes before even numbers are read)
        log(a.tag, name, "ODD summed", round(od["summed_DF_TXSS_excess"], 4), {c[:4]: round(od[c]["f1"], 3) for c in CALLERS},
            "d vs genus", [round(x, 4) for x in item["vs_genus_odd"]["summed"]],
            {c[:4]: round(item["vs_genus_odd"][c]["d_f1"][0], 4) for c in CALLERS},
            "ribo odd", rib_odd["recovered"], rib_odd["resolved"], rib_odd["parvarchaeaceae"],
            "cond odd", item["conditions_vs_genus_odd"], item.get("refcheck", ""), f"{item['seconds']}s")
    out["seconds"] = round(time.time() - t0, 1)
    jdump(LF / "results" / f"tok_{a.tag}.json", out)
    odd = dict(tag=a.tag, engine=out["engine"], readings={
        k: dict(summary_odd=v["summary"]["odd"], ribo_odd=v["ribo_odd"], ribo_genus_odd=v["ribo_genus_odd"],
                vs_genus_odd=v["vs_genus_odd"], conditions_vs_genus_odd=v["conditions_vs_genus_odd"], loci=v["loci"])
        for k, v in out["readings"].items()})
    jdump(LF / "results" / f"tok_{a.tag}.odd.json", odd)
    log("wrote", LF / "results" / f"tok_{a.tag}.json", "and .odd.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score",))
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", default=str(TOK))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--refcheck", action="store_true")
    a = ap.parse_args()
    {"score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
