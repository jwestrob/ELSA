#!/usr/bin/env python
"""Table over all arms checked under $RD/ribosomal/arms/, with paired bootstrap intervals of
resolved-count differences against the base arm (g2_m16 runs; one run per genome, so resampling runs
resamples genomes). Writes $RD/ribosomal/arms_table.tsv and arms_table.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
A = RD / "ribosomal/arms"
BASE = "results_dpann_v5__dpann__mmseqs__seq__e3c50s75__w3"
SETS = ["conserved@2", "conserved_g2@2", "unit@0.5", "ladder_conserved@12", "ladder_conserved@80",
        "ladder_conserved_g2@80"]
B = 2000


def main():
    rng = np.random.default_rng(17)
    base_pr = pd.read_parquet(A / BASE / "ribosomal_runs.parquet").sort_values("run_id")
    n = len(base_pr)
    boot = rng.integers(0, n, size=(B, n))
    rows = []
    for d in sorted(p for p in A.iterdir() if p.is_dir() and (p / "ribosomal_check.json").exists()):
        j = json.load(open(d / "ribosomal_check.json"))
        r = j["results"]["g2_m16"]
        c = j["controls"]["g2_m16"]
        pr = pd.read_parquet(d / "ribosomal_runs.parquet").sort_values("run_id")
        assert np.array_equal(pr.run_id.to_numpy(), base_pr.run_id.to_numpy())
        row = dict(arm_dir=d.name, w=j["w"], rule=j["rule"], tokens=j["tokens"], sig_pairs=j["significant_pairs"],
                   genes_covered=round(j["genes_covered_frac"], 3) if j["genes_covered_frac"] else None,
                   modules=j["modules"])
        for s in SETS:
            x = r.get(s)
            if x is None:
                continue
            tag = s.replace("ladder_", "")
            row[f"{tag}:recovered"] = x["recovered"]
            row[f"{tag}:resolved"] = x["resolved"]
            row[f"{tag}:med_prec"] = round(x["median_precision_recovered"], 3) if x["median_precision_recovered"] else None
            row[f"{tag}:mean_frag"] = round(x["mean_fragments"], 3)
            row[f"{tag}:purity"] = round(x["module_purity"], 3) if x["module_purity"] is not None else None
            row[f"{tag}:parv"] = x["parvarchaeaceae"]
            row[f"{tag}:parv_resolved"] = x["parvarchaeaceae_resolved"]
            row[f"{tag}:med_ext"] = f"{x['median_ext_left_recovered']:g}/{x['median_ext_right_recovered']:g}" \
                if x["median_ext_left_recovered"] is not None else None
            if s in c:
                row[f"{tag}:ctrl_rec"] = round(c[s]["random"]["recovered_frac"], 3)
                row[f"{tag}:ctrl_res"] = round(c[s]["random"]["resolved_frac"], 3)
            col = f"{s}:resolved"
            if col in pr and col in base_pr:
                diff = pr[col].to_numpy().astype(int) - base_pr[col].to_numpy().astype(int)
                bs = diff[boot].sum(axis=1)
                row[f"{tag}:d_resolved_vs_base"] = int(diff.sum())
                row[f"{tag}:d_ci"] = f"[{np.quantile(bs, .025):.0f}, {np.quantile(bs, .975):.0f}]"
        row["oracle_resolved"] = r.get("any_ladder_level_oracle", {}).get("resolved")
        row["mean_coverage@2"] = round(r["conserved@2"]["mean_coverage"], 4)
        row["no_locus@2"] = r["conserved@2"]["no_locus"]
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RD / "ribosomal/arms_table.tsv", sep="\t", index=False)
    t.to_json(RD / "ribosomal/arms_table.json", orient="records", indent=1)
    show = ["arm_dir", "w", "tokens", "conserved@2:recovered", "conserved@2:resolved", "conserved@2:d_ci",
            "conserved@2:med_prec", "conserved@2:mean_frag", "conserved@2:purity", "conserved@2:parv",
            "conserved@80:resolved", "conserved@80:d_ci", "unit@0.5:resolved", "oracle_resolved"]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 200):
        print(t[[c for c in show if c in t]].to_string(index=False))


if __name__ == "__main__":
    main()
