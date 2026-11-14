#!/usr/bin/env python
"""Reading inventory (coordinator follow-up 2; predeclared.json followup_2), descriptive.

For an arm: each of the 11 readings of loci_all_genomes scored ALONE (conserved and conserved_g2 at levels 2, 5, 12,
30, 80 recomputed in full from boundary_support.npy with the engine's run rule, because the table keeps a locus only
once per kind; unit@0.5 from the table) and the full table, on the package gold (min 2 members, 200 controls,
seed 17). Per caller on even genomes: F1, control F1, excess; ribosomal resolved (all 1,686 runs) per reading.
Best-locus attribution: the full-table F1 of an instance equals the maximum over readings of the reading's own best
F1 (checked); per caller, share of instances where a reading attains that maximum (attains), where it is the only
one (sole), and where it is the first attaining reading in table order (assigned; sums to 1 with 'none' = F1 0).
Writes results/inventory_<arm>.json and results/inventory_<arm>.tsv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blib as B  # noqa: E402

CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
READINGS = [("conserved", lv) for lv in (2, 5, 12, 30, 80)] + [("conserved_g2", lv) for lv in (2, 5, 12, 30, 80)] + [("unit", 0.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    L = B.Landscape(B.ARMS[a.arm])
    gcode = B.odd_even_code()
    runs = B.ribo_runs()
    full = pd.read_parquet(L.rdir / "loci_all_genomes.parquet")
    contigs = pd.read_parquet(B.PKG / "universe/contigs.parquet", columns=["genome", "contig", "first_gid"])
    full = full.merge(contigs, on=["genome", "contig"], how="left")
    full["gid_s"] = full.first_gid.astype(np.int64) + full.start
    full["gid_e"] = full.first_gid.astype(np.int64) + full.end
    per_reading, check = {}, {}
    for kind, lv in READINGS:
        name = f"{kind}@{lv:g}"
        if kind == "unit":
            df = full[(full.kind == "unit") & (full.level == lv)].copy()
        else:
            s, e = B.R.runs_at_least(L.b, L.cs, L.ce, float(lv), 0 if kind == "conserved" else 2)
            df = L.table(s, e)
            tab = full[(full.kind == kind) & (full.level == lv)]
            rec = set(zip(df.gid_s.to_numpy().tolist(), df.gid_e.to_numpy().tolist()))
            tb = set(zip(tab.gid_s.to_numpy().tolist(), tab.gid_e.to_numpy().tolist()))
            check[name] = dict(recomputed=len(rec), table_rows=len(tb), table_subset_of_recomputed=tb <= rec)
        per, _ = B.score_table(B.PKG, df, a.threads)
        rib = B.ribo_summary(runs, B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy()))
        per_reading[name] = (per.set_index("instance_id"), rib, B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), L.u.n))
        B.log(a.arm, name, check.get(name), "ribo", rib["resolved"], f"{time.time()-t0:.0f}s")
    perF, _ = B.score_table(B.PKG, full, a.threads)
    perF = perF.set_index("instance_id")
    ribF = B.ribo_summary(runs, B.ribo_score(runs, full.gid_s.to_numpy(), full.gid_e.to_numpy()))
    names = [f"{k}@{lv:g}" for k, lv in READINGS]
    ids = perF.index
    M = np.column_stack([per_reading[n][0].loc[ids].f1.to_numpy() for n in names])
    mx = M.max(1)
    check["max_over_readings_equals_full_F1"] = bool(np.allclose(mx, perF.f1.to_numpy(), atol=1e-12))
    check["max_abs_diff"] = float(np.abs(mx - perF.f1.to_numpy()).max())
    att = np.abs(M - perF.f1.to_numpy()[:, None]) <= 1e-12
    att &= (perF.f1.to_numpy() > 0)[:, None]
    first = np.where(att.any(1), att.argmax(1), -1)
    even = (perF.genome.map(gcode).to_numpy() % 2 == 0)
    caller = perF.caller.to_numpy()
    rows = []
    for n_i, n in enumerate(names + ["full_table"]):
        per = per_reading[n][0].loc[ids] if n != "full_table" else perF
        row = dict(arm=a.arm, reading=n)
        for c in CALLERS:
            k = even & (caller == c)
            row[f"{c}_n"] = int(k.sum())
            row[f"{c}_F1"] = float(per.f1.to_numpy()[k].mean())
            row[f"{c}_control"] = float(per.control_mean_f1.to_numpy()[k].mean())
            row[f"{c}_excess"] = float((per.f1 - per.control_mean_f1).to_numpy()[k].mean())
            if n != "full_table":
                row[f"{c}_attains"] = float(att[k, n_i].mean())
                row[f"{c}_sole"] = float((att[k, n_i] & (att[k].sum(1) == 1)).mean())
                row[f"{c}_assigned"] = float((first[k] == n_i).mean())
            else:
                row[f"{c}_attains"] = row[f"{c}_sole"] = None
                row[f"{c}_assigned"] = float((first[k] == -1).mean())  # share with F1 0 (no locus) in the full-table row
        row["summed_DF_TXSS_excess"] = row["DefenseFinder_excess"] + row["TXSScan_excess"]
        rib = per_reading[n][1] if n != "full_table" else ribF
        loc = per_reading[n][2] if n != "full_table" else B.locus_stats(full.gid_s.to_numpy(), full.gid_e.to_numpy(), L.u.n)
        row.update(ribo_resolved=rib["resolved"], ribo_recovered=rib["recovered"], loci=loc["loci"],
                   genes_in_loci=loc["genes_covered"], median_span=loc["median_span"])
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(B.BD / f"results/inventory_{a.arm}.tsv", sep="\t", index=False)
    with open(B.BD / f"results/inventory_{a.arm}.json", "w") as fh:
        json.dump(dict(arm=a.arm, checks=check, rows=rows), fh, indent=1,
                  default=lambda x: x.item() if hasattr(x, "item") else str(x))
    B.log("checks", check)
    B.log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
