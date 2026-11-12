#!/usr/bin/env python
"""Sensitivity of the module-based rules (C, G, H) to the Louvain seed (17-21 at the tuned resolution 0.3) and to
the resolution near the tuned value (0.2, 0.25, 0.35, 0.5, seed 17); selection is already fixed, this is reported
as robustness only. Also a mechanism diagnostic: per ribosomal run, the number of distinct pair-module labels
(res 0.3) on its internal boundaries and whether the boundaries just outside the run carry another label.
Writes results/robust_rescue3.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blib as B  # noqa: E402
import rules as RU  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    L = B.Landscape(B.ARMS["rescue3"])
    bld = RU.Builder(L)
    gcode = B.odd_even_code()
    runs = B.ribo_runs()
    r1 = pd.read_parquet(B.BD / "scores/rescue3/R1_conserved2.parquet").set_index("instance_id")
    out = dict(configs=[])
    cfgs = []
    for seed in (17, 18, 19, 20, 21):
        cfgs += [("C", dict(res=0.3, seed=seed)), ("G", dict(res=0.3, h=1, gain=0, seed=seed)),
                 ("H", dict(res=0.3, f=0.5, seed=seed))]
    for res in (0.2, 0.25, 0.35, 0.5):
        cfgs += [("C", dict(res=res)), ("G", dict(res=res, h=1, gain=0)), ("H", dict(res=res, f=0.5))]
    for fam, p in cfgs:
        s, e = bld.build(fam, p)
        df = L.table(s, e)
        per, _ = B.score_table(B.PKG, df, a.threads)
        per = per.set_index("instance_id")
        pr = RU.paired(per, r1, gcode, "even")
        rib = B.ribo_summary(runs, B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy()))
        cond = RU.test_conditions(rib, pr)
        row = dict(family=fam, params=p, conditions=cond, summed=pr["summed"],
                   d_f1={c: pr[c]["d_f1"][0] for c in RU.CALLERS}, ribo=rib)
        out["configs"].append(row)
        B.log(fam, p, cond, [round(x, 3) for x in pr["summed"]], {c[:4]: round(v, 3) for c, v in row["d_f1"].items()},
              rib["resolved"], rib["recovered"], rib["parvarchaeaceae"], rib["parvarchaeaceae_resolved"], f"{time.time()-t0:.0f}s")
    # mechanism diagnostic at res 0.3 seed 17 and at res 10 (engine modules)
    diag = {}
    for res in (0.3, 1, 3, 10):
        _, pm = bld.modules(res)
        lab = L.boundary_labels(pm)
        n_lab, out_diff = [], []
        for r in runs.itertuples():
            ks = np.arange(r.gid_start, r.gid_end)
            v = lab[ks]
            v = v[v >= 0]
            u, c = np.unique(v, return_counts=True)
            n_lab.append(len(u))
            top = u[c.argmax()] if len(u) else -9
            flank = [lab[k] for k in (r.gid_start - 1, r.gid_end) if 0 <= k < L.u.n and L.b[k] >= 2]
            out_diff.append(np.mean([f != top for f in flank]) if flank else np.nan)
        n_lab = np.array(n_lab)
        diag[str(res)] = dict(runs_one_label=int((n_lab == 1).sum()), runs_two_labels=int((n_lab == 2).sum()),
                              median_labels=float(np.median(n_lab)), flank_label_differs=float(np.nanmean(out_diff)))
        B.log("diag res", res, diag[str(res)])
    out["ribosomal_label_diagnostic"] = diag
    with open(B.BD / "results/robust_rescue3.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    B.log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
