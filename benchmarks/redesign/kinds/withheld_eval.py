#!/usr/bin/env python
"""Withheld-family test: AUROC of the selected kappa per curated system type and pooled, at levels
full / L1 (query's own MMseqs2 families withheld) / L2 (all families of the type withheld) / L2ctl (matched random
families withheld). Genome bootstrap 2,000 draws, seed 17 (pooled and per type)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_eval import Boot, log  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
W = RD / "kinds/withheld"
DEPTHS = [30, 100, 300, 1000]
STATS = ["A", "G", "F", "FX", "M", "X", "FXo", "Xo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    sel = json.load(open(RD / "kinds/bench/summary.json"))["selected"]
    s_name, D, form = sel.split("_")
    v, s = DEPTHS.index(int(D[1:])), STATS.index(s_name)
    d = np.load(W / "design.npz")
    qt, pos, qrow = d["qtype"], d["is_pos"], d["qrow"]
    gc = np.load(RD / "kinds/prep/gc.npy")[d["qa"]]
    types = pd.read_csv(W / "types.tsv", sep="\t")
    K = {}
    for lvl in ("full", "L1", "L2", "L2ctl"):
        O = np.load(W / f"pass_{lvl}_obs.npz")["out"][:, v, s].astype(np.float64)
        R0 = np.mean([np.load(W / f"pass_{lvl}_null{r}.npz")["out"][:, v, s] for r in range(3)], axis=0)
        K[lvl] = np.log((O + 1) / (R0 + 1)) if form == "ratio" else O - R0
        K[lvl + "_R"] = O
    rows = []
    for t, ty in enumerate(types.type):
        m = qt == t
        r = dict(type=ty, n_pos=int((m & pos).sum()))
        for lvl in ("full", "L1", "L2", "L2ctl"):
            x = K[lvl][m]
            y = pos[m]
            r[f"auc_{lvl}"] = Boot(gc[m], draws=0).auc(x, y, draws=False)[0]
            q99 = np.quantile(x[~y], 0.99)
            r[f"frac_pos_above_negq99_{lvl}"] = float((x[y] > q99).mean())
            r[f"pos_R_median_{lvl}"] = float(np.median(K[lvl + "_R"][m][y]))
        rows.append(r)
    tt = types.merge(pd.DataFrame(rows), on=["type", "n_pos"])
    tt.to_csv(W / "withheld_by_type.tsv", sep="\t", index=False)
    log(tt.round(3).to_string(index=False))
    # pooled over all types and per caller
    pooled = {}
    for name, cm in (("all", np.ones(len(types), bool)),) + tuple(
            (c, types.type.str.startswith(c + ":").to_numpy()) for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper")):
        tsel = np.flatnonzero(cm)
        if not len(tsel):
            continue
        m = np.isin(qt, tsel)
        bt = Boot(gc[m], draws=2000, seed=17)
        res = {lvl: bt.auc(K[lvl][m], pos[m]) for lvl in ("full", "L1", "L2", "L2ctl")}
        pooled[name] = {lvl: dict(auc=float(p), lo=float(np.quantile(dr, .025)), hi=float(np.quantile(dr, .975)))
                        for lvl, (p, dr) in res.items()}
        for l1, l2 in (("full", "L2"), ("full", "L1"), ("L2ctl", "L2")):
            dd = res[l1][1] - res[l2][1]
            pooled[name][f"{l1}_minus_{l2}"] = dict(d=float(res[l1][0] - res[l2][0]), lo=float(np.quantile(dd, .025)),
                                                    hi=float(np.quantile(dd, .975)))
        pooled[name]["n_types"] = int(len(tsel))
        pooled[name]["n_pos"] = int((m & pos).sum())
    log(json.dumps(pooled, indent=1))
    json.dump(dict(selected=sel, pooled=pooled), open(W / "withheld_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
