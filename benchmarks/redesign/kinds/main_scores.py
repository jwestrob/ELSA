#!/usr/bin/env python
"""Selected kappa on all adjacencies with every other genome as reference (main passes).

Writes $RD/kinds/main/scores.parquet (one row per adjacency = adj1 row) and summary_main.json:
  - kappa of the selected variant, R, R0, ablation counts at the selected depth
  - random same-genome pair kappa (null calibration: q99, q99.9)
  - share of adjacencies above the random-pair q99 by engine status and family status
  - screen-1 co-fold benchmark (grammar evaluation 2): AUROC for ipTM >= 0.75 of kappa vs critic 0.692 /
    engine 0.748 on the same 597 folded pairs
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
M = RD / "kinds/main"
DEPTHS = [30, 100, 300, 1000]
STATS = ["A", "G", "F", "FX", "M", "X", "FXo", "Xo"]
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    sel = json.load(open(RD / "kinds/bench/summary.json"))["selected"]
    s_name, D, form = sel.split("_")
    v, s = DEPTHS.index(int(D[1:])), STATS.index(s_name)
    log("selected variant", sel)
    obs = np.load(M / "pass_obs.npz")
    O, H, n_adj = obs["out"], obs["hom"], int(obs["n_adj"])
    N = [np.load(M / f"pass_null{r}.npz")["out"] for r in range(3)]

    def kap(si, vi=v, f=form):
        R = O[:, vi, si].astype(np.float64)
        R0 = np.mean([x[:, vi, si] for x in N], axis=0)
        return (np.log((R + 1) / (R0 + 1)) if f == "ratio" else R - R0), R, R0

    k, R, R0 = kap(s)
    adj = pd.read_parquet(RD / "grammar/prep/adj1.parquet")
    assert len(adj) == n_adj
    cos = np.load(RD / "kinds/prep/adj_cos.npy")
    df = adj[["i", "j", "genome_code", "config", "ta", "tb", "singleton_either", "same_token", "unannotated_both",
              "gold_none", "eng_L", "eng_G", "eng_called", "eng_nlogP", "raw_cos"]].copy()
    df["kappa"] = k[:n_adj]
    df["R"] = R[:n_adj].astype(np.int32)
    df["R0"] = R0[:n_adj].astype(np.float32)
    for si, nm in enumerate(STATS):
        df[f"n_{nm}"] = O[:n_adj, v, si]
    df["hom_b_in_a"] = H[:n_adj, 0]
    df["hom_a_in_b"] = H[:n_adj, 1]
    df["wcos"] = cos
    df["homologous"] = (cos >= 0.32) | (df.hom_b_in_a <= 1000) | (df.hom_a_in_b <= 1000) | df.same_token
    df["uncallable"] = df.singleton_either | (df.eng_L < 2)
    kr = k[n_adj:]
    q99, q999 = np.quantile(kr, [0.99, 0.999])
    df["above_q99"] = df.kappa > q99
    df.to_parquet(M / "scores.parquet", index=False)
    log("written scores", len(df))
    summ = dict(selected=sel, random_pairs=int(len(kr)), random_kappa_q50_q99_q999=[float(np.quantile(kr, .5)),
                float(q99), float(q999)], random_frac_R_gt_0=float((O[n_adj:, v, s] > 0).mean()),
                overflow_obs=int(obs["ovf"].sum()))
    grp = []
    for nm, m in (("all adjacencies", np.ones(len(df), bool)),
                  ("non-homologous", ~df.homologous),
                  ("engine-called", df.eng_called & ~df.homologous),
                  ("uncallable", df.uncallable & ~df.homologous),
                  ("uncallable, singleton member", df.singleton_either & ~df.homologous),
                  ("uncallable, L<2, no singleton", ~df.singleton_either & (df.eng_L < 2) & ~df.homologous),
                  ("uncallable, both unannotated, no gold", df.uncallable & ~df.homologous & df.unannotated_both & df.gold_none)):
        grp.append(dict(set=nm, n=int(m.sum()), frac_above_q99=float(df.above_q99[m].mean()),
                        frac_above_q999=float((df.kappa[m] > q999).mean()), frac_R_gt_0=float((df.R[m] > 0).mean())))
    summ["above_null"] = grp
    log(pd.DataFrame(grp).to_string(index=False))
    # screen-1 co-fold benchmark (grammar evaluation 2)
    f = pd.read_parquet(RD / "grammar/prep/folded.parquet")
    lut = pd.Series(np.arange(len(df)), index=df.i.to_numpy())
    lo = np.minimum(f.g0, f.g1)
    f["row"] = np.where(f.offset == 1, lut.reindex(lo).to_numpy(), np.nan)
    ff = f[f.row.notna()].copy()
    ff["kappa"] = df.kappa.to_numpy()[ff.row.astype(int)]
    ff["R"] = df.R.to_numpy()[ff.row.astype(int)]
    y = (ff.iptm >= 0.75).to_numpy()
    cs = pd.read_parquet(RD / "grammar/eval/folded_scored_cv.parquet")
    summ["screen1_cofold"] = dict(n_folded=int(len(f)), n_adjacent=int(len(ff)), hits=int(y.sum()),
                                  auc_kappa=float(auc(ff.kappa.to_numpy(), y)),
                                  auc_engine_nlogP=float(auc(ff.eng_nlogP.to_numpy(), y)),
                                  by_class={c: dict(n=int((ff.cls == c).sum()), hits=int(y[(ff.cls == c).to_numpy()].sum()),
                                                    auc_kappa=(float(auc(ff.kappa.to_numpy()[(ff.cls == c).to_numpy()],
                                                                          y[(ff.cls == c).to_numpy()]))
                                                               if 0 < y[(ff.cls == c).to_numpy()].sum() < (ff.cls == c).sum() else None))
                                            for c in ("cognate", "uncalled")},
                                  critic_columns=list(cs.columns))
    ff[["fold_id", "cls", "g0", "g1", "iptm", "pass_c123", "eng_nlogP", "kappa", "R"]].to_csv(M / "screen1_kappa.tsv",
                                                                                         sep="\t", index=False)
    log("screen-1 cofold", summ["screen1_cofold"])
    json.dump(summ, open(M / "summary_main.json", "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
