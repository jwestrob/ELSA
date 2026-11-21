#!/usr/bin/env python
"""POST HOC (decided after the even-genome bench results): realisation unit resolution. The F count with embedding
families of the zca10 mutual-kNN graph at resolutions r100 / r300 / r1000 (run-time admissible) against r3000 (the
predeclared unit) and MMseqs2 families (evaluation-only), F_D1000_excess, even genomes, against the critic."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_eval import Boot, kappa, log, CALLERS  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
K = RD / "kinds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    adj = pd.read_parquet(RD / "grammar/prep/adj1.parquet", columns=["genome_code", "singleton_either", "gold_none"]
                          + [f"pos_{c}" for c in CALLERS])
    z = np.load(RD / "grammar/eval/assembled_cv.npz")
    crit = z["primary"].astype(np.float64)
    G3 = z["G3_train"]
    g = adj.genome_code.to_numpy()
    even = g % 2 == 0
    strata = {"all": np.ones(len(g), bool), "never_w3": G3 == 0, ">=5_w3": G3 >= 5,
              "singleton_either": adj.singleton_either.to_numpy()}
    meth = {"critic": crit}
    for unit, d in (("r3000", K / "bench"), ("r1000", K / "bench_r1000"), ("r300", K / "bench_r300"),
                    ("r100", K / "bench_r100")):
        if not (d / "pass_null2.npz").exists():
            continue
        obs = np.load(d / "pass_obs.npz")["out"]
        nl = [np.load(d / f"pass_null{r}.npz")["out"] for r in range(3)]
        meth[f"F_{unit}"] = kappa(obs, nl, 3, 2, "excess")
        meth[f"FX_{unit}"] = kappa(obs, nl, 3, 3, "excess")
        if unit == "r3000":
            meth["M_mmseqs"] = kappa(obs, nl, 3, 4, "excess")
            meth["A_descent"] = kappa(obs, nl, 3, 0, "excess")
    rows = []
    for caller in CALLERS:
        for st, sm in strata.items():
            p = adj[f"pos_{caller}"].to_numpy() & sm & even
            q = adj.gold_none.to_numpy() & sm & even
            sel = p | q
            bt = Boot(g[sel], draws=2000, seed=17)
            res = {m: bt.auc(x[sel], p[sel]) for m, x in meth.items()}
            for m, (pt, dr) in res.items():
                dd = dr - res["critic"][1]
                rows.append(dict(caller=caller, stratum=st, n_pos=int(p.sum()), method=m, auc=pt,
                                 lo=np.quantile(dr, .025), hi=np.quantile(dr, .975), minus_critic=pt - res["critic"][0],
                                 mc_lo=np.quantile(dd, .025), mc_hi=np.quantile(dd, .975)))
            log(caller, st, {m: round(v[0], 3) for m, v in res.items()})
    t = pd.DataFrame(rows)
    t.to_csv(K / "bench_units_posthoc.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
