#!/usr/bin/env python
"""Kind recurrence on the grammar critic benchmark (held-out GTDB orders; reference = training side).

1. Reproduce the critic's stored AUROC (DefenseFinder no-gold, never_w3 0.7676, singleton_either 0.7863).
2. Select one kappa variant on ODD genome_code (predeclared rule).
3. Test on EVEN genome_code against the critic (paired genome bootstrap, 2,000 draws, seed 17); all genomes pooled
   reported for reference (contains the tuning half).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
B = RD / "kinds/bench"
DEPTHS = [30, 100, 300, 1000]
STATS = ["A", "G", "F", "FX", "M", "X", "FXo", "Xo"]
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


@njit(cache=True)
def _auc_w(yy, gg, starts, w):
    n = len(yy)
    cum_n, num, wp_tot, wn_tot = 0.0, 0.0, 0.0, 0.0
    for k in range(len(starts)):
        s = starts[k]
        e = starts[k + 1] if k + 1 < len(starts) else n
        wp, wn = 0.0, 0.0
        for i in range(s, e):
            x = w[gg[i]]
            if yy[i]:
                wp += x
            else:
                wn += x
        num += wp * (cum_n + 0.5 * wn)
        cum_n += wn
        wp_tot += wp
        wn_tot += wn
    if wp_tot == 0.0 or wn_tot == 0.0:
        return np.nan
    return num / (wp_tot * wn_tot)


@njit(parallel=True, cache=True)
def _auc_draws(yy, gg, starts, W):
    out = np.empty(W.shape[0])
    for d in prange(W.shape[0]):
        out[d] = _auc_w(yy, gg, starts, W[d])
    return out


class Boot:
    """Genome bootstrap (grammar/evaluate.py, draws and seed as set here)."""
    def __init__(self, groups, draws=2000, seed=17):
        codes, self.inv = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(seed)
        G = len(codes)
        self.W = rng.multinomial(G, np.full(G, 1.0 / G), size=draws).astype(np.float64) if draws else None
        self.one = np.ones(G)

    def auc(self, s, y, draws=True):
        o = np.argsort(s, kind="mergesort")
        ss, yy, gg = s[o], y[o], self.inv[o]
        starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1].astype(np.int64)
        pt = _auc_w(yy, gg, starts, self.one)
        if not draws or self.W is None:
            return pt, None
        return pt, _auc_draws(yy, gg, starts, self.W)


def kappa(obs, nulls, v, s, form):
    R = obs[:, v, s].astype(np.float64)
    R0 = np.mean([x[:, v, s] for x in nulls], axis=0).astype(np.float64)
    if form == "ratio":
        return np.log((R + 1.0) / (R0 + 1.0))
    return R - R0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    adj = pd.read_parquet(RD / "grammar/prep/adj1.parquet",
                          columns=["genome_code", "singleton_either", "same_token", "gold_none"]
                          + [f"pos_{c}" for c in CALLERS] + [f"straddle_{c}" for c in CALLERS])
    z = np.load(RD / "grammar/eval/assembled_cv.npz")
    sc = {k: z[k].astype(np.float64) for k in ("primary", "pmi_a1", "raw_cos", "G3_train", "eng_nlogP")}
    fold_of = z["fold_of"]
    g = adj.genome_code.to_numpy()
    G3 = sc["G3_train"]
    strata = {"all": np.ones(len(adj), bool), "never_w3": G3 == 0, "1-4_w3": (G3 >= 1) & (G3 <= 4),
              ">=5_w3": G3 >= 5, "singleton_either": adj.singleton_either.to_numpy()}
    covered = fold_of >= 0
    nogold = adj.gold_none.to_numpy() & covered
    summary = {}

    # ------------------------------------------------------------ 1. reproduce the critic
    rep = {}
    for st, ref in (("never_w3", 0.767587), ("singleton_either", 0.786264)):
        p = adj.pos_DefenseFinder.to_numpy() & covered & strata[st]
        q = nogold & strata[st]
        sel = p | q
        pt, _ = Boot(g[sel], draws=0).auc(sc["primary"][sel], p[sel], draws=False)
        rep[st] = dict(reproduced=round(pt, 6), stored=ref, equal_to_6dp=bool(abs(pt - ref) < 5e-7))
    log("reproduce critic", rep)
    summary["reproduce_critic"] = rep
    if not all(r["equal_to_6dp"] for r in rep.values()):
        print("critic reference not reproduced", file=sys.stderr)

    # ------------------------------------------------------------ kappa variants
    obs = np.load(B / "pass_obs.npz")["out"]
    nulls = [np.load(B / f"pass_null{r}.npz")["out"] for r in range(3)]
    ovf = np.load(B / "pass_obs.npz")["ovf"]
    log("bench passes loaded", obs.shape, "missing", int((obs[:, 0, 0] < 0).sum()), "overflow", int(ovf.sum()))
    variants = {}
    for s in STATS:
        for v, D in enumerate(DEPTHS):
            for form in ("ratio", "excess"):
                variants[f"{s}_D{D}_{form}"] = (v, STATS.index(s), form)

    def score(name):
        v, s, form = variants[name]
        return kappa(obs, nulls, v, s, form)

    # ------------------------------------------------------------ 2. selection on odd genomes
    odd = (g % 2) == 1
    even = ~odd
    grid = [f"{s}_D{D}_{f}" for s in ("F", "FX") for D in DEPTHS for f in ("ratio", "excess")]
    rows = []
    for nm in grid:
        x = score(nm)
        r = dict(variant=nm)
        for st in ("never_w3", "singleton_either", "all"):
            p = adj.pos_DefenseFinder.to_numpy() & covered & strata[st] & odd
            q = nogold & strata[st] & odd
            sel = p | q
            r[st], _ = Boot(g[sel], draws=0).auc(x[sel], p[sel], draws=False)
            r[f"n_pos_{st}"] = int(p.sum())
        r["criterion"] = (r["never_w3"] + r["singleton_either"]) / 2
        rows.append(r)
    selt = pd.DataFrame(rows).sort_values(["criterion", "all"], ascending=False)
    selt.to_csv(B / "select_odd.tsv", sep="\t", index=False)
    best = selt.variant.iloc[0]
    log("selection (odd genomes)\n" + selt.round(4).to_string(index=False))
    log("SELECTED", best)
    summary["selected"] = best
    summary["selection_table"] = selt.round(5).to_dict("records")
    json.dump(summary, open(B / "summary.json", "w"), indent=1)

    # ------------------------------------------------------------ 3. test (even) + pooled
    v_sel, s_sel, f_sel = variants[best]
    D_sel = DEPTHS[v_sel]
    methods = {"kappa": score(best), "critic": sc["primary"], "pmi_a1": sc["pmi_a1"], "raw_cos": sc["raw_cos"],
               "engine_nlogP": sc["eng_nlogP"]}
    for s in STATS:
        nm = f"{s}_D{D_sel}_{f_sel}"
        methods["abl_" + nm] = score(nm)
    out_rows = []
    for split_name, gm in (("even", even), ("all", np.ones(len(g), bool)), ("odd", odd)):
        for caller in CALLERS:
            for st, smask in strata.items():
                if split_name == "odd" and st not in ("never_w3", "singleton_either", "all"):
                    continue
                p = adj[f"pos_{caller}"].to_numpy() & covered & smask & gm
                q = nogold & smask & gm
                if p.sum() < 5:
                    continue
                sel = p | q
                y = p[sel]
                bt = Boot(g[sel], draws=2000, seed=17)
                res = {}
                for m, x in methods.items():
                    res[m] = bt.auc(x[sel], y)
                for m, (pt, dr) in res.items():
                    r = dict(split=split_name, caller=caller, stratum=st, n_pos=int(p.sum()), n_neg=int(q.sum()),
                             n_genomes=int(len(np.unique(g[sel]))), method=m, auc=pt,
                             lo=float(np.nanquantile(dr, .025)), hi=float(np.nanquantile(dr, .975)))
                    if m != "critic":
                        d = dr - res["critic"][1]
                        r.update(minus_critic=pt - res["critic"][0], mc_lo=float(np.nanquantile(d, .025)),
                                 mc_hi=float(np.nanquantile(d, .975)))
                    out_rows.append(r)
                log(split_name, caller, st, int(p.sum()), {m: round(res[m][0], 3) for m in
                                                            ("kappa", "critic", "pmi_a1", "raw_cos")})
    t = pd.DataFrame(out_rows)
    t.to_csv(B / "eval_test.tsv", sep="\t", index=False)
    key = t[(t.caller == "DefenseFinder") & (t.method.isin(["kappa", "critic", "pmi_a1", "raw_cos"]))]
    summary["test_DefenseFinder"] = key.round(4).to_dict("records")
    json.dump(summary, open(B / "summary.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
