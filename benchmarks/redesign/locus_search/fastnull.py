#!/usr/bin/env python
"""Fast null for locus search: E-values from a few null replicates with a fitted exponential tail.

The circular-shift null of locus_search.null_tail gives, per replicate, the descending scores of the
non-overlapping null hits. Pooled over R replicates, E_emp(S) = #(null hits >= S) / R. Below
E = 1/R the empirical estimate is 0, so a small R cannot resolve E <= 0.01. Above a threshold u
(the score exceeded by k0 = c * R pooled null hits, i.e. c hits per replicate) the excess scores
are modelled as exponential (peaks over threshold, shape 0):

    E_fit(S) = (k0 / R) * exp(-(S - u) / beta),  beta = mean excess of the k0 top null hits over u,

and E_emp is used below u. With too few null hits the empirical estimate is returned.

  fastnull.py calib --threads T     calibration of the fitted tail against 200-replicate E-values
                                    (round-2 final runs, null tails stored per run)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def evalue_emp(scores, tail, R):
    flat = np.sort(tail[np.isfinite(tail)])
    return (len(flat) - np.searchsorted(flat, scores, side="left")) / R


def fit_exp(tail, R, c=1.0, min_exc=5):
    """(u, k0, beta) of the exponential tail, or None when the pooled null is too small."""
    flat = tail[np.isfinite(tail)]
    k0 = max(int(round(c * R)), min_exc)
    if len(flat) < k0 + 1:
        return None
    top = -np.sort(-flat)[:k0 + 1]
    u = top[k0]
    beta = float((top[:k0] - u).mean())
    if not beta > 0:
        return None
    return float(u), k0, beta


def evalue_fit(scores, tail, R, c=1.0):
    scores = np.asarray(scores, float)
    E = evalue_emp(scores, tail, R)
    p = fit_exp(tail, R, c)
    if p is None:
        return E
    u, k0, beta = p
    return np.where(scores > u, (k0 / R) * np.exp(-(scores - u) / beta), E)


def evalue_hill(scores, tail, R, c=1.0):
    """Pareto tail in the score (exponential in log score): E = (k0/R) (S/u)^-alpha above u, alpha by Hill."""
    scores = np.asarray(scores, float)
    E = evalue_emp(scores, tail, R)
    flat = tail[np.isfinite(tail)]
    k0 = max(int(round(c * R)), 5)
    if len(flat) < k0 + 1:
        return E
    top = -np.sort(-flat)[:k0 + 1]
    u = top[k0]
    if not u > 0:
        return E
    m = float(np.log(top[:k0] / u).mean())
    if not m > 0:
        return E
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Ef = (k0 / R) * np.power(np.maximum(scores, 1e-300) / u, -1.0 / m)
    return np.where(scores > u, Ef, E)


def evalue_gumbel(scores, tail, R, c=1.0):
    """Gumbel fitted to the per-replicate maxima by moments; E = exp(-(S - mu) / beta) above the median maximum."""
    scores = np.asarray(scores, float)
    E = evalue_emp(scores, tail, R)
    mx = tail[:, 0][np.isfinite(tail[:, 0])]
    if len(mx) < 5 or not mx.std() > 0:
        return E
    beta = mx.std(ddof=1) * np.sqrt(6) / np.pi
    mu = mx.mean() - 0.5772156649 * beta
    with np.errstate(over="ignore"):
        Ef = -np.log1p(-np.minimum(1 - np.exp(-np.exp(-(scores - mu) / beta)), 1 - 1e-16))
    return np.where(scores > np.median(mx), Ef, E)


MODEL_FN = {"exp": lambda s, t, R, c: evalue_fit(s, t, R, c), "hill": evalue_hill, "gum": evalue_gumbel,
            "emp": lambda s, t, R, c: evalue_emp(s, t, R)}


def evalue_gpd(scores, tail, R, c=1.0):
    """Generalized Pareto tail (shape free, location 0) over the same threshold; for comparison."""
    from scipy.stats import genpareto
    scores = np.asarray(scores, float)
    E = evalue_emp(scores, tail, R)
    flat = tail[np.isfinite(tail)]
    k0 = max(int(round(c * R)), 5)
    if len(flat) < k0 + 1:
        return E
    top = -np.sort(-flat)[:k0 + 1]
    u = top[k0]
    exc = top[:k0] - u
    if not exc.mean() > 0:
        return E
    xi, _, sc = genpareto.fit(exc, floc=0)
    return np.where(scores > u, (k0 / R) * genpareto.sf(scores - u, xi, loc=0, scale=sc), E)


# =========================================================================== calibration
XGRID = 10.0 ** np.linspace(-4, 1, 26)
J001 = int(np.argmin(np.abs(np.log10(XGRID) + 2)))
CAL = {}


def _model_task(model):
    """Per run: holdout null counts at each XGRID value (per replicate) and E of the observed hits."""
    T, Hg = CAL["T"], CAL["Hg"]
    kind, rr, cc = model.split(":")
    rr, cc = int(rr), float(cc)
    fn = MODEL_FN[kind]
    cnt = np.zeros((len(T), len(XGRID)))
    hits = []
    for i, t in enumerate(T.itertuples()):
        tail = t.tails.astype(np.float64)
        fit, hold = tail[:rr], tail[rr:]
        hs = hold[np.isfinite(hold)]
        e = np.sort(fn(hs, fit, rr, cc))
        cnt[i] = np.searchsorted(e, XGRID, side="right") / hold.shape[0]
        h = Hg.get((t.qid, t.method))
        if h is not None and len(h):
            hits.append(pd.DataFrame(dict(run=i, method=t.method, E200=h.evalue.to_numpy(), Efit=fn(h.score.to_numpy(), fit, rr, cc),
                                          label=h.label.to_numpy())))
    return model, cnt, pd.concat(hits, ignore_index=True)


def _summ(cnt_rows, HH, rng, xs=(0.001, 0.01, 0.1, 1.0)):
    d = {"null_calibration": {}, "qq_bins": [], "decision_E0.01": {}}
    for x in xs:
        j = int(np.argmin(np.abs(np.log10(XGRID) - np.log10(x))))
        v = cnt_rows[:, j]
        pick = rng.integers(0, len(v), size=(2000, len(v)))
        bm = v[pick].mean(1)
        d["null_calibration"][str(x)] = dict(expected=x, observed=float(v.mean()), ratio=float(v.mean() / x),
                                             ci=[float(np.quantile(bm, .025) / x), float(np.quantile(bm, .975) / x)])
    bins = [(0.0, 0.002), (0.002, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 10.0)]
    for lo, hi in bins:
        sel = (HH.E200 <= hi) & ((HH.E200 > lo) if lo > 0 else True)
        if not sel.any():
            continue
        r = np.log10(np.maximum(HH.Efit[sel].to_numpy(), 1e-12)) - np.log10(np.maximum(HH.E200[sel].to_numpy(), 1e-12))
        d["qq_bins"].append(dict(E200_lo=lo, E200_hi=hi, hits=int(sel.sum()), median_E200=float(HH.E200[sel].median()),
                                 median_Efit=float(HH.Efit[sel].median()),
                                 log10_ratio_q10_50_90=np.quantile(r, [.1, .5, .9]).round(3).tolist()))
    a200, af = HH.E200 <= 0.01, HH.Efit <= 0.01
    d["decision_E0.01"] = dict(both=int((a200 & af).sum()), only200=int((a200 & ~af).sum()), onlyfit=int((~a200 & af).sum()),
                               agreement=float((a200 == af).mean()))
    for m, g in HH.groupby("method"):
        d["decision_E0.01"][m] = dict(n200=int((g.E200 <= 0.01).sum()), nfit=int((g.Efit <= 0.01).sum()),
                                      unlabelled200=int(((g.E200 <= 0.01) & (g.label == 0)).sum()),
                                      unlabelledfit=int(((g.Efit <= 0.01) & (g.label == 0)).sum()))
    qs = np.linspace(0.01, 0.99, 25)
    d["qq_quantiles"] = dict(q=qs.round(3).tolist(),
                             log10_E200=np.quantile(np.log10(np.maximum(HH.E200, 1e-6)), qs).round(3).tolist(),
                             log10_Efit=np.quantile(np.log10(np.maximum(HH.Efit, 1e-6)), qs).round(3).tolist())
    return d


def calibration_curve(cnt_rows):
    """Monotone map from a model's E to the observed null rate (per replicate), on XGRID."""
    r = np.maximum.accumulate(np.maximum(cnt_rows.mean(0), 1e-6))
    return r


def apply_curve(E, r):
    lx, lr = np.log10(XGRID), np.log10(r)
    le = np.log10(np.maximum(np.asarray(E, float), 1e-12))
    out = np.interp(le, lx, lr)
    lo = le < lx[0]                                   # below the grid: keep the ratio at the grid minimum
    out[lo] = le[lo] + (lr[0] - lx[0])
    return 10.0 ** out


def learn_curves(out, kind="gum", R=40, c=1.0, sets=("dfall", "xeven")):
    """Per-method calibration curves (null rate per replicate at XGRID) of a small-R tail model, from the stored
    200-replicate tails: model fitted on replicates [0, R), counted on the independent replicates [R, 200)."""
    fn = MODEL_FN[kind]
    rows = {}
    for tag in sets:
        z = np.load(Path(out) / f"final_{tag}_tails.npz", allow_pickle=True)
        for m, tail in zip(z["method"], z["tails"]):
            tail = tail.astype(np.float64)
            fit, hold = tail[:R], tail[R:]
            e = np.sort(fn(hold[np.isfinite(hold)], fit, R, c))
            rows.setdefault(str(m), []).append(np.searchsorted(e, XGRID, side="right") / hold.shape[0])
    return {m: calibration_curve(np.asarray(v)) for m, v in rows.items()}


def calib(a):
    import multiprocessing as mp
    out = a.out
    t0 = time.time()
    runs = []
    for tag in a.sets.split(","):
        z = np.load(out / f"final_{tag}_tails.npz", allow_pickle=True)
        runs.append(pd.DataFrame(dict(set=tag, qid=z["qid"], method=z["method"])))
        runs[-1]["tails"] = list(z["tails"])
    T = pd.concat(runs, ignore_index=True)
    Q = pd.read_parquet(Path(str(out).replace("out2", "out")) / "queries.parquet")
    T["genome_code"] = T.qid.map(dict(zip(Q.qid, Q.genome_code)))
    T["test"] = (T.genome_code % 4 == 2).to_numpy()   # held-out quarter of the genomes for calibrated models
    H = pd.concat([pd.read_parquet(out / f"final_{tag}_hits.parquet") for tag in a.sets.split(",")], ignore_index=True)
    M = pd.concat([pd.read_parquet(out / f"final_{tag}_metrics.parquet") for tag in a.sets.split(",")], ignore_index=True)
    R = T.tails.iloc[0].shape[0]
    CAL.update(T=T, Hg={k: g for k, g in H.groupby(["qid", "method"])})
    print(f"{len(T):,} runs ({T.qid.nunique():,} queries; test runs {int(T.test.sum()):,}) with {R}-replicate tails, "
          f"{len(H):,} hits (E200 <= 10), {time.time() - t0:.0f}s", flush=True)
    models = a.models.split(",")
    with mp.get_context("fork").Pool(min(a.threads, len(models))) as pool:
        outs = pool.map(_model_task, models)
    rng = np.random.default_rng(17)
    res = dict(R_full=R, runs=int(len(T)), queries=int(T.qid.nunique()), methods=sorted(T.method.unique().tolist()),
               test_runs=int(T.test.sum()), test_queries=int(T[T.test].qid.nunique()), models={}, calibrated={})
    test = T.test.to_numpy()
    for model, cnt, HH in outs:
        res["models"][model] = _summ(cnt, HH, rng)
        res["models"][model]["per_method_ratio_E0.01"] = {
            m: float(cnt[(T.method == m).to_numpy(), J001].mean() / XGRID[J001]) for m in sorted(T.method.unique())}
        # calibrated: per-method curve learned on train genomes, evaluated on held-out genomes
        HH["test"] = test[HH.run.to_numpy()]
        cal_cnt = np.zeros_like(cnt)
        Hc = HH[HH.test].copy()
        curves = {}
        for m in sorted(T.method.unique()):
            tr = (~test) & (T.method == m).to_numpy()
            curves[m] = calibration_curve(cnt[tr])
            sel = Hc.method == m
            Hc.loc[sel, "Efit"] = apply_curve(Hc.Efit[sel], curves[m])
        # held-out null: recompute counts through the curve (E_cal <= x  <=>  E_model <= curve^-1(x))
        for i in np.flatnonzero(test):
            r = curves[T.method.iloc[i]]
            # cnt[i] holds P(E_model <= XGRID); E_cal <= x  <=>  E_model <= x' with r(x') = x
            xprime = 10.0 ** np.interp(np.log10(XGRID), np.log10(r), np.log10(XGRID))
            cal_cnt[i] = np.interp(np.log10(xprime), np.log10(XGRID), cnt[i])
        res["calibrated"][model] = _summ(cal_cnt[test], Hc, rng)
        res["calibrated"][model]["curves_ratio_at_E0.01"] = {m: float(c[J001] / XGRID[J001]) for m, c in curves.items()}
        print(model, "raw", json.dumps({k: round(v["ratio"], 3) for k, v in res["models"][model]["null_calibration"].items()}),
              "calibrated", json.dumps({k: round(v["ratio"], 3) for k, v in res["calibrated"][model]["null_calibration"].items()}),
              flush=True)
    tm = M.groupby("method")[["sec_scan", "sec_null"]].mean()
    res["seconds_per_query"] = {m: dict(scan=float(r.sec_scan), null200=float(r.sec_null),
                                        null_per_replicate=float(r.sec_null / R)) for m, r in tm.iterrows()}
    json.dump(res, open(out / a.outfile, "w"), indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calib"])
    ap.add_argument("--out", type=Path, default=Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/"
                                                     "redesign_20261111/locus_search/out2"))
    ap.add_argument("--sets", default="dfall")
    ap.add_argument("--models", default="emp:40:1,emp:100:1,exp:40:1,exp:40:0.5,exp:40:0.25,hill:20:0.5,hill:30:0.5,"
                                        "hill:40:0.5,hill:40:1,hill:40:0.25,gum:40:1,exp:20:0.5,exp:30:0.5")
    ap.add_argument("--outfile", default="fastnull_calibration2.json")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    calib(a)


if __name__ == "__main__":
    main()
