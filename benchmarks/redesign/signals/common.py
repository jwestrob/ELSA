"""Shared paths and helpers for workstream ws_signals (biotite)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
ENG = RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"
LF = RD / "lineagefree"
DB = RD / "ws_dpann/input/sharur.duckdb"
W = RD / "signals/work"
BLK = 200_000
SEED = 17


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def load_genes():
    g = pd.read_parquet(PKG / "universe/genes.parquet",
                        columns=["gid", "genome_code", "contig_code", "position_index", "strand", "start", "end", "length"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    return g


def load_tokens():
    t = pd.read_parquet(TOK)
    t = t.sort_values("gid")
    assert (t.gid.to_numpy() == np.arange(len(t))).all()
    return t.token.to_numpy(np.int64)


def load_groups(name):
    d = pd.read_parquet(LF / f"groups/{name}.parquet").sort_values("genome_code")
    assert (d.genome_code.to_numpy() == np.arange(len(d))).all()
    return d.group.to_numpy(np.int64)


# ---------------------------------------------------------------- zca10 (copied from softrec/embed_tokens.py)
def _unit(X):
    for s in range(0, len(X), BLK):
        n = np.linalg.norm(X[s:s + BLK], axis=1, keepdims=True)
        n[n == 0] = 1
        X[s:s + BLK] /= n


def _wmean(X, w):
    mu = np.zeros(X.shape[1], dtype=np.float64)
    for s in range(0, len(X), BLK):
        mu += (X[s:s + BLK].astype(np.float64) * w[s:s + BLK, None]).sum(0)
    return mu / w.sum()


def _wcov(X, w):
    d = X.shape[1]
    C = np.zeros((d, d), dtype=np.float64)
    for s in range(0, len(X), BLK):
        b = X[s:s + BLK].astype(np.float64)
        C += (b * w[s:s + BLK, None]).T @ b
    return C / w.sum()


def zca10_uid():
    """Returns X (U x 960, float32, zca10, unit) and the valid mask, exactly as softrec's transform(kind='zca10')."""
    X = np.load(EMB / "final" / "mean_raw_f32__uid.npy")
    u = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    valid = np.empty(len(X), dtype=bool)
    for s in range(0, len(X), BLK):
        valid[s:s + BLK] = np.isfinite(X[s:s + BLK]).all(1)
    X[~valid] = 0
    cnt = np.bincount(u[u >= 0], minlength=len(X)).astype(np.float64)
    cnt[~valid] = 0
    _unit(X)
    mu = _wmean(X, cnt)
    for s in range(0, len(X), BLK):
        X[s:s + BLK] -= mu.astype(np.float32)
    X[~valid] = 0
    ev, Uv = np.linalg.eigh(_wcov(X, cnt))
    ev = np.maximum(ev, 0)
    lam = 0.9 * ev + 0.1 * ev.mean()
    Wz = ((Uv / np.sqrt(lam)) @ Uv.T).astype(np.float32)
    for s in range(0, len(X), BLK):
        X[s:s + BLK] = X[s:s + BLK] @ Wz
    _unit(X)
    return X, valid, u


# ---------------------------------------------------------------- statistics
def auc_w(score, y, w=None):
    """Weighted AUROC (ties count one half). score, y (bool), w (non-negative weights)."""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    w = np.ones(len(score)) if w is None else np.asarray(w, np.float64)
    ok = np.isfinite(score) & (w > 0)
    score, y, w = score[ok], y[ok], w[ok]
    wp, wn = w[y].sum(), w[~y].sum()
    if wp == 0 or wn == 0:
        return np.nan
    u, inv = np.unique(score, return_inverse=True)
    pos = np.bincount(inv, weights=w * y, minlength=len(u))
    neg = np.bincount(inv, weights=w * ~y, minlength=len(u))
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (wp * wn))


def boot_weights(n_units, draws=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=draws).astype(np.float64)


def ci(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [np.nan, np.nan]
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
