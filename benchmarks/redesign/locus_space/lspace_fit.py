#!/usr/bin/env python
"""Locus space, part 5: a fitted linear map on locus (pooled gene) vectors (workstream ws_locusspace).

Arms (fitted on odd-genome core loci only; no labels, no tokens):
  LZ  centre + soft whitening of the locus mean vectors: W1 = (S + a * tr(S)/d * I)^(-1/2), S the covariance of
      odd-genome core-locus vectors, a = 0.1 (as the gene-level zca10)
  LW  LZ followed by whitening of the within-pair covariance of self-supervised positive pairs:
      W2 = (S_pos + a * tr(S_pos)/d * I)^(-1/2) in LZ coordinates. Positive pair = two odd-genome cores in different
      genome groups (emb_hom 0.75) that are mutual 15-NN in the MEAN map graph, with Chamfer (gene-level set kernel)
      >= 0.5, and whose flanks differ (cosine of the summed vectors of up to 3 genes on each side < 0.2).
Because pooling is linear, a linear map on gene vectors before pooling equals the same map on the pooled vector.
Evaluation: gold 1-NN typing (part-1 protocol) and the locus-index search (part-4 protocol) with transformed vectors.
  lspace_fit.py fit --threads T
Predeclaration: predeclared.json part_5_fit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lspace as LS  # noqa: E402
from lspace import log, dump, OUT, RD, SEED  # noqa: E402

MAP = OUT / "map"
FIT = OUT / "fit"
ALPHA = 0.1
CH_MIN = 0.5
FLANK_MAX = 0.2
FLANK = 3


def inv_sqrt(S, alpha):
    d = S.shape[0]
    S = S + alpha * np.trace(S) / d * np.eye(d)
    w, V = np.linalg.eigh(S)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


@njit(parallel=True, cache=True)
def pair_chamfer(X, s, e, pa, pb, out):
    for t in prange(pa.shape[0]):
        a, b = pa[t], pb[t]
        a0, a1, b0, b1 = s[a], e[a] + 1, s[b], e[b] + 1
        sa = 0.0
        for i in range(a0, a1):
            m = -1e30
            for j in range(b0, b1):
                v = 0.0
                for d in range(X.shape[1]):
                    v += X[i, d] * X[j, d]
                if v > m:
                    m = v
            sa += m
        sb = 0.0
        for j in range(b0, b1):
            m = -1e30
            for i in range(a0, a1):
                v = 0.0
                for d in range(X.shape[1]):
                    v += X[i, d] * X[j, d]
                if v > m:
                    m = v
            sb += m
        out[t] = 0.5 * (sa / (a1 - a0) + sb / (b1 - b0))


@njit(parallel=True, cache=True)
def flank_vectors(X, cc, s, e, k, out):
    n = X.shape[0]
    for t in prange(s.shape[0]):
        for d in range(X.shape[1]):
            out[t, d] = 0.0
        c = cc[s[t]]
        for g in range(s[t] - k, s[t]):
            if g >= 0 and cc[g] == c:
                for d in range(X.shape[1]):
                    out[t, d] += X[g, d]
        for g in range(e[t] + 1, e[t] + 1 + k):
            if g < n and cc[g] == c:
                for d in range(X.shape[1]):
                    out[t, d] += X[g, d]


def stage_fit(a):
    import numba
    numba.set_num_threads(a.threads)
    FIT.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    res = dict(stage="fit", alpha=ALPHA, ch_min=CH_MIN, flank_max=FLANK_MAX, flank_genes=FLANK)
    U = LS.Universe(vectors=False)
    X = np.load(LS.ZROW)
    D = pd.read_parquet(MAP / "dh_loci.parquet")
    core_rows = np.flatnonzero((D.is_core & (D.span >= 2)).to_numpy())
    C = D.iloc[core_rows].reset_index(drop=True)
    V = np.load(MAP / "V_MEAN.npy").astype(np.float32)
    Vc = V[core_rows]
    s, e = C.gid_s.to_numpy(np.int64), C.gid_e.to_numpy(np.int64)
    gcode = U.gc[s]
    grp = U.groups["g075"][gcode]
    odd = (gcode % 2 == 1)
    # ---- LZ
    mu = Vc[odd].mean(0)
    Z = Vc[odd] - mu
    S = (Z.T @ Z / len(Z)).astype(np.float64)
    W1 = inv_sqrt(S, ALPHA).astype(np.float32)
    res["n_odd_cores"] = int(odd.sum())
    # ---- positives
    I = np.load(MAP / "knn_I_MEAN.npy").astype(np.int64)
    n = len(C)
    rows = np.repeat(np.arange(n), I.shape[1])
    cols = I.ravel()
    ok = (cols >= 0) & (cols != rows)
    rows, cols = rows[ok], cols[ok]
    key = rows * n + cols
    rkey = cols * n + rows
    mutual = np.isin(key, rkey, assume_unique=False)
    m = mutual & (rows < cols) & odd[rows] & odd[cols] & (grp[rows] != grp[cols])
    pa, pb = rows[m], cols[m]
    res["candidates_mutual_odd_crossgroup"] = int(len(pa))
    ch = np.empty(len(pa), np.float64)
    pair_chamfer(X, s, e, pa, pb, ch)
    F = np.empty((n, X.shape[1]), np.float32)
    flank_vectors(X, U.cc, s, e, FLANK, F)
    fn = np.linalg.norm(F, axis=1)
    Fn = F / np.maximum(fn[:, None], 1e-12)
    fc = np.einsum("ij,ij->i", Fn[pa], Fn[pb])
    has_fl = (fn[pa] > 0) & (fn[pb] > 0)
    pos = (ch >= CH_MIN) & has_fl & (fc < FLANK_MAX)
    res["positives"] = int(pos.sum())
    res["positives_frac_of_candidates"] = float(pos.mean())
    res["candidate_chamfer_q"] = np.quantile(ch, [.1, .5, .9]).tolist()
    res["candidate_flank_cos_q"] = np.quantile(fc[has_fl], [.1, .5, .9]).tolist()
    log(f"positives {pos.sum():,} of {len(pa):,} candidates")
    Za, Zb = (Vc[pa[pos]] - mu) @ W1.T, (Vc[pb[pos]] - mu) @ W1.T
    Dd = (Za - Zb).astype(np.float64)
    Sp = Dd.T @ Dd / len(Dd)
    W2 = inv_sqrt(Sp, ALPHA).astype(np.float32)
    w_eig = np.linalg.eigvalsh(Sp)
    res["within_pair_eig_q"] = np.quantile(w_eig, [0, .1, .5, .9, 1]).tolist()
    np.savez(FIT / "maps.npz", mu=mu, W1=W1, W2=W2)
    # ---- evaluation on gold (part-1 protocol): member-gene mean vectors
    I_g = LS.load_gold(U)
    Lg = LS.LocusSet(U, list(I_g.mem))
    Xu = X[Lg.ug]
    Vg = np.add.reduceat(Xu[Lg.idx], Lg.ptr[:-1], axis=0)
    Vg /= np.linalg.norm(Vg, axis=1, keepdims=True)
    reps = {}
    for name, T in (("MEAN", None), ("LZ", W1), ("LW", W2 @ W1)):
        Y = Vg if T is None else (Vg - mu) @ T.T
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        reps[name] = (Y @ Y.T).astype(np.float32)
    P = LS.typing_eval("fit", reps, Lg.mod, I_g, None)
    P.to_parquet(FIT / "typing_perquery.parquet", index=False)
    res["typing"] = LS.summarize_typing(P)
    res["paired"] = {f"{x}-{y}|{c}": LS.paired(P, x, y, c) for x, y in (("LZ", "MEAN"), ("LW", "MEAN"), ("LW", "LZ"))
                     for c in LS.CALLERS_Q}
    dump(res, FIT / "fit_results.json")
    # transformed DH locus vectors for the search arm
    for name, T in (("LZ", W1), ("LW", W2 @ W1)):
        Y = (V - mu) @ T.T
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        np.save(MAP / f"V_{name}.npy", Y.astype(np.float16))
    res["seconds"] = time.time() - T0
    dump(res, FIT / "fit_results.json")
    log(f"fit done {res['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 4)))
    a = ap.parse_args()
    stage_fit(a)


if __name__ == "__main__":
    main()
