#!/usr/bin/env python
"""Execution wrapper for the SECONDARY configuration while the family tokens are queued (coordinator, 2026-11-14).
fz.py is imported unchanged (sha256 32222775...). Two token-dependent extras of the copied code are deferred:
  prep  fz.cmd_prep saves kappa_var_knn50.npy, then reads the token cohesion for one descriptive number (Spearman with
        the token cohesion). Run 1 stops there (FileNotFoundError on kappa_var_tokens.npy, checked); run 2 finds the kNN
        cohesion on disk and builds the exclusion masks. The description lines of the kNN cohesion are recomputed with
        the same expressions once the tokens exist ('describe').
  arm   fz.cmd_arm builds trees for every entry of fz.KAPPA; the token-cohesion check tree (outputs *tok*, never scored)
        is dropped from the dict for this run. The knn outputs do not read the token array. When the tokens exist the
        unchanged fz.py arm is rerun and its L1 / L4bp_t0.02 are checked identical to these.
usage: fzrun.py prep|arm|describe --threads T
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fz  # noqa: E402


def cmd_prep(a):
    try:
        fz.cmd_prep(a)
        fz.log("prep completed in one run (token cohesion present)")
        return
    except FileNotFoundError as ex:
        assert "kappa_var_tokens" in str(ex), ex
        assert fz.KAPPA["knn"].exists()
        fz.log("run 1 stopped at the token-cohesion description as expected; kNN cohesion saved:", fz.KAPPA["knn"])
    fz.cmd_prep(a)


def cmd_arm(a):
    fz.KAPPA = {"knn": fz.KAPPA["knn"]}
    a.arm = "Z"
    fz.cmd_arm(a)


def cmd_describe(a):
    """kappa_knn50 description of sdu.cmd_prep (same expressions), written to results/prep_lut_kappa_knn50.json."""
    import numba
    numba.set_num_threads(a.threads)
    X = np.load(fz.LL / "emb/zca10_f32__uid.npy")
    uq = fz.uid_of_gene()
    g = pd.read_parquet(fz.UNIV / "genes.parquet", columns=["gid", "genome_code", "contig_code"])
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    n = len(g)
    rng = np.random.default_rng(17)
    ga = rng.integers(0, n, 1_000_000)
    gb = rng.integers(0, n, 1_000_000)
    okp = gc[ga] != gc[gb]
    o = np.empty(int(okp.sum()))
    fz.gene_pair_cos(ga[okp], gb[okp], X, uq, o)
    c0 = float(np.nanmean(o))
    kappa = np.load(fz.KAPPA["knn"]).astype(np.float64)
    tok = np.load(fz.KAPPA["tok"]).astype(np.float64)
    same = cc[:-1] == cc[1:]
    res = dict(gene_pair_floor=c0, q05_q25_q50_q75_q95=np.nanquantile(kappa, [.05, .25, .5, .75, .95]).tolist(),
               nan_within_contig=float(np.isnan(kappa[:-1][same]).mean()),
               share_below_02_05_1_2=[float(np.nanmean(kappa[:-1][same] < t)) for t in (.02, .05, .1, .2)],
               spearman_with_token_kappa=float(pd.Series(kappa).corr(pd.Series(tok), method="spearman")))
    fz.ll.jdump(res, fz.LL / "results/prep_lut_kappa_knn50.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "arm", "describe"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arm", default="Z")
    a = ap.parse_args()
    {"prep": cmd_prep, "arm": cmd_arm, "describe": cmd_describe}[a.cmd](a)


if __name__ == "__main__":
    main()
