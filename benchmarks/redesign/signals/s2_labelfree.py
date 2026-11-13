#!/usr/bin/env python
"""Post hoc, label-free variant of signal 2 (written after the predeclared signal-2 results were read).

No group labels and no classifier: the lineage subspace is the top K right singular vectors of the genome-background
matrix (b_g = mean family-centred residual of genome g, centred over genomes). Each eligible gene is projected,
z_i = P^T r_i; s2_analyze.py (SIG_TAG=_lf) then uses z in place of the centred log-probabilities (host-relative
direction u_i = z_i - mean z over the genome's other genes) and f_lf = -cos(z_i, zbar_g^(-i)) in place of f.
Writes s2_v_famjac_f16_lf.npy, s2_gene_lf.parquet, s2_labelfree.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PKG, W, EMB, BLK, log, load_tokens  # noqa: E402

K = 64


def main():
    t0 = time.time()
    tok = load_tokens()
    uid = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    gs = pd.read_parquet(W / "gene_stats.parquet", columns=["eligible"])
    elig = gs.eligible.to_numpy(bool)
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["genome_code"])
    gc = g.genome_code.to_numpy(np.int64)
    ge = np.flatnonzero(elig)
    X = np.load(W / "zca10_uid_f32.npy")
    fam_row, S, m = np.load(W / "fam_row.npy"), np.load(W / "fam_sum.npy"), np.load(W / "fam_m.npy")
    B, ng = np.load(W / "genome_B.npy"), np.load(W / "genome_n.npy").astype(np.float64)
    bm = B / np.maximum(ng, 1)[:, None]
    w = ng / ng.sum()
    bc = (bm - (w[:, None] * bm).sum(0)) * np.sqrt(w)[:, None]
    _, sv, Vt = np.linalg.svd(bc, full_matrices=False)
    P = Vt[:K].T.astype(np.float32)
    ev = sv ** 2
    info = dict(K=K, share_of_background_variance_topK=float(ev[:K].sum() / ev.sum()), top5=(ev[:5] / ev.sum()).tolist())
    Z = np.lib.format.open_memmap(W / "s2_v_famjac_f16_lf.npy", mode="w+", dtype=np.float16, shape=(len(ge), K))
    Zf = np.zeros((len(ge), K), np.float32)
    for s in range(0, len(ge), BLK):
        ix = ge[s:s + BLK]
        mm = m[tok[ix]].astype(np.float32)[:, None]
        R = X[uid[ix]] * (mm / (mm - 1)) - S[fam_row[tok[ix]]] / (mm - 1)
        Zf[s:s + BLK] = R @ P
    Z[:] = Zf.astype(np.float16)
    Z.flush()
    G = 3564
    import scipy.sparse as sp
    Sg = np.asarray(sp.csr_matrix((np.ones(len(ge)), (gc[ge], np.arange(len(ge)))), shape=(G, len(ge))) @ Zf, np.float64)
    nge = np.bincount(gc[ge], minlength=G).astype(np.float64)
    zb = (Sg[gc[ge]] - Zf) / np.maximum(nge[gc[ge]] - 1, 1)[:, None]
    f = -(Zf * zb).sum(1) / np.maximum(np.linalg.norm(Zf, axis=1) * np.linalg.norm(zb, axis=1), 1e-12)
    pd.DataFrame(dict(gid=ge, f=f)).to_parquet(W / "s2_gene_lf.parquet", index=False)
    info["seconds"] = time.time() - t0
    info["f_lf_q"] = np.quantile(f, [0.1, 0.5, 0.9]).tolist()
    json.dump(info, open(W / "s2_labelfree.json", "w"), indent=1)
    log(json.dumps(info))


if __name__ == "__main__":
    main()
