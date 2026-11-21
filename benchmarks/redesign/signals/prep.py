#!/usr/bin/env python
"""Stage 0 of ws_signals: zca10 vectors, family sums, genome backgrounds, per-gene residual statistics.

Writes to $RD/signals/work:
  zca10_uid_f32.npy   U x 960 float32 (zca10, unit), row = uid
  fam_row.npy         token -> row in fam_sum (-1 when the family has < 3 distinct uids)
  fam_sum.npy         n_elig_fam x 960 float32: sum of the family's distinct-uid vectors
  fam_m.npy           token -> number of distinct uids (int32)
  genome_B.npy        3564 x 960 float64: sum over the genome's eligible genes of r_i (family-centred LOO residual)
  genome_n.npy        eligible genes per genome
  gene_stats.parquet  gid, eligible, rnorm (|r_i|), cos_b (cos(r_i, b_g^(-i))), fam_m
  prep_summary.json   counts and the variance decomposition of residuals into genome and group shares
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W, BLK, log, load_genes, load_tokens, load_groups, zca10_uid  # noqa: E402


def main():
    t0 = time.time()
    W.mkdir(parents=True, exist_ok=True)
    X, valid, uid = zca10_uid()
    U, d = X.shape
    log("zca10", X.shape, "valid", int(valid.sum()), f"{time.time() - t0:.0f}s")
    np.save(W / "zca10_uid_f32.npy", X)
    log("saved zca10")
    g = load_genes()
    tok = load_tokens()
    n = len(g)
    T = int(tok.max()) + 1
    gc = g.genome_code.to_numpy(np.int64)
    G = int(gc.max()) + 1
    assert valid[uid].all(), "a gene without a vector"

    key = np.unique(tok * U + uid)
    tk, uk = key // U, key % U
    m = np.bincount(tk, minlength=T).astype(np.int32)
    np.save(W / "fam_m.npy", m)
    elig_tok = m >= 3
    fam_row = np.full(T, -1, dtype=np.int64)
    fam_row[elig_tok] = np.arange(int(elig_tok.sum()))
    sel = elig_tok[tk]
    A = sp.csr_matrix((np.ones(int(sel.sum()), dtype=np.float32), (fam_row[tk[sel]], uk[sel])),
                      shape=(int(elig_tok.sum()), U))
    S = np.asarray(A @ X, dtype=np.float32)
    np.save(W / "fam_row.npy", fam_row)
    np.save(W / "fam_sum.npy", S)
    log("families", T, "eligible (>= 3 distinct uids)", int(elig_tok.sum()), "distinct (token, uid)", len(key))

    elig = elig_tok[tok]
    ge = np.flatnonzero(elig)
    mi = m[tok[ge]].astype(np.float64)
    a = mi / (mi - 1)
    A1 = sp.csr_matrix((a, (gc[ge], uid[ge])), shape=(G, U))
    A2 = sp.csr_matrix((1.0 / (mi - 1), (gc[ge], fam_row[tok[ge]])), shape=(G, S.shape[0]))
    B = np.zeros((G, d), dtype=np.float64)       # column blocks: a float64 copy of X would take 19 GB
    for c0 in range(0, d, 120):
        B[:, c0:c0 + 120] = A1 @ X[:, c0:c0 + 120].astype(np.float64) - A2 @ S[:, c0:c0 + 120].astype(np.float64)
    ng = np.bincount(gc[ge], minlength=G)
    np.save(W / "genome_B.npy", B)
    np.save(W / "genome_n.npy", ng)
    log("genome backgrounds", G, "eligible genes", len(ge), f"{time.time() - t0:.0f}s")

    rnorm = np.full(n, np.nan, dtype=np.float32)
    cosb = np.full(n, np.nan, dtype=np.float32)
    ss_r = 0.0
    for s in range(0, len(ge), BLK):
        ix = ge[s:s + BLK]
        mm = m[tok[ix]].astype(np.float32)[:, None]
        R = X[uid[ix]] * (mm / (mm - 1)) - S[fam_row[tok[ix]]] / (mm - 1)
        rn = np.linalg.norm(R, axis=1)
        rnorm[ix] = rn
        ss_r += float((rn.astype(np.float64) ** 2).sum())
        bl = (B[gc[ix]] - R) / np.maximum(ng[gc[ix]] - 1, 1)[:, None]
        cosb[ix] = (R * bl).sum(1) / np.maximum(rn * np.linalg.norm(bl, axis=1), 1e-12)
    # variance shares: genome means, group means (data-derived groups), all on eligible genes
    bmean = B / np.maximum(ng, 1)[:, None]
    ss_genome = float((ng * (bmean ** 2).sum(1)).sum())
    out = dict(U=U, genes=n, tokens=T, eligible_tokens=int(elig_tok.sum()), eligible_genes=int(len(ge)),
               ss_residual=ss_r, share_genome_means=ss_genome / ss_r,
               expected_share_if_no_genome_component=float((ng > 0).sum() / len(ge)),
               genome_bnorm_q=np.quantile(np.linalg.norm(bmean[ng > 0], axis=1), [0.1, 0.5, 0.9]).tolist(),
               rnorm_q=np.nanquantile(rnorm, [0.1, 0.5, 0.9]).tolist(),
               cos_b_q=np.nanquantile(cosb, [0.1, 0.5, 0.9]).tolist())
    for nm in ("fam_jac_average_t0.2", "emb_hom_average_t0.75", "emb_mean_average_t0.753"):
        grp = load_groups(nm)
        K = int(grp.max()) + 1
        Bs = np.zeros((K, d))
        np.add.at(Bs, grp, B)
        nk = np.bincount(grp, weights=ng, minlength=K)
        out[f"share_group_means__{nm}"] = float(((Bs ** 2).sum(1) / np.maximum(nk, 1)).sum() / ss_r)
        out[f"groups__{nm}"] = int((nk > 0).sum())
    pd.DataFrame(dict(gid=np.arange(n), eligible=elig, rnorm=rnorm, cos_b=cosb, fam_m=m[tok])).to_parquet(
        W / "gene_stats.parquet", index=False)
    out["seconds"] = time.time() - t0
    json.dump(out, open(W / "prep_summary.json", "w"), indent=1)
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
