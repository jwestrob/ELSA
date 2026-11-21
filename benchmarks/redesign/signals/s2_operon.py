#!/usr/bin/env python
"""Exploratory (written after the predeclared tests were fixed, before their results were read): provenance signal on
the showcase operon A-B-C (tokens 81718, 393054, 130016). Stratum-adjusted foreignness of the operon's genes (same
strata as s2_analyze.py) and coherence of its adjacent pairs against all adjacent pairs; genome bootstrap.
Single-threaded; writes work/s2_operon.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W, log, load_genes, load_tokens, load_groups, boot_weights, ci  # noqa: E402

G = 3564
TAG = os.environ.get("SIG_TAG", "")
TOKS = dict(A=81718, B=393054, C=130016)


def main():
    g = load_genes()
    tok = load_tokens()
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    n = len(g)
    gs = pd.read_parquet(W / "gene_stats.parquet")
    elig = gs.eligible.to_numpy(bool)
    ge = np.flatnonzero(elig)
    s2 = pd.read_parquet(W / f"s2_gene{TAG}.parquet")
    f = np.full(n, np.nan)
    f[ge] = s2.f.to_numpy()
    m = gs.fam_m.to_numpy()
    famjac = load_groups("fam_jac_average_t0.2")
    tg = pd.DataFrame(dict(t=tok, grp=famjac[gc])).drop_duplicates()
    breadth = tg.groupby("t").size().reindex(np.arange(int(tok.max()) + 1)).fillna(0).to_numpy()[tok]
    rn = gs.rnorm.to_numpy()
    dec = np.zeros(n, int)
    dec[ge] = np.minimum((pd.Series(rn[ge]).rank(pct=True).to_numpy() * 10).astype(int), 9)
    strat = (np.digitize(m, [6, 11, 31, 101]) * 4 + np.digitize(breadth, [2, 4, 11])) * 10 + dec
    smean = pd.Series(f[ge]).groupby(strat[ge]).mean()
    adj = np.full(n, np.nan)
    adj[ge] = f[ge] - smean.reindex(strat[ge]).to_numpy()
    Wb = boot_weights(G)
    out = {}
    for nm, t in TOKS.items():
        ix = np.flatnonzero((tok == t) & elig)
        s_, c_ = np.bincount(gc[ix], weights=adj[ix], minlength=G), np.bincount(gc[ix], minlength=G).astype(float)
        out[nm] = dict(genes=int(len(ix)), mean_f=float(f[ix].mean()), background_mean_f=float(np.nanmean(f[ge])),
                       stratum_adjusted=float(s_.sum() / c_.sum()), ci=ci((Wb @ s_) / (Wb @ c_)),
                       frac_host_rank0=float((s2.set_index("gid").host_rank.reindex(ix) == 0).mean()))
    # coherence of adjacent operon pairs vs all adjacent eligible pairs
    V = np.load(W / f"s2_v_famjac_f16{TAG}.npy", mmap_mode="r")
    row = np.full(n, -1, np.int64)
    row[ge] = np.arange(len(ge))
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(len(ge)), (gc[ge], np.arange(len(ge)))), shape=(G, len(ge)))
    Sg = np.asarray(A @ np.asarray(V, np.float32), np.float64)
    ng = np.bincount(gc[ge], minlength=G).astype(float)

    def u_of(x):
        v = np.asarray(V[row[x]], np.float64)
        return v - (Sg[gc[x]] - v) / np.maximum(ng[gc[x]] - 1, 1)[:, None]

    def coh(a, b):
        ua, ub = u_of(a), u_of(b)
        return (ua * ub).sum(1) / np.maximum(np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1), 1e-12)
    i = np.arange(n - 1)
    ok = (cc[i] == cc[i + 1]) & elig[i] & elig[i + 1]
    i = i[ok]
    rng = np.random.default_rng(17)
    bi = np.sort(rng.choice(i, 1_000_000, replace=False))
    cb = coh(bi, bi + 1)
    out["background_off1_mean_c"] = float(cb.mean())
    for nm, (x, y) in dict(AB=("A", "B"), BC=("B", "C")).items():
        tx, ty = TOKS[x], TOKS[y]
        sel = i[((tok[i] == tx) & (tok[i + 1] == ty)) | ((tok[i] == ty) & (tok[i + 1] == tx))]
        c = coh(sel, sel + 1)
        s_, c_ = np.bincount(gc[sel], weights=c, minlength=G), np.bincount(gc[sel], minlength=G).astype(float)
        out[f"coh_{nm}"] = dict(pairs=int(len(sel)), mean_c=float(c.mean()), ci=ci((Wb @ s_) / (Wb @ c_)))
    json.dump(out, open(W / f"s2_operon{TAG}.json", "w"), indent=1)
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
