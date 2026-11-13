#!/usr/bin/env python
"""Taxonomy-free lineage units for the recurrence engine (workstream lineagefree).

The engine counts a co-located token pair once per lineage so that near-identical genomes do not count as
independent observations. This module derives the lineage unit from data the pipeline already has; it reads no
taxonomy (only genome, genome_code and n_genes from genomes.parquet).

usage
  genome_groups.py famsim --tokens TOK.parquet --universe UNIV_DIR --out DIR
      Genome x genome similarity from shared family content (gene -> family token):
        fam_jac.npy   Jaccard index of the two genomes' family sets
        fam_con.npy   directed containment C[A, B] = share of A's genes whose family occurs in B
  genome_groups.py embsim --exact EXACT_DIR --uid UID_OF_ROW.npy --universe UNIV_DIR --out DIR --threads T
      Genome x genome similarity from the stored per-genome best matches (every unique sequence's maximum cosine in
      every genome, entries >= the storage floor 0.32):
        emb_mean.npy  directed mean over A's genes of the best-match cosine in B (entries below the floor count as
                      the floor)
        emb_hom.npy   directed mean best-match cosine over A's genes whose best match in B is >= 0.5 (homolog level;
                      an AAI analogue, insensitive to genome completeness), with emb_homn.npy its gene count
        emb_id95.npy  directed share of A's genes with a best match >= 0.95 in B
  genome_groups.py groups --sim DIR/<name>.npy --sym mean|max|min --theta X [--theta Y ...] --out DIR [--tag TAG]
      Symmetrise S, then for every theta:
        groups: average-linkage clusters of 1 - S cut at 1 - theta (method --method average|complete|single)
        weights: w(g) = 1 / #{h : S(g, h) >= theta} (h = g included)
      Writes <tag>_<method>_t<theta>.parquet with columns genome, genome_code, group, weight, n_close.
  genome_groups.py describe --sim DIR/<name>.npy --sym mean --out JSON
      Similarity distribution (pairs and nearest neighbours), density antimodes and dendrogram merge-height gaps
      (the internal threshold criteria), cluster count over a theta ladder.

Run time is reported per command (seconds) in the JSON written next to the outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

FLOOR = 0.32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def universe_genes(udir):
    g = pd.read_parquet(Path(udir) / "genes.parquet", columns=["gid", "genome_code"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    gm = pd.read_parquet(Path(udir) / "genomes.parquet", columns=["genome", "genome_code", "n_genes"])
    gm = gm.sort_values("genome_code").reset_index(drop=True)
    assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
    gc = g.genome_code.to_numpy(np.int64)
    assert (np.diff(gc) >= 0).all()
    return gc, gm


# ============================================================================== family similarity
def cmd_famsim(a):
    from scipy import sparse
    t0 = time.time()
    gc, gm = universe_genes(a.universe)
    t = pd.read_parquet(a.tokens, columns=["gid", "token"])
    tok = np.full(len(gc), -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    ok = tok >= 0
    ng, T = len(gm), int(tok.max()) + 1
    X = sparse.csr_matrix((np.ones(ok.sum()), (gc[ok], tok[ok])), shape=(ng, T))   # gene counts per family
    X.sum_duplicates()
    P = X.copy()
    P.data[:] = 1.0                                                                     # family presence
    inter = (P @ P.T).toarray()
    nf = np.asarray(P.sum(1)).ravel()
    jac = inter / (nf[:, None] + nf[None, :] - inter)
    con = (X @ P.T).toarray() / gm.n_genes.to_numpy(np.float64)[:, None]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "fam_jac.npy", jac.astype(np.float32))
    np.save(out / "fam_con.npy", con.astype(np.float32))
    info = dict(cmd="famsim", tokens=str(a.tokens), genomes=ng, families=T, genes_assigned=int(ok.sum()),
                seconds=round(time.time() - t0, 1))
    (out / "famsim.json").write_text(json.dumps(info, indent=1) + "\n")
    log("famsim", info)


# ============================================================================== embedding similarity
@njit(parallel=True, cache=True)
def _emb_accumulate(gs, ge, uq, off, gen, val16, lut, s_all, n_all, s_hom, n_hom, n_id):
    ng = len(gs)
    for A in prange(ng):
        for a in range(gs[A], ge[A]):
            u = uq[a]
            for e in range(off[u], off[u + 1]):
                B = gen[e]
                v = lut[val16[e]]
                s_all[A, B] += v
                n_all[A, B] += 1
                if v >= 0.5:
                    s_hom[A, B] += v
                    n_hom[A, B] += 1
                if v >= 0.95:
                    n_id[A, B] += 1


def cmd_embsim(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    gc, gm = universe_genes(a.universe)
    ng = len(gm)
    uq = np.load(a.uid).astype(np.int64)
    assert len(uq) == len(gc) and (uq >= 0).all()
    ex = Path(a.exact)
    counts = np.load(ex / "counts.npy").astype(np.int64)
    off = np.r_[0, np.cumsum(counts)].astype(np.int64)
    n_ent = int(off[-1])
    gen = np.fromfile(ex / "genome.bin", dtype=np.int16)
    assert len(gen) == n_ent, (len(gen), n_ent)
    val16 = np.fromfile(ex / "value.bin", dtype=np.uint16)
    assert len(val16) == n_ent
    t_read = time.time() - t0
    log("read", n_ent, "entries", f"{t_read:.0f}s")
    lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    lut[~np.isfinite(lut)] = 0.0
    brk = np.flatnonzero(np.diff(gc)) + 1
    gs = np.r_[0, brk].astype(np.int64)
    ge = np.r_[brk, len(gc)].astype(np.int64)
    s_all = np.zeros((ng, ng), np.float64)
    n_all = np.zeros((ng, ng), np.int64)
    s_hom = np.zeros((ng, ng), np.float64)
    n_hom = np.zeros((ng, ng), np.int64)
    n_id = np.zeros((ng, ng), np.int64)
    t1 = time.time()
    _emb_accumulate(gs, ge, uq, off, gen.astype(np.int64), val16, lut, s_all, n_all, s_hom, n_hom, n_id)
    t_acc = time.time() - t1
    n_g = gm.n_genes.to_numpy(np.float64)
    assert np.array_equal(ge - gs, gm.n_genes.to_numpy())
    mean = (s_all + FLOOR * (n_g[:, None] - n_all)) / n_g[:, None]
    hom = np.where(n_hom > 0, s_hom / np.maximum(n_hom, 1), FLOOR)
    id95 = n_id / n_g[:, None]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "emb_mean.npy", mean.astype(np.float32))
    np.save(out / "emb_hom.npy", hom.astype(np.float32))
    np.save(out / "emb_homn.npy", n_hom.astype(np.int32))
    np.save(out / "emb_id95.npy", id95.astype(np.float32))
    info = dict(cmd="embsim", exact=str(ex), entries=n_ent, genomes=ng, threads=a.threads,
                diag_mean_q01_q50=np.quantile(np.diag(mean), [0.01, 0.5]).tolist(),
                seconds_read=round(t_read, 1), seconds_accumulate=round(t_acc, 1), seconds=round(time.time() - t0, 1))
    (out / "embsim.json").write_text(json.dumps(info, indent=1) + "\n")
    log("embsim", info)


# ============================================================================== groups and weights
def symmetrise(S, how):
    S = S.astype(np.float64)
    if how == "mean":
        Z = 0.5 * (S + S.T)
    elif how == "max":
        Z = np.maximum(S, S.T)
    elif how == "min":
        Z = np.minimum(S, S.T)
    else:
        raise ValueError(how)
    np.fill_diagonal(Z, 1.0)
    return Z


def linkage_of(Z, method):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    D = np.clip(1.0 - Z, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method=method)


def groups_at(Lk, theta):
    from scipy.cluster.hierarchy import fcluster
    lab = fcluster(Lk, t=1.0 - theta, criterion="distance")
    _, lab = np.unique(lab, return_inverse=True)
    return lab.astype(np.int64)


def weights_at(Z, theta):
    n_close = (Z >= theta).sum(1)
    return 1.0 / n_close, n_close


def cmd_groups(a):
    t0 = time.time()
    gm = pd.read_parquet(Path(a.universe) / "genomes.parquet", columns=["genome", "genome_code"])
    gm = gm.sort_values("genome_code").reset_index(drop=True)
    Z = symmetrise(np.load(a.sim), a.sym)
    Lk = linkage_of(Z, a.method)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or Path(a.sim).stem + "_" + a.sym
    rows = []
    for th in a.theta:
        lab = groups_at(Lk, th)
        w, nc = weights_at(Z, th)
        df = pd.DataFrame(dict(genome=gm.genome, genome_code=gm.genome_code, group=lab, weight=w, n_close=nc))
        fn = out / f"{tag}_{a.method}_t{th:g}.parquet"
        df.to_parquet(fn, index=False)
        rows.append(dict(theta=th, groups=int(lab.max()) + 1, effective_genomes=float(w.sum()),
                         largest_group=int(np.bincount(lab).max()), singletons=int((np.bincount(lab) == 1).sum()), file=str(fn)))
        log(rows[-1])
    (out / f"{tag}_{a.method}.json").write_text(json.dumps(dict(sim=str(a.sim), sym=a.sym, method=a.method, rows=rows,
                                                                 seconds=round(time.time() - t0, 1)), indent=1) + "\n")


# ============================================================================== distribution and internal criteria
def antimodes(x, lo, hi, bins=200, bw=None):
    """Interior minima of a Gaussian KDE of x on [lo, hi] (grid of `bins` points), with depth = min of the two
    flanking maxima divided by the minimum (log10)."""
    from scipy.stats import gaussian_kde
    x = x[(x >= lo) & (x <= hi)]
    if len(x) > 400_000:
        x = np.random.default_rng(17).choice(x, 400_000, replace=False)
    kde = gaussian_kde(x, bw_method=bw)
    g = np.linspace(lo, hi, bins)
    f = kde(g)
    out = []
    for i in range(1, bins - 1):
        if f[i] < f[i - 1] and f[i] <= f[i + 1]:
            lm, rm = f[:i].max(), f[i + 1:].max()
            out.append(dict(at=float(g[i]), density=float(f[i]), depth_log10=float(np.log10(min(lm, rm) / max(f[i], 1e-300)))))
    return out, g.tolist(), f.tolist()


def cmd_describe(a):
    t0 = time.time()
    Z = symmetrise(np.load(a.sim), a.sym)
    n = len(Z)
    iu = np.triu_indices(n, 1)
    pairs = Z[iu]
    off = Z.copy()
    np.fill_diagonal(off, -np.inf)
    nn = off.max(1)
    qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    res = dict(sim=str(a.sim), sym=a.sym, genomes=n, pairs=int(len(pairs)),
               pair_quantiles=dict(zip(map(str, qs), np.quantile(pairs, qs).round(4).tolist())),
               nn_quantiles=dict(zip(map(str, qs), np.quantile(nn, qs).round(4).tolist())))
    edges = np.linspace(0, 1, 101)
    res["pair_hist_0_1_by_0.01"] = np.histogram(pairs, edges)[0].tolist()
    res["nn_hist_0_1_by_0.01"] = np.histogram(nn, edges)[0].tolist()
    # antimodes of the density of pairwise similarity above the bulk (pairs above the 99th percentile) and of
    # nearest-neighbour similarity
    lo = float(np.quantile(pairs, 0.99))
    res["pair_antimodes_above_q99"], _, _ = antimodes(pairs, lo, 1.0)
    res["nn_antimodes"], _, _ = antimodes(nn, float(nn.min()), 1.0)
    # average-linkage dendrogram: merge heights (as similarity) and the cluster count over a theta ladder
    for method in ("average", "complete"):
        Lk = linkage_of(Z, method)
        h = 1.0 - Lk[:, 2]
        ladder = np.round(np.arange(0.05, 1.0001, 0.01), 3)
        cnt = [int(groups_at(Lk, th).max()) + 1 for th in ladder]
        # largest gaps between successive merge similarities (sorted), excluding the last 1% of merges
        hs = np.sort(h)[::-1]
        gaps = hs[:-1] - hs[1:]
        top = np.argsort(gaps)[::-1][:10]
        res[f"{method}_clusters_by_theta"] = dict(zip(map(str, ladder.tolist()), cnt))
        res[f"{method}_largest_merge_gaps"] = [dict(upper=float(hs[i]), lower=float(hs[i + 1]), gap=float(gaps[i]),
                                                    clusters_between=int(i + 1 + 0)) for i in top]
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    log("describe", a.sim, res["pair_quantiles"], res["nn_quantiles"], res["pair_antimodes_above_q99"][:5],
        res["nn_antimodes"][:5], res["seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("famsim")
    p.add_argument("--tokens", required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("embsim")
    p.add_argument("--exact", required=True)
    p.add_argument("--uid", required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, required=True)
    p = sub.add_parser("groups")
    p.add_argument("--sim", required=True)
    p.add_argument("--sym", default="mean")
    p.add_argument("--method", default="average")
    p.add_argument("--theta", type=float, nargs="+", required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    p = sub.add_parser("describe")
    p.add_argument("--sim", required=True)
    p.add_argument("--sym", default="mean")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"famsim": cmd_famsim, "embsim": cmd_embsim, "groups": cmd_groups, "describe": cmd_describe}[a.cmd](a)


if __name__ == "__main__":
    main()
