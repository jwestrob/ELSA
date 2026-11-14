#!/usr/bin/env python
"""Embedding-derived inputs for the Spacedust collection (workstream transfer_spacedust).

zca       stored ESM-C 300M vectors (weights identical to ESM++ small) -> unique-sequence zca10 vectors, refitted here
          exactly as embed_tokens.transform('zca10'): L2 normalise, subtract the gene-weighted mean, soft-ZCA alpha 0.1 of
          the gene-weighted covariance (lambda' = 0.9 lambda + 0.1 mean lambda), renormalise. No labels.
          Writes emb/zca10_f32__uid.npy, zca10_f16__uid.npy, meta.npz (uq = uid per gene, g_start, g_end), zca.json.
cohesion  ws_context token cohesion (context_units/cuts.py side_sums / gene_cohesion, copied) on the rescue3 tokens:
          kappa[k] for boundary (k, k+1); writes emb/kappa_var_tokens.npy, cohesion.json.
groups    genome x genome emb_mean / emb_hom / emb_id95 from the GPU table's accumulated sums (gtable.py), then
          average-linkage groups on 1 - emb_hom (mean-symmetrised) at theta 0.75 (lineagefree genome_groups.py, copied),
          plus the description (nearest-neighbour similarity, cluster counts over a ladder).

usage: emb.py zca|cohesion|groups --threads T
"""
import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
TS = RD / "transfer_spacedust"
E = TS / "emb"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/model_scaling_20260913/results/esmc300m/embedding")
UNIV = RD / "universe"
TOK = TS / "tok/mmseqs__seq__e3c50s75_rescue3__spacedust.parquet"
BLK = 200_000
NREF = 4190383


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


# ------------------------------------------------------------------ zca10 (embed_tokens.py, copied)
def unit(X):
    for s in range(0, len(X), BLK):
        n = np.linalg.norm(X[s:s + BLK], axis=1, keepdims=True)
        n[n == 0] = 1
        X[s:s + BLK] /= n


def weighted_mean(X, w):
    mu = np.zeros(X.shape[1], dtype=np.float64)
    for s in range(0, len(X), BLK):
        mu += (X[s:s + BLK].astype(np.float64) * w[s:s + BLK, None]).sum(0)
    return mu / w.sum()


def weighted_cov(X, w):
    d = X.shape[1]
    C = np.zeros((d, d), dtype=np.float64)
    for s in range(0, len(X), BLK):
        b = X[s:s + BLK].astype(np.float64)
        C += (b * w[s:s + BLK, None]).T @ b
    return C / w.sum()


def soft_zca(C, alpha=0.1):
    ev, U = np.linalg.eigh(C)
    ev = np.maximum(ev, 0)
    lam = (1 - alpha) * ev + alpha * ev.mean()
    return ((U / np.sqrt(lam)) @ U.T).astype(np.float32), ev


def transform_zca10(X, valid, cnt):
    info = dict(transform="zca10")
    unit(X)
    mu = weighted_mean(X, cnt)
    info["global_mean_norm"] = float(np.linalg.norm(mu))
    for s in range(0, len(X), BLK):
        X[s:s + BLK] -= mu.astype(np.float32)
    X[~valid] = 0
    W, ev = soft_zca(weighted_cov(X, cnt))
    info["eigenvalue_q"] = np.quantile(ev, [0, 0.5, 0.9, 1]).tolist()
    info["participation_ratio_before"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    for s in range(0, len(X), BLK):
        X[s:s + BLK] = X[s:s + BLK] @ W
    unit(X)
    return info


def cmd_zca(a):
    t0 = time.time()
    E.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(UNIV / "genes.parquet", columns=["gid", "side", "row_index", "gene_id", "genome_code"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    n = len(g)
    # embedding rows must be the universe rows: same gene_id in the same physical order on each side
    for side, sub in (("reference", "ref"), ("query", "query")):
        m = pd.read_parquet(EMB / side / "metadata.parquet", columns=["gene_id"]) if (EMB / side / "metadata.parquet").exists() else None
        gg = g[g.side == sub]
        if m is not None:
            assert len(m) == len(gg) and (m.gene_id.to_numpy() == gg.gene_id.to_numpy()[np.argsort(gg.row_index.to_numpy())]).all(), side
            log(side, "metadata gene_id order equals the universe row_index order")
        else:
            log(side, "no metadata.parquet beside the vectors; relying on the manifest's ordered_gene_ids hash")
    uq = np.load(TS / "fam/gid2uid.npy").astype(np.int64)
    U = int(uq.max()) + 1
    cnt = np.bincount(uq, minlength=U).astype(np.float64)
    first = np.full(U, -1, np.int64)
    first[uq[::-1]] = np.arange(n)[::-1]
    R = np.load(EMB / "reference/embeddings.npy", mmap_mode="r")
    Q = np.load(EMB / "query/embeddings.npy", mmap_mode="r")
    assert R.shape == (NREF, 960) and Q.shape == (n - NREF, 960)
    X = np.empty((U, 960), np.float32)
    rows = first
    for s in range(0, U, BLK):
        r = rows[s:s + BLK]
        isq = r >= NREF
        out = np.empty((len(r), 960), np.float32)
        if (~isq).any():
            idx = r[~isq]
            o = np.argsort(idx)
            tmp = np.asarray(R[idx[o]])
            out[np.flatnonzero(~isq)[o]] = tmp
        if isq.any():
            idx = r[isq] - NREF
            out[isq] = np.asarray(Q[idx])
        X[s:s + BLK] = out
    valid = np.isfinite(X).all(1)
    log("unique vectors", X.shape, "invalid", int((~valid).sum()), f"{time.time()-t0:.0f}s")
    # numerics of repeated sequences (different batches): cosine between copies of one uid
    dup = np.flatnonzero(cnt >= 2)
    rng = np.random.default_rng(17)
    pick = rng.choice(dup, min(20000, len(dup)), replace=False) if len(dup) else np.zeros(0, np.int64)
    order = np.argsort(uq, kind="stable")
    st = np.searchsorted(uq[order], pick)
    cs = []
    for ga, gb in zip(order[st], order[st + 1]):
        va = R[ga] if ga < NREF else Q[ga - NREF]
        vb = R[gb] if gb < NREF else Q[gb - NREF]
        cs.append(float(np.dot(va, vb) / np.linalg.norm(va) / np.linalg.norm(vb)))
    cs = np.asarray(cs)
    X[~valid] = 0
    info = transform_zca10(X, valid, cnt)
    np.save(E / "zca10_f32__uid.npy", X)
    np.save(E / "zca10_f16__uid.npy", X.astype(np.float16))
    gc = g.genome_code.to_numpy(np.int64)
    ch = np.flatnonzero(np.diff(gc)) + 1
    np.savez(E / "meta.npz", uq=uq, g_start=np.r_[0, ch].astype(np.int64), g_end=np.r_[ch, n].astype(np.int64), cnt=cnt)
    info.update(genes=n, unique=U, invalid_unique=int((~valid).sum()),
                duplicate_copy_cosine_n=int(len(cs)),
                duplicate_copy_cosine_min_q01_q50=[float(cs.min()), float(np.quantile(cs, .01)), float(np.median(cs))] if len(cs) else [],
                source=str(EMB), seconds=round(time.time() - t0, 1), peak_rss_gb=rss_gb())
    (E / "zca.json").write_text(json.dumps(info, indent=1) + "\n")
    log("zca10", info)


# ------------------------------------------------------------------ cohesion (context_units/cuts.py, copied)
@njit(cache=True)
def side_sums(tok, uid, X, strand, cid, T):
    d = X.shape[1]
    n = len(tok)
    A = np.zeros((2, T, d), dtype=np.float32)
    cnt = np.zeros((2, T), dtype=np.int64)
    for m in range(n):
        t = tok[m]
        if t < 0:
            continue
        for side in range(2):
            step = 1 if (strand[m] == 1) == (side == 0) else -1
            nm = m + step
            if nm < 0 or nm >= n or cid[nm] != cid[m]:
                continue
            q = uid[nm]
            cnt[side, t] += 1
            for j in range(d):
                A[side, t, j] += X[q, j]
    return A, cnt


@njit(parallel=True, cache=True)
def side_norms(A):
    out = np.zeros((A.shape[0], A.shape[1]))
    for t in prange(A.shape[1]):
        for s in range(A.shape[0]):
            x = 0.0
            for j in range(A.shape[2]):
                x += np.float64(A[s, t, j]) * A[s, t, j]
            out[s, t] = x
    return out


@njit(parallel=True, cache=True)
def gene_cohesion(tok, uid, X, strand, cid, A, cnt, normA2, c0, out_right, out_left):
    n = len(tok)
    d = X.shape[1]
    for i in prange(n):
        out_right[i] = np.nan
        out_left[i] = np.nan
        t = tok[i]
        if t < 0:
            continue
        for which in range(2):
            nb = i + 1 if which == 0 else i - 1
            if nb < 0 or nb >= n or cid[nb] != cid[i]:
                continue
            side = 0 if (strand[i] == 1) == (which == 0) else 1
            nn = cnt[side, t] - 1
            if nn < 2:
                continue
            q = uid[nb]
            dot = 0.0
            for j in range(d):
                dot += A[side, t, j] * X[q, j]
            s2 = normA2[side, t] - 2.0 * dot + 1.0
            rho = (s2 - nn) / (nn * (nn - 1.0))
            rho = (rho - c0) / (1.0 - c0)
            if which == 0:
                out_right[i] = rho
            else:
                out_left[i] = rho


@njit(parallel=True, cache=True)
def gene_pair_cos(ga, gb, X, uid, out):
    d = X.shape[1]
    for s in prange(len(ga)):
        qa, qb = uid[ga[s]], uid[gb[s]]
        x1 = 0.0
        x2 = 0.0
        x12 = 0.0
        for q in range(d):
            a = X[qa, q]
            b = X[qb, q]
            x1 += a * a
            x2 += b * b
            x12 += a * b
        out[s] = x12 / math.sqrt(x1 * x2) if x1 > 0 and x2 > 0 else np.nan


def cmd_cohesion(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    g = pd.read_parquet(UNIV / "genes.parquet", columns=["gid", "genome_code", "contig_code", "strand"])
    n = len(g)
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    cid = pd.factorize(cc)[0].astype(np.int64)
    strand = g.strand.to_numpy(np.int64)
    t = pd.read_parquet(TOK)
    tok = np.full(n, -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    T = int(tok.max()) + 1
    X = np.load(E / "zca10_f32__uid.npy")
    uid = np.load(E / "meta.npz")["uq"].astype(np.int64)
    rng = np.random.default_rng(17)
    ga = rng.integers(0, n, 1_000_000)
    gb = rng.integers(0, n, 1_000_000)
    ok = gc[ga] != gc[gb]
    o = np.empty(int(ok.sum()))
    gene_pair_cos(ga[ok], gb[ok], X, uid, o)
    c0 = float(np.nanmean(o))
    A, cnt = side_sums(tok, uid, X, strand, cid, T)
    normA2 = side_norms(A)
    kr = np.empty(n)
    kl = np.empty(n)
    gene_cohesion(tok, uid, X, strand, cid, A, cnt, normA2, c0, kr, kl)
    del A, X
    kappa = np.fmin(kr[:-1], kl[1:])
    same = cid[:-1] == cid[1:]
    kappa = np.r_[np.where(same, kappa, np.nan), np.nan]
    np.save(E / "kappa_var_tokens.npy", kappa.astype(np.float32))
    inside = kappa[:-1][same]
    info = dict(gene_pair_floor=c0, floor_pairs=int(ok.sum()),
                kappa_q05_q25_q50_q75_q95=np.nanquantile(inside, [.05, .25, .5, .75, .95]).tolist(),
                kappa_nan_frac=float(np.isnan(inside).mean()),
                frac_below={str(th): float((inside < th).mean()) for th in (0.02, 0.05, 0.1, 0.2)},
                tokens=T, seconds=round(time.time() - t0, 1), peak_rss_gb=rss_gb())
    (E / "cohesion.json").write_text(json.dumps(info, indent=1) + "\n")
    log("cohesion", info)


# ------------------------------------------------------------------ groups (lineagefree genome_groups.py, copied)
def symmetrise(S):
    Z = 0.5 * (S.astype(np.float64) + S.T.astype(np.float64))
    np.fill_diagonal(Z, 1.0)
    return Z


def linkage_of(Z, method="average"):
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


FLOOR = 0.32


def cmd_groups(a):
    t0 = time.time()
    G = TS / "gtable"
    gm = pd.read_parquet(UNIV / "genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    ng = len(gm)
    z = np.load(G / "sums.npz")
    s_all, n_all, s_hom, n_hom, n_id = (z[k] for k in ("s_all", "n_all", "s_hom", "n_hom", "n_id"))
    n_g = gm.n_genes.to_numpy(np.float64)
    mean = (s_all + FLOOR * (n_g[:, None] - n_all)) / n_g[:, None]
    hom = np.where(n_hom > 0, s_hom / np.maximum(n_hom, 1), FLOOR)
    id95 = n_id / n_g[:, None]
    out = TS / "groups"
    out.mkdir(parents=True, exist_ok=True)
    for nm, M in (("emb_mean", mean), ("emb_hom", hom), ("emb_id95", id95)):
        np.save(out / f"{nm}.npy", M.astype(np.float32))
    np.save(out / "emb_homn.npy", n_hom.astype(np.int32))
    Z = symmetrise(hom)
    Lk = linkage_of(Z)
    lab = groups_at(Lk, 0.75)
    pd.DataFrame(dict(genome=gm.genome, genome_code=gm.genome_code, group=lab)).to_parquet(
        out / "emb_hom_average_t0.75.parquet", index=False)
    sizes = np.bincount(lab)
    iu = np.triu_indices(ng, 1)
    off = Z.copy()
    np.fill_diagonal(off, -np.inf)
    nn = off.max(1)
    ladder = np.round(np.arange(0.5, 0.96, 0.05), 2)
    qg = gm.side.to_numpy() == "query"
    # taxonomy is read here for description only (GTDB family / genus AMI of the groups), never by the method
    from sklearn.metrics import adjusted_mutual_info_score as ami
    ref = ~qg & gm.family.notna().to_numpy()
    res = dict(genomes=ng, theta=0.75, groups=int(len(sizes)), largest=int(sizes.max()), singletons=int((sizes == 1).sum()),
               size_q50_q90_q99=np.quantile(sizes, [.5, .9, .99]).tolist(),
               pair_q50_q90_q99_q999=np.quantile(Z[iu], [.5, .9, .99, .999]).tolist(),
               nn_q10_q50_q90=np.quantile(nn, [.1, .5, .9]).tolist(),
               groups_by_theta={str(t): int(groups_at(Lk, t).max() + 1) for t in ladder},
               query_genome_groups={gm.genome[i]: int(lab[i]) for i in np.flatnonzero(qg)},
               query_group_sizes={gm.genome[i]: int(sizes[lab[i]]) for i in np.flatnonzero(qg)},
               ami_family_ref=float(ami(gm.family[ref], lab[ref])), ami_genus_ref=float(ami(gm.genus[ref].fillna("na"), lab[ref])),
               seconds=round(time.time() - t0, 1))
    (out / "groups.json").write_text(json.dumps(res, indent=1) + "\n")
    log("groups", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["zca", "cohesion", "groups"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    {"zca": cmd_zca, "cohesion": cmd_cohesion, "groups": cmd_groups}[a.cmd](a)


if __name__ == "__main__":
    main()
