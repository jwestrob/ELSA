#!/usr/bin/env python
"""Token arms from an embed_queue run, built at the unique-sequence level.

knn      transform the unique vectors and store their k nearest neighbours (faiss IVF, inner product)
graph    Leiden communities of the mutual-kNN graph of unique sequences -> row-level token arms
hybrid   sequence families merged by the embedding similarity of their centroids (Leiden CPM)

Inputs: an embed_queue run (final/mean_raw_f32__uid.npy, uid_of_row.npy) whose rows are the gids
of a universe (genes.parquet with gid, genome_code). Outputs: <out>/<arm>.parquet (gid, token),
<arm>.json and a line in <out>/INDEX.tsv. Identical sequences always share a token.

Transforms (weights = number of rows that carry each unique sequence):
  l2      L2-normalized residue mean
  gc      l2, minus the weighted global mean, renormalized
  zca10   gc before renormalization, soft-ZCA whitening with alpha 0.1
          (lambda' = 0.9 lambda + 0.1 mean(lambda)), renormalized
Centroid transforms for `hybrid` add `genomecentred` (each row minus the mean of its genome, as in
the Spacedust hybrid arms) and `genomecentred_zca10`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

BLK = 200_000


def stamp(*a):
    print(dt.datetime.now().isoformat(timespec="seconds"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


def load_run(emb: Path):
    X = np.load(emb / "final" / "mean_raw_f32__uid.npy")
    u = np.load(emb / "uid_of_row.npy").astype(np.int64)
    valid = np.empty(len(X), dtype=bool)
    for s in range(0, len(X), BLK):
        valid[s:s + BLK] = np.isfinite(X[s:s + BLK]).all(1)
    X[~valid] = 0
    cnt = np.bincount(u[u >= 0], minlength=len(X)).astype(np.float64)
    cnt[~valid] = 0
    return X, u, valid, cnt


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


def apply_linear(X, W):
    for s in range(0, len(X), BLK):
        X[s:s + BLK] = X[s:s + BLK] @ W


def transform(X, valid, cnt, kind):
    """In place. Returns provenance."""
    info = dict(transform=kind)
    unit(X)
    if kind == "l2":
        return info
    mu = weighted_mean(X, cnt)
    info["global_mean_norm"] = float(np.linalg.norm(mu))
    for s in range(0, len(X), BLK):
        X[s:s + BLK] -= mu.astype(np.float32)
    X[~valid] = 0
    if kind == "zca10":
        W, ev = soft_zca(weighted_cov(X, cnt))
        info["eigenvalue_q"] = np.quantile(ev, [0, 0.5, 0.9, 1]).tolist()
        info["participation_ratio_before"] = float(ev.sum() ** 2 / (ev ** 2).sum())
        apply_linear(X, W)
    elif kind != "gc":
        raise SystemExit(f"unknown transform {kind}")
    unit(X)
    return info


def ivf_knn(X, k, nlist, nprobe, seed, threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    nlist = int(min(nlist, max(1, len(X) // 64)))
    km = faiss.Kmeans(d, nlist, niter=20, seed=seed, spherical=True, max_points_per_centroid=64)
    km.train(np.ascontiguousarray(X[np.sort(rng.choice(len(X), min(len(X), nlist * 64), replace=False))]))
    quant = faiss.IndexFlatIP(d)
    quant.add(km.centroids)
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    index.add(X)
    index.nprobe = int(min(nprobe, nlist))
    D = np.empty((len(X), k + 1), dtype=np.float32)
    I = np.empty((len(X), k + 1), dtype=np.int64)
    t = time.time()
    for s in range(0, len(X), BLK):
        D[s:s + BLK], I[s:s + BLK] = index.search(X[s:s + BLK], k + 1)
        stamp(f"ivf search {min(s + BLK, len(X)):,}/{len(X):,} {time.time() - t:.0f}s")
    return drop_self(I, D)


def drop_self(I, D):
    """Drop the self hit (first column whenever it is the query itself; otherwise the last column)."""
    n = len(I)
    self_first = I[:, 0] == np.arange(n)
    Io = np.where(self_first[:, None], I[:, 1:], I[:, :-1])
    Do = np.where(self_first[:, None], D[:, 1:], D[:, :-1])
    Io[Io == np.arange(n)[:, None]] = -1
    return Io, Do


def hnsw_knn(X, k, m, efc, efs, threads):
    """Cosine kNN with a FAISS HNSW graph (unit vectors, inner product). The IVF index spent 75 min
    on 2.5 M DPANN vectors because dense families make unbalanced lists; a graph index has no lists."""
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(X.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efc
    t = time.time()
    for s in range(0, len(X), BLK):
        index.add(X[s:s + BLK])
        stamp(f"hnsw add {min(s + BLK, len(X)):,}/{len(X):,} {time.time() - t:.0f}s")
    index.hnsw.efSearch = max(efs, k + 1)
    D = np.empty((len(X), k + 1), dtype=np.float32)
    I = np.empty((len(X), k + 1), dtype=np.int64)
    t = time.time()
    for s in range(0, len(X), BLK):
        D[s:s + BLK], I[s:s + BLK] = index.search(X[s:s + BLK], k + 1)
        stamp(f"hnsw search {min(s + BLK, len(X)):,}/{len(X):,} {time.time() - t:.0f}s")
    return drop_self(I, D)


def exact_neighbours(X, q, k, threads):
    """Exact top-k (self excluded) for the query rows q, by brute force."""
    import faiss
    faiss.omp_set_num_threads(threads)
    flat = faiss.IndexFlatIP(X.shape[1])
    for s in range(0, len(X), BLK):
        flat.add(X[s:s + BLK])
    D, I = flat.search(np.ascontiguousarray(X[q]), k + 1)
    out = np.full((len(q), k), -1, dtype=np.int64)
    for r in range(len(q)):
        row = I[r][I[r] != q[r]][:k]
        out[r, :len(row)] = row
    return out


def recall_at(I_test, I_exact, ks):
    """Mean fraction of the exact top-k found among the first k test neighbours, for each k."""
    res = {}
    for kk in ks:
        hit = [len(np.intersect1d(I_test[r, :kk], I_exact[r, :kk])) / kk for r in range(len(I_exact))]
        res[f"recall@{kk}"] = round(float(np.mean(hit)), 4)
    return res


def mutual_edges(I, D):
    n = I.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), I.shape[1])
    dst = I.ravel().astype(np.int64)
    w = D.ravel()
    keep = (dst >= 0) & (dst != src)
    src, dst, w = src[keep], dst[keep], w[keep]
    key_f = np.sort(src * n + dst)
    key_r = dst * n + src
    pos = np.minimum(np.searchsorted(key_f, key_r), len(key_f) - 1)
    sel = (key_f[pos] == key_r) & (src < dst)
    return src[sel], dst[sel], w[sel]


def leiden(n, es, ed, ew, objective, resolution, seed):
    import random
    import igraph as ig
    G = ig.Graph(n=n, edges=np.column_stack([es, ed]).tolist(), directed=False)
    G.es["weight"] = ew.astype(float).tolist()
    random.seed(seed)
    ig.set_random_number_generator(random)
    part = G.community_leiden(objective_function=objective, weights="weight", resolution=resolution, n_iterations=-1)
    return np.asarray(part.membership, dtype=np.int64)


def write_arm(out: Path, arm: str, tokens: np.ndarray, man: dict, t0: float):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(tokens, dtype=np.int64)
    tmp = out / f".{arm}.parquet.tmp{os.getpid()}"
    pq.write_table(pa.table({"gid": np.arange(len(tokens), dtype=np.int64), "token": tokens}), tmp, compression="zstd")
    os.replace(tmp, out / f"{arm}.parquet")
    a = tokens[tokens >= 0]
    sizes = np.bincount(np.unique(a, return_inverse=True)[1])
    man = dict(man, arm=arm, rows=int(len(tokens)), n_assigned=int(len(a)), n_tokens=int(len(sizes)),
               singleton_tokens=int((sizes == 1).sum()), token_size_q50_q90_q99=np.quantile(sizes, [0.5, 0.9, 0.99]).tolist(),
               token_size_max=int(sizes.max()), genes_in_tokens_ge2=int(sizes[sizes >= 2].sum()),
               seconds=round(time.time() - t0, 1), peak_rss_gb=rss_gb(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               written=dt.datetime.now().isoformat(timespec="seconds"))
    tj = out / f".{arm}.json.tmp{os.getpid()}"
    tj.write_text(json.dumps(man, indent=1, default=str) + "\n")
    os.replace(tj, out / f"{arm}.json")
    with open(out / "INDEX.tsv", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        if fh.read() == "":
            fh.write("arm\tn_tokens\tn_assigned\tsingletons\tmax_size\tseconds\tdate\n")
        fh.write(f"{arm}\t{man['n_tokens']}\t{man['n_assigned']}\t{man['singleton_tokens']}\t{man['token_size_max']}\t{man['seconds']}\t{dt.date.today()}\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    stamp(f"WROTE {arm}: tokens={man['n_tokens']:,} singletons={man['singleton_tokens']:,} max={man['token_size_max']:,} "
          f"genes in tokens>=2: {man['genes_in_tokens_ge2']:,}")


def qtag(q):
    return f"{q:g}".replace(".", "p")


# --------------------------------------------------------------------------------------
def cmd_knn(a):
    t0 = time.time()
    X, u, valid, cnt = load_run(a.emb)
    info = transform(X, valid, cnt, a.transform)
    stamp("transformed", X.shape, info, f"{time.time() - t0:.0f}s rss {rss_gb()}")
    vi = np.flatnonzero(valid)
    Xv = X if len(vi) == len(X) else np.ascontiguousarray(X[vi])
    t1 = time.time()
    if a.index == "hnsw":
        I, D = hnsw_knn(Xv, a.k, a.hnsw_m, a.ef_construction, a.ef_search, a.threads)
        info.update(index="hnsw", hnsw_m=a.hnsw_m, ef_construction=a.ef_construction, ef_search=a.ef_search)
    else:
        I, D = ivf_knn(Xv, a.k, a.nlist, a.nprobe, a.seed, a.threads)
        info.update(index="ivf", nlist=a.nlist, nprobe=a.nprobe)
    info["knn_seconds"] = round(time.time() - t1, 1)
    stamp(f"kNN ({a.index}) {info['knn_seconds']}s rss {rss_gb()}")
    if a.exact_n:
        # recall of this index (and of an earlier kNN file) against brute force on a query sample
        q = np.sort(np.random.default_rng(a.seed + 1).choice(len(Xv), min(a.exact_n, len(Xv)), replace=False))
        t2 = time.time()
        Ie = exact_neighbours(Xv, q, a.k, a.threads)
        ks = [kk for kk in (1, 10, 20, a.k) if kk <= a.k]
        info["recall"] = recall_at(I[q], Ie, ks)
        stamp(f"exact check on {len(q):,} queries {time.time() - t2:.0f}s; {a.index}: {info['recall']}")
        if a.check_npz:
            old = np.load(a.check_npz, allow_pickle=True)["I"]
            inv = np.full(len(X), -1, dtype=np.int64)
            inv[vi] = np.arange(len(vi))
            Io = old[vi[q]].astype(np.int64)
            Io = np.where(Io >= 0, inv[np.maximum(Io, 0)], -1)
            info["recall_check_npz"] = dict(recall_at(Io, Ie, ks), file=str(a.check_npz))
            stamp(f"{a.check_npz.name}: {info['recall_check_npz']}")
    I = np.where(I >= 0, vi[np.maximum(I, 0)], -1)
    full_I = np.full((len(X), a.k), -1, dtype=np.int32)
    full_D = np.zeros((len(X), a.k), dtype=np.float32)
    full_I[vi], full_D[vi] = I, D
    rng = np.random.default_rng(a.seed)
    i, j = rng.choice(vi, 2_000_000), rng.choice(vi, 2_000_000)
    null = np.einsum("ij,ij->i", X[i], X[j])[i != j]
    a.work.mkdir(parents=True, exist_ok=True)
    f = a.work / f"knn__{a.tag}__{a.transform}__k{a.k}{'' if a.index == 'ivf' else '__' + a.index}.npz"
    np.savez(f, I=full_I, D=full_D, null=null.astype(np.float32), info=np.array(dict(info, emb=str(a.emb)), dtype=object))
    stamp("saved", f, "neighbour cosine q10/q50/q90 at rank 1 / 10 / k:",
          np.quantile(full_D[vi, 0], [0.1, 0.5, 0.9]).round(3), np.quantile(full_D[vi, min(9, a.k - 1)], [0.1, 0.5, 0.9]).round(3),
          np.quantile(full_D[vi, -1], [0.1, 0.5, 0.9]).round(3), "null q50/q99/q999:", np.quantile(null, [0.5, 0.99, 0.999]).round(3),
          f"{time.time() - t0:.0f}s rss {rss_gb()}")


def cmd_graph(a):
    t0 = time.time()
    z = np.load(a.knn, allow_pickle=True)
    I, D, null = z["I"][:, :a.k], z["D"][:, :a.k], z["null"]
    u = np.load(a.emb / "uid_of_row.npy").astype(np.int64)
    es, ed, ew = mutual_edges(I, D)
    stamp("mutual edges", len(es), "of", I.shape[0], "nodes; weight q10/q50/q90", np.quantile(ew, [0.1, 0.5, 0.9]).round(3))
    runs = [("mod", "modularity", float(r), f"r{qtag(float(r))}") for r in a.modularity.split(",") if r]
    runs += [("cpm", "CPM", float(np.quantile(null, float(q))), f"cpmq{qtag(float(q))}") for q in a.cpm_q.split(",") if q]
    if a.min_cos_q:
        thr = float(np.quantile(null, a.min_cos_q))
        keep = ew >= thr
        es, ed, ew = es[keep], ed[keep], ew[keep]
        stamp("edges kept at cosine >=", round(thr, 4), ":", len(es))
    for kind, obj, res, tag in runs:
        t1 = time.time()
        comp = leiden(I.shape[0], es, ed, ew, obj, res, a.seed)
        tok = np.where(u >= 0, comp[np.maximum(u, 0)], -1)
        arm = f"graph__{a.tag}__mknn{a.k}{'' if not a.min_cos_q else 'q' + qtag(a.min_cos_q)}{tag}"
        write_arm(a.out, arm, tok, dict(source="graph", knn=str(a.knn), k=a.k, objective=obj, resolution=res,
                                         min_cos_null_quantile=a.min_cos_q, edges=int(len(es)), info=z["info"].item()), t1)
    stamp("end", f"{time.time() - t0:.0f}s")


def cmd_hybrid(a):
    import duckdb
    from scipy.sparse import coo_matrix
    t0 = time.time()
    X, u, valid, cnt = load_run(a.emb)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    base = con.execute(f"SELECT token FROM read_parquet('{a.base}') ORDER BY gid").fetchnumpy()["token"].astype(np.int64)
    g = con.execute(f"SELECT genome_code FROM read_parquet('{a.universe}') ORDER BY gid").fetchnumpy()["genome_code"].astype(np.int64)
    assert len(base) == len(u) == len(g)
    rows = np.flatnonzero((u >= 0) & (base >= 0) & valid[np.maximum(u, 0)])
    uniq, fam = np.unique(base[rows], return_inverse=True)
    F, U, G = len(uniq), len(X), int(g.max()) + 1
    info = dict(centroid_transform=a.centroid_transform)
    ones = np.ones(len(rows), dtype=np.float32)
    Mfu = coo_matrix((ones, (fam, u[rows])), shape=(F, U)).tocsr()
    if a.centroid_transform in ("l2", "gc", "zca10"):
        info.update(transform(X, valid, cnt, a.centroid_transform))
        C = np.asarray(Mfu @ X, dtype=np.float32)
    else:
        unit(X)
        Mgu = coo_matrix((ones, (g[rows], u[rows])), shape=(G, U)).tocsr()
        ng = np.asarray(Mgu.sum(1)).ravel()
        Mu = np.asarray(Mgu @ X, dtype=np.float64) / np.maximum(ng, 1)[:, None]
        info["genome_mean_norm_q"] = np.quantile(np.linalg.norm(Mu, axis=1), [0, 0.5, 1]).round(4).tolist()
        Mfg = coo_matrix((ones, (fam, g[rows])), shape=(F, G)).tocsr()
        C = np.asarray(Mfu @ X, dtype=np.float64) - Mfg @ Mu           # sum over rows of (x - genome mean)
        if a.centroid_transform == "genomecentred_zca10":
            # covariance of the genome-centred rows: sum_rows x x^T - sum_g n_g mu_g mu_g^T
            S = weighted_cov(X, cnt) * cnt.sum() - (Mu * ng[:, None]).T @ Mu
            W, ev = soft_zca(S / len(rows))
            info["participation_ratio_before"] = float(ev.sum() ** 2 / (ev ** 2).sum())
            C = C @ W.astype(np.float64)
        elif a.centroid_transform != "genomecentred":
            raise SystemExit(f"unknown centroid transform {a.centroid_transform}")
        C = C.astype(np.float32)
    del X
    unit(C)
    has = np.linalg.norm(C, axis=1) > 0
    fidx = np.flatnonzero(has)
    Cv = np.ascontiguousarray(C[fidx])
    rng = np.random.default_rng(a.seed)
    i, j = rng.integers(0, len(fidx), 2_000_000), rng.integers(0, len(fidx), 2_000_000)
    null = np.einsum("ij,ij->i", Cv[i], Cv[j])[i != j]
    stamp("families", F, "with centroid", len(fidx), "null cos q50/q90/q99/q999", np.quantile(null, [0.5, 0.9, 0.99, 0.999]).round(4),
          f"{time.time() - t0:.0f}s rss {rss_gb()}")
    I, D = ivf_knn(Cv, a.k, a.nlist, a.nprobe, a.seed, a.threads)
    es, ed, ew = mutual_edges(I, D)
    stamp("mutual family edges", len(es), "cos q10/q50/q90", np.quantile(ew, [0.1, 0.5, 0.9]).round(4))
    t_base = time.time() - t0
    runs = [(f"q{qtag(float(q))}", float(np.quantile(null, float(q)))) for q in a.res_q.split(",") if q]
    runs += [(f"r{qtag(float(r))}", float(r)) for r in a.res.split(",") if r]
    nrow = len(base)
    for tag, res in runs:
        t1 = time.time()
        comp = leiden(len(fidx), es, ed, ew, "CPM", res, a.seed)
        merged = np.arange(F, dtype=np.int64)
        merged[fidx] = F + comp
        _, merged = np.unique(merged, return_inverse=True)
        tok = np.full(nrow, -1, dtype=np.int64)
        tok[rows] = merged[fam]
        rest = np.flatnonzero((tok < 0) & (base >= 0))                # rows without a vector keep their own family
        if len(rest):
            _, r2 = np.unique(base[rest], return_inverse=True)
            tok[rest] = merged.max() + 1 + r2
        csize = np.bincount(comp)
        arm = f"hybrid__{a.tag}__{a.centroid_transform}__{a.base_tag}cpm{tag}"
        write_arm(a.out, arm, tok, dict(source="hybrid", base=str(a.base), k=a.k, cpm_resolution=res, families=int(F),
                                         families_with_centroid=int(len(fidx)), mutual_edges=int(len(es)),
                                         merged_groups=int(len(csize)), merged_group_max_families=int(csize.max()),
                                         families_merged=int((csize[comp] > 1).sum()), info=info,
                                         null_q=np.quantile(null, [0.5, 0.9, 0.99, 0.999]).tolist(), seconds_base=round(t_base, 1)), t1)
    stamp("end", f"{time.time() - t0:.0f}s rss {rss_gb()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--emb", type=Path, required=True, help="embed_queue run directory")
        p.add_argument("--tag", required=True, help="model tag used in arm names, e.g. esmpp_small")
        p.add_argument("--threads", type=int, required=True)
        p.add_argument("--seed", type=int, default=17)

    p = sub.add_parser("knn")
    common(p)
    p.add_argument("--transform", default="zca10", choices=("l2", "gc", "zca10"))
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--nprobe", type=int, default=48)
    p.add_argument("--index", default="ivf", choices=("ivf", "hnsw"), help="hnsw writes knn__<tag>__<transform>__k<k>__hnsw.npz")
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=128)
    p.add_argument("--exact-n", type=int, default=0, help="check recall against brute force on this many sampled queries")
    p.add_argument("--check-npz", type=Path, default=None, help="an earlier kNN file to score against the same exact sample")
    p.add_argument("--work", type=Path, required=True)
    p.set_defaults(fn=cmd_knn)
    p = sub.add_parser("graph")
    common(p)
    p.add_argument("--knn", type=Path, required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--modularity", default="", help="comma-separated modularity resolutions")
    p.add_argument("--cpm-q", default="", help="comma-separated null quantiles used as CPM resolutions")
    p.add_argument("--min-cos-q", type=float, default=0.0, help="keep mutual edges with cosine >= this null quantile")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_graph)
    p = sub.add_parser("hybrid")
    common(p)
    p.add_argument("--base", type=Path, required=True, help="family token parquet (gid, token)")
    p.add_argument("--base-tag", required=True, help="short name of the base families in the arm name")
    p.add_argument("--universe", type=Path, required=True, help="genes.parquet with gid, genome_code")
    p.add_argument("--centroid-transform", default="genomecentred", choices=("l2", "gc", "zca10", "genomecentred", "genomecentred_zca10"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--res-q", default="0.5,0.9,0.99,0.999", help="CPM resolutions as quantiles of the random-pair centroid cosine")
    p.add_argument("--res", default="", help="CPM resolutions as cosine values")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_hybrid)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
