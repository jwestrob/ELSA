#!/usr/bin/env python
"""Legacy ELSA (gene-level kNN anchors -> collinear chaining -> blocks) on the DPANN package.

Workstream ws_baseline, ELSA redesign, 2026-11-12. Runs the research-checkout `elsa/` package
copied to $RD/baseline/src (branch agent/normalized-result-contract working tree).

usage:
  legacy.py zca                       fit soft-ZCA alpha 0.1 on DPANN (elsa.transform, shipped
                                      defaults: 200,000-row sample, seed 17) and write the vectors
  legacy.py chain   --arm raw|zca     kNN + anchors + chaining, exactly as run_chain_pipeline's
                                      first half (writes anchors.parquet, blocks_checkpoint.parquet)
  legacy.py perm    --arm A --seed S  null: permute gene positions within each contig (seed S, or
                                      'identity' for the replay check), re-chain the saved anchors
  legacy.py cluster --arm A           run_chain_pipeline resuming from the blocks checkpoint
                                      (overlap clustering, contained-cluster merge, CSV outputs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
B = RD / "baseline"
PKG = RD / "ws_dpann/package"
VEC_RAW = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/"
               "dpann__esmpp_small__20261111/final/l2_f32__row.npy")
VEC_ZCA = B / "vectors/esmpp_small__soft_zca_a0.1.npy"
ARMS = {"raw": dict(vectors=VEC_RAW, threshold=0.85, transform="none"),
        "zca": dict(vectors=VEC_ZCA, threshold=0.30, transform="soft_zca")}
sys.path.insert(0, str(B / "src"))
SUBSET = int(os.environ.get("BL_SUBSET", "0"))      # smoke test: first SUBSET genomes (genome_code < SUBSET)


def run_dir(arm):
    return B / "runs" / (f"{arm}_smoke{SUBSET}" if SUBSET else arm)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def threads():
    return int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))


def maxrss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2   # Linux: KiB


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def src_hashes():
    return {f: sha(B / "src/elsa" / f) for f in ("analyze/pipeline.py", "seed.py", "chain.py", "cluster.py",
                                                 "index.py", "transform.py")}


def universe():
    g = pd.read_parquet(PKG / "universe/genes.parquet",
                        columns=["gid", "protein_id", "genome", "contig", "contig_code", "position_index",
                                 "strand", "start", "end"])
    assert np.array_equal(g.gid.to_numpy(), np.arange(len(g))), "gid must be 0..n-1"
    assert (np.diff(g.contig_code.to_numpy()) >= 0).all(), "gid order must follow contig_code"
    if SUBSET:
        gc = pd.read_parquet(PKG / "universe/genes.parquet", columns=["genome_code"]).genome_code.to_numpy()
        g = g[gc < SUBSET].reset_index(drop=True)
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
    return g


def genes_for_pipeline(g):
    """genes_df as run_chain_pipeline expects it; verifies that its stable sort is the identity and
    that its position_index equals the package position_index."""
    gdf = pd.DataFrame({"sample_id": g.genome.to_numpy(), "contig_id": g.contig.to_numpy(),
                        "gene_id": g.protein_id.to_numpy(), "start": g.start.to_numpy(),
                        "end": g.end.to_numpy(), "strand": g.strand.to_numpy()})
    order = gdf.sort_values(["sample_id", "contig_id", "start", "end"], kind="stable").index.to_numpy()
    assert np.array_equal(order, np.arange(len(gdf))), "pipeline sort is not the identity on gid order"
    gdf["position_index"] = gdf.groupby(["sample_id", "contig_id"]).cumcount()
    assert np.array_equal(gdf.position_index.to_numpy(), g.position_index.to_numpy())
    return gdf


def config(arm, T):
    from elsa.analyze.pipeline import ChainConfig
    a = ARMS[arm]
    # canonical parameters (CLAUDE.md): k 50, max gap 2, min chain 2, no gap penalty; all others default
    return ChainConfig(embedding_transform=a["transform"], hnsw_k=50, similarity_threshold=a["threshold"],
                       max_gap_genes=2, min_chain_size=2, gap_penalty_scale=0.0, n_jobs=T)


def patch_hnsw_threads(T):
    """build_gene_index calls add_items with hnswlib's default thread count; pin it to T."""
    import hnswlib
    import elsa.index as ei

    class _Shim:
        def Index(self, *a, **k):
            idx = hnswlib.Index(*a, **k)
            idx.set_num_threads(T)
            return idx
    ei.hnswlib = _Shim()


# ----------------------------------------------------------------------------------------- zca
def cmd_zca(a):
    from elsa.transform import fit_transform, mean_norm
    t0 = time.time()
    X = np.load(VEC_RAW, mmap_mode="r")
    n, d = X.shape
    log("fit soft_zca alpha 0.1 on", X.shape)
    tr = fit_transform("soft_zca", X, alpha=0.1)              # shipped defaults: sample 200,000, seed 17
    VEC_ZCA.parent.mkdir(parents=True, exist_ok=True)
    tr.save(VEC_ZCA.parent / "soft_zca_a0.1.npz")
    tmp = VEC_ZCA.with_name(VEC_ZCA.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, d))
    step = 200_000
    for s in range(0, n, step):
        e = min(n, s + step)
        out[s:e] = tr.apply(np.asarray(X[s:e])).astype(np.float32)
    out.flush()
    del out
    os.replace(tmp, VEC_ZCA)
    Z = np.load(VEC_ZCA, mmap_mode="r")
    rng = np.random.default_rng(17)
    i, j = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    keep = i != j
    i, j = i[keep], j[keep]
    oi, oj = np.argsort(i), np.argsort(j)
    xi = np.asarray(X[np.sort(i)])[np.argsort(oi)]
    xj = np.asarray(X[np.sort(j)])[np.argsort(oj)]
    zi = np.asarray(Z[np.sort(i)])[np.argsort(oi)]
    zj = np.asarray(Z[np.sort(j)])[np.argsort(oj)]
    rc = np.einsum("ij,ij->i", xi, xj)
    zc = np.einsum("ij,ij->i", zi, zj)
    qs = [0.5, 0.9, 0.99, 0.999]
    samp = np.sort(rng.choice(n, 100_000, replace=False))
    res = dict(n=n, dim=d, transform=tr.describe(), fit_sample=200_000, fit_seed=17,
               mean_norm_raw=mean_norm(np.asarray(X[samp])), mean_norm_zca=mean_norm(np.asarray(Z[samp])),
               random_pair_cosine_raw={str(q): float(np.quantile(rc, q)) for q in qs},
               random_pair_cosine_zca={str(q): float(np.quantile(zc, q)) for q in qs},
               random_pairs=int(len(rc)), seconds=round(time.time() - t0, 1), maxrss_gb=round(maxrss_gb(), 2),
               src=src_hashes(), output=str(VEC_ZCA))
    write_json(VEC_ZCA.parent / "soft_zca_a0.1.json", res)
    log(res)


# --------------------------------------------------------------------------------------- chain
def cmd_chain(a):
    T = threads()
    patch_hnsw_threads(T)
    from elsa.analyze import pipeline as pl
    out = run_dir(a.arm)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = universe()
    gdf = genes_for_pipeline(g)
    del g
    vec = ARMS[a.arm]["vectors"]
    emb = np.load(vec) if not SUBSET else np.array(np.load(vec, mmap_mode="r")[:len(gdf)])
    # writable, in memory (as the CLI hands it over)
    assert emb.shape == (len(gdf), 960) and emb.dtype == np.float32
    assert np.isfinite(emb[::997]).all()
    t_load = time.time() - t0
    cfg = config(a.arm, T)
    log("config", cfg)
    t1 = time.time()
    blocks = pl._run_gene_chaining(
        gdf, emb, hnsw_k=cfg.hnsw_k, similarity_threshold=cfg.similarity_threshold, max_gap=cfg.max_gap_genes,
        min_chain_size=cfg.min_chain_size, gap_penalty_scale=cfg.gap_penalty_scale, hnsw_m=cfg.hnsw_m,
        hnsw_ef_construction=cfg.hnsw_ef_construction, hnsw_ef_search=cfg.hnsw_ef_search,
        index_backend=cfg.index_backend, prebuilt_index=None, faiss_nprobe=cfg.faiss_nprobe, _presorted=True,
        checkpoint_dir=out, n_jobs=cfg.n_jobs, extend_anchors=cfg.extend_anchors,
        anchor_extend_radius=cfg.anchor_extend_radius, anchor_extend_threshold=cfg.anchor_extend_threshold)
    t_chain = time.time() - t1
    t2 = time.time()
    pl._save_blocks_checkpoint(blocks, out / "blocks_checkpoint.parquet")
    t_save = time.time() - t2
    n_anchor_rows = None
    try:
        import pyarrow.parquet as pq
        n_anchor_rows = pq.ParquetFile(out / "anchors.parquet").metadata.num_rows
    except Exception as e:                                      # noqa: BLE001
        log("anchor row count failed", e)
    na = blocks.n_anchors.to_numpy()
    res = dict(arm=a.arm, vectors=str(vec), threshold=cfg.similarity_threshold, transform=cfg.embedding_transform,
               config=cfg.__dict__, threads=T, genes=int(len(gdf)), anchors=n_anchor_rows,
               blocks=int(len(blocks)), anchors_in_blocks=int(na.sum()),
               blocks_by_size={str(k): int((np.minimum(na, 6) == k).sum()) for k in range(2, 7)},
               seconds=dict(load=round(t_load, 1), knn_and_chain=round(t_chain, 1), checkpoint=round(t_save, 1),
                            total=round(time.time() - t0, 1)),
               maxrss_gb=round(maxrss_gb(), 2), src=src_hashes())
    write_json(out / "chain_summary.json", res)
    log(res)


# ---------------------------------------------------------------------------------------- perm
def perm_positions(g, seed):
    """Uniform random permutation of positions within each contig (gid order is contig-contiguous)."""
    n = len(g)
    cc = g.contig_code.to_numpy()
    first = np.r_[0, np.flatnonzero(np.diff(cc)) + 1]
    first_of_gene = np.repeat(first, np.diff(np.r_[first, n]))
    if seed == "identity":
        return g.position_index.to_numpy().astype(np.int64)
    rng = np.random.default_rng(int(seed))
    order = np.lexsort((rng.random(n), cc))
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n, dtype=np.int64) - first_of_gene[order]
    return pos


def load_anchors(out, g):
    """Saved anchors with categories restored to exactly the run's (sorted unique over the whole gene
    table, ordered), so codes, canonical swaps and group order equal the observed run."""
    cols = ["query_idx", "target_idx", "query_genome", "target_genome", "query_contig", "target_contig",
            "similarity", "orientation"]
    A = pd.read_parquet(out / "anchors.parquet", columns=cols)
    full = {"genome": np.unique(g.genome.to_numpy()), "contig": np.unique(g.contig.to_numpy())}
    for c in ("query_genome", "target_genome", "query_contig", "target_contig"):
        if not isinstance(A[c].dtype, pd.CategoricalDtype):
            A[c] = A[c].astype("category")
        cats = full[c.split("_")[1]]
        assert set(A[c].cat.categories) <= set(cats)
        A[c] = A[c].cat.set_categories(cats, ordered=True)
    return A


def cmd_perm(a):
    T = threads()
    from elsa.seed import group_anchors_by_contig_pair
    from elsa.chain import chain_groups_batched
    out = run_dir(a.arm)
    t0 = time.time()
    g = universe()
    contigs = pd.read_parquet(PKG / "universe/contigs.parquet")
    assert contigs.contig.is_unique
    cinfo = contigs.set_index("contig")
    A = load_anchors(out, g)
    log("anchors", len(A), f"{time.time() - t0:.1f}s")
    for side in ("query", "target"):
        cats = A[f"{side}_contig"].cat.categories.to_numpy()
        fg = cinfo.first_gid.reindex(cats).to_numpy()
        ccode = cinfo.contig_code.reindex(cats).to_numpy()
        assert not np.isnan(fg).any(), "anchor contig absent from the universe"
        codes = A[f"{side}_contig"].cat.codes.to_numpy()
        gid = fg.astype(np.int64)[codes] + A[f"{side}_idx"].to_numpy(np.int64)
        # every anchor gene must fall inside its own contig
        assert np.array_equal(g.contig_code.to_numpy()[gid], ccode.astype(np.int64)[codes])
        A[f"{side}_gene_id"] = gid
        del codes, gid
    pos = perm_positions(g, a.seed)
    for side in ("query", "target"):
        A[f"{side}_idx"] = pos[A[f"{side}_gene_id"].to_numpy()]
    log("positions assigned", a.seed, f"{time.time() - t0:.1f}s")
    t1 = time.time()
    grouped = group_anchors_by_contig_pair(A)
    del A
    blocks = chain_groups_batched(grouped, max_gap=2, min_size=2, gap_penalty_scale=0.0, extract_blocks=True)
    t_chain = time.time() - t1
    keep = ["query_genome", "target_genome", "query_contig", "target_contig", "query_start", "query_end",
            "target_start", "target_end", "n_anchors", "chain_score", "orientation"]
    if a.seed == "identity":
        keep += ["anchor_query_gene_ids", "anchor_target_gene_ids"]
    blocks = blocks[keep]
    for c in ("query_genome", "target_genome", "query_contig", "target_contig"):
        blocks[c] = blocks[c].astype(str)
    tag = f"perm_{a.seed}"
    blocks.to_parquet(out / f"{tag}_blocks.parquet", index=False)
    na = blocks.n_anchors.to_numpy()
    res = dict(arm=a.arm, seed=a.seed, threads=T, blocks=int(len(blocks)),
               blocks_by_size={str(k): int((np.minimum(na, 6) == k).sum()) for k in range(2, 7)},
               seconds=dict(chain=round(t_chain, 1), total=round(time.time() - t0, 1)), maxrss_gb=round(maxrss_gb(), 2))
    if a.seed == "identity":
        obs = pd.read_parquet(out / "blocks_checkpoint.parquet",
                              columns=keep[:-2])
        key = keep[:9]
        x = obs[key].astype({c: str for c in key[:4]}).sort_values(key).reset_index(drop=True)
        y = blocks[key].sort_values(key).reset_index(drop=True)
        res["replay"] = dict(observed_blocks=int(len(obs)), equal_blocks=bool(len(x) == len(y) and x.equals(y)),
                             max_abs_score_diff=float(np.abs(np.sort(obs.chain_score.to_numpy()) -
                                                             np.sort(blocks.chain_score.to_numpy())).max())
                             if len(obs) == len(blocks) else None)
    write_json(out / f"{tag}_summary.json", res)
    log(res)


# ------------------------------------------------------------------------------------- cluster
def cmd_cluster(a):
    T = threads()
    from elsa.analyze import pipeline as pl
    out = run_dir(a.arm)
    assert (out / "blocks_checkpoint.parquet").exists()
    t0 = time.time()
    g = universe()
    gdf = genes_for_pipeline(g)
    del g
    cfg = config(a.arm, T)
    dummy = np.zeros((len(gdf), 1), dtype=np.float32)          # unused: resumes from the blocks checkpoint
    s = pl.run_chain_pipeline(genes_df=gdf, output_dir=out, config=cfg, embeddings=dummy, run_schema=False)
    res = dict(arm=a.arm, threads=T, summary=s.__dict__, seconds=round(time.time() - t0, 1),
               maxrss_gb=round(maxrss_gb(), 2))
    write_json(out / "cluster_summary.json", res)
    log(res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("zca")
    for name in ("chain", "perm", "cluster"):
        p = sub.add_parser(name)
        p.add_argument("--arm", choices=sorted(ARMS), required=True)
        if name == "perm":
            p.add_argument("--seed", required=True)
    a = ap.parse_args()
    dict(zca=cmd_zca, chain=cmd_chain, perm=cmd_perm, cluster=cmd_cluster)[a.cmd](a)


if __name__ == "__main__":
    main()
