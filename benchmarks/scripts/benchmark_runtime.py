#!/usr/bin/env python3
"""
Time each pipeline stage and report memory usage.

Instruments: index build, anchor seeding, chaining + extraction, clustering.
Runs on both the 20-genome E. coli and 30-genome cross-species datasets.

Usage:
    python benchmarks/scripts/benchmark_runtime.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from elsa.index import build_gene_index
from elsa.seed import find_cross_genome_anchors, group_anchors_by_contig_pair
from elsa.chain import chain_anchors_lis, extract_nonoverlapping_chains
from elsa.cluster import cluster_blocks_by_overlap


def get_rss_mb() -> float:
    """Current resident memory in MB (Linux/macOS)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on Linux
    except Exception:
        return 0.0


def benchmark_dataset(label: str, genes_path: Path) -> dict:
    """Time each pipeline stage on one dataset."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[Runtime] {label}: {genes_path}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    rss_start = get_rss_mb()
    t_total_start = time.time()

    # 1. Load
    t0 = time.time()
    genes_df = pd.read_parquet(genes_path)
    genes_df = genes_df.sort_values(["sample_id", "contig_id", "start", "end"])
    genes_df["position_index"] = genes_df.groupby(["sample_id", "contig_id"]).cumcount()
    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    embeddings = genes_df[emb_cols].values.astype(np.float32)
    info_cols = ["gene_id", "sample_id", "contig_id", "position_index"]
    if "strand" in genes_df.columns:
        info_cols.append("strand")
    gene_info = genes_df[info_cols].reset_index(drop=True)
    t_load = time.time() - t0

    n_genes = len(genes_df)
    n_genomes = genes_df["sample_id"].nunique()
    dim = len(emb_cols)
    rss_after_load = get_rss_mb()

    print(f"[Runtime] Loaded {n_genes:,} genes, {n_genomes} genomes, dim={dim} ({t_load:.1f}s)",
          file=sys.stderr, flush=True)

    # 2. Index build
    t0 = time.time()
    index_tuple = build_gene_index(
        embeddings, index_backend="faiss_ivfflat", faiss_nprobe=32,
    )
    t_index = time.time() - t0
    rss_after_index = get_rss_mb()
    print(f"[Runtime] Index built ({t_index:.1f}s)", file=sys.stderr, flush=True)

    # 3. Anchor seeding
    t0 = time.time()
    anchors = find_cross_genome_anchors(
        index_tuple, embeddings, gene_info,
        k=50, similarity_threshold=0.85,
    )
    t_seed = time.time() - t0
    rss_after_seed = get_rss_mb()
    print(f"[Runtime] Seeded {len(anchors):,} anchors ({t_seed:.1f}s)", file=sys.stderr, flush=True)

    # 4. Group + chain + extract
    t0 = time.time()
    groups = group_anchors_by_contig_pair(anchors)
    all_blocks = []
    block_id = 0
    n_chains = 0
    for key, group_anchors in groups.items():
        if len(group_anchors) < 2:
            continue
        chains = chain_anchors_lis(group_anchors, max_gap=2, min_size=2, gap_penalty_scale=0.0)
        if not chains:
            continue
        blocks = extract_nonoverlapping_chains(chains, block_id_start=block_id)
        all_blocks.extend(blocks)
        block_id += len(blocks)
        n_chains += len(chains)
    t_chain = time.time() - t0
    rss_after_chain = get_rss_mb()
    print(f"[Runtime] Chained → {len(all_blocks):,} blocks from {n_chains:,} chains ({t_chain:.1f}s)",
          file=sys.stderr, flush=True)

    # 5. Clustering
    t0 = time.time()
    block_to_cluster, clusters_df = cluster_blocks_by_overlap(
        all_blocks, jaccard_tau=0.3, mutual_k=5, min_genome_support=2,
    )
    t_cluster = time.time() - t0
    rss_after_cluster = get_rss_mb()
    print(f"[Runtime] Clustered → {len(clusters_df):,} clusters ({t_cluster:.1f}s)",
          file=sys.stderr, flush=True)

    t_total = time.time() - t_total_start

    return {
        "label": label,
        "n_genomes": n_genomes,
        "n_genes": n_genes,
        "dim": dim,
        "n_anchors": len(anchors),
        "n_blocks": len(all_blocks),
        "n_clusters": len(clusters_df),
        "timings": {
            "load_s": round(t_load, 1),
            "index_s": round(t_index, 1),
            "seed_s": round(t_seed, 1),
            "chain_s": round(t_chain, 1),
            "cluster_s": round(t_cluster, 1),
            "total_s": round(t_total, 1),
        },
        "memory_mb": {
            "start": round(rss_start, 0),
            "after_load": round(rss_after_load, 0),
            "after_index": round(rss_after_index, 0),
            "after_seed": round(rss_after_seed, 0),
            "after_chain": round(rss_after_chain, 0),
            "after_cluster": round(rss_after_cluster, 0),
            "peak_delta": round(rss_after_cluster - rss_start, 0),
        },
    }


def main():
    output_dir = PROJECT_ROOT / "benchmarks" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("E. coli (20 genomes, 90k genes)",
         PROJECT_ROOT / "benchmarks" / "elsa_output" / "ecoli" / "elsa_index" / "ingest" / "genes.parquet"),
        ("Enterobacteriaceae (30 genomes, 143k genes)",
         PROJECT_ROOT / "benchmarks" / "elsa_output" / "cross_species" / "ingest" / "genes.parquet"),
    ]

    results = []
    for label, path in datasets:
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        r = benchmark_dataset(label, path)
        results.append(r)

    # Write JSON
    json_path = output_dir / "runtime_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Runtime] Wrote {json_path}", file=sys.stderr)

    # Write markdown
    md_path = output_dir / "runtime_benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write("# Runtime and Resource Usage\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d')}\n")
        f.write("**System**: Linux, 32 GB RAM, AMD CPU\n")
        f.write("**Index backend**: FAISS IVFFlat (nprobe=32)\n\n")

        f.write("## Per-Stage Timing\n\n")
        f.write("| Stage | ")
        f.write(" | ".join(r["label"] for r in results))
        f.write(" |\n")
        f.write("|-------|" + "|".join("------" for _ in results) + "|\n")

        stages = [
            ("Load parquet", "load_s"),
            ("Build index", "index_s"),
            ("Anchor seeding (kNN)", "seed_s"),
            ("Chain + extract", "chain_s"),
            ("Cluster", "cluster_s"),
            ("**Total**", "total_s"),
        ]
        for label, key in stages:
            f.write(f"| {label} |")
            for r in results:
                f.write(f" {r['timings'][key]:.1f}s |")
            f.write("\n")

        f.write("\n## Dataset Summary\n\n")
        f.write("| Metric | ")
        f.write(" | ".join(r["label"] for r in results))
        f.write(" |\n")
        f.write("|--------|" + "|".join("------" for _ in results) + "|\n")
        for metric, key in [
            ("Genomes", "n_genomes"), ("Genes", "n_genes"),
            ("Embedding dim", "dim"),
            ("Cross-genome anchors", "n_anchors"),
            ("Syntenic blocks", "n_blocks"),
            ("Clusters", "n_clusters"),
        ]:
            f.write(f"| {metric} |")
            for r in results:
                f.write(f" {r[key]:,} |")
            f.write("\n")

        f.write("\n## Peak Memory (RSS delta)\n\n")
        for r in results:
            f.write(f"- **{r['label']}**: {r['memory_mb']['peak_delta']:.0f} MB\n")

    print(f"[Runtime] Wrote {md_path}", file=sys.stderr)

    # Console summary
    print("\n=== Runtime Summary ===")
    for r in results:
        t = r["timings"]
        print(f"\n{r['label']}:")
        print(f"  Load: {t['load_s']}s | Index: {t['index_s']}s | Seed: {t['seed_s']}s | "
              f"Chain: {t['chain_s']}s | Cluster: {t['cluster_s']}s | Total: {t['total_s']}s")


if __name__ == "__main__":
    main()
