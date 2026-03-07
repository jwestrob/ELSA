#!/usr/bin/env python3
"""
Benchmark FAISS / HNSW / sklearn index backends for ELSA.

Compares recall@k, index build time, query time, and memory usage
across all supported backends.

Usage:
    python benchmarks/scripts/benchmark_index_backends.py \
        --genes benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet \
        --n-queries 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from elsa.index import build_gene_index, HNSWLIB_AVAILABLE, FAISS_AVAILABLE


def get_index_memory(index_type: str, index) -> int:
    """Estimate index memory in bytes."""
    if index_type == "faiss":
        import faiss
        try:
            # Write index to a temp buffer to measure size
            writer = faiss.VectorIOWriter()
            faiss.write_index(index, writer)
            return writer.data.size()
        except Exception:
            return 0
    elif index_type == "hnsw":
        # hnswlib doesn't expose memory directly; estimate from params
        # Each element: dim*4 bytes (float32) + M*2*8 bytes (graph edges)
        n = index.get_current_count()
        dim = index.dim
        m = index.M
        return n * (dim * 4 + m * 2 * 8)
    else:
        return 0


def compute_recall(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute recall@k: fraction of ground-truth neighbors recovered."""
    n, k = ground_truth.shape
    hits = 0
    total = 0
    for i in range(n):
        gt_set = set(ground_truth[i])
        gt_set.discard(-1)
        pred_set = set(predictions[i])
        pred_set.discard(-1)
        if gt_set:
            hits += len(gt_set & pred_set)
            total += len(gt_set)
    return hits / total if total > 0 else 0.0


def run_benchmark(
    embeddings: np.ndarray,
    n_queries: int = 1000,
    k: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Run recall/speed/memory benchmark across backends."""
    n, dim = embeddings.shape
    rng = np.random.RandomState(seed)
    query_indices = rng.choice(n, size=min(n_queries, n), replace=False)

    # Normalize once (same as build_gene_index does internally)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = (embeddings / (norms + 1e-9)).astype(np.float32)
    query_vecs = normalized[query_indices]

    k_query = min(k + 10, n)

    # --- Ground truth: brute-force exact search ---
    print("Computing ground truth (brute-force inner product)...", flush=True)
    if FAISS_AVAILABLE:
        import faiss
        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(np.ascontiguousarray(normalized))
        gt_sims, gt_labels = flat_index.search(np.ascontiguousarray(query_vecs), k_query)
    else:
        # numpy fallback
        gt_sims = query_vecs @ normalized.T
        gt_labels = np.argsort(-gt_sims, axis=1)[:, :k_query]

    # --- Define backends to test ---
    backends = []

    if HNSWLIB_AVAILABLE:
        for ef in [128, 256, 512]:
            backends.append((f"hnsw (M=32, ef={ef})", "hnsw", {"ef_search": ef}))

    if FAISS_AVAILABLE:
        backends.append(("faiss_flat", "faiss_flat", {}))
        for nprobe in [8, 16, 32, 64, 128]:
            backends.append((f"faiss_ivfflat (nprobe={nprobe})", "faiss_ivfflat", {"faiss_nprobe": nprobe}))
        for nprobe in [8, 16, 32, 64]:
            backends.append((f"faiss_ivfpq (nprobe={nprobe})", "faiss_ivfpq", {"faiss_nprobe": nprobe}))
            backends.append((f"faiss_ivfsq (nprobe={nprobe})", "faiss_ivfsq", {"faiss_nprobe": nprobe}))

    backends.append(("sklearn (brute)", "sklearn", {}))

    results = []
    for name, backend, extra_kwargs in backends:
        print(f"\n--- {name} ---", flush=True)

        # Build
        t0 = time.perf_counter()
        try:
            idx_type, idx = build_gene_index(
                embeddings,
                index_backend=backend,
                **extra_kwargs,
            )
        except (ImportError, ValueError) as e:
            print(f"  SKIPPED: {e}")
            continue
        build_time = time.perf_counter() - t0
        print(f"  Build: {build_time:.2f}s", flush=True)

        # Memory
        mem_bytes = get_index_memory(idx_type, idx)
        mem_mb = mem_bytes / (1024 * 1024)
        print(f"  Memory: {mem_mb:.1f} MB", flush=True)

        # Query
        t0 = time.perf_counter()
        if idx_type == "hnsw":
            labels, distances = idx.knn_query(query_vecs, k=k_query)
        elif idx_type == "faiss":
            _, labels = idx.search(np.ascontiguousarray(query_vecs), k_query)
            labels = labels.astype(np.intp)
        else:  # sklearn
            _, labels = idx.kneighbors(query_vecs, n_neighbors=k_query)
        query_time = time.perf_counter() - t0
        print(f"  Query ({len(query_indices)} queries): {query_time:.3f}s", flush=True)

        # Recall
        recall = compute_recall(gt_labels, labels)
        print(f"  Recall@{k_query}: {recall:.4f}", flush=True)

        results.append({
            "backend": name,
            "build_time_s": round(build_time, 3),
            "query_time_s": round(query_time, 4),
            "memory_mb": round(mem_mb, 1),
            f"recall@{k_query}": round(recall, 4),
            "n_genes": n,
            "dim": dim,
            "n_queries": len(query_indices),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ELSA index backends")
    parser.add_argument("--genes", required=True, type=Path,
                        help="Path to genes.parquet")
    parser.add_argument("--n-queries", type=int, default=1000,
                        help="Number of random query genes (default: 1000)")
    parser.add_argument("--k", type=int, default=50,
                        help="Number of neighbors (default: 50)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: print table)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if not args.genes.exists():
        print(f"ERROR: genes.parquet not found: {args.genes}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embeddings from {args.genes}...")
    df = pd.read_parquet(args.genes)
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")])
    embeddings = df[emb_cols].values.astype(np.float32)
    print(f"Loaded {embeddings.shape[0]} genes, dim={embeddings.shape[1]}")

    print(f"\nBackend availability: hnswlib={HNSWLIB_AVAILABLE}, faiss={FAISS_AVAILABLE}")

    results_df = run_benchmark(
        embeddings,
        n_queries=args.n_queries,
        k=args.k,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Index Backend Benchmark")
        for col in results_df.columns:
            table.add_column(col, justify="right" if col != "backend" else "left")
        for _, row in results_df.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)
    except ImportError:
        print(results_df.to_string(index=False))

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
