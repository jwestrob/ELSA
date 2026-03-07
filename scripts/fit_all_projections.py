#!/usr/bin/env python3
"""Fit all projection techniques from UniRef50 raw embeddings.

Run this AFTER fit_uniref50_pca.py finishes embedding:

    python scripts/fit_all_projections.py \
        --raw data/frozen_pca/uniref50_raw.parquet \
        --output data/frozen_pca/ \
        --dim 256

Fits five techniques:
  1. pca              — Standard PCA (480D -> 256D)
  2. pca_whiten       — PCA with whitening
  3. abt_k1           — All-but-the-top (remove PC1) + PCA
  4. abt_k3           — All-but-the-top (remove PC1-3) + PCA
  5. opq              — FAISS OPQ rotation (optimized for retrieval)
  6. whiten_raw       — Whitened 480D (no reduction, baseline)
"""

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

console = Console()


def load_raw_matrix(raw_path: Path) -> np.ndarray:
    """Load raw embedding matrix from parquet."""
    pf = pq.ParquetFile(raw_path)
    raw_cols = sorted([c for c in pf.schema.names if c.startswith("raw_")])
    table = pf.read(columns=raw_cols)
    matrix = np.column_stack(
        [col.to_numpy(zero_copy_only=False) for col in table.columns]
    ).astype(np.float32)
    del table
    console.print(f"Loaded {matrix.shape[0]:,} embeddings, dim={matrix.shape[1]}")
    return matrix


def subsample(matrix: np.ndarray, n: int) -> np.ndarray:
    if matrix.shape[0] <= n:
        return matrix
    rng = np.random.default_rng(42)
    return matrix[rng.choice(matrix.shape[0], n, replace=False)]


def sanity_check(name: str, matrix: np.ndarray, projected: np.ndarray):
    """Print norm and cosine stats for a projection."""
    norms = np.linalg.norm(projected[:200], axis=1)
    # Mean pairwise cosine on a small sample
    sample = projected[:200]
    sample_normed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8)
    cos_matrix = sample_normed @ sample_normed.T
    # Off-diagonal mean
    mask = ~np.eye(len(sample), dtype=bool)
    mean_cos = cos_matrix[mask].mean()
    console.print(
        f"  [{name}] dim={projected.shape[1]} "
        f"norm: {norms.mean():.3f}+/-{norms.std():.3f}  "
        f"mean_cos: {mean_cos:.4f}"
    )


# ---------------------------------------------------------------------------
# Technique 1: Standard PCA
# ---------------------------------------------------------------------------
def fit_standard_pca(matrix: np.ndarray, dim: int, output_dir: Path):
    console.print("\n[bold]1. Standard PCA[/bold]")
    fit_data = subsample(matrix, 50000)
    pca = PCA(n_components=dim, random_state=42)
    pca.fit(fit_data)
    console.print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    path = output_dir / "pca_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(pca, f)
    console.print(f"  Saved: {path}")

    sanity_check("pca", matrix, pca.transform(matrix[:200]))
    return path


# ---------------------------------------------------------------------------
# Technique 2: PCA + Whitening
# ---------------------------------------------------------------------------
def fit_pca_whiten(matrix: np.ndarray, dim: int, output_dir: Path):
    console.print("\n[bold]2. PCA + Whitening[/bold]")
    fit_data = subsample(matrix, 50000)
    pca = PCA(n_components=dim, whiten=True, random_state=42)
    pca.fit(fit_data)
    console.print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    path = output_dir / "pca_whiten_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(pca, f)
    console.print(f"  Saved: {path}")

    sanity_check("pca_whiten", matrix, pca.transform(matrix[:200]))
    return path


# ---------------------------------------------------------------------------
# Technique 3 & 4: All-but-the-top PCA
# ---------------------------------------------------------------------------
def fit_abt_pca(matrix: np.ndarray, dim: int, k: int, output_dir: Path):
    console.print(f"\n[bold]3. All-but-the-top PCA (k={k})[/bold]")
    fit_data = subsample(matrix, 50000)

    # Step 1: Compute mean and top-k principal components
    mean = fit_data.mean(axis=0)
    centered = fit_data - mean

    pca_full = PCA(n_components=k, random_state=42)
    pca_full.fit(centered)
    top_components = pca_full.components_  # (k, 480)

    # Step 2: Remove top-k components from centered data
    # Projection onto top-k subspace
    projections = centered @ top_components.T  # (N, k)
    residual = centered - projections @ top_components  # (N, 480)

    # Step 3: Fit PCA on the residual
    pca_residual = PCA(n_components=dim, random_state=42)
    pca_residual.fit(residual)
    console.print(f"  Residual explained variance: {pca_residual.explained_variance_ratio_.sum():.4f}")

    # Save all components needed for transform
    abt_data = {
        "mean": mean,
        "top_components": top_components,  # (k, 480) — components to remove
        "pca": pca_residual,               # PCA on residual
        "k": k,
    }

    suffix = f"k{k}"
    path = output_dir / f"abt_pca_{suffix}.pkl"
    with open(path, "wb") as f:
        pickle.dump(abt_data, f)
    console.print(f"  Saved: {path}")

    # Sanity check
    test = matrix[:200]
    test_centered = test - mean
    test_proj = test_centered @ top_components.T
    test_residual = test_centered - test_proj @ top_components
    test_out = pca_residual.transform(test_residual)
    sanity_check(f"abt_k{k}", matrix, test_out)

    return path


# ---------------------------------------------------------------------------
# Technique 5: FAISS OPQ
# ---------------------------------------------------------------------------
def fit_opq(matrix: np.ndarray, dim: int, output_dir: Path):
    console.print(f"\n[bold]4. FAISS OPQ[/bold]")
    fit_data = subsample(matrix, 50000).copy()  # FAISS needs contiguous

    # OPQ requires dim to be divisible by M (number of sub-quantizers)
    # Use M = number of subspaces. dim=256 is divisible by many values.
    # Common choice: M = dim/4 or M = dim/8
    M = min(64, dim)  # sub-quantizers
    nbits = 8         # bits per sub-quantizer

    console.print(f"  Training OPQ: {matrix.shape[1]}D -> {dim}D, M={M}, nbits={nbits}")
    console.print(f"  Training on {len(fit_data):,} vectors...")

    # OPQ trains a rotation matrix that optimizes PQ distortion
    # OPQMatrix does rotation + dimensionality reduction
    opq = faiss.OPQMatrix(matrix.shape[1], M, dim)
    opq.train(fit_data)

    console.print(f"  OPQ trained: is_orthonormal={opq.is_orthonormal}")

    path = output_dir / "opq_transform.pkl"
    # FAISS OPQ can be serialized via pickle or faiss.write_VectorTransform
    faiss_path = output_dir / "opq_transform.faissindex"
    faiss.write_VectorTransform(opq, str(faiss_path))
    console.print(f"  Saved: {faiss_path}")

    # Also save as pickle for compatibility with our pipeline
    with open(path, "wb") as f:
        pickle.dump({"type": "opq", "faiss_path": str(faiss_path), "dim_in": matrix.shape[1], "dim_out": dim}, f)

    # Sanity check
    test = matrix[:200].copy()
    test_out = opq.apply(test)
    sanity_check("opq", matrix, test_out)

    return path


# ---------------------------------------------------------------------------
# Technique 6: Whitened raw (no reduction)
# ---------------------------------------------------------------------------
def fit_whiten_raw(matrix: np.ndarray, output_dir: Path):
    console.print(f"\n[bold]5. Whitened raw (480D baseline)[/bold]")
    fit_data = subsample(matrix, 50000)

    scaler = StandardScaler()
    scaler.fit(fit_data)

    path = output_dir / "whiten_scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    console.print(f"  Saved: {path}")

    sanity_check("whiten_raw", matrix, scaler.transform(matrix[:200]))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit all projection techniques from UniRef50 raw embeddings"
    )
    parser.add_argument("--raw", type=Path, required=True,
                        help="Path to uniref50_raw.parquet")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/frozen_pca"),
                        help="Output directory for all models")
    parser.add_argument("--dim", type=int, default=256,
                        help="Target dimensionality for reduction techniques")
    parser.add_argument("--techniques", nargs="+",
                        default=["pca", "pca_whiten", "abt_k1", "abt_k3", "opq", "whiten_raw"],
                        help="Which techniques to fit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    matrix = load_raw_matrix(args.raw)

    results = {}
    techniques = set(args.techniques)

    if "pca" in techniques:
        results["pca"] = fit_standard_pca(matrix, args.dim, args.output)

    if "pca_whiten" in techniques:
        results["pca_whiten"] = fit_pca_whiten(matrix, args.dim, args.output)

    if "abt_k1" in techniques:
        results["abt_k1"] = fit_abt_pca(matrix, args.dim, k=1, output_dir=args.output)

    if "abt_k3" in techniques:
        results["abt_k3"] = fit_abt_pca(matrix, args.dim, k=3, output_dir=args.output)

    if "opq" in techniques:
        results["opq"] = fit_opq(matrix, args.dim, args.output)

    if "whiten_raw" in techniques:
        results["whiten_raw"] = fit_whiten_raw(matrix, args.output)

    console.print("\n[bold green]All techniques fitted![/bold green]")
    console.print("\nSaved models:")
    for name, path in results.items():
        console.print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
