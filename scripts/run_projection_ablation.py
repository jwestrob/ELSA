#!/usr/bin/env python3
"""Run the projection ablation benchmark.

Applies each frozen transform to E. coli 480D embeddings, runs the chain
pipeline, and evaluates operon recall.

Usage:
    python scripts/run_projection_ablation.py

Prerequisites:
    - data/frozen_pca/ contains all fitted transforms (from fit_all_projections.py)
    - benchmarks/elsa_output/ecoli/elsa_index_nopca/ingest/genes.parquet (480D E. coli)
    - benchmarks/ground_truth/ecoli_operon_gt_v2.tsv
"""

import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Paths
ECOLI_RAW = Path("benchmarks/elsa_output/ecoli/elsa_index_nopca/ingest/genes.parquet")
TRANSFORMS_DIR = Path("data/frozen_pca")
OUTPUT_BASE = Path("benchmarks/results/projection_ablation")
OPERON_GT = Path("benchmarks/ground_truth/ecoli_operon_gt_v2.tsv")

# Techniques and their transform files
TECHNIQUES = {
    "per_dataset_pca": None,  # baseline — use existing projected embeddings
    "frozen_pca": "pca_model.pkl",
    "frozen_pca_whiten": "pca_whiten_model.pkl",
    "abt_k1": "abt_pca_k1.pkl",
    "abt_k3": "abt_pca_k3.pkl",
    "opq": "opq_transform.faissindex",
    "whiten_raw_480": "whiten_scaler.pkl",
}


def load_raw_ecoli() -> tuple[pd.DataFrame, np.ndarray]:
    """Load E. coli 480D embeddings."""
    df = pd.read_parquet(ECOLI_RAW)
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")])
    matrix = df[emb_cols].values.astype(np.float32)
    meta_cols = [c for c in df.columns if not c.startswith("emb_")]
    return df[meta_cols].copy(), matrix, emb_cols


def apply_transform(name: str, matrix: np.ndarray) -> np.ndarray:
    """Apply a projection transform to raw embeddings."""
    if name == "per_dataset_pca":
        # Fit PCA on this dataset's own embeddings (the current default behavior)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=256, random_state=42)
        pca.fit(matrix[:50000] if len(matrix) > 50000 else matrix)
        console.print(f"  Per-dataset PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        return pca.transform(matrix)

    transform_file = TRANSFORMS_DIR / TECHNIQUES[name]

    if name == "frozen_pca":
        with open(transform_file, "rb") as f:
            pca = pickle.load(f)
        return pca.transform(matrix)

    elif name == "frozen_pca_whiten":
        with open(transform_file, "rb") as f:
            pca = pickle.load(f)
        return pca.transform(matrix)

    elif name.startswith("abt_"):
        with open(transform_file, "rb") as f:
            abt = pickle.load(f)
        centered = matrix - abt["mean"]
        proj = centered @ abt["top_components"].T
        residual = centered - proj @ abt["top_components"]
        return abt["pca"].transform(residual)

    elif name == "opq":
        opq = faiss.read_VectorTransform(str(transform_file))
        return opq.apply(matrix.copy())

    elif name == "whiten_raw_480":
        with open(transform_file, "rb") as f:
            scaler = pickle.load(f)
        return scaler.transform(matrix)

    else:
        raise ValueError(f"Unknown technique: {name}")


def save_genes_parquet(meta_df: pd.DataFrame, projected: np.ndarray,
                       output_dir: Path, l2_normalize: bool = True):
    """Save projected embeddings as genes.parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if l2_normalize:
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / (norms + 1e-8)

    df = meta_df.copy()
    for i in range(projected.shape[1]):
        df[f"emb_{i:03d}"] = projected[:, i].astype(np.float16)

    out_path = output_dir / "genes.parquet"
    df.to_parquet(out_path, compression="snappy", index=False)
    console.print(f"  Saved {len(df):,} genes ({projected.shape[1]}D) -> {out_path}")
    return out_path


def run_chain_pipeline(genes_path: Path, output_dir: Path) -> Path:
    """Run the micro chain pipeline."""
    from elsa.analyze.micro_chain import run_micro_chain_pipeline, MicroChainConfig

    config = MicroChainConfig(
        index_backend="faiss_ivfflat",
        faiss_nprobe=32,
        hnsw_k=50,
        similarity_threshold=0.85,
        max_gap_genes=2,
        min_chain_size=2,
        jaccard_tau=0.3,
        mutual_k=5,
        df_max=500,
        min_genome_support=2,
    )

    chain_dir = output_dir / "micro_chain"
    summary = run_micro_chain_pipeline(genes_path, chain_dir, config=config)

    console.print(
        f"  blocks={summary.num_blocks} clusters={summary.num_clusters} "
        f"mean_size={summary.mean_block_size:.1f}"
    )
    return chain_dir / "micro_chain_blocks.csv"


def evaluate_operon_recall(blocks_path: Path, output_dir: Path) -> dict:
    """Evaluate operon recall."""
    sys.path.insert(0, str(Path("benchmarks/scripts")))
    from evaluate_operon_recall import load_operon_gt, load_elsa_blocks, evaluate_elsa

    operon_gt = load_operon_gt(OPERON_GT)
    elsa_blocks = load_elsa_blocks(blocks_path)

    gt_genomes = set(operon_gt["genome_a"].unique()) | set(operon_gt["genome_b"].unique())
    elsa_ecoli = elsa_blocks[
        (elsa_blocks["query_genome"].isin(gt_genomes))
        & (elsa_blocks["target_genome"].isin(gt_genomes))
    ]

    results = evaluate_elsa(elsa_ecoli, operon_gt, threshold=0.5)

    # Save per-technique CSV
    csv_path = output_dir / "operon_recall.csv"
    pd.DataFrame([{
        "strict_recall": results["strict_recall"],
        "independent_recall": results["independent_recall"],
        "any_recall": results["any_recall"],
        "strict_found": results["strict_found"],
        "independent_found": results["independent_found"],
        "any_found": results["any_found"],
        "total": results["total_operons"],
    }]).to_csv(csv_path, index=False)

    return results


def main():
    console.print("[bold]Projection Ablation Benchmark[/bold]")
    console.print(f"E. coli raw embeddings: {ECOLI_RAW}")
    console.print(f"Operon ground truth: {OPERON_GT}\n")

    # Load raw embeddings once
    meta_df, raw_matrix, _ = load_raw_ecoli()
    console.print(f"Loaded {len(meta_df):,} proteins, {raw_matrix.shape[1]}D\n")

    all_results = {}

    for name in TECHNIQUES:
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]Technique: {name}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")

        output_dir = OUTPUT_BASE / name

        # Step 1: Apply transform
        console.print("  Applying transform...")
        projected = apply_transform(name, raw_matrix)
        console.print(f"  Output dim: {projected.shape[1]}")

        # Step 2: Save genes.parquet
        genes_path = save_genes_parquet(meta_df, projected, output_dir)

        # Step 3: Run chain pipeline
        console.print("  Running chain pipeline...")
        blocks_path = run_chain_pipeline(genes_path, output_dir)

        # Step 4: Evaluate operon recall
        console.print("  Evaluating operon recall...")
        results = evaluate_operon_recall(blocks_path, output_dir)

        all_results[name] = {
            "dim": projected.shape[1],
            "strict": results["strict_recall"],
            "independent": results["independent_recall"],
            "any_cov": results["any_recall"],
            "strict_n": results["strict_found"],
            "indep_n": results["independent_found"],
            "any_n": results["any_found"],
            "total": results["total_operons"],
        }

        console.print(
            f"  [green]Strict: {results['strict_recall']:.1%}  "
            f"Independent: {results['independent_recall']:.1%}  "
            f"Any: {results['any_recall']:.1%}[/green]"
        )

    # Summary table
    console.print(f"\n\n[bold]{'='*70}[/bold]")
    console.print("[bold]PROJECTION ABLATION RESULTS[/bold]")
    console.print(f"[bold]{'='*70}[/bold]\n")

    table = Table(title="Operon Recall by Projection Technique")
    table.add_column("Technique", style="cyan")
    table.add_column("Dim", justify="right")
    table.add_column("Strict", justify="right")
    table.add_column("Independent", justify="right")
    table.add_column("Any Coverage", justify="right")

    for name, r in all_results.items():
        table.add_row(
            name,
            str(r["dim"]),
            f"{r['strict']:.1%}",
            f"{r['independent']:.1%}",
            f"{r['any_cov']:.1%}",
        )
    console.print(table)

    # Save summary CSV
    summary_path = OUTPUT_BASE / "ablation_summary.csv"
    pd.DataFrame([
        {"technique": name, **vals} for name, vals in all_results.items()
    ]).to_csv(summary_path, index=False)
    console.print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
