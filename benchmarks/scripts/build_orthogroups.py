#!/usr/bin/env python3
"""
Phase 1: Assign genes to orthogroups based on embedding similarity.

Uses Leiden clustering on a kNN graph of gene embeddings to group
genes into orthogroups (gene families).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def load_gene_embeddings(genes_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load genes.parquet and extract embeddings."""
    df = pd.read_parquet(genes_path)

    # Find embedding columns
    emb_cols = sorted([c for c in df.columns if c.startswith('emb_')])
    if not emb_cols:
        raise ValueError("No embedding columns found")

    embeddings = df[emb_cols].values.astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    print(f"Loaded {len(df)} genes with {len(emb_cols)}-dim embeddings")
    print(f"Genomes: {df['sample_id'].nunique()}")

    return df, embeddings


def build_knn_graph(embeddings: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Build k-nearest neighbor graph using HNSW."""
    try:
        import hnswlib

        n, dim = embeddings.shape
        print(f"Building HNSW index for {n} genes...")

        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=32)
        index.add_items(embeddings, np.arange(n))
        index.set_ef(max(k * 2, 100))

        print(f"Querying {k} nearest neighbors...")
        labels, distances = index.knn_query(embeddings, k=k+1)  # +1 to exclude self

        # Convert cosine distance to similarity
        similarities = 1 - distances

        # Exclude self (first neighbor)
        return labels[:, 1:], similarities[:, 1:]

    except ImportError:
        print("hnswlib not available, using sklearn...")
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
        nn.fit(embeddings)
        distances, labels = nn.kneighbors(embeddings)
        similarities = 1 - distances

        return labels[:, 1:], similarities[:, 1:]


def cluster_leiden(
    n_nodes: int,
    knn_indices: np.ndarray,
    knn_similarities: np.ndarray,
    similarity_threshold: float = 0.85,
    resolution: float = 1.0,
) -> np.ndarray:
    """Cluster genes using Leiden algorithm on thresholded kNN graph."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError("Please install igraph and leidenalg: pip install igraph leidenalg")

    print(f"Building graph with similarity threshold {similarity_threshold}...")

    # Build edge list with weights
    edges = []
    weights = []

    for i in range(n_nodes):
        for j_idx, sim in zip(knn_indices[i], knn_similarities[i]):
            if sim >= similarity_threshold and i < j_idx:  # Avoid duplicates
                edges.append((i, int(j_idx)))
                weights.append(float(sim))

    print(f"Graph has {len(edges)} edges (threshold={similarity_threshold})")

    if len(edges) == 0:
        print("WARNING: No edges above threshold! Returning singletons.")
        return np.arange(n_nodes)

    # Create graph
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es['weight'] = weights

    # Run Leiden clustering
    print(f"Running Leiden clustering (resolution={resolution})...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=42
    )

    labels = np.array(partition.membership)
    n_clusters = len(set(labels))
    print(f"Found {n_clusters} orthogroups")

    return labels


def analyze_orthogroups(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Analyze orthogroup statistics."""
    df = df.copy()
    df['orthogroup'] = [f"OG_{l:05d}" for l in labels]

    # Size distribution
    og_sizes = df.groupby('orthogroup').size()

    # Genome coverage per OG
    og_genomes = df.groupby('orthogroup')['sample_id'].nunique()

    stats = {
        'n_orthogroups': len(og_sizes),
        'n_genes': len(df),
        'n_genomes': df['sample_id'].nunique(),
        'og_size_mean': float(og_sizes.mean()),
        'og_size_median': float(og_sizes.median()),
        'og_size_max': int(og_sizes.max()),
        'og_genome_coverage_mean': float(og_genomes.mean()),
        'singletons': int((og_sizes == 1).sum()),
        'core_ogs': int((og_genomes == df['sample_id'].nunique()).sum()),
    }

    print(f"\nOrthogroup Statistics:")
    print(f"  Total orthogroups: {stats['n_orthogroups']}")
    print(f"  Mean size: {stats['og_size_mean']:.1f} genes")
    print(f"  Median size: {stats['og_size_median']:.0f} genes")
    print(f"  Max size: {stats['og_size_max']} genes")
    print(f"  Singletons: {stats['singletons']} ({stats['singletons']/stats['n_orthogroups']*100:.1f}%)")
    print(f"  Core OGs (in all genomes): {stats['core_ogs']}")
    print(f"  Mean genome coverage: {stats['og_genome_coverage_mean']:.1f} genomes")

    return stats


def build_orthogroups(
    genes_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.85,
    k_neighbors: int = 30,
    resolution: float = 1.0,
) -> pd.DataFrame:
    """Build orthogroups from gene embeddings."""

    # Load data
    df, embeddings = load_gene_embeddings(genes_path)

    # Build kNN graph
    knn_indices, knn_similarities = build_knn_graph(embeddings, k=k_neighbors)

    # Cluster
    labels = cluster_leiden(
        len(df), knn_indices, knn_similarities,
        similarity_threshold=similarity_threshold,
        resolution=resolution,
    )

    # Add orthogroup labels to dataframe
    df['orthogroup'] = [f"OG_{l:05d}" for l in labels]

    # Analyze
    stats = analyze_orthogroups(df, labels)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save orthogroup assignments
    out_df = df[['gene_id', 'sample_id', 'contig_id', 'start', 'end', 'strand', 'orthogroup']].copy()
    out_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nWrote {output_path}")

    # Save stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")

    return out_df


def main():
    parser = argparse.ArgumentParser(description="Build orthogroups from gene embeddings")
    parser.add_argument("genes_parquet", type=Path, help="Path to genes.parquet")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output TSV path")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for same orthogroup (default: 0.85)")
    parser.add_argument("--k-neighbors", type=int, default=30,
                        help="Number of neighbors for kNN graph (default: 30)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Leiden resolution parameter (default: 1.0)")

    args = parser.parse_args()

    build_orthogroups(
        args.genes_parquet,
        args.output,
        similarity_threshold=args.similarity_threshold,
        k_neighbors=args.k_neighbors,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
