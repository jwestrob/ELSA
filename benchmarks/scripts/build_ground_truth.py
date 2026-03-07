#!/usr/bin/env python3
"""
Build ground truth conserved syntenic blocks from ELSA embeddings.

Uses HNSW for efficient kNN search instead of brute-force pairwise comparison.

Ground truth definition:
- Proteins are "the same" if cosine similarity > threshold (default 0.9)
- A conserved block is a set of ≥3 adjacent proteins that are "the same"
  across ≥2 genomes
- Adjacency allows up to N intervening genes (default 2)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConservedBlock:
    """A conserved syntenic block found across genomes."""
    block_id: int
    genes_by_genome: dict[str, list[str]] = field(default_factory=dict)
    n_genomes: int = 0
    n_genes: int = 0

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "n_genes": self.n_genes,
            "n_genomes": self.n_genomes,
            "genomes": list(self.genes_by_genome.keys()),
            "genes_by_genome": self.genes_by_genome,
        }


def load_genes_parquet(genes_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load genes.parquet with embeddings."""
    df = pd.read_parquet(genes_path)

    # Find embedding columns
    emb_cols = [c for c in df.columns if c.startswith('emb_') or
                c.replace('.', '').replace('-', '').lstrip('-').isdigit()]

    if not emb_cols:
        # Try numeric columns excluding known non-embedding ones
        exclude = {'start', 'end', 'strand', 'position_index', 'index'}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        emb_cols = [c for c in numeric_cols if c not in exclude and
                    not c.endswith('_id') and not c.endswith('_idx')]

    print(f"Found {len(emb_cols)} embedding dimensions")
    print(f"Loaded {len(df)} genes from {df['sample_id'].nunique()} genomes")

    return df, emb_cols


def compute_gene_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute position index for each gene within its contig."""
    df = df.copy()
    df = df.sort_values(['sample_id', 'contig_id', 'start'])
    df['position_index'] = df.groupby(['sample_id', 'contig_id']).cumcount()
    return df


def build_hnsw_index(embeddings: np.ndarray, ef_construction: int = 200, M: int = 32):
    """Build HNSW index for fast kNN search."""
    try:
        import hnswlib
    except ImportError:
        print("hnswlib not available, falling back to sklearn")
        return None

    n, dim = embeddings.shape

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)

    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(normalized, np.arange(n))
    index.set_ef(128)

    return index


def find_similar_proteins_hnsw(
    embeddings: np.ndarray,
    genome_labels: np.ndarray,
    k: int = 50,
    threshold: float = 0.9,
) -> list[tuple[int, int, float]]:
    """Find similar proteins using HNSW kNN search.

    Only returns cross-genome pairs (same genome pairs filtered out).
    """
    n, dim = embeddings.shape

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)

    # Try HNSW first
    try:
        import hnswlib

        print(f"Building HNSW index for {n} proteins...")
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=32)
        index.add_items(normalized, np.arange(n))
        index.set_ef(max(k * 2, 128))

        print(f"Querying {k} nearest neighbors per protein...")
        labels, distances = index.knn_query(normalized, k=k)

        # Convert cosine distance to similarity
        # hnswlib returns 1 - cos_sim for cosine space
        similarities = 1 - distances

    except ImportError:
        print("hnswlib not available, using sklearn NearestNeighbors...")
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        nn.fit(normalized)
        distances, labels = nn.kneighbors(normalized)
        similarities = 1 - distances

    # Collect cross-genome pairs above threshold
    pairs = []
    seen = set()

    print("Filtering to cross-genome pairs with similarity > threshold...")
    for i in range(n):
        genome_i = genome_labels[i]
        for j_idx in range(k):
            j = labels[i, j_idx]
            sim = similarities[i, j_idx]

            if sim < threshold:
                continue
            if i == j:
                continue

            genome_j = genome_labels[j]
            if genome_i == genome_j:
                continue  # Skip same-genome pairs

            # Canonical ordering to avoid duplicates
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            pairs.append((int(i), int(j), float(sim)))

    print(f"Found {len(pairs)} cross-genome similar pairs")
    return pairs


def find_adjacent_blocks(
    df: pd.DataFrame,
    similar_pairs: list[tuple[int, int, float]],
    max_gap: int = 2,
    min_block_size: int = 3,
) -> list[dict]:
    """Find blocks of adjacent similar genes."""

    if not similar_pairs:
        return []

    # Build lookup structures
    gene_info = df[['gene_id', 'sample_id', 'contig_id', 'position_index']].reset_index(drop=True)

    # Group pairs by genome pair and contig pair
    contig_pairs = defaultdict(list)

    for idx_a, idx_b, sim in similar_pairs:
        info_a = gene_info.iloc[idx_a]
        info_b = gene_info.iloc[idx_b]

        key = (
            info_a['sample_id'], info_a['contig_id'],
            info_b['sample_id'], info_b['contig_id']
        )

        contig_pairs[key].append({
            'idx_a': idx_a, 'idx_b': idx_b,
            'pos_a': info_a['position_index'],
            'pos_b': info_b['position_index'],
            'gene_id_a': info_a['gene_id'],
            'gene_id_b': info_b['gene_id'],
            'sim': sim,
        })

    blocks = []

    print(f"Finding collinear chains in {len(contig_pairs)} contig pairs...")

    for key, matches in contig_pairs.items():
        if len(matches) < min_block_size:
            continue

        genome_a, contig_a, genome_b, contig_b = key

        # Find collinear chains
        chains = find_collinear_chains(matches, max_gap, min_block_size)

        for chain in chains:
            blocks.append({
                'genome_a': genome_a,
                'genome_b': genome_b,
                'contig_a': contig_a,
                'contig_b': contig_b,
                'genes_a': [m['gene_id_a'] for m in chain],
                'genes_b': [m['gene_id_b'] for m in chain],
            })

    return blocks


def find_collinear_chains(
    matches: list[dict],
    max_gap: int,
    min_size: int,
) -> list[list[dict]]:
    """Find collinear chains using dynamic programming."""

    if len(matches) < min_size:
        return []

    all_chains = []

    # Try both orientations
    for reverse_b in [False, True]:
        sorted_matches = sorted(matches, key=lambda x: x['pos_a'])

        if reverse_b:
            for m in sorted_matches:
                m['_cmp'] = -m['pos_b']
        else:
            for m in sorted_matches:
                m['_cmp'] = m['pos_b']

        n = len(sorted_matches)
        dp = [(1, -1) for _ in range(n)]

        for i in range(1, n):
            best_len, best_prev = 1, -1

            for j in range(i):
                gap_a = sorted_matches[i]['pos_a'] - sorted_matches[j]['pos_a'] - 1
                if gap_a > max_gap:
                    continue

                if sorted_matches[i]['_cmp'] <= sorted_matches[j]['_cmp']:
                    continue

                gap_b = abs(sorted_matches[i]['pos_b'] - sorted_matches[j]['pos_b']) - 1
                if gap_b > max_gap:
                    continue

                if dp[j][0] + 1 > best_len:
                    best_len = dp[j][0] + 1
                    best_prev = j

            dp[i] = (best_len, best_prev)

        # Backtrack
        used = set()
        for i in range(n - 1, -1, -1):
            if i in used or dp[i][0] < min_size:
                continue

            chain = []
            j = i
            while j >= 0:
                chain.append(sorted_matches[j])
                used.add(j)
                j = dp[j][1]
            chain.reverse()
            all_chains.append(chain)

    return all_chains


def pairwise_to_conserved(
    pairwise_blocks: list[dict],
) -> list[ConservedBlock]:
    """Convert pairwise blocks directly to ConservedBlock format (no merging)."""
    conserved = []
    for i, block in enumerate(pairwise_blocks):
        genes_by_genome = {
            block['genome_a']: sorted(block['genes_a']),
            block['genome_b']: sorted(block['genes_b']),
        }
        avg_genes = (len(block['genes_a']) + len(block['genes_b'])) // 2
        conserved.append(ConservedBlock(
            block_id=i,
            genes_by_genome=genes_by_genome,
            n_genomes=2,
            n_genes=avg_genes,
        ))
    return conserved


def merge_blocks(
    pairwise_blocks: list[dict],
    min_genomes: int = 2,
    merge_jaccard_threshold: float = 0.0,
) -> list[ConservedBlock]:
    """Merge pairwise blocks into multi-genome conserved blocks.

    Args:
        pairwise_blocks: List of pairwise block dicts
        min_genomes: Minimum genomes for a valid block
        merge_jaccard_threshold: Only merge blocks with Jaccard >= this threshold.
            If 0, uses legacy single-gene overlap merging.
    """
    if not pairwise_blocks:
        return []

    # Convert to sets for easier comparison
    block_gene_sets = []
    for block in pairwise_blocks:
        genes = set(block['genes_a']) | set(block['genes_b'])
        block_gene_sets.append(genes)

    # Union-find with Jaccard threshold
    parent = list(range(len(pairwise_blocks)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    if merge_jaccard_threshold > 0:
        # Merge only if blocks have high Jaccard overlap
        print(f"  Using Jaccard threshold {merge_jaccard_threshold} for merging...")
        for i in range(len(pairwise_blocks)):
            for j in range(i + 1, len(pairwise_blocks)):
                intersection = len(block_gene_sets[i] & block_gene_sets[j])
                union_size = len(block_gene_sets[i] | block_gene_sets[j])
                if union_size > 0:
                    jaccard = intersection / union_size
                    if jaccard >= merge_jaccard_threshold:
                        union(i, j)
    else:
        # Legacy: merge if ANY gene is shared
        gene_to_blocks = defaultdict(set)
        for i, block in enumerate(pairwise_blocks):
            for gene in block['genes_a']:
                gene_to_blocks[gene].add(i)
            for gene in block['genes_b']:
                gene_to_blocks[gene].add(i)

        for gene, block_ids in gene_to_blocks.items():
            block_list = list(block_ids)
            for i in range(1, len(block_list)):
                union(block_list[0], block_list[i])

    # Group by root
    groups = defaultdict(list)
    for i in range(len(pairwise_blocks)):
        groups[find(i)].append(pairwise_blocks[i])

    # Create conserved blocks
    conserved = []

    for group_blocks in groups.values():
        genes_by_genome = defaultdict(set)

        for block in group_blocks:
            genes_by_genome[block['genome_a']].update(block['genes_a'])
            genes_by_genome[block['genome_b']].update(block['genes_b'])

        if len(genes_by_genome) >= min_genomes:
            avg_genes = sum(len(g) for g in genes_by_genome.values()) // len(genes_by_genome)
            conserved.append(ConservedBlock(
                block_id=len(conserved),
                genes_by_genome={k: sorted(v) for k, v in genes_by_genome.items()},
                n_genomes=len(genes_by_genome),
                n_genes=avg_genes,
            ))

    return conserved


def build_ground_truth(
    genes_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.85,
    max_gap: int = 2,
    min_block_size: int = 3,
    min_genomes: int = 2,
    k_neighbors: int = 50,
    no_merge: bool = False,
    merge_jaccard_threshold: float = 0.0,
) -> list[ConservedBlock]:
    """Build ground truth using HNSW for efficient search.

    Args:
        no_merge: If True, output pairwise blocks without merging
        merge_jaccard_threshold: If > 0, only merge blocks with Jaccard >= threshold
    """

    print(f"Loading genes from {genes_path}")
    df, emb_cols = load_genes_parquet(genes_path)

    print("Computing gene positions...")
    df = compute_gene_positions(df)

    genomes = df['sample_id'].unique()
    print(f"Found {len(genomes)} genomes")

    # Extract embeddings and genome labels
    embeddings = df[emb_cols].values.astype(np.float32)
    genome_labels = df['sample_id'].values

    # Find similar proteins using HNSW
    similar_pairs = find_similar_proteins_hnsw(
        embeddings, genome_labels,
        k=k_neighbors, threshold=similarity_threshold
    )

    # Find adjacent blocks
    print("Finding adjacent blocks...")
    pairwise_blocks = find_adjacent_blocks(
        df, similar_pairs,
        max_gap=max_gap, min_block_size=min_block_size
    )
    print(f"Found {len(pairwise_blocks)} pairwise blocks")

    # Convert/merge into conserved blocks
    if no_merge:
        print("Skipping merge (--no-merge), outputting pairwise blocks...")
        conserved = pairwise_to_conserved(pairwise_blocks)
    else:
        print("Merging into conserved blocks...")
        conserved = merge_blocks(
            pairwise_blocks,
            min_genomes=min_genomes,
            merge_jaccard_threshold=merge_jaccard_threshold
        )
    print(f"Found {len(conserved)} conserved blocks")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [{
        'block_id': f"GT_{b.block_id:05d}",
        'n_genes': b.n_genes,
        'n_genomes': b.n_genomes,
        'genomes': ','.join(sorted(b.genes_by_genome.keys())),
        'genes_json': json.dumps(b.genes_by_genome),
    } for b in conserved]

    pd.DataFrame(rows).to_csv(output_path, sep='\t', index=False)
    print(f"Wrote {output_path}")

    # Also write JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump([b.to_dict() for b in conserved], f, indent=2)
    print(f"Wrote {json_path}")

    return conserved


def main():
    parser = argparse.ArgumentParser(description="Build ground truth using HNSW")
    parser.add_argument("genes_parquet", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-block-size", type=int, default=3)
    parser.add_argument("--min-genomes", type=int, default=2)
    parser.add_argument("--k-neighbors", type=int, default=50)
    parser.add_argument("--no-merge", action="store_true",
                        help="Output pairwise blocks without merging (avoids mega-block problem)")
    parser.add_argument("--merge-jaccard", type=float, default=0.0,
                        help="Only merge blocks with Jaccard >= threshold (0 = legacy single-gene merge)")

    args = parser.parse_args()

    build_ground_truth(
        args.genes_parquet,
        args.output,
        similarity_threshold=args.similarity_threshold,
        max_gap=args.max_gap,
        min_block_size=args.min_block_size,
        min_genomes=args.min_genomes,
        k_neighbors=args.k_neighbors,
        no_merge=args.no_merge,
        merge_jaccard_threshold=args.merge_jaccard,
    )


if __name__ == "__main__":
    main()
