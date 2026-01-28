#!/usr/bin/env python3
"""
Build ground truth conserved syntenic blocks from ELSA embeddings.

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
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


@dataclass
class Gene:
    """A gene with its embedding and position."""
    gene_id: str
    sample_id: str
    contig_id: str
    start: int
    end: int
    strand: int
    embedding: np.ndarray
    position_index: int = 0  # Position in contig gene order


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


def load_genes_parquet(genes_path: Path) -> pd.DataFrame:
    """Load genes.parquet with embeddings."""
    df = pd.read_parquet(genes_path)

    # Find embedding columns (they start with a number or are named emb_*)
    emb_cols = [c for c in df.columns if c.startswith('emb_') or
                (c.replace('.', '').replace('-', '').isdigit())]

    if not emb_cols:
        # Try to find columns that look like embedding dimensions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude known non-embedding columns
        exclude = {'start', 'end', 'strand', 'position_index', 'index'}
        emb_cols = [c for c in numeric_cols if c not in exclude and
                    not c.endswith('_id') and not c.endswith('_idx')]

    print(f"Found {len(emb_cols)} embedding dimensions")
    print(f"Loaded {len(df)} genes from {df['sample_id'].nunique()} genomes")

    return df, emb_cols


def compute_gene_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute position index for each gene within its contig."""
    df = df.copy()

    # Sort by genome, contig, start position
    df = df.sort_values(['sample_id', 'contig_id', 'start'])

    # Assign position index within each contig
    df['position_index'] = df.groupby(['sample_id', 'contig_id']).cumcount()

    return df


def find_similar_proteins(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    threshold: float = 0.9,
) -> list[tuple[int, int, float]]:
    """Find pairs of similar proteins between two sets of embeddings.

    Returns list of (idx_a, idx_b, similarity) tuples.
    """
    # Normalize embeddings for cosine similarity
    norm_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-9)
    norm_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-9)

    # Compute cosine similarity matrix
    sim_matrix = norm_a @ norm_b.T

    # Find pairs above threshold
    pairs = []
    rows, cols = np.where(sim_matrix >= threshold)
    for i, j in zip(rows, cols):
        pairs.append((int(i), int(j), float(sim_matrix[i, j])))

    return pairs


def find_adjacent_matches(
    genes_a: pd.DataFrame,
    genes_b: pd.DataFrame,
    similar_pairs: list[tuple[int, int, float]],
    max_gap: int = 2,
    min_block_size: int = 3,
) -> list[dict]:
    """Find blocks of adjacent similar genes between two genomes.

    Args:
        genes_a: Genes from genome A with position_index
        genes_b: Genes from genome B with position_index
        similar_pairs: List of (idx_a, idx_b, similarity) tuples
        max_gap: Maximum gap (intervening genes) allowed
        min_block_size: Minimum number of genes in a block

    Returns:
        List of blocks, each with genes from both genomes
    """
    if not similar_pairs:
        return []

    # Build lookup from index to gene info
    genes_a = genes_a.reset_index(drop=True)
    genes_b = genes_b.reset_index(drop=True)

    # Group similar pairs by contig pair
    contig_pairs = defaultdict(list)
    for idx_a, idx_b, sim in similar_pairs:
        if idx_a >= len(genes_a) or idx_b >= len(genes_b):
            continue
        contig_a = genes_a.iloc[idx_a]['contig_id']
        contig_b = genes_b.iloc[idx_b]['contig_id']
        pos_a = genes_a.iloc[idx_a]['position_index']
        pos_b = genes_b.iloc[idx_b]['position_index']
        gene_id_a = genes_a.iloc[idx_a]['gene_id']
        gene_id_b = genes_b.iloc[idx_b]['gene_id']

        contig_pairs[(contig_a, contig_b)].append({
            'idx_a': idx_a, 'idx_b': idx_b,
            'pos_a': pos_a, 'pos_b': pos_b,
            'gene_id_a': gene_id_a, 'gene_id_b': gene_id_b,
            'sim': sim,
        })

    blocks = []

    for (contig_a, contig_b), matches in contig_pairs.items():
        if len(matches) < min_block_size:
            continue

        # Sort by position in genome A
        matches = sorted(matches, key=lambda x: x['pos_a'])

        # Find chains of adjacent matches
        # Use dynamic programming to find longest chains
        chains = find_collinear_chains(matches, max_gap, min_block_size)

        for chain in chains:
            blocks.append({
                'contig_a': contig_a,
                'contig_b': contig_b,
                'genes_a': [m['gene_id_a'] for m in chain],
                'genes_b': [m['gene_id_b'] for m in chain],
                'positions_a': [m['pos_a'] for m in chain],
                'positions_b': [m['pos_b'] for m in chain],
                'similarities': [m['sim'] for m in chain],
            })

    return blocks


def find_collinear_chains(
    matches: list[dict],
    max_gap: int,
    min_size: int,
) -> list[list[dict]]:
    """Find collinear chains of matches (similar to LIS with gap constraint).

    A chain is collinear if positions in both genomes are monotonically
    increasing (allowing for strand flip = monotonically decreasing in B).
    """
    if len(matches) < min_size:
        return []

    n = len(matches)

    # Try both orientations (forward and reverse in genome B)
    all_chains = []

    for reverse_b in [False, True]:
        # Sort matches by position in A
        sorted_matches = sorted(matches, key=lambda x: x['pos_a'])

        if reverse_b:
            # For reverse orientation, we want decreasing positions in B
            # So we negate pos_b for the comparison
            for m in sorted_matches:
                m['_pos_b_cmp'] = -m['pos_b']
        else:
            for m in sorted_matches:
                m['_pos_b_cmp'] = m['pos_b']

        # Find longest increasing subsequence with gap constraint
        # dp[i] = (chain_length, prev_index)
        dp = [(1, -1) for _ in range(n)]

        for i in range(1, n):
            best_len = 1
            best_prev = -1

            for j in range(i):
                # Check gap constraint in genome A
                gap_a = sorted_matches[i]['pos_a'] - sorted_matches[j]['pos_a'] - 1
                if gap_a > max_gap:
                    continue

                # Check collinearity in genome B
                if sorted_matches[i]['_pos_b_cmp'] <= sorted_matches[j]['_pos_b_cmp']:
                    continue

                # Check gap constraint in genome B
                gap_b = abs(sorted_matches[i]['pos_b'] - sorted_matches[j]['pos_b']) - 1
                if gap_b > max_gap:
                    continue

                if dp[j][0] + 1 > best_len:
                    best_len = dp[j][0] + 1
                    best_prev = j

            dp[i] = (best_len, best_prev)

        # Backtrack to find chains
        used = set()
        for i in range(n - 1, -1, -1):
            if i in used:
                continue
            if dp[i][0] >= min_size:
                # Backtrack to get chain
                chain = []
                j = i
                while j >= 0:
                    chain.append(sorted_matches[j])
                    used.add(j)
                    j = dp[j][1]
                chain.reverse()
                all_chains.append(chain)

    return all_chains


def merge_pairwise_blocks(
    pairwise_blocks: dict[tuple[str, str], list[dict]],
    min_genomes: int = 2,
) -> list[ConservedBlock]:
    """Merge pairwise blocks into multi-genome conserved blocks.

    Two blocks are merged if they share significant gene overlap.
    """
    # Collect all unique genes and their block memberships
    gene_to_blocks = defaultdict(set)
    block_id_counter = 0
    raw_blocks = []

    for (genome_a, genome_b), blocks in pairwise_blocks.items():
        for block in blocks:
            block_id = block_id_counter
            block_id_counter += 1

            raw_blocks.append({
                'id': block_id,
                'genomes': {genome_a, genome_b},
                'genes': {
                    genome_a: set(block['genes_a']),
                    genome_b: set(block['genes_b']),
                },
            })

            for gene in block['genes_a']:
                gene_to_blocks[gene].add(block_id)
            for gene in block['genes_b']:
                gene_to_blocks[gene].add(block_id)

    if not raw_blocks:
        return []

    # Union-find to merge overlapping blocks
    parent = list(range(len(raw_blocks)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Merge blocks that share genes
    for gene, block_ids in gene_to_blocks.items():
        block_list = list(block_ids)
        for i in range(1, len(block_list)):
            union(block_list[0], block_list[i])

    # Group blocks by their root
    merged_groups = defaultdict(list)
    for i, block in enumerate(raw_blocks):
        root = find(i)
        merged_groups[root].append(block)

    # Create final conserved blocks
    conserved_blocks = []
    final_block_id = 0

    for group in merged_groups.values():
        # Merge all blocks in the group
        all_genomes = set()
        genes_by_genome = defaultdict(set)

        for block in group:
            all_genomes.update(block['genomes'])
            for genome, genes in block['genes'].items():
                genes_by_genome[genome].update(genes)

        if len(all_genomes) >= min_genomes:
            cb = ConservedBlock(
                block_id=final_block_id,
                genes_by_genome={g: sorted(genes) for g, genes in genes_by_genome.items()},
                n_genomes=len(all_genomes),
                n_genes=sum(len(genes) for genes in genes_by_genome.values()) // len(all_genomes),
            )
            conserved_blocks.append(cb)
            final_block_id += 1

    return conserved_blocks


def build_ground_truth(
    genes_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.9,
    max_gap: int = 2,
    min_block_size: int = 3,
    min_genomes: int = 2,
) -> list[ConservedBlock]:
    """Build ground truth conserved blocks from ELSA embeddings.

    Args:
        genes_path: Path to genes.parquet from ELSA
        output_path: Path to write ground truth TSV
        similarity_threshold: Cosine similarity threshold for "same protein"
        max_gap: Maximum intervening genes allowed
        min_block_size: Minimum genes per block
        min_genomes: Minimum genomes for conservation

    Returns:
        List of ConservedBlock objects
    """
    print(f"Loading genes from {genes_path}")
    df, emb_cols = load_genes_parquet(genes_path)

    # Compute gene positions within contigs
    print("Computing gene positions...")
    df = compute_gene_positions(df)

    # Get list of genomes
    genomes = df['sample_id'].unique().tolist()
    print(f"Found {len(genomes)} genomes")

    # Extract embeddings as numpy array
    embeddings = df[emb_cols].values.astype(np.float32)

    # Find pairwise conserved blocks
    pairwise_blocks = {}
    n_pairs = len(genomes) * (len(genomes) - 1) // 2
    pair_count = 0

    print(f"Comparing {n_pairs} genome pairs...")

    for i, genome_a in enumerate(genomes):
        for genome_b in genomes[i+1:]:
            pair_count += 1
            if pair_count % 10 == 0:
                print(f"  Progress: {pair_count}/{n_pairs} pairs")

            # Get genes for each genome
            mask_a = df['sample_id'] == genome_a
            mask_b = df['sample_id'] == genome_b

            genes_a = df[mask_a].copy()
            genes_b = df[mask_b].copy()

            emb_a = embeddings[mask_a.values]
            emb_b = embeddings[mask_b.values]

            # Find similar proteins
            similar_pairs = find_similar_proteins(
                emb_a, emb_b, threshold=similarity_threshold
            )

            if not similar_pairs:
                continue

            # Find adjacent blocks
            blocks = find_adjacent_matches(
                genes_a, genes_b, similar_pairs,
                max_gap=max_gap, min_block_size=min_block_size
            )

            if blocks:
                pairwise_blocks[(genome_a, genome_b)] = blocks

    print(f"Found blocks in {len(pairwise_blocks)} genome pairs")

    # Merge into multi-genome conserved blocks
    print("Merging pairwise blocks...")
    conserved_blocks = merge_pairwise_blocks(pairwise_blocks, min_genomes=min_genomes)

    print(f"Found {len(conserved_blocks)} conserved blocks")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as TSV
    rows = []
    for block in conserved_blocks:
        rows.append({
            'block_id': f"GT_{block.block_id:05d}",
            'n_genes': block.n_genes,
            'n_genomes': block.n_genomes,
            'genomes': ','.join(sorted(block.genes_by_genome.keys())),
            'genes_json': json.dumps(block.genes_by_genome),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep='\t', index=False)
    print(f"Wrote ground truth to {output_path}")

    # Also write detailed JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump([b.to_dict() for b in conserved_blocks], f, indent=2)
    print(f"Wrote detailed JSON to {json_path}")

    return conserved_blocks


def main():
    parser = argparse.ArgumentParser(
        description="Build ground truth conserved blocks from ELSA embeddings"
    )
    parser.add_argument(
        "genes_parquet",
        type=Path,
        help="Path to genes.parquet from ELSA embed output"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path for ground truth TSV"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for 'same protein' (default: 0.9)"
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum intervening genes allowed in a block (default: 2)"
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=3,
        help="Minimum genes per block (default: 3)"
    )
    parser.add_argument(
        "--min-genomes",
        type=int,
        default=2,
        help="Minimum genomes for conservation (default: 2)"
    )

    args = parser.parse_args()

    build_ground_truth(
        args.genes_parquet,
        args.output,
        similarity_threshold=args.similarity_threshold,
        max_gap=args.max_gap,
        min_block_size=args.min_block_size,
        min_genomes=args.min_genomes,
    )


if __name__ == "__main__":
    main()
