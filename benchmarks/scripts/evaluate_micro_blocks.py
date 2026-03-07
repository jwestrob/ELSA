#!/usr/bin/env python3
"""
Evaluate micro-synteny blocks against ground truth.

Supports both legacy format (fixed 3-gene windows) and new chain format
(variable-length blocks from anchor chaining).

Legacy format columns:
  genome_id, contig_id, start_index, end_index, cluster_id, n_genes

Chain format columns:
  block_id, cluster_id, query_genome, target_genome, query_contig, target_contig,
  query_start, query_end, target_start, target_end, n_anchors, chain_score, orientation
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def load_ground_truth(gt_path: Path) -> list[dict]:
    """Load ground truth blocks from JSON."""
    with open(gt_path) as f:
        return json.load(f)


def load_micro_blocks(blocks_path: Path) -> pd.DataFrame:
    """Load micro blocks with cluster assignments."""
    return pd.read_csv(blocks_path)


def detect_format(df: pd.DataFrame) -> str:
    """Detect whether this is legacy or chain format."""
    if 'query_genome' in df.columns and 'target_genome' in df.columns:
        return 'chain'
    elif 'genome_id' in df.columns:
        return 'legacy'
    else:
        raise ValueError(f"Unknown format. Columns: {list(df.columns)}")


def build_gene_to_gt_map(gt_blocks: list[dict]) -> dict:
    """Build mapping from gene_id -> list of GT block IDs that contain it."""
    gene_to_gt = defaultdict(set)
    for block in gt_blocks:
        block_id = block['block_id']
        for genome, genes in block.get('genes_by_genome', {}).items():
            for gene in genes:
                gene_to_gt[gene].add(block_id)
    return gene_to_gt


def load_gene_mapping(genes_parquet_path: Optional[Path]) -> Optional[dict]:
    """Load gene position-to-ID mapping from genes.parquet.

    Returns mapping: (genome_id, contig_id, position_index) -> gene_id
    """
    if genes_parquet_path is None or not genes_parquet_path.exists():
        return None

    df = pd.read_parquet(genes_parquet_path)
    df = df.sort_values(['sample_id', 'contig_id', 'start', 'end'])
    df['position_index'] = df.groupby(['sample_id', 'contig_id']).cumcount()

    mapping = {}
    for _, row in df.iterrows():
        key = (str(row['sample_id']), str(row['contig_id']), int(row['position_index']))
        mapping[key] = str(row['gene_id'])

    return mapping


def extract_genes_legacy(
    micro_df: pd.DataFrame,
    gene_mapping: Optional[dict] = None,
) -> dict:
    """Extract cluster -> genome -> gene set mapping for legacy format."""
    cluster_genes = defaultdict(lambda: defaultdict(set))

    for _, row in micro_df.iterrows():
        cluster_id = row['cluster_id']
        genome_id = str(row['genome_id'])
        contig_id = str(row['contig_id'])

        for idx in range(int(row['start_index']), int(row['end_index']) + 1):
            if gene_mapping:
                # Use actual gene ID from mapping
                key = (genome_id, contig_id, idx)
                gene_id = gene_mapping.get(key)
                if gene_id:
                    cluster_genes[cluster_id][genome_id].add(gene_id)
            else:
                # Use synthetic gene ID
                gene_id = f"{genome_id}_{contig_id}_{idx}"
                cluster_genes[cluster_id][genome_id].add(gene_id)

    return cluster_genes


def extract_genes_chain(
    micro_df: pd.DataFrame,
    gene_mapping: Optional[dict] = None,
) -> dict:
    """Extract cluster -> genome -> gene set mapping for chain format."""
    cluster_genes = defaultdict(lambda: defaultdict(set))

    for _, row in micro_df.iterrows():
        cluster_id = row['cluster_id']

        # Query side
        query_genome = str(row['query_genome'])
        query_contig = str(row['query_contig'])
        for idx in range(int(row['query_start']), int(row['query_end']) + 1):
            if gene_mapping:
                key = (query_genome, query_contig, idx)
                gene_id = gene_mapping.get(key)
                if gene_id:
                    cluster_genes[cluster_id][query_genome].add(gene_id)
            else:
                gene_id = f"{query_genome}_{query_contig}_{idx}"
                cluster_genes[cluster_id][query_genome].add(gene_id)

        # Target side
        target_genome = str(row['target_genome'])
        target_contig = str(row['target_contig'])
        for idx in range(int(row['target_start']), int(row['target_end']) + 1):
            if gene_mapping:
                key = (target_genome, target_contig, idx)
                gene_id = gene_mapping.get(key)
                if gene_id:
                    cluster_genes[cluster_id][target_genome].add(gene_id)
            else:
                gene_id = f"{target_genome}_{target_contig}_{idx}"
                cluster_genes[cluster_id][target_genome].add(gene_id)

    return cluster_genes


def evaluate_micro_blocks(
    gt_blocks: list[dict],
    micro_df: pd.DataFrame,
    min_overlap: int = 2,
    genes_parquet_path: Optional[Path] = None,
) -> dict:
    """Evaluate micro blocks against ground truth."""

    # Detect format and load gene mapping if available
    format_type = detect_format(micro_df)
    gene_mapping = load_gene_mapping(genes_parquet_path)

    print(f"  Format: {format_type}")
    if gene_mapping:
        print(f"  Gene mapping: {len(gene_mapping)} positions")

    # Build gene -> GT block mapping
    gene_to_gt = build_gene_to_gt_map(gt_blocks)

    # Extract cluster genes based on format
    if format_type == 'chain':
        cluster_genes = extract_genes_chain(micro_df, gene_mapping)
    else:
        cluster_genes = extract_genes_legacy(micro_df, gene_mapping)

    # Count metrics
    n_micro_clusters = len([c for c in cluster_genes.keys() if c != 0])  # Exclude sink
    n_gt_blocks = len(gt_blocks)

    # For each micro cluster, find matching GT blocks
    micro_to_gt_matches = {}
    gt_matched = set()

    for cluster_id, genes_by_genome in cluster_genes.items():
        if cluster_id == 0:  # Skip sink cluster
            continue

        # Get all genes in this micro cluster
        all_genes = set()
        for genome_genes in genes_by_genome.values():
            all_genes.update(genome_genes)

        # Find GT blocks that overlap
        matching_gt = set()
        for gene in all_genes:
            if gene in gene_to_gt:
                matching_gt.update(gene_to_gt[gene])

        # Filter to GT blocks with significant overlap
        significant_matches = set()
        for gt_bid in matching_gt:
            # Find GT block by ID (may be string or int)
            gt_block = None
            for b in gt_blocks:
                if str(b['block_id']) == str(gt_bid) or b['block_id'] == gt_bid:
                    gt_block = b
                    break

            if gt_block is None:
                continue

            gt_genes = set()
            for genome, genes in gt_block.get('genes_by_genome', {}).items():
                gt_genes.update(genes)

            overlap = len(all_genes & gt_genes)
            if overlap >= min_overlap:
                significant_matches.add(gt_bid)
                gt_matched.add(gt_bid)

        micro_to_gt_matches[cluster_id] = significant_matches

    # Calculate metrics
    matched_micro = sum(1 for matches in micro_to_gt_matches.values() if matches)

    recall = len(gt_matched) / n_gt_blocks if n_gt_blocks > 0 else 0.0
    precision = matched_micro / n_micro_clusters if n_micro_clusters > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    # Size statistics - handle both formats
    if 'n_genes' in micro_df.columns:
        block_sizes = micro_df['n_genes'].values
    elif 'n_anchors' in micro_df.columns:
        block_sizes = micro_df['n_anchors'].values
    else:
        # Compute from spans
        if format_type == 'chain':
            block_sizes = (micro_df['query_end'] - micro_df['query_start'] + 1).values
        else:
            block_sizes = (micro_df['end_index'] - micro_df['start_index'] + 1).values

    # Additional chain-specific metrics
    result = {
        'format': format_type,
        'n_gt_blocks': n_gt_blocks,
        'n_micro_clusters': n_micro_clusters,
        'n_micro_blocks': len(micro_df),
        'n_matched_gt': len(gt_matched),
        'n_matched_micro': matched_micro,
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
        'mean_block_size': round(float(np.mean(block_sizes)), 2),
        'min_block_size': int(np.min(block_sizes)),
        'max_block_size': int(np.max(block_sizes)),
    }

    # Add chain-specific metrics
    if format_type == 'chain' and 'chain_score' in micro_df.columns:
        result['mean_chain_score'] = round(float(micro_df['chain_score'].mean()), 4)
        result['n_forward'] = int((micro_df['orientation'] == 1).sum())
        result['n_inverted'] = int((micro_df['orientation'] == -1).sum())

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate micro-synteny blocks against ground truth"
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to ground truth blocks (JSON)"
    )
    parser.add_argument(
        "micro_blocks",
        type=Path,
        help="Path to micro blocks CSV (legacy or chain format)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for evaluation results (JSON)"
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum gene overlap to count as a match (default: 2)"
    )
    parser.add_argument(
        "--genes-parquet",
        type=Path,
        help="Path to genes.parquet for accurate gene ID mapping"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading ground truth from {args.ground_truth}")
    gt_blocks = load_ground_truth(args.ground_truth)
    print(f"  Loaded {len(gt_blocks)} ground truth blocks")

    print(f"Loading micro blocks from {args.micro_blocks}")
    micro_df = load_micro_blocks(args.micro_blocks)
    print(f"  Loaded {len(micro_df)} micro blocks")
    n_clusters = micro_df['cluster_id'].nunique()
    print(f"  {n_clusters} unique clusters")

    # Evaluate
    result = evaluate_micro_blocks(
        gt_blocks, micro_df, args.min_overlap,
        genes_parquet_path=args.genes_parquet
    )

    # Print results
    print("\n" + "="*50)
    print("MICRO-SYNTENY EVALUATION RESULTS")
    print("="*50)
    print(f"Format:               {result['format']}")
    print(f"Ground Truth Blocks:  {result['n_gt_blocks']}")
    print(f"Micro Clusters:       {result['n_micro_clusters']}")
    print(f"Micro Blocks:         {result['n_micro_blocks']}")
    print("-"*50)
    print(f"Matched GT:           {result['n_matched_gt']}")
    print(f"Matched Micro:        {result['n_matched_micro']}")
    print("-"*50)
    print(f"Recall:               {result['recall']:.4f}")
    print(f"Precision:            {result['precision']:.4f}")
    print(f"F1 Score:             {result['f1']:.4f}")
    print("-"*50)
    print(f"Block Size:           {result['mean_block_size']:.1f} genes (range: {result['min_block_size']}-{result['max_block_size']})")

    if 'mean_chain_score' in result:
        print(f"Mean Chain Score:     {result['mean_chain_score']:.4f}")
        print(f"Orientation:          {result['n_forward']} forward, {result['n_inverted']} inverted")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
