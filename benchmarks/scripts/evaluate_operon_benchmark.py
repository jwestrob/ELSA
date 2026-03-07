#!/usr/bin/env python3
"""
Evaluate ELSA micro-chain blocks against operon ground truth.

This script:
1. Loads operon ground truth (conserved operon instances)
2. Loads ELSA micro-chain blocks
3. Computes recall, precision, and boundary accuracy

Metrics:
- Operon Recall: fraction of conserved operons detected by ELSA
- Operon Precision: fraction of ELSA blocks that overlap operons
- Boundary F1: how well block boundaries match operon boundaries

Usage:
    python benchmarks/scripts/evaluate_operon_benchmark.py --organism bsubtilis
"""

import argparse
import json
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


@dataclass
class OperonInstance:
    """A conserved operon instance between two genomes."""
    operon_id: str
    genome_a: str
    genome_b: str
    contig_a: str
    contig_b: str
    gene_idx_start_a: int
    gene_idx_end_a: int
    gene_idx_start_b: int
    gene_idx_end_b: int
    n_genes_a: int
    n_genes_b: int


@dataclass
class ELSABlock:
    """An ELSA micro-chain block."""
    block_id: int
    cluster_id: int
    query_genome: str
    target_genome: str
    query_contig: str
    target_contig: str
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    n_genes: int


def load_operon_ground_truth(gt_file: Path) -> list[OperonInstance]:
    """Load operon ground truth from TSV file."""
    operons = []

    with open(gt_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            operons.append(OperonInstance(
                operon_id=row['operon_id'],
                genome_a=row['genome_a'],
                genome_b=row['genome_b'],
                contig_a=row['contig_a'],
                contig_b=row['contig_b'],
                gene_idx_start_a=int(row['gene_idx_start_a']),
                gene_idx_end_a=int(row['gene_idx_end_a']),
                gene_idx_start_b=int(row['gene_idx_start_b']),
                gene_idx_end_b=int(row['gene_idx_end_b']),
                n_genes_a=int(row['n_genes_a']),
                n_genes_b=int(row['n_genes_b']),
            ))

    return operons


def load_elsa_blocks(blocks_file: Path) -> list[ELSABlock]:
    """Load ELSA micro-chain blocks from CSV file."""
    df = pd.read_csv(blocks_file)
    blocks = []

    for _, row in df.iterrows():
        blocks.append(ELSABlock(
            block_id=int(row['block_id']),
            cluster_id=int(row['cluster_id']),
            query_genome=row['query_genome'],
            target_genome=row['target_genome'],
            query_contig=row['query_contig'],
            target_contig=row['target_contig'],
            query_start=int(row['query_start']),
            query_end=int(row['query_end']),
            target_start=int(row['target_start']),
            target_end=int(row['target_end']),
            n_genes=int(row['n_genes']),
        ))

    return blocks


def ranges_overlap(start1: int, end1: int, start2: int, end2: int, min_overlap: float = 0.5) -> bool:
    """Check if two ranges overlap by at least min_overlap fraction."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start >= overlap_end:
        return False

    overlap_len = overlap_end - overlap_start + 1
    range1_len = end1 - start1 + 1
    range2_len = end2 - start2 + 1

    # Check if overlap is significant relative to the smaller range
    min_len = min(range1_len, range2_len)
    return overlap_len >= min_len * min_overlap


def evaluate_operon_detection(
    operons: list[OperonInstance],
    blocks: list[ELSABlock],
    min_overlap: float = 0.5,
) -> dict:
    """
    Evaluate ELSA blocks against operon ground truth.

    Returns:
        Dictionary with evaluation metrics.
    """

    # Build index of ELSA blocks by genome pair
    block_index = defaultdict(list)
    for block in blocks:
        # Index by both orderings of genome pair
        key1 = (block.query_genome, block.target_genome)
        key2 = (block.target_genome, block.query_genome)
        block_index[key1].append(block)
        block_index[key2].append(block)

    # Track which operons are detected
    detected_operons = []
    undetected_operons = []
    operon_matches = []

    for operon in operons:
        key = (operon.genome_a, operon.genome_b)
        candidate_blocks = block_index.get(key, [])

        # Check if any ELSA block overlaps this operon
        found = False
        best_match = None
        best_overlap = 0

        for block in candidate_blocks:
            # Match genome A to query or target
            if block.query_genome == operon.genome_a:
                block_start_a = block.query_start
                block_end_a = block.query_end
                block_start_b = block.target_start
                block_end_b = block.target_end
                block_contig_a = block.query_contig
                block_contig_b = block.target_contig
            else:
                block_start_a = block.target_start
                block_end_a = block.target_end
                block_start_b = block.query_start
                block_end_b = block.query_end
                block_contig_a = block.target_contig
                block_contig_b = block.query_contig

            # Check contig match
            if block_contig_a != operon.contig_a or block_contig_b != operon.contig_b:
                continue

            # Check overlap in both genomes
            overlap_a = ranges_overlap(
                block_start_a, block_end_a,
                operon.gene_idx_start_a, operon.gene_idx_end_a,
                min_overlap
            )
            overlap_b = ranges_overlap(
                block_start_b, block_end_b,
                operon.gene_idx_start_b, operon.gene_idx_end_b,
                min_overlap
            )

            if overlap_a and overlap_b:
                found = True
                # Calculate overlap score
                overlap_start_a = max(block_start_a, operon.gene_idx_start_a)
                overlap_end_a = min(block_end_a, operon.gene_idx_end_a)
                overlap_len = overlap_end_a - overlap_start_a + 1
                operon_len = operon.gene_idx_end_a - operon.gene_idx_start_a + 1
                overlap_score = overlap_len / operon_len

                if overlap_score > best_overlap:
                    best_overlap = overlap_score
                    best_match = block

        if found:
            detected_operons.append(operon)
            operon_matches.append({
                'operon_id': operon.operon_id,
                'genome_a': operon.genome_a,
                'genome_b': operon.genome_b,
                'block_id': best_match.block_id,
                'cluster_id': best_match.cluster_id,
                'overlap_score': best_overlap,
            })
        else:
            undetected_operons.append(operon)

    # Track which blocks match operons (for precision)
    block_matches = defaultdict(list)
    for match in operon_matches:
        block_matches[match['block_id']].append(match['operon_id'])

    # Compute metrics
    total_operons = len(operons)
    detected_count = len(detected_operons)
    recall = detected_count / total_operons if total_operons > 0 else 0

    # Precision: what fraction of ELSA blocks overlap at least one operon
    blocks_with_matches = len(block_matches)
    total_blocks = len(blocks)
    precision = blocks_with_matches / total_blocks if total_blocks > 0 else 0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-operon statistics
    operon_stats = defaultdict(lambda: {'detected': 0, 'total': 0})
    for operon in operons:
        operon_stats[operon.operon_id]['total'] += 1
    for operon in detected_operons:
        operon_stats[operon.operon_id]['detected'] += 1

    per_operon_recall = {
        op_id: stats['detected'] / stats['total']
        for op_id, stats in operon_stats.items()
    }

    results = {
        'metrics': {
            'operon_recall': recall,
            'operon_precision': precision,
            'f1_score': f1,
            'total_operon_instances': total_operons,
            'detected_operon_instances': detected_count,
            'total_elsa_blocks': total_blocks,
            'elsa_blocks_matching_operons': blocks_with_matches,
        },
        'per_operon_recall': per_operon_recall,
        'operon_matches': operon_matches[:100],  # Sample of matches
        'undetected_sample': [
            {
                'operon_id': op.operon_id,
                'genome_a': op.genome_a,
                'genome_b': op.genome_b,
                'n_genes': op.n_genes_a,
            }
            for op in undetected_operons[:50]
        ],
    }

    return results


def print_results(results: dict):
    """Print evaluation results in a formatted way."""
    metrics = results['metrics']

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n### Overall Metrics")
    print(f"  Operon Recall:    {metrics['operon_recall']:.1%} ({metrics['detected_operon_instances']}/{metrics['total_operon_instances']})")
    print(f"  Operon Precision: {metrics['operon_precision']:.1%} ({metrics['elsa_blocks_matching_operons']}/{metrics['total_elsa_blocks']})")
    print(f"  F1 Score:         {metrics['f1_score']:.3f}")

    print("\n### Per-Operon Recall (top 10 by recall)")
    per_operon = results['per_operon_recall']
    sorted_operons = sorted(per_operon.items(), key=lambda x: x[1], reverse=True)
    for op_id, recall in sorted_operons[:10]:
        print(f"  {op_id:25s}: {recall:.1%}")

    print("\n### Sample Undetected Operons")
    for undet in results['undetected_sample'][:10]:
        print(f"  {undet['operon_id']:25s} ({undet['genome_a']} <-> {undet['genome_b']}, {undet['n_genes']} genes)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ELSA blocks against operon ground truth')
    parser.add_argument('--organism', required=True, choices=['bsubtilis', 'ecoli'],
                        help='Organism to evaluate')
    parser.add_argument('--min-overlap', type=float, default=0.5,
                        help='Minimum overlap fraction for detection')
    parser.add_argument('--blocks-file', type=Path,
                        help='Path to ELSA blocks CSV (overrides default)')
    args = parser.parse_args()

    # Set paths based on organism (uses coordinate-based v2 ground truth)
    if args.organism == 'bsubtilis':
        gt_file = BENCHMARKS_DIR / 'ground_truth' / 'bsubtilis_operon_gt_v2.tsv'
        blocks_file = args.blocks_file or BENCHMARKS_DIR / 'results' / 'bacillus_chain' / 'micro_chain_blocks.csv'
        output_file = BENCHMARKS_DIR / 'evaluation' / 'bsubtilis_operon_eval.json'
    else:
        gt_file = BENCHMARKS_DIR / 'ground_truth' / 'ecoli_operon_gt_v2.tsv'
        blocks_file = args.blocks_file or BENCHMARKS_DIR / 'results' / 'ecoli_chain' / 'micro_chain_blocks.csv'
        output_file = BENCHMARKS_DIR / 'evaluation' / 'ecoli_operon_eval.json'

    print("=" * 60)
    print(f"Evaluating ELSA blocks against operon ground truth")
    print(f"Organism: {args.organism}")
    print("=" * 60)

    # Check files exist
    if not gt_file.exists():
        print(f"Error: Ground truth file not found: {gt_file}")
        return

    if not blocks_file.exists():
        print(f"Error: Blocks file not found: {blocks_file}")
        return

    # Load data
    print("\nLoading data...")
    operons = load_operon_ground_truth(gt_file)
    print(f"  Loaded {len(operons)} operon instances")

    blocks = load_elsa_blocks(blocks_file)
    print(f"  Loaded {len(blocks)} ELSA blocks")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_operon_detection(operons, blocks, args.min_overlap)

    # Print results
    print_results(results)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
