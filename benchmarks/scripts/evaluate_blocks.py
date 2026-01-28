#!/usr/bin/env python3
"""
Evaluate ELSA syntenic blocks against ground truth.

Metrics:
- Recall: % of ground truth blocks overlapped by ELSA blocks
- Precision: % of ELSA blocks overlapping ground truth
- F1: Harmonic mean
- Mean Jaccard: Average gene-set overlap for matched blocks
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class EvaluationResult:
    """Results from evaluating ELSA blocks against ground truth."""
    n_gt_blocks: int
    n_elsa_blocks: int
    n_matched_gt: int
    n_matched_elsa: int
    recall: float
    precision: float
    f1: float
    mean_jaccard: float
    fragmentation_rate: float  # GT blocks matched by multiple ELSA blocks
    merge_rate: float  # ELSA blocks matching multiple GT blocks

    def to_dict(self) -> dict:
        return {
            'n_gt_blocks': self.n_gt_blocks,
            'n_elsa_blocks': self.n_elsa_blocks,
            'n_matched_gt': self.n_matched_gt,
            'n_matched_elsa': self.n_matched_elsa,
            'recall': round(self.recall, 4),
            'precision': round(self.precision, 4),
            'f1': round(self.f1, 4),
            'mean_jaccard': round(self.mean_jaccard, 4),
            'fragmentation_rate': round(self.fragmentation_rate, 4),
            'merge_rate': round(self.merge_rate, 4),
        }


def load_ground_truth(gt_path: Path) -> list[dict]:
    """Load ground truth blocks from TSV or JSON."""
    if gt_path.suffix == '.json':
        with open(gt_path) as f:
            return json.load(f)
    else:
        # TSV format
        df = pd.read_csv(gt_path, sep='\t')
        blocks = []
        for _, row in df.iterrows():
            genes_by_genome = json.loads(row['genes_json'])
            blocks.append({
                'block_id': row['block_id'],
                'n_genes': row['n_genes'],
                'n_genomes': row['n_genomes'],
                'genes_by_genome': genes_by_genome,
            })
        return blocks


def load_elsa_blocks(blocks_path: Path) -> list[dict]:
    """Load ELSA syntenic blocks from CSV."""
    df = pd.read_csv(blocks_path)

    # Group by block_id and collect genes
    blocks = []

    # Check what columns we have
    if 'block_id' not in df.columns:
        print(f"Warning: No block_id column in {blocks_path}")
        return []

    # Try to find gene information columns
    gene_cols = [c for c in df.columns if 'gene' in c.lower() or 'window' in c.lower()]

    for block_id, group in df.groupby('block_id'):
        genes_by_genome = defaultdict(set)

        # Extract genes from the block
        # This depends on the ELSA output format
        for _, row in group.iterrows():
            # Try different possible column names
            if 'query_genome' in row and 'target_genome' in row:
                qg = row.get('query_genome', row.get('query_locus', '').split(':')[0])
                tg = row.get('target_genome', row.get('target_locus', '').split(':')[0])

                # Try to get gene lists
                if 'query_windows_json' in row:
                    try:
                        q_windows = json.loads(row['query_windows_json'])
                        for w in q_windows:
                            if isinstance(w, dict) and 'gene_id' in w:
                                genes_by_genome[qg].add(w['gene_id'])
                            elif isinstance(w, str):
                                genes_by_genome[qg].add(w)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if 'target_windows_json' in row:
                    try:
                        t_windows = json.loads(row['target_windows_json'])
                        for w in t_windows:
                            if isinstance(w, dict) and 'gene_id' in w:
                                genes_by_genome[tg].add(w['gene_id'])
                            elif isinstance(w, str):
                                genes_by_genome[tg].add(w)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Alternative: sample_id based
            if 'sample_id' in row:
                sample = row['sample_id']
                if 'gene_id' in row:
                    genes_by_genome[sample].add(row['gene_id'])

        if genes_by_genome:
            blocks.append({
                'block_id': block_id,
                'genes_by_genome': {k: list(v) for k, v in genes_by_genome.items()},
                'n_genomes': len(genes_by_genome),
            })

    return blocks


def compute_block_overlap(
    block_a: dict,
    block_b: dict,
) -> tuple[float, int]:
    """Compute Jaccard overlap between two blocks.

    Returns (jaccard_score, n_shared_genes).
    """
    # Get all genes from each block
    genes_a = set()
    genes_b = set()

    for genome, genes in block_a.get('genes_by_genome', {}).items():
        genes_a.update(genes)

    for genome, genes in block_b.get('genes_by_genome', {}).items():
        genes_b.update(genes)

    if not genes_a or not genes_b:
        return 0.0, 0

    intersection = genes_a & genes_b
    union = genes_a | genes_b

    jaccard = len(intersection) / len(union) if union else 0.0

    return jaccard, len(intersection)


def evaluate_blocks(
    gt_blocks: list[dict],
    elsa_blocks: list[dict],
    overlap_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate ELSA blocks against ground truth.

    Args:
        gt_blocks: Ground truth conserved blocks
        elsa_blocks: ELSA-predicted syntenic blocks
        overlap_threshold: Minimum Jaccard for a "match"

    Returns:
        EvaluationResult with metrics
    """
    n_gt = len(gt_blocks)
    n_elsa = len(elsa_blocks)

    if n_gt == 0:
        print("Warning: No ground truth blocks")
        return EvaluationResult(
            n_gt_blocks=0, n_elsa_blocks=n_elsa,
            n_matched_gt=0, n_matched_elsa=0,
            recall=0.0, precision=0.0, f1=0.0, mean_jaccard=0.0,
            fragmentation_rate=0.0, merge_rate=0.0
        )

    if n_elsa == 0:
        print("Warning: No ELSA blocks")
        return EvaluationResult(
            n_gt_blocks=n_gt, n_elsa_blocks=0,
            n_matched_gt=0, n_matched_elsa=0,
            recall=0.0, precision=0.0, f1=0.0, mean_jaccard=0.0,
            fragmentation_rate=0.0, merge_rate=0.0
        )

    # Compute all pairwise overlaps
    gt_matches = defaultdict(list)  # gt_idx -> list of (elsa_idx, jaccard)
    elsa_matches = defaultdict(list)  # elsa_idx -> list of (gt_idx, jaccard)
    all_jaccards = []

    print(f"Computing {n_gt} x {n_elsa} block overlaps...")

    for i, gt_block in enumerate(gt_blocks):
        for j, elsa_block in enumerate(elsa_blocks):
            jaccard, n_shared = compute_block_overlap(gt_block, elsa_block)

            if jaccard >= overlap_threshold or n_shared >= 3:
                gt_matches[i].append((j, jaccard))
                elsa_matches[j].append((i, jaccard))
                all_jaccards.append(jaccard)

    # Compute metrics
    matched_gt = set(gt_matches.keys())
    matched_elsa = set(elsa_matches.keys())

    recall = len(matched_gt) / n_gt
    precision = len(matched_elsa) / n_elsa if n_elsa > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    mean_jaccard = np.mean(all_jaccards) if all_jaccards else 0.0

    # Fragmentation: GT blocks matched by multiple ELSA blocks
    fragmented = sum(1 for matches in gt_matches.values() if len(matches) > 1)
    fragmentation_rate = fragmented / n_gt if n_gt > 0 else 0.0

    # Merge: ELSA blocks matching multiple GT blocks
    merged = sum(1 for matches in elsa_matches.values() if len(matches) > 1)
    merge_rate = merged / n_elsa if n_elsa > 0 else 0.0

    return EvaluationResult(
        n_gt_blocks=n_gt,
        n_elsa_blocks=n_elsa,
        n_matched_gt=len(matched_gt),
        n_matched_elsa=len(matched_elsa),
        recall=recall,
        precision=precision,
        f1=f1,
        mean_jaccard=mean_jaccard,
        fragmentation_rate=fragmentation_rate,
        merge_rate=merge_rate,
    )


def evaluate_with_pfam(
    elsa_blocks: list[dict],
    pfam_annotations: dict,
) -> float:
    """Compute PFAM coherence for ELSA blocks.

    Returns fraction of blocks where genes share PFAM domains.
    """
    if not pfam_annotations:
        return 0.0

    coherent = 0
    total = 0

    for block in elsa_blocks:
        genes = []
        for genome_genes in block.get('genes_by_genome', {}).values():
            genes.extend(genome_genes)

        if len(genes) < 2:
            continue

        # Get PFAM domains for each gene
        pfam_sets = []
        for gene in genes:
            domains = set(pfam_annotations.get(gene, []))
            if domains:
                pfam_sets.append(domains)

        if len(pfam_sets) >= 2:
            total += 1
            # Check if any domain is shared
            common = pfam_sets[0]
            for s in pfam_sets[1:]:
                common = common & s
            if common:
                coherent += 1

    return coherent / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ELSA blocks against ground truth"
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to ground truth blocks (TSV or JSON)"
    )
    parser.add_argument(
        "elsa_blocks",
        type=Path,
        help="Path to ELSA syntenic_blocks.csv"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for evaluation results (JSON)"
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Minimum Jaccard overlap for a match (default: 0.5)"
    )
    parser.add_argument(
        "--pfam",
        type=Path,
        help="Optional PFAM annotations JSON for coherence check"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading ground truth from {args.ground_truth}")
    gt_blocks = load_ground_truth(args.ground_truth)
    print(f"  Loaded {len(gt_blocks)} ground truth blocks")

    print(f"Loading ELSA blocks from {args.elsa_blocks}")
    elsa_blocks = load_elsa_blocks(args.elsa_blocks)
    print(f"  Loaded {len(elsa_blocks)} ELSA blocks")

    # Evaluate
    result = evaluate_blocks(gt_blocks, elsa_blocks, args.overlap_threshold)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Ground Truth Blocks:  {result.n_gt_blocks}")
    print(f"ELSA Blocks:          {result.n_elsa_blocks}")
    print(f"Matched GT:           {result.n_matched_gt}")
    print(f"Matched ELSA:         {result.n_matched_elsa}")
    print("-"*50)
    print(f"Recall:               {result.recall:.4f}")
    print(f"Precision:            {result.precision:.4f}")
    print(f"F1 Score:             {result.f1:.4f}")
    print(f"Mean Jaccard:         {result.mean_jaccard:.4f}")
    print("-"*50)
    print(f"Fragmentation Rate:   {result.fragmentation_rate:.4f}")
    print(f"Merge Rate:           {result.merge_rate:.4f}")

    # PFAM coherence if provided
    if args.pfam and args.pfam.exists():
        with open(args.pfam) as f:
            pfam_data = json.load(f)
        coherence = evaluate_with_pfam(elsa_blocks, pfam_data)
        print(f"PFAM Coherence:       {coherence:.4f}")
        result_dict = result.to_dict()
        result_dict['pfam_coherence'] = round(coherence, 4)
    else:
        result_dict = result.to_dict()

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
