#!/usr/bin/env python3
"""
Validate ELSA blocks against orthogroup data.

For each ELSA block, check if genes in the aligned regions are orthologs.
This validates that ELSA is finding true synteny, not spurious alignments.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


def load_orthogroups(og_file: Path) -> dict:
    """Load orthogroup assignments into a gene_id -> orthogroup dict."""
    df = pd.read_csv(og_file, sep='\t')
    return dict(zip(df['gene_id'], df['orthogroup']))


def get_genes_in_range(genes_df: pd.DataFrame, sample_id: str, contig: str,
                       start_idx: int, end_idx: int) -> list[str]:
    """Get gene IDs for genes in a range."""
    mask = (
        (genes_df['sample_id'] == sample_id) &
        (genes_df['contig_id'] == contig)
    )
    sample_genes = genes_df[mask].reset_index(drop=True)

    # Gene indices are 0-based positions within sample
    if start_idx < 0 or end_idx >= len(sample_genes):
        return []

    return sample_genes.iloc[start_idx:end_idx+1]['gene_id'].tolist()


def validate_block(block: dict, genes_df: pd.DataFrame, og_map: dict) -> dict:
    """Validate a single block by checking ortholog content."""
    # Get genes in query region
    query_genes = get_genes_in_range(
        genes_df, block['query_genome'], block['query_contig'],
        block['query_start'], block['query_end']
    )

    # Get genes in target region
    target_genes = get_genes_in_range(
        genes_df, block['target_genome'], block['target_contig'],
        block['target_start'], block['target_end']
    )

    if not query_genes or not target_genes:
        return {'valid': False, 'reason': 'no_genes'}

    # Get orthogroups for each region
    query_ogs = set(og_map.get(g, None) for g in query_genes) - {None}
    target_ogs = set(og_map.get(g, None) for g in target_genes) - {None}

    # Check overlap
    shared_ogs = query_ogs & target_ogs

    # Calculate metrics
    query_og_coverage = len(query_ogs) / len(query_genes) if query_genes else 0
    target_og_coverage = len(target_ogs) / len(target_genes) if target_genes else 0

    # For genes with orthogroups, what fraction share orthogroups?
    query_genes_with_shared = sum(1 for g in query_genes if og_map.get(g) in shared_ogs)
    target_genes_with_shared = sum(1 for g in target_genes if og_map.get(g) in shared_ogs)

    ortholog_fraction_query = query_genes_with_shared / len(query_genes) if query_genes else 0
    ortholog_fraction_target = target_genes_with_shared / len(target_genes) if target_genes else 0

    return {
        'block_id': block['block_id'],
        'n_query_genes': len(query_genes),
        'n_target_genes': len(target_genes),
        'n_shared_ogs': len(shared_ogs),
        'ortholog_fraction_query': ortholog_fraction_query,
        'ortholog_fraction_target': ortholog_fraction_target,
        'min_ortholog_fraction': min(ortholog_fraction_query, ortholog_fraction_target),
        'valid': True,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate ELSA blocks against orthogroups')
    parser.add_argument('--organism', required=True, choices=['ecoli', 'bsubtilis'])
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Sample N blocks randomly (0 = all blocks)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    # Set paths
    if args.organism == 'ecoli':
        blocks_file = BENCHMARKS_DIR / 'results' / 'ecoli_chain' / 'micro_chain_blocks.csv'
        genes_file = BENCHMARKS_DIR / 'elsa_output' / 'ecoli' / 'elsa_index' / 'ingest' / 'genes.parquet'
        og_file = BENCHMARKS_DIR / 'ground_truth' / 'orthogroups.tsv'
        output_file = BENCHMARKS_DIR / 'evaluation' / 'ecoli_ortholog_validation.json'
    else:
        blocks_file = BENCHMARKS_DIR / 'results' / 'bacillus_chain' / 'micro_chain_blocks.csv'
        genes_file = BENCHMARKS_DIR / 'elsa_output' / 'bacillus' / 'elsa_index' / 'ingest' / 'genes.parquet'
        og_file = BENCHMARKS_DIR / 'ground_truth' / 'orthogroups_bsubtilis.tsv'
        output_file = BENCHMARKS_DIR / 'evaluation' / 'bsubtilis_ortholog_validation.json'

    print("=" * 60)
    print(f"Validating ELSA blocks against orthogroup data")
    print(f"Organism: {args.organism}")
    print("=" * 60)

    # Check files
    if not og_file.exists():
        print(f"Error: Orthogroup file not found: {og_file}")
        print("Run OrthoFinder first to generate orthogroup assignments.")
        return

    # Load data
    print("\n[1/3] Loading data...")
    blocks_df = pd.read_csv(blocks_file)
    print(f"  Loaded {len(blocks_df)} ELSA blocks")

    genes_df = pd.read_parquet(genes_file)
    print(f"  Loaded {len(genes_df)} genes")

    og_map = load_orthogroups(og_file)
    print(f"  Loaded {len(og_map)} orthogroup assignments")

    # Sample if requested
    if args.sample_size > 0 and args.sample_size < len(blocks_df):
        blocks_df = blocks_df.sample(n=args.sample_size, random_state=42)
        print(f"  Sampled {len(blocks_df)} blocks")

    # Validate blocks
    print("\n[2/3] Validating blocks...")
    results = []

    for i, (_, block) in enumerate(blocks_df.iterrows()):
        result = validate_block(block.to_dict(), genes_df, og_map)
        results.append(result)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(blocks_df)} blocks...")

    # Compute summary statistics
    print("\n[3/3] Computing summary...")
    valid_results = [r for r in results if r.get('valid', False)]

    if not valid_results:
        print("No valid results!")
        return

    ortholog_fractions = [r['min_ortholog_fraction'] for r in valid_results]

    # Thresholds
    thresholds = [0.5, 0.75, 0.9, 0.95]
    threshold_counts = {t: sum(1 for f in ortholog_fractions if f >= t) for t in thresholds}

    summary = {
        'total_blocks': len(blocks_df),
        'validated_blocks': len(valid_results),
        'mean_ortholog_fraction': np.mean(ortholog_fractions),
        'median_ortholog_fraction': np.median(ortholog_fractions),
        'std_ortholog_fraction': np.std(ortholog_fractions),
        'min_ortholog_fraction': np.min(ortholog_fractions),
        'max_ortholog_fraction': np.max(ortholog_fractions),
        'blocks_above_threshold': {
            f'{int(t*100)}%': {
                'count': threshold_counts[t],
                'fraction': threshold_counts[t] / len(valid_results)
            }
            for t in thresholds
        },
    }

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nBlocks validated: {summary['validated_blocks']}/{summary['total_blocks']}")
    print(f"\nOrtholog fraction (genes in shared orthogroups):")
    print(f"  Mean:   {summary['mean_ortholog_fraction']:.1%}")
    print(f"  Median: {summary['median_ortholog_fraction']:.1%}")
    print(f"  Std:    {summary['std_ortholog_fraction']:.1%}")
    print(f"\nBlocks by ortholog threshold:")
    for t in thresholds:
        info = summary['blocks_above_threshold'][f'{int(t*100)}%']
        print(f"  >={t*100:.0f}%: {info['count']} blocks ({info['fraction']:.1%})")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'sample_results': valid_results[:100],
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
