#!/usr/bin/env python3
"""
Analyze block fragmentation patterns between ELSA and MCScanX.

Identifies:
1. ELSA block pairs that fall within single MCScanX blocks
2. Whether these represent true separate syntenic events or fragmentation artifacts
3. Evidence of inversions, insertions, or rearrangements between ELSA blocks

This helps understand:
- When ELSA correctly splits merged MCScanX blocks
- When ELSA over-fragments continuous syntenic regions
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


def parse_chrom(chrom: str) -> tuple:
    """Parse genome_contig into (genome, contig)."""
    m = re.match(r'(GCF_\d+\.\d+)_(.+)', chrom)
    if m:
        return m.group(1), m.group(2)
    return None, None


def build_gene_index(gff_path: Path) -> dict:
    """Build mapping from MCScanX internal ID to (genome, contig, gene_idx)."""
    chrom_genes = defaultdict(list)

    with open(gff_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom = parts[0]
                internal_id = parts[1]
                start = int(parts[2])
                chrom_genes[chrom].append((start, internal_id))

    gene_to_idx = {}
    for chrom, genes in chrom_genes.items():
        genes.sort(key=lambda x: x[0])
        genome, contig = parse_chrom(chrom)
        if genome is None:
            continue
        for gene_idx, (start, internal_id) in enumerate(genes):
            gene_to_idx[internal_id] = (genome, contig, gene_idx)

    return gene_to_idx


def parse_mcscanx_blocks(coll_path: Path, gene_to_idx: dict) -> list:
    """Parse MCScanX collinearity file."""
    blocks = []
    current_block = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith('## Alignment'):
                if current_block and current_block['gene_pairs']:
                    pairs = current_block['gene_pairs']
                    current_block['query_start'] = min(p[0] for p in pairs)
                    current_block['query_end'] = max(p[0] for p in pairs)
                    current_block['target_start'] = min(p[1] for p in pairs)
                    current_block['target_end'] = max(p[1] for p in pairs)
                    current_block['n_genes'] = len(pairs)
                    blocks.append(current_block)

                parts = line.split()
                block_id = int(parts[2].rstrip(':'))

                chrom_info = None
                orientation = 'plus' if 'plus' in line else 'minus'
                for p in parts:
                    if '&' in p:
                        chrom_info = p
                        break

                if chrom_info:
                    chrom_a, chrom_b = chrom_info.split('&')
                    genome_a, contig_a = parse_chrom(chrom_a)
                    genome_b, contig_b = parse_chrom(chrom_b)

                    current_block = {
                        'block_id': block_id,
                        'genome_a': genome_a,
                        'genome_b': genome_b,
                        'contig_a': contig_a,
                        'contig_b': contig_b,
                        'orientation': orientation,
                        'gene_pairs': [],
                    }
                else:
                    current_block = None

            elif current_block and line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    gene_a = parts[1]
                    gene_b = parts[2]

                    if gene_a in gene_to_idx and gene_b in gene_to_idx:
                        _, _, idx_a = gene_to_idx[gene_a]
                        _, _, idx_b = gene_to_idx[gene_b]
                        current_block['gene_pairs'].append((idx_a, idx_b))

    if current_block and current_block['gene_pairs']:
        pairs = current_block['gene_pairs']
        current_block['query_start'] = min(p[0] for p in pairs)
        current_block['query_end'] = max(p[0] for p in pairs)
        current_block['target_start'] = min(p[1] for p in pairs)
        current_block['target_end'] = max(p[1] for p in pairs)
        current_block['n_genes'] = len(pairs)
        blocks.append(current_block)

    return blocks


def ranges_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two ranges overlap."""
    return start1 <= end2 and start2 <= end1


def contains_range(outer_start: int, outer_end: int, inner_start: int, inner_end: int) -> bool:
    """Check if outer range fully contains inner range."""
    return outer_start <= inner_start and outer_end >= inner_end


def find_elsa_blocks_in_mcscanx(elsa_blocks: pd.DataFrame, mcscanx_blocks: list) -> list:
    """
    Find ELSA blocks that fall within single MCScanX blocks.

    Returns list of (mcscanx_block, [elsa_blocks]) tuples.
    """
    # Index MCScanX blocks by genome pair and contig
    mcscanx_index = defaultdict(list)
    for block in mcscanx_blocks:
        key = (block['genome_a'], block['genome_b'], block['contig_a'], block['contig_b'])
        mcscanx_index[key].append(block)
        # Also add reverse key
        key_rev = (block['genome_b'], block['genome_a'], block['contig_b'], block['contig_a'])
        mcscanx_index[key_rev].append(block)

    results = []

    for mcs_block in mcscanx_blocks:
        key = (mcs_block['genome_a'], mcs_block['genome_b'],
               mcs_block['contig_a'], mcs_block['contig_b'])

        # Find ELSA blocks for this genome/contig pair
        elsa_matching = elsa_blocks[
            ((elsa_blocks['query_genome'] == mcs_block['genome_a']) &
             (elsa_blocks['target_genome'] == mcs_block['genome_b']) &
             (elsa_blocks['query_contig'] == mcs_block['contig_a']) &
             (elsa_blocks['target_contig'] == mcs_block['contig_b'])) |
            ((elsa_blocks['query_genome'] == mcs_block['genome_b']) &
             (elsa_blocks['target_genome'] == mcs_block['genome_a']) &
             (elsa_blocks['query_contig'] == mcs_block['contig_b']) &
             (elsa_blocks['target_contig'] == mcs_block['contig_a']))
        ]

        contained_elsa = []

        for _, elsa in elsa_matching.iterrows():
            # Normalize orientation
            if elsa['query_genome'] == mcs_block['genome_a']:
                e_start_a, e_end_a = elsa['query_start'], elsa['query_end']
                e_start_b, e_end_b = elsa['target_start'], elsa['target_end']
            else:
                e_start_a, e_end_a = elsa['target_start'], elsa['target_end']
                e_start_b, e_end_b = elsa['query_start'], elsa['query_end']

            # Check if ELSA block is contained within MCScanX block
            if (contains_range(mcs_block['query_start'], mcs_block['query_end'],
                              e_start_a, e_end_a) and
                contains_range(mcs_block['target_start'], mcs_block['target_end'],
                              e_start_b, e_end_b)):
                contained_elsa.append({
                    'block_id': elsa['block_id'],
                    'start_a': e_start_a,
                    'end_a': e_end_a,
                    'start_b': e_start_b,
                    'end_b': e_end_b,
                    'n_genes': elsa['n_genes'],
                    'orientation': elsa['orientation'],
                })

        if len(contained_elsa) >= 2:
            results.append({
                'mcscanx_block': mcs_block,
                'elsa_blocks': contained_elsa,
                'n_elsa_blocks': len(contained_elsa),
            })

    return results


def analyze_fragmentation(container: dict) -> dict:
    """
    Analyze whether ELSA blocks within a MCScanX block represent true separate events.

    Evidence of true separation:
    - Gaps between ELSA blocks (insertions)
    - Different orientations (inversions)
    - Non-overlapping gene ranges

    Evidence of over-fragmentation:
    - Adjacent blocks with same orientation
    - Blocks that could merge into one continuous chain
    """
    mcs = container['mcscanx_block']
    elsa_blocks = sorted(container['elsa_blocks'], key=lambda x: x['start_a'])

    analysis = {
        'mcscanx_id': mcs['block_id'],
        'mcscanx_span_a': mcs['query_end'] - mcs['query_start'] + 1,
        'mcscanx_span_b': mcs['target_end'] - mcs['target_start'] + 1,
        'mcscanx_n_genes': mcs['n_genes'],
        'n_elsa_blocks': len(elsa_blocks),
        'total_elsa_genes': sum(b['n_genes'] for b in elsa_blocks),
    }

    # Check for orientation differences
    orientations = set(b['orientation'] for b in elsa_blocks)
    analysis['has_mixed_orientations'] = len(orientations) > 1
    analysis['n_positive'] = sum(1 for b in elsa_blocks if b['orientation'] == 1)
    analysis['n_negative'] = sum(1 for b in elsa_blocks if b['orientation'] == -1)

    # Check gaps between consecutive blocks
    gaps_a = []
    gaps_b = []
    for i in range(1, len(elsa_blocks)):
        prev = elsa_blocks[i-1]
        curr = elsa_blocks[i]

        gap_a = curr['start_a'] - prev['end_a'] - 1
        gaps_a.append(gap_a)

        # For gaps in B, need to consider orientation
        if prev['orientation'] == curr['orientation'] == 1:
            gap_b = curr['start_b'] - prev['end_b'] - 1
        elif prev['orientation'] == curr['orientation'] == -1:
            gap_b = prev['start_b'] - curr['end_b'] - 1
        else:
            # Mixed orientation - can't compute meaningful gap
            gap_b = None

        if gap_b is not None:
            gaps_b.append(gap_b)

    analysis['mean_gap_a'] = np.mean(gaps_a) if gaps_a else 0
    analysis['max_gap_a'] = max(gaps_a) if gaps_a else 0
    analysis['mean_gap_b'] = np.mean(gaps_b) if gaps_b else 0
    analysis['max_gap_b'] = max(gaps_b) if gaps_b else 0

    # Determine fragmentation type
    if analysis['has_mixed_orientations']:
        analysis['fragmentation_type'] = 'inversion'
        analysis['likely_true_separation'] = True
    elif analysis['max_gap_a'] > 10 or analysis['max_gap_b'] > 10:
        analysis['fragmentation_type'] = 'insertion'
        analysis['likely_true_separation'] = True
    elif analysis['mean_gap_a'] < 2 and analysis['mean_gap_b'] < 2:
        analysis['fragmentation_type'] = 'over_fragmentation'
        analysis['likely_true_separation'] = False
    else:
        analysis['fragmentation_type'] = 'moderate_gap'
        analysis['likely_true_separation'] = analysis['mean_gap_a'] > 5

    # Coverage analysis
    total_elsa_span_a = sum(b['end_a'] - b['start_a'] + 1 for b in elsa_blocks)
    analysis['elsa_coverage_a'] = total_elsa_span_a / analysis['mcscanx_span_a']

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elsa-blocks',
                        default=BENCHMARKS_DIR / 'results' / 'cross_species_chain' / 'micro_chain' / 'micro_chain_blocks.csv')
    parser.add_argument('--mcscanx-gff',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.gff')
    parser.add_argument('--mcscanx-collinearity',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.collinearity')
    parser.add_argument('--output',
                        default=BENCHMARKS_DIR / 'evaluation' / 'fragmentation_analysis.md')
    args = parser.parse_args()

    output_path = Path(args.output)

    print("=" * 70)
    print("Block Fragmentation Analysis")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading ELSA blocks...")
    elsa_blocks = pd.read_csv(args.elsa_blocks)
    print(f"  Loaded {len(elsa_blocks):,} ELSA blocks")

    print("\n[2/4] Building gene index from MCScanX GFF...")
    gene_to_idx = build_gene_index(Path(args.mcscanx_gff))
    print(f"  Indexed {len(gene_to_idx):,} genes")

    print("\n[3/4] Parsing MCScanX blocks...")
    mcscanx_blocks = parse_mcscanx_blocks(Path(args.mcscanx_collinearity), gene_to_idx)
    print(f"  Parsed {len(mcscanx_blocks):,} blocks")

    print("\n[4/4] Analyzing fragmentation patterns...")
    containers = find_elsa_blocks_in_mcscanx(elsa_blocks, mcscanx_blocks)
    print(f"  Found {len(containers):,} MCScanX blocks containing multiple ELSA blocks")

    # Analyze each container
    analyses = []
    for container in containers:
        analysis = analyze_fragmentation(container)
        analyses.append(analysis)

    df = pd.DataFrame(analyses)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if len(df) > 0:
        print(f"\nMCScanX blocks with multiple ELSA blocks: {len(df):,}")
        print(f"Total ELSA blocks in these containers: {df['n_elsa_blocks'].sum():,}")
        print(f"Mean ELSA blocks per container: {df['n_elsa_blocks'].mean():.1f}")
        print(f"Max ELSA blocks in single container: {df['n_elsa_blocks'].max()}")

        print("\n### Fragmentation Types ###")
        for frag_type in df['fragmentation_type'].unique():
            count = (df['fragmentation_type'] == frag_type).sum()
            print(f"  {frag_type}: {count:,} ({count/len(df)*100:.1f}%)")

        print("\n### Likely True Separations ###")
        true_sep = df['likely_true_separation'].sum()
        print(f"  True separation: {true_sep:,} ({true_sep/len(df)*100:.1f}%)")
        print(f"  Over-fragmentation: {len(df) - true_sep:,} ({(len(df)-true_sep)/len(df)*100:.1f}%)")

        print("\n### Mixed Orientation (Inversions) ###")
        mixed = df['has_mixed_orientations'].sum()
        print(f"  Containers with inversions: {mixed:,} ({mixed/len(df)*100:.1f}%)")

        print("\n### Gap Statistics ###")
        print(f"  Mean gap in genome A: {df['mean_gap_a'].mean():.1f} genes")
        print(f"  Mean max gap in genome A: {df['max_gap_a'].mean():.1f} genes")

    # Generate report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Block Fragmentation Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis examines cases where multiple ELSA blocks fall within\n")
        f.write("a single MCScanX block, to understand:\n\n")
        f.write("1. **True separation**: ELSA correctly identifies distinct syntenic events\n")
        f.write("   that MCScanX incorrectly merged (inversions, insertions, rearrangements)\n\n")
        f.write("2. **Over-fragmentation**: ELSA unnecessarily splits continuous syntenic\n")
        f.write("   regions that MCScanX correctly unified\n\n")

        f.write("## Summary\n\n")

        if len(df) > 0:
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| MCScanX blocks with multiple ELSA blocks | {len(df):,} |\n")
            f.write(f"| Total ELSA blocks in containers | {df['n_elsa_blocks'].sum():,} |\n")
            f.write(f"| Mean ELSA blocks per container | {df['n_elsa_blocks'].mean():.1f} |\n")
            f.write(f"| Max ELSA blocks in container | {df['n_elsa_blocks'].max()} |\n\n")

            f.write("## Fragmentation Classification\n\n")
            f.write("| Type | Count | Percentage | Description |\n")
            f.write("|------|-------|------------|-------------|\n")

            type_descriptions = {
                'inversion': 'ELSA blocks have different orientations (true rearrangement)',
                'insertion': 'Large gaps between blocks suggest insertions/deletions',
                'moderate_gap': 'Moderate gaps may indicate true or false separation',
                'over_fragmentation': 'Small gaps suggest ELSA over-split a continuous region',
            }

            for frag_type in ['inversion', 'insertion', 'moderate_gap', 'over_fragmentation']:
                count = (df['fragmentation_type'] == frag_type).sum()
                desc = type_descriptions.get(frag_type, '')
                f.write(f"| {frag_type} | {count:,} | {count/len(df)*100:.1f}% | {desc} |\n")

            f.write("\n## Assessment Summary\n\n")

            true_sep = df['likely_true_separation'].sum()
            over_frag = len(df) - true_sep

            f.write(f"| Assessment | Count | Percentage |\n")
            f.write(f"|------------|-------|------------|\n")
            f.write(f"| Likely true separation | {true_sep:,} | {true_sep/len(df)*100:.1f}% |\n")
            f.write(f"| Possible over-fragmentation | {over_frag:,} | {over_frag/len(df)*100:.1f}% |\n\n")

            f.write("## Interpretation\n\n")

            inversion_count = (df['fragmentation_type'] == 'inversion').sum()
            insertion_count = (df['fragmentation_type'] == 'insertion').sum()

            f.write(f"### Evidence of Correct ELSA Splitting\n\n")
            f.write(f"**{inversion_count:,} cases of inversions**: ELSA correctly separated\n")
            f.write(f"regions with different orientations that MCScanX merged together.\n\n")
            f.write(f"**{insertion_count:,} cases of insertions**: Large gaps (>10 genes)\n")
            f.write(f"between ELSA blocks suggest true structural differences.\n\n")

            f.write(f"### Potential Over-fragmentation\n\n")
            f.write(f"**{over_frag:,} cases** where ELSA may have unnecessarily split\n")
            f.write(f"continuous syntenic regions. This could be due to:\n")
            f.write(f"- Embedding similarity threshold being too strict\n")
            f.write(f"- Chaining algorithm breaking on minor variations\n")
            f.write(f"- True but small structural variations ELSA detected\n\n")

            # Add example cases
            f.write("## Example Cases\n\n")

            # Show an inversion example
            inversions = df[df['fragmentation_type'] == 'inversion']
            if len(inversions) > 0:
                ex = inversions.iloc[0]
                f.write("### Inversion Example\n\n")
                f.write(f"MCScanX block {ex['mcscanx_id']} (span: {ex['mcscanx_span_a']} genes)\n")
                f.write(f"contains {ex['n_elsa_blocks']} ELSA blocks:\n")
                f.write(f"- Positive orientation: {ex['n_positive']}\n")
                f.write(f"- Negative orientation: {ex['n_negative']}\n\n")
                f.write("This is a clear case where ELSA correctly identified an inversion.\n\n")

            # Show over-fragmentation example
            over_frags = df[df['fragmentation_type'] == 'over_fragmentation']
            if len(over_frags) > 0:
                ex = over_frags.iloc[0]
                f.write("### Over-fragmentation Example\n\n")
                f.write(f"MCScanX block {ex['mcscanx_id']} (span: {ex['mcscanx_span_a']} genes)\n")
                f.write(f"contains {ex['n_elsa_blocks']} ELSA blocks with:\n")
                f.write(f"- Mean gap: {ex['mean_gap_a']:.1f} genes\n")
                f.write(f"- Max gap: {ex['max_gap_a']:.0f} genes\n\n")
                f.write("Small gaps suggest these could potentially be merged.\n\n")

        else:
            f.write("No MCScanX blocks were found containing multiple ELSA blocks.\n")
            f.write("This could indicate that ELSA's block boundaries largely align with MCScanX.\n")

    print(f"\nReport saved to: {output_path}")

    # Save detailed data
    if len(df) > 0:
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Detailed data saved to: {csv_path}")


if __name__ == '__main__':
    main()
