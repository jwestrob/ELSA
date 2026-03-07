#!/usr/bin/env python3
"""
Analyze MCScanX over-prediction patterns.

Quantifies how often MCScanX blocks contain gene pairs that don't follow
expected collinearity patterns:
1. Diagonal coherence - do gene indices increase/decrease monotonically?
2. Gap analysis - identify blocks with large internal gaps
3. Inversion detection - find blocks with mixed orientation signals

This reveals cases where MCScanX creates single large blocks that span
multiple independent syntenic events.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

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


def parse_collinearity_detailed(coll_path: Path, gene_to_idx: dict) -> list:
    """
    Parse MCScanX collinearity file to extract blocks with full gene pair lists.

    Returns list of block dicts with:
    - gene_pairs: list of (idx_a, idx_b) tuples
    - orientation: 'plus' or 'minus'
    - various metadata
    """
    blocks = []
    current_block = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()

            # Block header
            if line.startswith('## Alignment'):
                if current_block and current_block['gene_pairs']:
                    blocks.append(current_block)

                # Parse header
                parts = line.split()
                block_id = int(parts[2].rstrip(':'))

                # Find chromosome info
                chrom_info = None
                for p in parts:
                    if '&' in p:
                        chrom_info = p
                        break

                if chrom_info:
                    chrom_a, chrom_b = chrom_info.split('&')
                    genome_a, contig_a = parse_chrom(chrom_a)
                    genome_b, contig_b = parse_chrom(chrom_b)

                    orientation = 'plus' if 'plus' in line else 'minus'

                    current_block = {
                        'block_id': block_id,
                        'genome_a': genome_a,
                        'genome_b': genome_b,
                        'contig_a': contig_a,
                        'contig_b': contig_b,
                        'orientation': orientation,
                        'gene_pairs': [],
                        'raw_pairs': [],  # Original gene IDs
                    }
                else:
                    current_block = None

            # Gene pair line
            elif current_block and line and not line.startswith('#'):
                # Format: N-  M:[TAB]id_a[TAB]id_b[TAB]e_value
                parts = line.split('\t')
                if len(parts) >= 3:
                    gene_a = parts[1].strip()
                    gene_b = parts[2].strip()

                    current_block['raw_pairs'].append((gene_a, gene_b))

                    if gene_a in gene_to_idx and gene_b in gene_to_idx:
                        _, _, idx_a = gene_to_idx[gene_a]
                        _, _, idx_b = gene_to_idx[gene_b]
                        current_block['gene_pairs'].append((idx_a, idx_b))

    # Don't forget last block
    if current_block and current_block['gene_pairs']:
        blocks.append(current_block)

    return blocks


def analyze_block_coherence(block: dict) -> dict:
    """
    Analyze the diagonal coherence of gene pairs in a block.

    Perfect collinearity: gene indices should increase together (plus)
    or one increases while other decreases (minus).

    Returns metrics measuring deviations from this expectation.
    """
    pairs = block['gene_pairs']
    if len(pairs) < 2:
        return {
            'n_pairs': len(pairs),
            'coherent': True,
            'n_violations': 0,
            'max_gap_a': 0,
            'max_gap_b': 0,
            'diagonal_mad': 0.0,
        }

    # Sort by position in genome A
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    idx_a = [p[0] for p in pairs_sorted]
    idx_b = [p[1] for p in pairs_sorted]

    # Check expected pattern
    expected_orientation = block['orientation']

    # Calculate diagonal offset (should be constant for perfect collinearity)
    # For 'plus': idx_a - idx_b should be constant
    # For 'minus': idx_a + idx_b should be constant
    if expected_orientation == 'plus':
        diagonals = [a - b for a, b in pairs_sorted]
    else:
        diagonals = [a + b for a, b in pairs_sorted]

    diagonal_median = np.median(diagonals)
    diagonal_mad = np.median(np.abs(np.array(diagonals) - diagonal_median))

    # Count monotonicity violations in B sequence
    # For 'plus': idx_b should be monotonically increasing
    # For 'minus': idx_b should be monotonically decreasing
    violations = 0
    for i in range(1, len(idx_b)):
        if expected_orientation == 'plus':
            if idx_b[i] < idx_b[i-1]:
                violations += 1
        else:
            if idx_b[i] > idx_b[i-1]:
                violations += 1

    # Calculate maximum gaps
    gaps_a = [idx_a[i] - idx_a[i-1] for i in range(1, len(idx_a))]
    gaps_b = [abs(idx_b[i] - idx_b[i-1]) for i in range(1, len(idx_b))]

    max_gap_a = max(gaps_a) if gaps_a else 0
    max_gap_b = max(gaps_b) if gaps_b else 0

    # A block is "coherent" if it has few violations and consistent diagonal
    coherent = violations <= len(pairs) * 0.1 and diagonal_mad < 3

    return {
        'n_pairs': len(pairs),
        'coherent': coherent,
        'n_violations': violations,
        'violation_rate': violations / max(1, len(pairs) - 1),
        'max_gap_a': max_gap_a,
        'max_gap_b': max_gap_b,
        'mean_gap_a': np.mean(gaps_a) if gaps_a else 0,
        'mean_gap_b': np.mean(gaps_b) if gaps_b else 0,
        'diagonal_mad': diagonal_mad,
        'diagonal_std': np.std(diagonals) if len(diagonals) > 1 else 0,
        'span_a': max(idx_a) - min(idx_a) + 1,
        'span_b': max(idx_b) - min(idx_b) + 1,
        'coverage_a': len(pairs) / (max(idx_a) - min(idx_a) + 1) if max(idx_a) > min(idx_a) else 1,
        'coverage_b': len(pairs) / (max(idx_b) - min(idx_b) + 1) if max(idx_b) > min(idx_b) else 1,
    }


def detect_sub_blocks(block: dict, max_gap: int = 10) -> list:
    """
    Detect potential sub-blocks within a large MCScanX block.

    Sub-blocks are separated by gaps > max_gap genes in either genome.
    Returns list of (start_pair_idx, end_pair_idx) for each sub-block.
    """
    pairs = block['gene_pairs']
    if len(pairs) < 2:
        return [(0, len(pairs) - 1)]

    # Sort by position in genome A
    pairs_sorted = sorted(enumerate(pairs), key=lambda x: x[1][0])

    sub_blocks = []
    current_start = 0

    for i in range(1, len(pairs_sorted)):
        orig_idx_prev, (a_prev, b_prev) = pairs_sorted[i-1]
        orig_idx_curr, (a_curr, b_curr) = pairs_sorted[i]

        gap_a = a_curr - a_prev
        gap_b = abs(b_curr - b_prev)

        if gap_a > max_gap or gap_b > max_gap:
            # Found a break point
            sub_blocks.append((current_start, i - 1))
            current_start = i

    # Add final sub-block
    sub_blocks.append((current_start, len(pairs) - 1))

    return sub_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcscanx-gff',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.gff',
                        help='MCScanX GFF file')
    parser.add_argument('--mcscanx-collinearity',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.collinearity',
                        help='MCScanX collinearity file')
    parser.add_argument('--output',
                        default=BENCHMARKS_DIR / 'evaluation' / 'mcscanx_overprediction_analysis.md',
                        help='Output report')
    args = parser.parse_args()

    gff_path = Path(args.mcscanx_gff)
    coll_path = Path(args.mcscanx_collinearity)
    output_path = Path(args.output)

    print("=" * 70)
    print("MCScanX Over-Prediction Analysis")
    print("=" * 70)

    print("\n[1/3] Building gene index...")
    gene_to_idx = build_gene_index(gff_path)
    print(f"  Indexed {len(gene_to_idx):,} genes")

    print("\n[2/3] Parsing collinearity file...")
    blocks = parse_collinearity_detailed(coll_path, gene_to_idx)
    print(f"  Parsed {len(blocks):,} blocks")

    print("\n[3/3] Analyzing block coherence...")
    results = []
    for block in blocks:
        coherence = analyze_block_coherence(block)
        sub_blocks = detect_sub_blocks(block, max_gap=10)

        results.append({
            'block_id': block['block_id'],
            'genome_a': block['genome_a'],
            'genome_b': block['genome_b'],
            'orientation': block['orientation'],
            **coherence,
            'n_sub_blocks': len(sub_blocks),
            'is_fragmented': len(sub_blocks) > 1,
        })

    df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Coherence analysis
    n_coherent = df['coherent'].sum()
    n_incoherent = len(df) - n_coherent
    print(f"\nBlock coherence:")
    print(f"  Coherent blocks:   {n_coherent:,} ({n_coherent/len(df)*100:.1f}%)")
    print(f"  Incoherent blocks: {n_incoherent:,} ({n_incoherent/len(df)*100:.1f}%)")

    # Violation analysis
    high_violation = df[df['violation_rate'] > 0.1]
    print(f"\nMonotonicity violations (>10% of pairs):")
    print(f"  Blocks with high violations: {len(high_violation):,} ({len(high_violation)/len(df)*100:.1f}%)")

    # Gap analysis
    large_gap = df[(df['max_gap_a'] > 20) | (df['max_gap_b'] > 20)]
    print(f"\nLarge internal gaps (>20 genes):")
    print(f"  Blocks with large gaps: {len(large_gap):,} ({len(large_gap)/len(df)*100:.1f}%)")

    # Sub-block (fragmentation) analysis
    fragmented = df[df['is_fragmented']]
    print(f"\nPotential fragmentation (could be split into sub-blocks):")
    print(f"  Fragmented blocks: {len(fragmented):,} ({len(fragmented)/len(df)*100:.1f}%)")
    if len(fragmented) > 0:
        print(f"  Mean sub-blocks per fragmented block: {fragmented['n_sub_blocks'].mean():.1f}")
        print(f"  Max sub-blocks in single block: {fragmented['n_sub_blocks'].max()}")

    # Coverage analysis
    low_coverage = df[(df['coverage_a'] < 0.5) | (df['coverage_b'] < 0.5)]
    print(f"\nLow coverage (sparse blocks, <50% of span covered):")
    print(f"  Sparse blocks: {len(low_coverage):,} ({len(low_coverage)/len(df)*100:.1f}%)")

    # Large block analysis (most relevant for over-prediction)
    large_blocks = df[df['span_a'] > 50]
    print(f"\nLarge blocks (>50 genes span):")
    print(f"  Count: {len(large_blocks):,}")
    if len(large_blocks) > 0:
        print(f"  Mean coverage: {large_blocks['coverage_a'].mean():.1%}")
        print(f"  Fragmented: {large_blocks['is_fragmented'].sum():,} ({large_blocks['is_fragmented'].sum()/len(large_blocks)*100:.1f}%)")
        print(f"  High violations: {(large_blocks['violation_rate'] > 0.1).sum():,}")

    # Generate report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# MCScanX Over-Prediction Analysis\n\n")
        f.write("## Summary\n\n")
        f.write("This analysis examines MCScanX blocks for signs of over-prediction:\n")
        f.write("large blocks that combine multiple distinct syntenic events.\n\n")

        f.write("## Key Findings\n\n")

        f.write("### Block Coherence\n\n")
        f.write("A 'coherent' block has consistent gene-to-gene correspondences with:\n")
        f.write("- Few monotonicity violations (genes stay in order)\n")
        f.write("- Low diagonal deviation (consistent offset between genomes)\n\n")
        f.write(f"| Category | Count | Percentage |\n")
        f.write(f"|----------|-------|------------|\n")
        f.write(f"| Coherent | {n_coherent:,} | {n_coherent/len(df)*100:.1f}% |\n")
        f.write(f"| Incoherent | {n_incoherent:,} | {n_incoherent/len(df)*100:.1f}% |\n\n")

        f.write("### Fragmentation Potential\n\n")
        f.write("Blocks that could be split into smaller, more coherent sub-blocks\n")
        f.write("(detected via gaps >10 genes in either genome):\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Fragmented blocks | {len(fragmented):,} ({len(fragmented)/len(df)*100:.1f}%) |\n")
        if len(fragmented) > 0:
            f.write(f"| Mean sub-blocks | {fragmented['n_sub_blocks'].mean():.1f} |\n")
            f.write(f"| Max sub-blocks | {fragmented['n_sub_blocks'].max()} |\n")
        f.write("\n")

        f.write("### Large Block Analysis (>50 genes)\n\n")
        f.write("Large blocks are most susceptible to over-prediction:\n\n")
        if len(large_blocks) > 0:
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total large blocks | {len(large_blocks):,} |\n")
            f.write(f"| Mean span (genome A) | {large_blocks['span_a'].mean():.1f} genes |\n")
            f.write(f"| Mean coverage | {large_blocks['coverage_a'].mean():.1%} |\n")
            f.write(f"| Fragmented | {large_blocks['is_fragmented'].sum()} ({large_blocks['is_fragmented'].sum()/len(large_blocks)*100:.1f}%) |\n")
            f.write(f"| High violation rate | {(large_blocks['violation_rate'] > 0.1).sum()} |\n")
        f.write("\n")

        f.write("### Sparse Blocks\n\n")
        f.write("Blocks where matched gene pairs cover <50% of the genomic span:\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Sparse blocks | {len(low_coverage):,} ({len(low_coverage)/len(df)*100:.1f}%) |\n")
        if len(low_coverage) > 0:
            f.write(f"| Mean coverage (A) | {low_coverage['coverage_a'].mean():.1%} |\n")
            f.write(f"| Mean span (A) | {low_coverage['span_a'].mean():.1f} genes |\n")
        f.write("\n")

        f.write("## Interpretation\n\n")
        f.write("**Fragmented blocks** suggest MCScanX may be joining distinct syntenic\n")
        f.write("regions that happen to have similar BLAST hit patterns. ELSA's embedding-based\n")
        f.write("approach with chaining would separate these into distinct blocks.\n\n")

        f.write("**Sparse blocks** indicate that while MCScanX found BLAST hits spanning\n")
        f.write("a large region, many intermediate genes don't have correspondences.\n")
        f.write("This can lead to 'accidental' operon coverage where the block spans\n")
        f.write("an operon without the operon genes actually matching.\n\n")

        f.write("## Top 10 Most Fragmented Large Blocks\n\n")
        f.write("These blocks may represent multiple independent syntenic events merged together:\n\n")
        top_fragmented = large_blocks.nlargest(10, 'n_sub_blocks')[
            ['block_id', 'genome_a', 'genome_b', 'span_a', 'n_pairs', 'n_sub_blocks', 'coverage_a']
        ]
        if len(top_fragmented) > 0:
            f.write("| Block ID | Genomes | Span | Pairs | Sub-blocks | Coverage |\n")
            f.write("|----------|---------|------|-------|------------|----------|\n")
            for _, row in top_fragmented.iterrows():
                f.write(f"| {row['block_id']} | {row['genome_a']}↔{row['genome_b']} | ")
                f.write(f"{row['span_a']} | {row['n_pairs']} | {row['n_sub_blocks']} | {row['coverage_a']:.1%} |\n")

    print(f"\nReport saved to: {output_path}")

    # Save detailed data
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed data saved to: {csv_path}")


if __name__ == '__main__':
    main()
