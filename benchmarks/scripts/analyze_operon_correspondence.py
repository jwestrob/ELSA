#!/usr/bin/env python3
"""
Analyze operon-level gene correspondence for MCScanX "wins".

For cases where MCScanX achieved strict recall but ELSA didn't, verify
whether the operon genes actually correspond to each other within the
MCScanX block.

This distinguishes:
1. True correspondence: Operon genes in genome A map to operon genes in genome B
2. Accidental span: Block spans both operon positions, but genes don't correspond

The key insight from the plan: MCScanX's "strict recall" may be artificially
inflated by large blocks that accidentally span operon positions without
the operon genes actually mapping to each other.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

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
    idx_to_gene = {}

    for chrom, genes in chrom_genes.items():
        genes.sort(key=lambda x: x[0])
        genome, contig = parse_chrom(chrom)
        if genome is None:
            continue
        for gene_idx, (start, internal_id) in enumerate(genes):
            gene_to_idx[internal_id] = (genome, contig, gene_idx)
            idx_to_gene[(genome, contig, gene_idx)] = internal_id

    return gene_to_idx, idx_to_gene


def parse_mcscanx_blocks(coll_path: Path, gene_to_idx: dict) -> list:
    """Parse MCScanX collinearity file with full gene pair information."""
    blocks = []
    current_block = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith('## Alignment'):
                if current_block and current_block['gene_pairs']:
                    current_block['query_start'] = min(p[0] for p in current_block['gene_pairs'])
                    current_block['query_end'] = max(p[0] for p in current_block['gene_pairs'])
                    current_block['target_start'] = min(p[1] for p in current_block['gene_pairs'])
                    current_block['target_end'] = max(p[1] for p in current_block['gene_pairs'])
                    blocks.append(current_block)

                parts = line.split()
                block_id = int(parts[2].rstrip(':'))

                chrom_info = None
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
                        'gene_pairs': [],
                        'gene_pair_map': {},  # idx_a -> idx_b
                    }
                else:
                    current_block = None

            elif current_block and line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    gene_a = parts[1].strip()
                    gene_b = parts[2].strip()

                    if gene_a in gene_to_idx and gene_b in gene_to_idx:
                        _, _, idx_a = gene_to_idx[gene_a]
                        _, _, idx_b = gene_to_idx[gene_b]
                        current_block['gene_pairs'].append((idx_a, idx_b))
                        current_block['gene_pair_map'][idx_a] = idx_b

    if current_block and current_block['gene_pairs']:
        current_block['query_start'] = min(p[0] for p in current_block['gene_pairs'])
        current_block['query_end'] = max(p[0] for p in current_block['gene_pairs'])
        current_block['target_start'] = min(p[1] for p in current_block['gene_pairs'])
        current_block['target_end'] = max(p[1] for p in current_block['gene_pairs'])
        blocks.append(current_block)

    return blocks


def check_overlap(block_start, block_end, operon_start, operon_end, threshold=0.5):
    """Check if block overlaps operon by at least threshold fraction."""
    overlap_start = max(block_start, operon_start)
    overlap_end = min(block_end, operon_end)

    if overlap_start > overlap_end:
        return False, 0.0

    overlap_size = overlap_end - overlap_start + 1
    operon_size = operon_end - operon_start + 1

    overlap_frac = overlap_size / operon_size
    return overlap_frac >= threshold, overlap_frac


def analyze_operon_correspondence(operon: pd.Series, blocks: list,
                                   threshold: float = 0.5) -> dict:
    """
    For a given operon, find MCScanX blocks that achieve strict recall
    and check if the operon genes actually correspond.
    """
    genome_a = operon['genome_a']
    genome_b = operon['genome_b']
    contig_a = operon['contig_a']
    contig_b = operon['contig_b']

    op_start_a = operon['gene_idx_start_a']
    op_end_a = operon['gene_idx_end_a']
    op_start_b = operon['gene_idx_start_b']
    op_end_b = operon['gene_idx_end_b']

    # Find matching blocks
    matching_blocks = [b for b in blocks
                       if (b['genome_a'] == genome_a and b['genome_b'] == genome_b and
                           b['contig_a'] == contig_a and b['contig_b'] == contig_b) or
                          (b['genome_a'] == genome_b and b['genome_b'] == genome_a and
                           b['contig_a'] == contig_b and b['contig_b'] == contig_a)]

    result = {
        'operon_id': operon['operon_id'],
        'genome_a': genome_a,
        'genome_b': genome_b,
        'operon_size_a': op_end_a - op_start_a + 1,
        'operon_size_b': op_end_b - op_start_b + 1,
        'has_strict_block': False,
        'strict_block_ids': [],
        'genes_correspond': False,
        'correspondence_score': 0.0,
        'block_span_a': 0,
        'block_span_b': 0,
        'classification': 'no_block',
    }

    strict_blocks = []

    for block in matching_blocks:
        # Normalize orientation
        if block['genome_a'] == genome_a:
            b_start_a = block['query_start']
            b_end_a = block['query_end']
            b_start_b = block['target_start']
            b_end_b = block['target_end']
            pair_map = block['gene_pair_map']
        else:
            b_start_a = block['target_start']
            b_end_a = block['target_end']
            b_start_b = block['query_start']
            b_end_b = block['query_end']
            # Invert the pair map
            pair_map = {v: k for k, v in block['gene_pair_map'].items()}

        overlap_a, frac_a = check_overlap(b_start_a, b_end_a, op_start_a, op_end_a, threshold)
        overlap_b, frac_b = check_overlap(b_start_b, b_end_b, op_start_b, op_end_b, threshold)

        if overlap_a and overlap_b:
            strict_blocks.append({
                'block': block,
                'pair_map': pair_map,
                'span_a': b_end_a - b_start_a + 1,
                'span_b': b_end_b - b_start_b + 1,
            })

    if not strict_blocks:
        return result

    result['has_strict_block'] = True
    result['strict_block_ids'] = [b['block']['block_id'] for b in strict_blocks]

    # Check gene correspondence for each strict block
    best_correspondence = 0.0
    best_block_info = None

    for block_info in strict_blocks:
        pair_map = block_info['pair_map']

        # Count how many operon genes in A map to operon genes in B
        operon_genes_a = set(range(op_start_a, op_end_a + 1))
        operon_genes_b = set(range(op_start_b, op_end_b + 1))

        corresponding_count = 0
        for gene_a in operon_genes_a:
            if gene_a in pair_map:
                mapped_b = pair_map[gene_a]
                if mapped_b in operon_genes_b:
                    corresponding_count += 1

        # Correspondence score: fraction of operon A genes that map to operon B genes
        correspondence = corresponding_count / len(operon_genes_a) if operon_genes_a else 0

        if correspondence > best_correspondence:
            best_correspondence = correspondence
            best_block_info = block_info

    result['correspondence_score'] = best_correspondence
    result['genes_correspond'] = best_correspondence >= 0.5

    if best_block_info:
        result['block_span_a'] = best_block_info['span_a']
        result['block_span_b'] = best_block_info['span_b']

    # Classification
    if best_correspondence >= 0.9:
        result['classification'] = 'true_correspondence'
    elif best_correspondence >= 0.5:
        result['classification'] = 'partial_correspondence'
    elif best_correspondence > 0:
        result['classification'] = 'weak_correspondence'
    else:
        result['classification'] = 'accidental_span'

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operon-gt',
                        default=BENCHMARKS_DIR / 'ground_truth' / 'ecoli_operon_gt_v2.tsv')
    parser.add_argument('--elsa-blocks',
                        default=BENCHMARKS_DIR / 'results' / 'cross_species_chain' / 'micro_chain' / 'micro_chain_blocks.csv')
    parser.add_argument('--mcscanx-gff',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.gff')
    parser.add_argument('--mcscanx-collinearity',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.collinearity')
    parser.add_argument('--output',
                        default=BENCHMARKS_DIR / 'evaluation' / 'operon_correspondence_analysis.md')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    output_path = Path(args.output)

    print("=" * 70)
    print("Operon Correspondence Analysis")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading operon ground truth...")
    operon_gt = pd.read_csv(args.operon_gt, sep='\t')
    print(f"  Loaded {len(operon_gt):,} operon instances")

    print("\n[2/4] Building gene index from MCScanX GFF...")
    gene_to_idx, idx_to_gene = build_gene_index(Path(args.mcscanx_gff))
    print(f"  Indexed {len(gene_to_idx):,} genes")

    print("\n[3/4] Parsing MCScanX blocks...")
    mcscanx_blocks = parse_mcscanx_blocks(Path(args.mcscanx_collinearity), gene_to_idx)
    print(f"  Parsed {len(mcscanx_blocks):,} blocks")

    # Filter to E. coli genomes
    gt_genomes = set(operon_gt['genome_a'].unique()) | set(operon_gt['genome_b'].unique())
    mcscanx_blocks = [b for b in mcscanx_blocks
                      if b['genome_a'] in gt_genomes and b['genome_b'] in gt_genomes]
    print(f"  Filtered to {len(mcscanx_blocks):,} E. coli blocks")

    print("\n[4/4] Analyzing operon correspondence...")
    results = []
    for _, operon in operon_gt.iterrows():
        result = analyze_operon_correspondence(operon, mcscanx_blocks, args.threshold)
        results.append(result)

    df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total = len(df)
    has_strict = df['has_strict_block'].sum()

    print(f"\nTotal operon instances: {total:,}")
    print(f"Operons with MCScanX strict coverage: {has_strict:,} ({has_strict/total*100:.1f}%)")

    if has_strict > 0:
        strict_df = df[df['has_strict_block']]

        print(f"\n### Classification of MCScanX 'strict recall' cases ###")
        for classification in ['true_correspondence', 'partial_correspondence',
                               'weak_correspondence', 'accidental_span']:
            count = (strict_df['classification'] == classification).sum()
            pct = count / has_strict * 100
            print(f"  {classification}: {count:,} ({pct:.1f}%)")

        print(f"\n### Correspondence scores for strict blocks ###")
        print(f"  Mean: {strict_df['correspondence_score'].mean():.1%}")
        print(f"  Median: {strict_df['correspondence_score'].median():.1%}")
        print(f"  Std: {strict_df['correspondence_score'].std():.1%}")

        true_corr = (strict_df['correspondence_score'] >= 0.5).sum()
        accidental = (strict_df['correspondence_score'] < 0.5).sum()
        print(f"\n### Summary ###")
        print(f"  True correspondence (≥50%): {true_corr:,} ({true_corr/has_strict*100:.1f}%)")
        print(f"  Accidental span (<50%): {accidental:,} ({accidental/has_strict*100:.1f}%)")

        # Block size analysis for accidental spans
        accidental_df = strict_df[strict_df['correspondence_score'] < 0.5]
        if len(accidental_df) > 0:
            print(f"\n### Accidental span block characteristics ###")
            print(f"  Mean block span (A): {accidental_df['block_span_a'].mean():.1f} genes")
            print(f"  Mean block span (B): {accidental_df['block_span_b'].mean():.1f} genes")
            print(f"  Mean operon size: {accidental_df['operon_size_a'].mean():.1f} genes")

    # Generate report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Operon Correspondence Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis examines whether MCScanX's 'strict recall' on operons\n")
        f.write("represents true gene-to-gene correspondence or accidental span.\n\n")

        f.write("### Key Question\n\n")
        f.write("When MCScanX reports a block that covers an operon in both genomes,\n")
        f.write("do the operon genes in genome A actually map to operon genes in genome B\n")
        f.write("within that block's collinearity file?\n\n")

        f.write("## Results Summary\n\n")

        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total operon instances | {total:,} |\n")
        f.write(f"| MCScanX strict coverage | {has_strict:,} ({has_strict/total*100:.1f}%) |\n\n")

        if has_strict > 0:
            strict_df = df[df['has_strict_block']]

            f.write("### Classification of Strict Recall Cases\n\n")
            f.write("| Classification | Count | Percentage |\n")
            f.write("|----------------|-------|------------|\n")

            for classification, label in [
                ('true_correspondence', 'True correspondence (≥90%)'),
                ('partial_correspondence', 'Partial correspondence (50-89%)'),
                ('weak_correspondence', 'Weak correspondence (1-49%)'),
                ('accidental_span', 'Accidental span (0%)'),
            ]:
                count = (strict_df['classification'] == classification).sum()
                f.write(f"| {label} | {count:,} | {count/has_strict*100:.1f}% |\n")

            f.write("\n### Correspondence Score Distribution\n\n")
            f.write(f"| Statistic | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| Mean | {strict_df['correspondence_score'].mean():.1%} |\n")
            f.write(f"| Median | {strict_df['correspondence_score'].median():.1%} |\n")
            f.write(f"| Std Dev | {strict_df['correspondence_score'].std():.1%} |\n")
            f.write(f"| Min | {strict_df['correspondence_score'].min():.1%} |\n")
            f.write(f"| Max | {strict_df['correspondence_score'].max():.1%} |\n\n")

            # Calculate adjusted strict recall
            true_strict = (strict_df['correspondence_score'] >= 0.5).sum()
            adjusted_strict_recall = true_strict / total

            f.write("## Adjusted Metrics\n\n")
            f.write("If we require actual gene correspondence (≥50% of operon genes map\n")
            f.write("to each other), the metrics change:\n\n")
            f.write(f"| Metric | Original | Adjusted |\n")
            f.write(f"|--------|----------|----------|\n")
            f.write(f"| MCScanX strict recall | {has_strict/total*100:.1f}% | {adjusted_strict_recall*100:.1f}% |\n")
            f.write(f"| Cases counted | {has_strict:,} | {true_strict:,} |\n\n")

            accidental_count = has_strict - true_strict
            f.write(f"**Implication**: {accidental_count:,} ({accidental_count/has_strict*100:.1f}%) of\n")
            f.write(f"MCScanX's 'strict recall' cases are likely false positives where large\n")
            f.write(f"blocks accidentally span operon positions without genes corresponding.\n\n")

        f.write("## Interpretation\n\n")
        f.write("### Why Accidental Spans Occur\n\n")
        f.write("1. **Large blocks**: MCScanX creates large collinear blocks from BLAST hits.\n")
        f.write("   A 400-gene block might span multiple small operons (2-10 genes each)\n")
        f.write("   without the operon genes being explicitly linked.\n\n")
        f.write("2. **Sparse coverage**: Within large blocks, not every gene has a BLAST hit.\n")
        f.write("   The block 'spans' positions without having gene correspondences there.\n\n")
        f.write("3. **Genome rearrangements**: Operons may exist at different relative positions\n")
        f.write("   in each genome. A block might span both positions accidentally.\n\n")

        f.write("### ELSA's Approach\n\n")
        f.write("ELSA's embedding-based chaining creates smaller, more precise blocks.\n")
        f.write("While this may result in lower 'strict recall' (same block covering both),\n")
        f.write("the gene correspondences within blocks are more reliable.\n\n")

        f.write("## Case Studies\n\n")

        # Show examples of each type
        if has_strict > 0:
            for classification in ['accidental_span', 'true_correspondence']:
                examples = strict_df[strict_df['classification'] == classification].head(3)
                if len(examples) > 0:
                    f.write(f"### {classification.replace('_', ' ').title()} Examples\n\n")
                    for _, ex in examples.iterrows():
                        f.write(f"**{ex['operon_id']}** ({ex['genome_a']} ↔ {ex['genome_b']})\n")
                        f.write(f"- Operon size: {ex['operon_size_a']} genes\n")
                        f.write(f"- Block span: {ex['block_span_a']} genes\n")
                        f.write(f"- Correspondence: {ex['correspondence_score']:.1%}\n\n")

    print(f"\nReport saved to: {output_path}")

    # Save detailed data
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed data saved to: {csv_path}")


if __name__ == '__main__':
    main()
