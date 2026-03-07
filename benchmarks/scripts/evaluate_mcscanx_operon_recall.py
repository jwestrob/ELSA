#!/usr/bin/env python3
"""
Evaluate MCScanX against operon ground truth.

Parses the collinearity file to get actual gene positions and evaluates
operon recall with the same metrics as ELSA evaluation.
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re


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


def parse_collinearity(coll_path: Path, gene_to_idx: dict) -> list:
    """
    Parse MCScanX collinearity file to extract blocks with gene indices.

    Returns list of dicts with block info including gene index ranges.
    """
    blocks = []
    current_block = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()

            # Block header
            if line.startswith('## Alignment'):
                if current_block and current_block['genes_a'] and current_block['genes_b']:
                    # Finalize previous block
                    current_block['query_start'] = min(current_block['genes_a'])
                    current_block['query_end'] = max(current_block['genes_a'])
                    current_block['target_start'] = min(current_block['genes_b'])
                    current_block['target_end'] = max(current_block['genes_b'])
                    blocks.append(current_block)

                # Parse header: ## Alignment N: score=X e_value=Y N=Z chrom_a&chrom_b orientation
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
                        'genes_a': [],
                        'genes_b': [],
                    }
                else:
                    current_block = None

            # Gene pair line
            elif current_block and line and not line.startswith('#'):
                # Format: N-  M:[TAB]id_a[TAB]id_b[TAB]e_value
                # Must split on tabs — the "N-  M:" prefix has internal spaces
                parts = line.split('\t')
                if len(parts) >= 3:
                    gene_a = parts[1].strip()
                    gene_b = parts[2].strip()

                    if gene_a in gene_to_idx and gene_b in gene_to_idx:
                        _, _, idx_a = gene_to_idx[gene_a]
                        _, _, idx_b = gene_to_idx[gene_b]
                        current_block['genes_a'].append(idx_a)
                        current_block['genes_b'].append(idx_b)

    # Don't forget last block
    if current_block and current_block['genes_a'] and current_block['genes_b']:
        current_block['query_start'] = min(current_block['genes_a'])
        current_block['query_end'] = max(current_block['genes_a'])
        current_block['target_start'] = min(current_block['genes_b'])
        current_block['target_end'] = max(current_block['genes_b'])
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


def evaluate_mcscanx(blocks: list, operon_gt: pd.DataFrame,
                     gt_genomes: set, threshold: float = 0.5) -> dict:
    """Evaluate MCScanX blocks against operon ground truth."""

    # Filter to E. coli blocks
    ecoli_blocks = [b for b in blocks
                    if b['genome_a'] in gt_genomes and b['genome_b'] in gt_genomes]

    results = {
        'total_operons': len(operon_gt),
        'total_blocks': len(ecoli_blocks),
        'strict_found': 0,
        'strict_missed': 0,
        'independent_found': 0,
        'independent_missed': 0,
        'any_found': 0,
        'any_missed': 0,
        'strict_missed_operons': [],
    }

    # Index blocks by genome pair for faster lookup
    block_index = defaultdict(list)
    for block in ecoli_blocks:
        key1 = (block['genome_a'], block['genome_b'], block['contig_a'], block['contig_b'])
        key2 = (block['genome_b'], block['genome_a'], block['contig_b'], block['contig_a'])
        block_index[key1].append(block)
        block_index[key2].append(block)

    for _, operon in operon_gt.iterrows():
        operon_id = operon['operon_id']
        genome_a = operon['genome_a']
        genome_b = operon['genome_b']
        contig_a = operon['contig_a']
        contig_b = operon['contig_b']

        op_start_a = operon['gene_idx_start_a']
        op_end_a = operon['gene_idx_end_a']
        op_start_b = operon['gene_idx_start_b']
        op_end_b = operon['gene_idx_end_b']

        # Get matching blocks
        key = (genome_a, genome_b, contig_a, contig_b)
        matching = block_index.get(key, [])

        strict_found = False
        best_overlap_a = 0.0
        best_overlap_b = 0.0

        for block in matching:
            # Determine direction
            if block['genome_a'] == genome_a:
                b_start_a = block['query_start']
                b_end_a = block['query_end']
                b_start_b = block['target_start']
                b_end_b = block['target_end']
            else:
                b_start_a = block['target_start']
                b_end_a = block['target_end']
                b_start_b = block['query_start']
                b_end_b = block['query_end']

            overlap_a, frac_a = check_overlap(b_start_a, b_end_a, op_start_a, op_end_a, threshold)
            overlap_b, frac_b = check_overlap(b_start_b, b_end_b, op_start_b, op_end_b, threshold)

            best_overlap_a = max(best_overlap_a, frac_a)
            best_overlap_b = max(best_overlap_b, frac_b)

            if overlap_a and overlap_b:
                strict_found = True

        # Strict: same block covers both
        if strict_found:
            results['strict_found'] += 1
        else:
            results['strict_missed'] += 1
            results['strict_missed_operons'].append(operon_id)

        # Independent: covered in both (can be different blocks)
        if best_overlap_a >= threshold and best_overlap_b >= threshold:
            results['independent_found'] += 1
        else:
            results['independent_missed'] += 1

        # Any: covered in at least one genome
        if best_overlap_a >= threshold or best_overlap_b >= threshold:
            results['any_found'] += 1
        else:
            results['any_missed'] += 1

    total = results['total_operons']
    results['strict_recall'] = results['strict_found'] / total if total > 0 else 0
    results['independent_recall'] = results['independent_found'] / total if total > 0 else 0
    results['any_recall'] = results['any_found'] / total if total > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operon-gt', required=True, help='Operon ground truth TSV')
    parser.add_argument('--mcscanx-gff', required=True, help='MCScanX GFF file')
    parser.add_argument('--mcscanx-collinearity', required=True, help='MCScanX collinearity file')
    parser.add_argument('--output', required=True, help='Output report')
    parser.add_argument('--threshold', type=float, default=0.5, help='Overlap threshold')
    args = parser.parse_args()

    print("Loading operon ground truth...")
    operon_gt = pd.read_csv(args.operon_gt, sep='\t')
    gt_genomes = set(operon_gt['genome_a'].unique()) | set(operon_gt['genome_b'].unique())
    print(f"  {len(gt_genomes)} E. coli genomes, {len(operon_gt):,} operon instances")

    print("\nBuilding gene index from MCScanX GFF...")
    gene_to_idx = build_gene_index(Path(args.mcscanx_gff))
    print(f"  {len(gene_to_idx):,} genes indexed")

    print("\nParsing MCScanX collinearity file...")
    blocks = parse_collinearity(Path(args.mcscanx_collinearity), gene_to_idx)
    print(f"  {len(blocks):,} total blocks")

    print("\nEvaluating MCScanX...")
    results = evaluate_mcscanx(blocks, operon_gt, gt_genomes, args.threshold)
    print(f"  {results['total_blocks']:,} E. coli blocks")

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# MCScanX Operon Recall Evaluation\n\n")
        f.write(f"**Ground Truth**: E. coli operons from RegulonDB\n")
        f.write(f"**Overlap Threshold**: {args.threshold:.0%}\n")
        f.write(f"**Operon instances**: {results['total_operons']:,}\n")
        f.write(f"**MCScanX E. coli blocks**: {results['total_blocks']:,}\n\n")

        f.write("## Recall Metrics\n\n")
        f.write("| Metric | MCScanX |\n")
        f.write("|--------|--------|\n")
        f.write(f"| Strict recall | {results['strict_recall']:.1%} ({results['strict_found']:,}/{results['total_operons']:,}) |\n")
        f.write(f"| Independent recall | {results['independent_recall']:.1%} ({results['independent_found']:,}/{results['total_operons']:,}) |\n")
        f.write(f"| Any coverage recall | {results['any_recall']:.1%} ({results['any_found']:,}/{results['total_operons']:,}) |\n")

    print(f"\nReport saved to: {output_path}")
    print("\n=== SUMMARY ===")
    print(f"Strict recall:      {results['strict_recall']:.1%} ({results['strict_found']:,}/{results['total_operons']:,})")
    print(f"Independent recall: {results['independent_recall']:.1%} ({results['independent_found']:,}/{results['total_operons']:,})")
    print(f"Any coverage:       {results['any_recall']:.1%} ({results['any_found']:,}/{results['total_operons']:,})")


if __name__ == '__main__':
    main()
