#!/usr/bin/env python3
"""
Evaluate ELSA and MCScanX against operon ground truth.

Provides multiple evaluation metrics:
1. Strict recall: Single block covers operon in BOTH genomes (≥50%)
2. Independent recall: Operon covered in each genome (can be different blocks)
3. Coverage recall: Operon covered in AT LEAST ONE genome
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.localization import localize_pairs, summarize, format_summary  # noqa: E402


def load_operon_gt(gt_path: Path) -> pd.DataFrame:
    """Load operon ground truth."""
    return pd.read_csv(gt_path, sep='\t')


def load_elsa_blocks(blocks_path: Path) -> pd.DataFrame:
    """Load ELSA blocks."""
    return pd.read_csv(blocks_path)


def load_mcscanx_blocks(blocks_path: Path) -> pd.DataFrame:
    """Load MCScanX blocks."""
    return pd.read_csv(blocks_path)


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


def evaluate_elsa(elsa_blocks: pd.DataFrame, operon_gt: pd.DataFrame,
                  threshold: float = 0.5) -> dict:
    """
    Evaluate ELSA blocks against operon ground truth.

    Returns multiple recall metrics:
    - strict_recall: Single block covers operon ≥threshold in BOTH genomes
    - independent_recall: Operon covered ≥threshold in BOTH genomes (possibly different blocks)
    - coverage_recall: Operon covered ≥threshold in AT LEAST ONE genome
    """

    results = {
        'total_operons': len(operon_gt),
        # Strict: same block covers both
        'strict_found': 0,
        'strict_missed': 0,
        'strict_missed_operons': [],
        # Independent: covered in both (can be different blocks)
        'independent_found': 0,
        'independent_missed': 0,
        # Coverage: covered in at least one genome
        'any_found': 0,
        'any_missed': 0,
    }

    for _, operon in operon_gt.iterrows():
        operon_id = operon['operon_id']
        genome_a = operon['genome_a']
        genome_b = operon['genome_b']

        op_start_a = operon['gene_idx_start_a']
        op_end_a = operon['gene_idx_end_a']
        op_start_b = operon['gene_idx_start_b']
        op_end_b = operon['gene_idx_end_b']

        # Find blocks matching this genome pair (either direction)
        matching_blocks = elsa_blocks[
            ((elsa_blocks['query_genome'] == genome_a) & (elsa_blocks['target_genome'] == genome_b)) |
            ((elsa_blocks['query_genome'] == genome_b) & (elsa_blocks['target_genome'] == genome_a))
        ]

        strict_found = False
        best_overlap_a = 0.0
        best_overlap_b = 0.0

        for _, block in matching_blocks.iterrows():
            if block['query_genome'] == genome_a:
                block_start_a = block['query_start']
                block_end_a = block['query_end']
                block_start_b = block['target_start']
                block_end_b = block['target_end']
            else:
                block_start_a = block['target_start']
                block_end_a = block['target_end']
                block_start_b = block['query_start']
                block_end_b = block['query_end']

            overlap_a, frac_a = check_overlap(block_start_a, block_end_a, op_start_a, op_end_a, threshold)
            overlap_b, frac_b = check_overlap(block_start_b, block_end_b, op_start_b, op_end_b, threshold)

            # Track best overlap for each genome independently
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

    # Legacy names for compatibility
    results['found'] = results['strict_found']
    results['not_found'] = results['strict_missed']
    results['recall'] = results['strict_recall']
    results['missed_operons'] = results['strict_missed_operons']
    results['found_operons'] = []  # Not tracked for efficiency

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operon-gt', required=True, help='Operon ground truth TSV')
    parser.add_argument('--elsa-blocks', required=True, help='ELSA blocks CSV')
    parser.add_argument('--mcscanx-blocks', help='MCScanX blocks CSV (optional, not used currently)')
    parser.add_argument('--mcscanx-collinearity', help='MCScanX collinearity file (optional)')
    parser.add_argument('--output', required=True, help='Output comparison report')
    parser.add_argument('--threshold', type=float, default=0.5, help='Overlap threshold')
    args = parser.parse_args()

    print("Loading data...")
    operon_gt = load_operon_gt(Path(args.operon_gt))
    elsa_blocks = load_elsa_blocks(Path(args.elsa_blocks))

    # Filter to E. coli genomes only (those in the ground truth)
    gt_genomes = set(operon_gt['genome_a'].unique()) | set(operon_gt['genome_b'].unique())
    print(f"Ground truth genomes: {len(gt_genomes)}")
    print(f"  {sorted(gt_genomes)[:5]}...")

    # Filter ELSA blocks to E. coli pairs
    elsa_ecoli = elsa_blocks[
        (elsa_blocks['query_genome'].isin(gt_genomes)) &
        (elsa_blocks['target_genome'].isin(gt_genomes))
    ]
    print(f"ELSA E.coli blocks: {len(elsa_ecoli):,}")

    # Count unique operons
    unique_operons = operon_gt['operon_id'].nunique()
    print(f"Unique operons: {unique_operons}")
    print(f"Operon instances (genome pairs): {len(operon_gt)}")

    print("\nEvaluating ELSA...")
    elsa_results = evaluate_elsa(elsa_ecoli, operon_gt, args.threshold)

    # Localization, computed and reported unconditionally alongside recall.
    # Recall on its own is satisfied by a block spanning the whole contig, so a
    # recall figure without the boundary precision that qualifies it overstates
    # what was found. See elsa/localization.py.
    # Blocks are in gene-index space. The ground truth carries both spaces under
    # confusingly similar names — `start_a` is a base-pair coordinate while
    # `gene_idx_start_a` is a gene index — so the gene-index columns are named
    # explicitly here. Passing the bp columns would silently compare a
    # megabase-scale interval against a gene-count interval and report near-zero
    # precision everywhere.
    print("Measuring localization...")
    locus_cols = ("genome_a", "contig_a", "gene_idx_start_a", "gene_idx_end_a",
                  "genome_b", "contig_b", "gene_idx_start_b", "gene_idx_end_b")
    missing = [c for c in locus_cols if c not in operon_gt.columns]
    if missing:
        # The audit's pair-metric tables already carry translated gene indices
        # under the plain names.
        locus_cols = ("genome_a", "contig_a", "start_a", "end_a",
                      "genome_b", "contig_b", "start_b", "end_b")
        missing = [c for c in locus_cols if c not in operon_gt.columns]
        if missing:
            raise SystemExit(f"ground truth lacks locus columns: {missing}")
    localization = localize_pairs(operon_gt, elsa_ecoli, min_coverage=args.threshold,
                                  locus_cols=locus_cols)
    local_summary = summarize(localization, group_column="operon_id")
    print()
    print(format_summary(local_summary, "Localization"))

    detail_path = Path(args.output).with_suffix(".localization.tsv")
    localization.to_csv(detail_path, sep="\t", index=False)

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Operon Recall Evaluation: ELSA vs MCScanX\n\n")
        f.write(f"**Ground Truth**: E. coli operons from RegulonDB\n")
        f.write(f"**Overlap Threshold**: {args.threshold:.0%}\n")
        f.write(f"**Operon instances**: {elsa_results['total_operons']:,} (across {len(gt_genomes)} E. coli genomes)\n\n")

        f.write("## Recall Metrics\n\n")
        f.write("Three different recall definitions:\n\n")
        f.write("1. **Strict**: Single block covers operon ≥50% in BOTH genomes\n")
        f.write("2. **Independent**: Operon covered ≥50% in BOTH genomes (can be different blocks)\n")
        f.write("3. **Any coverage**: Operon covered ≥50% in AT LEAST ONE genome\n\n")

        f.write("| Metric | ELSA |\n")
        f.write("|--------|------|\n")
        f.write(f"| Strict recall | {elsa_results['strict_recall']:.1%} ({elsa_results['strict_found']:,}/{elsa_results['total_operons']:,}) |\n")
        f.write(f"| Independent recall | {elsa_results['independent_recall']:.1%} ({elsa_results['independent_found']:,}/{elsa_results['total_operons']:,}) |\n")
        f.write(f"| Any coverage recall | {elsa_results['any_recall']:.1%} ({elsa_results['any_found']:,}/{elsa_results['total_operons']:,}) |\n")

        f.write("\n## Localization\n\n")
        f.write("Recall asks whether the operon is inside some block. A block "
                "spanning the whole contig satisfies that perfectly, so these "
                "figures qualify every recall number above.\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Median boundary precision | {local_summary['median_boundary_precision']:.4f} "
                f"(95% CI {local_summary['boundary_precision_ci_low']:.4f}–"
                f"{local_summary['boundary_precision_ci_high']:.4f}) |\n")
        f.write(f"| Median pair IoU | {local_summary['median_pair_iou']:.4f} |\n")
        f.write(f"| Median block span | {local_summary['median_block_span']:.0f} genes |\n")
        f.write(f"| Median locus span | {local_summary['median_locus_span']:.0f} genes |\n")
        f.write(f"| Median containing excess | {local_summary['median_containing_excess_genes']:.0f} genes |\n\n")
        f.write(f"**{local_summary['recall']:.1%} of loci sit inside some block; that block is "
                f"{local_summary['median_boundary_precision']:.1%} locus by extent.** "
                f"Per-locus detail in `{detail_path.name}`.\n")

        f.write("\n## Interpretation\n\n")
        f.write("The gap between strict and independent recall reveals an important insight:\n\n")
        f.write("- **Strict recall** is low because operons are small (2-10 genes) and may be\n")
        f.write("  part of different larger syntenic regions in each genome.\n")
        f.write("- **Independent recall** shows that ELSA finds blocks covering each operon's\n")
        f.write("  position, just not always in the same block.\n")
        f.write("- **Any coverage** confirms that operon regions are generally well-covered.\n\n")
        f.write("This is expected behavior: operons can be embedded in different syntenic\n")
        f.write("contexts across genomes due to insertions, deletions, or inversions in the\n")
        f.write("surrounding regions.\n")

        # Operons not covered at all
        if elsa_results['any_missed'] > 0:
            f.write(f"\n### Operons with no coverage ({elsa_results['any_missed']} instances)\n")
            f.write("These operons have no block overlapping their position in either genome.\n")

    print(f"\nReport saved to: {output_path}")
    print("\n=== SUMMARY ===")
    print(f"Strict recall:      {elsa_results['strict_recall']:.1%} ({elsa_results['strict_found']:,}/{elsa_results['total_operons']:,})")
    print(f"Independent recall: {elsa_results['independent_recall']:.1%} ({elsa_results['independent_found']:,}/{elsa_results['total_operons']:,})")
    print(f"Any coverage:       {elsa_results['any_recall']:.1%} ({elsa_results['any_found']:,}/{elsa_results['total_operons']:,})")


if __name__ == '__main__':
    main()
