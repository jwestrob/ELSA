#!/usr/bin/env python3
"""
Create synteny dot plots comparing ELSA and MCScanX blocks.

Generates publication-quality dot plots showing:
1. Gene-by-gene correspondence patterns
2. ELSA block boundaries vs MCScanX block boundaries
3. Case studies highlighting differences

Useful for visualizing specific examples like the flgKL operon case.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent

# Colors
ELSA_COLOR = '#2ecc71'
MCSCANX_COLOR = '#3498db'
OPERON_COLOR = '#e74c3c'


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


def parse_mcscanx_blocks(coll_path: Path, gene_to_idx: dict,
                          genome_a: str, genome_b: str) -> list:
    """Parse MCScanX collinearity file for a specific genome pair."""
    blocks = []
    current_block = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith('## Alignment'):
                if current_block and current_block['gene_pairs']:
                    blocks.append(current_block)

                parts = line.split()
                block_id = int(parts[2].rstrip(':'))

                chrom_info = None
                for p in parts:
                    if '&' in p:
                        chrom_info = p
                        break

                if chrom_info:
                    chrom_a_str, chrom_b_str = chrom_info.split('&')
                    ga, ca = parse_chrom(chrom_a_str)
                    gb, cb = parse_chrom(chrom_b_str)

                    # Check if this matches our genome pair
                    if (ga == genome_a and gb == genome_b) or (ga == genome_b and gb == genome_a):
                        current_block = {
                            'block_id': block_id,
                            'genome_a': ga,
                            'genome_b': gb,
                            'contig_a': ca,
                            'contig_b': cb,
                            'gene_pairs': [],
                        }
                    else:
                        current_block = None
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

    if current_block and current_block['gene_pairs']:
        blocks.append(current_block)

    return blocks


def get_elsa_blocks(elsa_df: pd.DataFrame, genome_a: str, genome_b: str) -> list:
    """Filter ELSA blocks for a specific genome pair."""
    matching = elsa_df[
        ((elsa_df['query_genome'] == genome_a) & (elsa_df['target_genome'] == genome_b)) |
        ((elsa_df['query_genome'] == genome_b) & (elsa_df['target_genome'] == genome_a))
    ]

    blocks = []
    for _, row in matching.iterrows():
        if row['query_genome'] == genome_a:
            blocks.append({
                'block_id': row['block_id'],
                'start_a': row['query_start'],
                'end_a': row['query_end'],
                'start_b': row['target_start'],
                'end_b': row['target_end'],
            })
        else:
            blocks.append({
                'block_id': row['block_id'],
                'start_a': row['target_start'],
                'end_a': row['target_end'],
                'start_b': row['query_start'],
                'end_b': row['query_end'],
            })

    return blocks


def create_dotplot(mcscanx_blocks: list, elsa_blocks: list,
                   genome_a: str, genome_b: str,
                   output_path: Path, region: tuple = None,
                   operons: list = None):
    """
    Create a dot plot showing both MCScanX and ELSA blocks.

    Args:
        mcscanx_blocks: List of MCScanX blocks with gene_pairs
        elsa_blocks: List of ELSA blocks with start/end ranges
        genome_a, genome_b: Genome identifiers
        output_path: Where to save the figure
        region: Optional (x_min, x_max, y_min, y_max) to zoom
        operons: Optional list of {'name', 'pos_a', 'pos_b'} to highlight
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot MCScanX gene pairs as points
    mcs_x, mcs_y = [], []
    for block in mcscanx_blocks:
        for idx_a, idx_b in block['gene_pairs']:
            mcs_x.append(idx_a)
            mcs_y.append(idx_b)

    if mcs_x:
        ax.scatter(mcs_x, mcs_y, s=2, alpha=0.6, color=MCSCANX_COLOR,
                   label='MCScanX gene pairs', zorder=2)

    # Plot ELSA blocks as rectangles
    for i, block in enumerate(elsa_blocks):
        rect = patches.Rectangle(
            (block['start_a'], block['start_b']),
            block['end_a'] - block['start_a'],
            block['end_b'] - block['start_b'],
            linewidth=1.5, edgecolor=ELSA_COLOR, facecolor='none',
            alpha=0.8, zorder=3,
            label='ELSA blocks' if i == 0 else None
        )
        ax.add_patch(rect)

    # Highlight operons if provided
    if operons:
        for operon in operons:
            # Vertical line for genome A position
            ax.axvline(operon['pos_a'], color=OPERON_COLOR, linestyle='--',
                       alpha=0.7, linewidth=2)
            # Horizontal line for genome B position
            ax.axhline(operon['pos_b'], color=OPERON_COLOR, linestyle='--',
                       alpha=0.7, linewidth=2)
            # Annotation
            ax.annotate(operon['name'],
                        xy=(operon['pos_a'], operon['pos_b']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color=OPERON_COLOR,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Set axis labels
    ax.set_xlabel(f'{genome_a} (gene index)', fontsize=12)
    ax.set_ylabel(f'{genome_b} (gene index)', fontsize=12)
    ax.set_title(f'Synteny Dot Plot: {genome_a} vs {genome_b}', fontsize=14)

    # Apply region zoom if specified
    if region:
        ax.set_xlim(region[0], region[1])
        ax.set_ylim(region[2], region[3])

    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix('.pdf'))
    plt.close()

    print(f"  Saved: {output_path}")


def create_flgkl_case_study(mcscanx_blocks: list, elsa_blocks: list,
                             output_dir: Path):
    """Create detailed visualization of the flgKL operon case study."""
    # flgKL operon positions from the ground truth
    # Genome A (GCF_000599625.1): positions 2501-2502
    # Genome B (GCF_000599665.1): positions 1581-1582

    operons = [
        {'name': 'flgKL (A)', 'pos_a': 2501.5, 'pos_b': 1581.5},
    ]

    # Create zoomed view around the operon region
    region = (2400, 2700, 1400, 1800)

    output_path = output_dir / 'dotplot_flgkl_case_study.png'
    create_dotplot(mcscanx_blocks, elsa_blocks,
                   'GCF_000599625.1', 'GCF_000599665.1',
                   output_path, region=region, operons=operons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elsa-blocks',
                        default=BENCHMARKS_DIR / 'results' / 'cross_species_chain' / 'micro_chain' / 'micro_chain_blocks.csv')
    parser.add_argument('--mcscanx-gff',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.gff')
    parser.add_argument('--mcscanx-collinearity',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.collinearity')
    parser.add_argument('--output-dir',
                        default=BENCHMARKS_DIR / 'evaluation' / 'figures')
    parser.add_argument('--genome-a', default='GCF_000599625.1',
                        help='First genome for dot plot')
    parser.add_argument('--genome-b', default='GCF_000599665.1',
                        help='Second genome for dot plot')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating Synteny Dot Plots")
    print("=" * 70)

    # Load data
    print("\nLoading ELSA blocks...")
    elsa_df = pd.read_csv(args.elsa_blocks)
    print(f"  Loaded {len(elsa_df):,} ELSA blocks")

    print("\nBuilding gene index...")
    gene_to_idx = build_gene_index(Path(args.mcscanx_gff))
    print(f"  Indexed {len(gene_to_idx):,} genes")

    print("\nParsing MCScanX blocks...")
    mcscanx_all = parse_mcscanx_blocks(
        Path(args.mcscanx_collinearity), gene_to_idx,
        args.genome_a, args.genome_b
    )
    print(f"  Found {len(mcscanx_all):,} blocks for {args.genome_a} vs {args.genome_b}")

    # Get ELSA blocks for same genome pair
    elsa_blocks = get_elsa_blocks(elsa_df, args.genome_a, args.genome_b)
    print(f"  Found {len(elsa_blocks):,} ELSA blocks")

    # Create main dot plot
    print("\nCreating dot plots...")
    main_output = output_dir / f'dotplot_{args.genome_a}_vs_{args.genome_b}.png'
    create_dotplot(mcscanx_all, elsa_blocks,
                   args.genome_a, args.genome_b, main_output)

    # Create flgKL case study if this is the right genome pair
    if args.genome_a == 'GCF_000599625.1' and args.genome_b == 'GCF_000599665.1':
        print("\nCreating flgKL case study...")
        create_flgkl_case_study(mcscanx_all, elsa_blocks, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
