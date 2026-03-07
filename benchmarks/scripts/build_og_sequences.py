#!/usr/bin/env python3
"""
Phase 2: Build ordered orthogroup sequences for each contig.

Creates a representation of each genome as an ordered sequence of orthogroups,
which can then be used to find conserved neighborhoods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def build_og_sequences(
    orthogroups_path: Path,
    output_path: Path,
) -> dict:
    """Build ordered orthogroup sequences for each contig."""

    print(f"Loading orthogroups from {orthogroups_path}")
    df = pd.read_csv(orthogroups_path, sep='\t')
    print(f"Loaded {len(df)} genes")

    # Sort by sample, contig, position
    df = df.sort_values(['sample_id', 'contig_id', 'start'])

    # Build sequences
    sequences = {}
    stats = defaultdict(int)

    for (sample_id, contig_id), group in df.groupby(['sample_id', 'contig_id']):
        key = f"{sample_id}:{contig_id}"

        genes = group.sort_values('start')

        sequences[key] = {
            'sample_id': sample_id,
            'contig_id': contig_id,
            'n_genes': len(genes),
            'orthogroups': genes['orthogroup'].tolist(),
            'strands': genes['strand'].tolist(),
            'gene_ids': genes['gene_id'].tolist(),
            'positions': genes['start'].tolist(),
        }

        stats['n_contigs'] += 1
        stats['n_genes'] += len(genes)

    stats['n_samples'] = df['sample_id'].nunique()

    print(f"\nBuilt sequences for {stats['n_contigs']} contigs across {stats['n_samples']} samples")
    print(f"Total genes: {stats['n_genes']}")

    # Contig size distribution
    sizes = [s['n_genes'] for s in sequences.values()]
    print(f"Contig sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.0f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(sequences, f)
    print(f"\nWrote {output_path}")

    # Save stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dict(stats), f, indent=2)

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Build orthogroup sequences per contig")
    parser.add_argument("orthogroups_tsv", type=Path, help="Orthogroups TSV from Phase 1")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON path")

    args = parser.parse_args()

    build_og_sequences(args.orthogroups_tsv, args.output)


if __name__ == "__main__":
    main()
