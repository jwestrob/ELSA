#!/usr/bin/env python3
"""
Build operon-based ground truth for ELSA benchmarking.
V2: Uses coordinate-based matching between GFF annotations and genes.parquet.
    Multithreaded for speed.
"""

import argparse
import json
import re
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


def parse_operons(operon_file: Path) -> list[dict]:
    """Parse operon TSV file."""
    operons = []
    with open(operon_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            genes = [g.strip().lower() for g in row['genes'].split(',') if g.strip()]
            if len(genes) >= 2:
                operons.append({
                    'operon_id': row['operon_id'],
                    'genes': genes,
                    'evidence': row.get('evidence', 'unknown')
                })
    return operons


def parse_gff_genes(gff_file: Path) -> list[dict]:
    """Parse GFF file to get gene names and coordinates."""
    genes = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue

            gene_match = re.search(r'gene=([^;]+)', parts[8])
            if not gene_match:
                continue

            genes.append({
                'gene_name': gene_match.group(1).lower(),
                'contig': parts[0],
                'start': int(parts[3]),
                'end': int(parts[4]),
                'strand': parts[6]
            })
    return genes


def match_gff_to_parquet_fast(gff_genes: list[dict], parquet_df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Match GFF genes to parquet indices by coordinate overlap.
    Optimized with vectorized operations.
    """
    gene_to_indices = defaultdict(list)

    # Group parquet genes by contig for faster lookup
    parquet_by_contig = {}
    for contig in parquet_df['contig_id'].unique():
        mask = parquet_df['contig_id'] == contig
        contig_df = parquet_df[mask].copy()
        contig_df['_local_idx'] = range(len(contig_df))
        parquet_by_contig[contig] = contig_df

    # Build a position-to-index map for the full dataframe
    idx_map = {i: idx for idx, i in enumerate(parquet_df.index)}

    for gff_gene in gff_genes:
        contig = gff_gene['contig']
        if contig not in parquet_by_contig:
            continue

        contig_df = parquet_by_contig[contig]

        # Vectorized overlap check
        overlap_start = np.maximum(gff_gene['start'], contig_df['start'].values)
        overlap_end = np.minimum(gff_gene['end'], contig_df['end'].values)
        has_overlap = overlap_start < overlap_end

        if not has_overlap.any():
            continue

        overlap_len = overlap_end - overlap_start
        gff_len = gff_gene['end'] - gff_gene['start']
        parquet_len = contig_df['end'].values - contig_df['start'].values
        min_len = np.minimum(gff_len, parquet_len)

        overlap_frac = np.where(min_len > 0, overlap_len / min_len, 0)
        good_matches = has_overlap & (overlap_frac >= 0.5)

        for orig_idx in contig_df.index[good_matches]:
            parquet_idx = idx_map[orig_idx]
            gene_to_indices[gff_gene['gene_name']].append(parquet_idx)

    return dict(gene_to_indices)


def find_operon_in_genome(operon: dict, gene_to_indices: dict[str, list[int]],
                          parquet_df: pd.DataFrame, max_gap: int = 3) -> Optional[tuple]:
    """Find an operon's location in a genome.

    Handles genes with multiple copies by finding the best contiguous run.
    """
    gene_indices = {}
    for gene_name in operon['genes']:
        if gene_name in gene_to_indices:
            gene_indices[gene_name] = gene_to_indices[gene_name]

    if len(gene_indices) < 2:
        return None

    # Group by contig
    contig_positions = defaultdict(list)
    for gene_name, indices in gene_indices.items():
        for idx in indices:
            contig = parquet_df.iloc[idx]['contig_id']
            contig_positions[contig].append((idx, gene_name))

    best_result = None
    best_unique_genes = 0

    for contig, positions in contig_positions.items():
        positions.sort(key=lambda x: x[0])
        if len(positions) < 2:
            continue

        # Find best contiguous window using sliding window approach
        # to handle genes with multiple copies
        for start_i in range(len(positions)):
            seen_genes = set()
            window_positions = []

            for end_i in range(start_i, len(positions)):
                # Check gap from previous position
                if window_positions:
                    gap = positions[end_i][0] - window_positions[-1][0]
                    if gap > max_gap + 1:
                        break  # Gap too large, stop extending

                window_positions.append(positions[end_i])
                seen_genes.add(positions[end_i][1])

            # Check if this window is better
            if len(seen_genes) >= 2 and len(seen_genes) > best_unique_genes:
                best_unique_genes = len(seen_genes)
                indices = [p[0] for p in window_positions]
                gene_names = [p[1] for p in window_positions]
                best_result = (contig, min(indices), max(indices), gene_names)

    return best_result


def process_genome(args):
    """Process a single genome - for parallel execution."""
    sample_id, genes_parquet, annotation_dir, operons = args

    # Load parquet data for this sample
    df = pd.read_parquet(genes_parquet)
    parquet_df = df[df['sample_id'] == sample_id].reset_index(drop=True)

    # Load GFF
    gff_file = annotation_dir / f"{sample_id}.gff"
    if not gff_file.exists():
        return sample_id, None

    gff_genes = parse_gff_genes(gff_file)

    # Match GFF to parquet
    gene_to_indices = match_gff_to_parquet_fast(gff_genes, parquet_df)

    # Find operons
    operon_locations = {}
    for operon in operons:
        loc = find_operon_in_genome(operon, gene_to_indices, parquet_df)
        if loc:
            operon_locations[operon['operon_id']] = loc

    return sample_id, {
        'operon_locations': operon_locations,
        'n_genes': len(parquet_df),
    }


def main():
    parser = argparse.ArgumentParser(description='Build operon ground truth v2')
    parser.add_argument('--organism', required=True, choices=['bsubtilis', 'ecoli'])
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()

    if args.organism == 'bsubtilis':
        annotation_dir = BENCHMARKS_DIR / 'data' / 'bacillus' / 'annotations'
        operon_file = BENCHMARKS_DIR / 'operons' / 'bsubtilis' / 'operons.tsv'
        genes_parquet = BENCHMARKS_DIR / 'elsa_output' / 'bacillus' / 'elsa_index' / 'ingest' / 'genes.parquet'
        output_path = BENCHMARKS_DIR / 'ground_truth' / 'bsubtilis_operon_gt_v2'
    else:
        annotation_dir = BENCHMARKS_DIR / 'data' / 'ecoli' / 'annotations'
        operon_file = BENCHMARKS_DIR / 'operons' / 'ecoli' / 'operons.tsv'
        genes_parquet = BENCHMARKS_DIR / 'elsa_output' / 'ecoli' / 'elsa_index' / 'ingest' / 'genes.parquet'
        output_path = BENCHMARKS_DIR / 'ground_truth' / 'ecoli_operon_gt_v2'

    print("=" * 60)
    print(f"Building operon ground truth v2 for {args.organism}")
    print(f"Using {args.workers} workers")
    print("=" * 60)

    # Load operons
    print("\n[1/4] Loading operons...")
    operons = parse_operons(operon_file)
    print(f"  Loaded {len(operons)} operons")

    # Get sample IDs
    print("\n[2/4] Loading sample IDs...")
    df = pd.read_parquet(genes_parquet)
    sample_ids = sorted(df['sample_id'].unique())
    print(f"  Found {len(sample_ids)} genomes")
    del df  # Free memory

    # Process genomes in parallel
    print("\n[3/4] Processing genomes...")
    genome_data = {}

    work_items = [(sid, genes_parquet, annotation_dir, operons) for sid in sample_ids]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_genome, item): item[0] for item in work_items}

        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                sid, data = future.result()
                if data:
                    genome_data[sid] = data
                    print(f"  {sid}: {len(data['operon_locations'])} operons")
            except Exception as e:
                print(f"  {sample_id}: ERROR - {e}")

    # Generate pairwise comparisons
    print("\n[4/4] Generating pairwise ground truth...")
    results = []
    sample_list = list(genome_data.keys())

    for i, sample_a in enumerate(sample_list):
        for sample_b in sample_list[i+1:]:
            data_a = genome_data[sample_a]
            data_b = genome_data[sample_b]

            shared_operons = set(data_a['operon_locations'].keys()) & \
                           set(data_b['operon_locations'].keys())

            for operon_id in shared_operons:
                loc_a = data_a['operon_locations'][operon_id]
                loc_b = data_b['operon_locations'][operon_id]

                results.append({
                    'operon_id': operon_id,
                    'genome_a': sample_a,
                    'genome_b': sample_b,
                    'contig_a': loc_a[0],
                    'contig_b': loc_b[0],
                    'gene_idx_start_a': loc_a[1],
                    'gene_idx_end_a': loc_a[2],
                    'gene_idx_start_b': loc_b[1],
                    'gene_idx_end_b': loc_b[2],
                    'n_genes_a': len(loc_a[3]),
                    'n_genes_b': len(loc_b[3]),
                    'genes_a': ','.join(loc_a[3]),
                    'genes_b': ','.join(loc_b[3]),
                })

    print(f"  Generated {len(results)} operon instances")

    # Write output
    print("\nWriting output...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TSV
    tsv_path = output_path.with_suffix('.tsv')
    with open(tsv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(results)
    print(f"  Wrote {tsv_path}")

    # JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Wrote {json_path}")

    # Stats
    stats = {
        'total_instances': len(results),
        'unique_operons': len(set(r['operon_id'] for r in results)),
        'genome_pairs': len(set((r['genome_a'], r['genome_b']) for r in results)),
    }
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Instances: {stats['total_instances']}")
    print(f"  Unique operons: {stats['unique_operons']}")
    print(f"  Genome pairs: {stats['genome_pairs']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
