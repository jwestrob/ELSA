#!/usr/bin/env python3
"""
Phase 3-4: Find conserved gene neighborhoods.

Finds k-mers of orthogroups that appear in multiple genomes,
then merges overlapping k-mers into conserved blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ConservedBlock:
    """A conserved gene neighborhood."""
    block_id: str
    orthogroups: list[str]
    n_genomes: int
    genomes: list[str]
    instances: dict = field(default_factory=dict)  # genome -> {contig, genes, positions}

    def to_dict(self) -> dict:
        return {
            'block_id': self.block_id,
            'orthogroups': self.orthogroups,
            'n_orthogroups': len(self.orthogroups),
            'n_genomes': self.n_genomes,
            'genomes': self.genomes,
            'instances': self.instances,
        }


def canonicalize_kmer(kmer: tuple[str, ...]) -> tuple[str, ...]:
    """Canonicalize k-mer by choosing lexicographically smaller of forward/reverse."""
    rev = tuple(reversed(kmer))
    return min(kmer, rev)


def extract_kmers(
    og_sequence: list[str],
    k: int,
) -> list[tuple[tuple[str, ...], int]]:
    """Extract all k-mers from an orthogroup sequence.

    Returns list of (canonical_kmer, position) tuples.
    """
    kmers = []
    for i in range(len(og_sequence) - k + 1):
        kmer = tuple(og_sequence[i:i + k])
        canonical = canonicalize_kmer(kmer)
        kmers.append((canonical, i))
    return kmers


def find_conserved_kmers(
    sequences: dict,
    k: int,
    min_genome_support: int,
) -> dict[tuple[str, ...], dict]:
    """Find k-mers that appear in at least min_genome_support genomes.

    Returns dict: kmer -> {genomes: set, instances: [(contig_key, position), ...]}
    """
    print(f"Extracting {k}-mers from {len(sequences)} contigs...")

    # kmer -> {genomes: set, instances: list}
    kmer_data = defaultdict(lambda: {'genomes': set(), 'instances': []})

    for contig_key, seq_data in sequences.items():
        sample_id = seq_data['sample_id']
        og_seq = seq_data['orthogroups']

        kmers = extract_kmers(og_seq, k)
        for kmer, pos in kmers:
            kmer_data[kmer]['genomes'].add(sample_id)
            kmer_data[kmer]['instances'].append((contig_key, pos))

    # Filter by genome support
    n_total = len(kmer_data)
    conserved = {
        kmer: data for kmer, data in kmer_data.items()
        if len(data['genomes']) >= min_genome_support
    }

    print(f"Total unique {k}-mers: {n_total}")
    print(f"Conserved {k}-mers (≥{min_genome_support} genomes): {len(conserved)}")

    return conserved


def merge_overlapping_kmers(
    conserved_kmers: dict[tuple[str, ...], dict],
    sequences: dict,
    k: int,
    min_genome_support: int,
) -> list[ConservedBlock]:
    """Merge overlapping conserved k-mers into larger blocks.

    Strategy: For each genome, find runs of overlapping conserved k-mers
    and merge them into maximal blocks.
    """
    print(f"\nMerging overlapping k-mers...")

    # Build kmer -> set of genomes lookup
    kmer_genomes = {kmer: data['genomes'] for kmer, data in conserved_kmers.items()}

    # For each genome, find conserved regions
    genome_blocks = defaultdict(list)  # genome -> list of (contig_key, start_pos, end_pos, og_list)

    for contig_key, seq_data in sequences.items():
        sample_id = seq_data['sample_id']
        og_seq = seq_data['orthogroups']
        gene_ids = seq_data['gene_ids']

        # Find positions covered by conserved k-mers
        covered = [False] * len(og_seq)
        kmer_at_pos = {}  # position -> kmer

        for i in range(len(og_seq) - k + 1):
            kmer = tuple(og_seq[i:i + k])
            canonical = canonicalize_kmer(kmer)
            if canonical in kmer_genomes:
                for j in range(i, i + k):
                    covered[j] = True
                kmer_at_pos[i] = canonical

        # Find contiguous covered regions
        i = 0
        while i < len(og_seq):
            if covered[i]:
                # Start of a covered region
                start = i
                while i < len(og_seq) and covered[i]:
                    i += 1
                end = i  # exclusive

                # Extract the block
                block_ogs = og_seq[start:end]
                block_genes = gene_ids[start:end]

                genome_blocks[sample_id].append({
                    'contig_key': contig_key,
                    'start': start,
                    'end': end,
                    'orthogroups': block_ogs,
                    'gene_ids': block_genes,
                })
            else:
                i += 1

    # Now cluster blocks by orthogroup content
    # Two blocks are "the same" if they have the same orthogroup sequence (or reverse)
    block_signatures = defaultdict(list)  # canonical_og_tuple -> list of (genome, block_info)

    for genome, blocks in genome_blocks.items():
        for block in blocks:
            og_tuple = tuple(block['orthogroups'])
            canonical = canonicalize_kmer(og_tuple)
            block_signatures[canonical].append((genome, block))

    # Filter by genome support and create ConservedBlock objects
    conserved_blocks = []

    for og_tuple, instances in block_signatures.items():
        genomes = set(g for g, _ in instances)
        if len(genomes) >= min_genome_support:
            block_id = f"GT_{len(conserved_blocks):05d}"

            instance_dict = {}
            for genome, block_info in instances:
                contig_key = block_info['contig_key']
                contig_id = contig_key.split(':')[1] if ':' in contig_key else contig_key

                instance_dict[genome] = {
                    'contig': contig_id,
                    'genes': block_info['gene_ids'],
                    'start_pos': block_info['start'],
                    'end_pos': block_info['end'],
                }

            conserved_blocks.append(ConservedBlock(
                block_id=block_id,
                orthogroups=list(og_tuple),
                n_genomes=len(genomes),
                genomes=sorted(genomes),
                instances=instance_dict,
            ))

    # Sort by genome support (descending) then by length (descending)
    conserved_blocks.sort(key=lambda b: (-b.n_genomes, -len(b.orthogroups)))

    # Re-assign IDs after sorting
    for i, block in enumerate(conserved_blocks):
        block.block_id = f"GT_{i:05d}"

    print(f"Created {len(conserved_blocks)} conserved blocks")

    return conserved_blocks


def analyze_blocks(blocks: list[ConservedBlock], n_genomes: int) -> dict:
    """Analyze conserved block statistics."""
    if not blocks:
        return {'n_blocks': 0}

    sizes = [len(b.orthogroups) for b in blocks]
    supports = [b.n_genomes for b in blocks]

    stats = {
        'n_blocks': len(blocks),
        'size_min': min(sizes),
        'size_max': max(sizes),
        'size_mean': sum(sizes) / len(sizes),
        'support_min': min(supports),
        'support_max': max(supports),
        'support_mean': sum(supports) / len(supports),
        'core_blocks': sum(1 for b in blocks if b.n_genomes == n_genomes),
    }

    print(f"\nConserved Block Statistics:")
    print(f"  Total blocks: {stats['n_blocks']}")
    print(f"  Size range: {stats['size_min']}-{stats['size_max']} orthogroups")
    print(f"  Mean size: {stats['size_mean']:.1f} orthogroups")
    print(f"  Support range: {stats['support_min']}-{stats['support_max']} genomes")
    print(f"  Mean support: {stats['support_mean']:.1f} genomes")
    print(f"  Core blocks (all genomes): {stats['core_blocks']}")

    # Size distribution
    print(f"\n  Size distribution:")
    for s in [3, 4, 5, 10, 20, 50]:
        n = sum(1 for b in blocks if len(b.orthogroups) >= s)
        print(f"    ≥{s} OGs: {n} blocks")

    return stats


def find_conserved_neighborhoods(
    sequences_path: Path,
    output_path: Path,
    k: int = 3,
    min_genome_support: int = 10,
    min_genome_fraction: Optional[float] = None,
) -> list[ConservedBlock]:
    """Main function to find conserved gene neighborhoods."""

    print(f"Loading sequences from {sequences_path}")
    with open(sequences_path) as f:
        sequences = json.load(f)

    # Determine genome support threshold
    n_genomes = len(set(s['sample_id'] for s in sequences.values()))
    if min_genome_fraction is not None:
        min_support = max(2, int(n_genomes * min_genome_fraction))
        print(f"Using {min_genome_fraction*100:.0f}% genome support = {min_support} genomes")
    else:
        min_support = min_genome_support
        print(f"Using absolute genome support = {min_support} genomes")

    # Find conserved k-mers
    conserved_kmers = find_conserved_kmers(sequences, k, min_support)

    if not conserved_kmers:
        print("No conserved k-mers found!")
        return []

    # Merge into blocks
    blocks = merge_overlapping_kmers(conserved_kmers, sequences, k, min_support)

    # Analyze
    stats = analyze_blocks(blocks, n_genomes)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump([b.to_dict() for b in blocks], f, indent=2)
    print(f"\nWrote {output_path}")

    # Save stats
    stats_path = output_path.with_suffix('.stats.json')
    stats['k'] = k
    stats['min_genome_support'] = min_support
    stats['n_genomes'] = n_genomes
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Also save as TSV for easy viewing
    tsv_path = output_path.with_suffix('.tsv')
    with open(tsv_path, 'w') as f:
        f.write("block_id\tn_orthogroups\tn_genomes\torthogroups\tgenomes\n")
        for b in blocks:
            f.write(f"{b.block_id}\t{len(b.orthogroups)}\t{b.n_genomes}\t")
            f.write(f"{','.join(b.orthogroups)}\t{','.join(b.genomes)}\n")
    print(f"Wrote {tsv_path}")

    return blocks


def main():
    parser = argparse.ArgumentParser(description="Find conserved gene neighborhoods")
    parser.add_argument("sequences_json", type=Path, help="OG sequences from Phase 2")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("-k", "--kmer-size", type=int, default=3,
                        help="K-mer size (default: 3)")
    parser.add_argument("--min-genomes", type=int, default=None,
                        help="Minimum genome support (absolute)")
    parser.add_argument("--min-fraction", type=float, default=0.5,
                        help="Minimum genome support as fraction (default: 0.5)")

    args = parser.parse_args()

    min_support = args.min_genomes if args.min_genomes else None
    min_fraction = None if args.min_genomes else args.min_fraction

    find_conserved_neighborhoods(
        args.sequences_json,
        args.output,
        k=args.kmer_size,
        min_genome_support=min_support or 2,
        min_genome_fraction=min_fraction,
    )


if __name__ == "__main__":
    main()
