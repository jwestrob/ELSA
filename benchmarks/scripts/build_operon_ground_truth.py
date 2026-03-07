#!/usr/bin/env python3
"""
Build operon-based ground truth for ELSA benchmarking.

This script:
1. Parses operon definitions (operon_id -> gene list)
2. Parses GFF files to get gene names and positions for each genome
3. Identifies which operons are conserved across genome pairs
4. Outputs pairwise ground truth blocks

Usage:
    python benchmarks/scripts/build_operon_ground_truth.py --organism bsubtilis
    python benchmarks/scripts/build_operon_ground_truth.py --organism ecoli
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


@dataclass
class Gene:
    """Represents a gene with its annotation."""
    gene_id: str
    gene_name: str
    contig: str
    start: int
    end: int
    strand: str
    sample_id: str


@dataclass
class Operon:
    """Represents an operon definition."""
    operon_id: str
    genes: list[str]
    evidence: str


@dataclass
class ConservedOperon:
    """Represents a conserved operon instance between two genomes."""
    operon_id: str
    genome_a: str
    genome_b: str
    contig_a: str
    contig_b: str
    genes_a: list[str]
    genes_b: list[str]
    start_a: int
    end_a: int
    start_b: int
    end_b: int
    gene_idx_a: list[int]
    gene_idx_b: list[int]


def parse_operons(operon_file: Path) -> list[Operon]:
    """Parse operon TSV file into list of Operon objects."""
    operons = []
    with open(operon_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            genes = [g.strip() for g in row['genes'].split(',') if g.strip()]
            if len(genes) >= 2:  # Require at least 2 genes for operon
                operons.append(Operon(
                    operon_id=row['operon_id'],
                    genes=genes,
                    evidence=row.get('evidence', 'unknown')
                ))
    return operons


def parse_gff(gff_file: Path, sample_id: str) -> list[Gene]:
    """Parse GFF file to extract gene information."""
    genes = []
    gene_idx = 0

    with open(gff_file) as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type not in ('gene', 'CDS'):
                continue

            # Only process 'gene' features for position, get name from attributes
            if feature_type != 'gene':
                continue

            contig = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]

            # Extract gene name from attributes
            gene_name = None
            gene_id = None

            # Try to find gene name
            gene_match = re.search(r'gene=([^;]+)', attributes)
            if gene_match:
                gene_name = gene_match.group(1)

            # Get gene ID from Name or ID attribute
            id_match = re.search(r'ID=([^;]+)', attributes)
            if id_match:
                gene_id = id_match.group(1)

            name_match = re.search(r'Name=([^;]+)', attributes)
            if name_match and not gene_name:
                gene_name = name_match.group(1)

            # Get locus_tag as fallback
            locus_match = re.search(r'locus_tag=([^;]+)', attributes)
            if locus_match:
                if not gene_id:
                    gene_id = locus_match.group(1)
                if not gene_name:
                    gene_name = locus_match.group(1)

            if gene_name and gene_id:
                genes.append(Gene(
                    gene_id=gene_id,
                    gene_name=gene_name.lower(),  # Normalize to lowercase
                    contig=contig,
                    start=start,
                    end=end,
                    strand=strand,
                    sample_id=sample_id
                ))
                gene_idx += 1

    # Sort genes by position within each contig
    genes.sort(key=lambda g: (g.contig, g.start))

    return genes


def load_all_genomes(annotation_dir: Path) -> dict[str, list[Gene]]:
    """Load gene annotations from all GFF files in directory."""
    genomes = {}

    for gff_file in sorted(annotation_dir.glob('*.gff')):
        sample_id = gff_file.stem
        genes = parse_gff(gff_file, sample_id)
        if genes:
            genomes[sample_id] = genes
            print(f"  Loaded {sample_id}: {len(genes)} genes")

    return genomes


def build_gene_name_index(genomes: dict[str, list[Gene]]) -> dict[str, dict[str, list[Gene]]]:
    """Build index: gene_name -> genome_id -> list of Gene objects."""
    index = defaultdict(lambda: defaultdict(list))

    for genome_id, genes in genomes.items():
        for gene in genes:
            index[gene.gene_name][genome_id].append(gene)

    return index


def find_conserved_operons(
    operons: list[Operon],
    genomes: dict[str, list[Gene]],
    gene_index: dict[str, dict[str, list[Gene]]],
    min_genes: int = 2,
) -> list[ConservedOperon]:
    """
    Find conserved operon instances across genome pairs.

    An operon is conserved if:
    1. At least min_genes of its genes are present in both genomes
    2. The genes are syntenic (consecutive with same orientation)
    """
    conserved = []
    genome_ids = sorted(genomes.keys())

    # Build position index for each genome
    position_index = {}
    for genome_id, genes in genomes.items():
        pos_idx = {}
        for idx, gene in enumerate(genes):
            key = (gene.contig, gene.start)
            pos_idx[gene.gene_name] = (gene.contig, idx, gene)
        position_index[genome_id] = pos_idx

    for operon in operons:
        operon_genes = set(operon.genes)

        # Find genomes that have this operon
        genome_occurrences = {}  # genome_id -> list of (gene_name, contig, position, Gene)

        for genome_id in genome_ids:
            found_genes = []
            for gene_name in operon.genes:
                gene_name_lower = gene_name.lower()
                if gene_name_lower in gene_index and genome_id in gene_index[gene_name_lower]:
                    for gene in gene_index[gene_name_lower][genome_id]:
                        # Get position in genome
                        genes_list = genomes[genome_id]
                        for idx, g in enumerate(genes_list):
                            if g.gene_id == gene.gene_id:
                                found_genes.append((gene_name_lower, gene.contig, idx, gene))
                                break

            if len(found_genes) >= min_genes:
                genome_occurrences[genome_id] = found_genes

        # Create pairwise comparisons
        genome_list = list(genome_occurrences.keys())
        for i, genome_a in enumerate(genome_list):
            for genome_b in genome_list[i+1:]:
                genes_a = genome_occurrences[genome_a]
                genes_b = genome_occurrences[genome_b]

                # Check if genes are syntenic (consecutive positions on same contig)
                # Group by contig and check for consecutive genes
                def check_synteny(gene_list):
                    """Check if genes form a syntenic block."""
                    if not gene_list:
                        return None, None, None, None, None

                    # Group by contig
                    contig_genes = defaultdict(list)
                    for name, contig, pos, gene in gene_list:
                        contig_genes[contig].append((pos, name, gene))

                    # Find best contig (most genes)
                    best_contig = max(contig_genes.keys(), key=lambda c: len(contig_genes[c]))
                    contig_gene_list = sorted(contig_genes[best_contig])

                    if len(contig_gene_list) < min_genes:
                        return None, None, None, None, None

                    # Check if genes are roughly consecutive (allow small gaps)
                    positions = [p for p, n, g in contig_gene_list]
                    max_gap = 3  # Allow up to 3 genes between operon members

                    is_syntenic = True
                    for j in range(len(positions) - 1):
                        if positions[j+1] - positions[j] > max_gap + 1:
                            is_syntenic = False
                            break

                    if not is_syntenic:
                        return None, None, None, None, None

                    gene_names = [n for p, n, g in contig_gene_list]
                    gene_objs = [g for p, n, g in contig_gene_list]
                    gene_positions = positions

                    return best_contig, gene_names, gene_objs, gene_positions, is_syntenic

                contig_a, names_a, objs_a, pos_a, syn_a = check_synteny(genes_a)
                contig_b, names_b, objs_b, pos_b, syn_b = check_synteny(genes_b)

                if syn_a and syn_b and len(names_a) >= min_genes and len(names_b) >= min_genes:
                    # Check if the operons share enough common genes
                    common = set(names_a) & set(names_b)
                    if len(common) >= min_genes:
                        conserved.append(ConservedOperon(
                            operon_id=operon.operon_id,
                            genome_a=genome_a,
                            genome_b=genome_b,
                            contig_a=contig_a,
                            contig_b=contig_b,
                            genes_a=names_a,
                            genes_b=names_b,
                            start_a=min(g.start for g in objs_a),
                            end_a=max(g.end for g in objs_a),
                            start_b=min(g.start for g in objs_b),
                            end_b=max(g.end for g in objs_b),
                            gene_idx_a=pos_a,
                            gene_idx_b=pos_b,
                        ))

    return conserved


def write_ground_truth(conserved: list[ConservedOperon], output_path: Path):
    """Write conserved operons to TSV and JSON files."""

    # Write TSV
    tsv_path = output_path.with_suffix('.tsv')
    with open(tsv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'operon_id', 'genome_a', 'genome_b',
            'contig_a', 'contig_b',
            'start_a', 'end_a', 'start_b', 'end_b',
            'n_genes_a', 'n_genes_b',
            'genes_a', 'genes_b',
            'gene_idx_start_a', 'gene_idx_end_a',
            'gene_idx_start_b', 'gene_idx_end_b',
        ])
        for c in conserved:
            writer.writerow([
                c.operon_id, c.genome_a, c.genome_b,
                c.contig_a, c.contig_b,
                c.start_a, c.end_a, c.start_b, c.end_b,
                len(c.genes_a), len(c.genes_b),
                ','.join(c.genes_a), ','.join(c.genes_b),
                min(c.gene_idx_a), max(c.gene_idx_a),
                min(c.gene_idx_b), max(c.gene_idx_b),
            ])

    print(f"  Wrote {tsv_path}")

    # Write JSON
    json_path = output_path.with_suffix('.json')
    data = []
    for c in conserved:
        data.append({
            'operon_id': c.operon_id,
            'genome_a': c.genome_a,
            'genome_b': c.genome_b,
            'contig_a': c.contig_a,
            'contig_b': c.contig_b,
            'start_a': c.start_a,
            'end_a': c.end_a,
            'start_b': c.start_b,
            'end_b': c.end_b,
            'genes_a': c.genes_a,
            'genes_b': c.genes_b,
            'gene_idx_a': c.gene_idx_a,
            'gene_idx_b': c.gene_idx_b,
        })

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Wrote {json_path}")

    # Write stats
    stats = {
        'total_conserved_operons': len(conserved),
        'unique_operons': len(set(c.operon_id for c in conserved)),
        'genome_pairs': len(set((c.genome_a, c.genome_b) for c in conserved)),
        'avg_genes_per_operon': np.mean([len(c.genes_a) for c in conserved]) if conserved else 0,
    }

    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Wrote {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Build operon ground truth')
    parser.add_argument('--organism', required=True, choices=['bsubtilis', 'ecoli'],
                        help='Organism to process')
    parser.add_argument('--min-genes', type=int, default=2,
                        help='Minimum genes for operon conservation')
    args = parser.parse_args()

    # Set paths based on organism
    if args.organism == 'bsubtilis':
        annotation_dir = BENCHMARKS_DIR / 'data' / 'bacillus' / 'annotations'
        operon_file = BENCHMARKS_DIR / 'operons' / 'bsubtilis' / 'operons.tsv'
        output_path = BENCHMARKS_DIR / 'ground_truth' / 'bsubtilis_operon_gt'
    else:
        annotation_dir = BENCHMARKS_DIR / 'data' / 'ecoli' / 'annotations'
        operon_file = BENCHMARKS_DIR / 'operons' / 'ecoli' / 'operons.tsv'
        output_path = BENCHMARKS_DIR / 'ground_truth' / 'ecoli_operon_gt'

    print("=" * 60)
    print(f"Building operon ground truth for {args.organism}")
    print("=" * 60)

    # Check files exist
    if not annotation_dir.exists():
        print(f"Error: Annotation directory not found: {annotation_dir}")
        return

    if not operon_file.exists():
        print(f"Error: Operon file not found: {operon_file}")
        return

    # Load operons
    print("\n[1/4] Loading operon definitions...")
    operons = parse_operons(operon_file)
    print(f"  Loaded {len(operons)} operons with ≥2 genes")

    # Load genomes
    print("\n[2/4] Loading genome annotations...")
    genomes = load_all_genomes(annotation_dir)
    print(f"  Loaded {len(genomes)} genomes")

    # Build gene name index
    print("\n[3/4] Building gene index...")
    gene_index = build_gene_name_index(genomes)
    print(f"  Indexed {len(gene_index)} unique gene names")

    # Find conserved operons
    print("\n[4/4] Finding conserved operons...")
    conserved = find_conserved_operons(
        operons, genomes, gene_index,
        min_genes=args.min_genes
    )
    print(f"  Found {len(conserved)} conserved operon instances")

    # Write output
    print("\nWriting output files...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = write_ground_truth(conserved, output_path)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Conserved operon instances: {stats['total_conserved_operons']}")
    print(f"  Unique operons found: {stats['unique_operons']}")
    print(f"  Genome pairs: {stats['genome_pairs']}")
    print(f"  Avg genes per instance: {stats['avg_genes_per_operon']:.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
