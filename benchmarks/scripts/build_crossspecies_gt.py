#!/usr/bin/env python3
"""
Build cross-species ground truth for ELSA benchmarking.

Uses OrthoFinder results to identify gene pairs that are neighbors in multiple species.
These represent recombination-resistant associations - genes that stay together
despite genomic shuffling across evolutionary time.

Ground truth definition:
- For each orthogroup pair (OG_A, OG_B), check if they are neighbors in each genome
- "Neighbors" = within N genes of each other on the same contig
- A pair is "conserved" if they are neighbors in genomes from ≥M species

Output:
- conserved_pairs.tsv: OG pairs that are neighbors in multiple species
- conserved_blocks.json: Connected components of conserved pairs (gene neighborhoods)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import csv


def load_orthogroups(orthogroups_file: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load OrthoFinder orthogroups.

    Returns: {orthogroup_id: {genome_id: [gene_ids]}}
    """
    orthogroups = defaultdict(lambda: defaultdict(list))

    with open(orthogroups_file) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)

        # Header format: Orthogroup, genome1, genome2, ...
        genome_cols = header[1:]

        for row in reader:
            og_id = row[0]
            for i, genes_str in enumerate(row[1:]):
                if genes_str.strip():
                    genome_id = genome_cols[i]
                    genes = [g.strip() for g in genes_str.split(',') if g.strip()]
                    orthogroups[og_id][genome_id] = genes

    return dict(orthogroups)


def load_gene_positions(gff_dir: Path) -> Dict[str, Dict[str, Tuple[str, int]]]:
    """
    Load gene positions from GFF files or protein FASTAs.

    Returns: {genome_id: {gene_id: (contig_id, position_index)}}
    """
    # This is a simplified version - in practice you'd parse GFF files
    # For now, we'll extract from protein FASTA headers (Prodigal format)
    gene_positions = defaultdict(dict)

    for faa_file in gff_dir.glob("*.faa"):
        genome_id = faa_file.stem

        # Parse protein FASTA to get gene order
        contig_genes = defaultdict(list)

        with open(faa_file) as f:
            for line in f:
                if line.startswith('>'):
                    # Prodigal format: >contig_genenum # start # end # strand # ...
                    parts = line[1:].split(' # ')
                    gene_id = parts[0].strip()

                    # Extract contig from gene_id (e.g., "NZ_CP010585.1_1" -> "NZ_CP010585.1")
                    if '_' in gene_id:
                        contig_id = '_'.join(gene_id.rsplit('_', 1)[:-1])
                    else:
                        contig_id = gene_id

                    if len(parts) >= 2:
                        start = int(parts[1])
                        contig_genes[contig_id].append((start, gene_id))

        # Sort by position and assign indices
        for contig_id, genes in contig_genes.items():
            genes.sort(key=lambda x: x[0])
            for idx, (_, gene_id) in enumerate(genes):
                gene_positions[genome_id][gene_id] = (contig_id, idx)

    return dict(gene_positions)


def find_orthogroup_neighbors(
    orthogroups: Dict[str, Dict[str, List[str]]],
    gene_positions: Dict[str, Dict[str, Tuple[str, int]]],
    max_distance: int = 5,
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Find orthogroup pairs that are neighbors in each genome.

    Returns: {(og_a, og_b): {genome_id: distance}}
    where distance is the number of genes between them (0 = adjacent)
    """
    # Build gene -> orthogroup mapping
    gene_to_og = {}
    for og_id, genome_genes in orthogroups.items():
        for genome_id, genes in genome_genes.items():
            for gene in genes:
                gene_to_og[gene] = og_id

    # For each genome, find OG pairs that are neighbors
    og_pair_genomes = defaultdict(dict)

    for genome_id, positions in gene_positions.items():
        # Group genes by contig
        contig_genes = defaultdict(list)
        for gene_id, (contig_id, idx) in positions.items():
            if gene_id in gene_to_og:
                contig_genes[contig_id].append((idx, gene_id, gene_to_og[gene_id]))

        # For each contig, find neighboring OG pairs
        for contig_id, genes in contig_genes.items():
            genes.sort(key=lambda x: x[0])

            for i, (idx_i, gene_i, og_i) in enumerate(genes):
                for j in range(i + 1, len(genes)):
                    idx_j, gene_j, og_j = genes[j]
                    distance = idx_j - idx_i - 1  # genes between them

                    if distance > max_distance:
                        break  # genes are too far apart

                    if og_i != og_j:  # different orthogroups
                        # Canonical order for pair
                        og_pair = tuple(sorted([og_i, og_j]))

                        # Record the distance (keep minimum if multiple instances)
                        if genome_id not in og_pair_genomes[og_pair]:
                            og_pair_genomes[og_pair][genome_id] = distance
                        else:
                            og_pair_genomes[og_pair][genome_id] = min(
                                og_pair_genomes[og_pair][genome_id], distance
                            )

    return dict(og_pair_genomes)


def identify_species(genome_id: str) -> str:
    """
    Extract species from genome ID based on prefix.

    Expects format like: ecoli_GCF_xxx, bsub_GCF_xxx, spneumo_GCF_xxx
    """
    if genome_id.startswith('ecoli_'):
        return 'E.coli'
    elif genome_id.startswith('bsub_'):
        return 'B.subtilis'
    elif genome_id.startswith('spneumo_'):
        return 'S.pneumoniae'
    else:
        # Try to extract from the genome ID
        parts = genome_id.split('_')
        if len(parts) > 1:
            return parts[0]
        return 'unknown'


def filter_conserved_pairs(
    og_pair_genomes: Dict[Tuple[str, str], Dict[str, int]],
    min_species: int = 2,
    min_genomes_per_species: int = 1,
) -> List[Dict]:
    """
    Filter to pairs conserved across multiple species.

    Args:
        og_pair_genomes: {(og_a, og_b): {genome_id: distance}}
        min_species: Minimum number of species where pair must be neighbors
        min_genomes_per_species: Minimum genomes per species to count that species

    Returns: List of conserved pair records
    """
    conserved_pairs = []

    for (og_a, og_b), genome_distances in og_pair_genomes.items():
        # Group by species
        species_genomes = defaultdict(list)
        for genome_id, distance in genome_distances.items():
            species = identify_species(genome_id)
            species_genomes[species].append((genome_id, distance))

        # Count species with sufficient support
        supported_species = []
        for species, genomes in species_genomes.items():
            if len(genomes) >= min_genomes_per_species:
                supported_species.append(species)

        if len(supported_species) >= min_species:
            conserved_pairs.append({
                'og_a': og_a,
                'og_b': og_b,
                'n_species': len(supported_species),
                'species': supported_species,
                'n_genomes': len(genome_distances),
                'genomes': list(genome_distances.keys()),
                'mean_distance': sum(genome_distances.values()) / len(genome_distances),
            })

    # Sort by number of species (descending)
    conserved_pairs.sort(key=lambda x: (-x['n_species'], -x['n_genomes']))

    return conserved_pairs


def build_conserved_blocks(conserved_pairs: List[Dict]) -> List[Dict]:
    """
    Build connected components from conserved pairs.

    If OG_A-OG_B and OG_B-OG_C are both conserved, they form a block {A, B, C}.
    """
    # Build adjacency graph
    adj = defaultdict(set)
    pair_info = {}

    for pair in conserved_pairs:
        og_a, og_b = pair['og_a'], pair['og_b']
        adj[og_a].add(og_b)
        adj[og_b].add(og_a)
        pair_info[(og_a, og_b)] = pair
        pair_info[(og_b, og_a)] = pair

    # Find connected components
    visited = set()
    blocks = []

    def dfs(node: str, component: Set[str]):
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        for neighbor in adj[node]:
            dfs(neighbor, component)

    for og in adj:
        if og not in visited:
            component = set()
            dfs(og, component)
            if len(component) >= 2:
                blocks.append(component)

    # Build block records
    block_records = []
    for i, component in enumerate(sorted(blocks, key=lambda x: -len(x))):
        ogs = sorted(component)

        # Find species coverage for this block
        species_set = set()
        genome_set = set()
        for j, og_a in enumerate(ogs):
            for og_b in ogs[j+1:]:
                key = tuple(sorted([og_a, og_b]))
                if key in pair_info:
                    info = pair_info[key]
                    species_set.update(info['species'])
                    genome_set.update(info['genomes'])

        block_records.append({
            'block_id': f'XSGT_{i:05d}',
            'orthogroups': ogs,
            'n_orthogroups': len(ogs),
            'species': sorted(species_set),
            'n_species': len(species_set),
            'genomes': sorted(genome_set),
            'n_genomes': len(genome_set),
        })

    return block_records


def main():
    parser = argparse.ArgumentParser(
        description="Build cross-species ground truth from OrthoFinder results"
    )
    parser.add_argument(
        "orthofinder_dir",
        type=Path,
        help="OrthoFinder results directory (e.g., Results_Jan28/)"
    )
    parser.add_argument(
        "proteins_dir",
        type=Path,
        help="Directory with protein FASTA files (to get gene positions)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("crossspecies_gt"),
        help="Output directory"
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5,
        help="Maximum gene distance to consider neighbors (default: 5)"
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=2,
        help="Minimum species for conserved pair (default: 2)"
    )
    parser.add_argument(
        "--min-genomes-per-species",
        type=int,
        default=3,
        help="Minimum genomes per species to count (default: 3)"
    )

    args = parser.parse_args()

    # Find orthogroups file
    og_file = args.orthofinder_dir / "Orthogroups" / "Orthogroups.tsv"
    if not og_file.exists():
        # Try alternate location
        og_file = args.orthofinder_dir / "Orthogroups.tsv"

    if not og_file.exists():
        raise FileNotFoundError(f"Orthogroups.tsv not found in {args.orthofinder_dir}")

    print(f"Loading orthogroups from {og_file}...")
    orthogroups = load_orthogroups(og_file)
    print(f"  Loaded {len(orthogroups)} orthogroups")

    print(f"Loading gene positions from {args.proteins_dir}...")
    gene_positions = load_gene_positions(args.proteins_dir)
    print(f"  Loaded positions for {len(gene_positions)} genomes")

    print(f"Finding orthogroup neighbors (max distance: {args.max_distance})...")
    og_neighbors = find_orthogroup_neighbors(
        orthogroups, gene_positions, args.max_distance
    )
    print(f"  Found {len(og_neighbors)} OG pairs as neighbors somewhere")

    print(f"Filtering conserved pairs (min {args.min_species} species)...")
    conserved_pairs = filter_conserved_pairs(
        og_neighbors,
        min_species=args.min_species,
        min_genomes_per_species=args.min_genomes_per_species,
    )
    print(f"  Found {len(conserved_pairs)} conserved pairs")

    print("Building conserved blocks...")
    blocks = build_conserved_blocks(conserved_pairs)
    print(f"  Created {len(blocks)} blocks")

    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)

    pairs_file = args.output / "conserved_pairs.json"
    with open(pairs_file, 'w') as f:
        json.dump(conserved_pairs, f, indent=2)
    print(f"Saved {len(conserved_pairs)} conserved pairs to {pairs_file}")

    blocks_file = args.output / "conserved_blocks.json"
    with open(blocks_file, 'w') as f:
        json.dump(blocks, f, indent=2)
    print(f"Saved {len(blocks)} conserved blocks to {blocks_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-SPECIES GROUND TRUTH SUMMARY")
    print("=" * 60)

    if conserved_pairs:
        species_dist = defaultdict(int)
        for pair in conserved_pairs:
            species_dist[pair['n_species']] += 1

        print(f"Conserved pairs by species count:")
        for n_species in sorted(species_dist.keys(), reverse=True):
            print(f"  {n_species} species: {species_dist[n_species]} pairs")

    if blocks:
        print(f"\nTop 10 largest blocks:")
        for block in blocks[:10]:
            print(f"  {block['block_id']}: {block['n_orthogroups']} OGs, "
                  f"{block['n_species']} species, {block['n_genomes']} genomes")

    print("=" * 60)


if __name__ == "__main__":
    main()
