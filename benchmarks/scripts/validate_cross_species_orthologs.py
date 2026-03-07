#!/usr/bin/env python3
"""
Validate ELSA cross-species blocks against OrthoFinder orthogroups.

Fixed: NCBI WP_* protein accessions are shared across genomes.
Uses compound key (genome, contig, start) to avoid overwrites.
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re


def load_orthogroups(og_path: Path) -> dict:
    """Load gene -> orthogroup mapping."""
    gene_to_og = {}
    og_df = pd.read_csv(og_path, sep='\t')
    for _, row in og_df.iterrows():
        og_id = row['Orthogroup']
        for col in og_df.columns[1:]:
            if pd.notna(row[col]):
                for gene in str(row[col]).split(', '):
                    gene = gene.strip()
                    if gene:
                        gene_to_og[gene] = og_id
    return gene_to_og


def build_gene_coord_index(genes_df: pd.DataFrame) -> dict:
    """Build index: (genome, contig, gene_index) -> (genomic_start, genomic_end)."""
    index = {}
    for (sample_id, contig_id), group in genes_df.groupby(['sample_id', 'contig_id']):
        sorted_genes = group.sort_values('start')
        for idx, (_, row) in enumerate(sorted_genes.iterrows()):
            index[(sample_id, contig_id, idx)] = (row['start'], row['end'])
    return index


def load_gff_proteins(gff_dir: Path, genome_ids: list) -> dict:
    """
    Load proteins indexed by (genome, contig).
    Returns: {(genome, contig): [(start, end, protein_id), ...]}
    """
    protein_index = defaultdict(list)

    for gid in genome_ids:
        gff = gff_dir / f"{gid}.gff"
        if not gff.exists():
            continue

        with open(gff) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'CDS':
                    continue

                contig = parts[0]
                start = int(parts[3])
                end = int(parts[4])

                m = re.search(r'protein_id=([^;]+)', parts[8])
                if m:
                    protein_id = m.group(1)
                    protein_index[(gid, contig)].append((start, end, protein_id))

    # Sort each contig's proteins by start position
    for key in protein_index:
        protein_index[key] = sorted(protein_index[key])

    return dict(protein_index)


def get_block_genomic_coords(block, gene_coord_index: dict, prefix: str = 'query') -> tuple:
    """Convert block gene indices to genomic coordinates."""
    genome = block[f'{prefix}_genome']
    contig = block[f'{prefix}_contig']
    start_idx = int(block[f'{prefix}_start'])
    end_idx = int(block[f'{prefix}_end'])

    first_key = (genome, contig, start_idx)
    last_key = (genome, contig, end_idx)

    if first_key in gene_coord_index and last_key in gene_coord_index:
        first_start, _ = gene_coord_index[first_key]
        _, last_end = gene_coord_index[last_key]
        return (first_start, last_end)
    return None


def get_proteins_in_range(protein_index: dict, genome: str, contig: str,
                          start: int, end: int) -> list:
    """Get proteins overlapping a genomic range."""
    key = (genome, contig)
    if key not in protein_index:
        return []

    # Binary search could speed this up, but linear is fine for our data size
    return [pid for s, e, pid in protein_index[key] if s <= end and e >= start]


def validate_block(block, gene_coord_index: dict, protein_index: dict,
                   gene_to_og: dict) -> dict:
    """Validate a single block."""
    result = {
        'block_id': block['block_id'],
        'n_genes': block['n_genes'],
        'og_overlap': 0.0,
        'shared_ogs': 0,
        'query_ogs': 0,
        'target_ogs': 0,
    }

    # Get query genomic coords
    q_coords = get_block_genomic_coords(block, gene_coord_index, 'query')
    if not q_coords:
        return result

    # Get target genomic coords
    t_coords = get_block_genomic_coords(block, gene_coord_index, 'target')
    if not t_coords:
        return result

    # Find NCBI proteins in each region
    q_proteins = get_proteins_in_range(
        protein_index, block['query_genome'], block['query_contig'],
        q_coords[0], q_coords[1]
    )
    t_proteins = get_proteins_in_range(
        protein_index, block['target_genome'], block['target_contig'],
        t_coords[0], t_coords[1]
    )

    # Get orthogroups
    q_ogs = {gene_to_og[p] for p in q_proteins if p in gene_to_og}
    t_ogs = {gene_to_og[p] for p in t_proteins if p in gene_to_og}

    shared = q_ogs & t_ogs
    all_ogs = q_ogs | t_ogs

    result['query_ogs'] = len(q_ogs)
    result['target_ogs'] = len(t_ogs)
    result['shared_ogs'] = len(shared)
    result['og_overlap'] = len(shared) / len(all_ogs) if all_ogs else 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', required=True)
    parser.add_argument('--orthogroups', required=True)
    parser.add_argument('--genes', required=True)
    parser.add_argument('--gff-dir', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print("Loading orthogroups...")
    gene_to_og = load_orthogroups(Path(args.orthogroups))
    print(f"  {len(gene_to_og):,} mappings")

    print("Loading samples...")
    samples = pd.read_csv(args.samples, sep='\t')
    species_map = dict(zip(samples['sample_id'], samples['species']))
    genome_ids = samples['sample_id'].tolist()

    print("Building gene coordinate index...")
    genes = pd.read_parquet(args.genes, columns=['sample_id', 'contig_id', 'gene_id', 'start', 'end'])
    gene_coord_index = build_gene_coord_index(genes)
    print(f"  {len(gene_coord_index):,} gene positions")

    print("Loading GFF proteins...")
    protein_index = load_gff_proteins(Path(args.gff_dir), genome_ids)
    total_prots = sum(len(v) for v in protein_index.values())
    print(f"  {total_prots:,} proteins across {len(protein_index)} contigs")

    print("Loading blocks...")
    blocks = pd.read_csv(args.blocks)
    blocks['query_species'] = blocks['query_genome'].map(species_map)
    blocks['target_species'] = blocks['target_genome'].map(species_map)
    blocks['species_pair'] = blocks.apply(
        lambda r: '↔'.join(sorted([str(r['query_species']), str(r['target_species'])])), axis=1
    )
    blocks['is_cross'] = blocks['query_species'] != blocks['target_species']
    print(f"  {len(blocks):,} blocks")

    print("Validating...")
    results = []
    for idx, (_, block) in enumerate(blocks.iterrows()):
        if idx % 5000 == 0:
            print(f"  {idx:,}/{len(blocks):,}")
        results.append(validate_block(block, gene_coord_index, protein_index, gene_to_og))

    val = pd.DataFrame(results)
    val['species_pair'] = blocks['species_pair'].values
    val['is_cross'] = blocks['is_cross'].values

    # Filter to blocks with data
    val = val[(val['query_ogs'] > 0) | (val['target_ogs'] > 0)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, 'w') as f:
        f.write("# Cross-Species Ortholog Validation\n\n")
        f.write(f"**Dataset**: 30 Enterobacteriaceae (20 E.coli + 5 Salmonella + 5 Klebsiella)\n\n")
        f.write(f"**Blocks with OG data**: {len(val):,}\n\n")

        f.write("## Overall\n\n| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Blocks | {len(val):,} |\n")
        f.write(f"| Cross-genus | {val['is_cross'].sum():,} |\n")
        f.write(f"| Mean overlap | {val['og_overlap'].mean():.1%} |\n")
        f.write(f"| Median overlap | {val['og_overlap'].median():.1%} |\n")
        f.write(f"| ≥50% | {(val['og_overlap']>=0.5).sum():,} ({100*(val['og_overlap']>=0.5).mean():.1f}%) |\n")
        f.write(f"| ≥90% | {(val['og_overlap']>=0.9).sum():,} ({100*(val['og_overlap']>=0.9).mean():.1f}%) |\n\n")

        f.write("## By Species Pair\n\n| Pair | N | Mean | Median | ≥50% | ≥90% |\n|------|---|------|--------|------|------|\n")
        for pair in sorted(val['species_pair'].unique()):
            p = val[val['species_pair'] == pair]
            f.write(f"| {pair} | {len(p):,} | {p['og_overlap'].mean():.1%} | ")
            f.write(f"{p['og_overlap'].median():.1%} | {(p['og_overlap']>=0.5).sum():,} | ")
            f.write(f"{(p['og_overlap']>=0.9).sum():,} |\n")

        cross = val[val['is_cross']]
        f.write(f"\n## Cross-Genus (Key Result)\n\n| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Blocks | {len(cross):,} |\n")
        f.write(f"| Mean overlap | {cross['og_overlap'].mean():.1%} |\n")
        f.write(f"| Median overlap | {cross['og_overlap'].median():.1%} |\n")
        f.write(f"| ≥50% | {(cross['og_overlap']>=0.5).sum():,} ({100*(cross['og_overlap']>=0.5).mean():.1f}%) |\n")
        f.write(f"| ≥90% | {(cross['og_overlap']>=0.9).sum():,} ({100*(cross['og_overlap']>=0.9).mean():.1f}%) |\n")

        f.write("\n## Interpretation\n\n")
        overall_mean = val['og_overlap'].mean()
        cross_mean = cross['og_overlap'].mean()
        if cross_mean >= 0.7:
            f.write(f"**Excellent**: {cross_mean:.1%} cross-genus orthogroup overlap confirms ")
            f.write("ELSA detects true conserved synteny across genera.\n")
        elif cross_mean >= 0.5:
            f.write(f"**Good**: {cross_mean:.1%} cross-genus overlap shows ELSA finds ")
            f.write("conserved regions with some paralogs/novel genes.\n")
        else:
            f.write(f"**Moderate**: {cross_mean:.1%} overlap. Further investigation needed.\n")

    print(f"\nReport: {out}")
    print(f"\n=== SUMMARY ===")
    print(f"Blocks: {len(val):,}, Cross-genus: {val['is_cross'].sum():,}")
    print(f"Mean overlap: {val['og_overlap'].mean():.1%}")
    print(f"Cross-genus mean: {cross['og_overlap'].mean():.1%}")
    print(f"≥90% overlap: {(val['og_overlap']>=0.9).sum():,} ({100*(val['og_overlap']>=0.9).mean():.1f}%)")


if __name__ == '__main__':
    main()
