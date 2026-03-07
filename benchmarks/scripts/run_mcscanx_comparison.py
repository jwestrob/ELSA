#!/usr/bin/env python3
"""
Run MCScanX on the cross-species dataset and compare to ELSA results.

MCScanX input format:
- prefix.blast: gene1 gene2 evalue bitscore (or full BLAST m8 format)
- prefix.gff: chr gene start end

Output:
- prefix.collinearity: syntenic blocks
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict
import gzip
import re
import time


def load_sequence_ids(seq_ids_path: Path) -> dict:
    """Load OrthoFinder sequence ID mapping."""
    mapping = {}  # internal_id -> protein_id
    with open(seq_ids_path) as f:
        for line in f:
            # Format: "0_0: NP_446529.1 description"
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                internal_id = parts[0]
                protein_id = parts[1].split()[0]
                mapping[internal_id] = protein_id
    return mapping


def load_species_ids(species_ids_path: Path) -> dict:
    """Load OrthoFinder species ID mapping."""
    mapping = {}  # species_num -> genome_id
    with open(species_ids_path) as f:
        for line in f:
            # Format: "0: GCF_000006945.2.faa"
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                species_num = int(parts[0])
                genome_id = parts[1].replace('.faa', '')
                mapping[species_num] = genome_id
    return mapping


def create_mcscanx_gff(gff_dir: Path, genome_ids: list, output_path: Path,
                       seq_id_map: dict, species_map: dict):
    """Create MCScanX-format GFF from individual genome GFFs."""

    # Reverse map: protein_id -> internal_id
    protein_to_internal = {v: k for k, v in seq_id_map.items()}

    with open(output_path, 'w') as out:
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

                    # Get protein ID
                    m = re.search(r'protein_id=([^;]+)', parts[8])
                    if not m:
                        continue
                    protein_id = m.group(1)

                    # Get internal ID for MCScanX
                    if protein_id in protein_to_internal:
                        internal_id = protein_to_internal[protein_id]
                        # MCScanX format: chr gene start end
                        # Use genome_contig as chromosome
                        chrom = f"{gid}_{contig}"
                        out.write(f"{chrom}\t{internal_id}\t{start}\t{end}\n")


def combine_blast_files(blast_dir: Path, output_path: Path):
    """Combine all OrthoFinder BLAST results into single file for MCScanX."""
    blast_files = list(blast_dir.glob("Blast*.txt.gz"))

    with open(output_path, 'w') as out:
        for bf in blast_files:
            with gzip.open(bf, 'rt') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 12:
                        # MCScanX needs: gene1 gene2 evalue bitscore (or full m8)
                        # Keep full m8 format
                        out.write(line)


def parse_mcscanx_collinearity(collin_path: Path) -> pd.DataFrame:
    """Parse MCScanX collinearity output into DataFrame."""
    blocks = []
    current_block = None

    with open(collin_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('## Alignment'):
                # New block: ## Alignment 0: score=1234 e_value=0 N=50 chr1&chr2 plus
                parts = line.split()
                block_id = int(parts[2].rstrip(':'))
                score = float(parts[3].split('=')[1])
                n_genes = int(parts[5].split('=')[1])
                chroms = parts[6].split('&')
                orientation = parts[7] if len(parts) > 7 else 'plus'

                current_block = {
                    'block_id': block_id,
                    'score': score,
                    'n_genes': n_genes,
                    'query_chrom': chroms[0] if len(chroms) > 0 else '',
                    'target_chrom': chroms[1] if len(chroms) > 1 else '',
                    'orientation': orientation,
                    'genes': []
                }
            elif line.startswith('#') or not line:
                continue
            elif current_block is not None and '-' in line[:20]:
                # Gene pair: 0-  0: gene1 gene2
                current_block['genes'].append(line)

            # Save block when we see next alignment or EOF
            if current_block and current_block['genes']:
                if line.startswith('## Alignment') or not line:
                    if len(current_block['genes']) > 0:
                        blocks.append({
                            'block_id': current_block['block_id'],
                            'score': current_block['score'],
                            'n_genes': len(current_block['genes']),
                            'query_chrom': current_block['query_chrom'],
                            'target_chrom': current_block['target_chrom'],
                            'orientation': current_block['orientation'],
                        })
                    if line.startswith('## Alignment'):
                        # Don't reset, let next iteration handle it
                        pass
                    else:
                        current_block = None

    # Don't forget last block
    if current_block and current_block['genes']:
        blocks.append({
            'block_id': current_block['block_id'],
            'score': current_block['score'],
            'n_genes': len(current_block['genes']),
            'query_chrom': current_block['query_chrom'],
            'target_chrom': current_block['target_chrom'],
            'orientation': current_block['orientation'],
        })

    return pd.DataFrame(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orthofinder-dir', required=True,
                        help='OrthoFinder Results directory')
    parser.add_argument('--gff-dir', required=True,
                        help='Directory with genome GFF files')
    parser.add_argument('--samples', required=True,
                        help='samples.tsv file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for MCScanX results')
    parser.add_argument('--elsa-blocks', required=True,
                        help='ELSA blocks CSV for comparison')
    args = parser.parse_args()

    of_dir = Path(args.orthofinder_dir)
    work_dir = of_dir / 'WorkingDirectory'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OrthoFinder mappings...")
    seq_ids = load_sequence_ids(work_dir / 'SequenceIDs.txt')
    species_ids = load_species_ids(work_dir / 'SpeciesIDs.txt')
    print(f"  {len(seq_ids):,} sequences, {len(species_ids)} species")

    print("Loading sample info...")
    samples = pd.read_csv(args.samples, sep='\t')
    genome_ids = samples['sample_id'].tolist()
    species_map = dict(zip(samples['sample_id'], samples['species']))

    # Prepare MCScanX input
    prefix = out_dir / 'cross_species'

    print("Creating MCScanX GFF...")
    create_mcscanx_gff(Path(args.gff_dir), genome_ids,
                       Path(f"{prefix}.gff"), seq_ids, species_ids)

    print("Combining BLAST files...")
    combine_blast_files(work_dir, Path(f"{prefix}.blast"))

    # Check file sizes
    gff_size = Path(f"{prefix}.gff").stat().st_size
    blast_size = Path(f"{prefix}.blast").stat().st_size
    print(f"  GFF: {gff_size/1e6:.1f} MB, BLAST: {blast_size/1e6:.1f} MB")

    # Run MCScanX
    print("\nRunning MCScanX...")
    start_time = time.time()

    try:
        result = subprocess.run(
            ['MCScanX', str(prefix)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        mcscanx_time = time.time() - start_time
        print(f"  Completed in {mcscanx_time:.1f}s")

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return
    except subprocess.TimeoutExpired:
        print("  Timeout after 1 hour")
        return

    # Parse results
    collin_path = Path(f"{prefix}.collinearity")
    if not collin_path.exists():
        print("  No collinearity file generated")
        return

    print("\nParsing MCScanX results...")
    mcscanx_blocks = parse_mcscanx_collinearity(collin_path)
    print(f"  Found {len(mcscanx_blocks):,} syntenic blocks")

    # Load ELSA results for comparison
    print("\nLoading ELSA results...")
    elsa_blocks = pd.read_csv(args.elsa_blocks)
    print(f"  ELSA: {len(elsa_blocks):,} blocks")

    # Add species info
    def get_genome_from_chrom(chrom):
        # Format: GCF_XXXXX_contig
        parts = chrom.rsplit('_', 1)
        if len(parts) >= 2:
            return '_'.join(parts[:-1])
        return chrom

    if len(mcscanx_blocks) > 0:
        mcscanx_blocks['query_genome'] = mcscanx_blocks['query_chrom'].apply(
            lambda x: '_'.join(x.split('_')[:2]) if '_' in x else x)
        mcscanx_blocks['target_genome'] = mcscanx_blocks['target_chrom'].apply(
            lambda x: '_'.join(x.split('_')[:2]) if '_' in x else x)

        mcscanx_blocks['query_species'] = mcscanx_blocks['query_genome'].map(species_map)
        mcscanx_blocks['target_species'] = mcscanx_blocks['target_genome'].map(species_map)

        def sp_pair(row):
            s = sorted([str(row.get('query_species', '')), str(row.get('target_species', ''))])
            return f"{s[0]}↔{s[1]}"

        mcscanx_blocks['species_pair'] = mcscanx_blocks.apply(sp_pair, axis=1)
        mcscanx_blocks['is_cross'] = mcscanx_blocks['query_species'] != mcscanx_blocks['target_species']

    elsa_blocks['query_species'] = elsa_blocks['query_genome'].map(species_map)
    elsa_blocks['target_species'] = elsa_blocks['target_genome'].map(species_map)
    elsa_blocks['species_pair'] = elsa_blocks.apply(
        lambda r: '↔'.join(sorted([str(r['query_species']), str(r['target_species'])])), axis=1)
    elsa_blocks['is_cross'] = elsa_blocks['query_species'] != elsa_blocks['target_species']

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON: ELSA vs MCScanX")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'ELSA':>15} {'MCScanX':>15}")
    print("-" * 60)
    print(f"{'Total blocks':<30} {len(elsa_blocks):>15,} {len(mcscanx_blocks):>15,}")
    print(f"{'Cross-genus blocks':<30} {elsa_blocks['is_cross'].sum():>15,} {mcscanx_blocks['is_cross'].sum() if len(mcscanx_blocks) > 0 else 0:>15,}")
    print(f"{'Mean block size (genes)':<30} {elsa_blocks['n_genes'].mean():>15.1f} {mcscanx_blocks['n_genes'].mean() if len(mcscanx_blocks) > 0 else 0:>15.1f}")
    print(f"{'Max block size':<30} {elsa_blocks['n_genes'].max():>15,} {mcscanx_blocks['n_genes'].max() if len(mcscanx_blocks) > 0 else 0:>15,}")
    print(f"{'Runtime (seconds)':<30} {'~5':>15} {mcscanx_time:>15.1f}")

    # Save comparison
    comparison_path = out_dir / 'comparison_summary.md'
    with open(comparison_path, 'w') as f:
        f.write("# ELSA vs MCScanX Comparison\n\n")
        f.write(f"**Dataset**: 30 Enterobacteriaceae genomes\n\n")

        f.write("## Overall\n\n")
        f.write("| Metric | ELSA | MCScanX |\n")
        f.write("|--------|------|--------|\n")
        f.write(f"| Total blocks | {len(elsa_blocks):,} | {len(mcscanx_blocks):,} |\n")
        f.write(f"| Cross-genus blocks | {elsa_blocks['is_cross'].sum():,} | {mcscanx_blocks['is_cross'].sum() if len(mcscanx_blocks) > 0 else 0:,} |\n")
        f.write(f"| Mean block size | {elsa_blocks['n_genes'].mean():.1f} | {mcscanx_blocks['n_genes'].mean() if len(mcscanx_blocks) > 0 else 0:.1f} |\n")
        f.write(f"| Max block size | {elsa_blocks['n_genes'].max():,} | {mcscanx_blocks['n_genes'].max() if len(mcscanx_blocks) > 0 else 0:,} |\n")
        f.write(f"| Runtime | ~5s | {mcscanx_time:.1f}s |\n")

    # Save MCScanX blocks
    if len(mcscanx_blocks) > 0:
        mcscanx_blocks.to_csv(out_dir / 'mcscanx_blocks.csv', index=False)

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
