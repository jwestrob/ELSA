#!/usr/bin/env python3
"""
Prepare MCScanX input from OrthoFinder results.

Handles two protein source formats:
1. NCBI proteins (Salmonella/Klebsiella): Use GFF for coordinates
2. Prodigal proteins (E. coli): Parse coordinates from FASTA headers

FASTA header format from Prodigal:
>contig_genenum # start # end # strand # attributes
e.g., >NZ_CP007265.1_1 # 47 # 1450 # 1 # ID=1_1;...
"""

import argparse
import gzip
import re
from pathlib import Path
from collections import defaultdict


def load_sequence_ids(path: Path) -> dict:
    """Load OrthoFinder sequence ID mapping: internal_id -> protein_id"""
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                internal_id = parts[0]
                protein_id = parts[1].split()[0]
                mapping[internal_id] = protein_id
    return mapping


def load_species_ids(path: Path) -> dict:
    """Load OrthoFinder species ID mapping: species_num -> genome_id"""
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                species_num = parts[0]
                genome_id = parts[1].replace('.faa', '')
                mapping[species_num] = genome_id
    return mapping


def parse_prodigal_fasta(fasta_path: Path) -> dict:
    """
    Parse Prodigal protein FASTA to get gene coordinates.
    Returns: protein_id -> (contig, start, end)
    """
    coords = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # Format: >contig_gene # start # end # strand # attrs
                parts = line[1:].strip().split(' # ')
                if len(parts) >= 4:
                    protein_id = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    # Parse contig from protein ID (format: contig_genenum)
                    contig_parts = protein_id.rsplit('_', 1)
                    contig = contig_parts[0] if len(contig_parts) > 1 else protein_id
                    coords[protein_id] = (contig, start, end)
    return coords


def parse_gff(gff_path: Path) -> dict:
    """
    Parse GFF to get NCBI protein coordinates.
    Returns: protein_id -> (contig, start, end)
    """
    coords = {}
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 9 and parts[2] == 'CDS':
                contig = parts[0]
                start = int(parts[3])
                end = int(parts[4])
                m = re.search(r'protein_id=([^;]+)', parts[8])
                if m:
                    coords[m.group(1)] = (contig, start, end)
    return coords


def is_prodigal_format(protein_path: Path) -> bool:
    """Check if protein FASTA has Prodigal-style headers."""
    with open(protein_path) as f:
        for line in f:
            if line.startswith('>'):
                # Prodigal format has " # " delimiters
                return ' # ' in line
    return False


def prepare_inputs(work_dir: Path, gff_dir: Path, protein_dir: Path,
                   output_prefix: Path):
    """Prepare MCScanX .gff and .blast files."""

    seq_ids = load_sequence_ids(work_dir / 'SequenceIDs.txt')
    species_ids = load_species_ids(work_dir / 'SpeciesIDs.txt')

    print(f"Loaded {len(seq_ids):,} sequence IDs, {len(species_ids)} species")

    # Build protein_id -> (genome, contig, start, end) mapping
    protein_coords = {}

    for species_num, genome_id in species_ids.items():
        protein_path = protein_dir / f"{genome_id}.faa"
        gff_path = gff_dir / f"{genome_id}.gff"

        if not protein_path.exists():
            print(f"  Warning: No protein file for {genome_id}")
            continue

        if is_prodigal_format(protein_path):
            # E. coli: parse Prodigal headers
            coords = parse_prodigal_fasta(protein_path)
            print(f"  {genome_id}: {len(coords):,} proteins (Prodigal)")
        elif gff_path.exists():
            # Salmonella/Klebsiella: use GFF
            coords = parse_gff(gff_path)
            print(f"  {genome_id}: {len(coords):,} proteins (GFF)")
        else:
            print(f"  Warning: No coords for {genome_id}")
            continue

        for pid, (contig, start, end) in coords.items():
            protein_coords[pid] = (genome_id, contig, start, end)

    print(f"\nTotal proteins with coordinates: {len(protein_coords):,}")

    # Reverse map: protein_id -> internal_id
    protein_to_internal = {v: k for k, v in seq_ids.items()}

    # Write MCScanX GFF
    gff_out = Path(f"{output_prefix}.gff")
    written = 0
    with open(gff_out, 'w') as f:
        for pid, (genome, contig, start, end) in protein_coords.items():
            if pid in protein_to_internal:
                internal_id = protein_to_internal[pid]
                chrom = f"{genome}_{contig}"
                f.write(f"{chrom}\t{internal_id}\t{start}\t{end}\n")
                written += 1

    print(f"Wrote {written:,} genes to {gff_out}")

    # Combine BLAST files
    blast_out = Path(f"{output_prefix}.blast")
    blast_files = list(work_dir.glob("Blast*.txt.gz"))

    with open(blast_out, 'w') as out:
        for bf in blast_files:
            with gzip.open(bf, 'rt') as f:
                for line in f:
                    out.write(line)

    blast_size = blast_out.stat().st_size / 1e6
    print(f"Wrote {blast_size:.1f} MB to {blast_out}")

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True,
                        help='OrthoFinder WorkingDirectory')
    parser.add_argument('--gff-dir', required=True)
    parser.add_argument('--protein-dir', required=True)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()

    prepare_inputs(
        Path(args.work_dir),
        Path(args.gff_dir),
        Path(args.protein_dir),
        Path(args.output_prefix)
    )


if __name__ == '__main__':
    main()
