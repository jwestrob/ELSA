#!/usr/bin/env python3
"""
Download Salmonella and Klebsiella genomes for cross-species synteny benchmark.

These are close relatives of E. coli in the Enterobacteriaceae family.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BENCHMARKS_DIR / 'data' / 'enterobacteriaceae'

# Representative complete genomes from NCBI
# Selected for: complete genome, RefSeq, good quality

SALMONELLA_GENOMES = [
    # Salmonella enterica subsp. enterica serovar Typhimurium
    ('GCF_000006945.2', 'Salmonella_Typhimurium_LT2'),
    ('GCF_000022165.1', 'Salmonella_Typhimurium_14028S'),
    # Salmonella enterica subsp. enterica serovar Typhi
    ('GCF_000007545.1', 'Salmonella_Typhi_CT18'),
    ('GCF_000195995.1', 'Salmonella_Typhi_Ty2'),
    # Salmonella enterica subsp. enterica serovar Enteritidis
    ('GCF_000009505.1', 'Salmonella_Enteritidis_P125109'),
]

KLEBSIELLA_GENOMES = [
    # Klebsiella pneumoniae
    ('GCF_000240185.1', 'Klebsiella_pneumoniae_HS11286'),
    ('GCF_000016305.1', 'Klebsiella_pneumoniae_MGH78578'),
    ('GCF_000742755.1', 'Klebsiella_pneumoniae_KPNIH1'),
    ('GCF_000733495.1', 'Klebsiella_pneumoniae_1084'),
    ('GCF_000714595.1', 'Klebsiella_pneumoniae_KP617'),
]


def download_genome(accession: str, name: str, output_dir: Path) -> bool:
    """Download a genome using NCBI datasets CLI."""
    genome_dir = output_dir / 'genomes'
    protein_dir = output_dir / 'proteins'
    annotation_dir = output_dir / 'annotations'

    for d in [genome_dir, protein_dir, annotation_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    fna_file = genome_dir / f'{accession}.fna'
    if fna_file.exists():
        print(f'  {accession} already downloaded')
        return True

    print(f'  Downloading {accession} ({name})...')

    # Use ncbi-datasets CLI
    try:
        # Download genome package
        cmd = [
            'datasets', 'download', 'genome', 'accession', accession,
            '--include', 'genome,protein,gff3',
            '--filename', f'/tmp/{accession}.zip'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'    Error downloading: {result.stderr}')
            return False

        # Unzip
        subprocess.run(['unzip', '-o', f'/tmp/{accession}.zip', '-d', f'/tmp/{accession}'],
                      capture_output=True)

        # Find and copy files
        import glob

        # Genome sequence
        fna_files = glob.glob(f'/tmp/{accession}/ncbi_dataset/data/{accession}/*.fna')
        if fna_files:
            subprocess.run(['cp', fna_files[0], str(fna_file)])

        # Protein sequences
        faa_files = glob.glob(f'/tmp/{accession}/ncbi_dataset/data/{accession}/*.faa')
        if faa_files:
            subprocess.run(['cp', faa_files[0], str(protein_dir / f'{accession}.faa')])

        # GFF annotation
        gff_files = glob.glob(f'/tmp/{accession}/ncbi_dataset/data/{accession}/*.gff')
        if gff_files:
            subprocess.run(['cp', gff_files[0], str(annotation_dir / f'{accession}.gff')])

        # Cleanup
        subprocess.run(['rm', '-rf', f'/tmp/{accession}', f'/tmp/{accession}.zip'])

        return True

    except FileNotFoundError:
        print('    Error: ncbi-datasets CLI not installed')
        print('    Install with: conda install -c conda-forge ncbi-datasets-cli')
        return False


def main():
    print('=' * 60)
    print('Downloading Enterobacteriaceae genomes for cross-species benchmark')
    print('=' * 60)

    print(f'\nOutput directory: {OUTPUT_DIR}')

    print('\n[1/2] Downloading Salmonella genomes...')
    for accession, name in SALMONELLA_GENOMES:
        download_genome(accession, name, OUTPUT_DIR)

    print('\n[2/2] Downloading Klebsiella genomes...')
    for accession, name in KLEBSIELLA_GENOMES:
        download_genome(accession, name, OUTPUT_DIR)

    # Count what we have
    genomes = list((OUTPUT_DIR / 'genomes').glob('*.fna')) if (OUTPUT_DIR / 'genomes').exists() else []
    proteins = list((OUTPUT_DIR / 'proteins').glob('*.faa')) if (OUTPUT_DIR / 'proteins').exists() else []
    annotations = list((OUTPUT_DIR / 'annotations').glob('*.gff')) if (OUTPUT_DIR / 'annotations').exists() else []

    print('\n' + '=' * 60)
    print('Summary:')
    print(f'  Genomes: {len(genomes)}')
    print(f'  Proteins: {len(proteins)}')
    print(f'  Annotations: {len(annotations)}')
    print('=' * 60)

    if len(genomes) < 10:
        print('\nNote: Some downloads may have failed.')
        print('You can manually download from NCBI if needed.')


if __name__ == '__main__':
    main()
