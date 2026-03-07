#!/usr/bin/env python3
"""
Reorganize test data into proper directory structure for clean astra processing.

Current messy structure:
test_data/genomes/
├── genome1.fna (nucleotide)
├── genome1.gff (annotations) 
├── genome1.faa (proteins)
└── ...

New clean structure:
test_data/
├── sequences/         # Nucleotide FASTA files
├── annotations/       # GFF annotation files  
├── proteins/          # Protein FASTA files (for astra)
└── metadata/          # Any other metadata files
"""

import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reorganize_test_data(base_dir: Path):
    """Reorganize test data into clean directory structure."""
    
    genomes_dir = base_dir / "genomes"
    if not genomes_dir.exists():
        logger.error(f"Genomes directory not found: {genomes_dir}")
        return
    
    # Create new organized directories
    sequences_dir = base_dir / "sequences"
    annotations_dir = base_dir / "annotations" 
    proteins_dir = base_dir / "proteins"
    
    sequences_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    proteins_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created directories:")
    logger.info(f"  Sequences: {sequences_dir}")
    logger.info(f"  Annotations: {annotations_dir}")
    logger.info(f"  Proteins: {proteins_dir}")
    
    # Move files by extension
    moved_counts = {"fna": 0, "gff": 0, "faa": 0}
    
    for file_path in genomes_dir.iterdir():
        if not file_path.is_file():
            continue
            
        if file_path.suffix == ".fna":
            # Nucleotide sequences
            dest = sequences_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            moved_counts["fna"] += 1
            logger.info(f"Moved nucleotide: {file_path.name} → sequences/")
            
        elif file_path.suffix == ".gff":
            # Annotation files
            dest = annotations_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            moved_counts["gff"] += 1
            logger.info(f"Moved annotation: {file_path.name} → annotations/")
            
        elif file_path.suffix == ".faa":
            # Protein sequences
            dest = proteins_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            moved_counts["faa"] += 1
            logger.info(f"Moved proteins: {file_path.name} → proteins/")
            
        else:
            logger.warning(f"Unknown file type: {file_path.name}")
    
    # Remove empty genomes directory if everything moved
    try:
        genomes_dir.rmdir()
        logger.info("Removed empty genomes/ directory")
    except OSError:
        logger.info("Genomes directory not empty, leaving it")
    
    logger.info(f"\nReorganization complete!")
    logger.info(f"Moved {moved_counts['fna']} nucleotide files")
    logger.info(f"Moved {moved_counts['gff']} annotation files") 
    logger.info(f"Moved {moved_counts['faa']} protein files")
    
    logger.info(f"\nNew structure:")
    logger.info(f"test_data/")
    logger.info(f"├── sequences/     # {moved_counts['fna']} .fna files")
    logger.info(f"├── annotations/   # {moved_counts['gff']} .gff files")
    logger.info(f"├── proteins/      # {moved_counts['faa']} .faa files")
    logger.info(f"└── metadata/      # (preserved existing)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize test data structure")
    parser.add_argument("--test-data-dir", type=Path, default="../test_data",
                       help="Test data base directory")
    
    args = parser.parse_args()
    
    reorganize_test_data(args.test_data_dir)
    
    print(f"\n🎉 Reorganization complete!")
    print(f"\nNow you can run astra cleanly:")
    print(f"  astra search --prot_in {args.test_data_dir}/proteins --installed_hmms PFAM ...")
    print(f"\nAnd update your setup command:")
    print(f"  python setup_genome_browser.py \\")
    print(f"    --sequences-dir {args.test_data_dir}/sequences \\")
    print(f"    --annotations-dir {args.test_data_dir}/annotations \\")
    print(f"    --proteins-dir {args.test_data_dir}/proteins \\")
    print(f"    --blocks-file ../syntenic_analysis/syntenic_blocks.csv \\")
    print(f"    --clusters-file ../syntenic_analysis/syntenic_clusters.csv")

if __name__ == "__main__":
    main()