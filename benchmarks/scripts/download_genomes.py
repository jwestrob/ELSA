#!/usr/bin/env python3
"""Download genomes from NCBI for benchmarking."""

import subprocess
import json
import sys
from pathlib import Path
import shutil
import zipfile
import tempfile

# Configuration
SPECIES = {
    "ecoli": {
        "taxon": "Escherichia coli",
        "count": 20,
    },
    "bacillus": {
        "taxon": "Bacillus subtilis",
        "count": 20,
    },
    "spneumo": {
        "taxon": "Streptococcus pneumoniae",
        "count": 20,
    },
}

BASE_DIR = Path(__file__).parent.parent / "data"


def get_genome_accessions(taxon: str, count: int) -> list[str]:
    """Get list of RefSeq accessions for a species."""
    cmd = [
        "datasets", "summary", "genome", "taxon", taxon,
        "--assembly-level", "complete",
        "--assembly-source", "RefSeq",
        "--limit", str(count + 10),  # Get extras in case some fail
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error querying NCBI for {taxon}: {result.stderr}")
        return []

    data = json.loads(result.stdout)
    reports = data.get("reports", [])

    accessions = []
    for report in reports[:count]:
        acc = report.get("accession")
        if acc:
            accessions.append(acc)

    return accessions


def download_genome(accession: str, output_dir: Path) -> bool:
    """Download a single genome with proteins and GFF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        zippath = tmppath / "genome.zip"

        cmd = [
            "datasets", "download", "genome", "accession", accession,
            "--include", "genome,protein,gff3",
            "--filename", str(zippath),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error downloading {accession}: {result.stderr}")
            return False

        # Extract files
        try:
            with zipfile.ZipFile(zippath, 'r') as zf:
                zf.extractall(tmppath)

            # Find and move files
            data_dir = tmppath / "ncbi_dataset" / "data" / accession
            if not data_dir.exists():
                # Try alternative path structure
                for candidate in (tmppath / "ncbi_dataset" / "data").iterdir():
                    if candidate.is_dir():
                        data_dir = candidate
                        break

            if not data_dir.exists():
                print(f"  Could not find data directory for {accession}")
                return False

            # Copy files to appropriate directories
            genome_dir = output_dir / "genomes"
            protein_dir = output_dir / "proteins"
            annot_dir = output_dir / "annotations"

            for fna in data_dir.glob("*.fna"):
                shutil.copy(fna, genome_dir / f"{accession}.fna")
                break

            for faa in data_dir.glob("*.faa"):
                shutil.copy(faa, protein_dir / f"{accession}.faa")
                break

            for gff in data_dir.glob("*.gff"):
                shutil.copy(gff, annot_dir / f"{accession}.gff")
                break

            return True

        except Exception as e:
            print(f"  Error extracting {accession}: {e}")
            return False


def download_species(species_key: str, config: dict) -> int:
    """Download all genomes for a species."""
    taxon = config["taxon"]
    count = config["count"]
    output_dir = BASE_DIR / species_key

    print(f"\n{'='*60}")
    print(f"Downloading {count} {taxon} genomes")
    print(f"{'='*60}")

    # Get accessions
    print(f"Querying NCBI for accessions...")
    accessions = get_genome_accessions(taxon, count)

    if not accessions:
        print(f"No accessions found for {taxon}")
        return 0

    print(f"Found {len(accessions)} accessions")

    # Download each genome
    success_count = 0
    for i, acc in enumerate(accessions, 1):
        print(f"[{i}/{len(accessions)}] Downloading {acc}...")
        if download_genome(acc, output_dir):
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed")

        if success_count >= count:
            break

    print(f"\nDownloaded {success_count}/{count} genomes for {taxon}")
    return success_count


def main():
    # Check that datasets CLI is available
    result = subprocess.run(["which", "datasets"], capture_output=True)
    if result.returncode != 0:
        print("Error: NCBI datasets CLI not found. Install with:")
        print("  conda install -c conda-forge ncbi-datasets-cli")
        sys.exit(1)

    # Download each species
    total = 0
    for species_key, config in SPECIES.items():
        count = download_species(species_key, config)
        total += count

    print(f"\n{'='*60}")
    print(f"Total genomes downloaded: {total}")
    print(f"{'='*60}")

    # Summary
    for species_key in SPECIES:
        genome_dir = BASE_DIR / species_key / "genomes"
        n_genomes = len(list(genome_dir.glob("*.fna")))
        print(f"  {species_key}: {n_genomes} genomes")


if __name__ == "__main__":
    main()
