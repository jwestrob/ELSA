#!/usr/bin/env python3
"""
PFAM annotation processor for ELSA genome browser.
Integrates with astra command-line tool to generate PFAM domain annotations.

Runs astra once on the entire protein directory, then subsets the resulting
hits TSV per genome based on protein ID ownership.
"""

import subprocess
import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from Bio import SeqIO

logger = logging.getLogger(__name__)


class PfamAnnotator:
    """PFAM domain annotation using astra command-line tool."""

    def __init__(self, threads: int = 8, evalue_threshold: float = 1e-5):
        self.threads = threads
        self.evalue_threshold = evalue_threshold

    def check_astra_installation(self) -> bool:
        """Check if astra is installed and available."""
        try:
            result = subprocess.run(['astra', '--help'],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def run_astra_scan(self, protein_dir: Path, output_dir: Path,
                       timeout: int = 3600) -> Dict:
        """Run astra PFAM scan on the entire protein directory (once)."""
        start_time = time.time()

        result = {
            "protein_dir": str(protein_dir),
            "execution_status": "failed",
            "execution_time_seconds": 0.0,
            "error_message": None,
            "hits_file": None,
            "total_hits": 0,
            "unique_proteins": 0,
            "unique_domains": 0,
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "astra", "search",
                "--prot_in", str(protein_dir),
                "--installed_hmms", "PFAM",
                "--outdir", str(output_dir),
                "--threads", str(self.threads),
                "--cut_ga",
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            process = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=timeout)

            if process.returncode != 0:
                result["error_message"] = f"Astra failed: {process.stderr}"
                logger.error(f"Astra error: {process.stderr}")
                return result

            hits_file = output_dir / "PFAM_hits_df.tsv"
            if not hits_file.exists():
                potential_files = list(output_dir.glob("*hits*.tsv"))
                if potential_files:
                    hits_file = potential_files[0]
                else:
                    result["error_message"] = f"No hits file found in {output_dir}"
                    return result

            df = pd.read_csv(hits_file, sep='\t')
            result["total_hits"] = len(df)
            result["unique_proteins"] = df['sequence_id'].nunique() if 'sequence_id' in df.columns else 0
            result["unique_domains"] = df['hmm_name'].nunique() if 'hmm_name' in df.columns else 0
            result["hits_file"] = str(hits_file)
            result["execution_status"] = "success"

            logger.info(f"PFAM scan complete: {result['total_hits']} hits, "
                        f"{result['unique_proteins']} proteins, "
                        f"{result['unique_domains']} domains")

        except subprocess.TimeoutExpired:
            result["error_message"] = "Astra scan timed out"
            logger.error(f"Astra scan timed out after {timeout} seconds")

        except Exception as e:
            result["error_message"] = f"Unexpected error: {e}"
            logger.error(f"Unexpected error: {e}")

        result["execution_time_seconds"] = round(time.time() - start_time, 2)
        return result

    def create_domain_annotations(self, hits_df: pd.DataFrame) -> Dict[str, str]:
        """Create semicolon-separated PFAM domain annotations per protein."""
        domain_annotations = {}

        if hits_df.empty:
            return domain_annotations

        for protein_id, protein_hits in hits_df.groupby('sequence_id'):
            sorted_hits = protein_hits.sort_values('env_from')
            domains = [hit['hmm_name'] for _, hit in sorted_hits.iterrows()]
            domain_annotations[protein_id] = ';'.join(domains)

        return domain_annotations


def _build_protein_to_genome(genome_files: List[Tuple[str, Path]]) -> Dict[str, str]:
    """Build a mapping from protein_id -> genome_id by reading .faa headers."""
    protein_to_genome = {}
    for genome_id, protein_file in genome_files:
        for record in SeqIO.parse(protein_file, "fasta"):
            protein_to_genome[record.id] = genome_id
    return protein_to_genome


def batch_annotate_genomes(genome_files: List[Tuple[str, Path]],
                           output_dir: Path,
                           threads: int = 8,
                           **kwargs) -> Dict:
    """
    Annotate all genomes with PFAM domains.

    Runs astra once on the shared protein directory, then subsets the
    resulting hits per genome based on protein ID ownership.
    """
    annotator = PfamAnnotator(threads=threads)

    if not annotator.check_astra_installation():
        raise RuntimeError("Astra is not installed or not in PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # All .faa files should be in the same directory
    protein_dir = genome_files[0][1].parent

    # 1. Run astra once on the whole directory
    scan_result = annotator.run_astra_scan(protein_dir, output_dir)

    if scan_result["execution_status"] != "success":
        logger.error(f"Astra scan failed: {scan_result['error_message']}")
        return {
            "genome_annotations": {},
            "execution_stats": [scan_result],
            "summary": {
                "total_genomes": len(genome_files),
                "successful_genomes": 0,
                "failed_genomes": len(genome_files),
                "total_annotated_proteins": 0,
                "total_domains": 0,
            },
        }

    # 2. Build protein_id -> genome_id mapping from .faa headers
    logger.info("Building protein-to-genome mapping from .faa headers...")
    protein_to_genome = _build_protein_to_genome(genome_files)
    logger.info(f"Mapped {len(protein_to_genome)} proteins across "
                f"{len(genome_files)} genomes")

    # 3. Load the full hits TSV and filter by e-value
    hits_df = pd.read_csv(scan_result["hits_file"], sep='\t')
    hits_df = hits_df[hits_df['evalue'] <= annotator.evalue_threshold]
    logger.info(f"Filtered to {len(hits_df)} significant hits "
                f"(E-value <= {annotator.evalue_threshold})")

    # 4. Subset hits per genome
    hits_df = hits_df.copy()
    hits_df['genome_id'] = hits_df['sequence_id'].map(protein_to_genome)

    results = {"genome_annotations": {}, "execution_stats": [], "summary": {}}

    for genome_id, _ in genome_files:
        genome_hits = hits_df[hits_df['genome_id'] == genome_id]
        domain_annotations = annotator.create_domain_annotations(genome_hits)
        results["genome_annotations"][genome_id] = domain_annotations

        stats = {
            "genome_id": genome_id,
            "scan_result": {"execution_status": "success"},
            "annotated_proteins": len(domain_annotations),
            "total_domains": sum(len(d.split(';')) for d in domain_annotations.values() if d),
            "annotation_timestamp": datetime.now().isoformat(),
        }
        results["execution_stats"].append(stats)
        logger.info(f"  {genome_id}: {len(domain_annotations)} annotated proteins")

    # Warn about unmapped hits
    unmapped = hits_df['genome_id'].isna().sum()
    if unmapped:
        logger.warning(f"{unmapped} hits could not be mapped to a genome")

    results["summary"] = {
        "total_genomes": len(genome_files),
        "successful_genomes": len(genome_files),
        "failed_genomes": 0,
        "total_annotated_proteins": sum(
            s["annotated_proteins"] for s in results["execution_stats"]
        ),
        "total_domains": sum(
            s["total_domains"] for s in results["execution_stats"]
        ),
    }

    # Save results
    results_file = output_dir / "pfam_annotation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Batch annotation complete: {results['summary']}")
    return results


def main():
    """Command line interface for PFAM annotation."""
    import argparse

    parser = argparse.ArgumentParser(description="PFAM annotation for ELSA genomes")
    parser.add_argument("--protein-dir", type=Path, required=True,
                        help="Directory containing protein FASTA files")
    parser.add_argument("--output-dir", type=Path, default=Path("pfam_annotations"),
                        help="Output directory for annotations")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads for astra")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="(ignored, kept for CLI compat)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    protein_files = list(args.protein_dir.glob("*.faa"))
    if not protein_files:
        logger.error(f"No protein files found in {args.protein_dir}")
        return

    genome_files = [(pf.stem, pf) for pf in protein_files]
    logger.info(f"Found {len(genome_files)} genomes to annotate")

    results = batch_annotate_genomes(
        genome_files=genome_files,
        output_dir=args.output_dir,
        threads=args.threads,
    )

    print(f"\nPFAM Annotation Summary:")
    print(f"  Total genomes: {results['summary']['total_genomes']}")
    print(f"  Successful: {results['summary']['successful_genomes']}")
    print(f"  Failed: {results['summary']['failed_genomes']}")
    print(f"  Annotated proteins: {results['summary']['total_annotated_proteins']:,}")
    print(f"  Total domains: {results['summary']['total_domains']:,}")


if __name__ == "__main__":
    main()
