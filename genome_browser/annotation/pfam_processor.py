#!/usr/bin/env python3
"""
PFAM annotation processor for ELSA genome browser.
Integrates with astra command-line tool to generate PFAM domain annotations.
"""

import subprocess
import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class PfamAnnotator:
    """PFAM domain annotation using astra command-line tool."""
    
    def __init__(self, threads: int = 8, evalue_threshold: float = 1e-5):
        self.threads = threads
        self.evalue_threshold = evalue_threshold
        self.temp_dir = None
        
    def check_astra_installation(self) -> bool:
        """Check if astra is installed and available."""
        try:
            result = subprocess.run(['astra', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def run_astra_scan(self, protein_file: Path, output_dir: Path, 
                       timeout: int = 3600) -> Dict:
        """
        Run astra PFAM scan on a single protein file.
        
        Args:
            protein_file: Path to protein FASTA file
            output_dir: Directory for output files
            timeout: Timeout in seconds
            
        Returns:
            Dict with execution results and statistics
        """
        start_time = time.time()
        
        result = {
            "protein_file": str(protein_file),
            "execution_status": "failed",
            "execution_time_seconds": 0.0,
            "error_message": None,
            "hits_file": None,
            "total_hits": 0,
            "unique_proteins": 0,
            "unique_domains": 0
        }
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Build astra command - pass directory containing only protein files
            cmd = [
                "astra", "search",
                "--prot_in", str(protein_file.parent),  # Directory with clean protein files
                "--installed_hmms", "PFAM", 
                "--outdir", str(output_dir),
                "--threads", str(self.threads),
                "--cut_ga"  # Use gathering cutoffs
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Execute astra
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            if process.returncode != 0:
                result["error_message"] = f"Astra failed: {process.stderr}"
                logger.error(f"Astra error: {process.stderr}")
                return result
            
            # Find results file
            hits_file = output_dir / "PFAM_hits_df.tsv"
            if not hits_file.exists():
                # Try alternative file names
                potential_files = list(output_dir.glob("*hits*.tsv"))
                if potential_files:
                    hits_file = potential_files[0]
                else:
                    result["error_message"] = f"No hits file found in {output_dir}"
                    return result
            
            # Parse results
            try:
                df = pd.read_csv(hits_file, sep='\t')
                result["total_hits"] = len(df)
                result["unique_proteins"] = df['sequence_id'].nunique() if 'sequence_id' in df.columns else 0
                result["unique_domains"] = df['hmm_name'].nunique() if 'hmm_name' in df.columns else 0
                result["hits_file"] = str(hits_file)
                result["execution_status"] = "success"
                
                logger.info(f"PFAM scan complete: {result['total_hits']} hits, "
                           f"{result['unique_proteins']} proteins, "
                           f"{result['unique_domains']} domains")
                
            except Exception as e:
                result["error_message"] = f"Failed to parse results: {e}"
                logger.error(f"Parse error: {e}")
                
        except subprocess.TimeoutExpired:
            result["error_message"] = "Astra scan timed out"
            logger.error(f"Astra scan timed out after {timeout} seconds")
            
        except Exception as e:
            result["error_message"] = f"Unexpected error: {e}"
            logger.error(f"Unexpected error: {e}")
        
        result["execution_time_seconds"] = round(time.time() - start_time, 2)
        return result
    
    def process_hits_file(self, hits_file: Path) -> pd.DataFrame:
        """
        Process astra hits file to extract significant domains.
        
        Args:
            hits_file: Path to astra hits TSV file
            
        Returns:
            Filtered DataFrame with significant hits
        """
        try:
            df = pd.read_csv(hits_file, sep='\t')
            logger.info(f"Loaded {len(df)} total hits from {hits_file}")
            
            # Filter by E-value threshold
            significant = df[df['evalue'] <= self.evalue_threshold].copy()
            logger.info(f"Filtered to {len(significant)} significant hits "
                       f"(E-value <= {self.evalue_threshold})")
            
            # Sort by position for consistent domain ordering
            if 'env_from' in significant.columns:
                significant = significant.sort_values(['sequence_id', 'env_from'])
            
            return significant
            
        except Exception as e:
            logger.error(f"Error processing hits file {hits_file}: {e}")
            return pd.DataFrame()
    
    def create_domain_annotations(self, hits_df: pd.DataFrame) -> Dict[str, str]:
        """
        Create semicolon-separated PFAM domain annotations per protein.
        
        Args:
            hits_df: DataFrame of significant PFAM hits
            
        Returns:
            Dict mapping protein_id -> "domain1;domain2;domain3"
        """
        domain_annotations = {}
        
        if hits_df.empty:
            return domain_annotations
        
        # Group by protein and create domain strings
        for protein_id, protein_hits in hits_df.groupby('sequence_id'):
            # Sort by envelope start position
            sorted_hits = protein_hits.sort_values('env_from')
            
            # Extract domain names (keep version numbers)
            domains = []
            for _, hit in sorted_hits.iterrows():
                domain_name = hit['hmm_name']  # e.g., "PF00001.21"
                domains.append(domain_name)
            
            # Join with semicolons
            domain_string = ';'.join(domains)
            domain_annotations[protein_id] = domain_string
        
        logger.info(f"Created domain annotations for {len(domain_annotations)} proteins")
        return domain_annotations
    
    def annotate_genome(self, genome_id: str, protein_file: Path, 
                       output_dir: Path) -> Tuple[Dict[str, str], Dict]:
        """
        Run complete PFAM annotation pipeline for a genome.
        
        Args:
            genome_id: Genome identifier
            protein_file: Path to protein FASTA file
            output_dir: Output directory for results
            
        Returns:
            Tuple of (domain_annotations_dict, execution_stats)
        """
        logger.info(f"Starting PFAM annotation for genome: {genome_id}")
        
        # Create genome-specific output directory
        genome_output_dir = output_dir / f"{genome_id}_pfam"
        
        # Run astra scan
        scan_result = self.run_astra_scan(protein_file, genome_output_dir)
        
        domain_annotations = {}
        
        if scan_result["execution_status"] == "success":
            # Process hits file
            hits_file = Path(scan_result["hits_file"])
            hits_df = self.process_hits_file(hits_file)
            
            # Create domain annotations
            domain_annotations = self.create_domain_annotations(hits_df)
        
        # Compile execution stats
        execution_stats = {
            "genome_id": genome_id,
            "scan_result": scan_result,
            "annotated_proteins": len(domain_annotations),
            "total_domains": sum(len(domains.split(';')) for domains in domain_annotations.values() if domains),
            "annotation_timestamp": datetime.now().isoformat()
        }
        
        return domain_annotations, execution_stats

def batch_annotate_genomes(genome_files: List[Tuple[str, Path]], 
                          output_dir: Path,
                          threads: int = 8,
                          max_workers: int = 2) -> Dict:
    """
    Batch annotate multiple genomes with PFAM domains.
    
    Args:
        genome_files: List of (genome_id, protein_file_path) tuples
        output_dir: Output directory for all results
        threads: Threads per astra process
        max_workers: Number of concurrent astra processes
        
    Returns:
        Dict with all results and statistics
    """
    annotator = PfamAnnotator(threads=threads)
    
    # Check astra installation
    if not annotator.check_astra_installation():
        raise RuntimeError("Astra is not installed or not in PATH. "
                          "Please install astra and ensure it's accessible.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "genome_annotations": {},  # genome_id -> {protein_id: domain_string}
        "execution_stats": [],
        "summary": {}
    }
    
    # Process genomes (limit concurrency to avoid overwhelming system)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit annotation jobs
        future_to_genome = {
            executor.submit(annotator.annotate_genome, genome_id, protein_file, output_dir): genome_id
            for genome_id, protein_file in genome_files
        }
        
        # Collect results
        for future in as_completed(future_to_genome):
            genome_id = future_to_genome[future]
            try:
                domain_annotations, execution_stats = future.result()
                results["genome_annotations"][genome_id] = domain_annotations
                results["execution_stats"].append(execution_stats)
                
                logger.info(f"Completed {genome_id}: "
                           f"{len(domain_annotations)} annotated proteins")
                
            except Exception as e:
                logger.error(f"Failed to annotate {genome_id}: {e}")
                results["execution_stats"].append({
                    "genome_id": genome_id,
                    "scan_result": {"execution_status": "failed", "error_message": str(e)},
                    "annotated_proteins": 0,
                    "total_domains": 0
                })
    
    # Generate summary statistics
    successful_genomes = [s for s in results["execution_stats"] 
                         if s["scan_result"]["execution_status"] == "success"]
    
    results["summary"] = {
        "total_genomes": len(genome_files),
        "successful_genomes": len(successful_genomes),
        "failed_genomes": len(genome_files) - len(successful_genomes),
        "total_annotated_proteins": sum(s["annotated_proteins"] for s in successful_genomes),
        "total_domains": sum(s["total_domains"] for s in successful_genomes)
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
                       help="Threads per astra process")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Number of concurrent astra processes")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find protein files
    protein_files = list(args.protein_dir.glob("*.faa"))
    if not protein_files:
        logger.error(f"No protein files found in {args.protein_dir}")
        return
    
    # Create genome file list
    genome_files = []
    for protein_file in protein_files:
        genome_id = protein_file.stem  # Remove .faa extension
        genome_files.append((genome_id, protein_file))
    
    logger.info(f"Found {len(genome_files)} genomes to annotate")
    
    # Run batch annotation
    results = batch_annotate_genomes(
        genome_files=genome_files,
        output_dir=args.output_dir,
        threads=args.threads,
        max_workers=args.max_workers
    )
    
    # Display summary
    print(f"\nPFAM Annotation Summary:")
    print(f"  Total genomes: {results['summary']['total_genomes']}")
    print(f"  Successful: {results['summary']['successful_genomes']}")
    print(f"  Failed: {results['summary']['failed_genomes']}")
    print(f"  Annotated proteins: {results['summary']['total_annotated_proteins']:,}")
    print(f"  Total domains: {results['summary']['total_domains']:,}")

if __name__ == "__main__":
    main()