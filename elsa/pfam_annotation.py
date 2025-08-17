"""
PFAM domain annotation using astra command-line tool.
Integrates PFAM annotation into the main ELSA pipeline.
"""

import subprocess
import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import logging

console = Console()
logger = logging.getLogger(__name__)


class PFAMAnnotator:
    """PFAM domain annotation using astra command-line tool."""
    
    def __init__(self, threads: int = 4):
        self.threads = threads
    
    def check_astra_installation(self) -> bool:
        """Check if astra is installed and available."""
        try:
            result = subprocess.run(['astra', '--help'], 
                                  capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def run_astra_scan(self, protein_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Run astra PFAM scan on a single protein file.
        
        Args:
            protein_file: Path to protein FASTA file
            output_dir: Directory for astra output
            
        Returns:
            Dictionary with scan results and metadata
        """
            
        # Initialize result dictionary
        result = {
            "genome": protein_file.stem,
            "protein_file": str(protein_file),
            "output_dir": str(output_dir),
            "success": False,
            "hits_file": None,
            "num_hits": 0,
            "runtime_seconds": 0,
            "error_message": None
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
            
            start_time = time.time()
            
            # Execute astra
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )
            
            result["runtime_seconds"] = time.time() - start_time
            
            if process.returncode != 0:
                result["error_message"] = f"Astra failed: {process.stderr}"
                logger.error(f"Astra error: {process.stderr}")
                return result
            
            # Look for hits file
            hits_file = output_dir / "PFAM_hits_df.tsv"
            if hits_file.exists():
                result["hits_file"] = str(hits_file)
                # Count hits (excluding header)
                with open(hits_file) as f:
                    result["num_hits"] = sum(1 for _ in f) - 1
                result["success"] = True
                logger.info(f"Astra scan completed: {result['num_hits']} hits in {result['runtime_seconds']:.1f}s")
            else:
                result["error_message"] = "Astra completed but no hits file found"
                logger.warning("Astra completed but no hits file found")
            
            return result
        
        except Exception as e:
            result["error_message"] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error during astra scan: {e}")
            return result
    
    def process_astra_hits(self, hits_file: Path, evalue_threshold: float = 1e-5,
                          score_threshold: float = 25.0) -> pd.DataFrame:
        """
        Process astra hits file to extract significant domains.
        
        Args:
            hits_file: Path to astra hits TSV file
            evalue_threshold: Maximum E-value for significant hits
            score_threshold: Minimum bitscore for significant hits
            
        Returns:
            DataFrame with filtered domain hits
        """
        if not hits_file.exists():
            return pd.DataFrame()
        
        try:
            # Read hits file
            df = pd.read_csv(hits_file, sep='\t')
            
            # Filter for significant hits
            significant = df[
                (df['full_E-value'] <= evalue_threshold) &
                (df['full_score'] >= score_threshold)
            ].copy()
            
            # Sort by E-value (best first)
            significant = significant.sort_values('full_E-value')
            
            return significant
            
        except Exception as e:
            logger.error(f"Error processing hits file {hits_file}: {e}")
            return pd.DataFrame()
    
    def annotate_genome(self, protein_file: Path, pfam_output_dir: Path) -> Dict[str, Any]:
        """
        Annotate a single genome with PFAM domains.
        
        Args:
            protein_file: Path to protein FASTA file
            pfam_output_dir: Base directory for PFAM results
            
        Returns:
            Dictionary with annotation results
        """
        genome_name = protein_file.stem
        genome_output_dir = pfam_output_dir / f"{genome_name}_pfam"
        
        logger.info(f"Annotating {genome_name} with PFAM domains...")
        
        # Run astra scan
        scan_result = self.run_astra_scan(protein_file, genome_output_dir)
        
        if scan_result["success"]:
            # Process hits
            hits_df = self.process_astra_hits(Path(scan_result["hits_file"]))
            scan_result["processed_hits"] = len(hits_df)
            
            # Save processed hits
            if not hits_df.empty:
                processed_file = genome_output_dir / "processed_hits.tsv"
                hits_df.to_csv(processed_file, sep='\t', index=False)
                scan_result["processed_hits_file"] = str(processed_file)
        
        return scan_result


def run_pfam_annotation_pipeline(proteins_dir: Path, output_dir: Path, threads: int = 4) -> Path:
    """
    Run PFAM annotation pipeline on all protein files in a directory.
    
    Args:
        proteins_dir: Directory containing protein FASTA files (.faa)
        output_dir: Directory for PFAM annotation results
        threads: Threads for astra process
        
    Returns:
        Path to results JSON file
    """
    protein_files = list(proteins_dir.glob("*.faa"))
    console.print(f"[green]Running PFAM annotation on {len(protein_files)} genomes in {proteins_dir}...[/green]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check astra installation
    annotator = PFAMAnnotator(threads=threads)
    if not annotator.check_astra_installation():
        raise RuntimeError("Astra is not installed or not in PATH. "
                          "Please install astra and ensure it's accessible.")
    
    # Run astra on entire proteins directory
    console.print(f"Running astra on directory: {proteins_dir}")
    
    # Build astra command - single run on entire directory
    cmd = [
        "astra", "search",
        "--prot_in", str(proteins_dir),
        "--installed_hmms", "PFAM", 
        "--outdir", str(output_dir),
        "--threads", str(threads),
        "--cut_ga"  # Use gathering cutoffs
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    console.print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Execute astra
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True
        )
        
        runtime = time.time() - start_time
        
        if process.returncode != 0:
            error_msg = f"Astra failed: {process.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Look for hits file
        hits_file = output_dir / "PFAM_hits_df.tsv"
        if not hits_file.exists():
            raise RuntimeError("Astra completed but no hits file found")
        
        # Count hits and process into genome_annotations structure
        import pandas as pd
        from collections import defaultdict
        
        df = pd.read_csv(hits_file, sep='\t')
        num_hits = len(df)
        
        # Process TSV into genome_annotations structure
        protein_domains = defaultdict(list)
        for _, row in df.iterrows():
            protein_id = row['sequence_id']
            domain = row['hmm_name']
            protein_domains[protein_id].append(domain)
        
        # Group proteins by genome using filename matching
        genome_annotations = defaultdict(dict)
        protein_file_stems = [f.stem for f in protein_files]  # Remove .faa extension
        
        for protein_id, domains in protein_domains.items():
            # Extract genome ID by matching against protein file names
            genome_id = None
            if '|' in protein_id:
                after_pipe = protein_id.split('|', 1)[1]
                # Find best matching genome file
                for stem in protein_file_stems:
                    if after_pipe.startswith(stem):
                        genome_id = stem
                        break
            
            if not genome_id:
                # Fallback: extract before '.con' pattern
                if '.con' in protein_id:
                    before_con = protein_id.split('.con')[0]
                    genome_id = before_con.split('|')[-1] if '|' in before_con else before_con
                else:
                    genome_id = protein_id.split('_')[0]
            
            domain_string = ';'.join(sorted(set(domains)))  # Remove duplicates and sort
            genome_annotations[genome_id][protein_id] = domain_string
        
        # Create results summary with genome_annotations
        results = {
            "method": "astra_bulk",
            "proteins_dir": str(proteins_dir),
            "output_dir": str(output_dir),
            "num_genomes": len(protein_files),
            "total_hits": num_hits,
            "runtime_seconds": runtime,
            "success": True,
            "hits_file": str(hits_file),
            "genome_annotations": dict(genome_annotations)
        }
        
        # Save results summary
        results_file = output_dir / "pfam_annotation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"[green]PFAM annotation completed![/green]")
        console.print(f"  • {len(protein_files)} genomes processed")
        console.print(f"  • {num_hits:,} total domain hits found")
        console.print(f"  • Runtime: {runtime:.1f} seconds")
        console.print(f"  • Results saved to: {results_file}")
        
        return results_file
        
    except Exception as e:
        error_msg = f"PFAM annotation failed: {e}"
        console.print(f"[red]{error_msg}[/red]")
        
        # Save error results
        results = {
            "method": "astra_bulk",
            "proteins_dir": str(proteins_dir),
            "success": False,
            "error_message": str(e),
            "runtime_seconds": time.time() - start_time
        }
        
        results_file = output_dir / "pfam_annotation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        raise