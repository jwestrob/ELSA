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


def run_pfam_annotation_pipeline(protein_files: List[Path], output_dir: Path,
                                threads: int = 4) -> Path:
    """
    Run PFAM annotation pipeline on multiple protein files.
    
    Args:
        protein_files: List of protein FASTA files
        output_dir: Directory for PFAM annotation results
        threads: Threads per astra process
        
    Returns:
        Path to results JSON file
    """
    console.print(f"[green]Running PFAM annotation on {len(protein_files)} genomes...[/green]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize annotator
    annotator = PFAMAnnotator(threads=threads)
    
    # Check astra installation
    if not annotator.check_astra_installation():
        raise RuntimeError("Astra is not installed or not in PATH. "
                          "Please install astra and ensure it's accessible.")
    
    # Annotate each genome
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Annotating genomes...", total=len(protein_files))
        
        for protein_file in protein_files:
            genome_name = protein_file.stem
            progress.update(task, description=f"Processing {genome_name}")
            
            try:
                result = annotator.annotate_genome(protein_file, output_dir)
                results[genome_name] = result
                
                if result["success"]:
                    console.print(f"✓ {genome_name}: {result['num_hits']} domains found")
                else:
                    console.print(f"✗ {genome_name}: {result.get('error_message', 'Unknown error')}")
                    
            except Exception as e:
                console.print(f"✗ {genome_name}: Unexpected error: {e}")
                results[genome_name] = {
                    "genome": genome_name,
                    "success": False,
                    "error_message": str(e)
                }
            
            progress.advance(task)
    
    # Save results summary
    results_file = output_dir / "pfam_annotation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results.values() if r["success"])
    total_hits = sum(r.get("num_hits", 0) for r in results.values() if r["success"])
    
    console.print(f"[green]PFAM annotation completed![/green]")
    console.print(f"  • {successful}/{len(protein_files)} genomes annotated successfully")
    console.print(f"  • {total_hits:,} total domain hits found")
    console.print(f"  • Results saved to: {results_file}")
    
    return results_file