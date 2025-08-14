#!/usr/bin/env python3
"""
PFAM domain annotation integration for ELSA build pipeline.
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


class PfamAnnotationError(Exception):
    """Exception raised during PFAM annotation."""
    pass


class PfamAnnotator:
    """PFAM domain annotation using astra command-line tool."""
    
    def __init__(self, threads: int = 8):
        self.threads = threads
        # Using --cut_ga flag instead of e-value threshold for better quality
        
    def check_astra_installation(self) -> bool:
        """Check if astra is installed and available."""
        try:
            result = subprocess.run(['astra', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def run_astra_scan(self, protein_file: Path, output_dir: Path, 
                       timeout: int = 1800) -> Dict:
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
        sample_name = protein_file.stem
        
        # Create sample-specific output directory
        sample_output_dir = output_dir / f"{sample_name}_pfam"
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Astra command (matches genome browser format)
        cmd = [
            'astra', 'search',
            '--prot_in', str(protein_file.parent),  # Directory containing protein file
            '--installed_hmms', 'PFAM',
            '--outdir', str(sample_output_dir),
            '--threads', str(self.threads),
            '--cut_ga'  # Use gathering cutoffs (more stringent than e-value)
        ]
        
        log_file = sample_output_dir / "astra_search_log.txt"
        
        try:
            logger.info(f"Running astra scan for {sample_name}...")
            print(f"🔧 Astra command: {' '.join(cmd)}")  # Print to terminal for debugging
            logger.debug(f"Command: {' '.join(cmd)}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    text=True
                )
            
            runtime = time.time() - start_time
            
            if result.returncode != 0:
                with open(log_file, 'r') as f:
                    error_log = f.read()
                raise PfamAnnotationError(
                    f"Astra failed for {sample_name} (exit {result.returncode}): {error_log[-500:]}"
                )
            
            # Check for output files
            hits_file = sample_output_dir / "PFAM_hits_df.tsv"
            if not hits_file.exists():
                raise PfamAnnotationError(f"Expected output file not found: {hits_file}")
            
            # Read and validate results
            try:
                hits_df = pd.read_csv(hits_file, sep='\t')
                n_hits = len(hits_df)
            except Exception as e:
                raise PfamAnnotationError(f"Could not read PFAM hits file: {e}")
            
            return {
                'sample_name': sample_name,
                'success': True,
                'runtime_seconds': runtime,
                'n_hits': n_hits,
                'output_dir': str(sample_output_dir),
                'hits_file': str(hits_file)
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Astra scan timeout for {sample_name} after {timeout}s")
            return {
                'sample_name': sample_name,
                'success': False,
                'error': 'timeout',
                'runtime_seconds': timeout
            }
        except Exception as e:
            logger.error(f"Astra scan failed for {sample_name}: {e}")
            return {
                'sample_name': sample_name,
                'success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
    
    def annotate_proteins(self, protein_files: List[Path], output_dir: Path,
                         parallel: bool = True) -> Dict:
        """
        Run PFAM annotation on multiple protein files.
        
        Args:
            protein_files: List of protein FASTA files
            output_dir: Base output directory for annotations
            parallel: Whether to run scans in parallel
            
        Returns:
            Dict with aggregated results and statistics
        """
        if not self.check_astra_installation():
            raise PfamAnnotationError(
                "astra not found in PATH. Please install astra for PFAM annotation."
            )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        results = []
        
        if parallel and len(protein_files) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(4, len(protein_files))) as executor:
                future_to_file = {
                    executor.submit(self.run_astra_scan, pf, output_dir): pf 
                    for pf in protein_files
                }
                
                for future in as_completed(future_to_file):
                    protein_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            logger.info(f"✓ {result['sample_name']}: {result['n_hits']} hits "
                                      f"({result['runtime_seconds']:.1f}s)")
                        else:
                            logger.warning(f"✗ {result['sample_name']}: {result.get('error', 'failed')}")
                            
                    except Exception as e:
                        logger.error(f"Exception processing {protein_file}: {e}")
                        results.append({
                            'sample_name': protein_file.stem,
                            'success': False,
                            'error': str(e),
                            'runtime_seconds': 0
                        })
        else:
            # Sequential execution
            for protein_file in protein_files:
                result = self.run_astra_scan(protein_file, output_dir)
                results.append(result)
                
                if result['success']:
                    logger.info(f"✓ {result['sample_name']}: {result['n_hits']} hits "
                              f"({result['runtime_seconds']:.1f}s)")
                else:
                    logger.warning(f"✗ {result['sample_name']}: {result.get('error', 'failed')}")
        
        # Aggregate results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        total_hits = sum(r.get('n_hits', 0) for r in successful_results)
        total_runtime = time.time() - start_time
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(protein_files),
            'successful_samples': len(successful_results),
            'failed_samples': len(failed_results),
            'total_hits': total_hits,
            'total_runtime_seconds': total_runtime,
            'cutoff_method': 'ga_cutoffs',  # Using PFAM gathering cutoffs
            'parallel_execution': parallel,
            'results': results
        }
        
        summary_file = output_dir / "pfam_annotation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"PFAM annotation complete: {len(successful_results)}/{len(protein_files)} samples, "
                   f"{total_hits:,} hits, {total_runtime:.1f}s total")
        
        if failed_results:
            logger.warning(f"Failed samples: {[r['sample_name'] for r in failed_results]}")
        
        return summary
    
    def annotate_proteins_single_command(self, protein_dir: Path, output_dir: Path) -> Dict:
        """
        Run PFAM annotation on entire protein directory with single Astra command.
        
        Args:
            protein_dir: Directory containing all protein FASTA files
            output_dir: Output directory for annotations
            
        Returns:
            Dict with aggregated results and statistics
        """
        if not self.check_astra_installation():
            raise PfamAnnotationError(
                "astra not found in PATH. Please install astra for PFAM annotation."
            )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Single Astra command for all proteins
        cmd = [
            'astra', 'search',
            '--prot_in', str(protein_dir),
            '--installed_hmms', 'PFAM',
            '--outdir', str(output_dir),
            '--threads', str(self.threads),
            '--cut_ga'  # Use gathering cutoffs
        ]
        
        log_file = output_dir / "astra_search_log.txt"
        
        try:
            logger.info(f"Running single astra command for all proteins...")
            print(f"🔧 Astra command: {' '.join(cmd)}")  # Print to terminal for debugging
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    timeout=1800,  # 30 minutes timeout
                    text=True
                )
            
            runtime = time.time() - start_time
            
            if result.returncode != 0:
                with open(log_file, 'r') as f:
                    error_log = f.read()
                raise PfamAnnotationError(
                    f"Astra failed (exit {result.returncode}): {error_log[-500:]}"
                )
            
            # Check for output file - Astra should create PFAM_hits_df.tsv
            hits_file = output_dir / "PFAM_hits_df.tsv"
            if not hits_file.exists():
                raise PfamAnnotationError(f"Expected output file not found: {hits_file}")
            
            # Parse the combined results
            try:
                hits_df = pd.read_csv(hits_file, sep='\t')
                total_hits = len(hits_df)
                
                # Count unique samples from sequence IDs
                if 'sequence_id' in hits_df.columns:
                    # Extract sample names from sequence IDs (format: accn|SAMPLE_NAME.con.XXXX_GENE)
                    hits_df['sample_name'] = hits_df['sequence_id'].str.extract(r'accn\|([^.]+)')
                    unique_samples = hits_df['sample_name'].nunique()
                    unique_proteins = hits_df['sequence_id'].nunique() 
                    unique_domains = hits_df['hmm_name'].nunique()
                else:
                    unique_samples = 1  # Fallback
                    unique_proteins = 0
                    unique_domains = 0
                
                logger.info(f"✓ PFAM annotation complete: {total_hits:,} hits across {unique_samples} samples")
                logger.info(f"  Unique proteins: {unique_proteins:,}")
                logger.info(f"  Unique domains: {unique_domains:,}")
                
            except Exception as e:
                raise PfamAnnotationError(f"Could not parse PFAM results: {e}")
            
            # Create summary in expected format
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': unique_samples,
                'successful_samples': unique_samples, 
                'failed_samples': 0,
                'total_hits': total_hits,
                'unique_proteins': unique_proteins,
                'unique_domains': unique_domains,
                'total_runtime_seconds': runtime,
                'cutoff_method': 'ga_cutoffs',  # Using PFAM gathering cutoffs
                'parallel_execution': False,  # Single command
                'hits_file': str(hits_file)
            }
            
            return summary
            
        except subprocess.TimeoutExpired:
            logger.error(f"Astra command timeout after 1800s")
            raise PfamAnnotationError("Astra annotation timed out")
        except Exception as e:
            logger.error(f"Astra annotation failed: {e}")
            raise PfamAnnotationError(f"Astra annotation failed: {e}")
    
    def load_pfam_annotations(self, annotation_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load PFAM annotations from annotation directory.
        
        Args:
            annotation_dir: Directory containing PFAM annotation results
            
        Returns:
            Dict mapping sample_name -> PFAM hits DataFrame
        """
        annotation_dir = Path(annotation_dir)
        annotations = {}
        
        for sample_dir in annotation_dir.glob("*_pfam"):
            sample_name = sample_dir.name.replace("_pfam", "")
            hits_file = sample_dir / "PFAM_hits_df.tsv"
            
            if hits_file.exists():
                try:
                    df = pd.read_csv(hits_file, sep='\t')
                    annotations[sample_name] = df
                    logger.debug(f"Loaded {len(df)} PFAM hits for {sample_name}")
                except Exception as e:
                    logger.warning(f"Could not load PFAM hits for {sample_name}: {e}")
            else:
                logger.warning(f"PFAM hits file not found: {hits_file}")
        
        logger.info(f"Loaded PFAM annotations for {len(annotations)} samples")
        return annotations


def run_pfam_annotation_pipeline(config, manifest, work_dir: Path, 
                                threads: int = 8, force_annotation: bool = False) -> Optional[Dict]:
    """
    Run PFAM annotation pipeline similar to genome browser setup.
    
    Args:
        config: ELSA configuration object
        manifest: ELSA manifest object  
        work_dir: Working directory
        threads: Number of threads for annotation
        
    Returns:
        PFAM annotation summary dict, or None if skipped/failed
    """
    pfam_dir = work_dir / "pfam_annotations"
    protein_dir = Path("test_data/proteins")
    
    # Handle 'auto' threads value
    if threads == 'auto':
        import os
        threads = os.cpu_count() or 8
    
    # Check if PFAM annotations need to be run
    summary_file = pfam_dir / "pfam_annotation_results.json"
    
    # First check if we already have completed annotations (unless forced)
    if summary_file.exists() and not force_annotation:
        logger.info("PFAM annotations already exist, loading summary")
        try:
            with open(summary_file) as f:
                return json.load(f)
        except:
            logger.warning("Could not load existing PFAM summary, will re-run")
    elif force_annotation:
        logger.info("Force annotation flag set, will re-run PFAM annotation")
    
    # Check if we can use existing genome browser annotations (but only if not forced)
    existing_pfam_dir = Path("genome_browser/pfam_annotations")
    if existing_pfam_dir.exists() and not summary_file.exists() and not force_annotation:
        logger.info("Using existing PFAM annotations from genome_browser/")
        # Copy to expected location
        pfam_dir.parent.mkdir(parents=True, exist_ok=True)
        if not pfam_dir.exists():
            import shutil
            shutil.copytree(existing_pfam_dir, pfam_dir)
        
        # Create summary from existing data
        annotator = PfamAnnotator()
        annotations = annotator.load_pfam_annotations(pfam_dir)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(annotations),
            'successful_samples': len(annotations),
            'failed_samples': 0,
            'total_hits': sum(len(df) for df in annotations.values()) if annotations else 0,
            'total_runtime_seconds': 0,
            'cutoff_method': 'ga_cutoffs',  # Using PFAM gathering cutoffs
            'parallel_execution': False,
            'results': [{'sample_name': name, 'success': True, 'n_hits': len(df)} 
                       for name, df in annotations.items()]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    # If forced, clean up existing results
    if force_annotation and pfam_dir.exists():
        logger.info("Removing existing PFAM annotations for fresh run")
        import shutil
        shutil.rmtree(pfam_dir)
    
    # Find protein FASTA files in test_data/proteins/ 
    protein_dir = Path("test_data/proteins")
    if not protein_dir.exists():
        logger.warning("No test_data/proteins directory found, skipping PFAM annotation")
        return None
    
    protein_files = list(protein_dir.glob("*.faa"))
    if not protein_files:
        logger.warning("No protein FASTA files found in test_data/proteins/, skipping PFAM annotation")
        return None
    
    logger.info(f"Found {len(protein_files)} protein files for PFAM annotation")
    
    # Run PFAM annotation using ELSA's own PfamAnnotator with GA cutoffs
    try:
        # Handle 'auto' threads value
        if threads == 'auto':
            import os
            threads = os.cpu_count() or 8
        
        # Use ELSA's own PFAM annotator with GA cutoffs
        logger.info("Using ELSA's PfamAnnotator with GA cutoffs")
        annotator = PfamAnnotator(threads=threads)
        
        if not annotator.check_astra_installation():
            logger.warning("Astra not available, skipping PFAM annotation")
            return None
        
        # Run single PFAM annotation command on entire proteins directory
        logger.info(f"Running PFAM annotation with GA cutoffs on {len(protein_files)} files...")
        logger.info(f"This may take several minutes...")
        
        summary = annotator.annotate_proteins_single_command(protein_dir, pfam_dir)
        
        # Save summary with the expected filename
        summary_file = pfam_dir / "pfam_annotation_results.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ ELSA PFAM annotation completed: {summary['successful_samples']}/{summary['total_samples']} samples")
        logger.info(f"  Total PFAM domains found: {summary['total_hits']:,}")
        logger.info(f"  Using GA cutoffs for high-quality annotations")
        
        return summary
        
    except Exception as e:
        logger.error(f"PFAM annotation failed: {e}")
        # Don't fail the entire build pipeline for annotation issues
        return None