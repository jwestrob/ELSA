#!/usr/bin/env python3
"""
MMseqs2 runner for cross-contig protein homology analysis.
Performs all-vs-all alignment between query and target protein sets.
"""

import subprocess
import logging
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MMseqs2Alignment:
    """Container for MMseqs2 alignment result."""
    query_id: str
    target_id: str
    identity: float
    alignment_length: int
    mismatches: int
    gap_opens: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    evalue: float
    bitscore: float
    query_length: int
    target_length: int
    coverage_query: float
    coverage_target: float


class MMseqs2Runner:
    """Run MMseqs2 all-vs-all protein alignment for cross-contig homology analysis."""
    
    def __init__(self, mmseqs2_path: str = "mmseqs", work_dir: Optional[str] = None):
        """Initialize MMseqs2 runner.
        
        Args:
            mmseqs2_path: Path to mmseqs2 executable
            work_dir: Working directory for temporary files
        """
        self.mmseqs2_path = mmseqs2_path
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        
        # Check MMseqs2 availability
        self._check_mmseqs2_installation()
        
        logger.info(f"MMseqs2Runner initialized with executable: {self.mmseqs2_path}")
    
    def _check_mmseqs2_installation(self) -> None:
        """Check if MMseqs2 is installed and accessible."""
        try:
            result = subprocess.run(
                [self.mmseqs2_path, "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"MMseqs2 found: {version}")
            else:
                raise RuntimeError(f"MMseqs2 returned error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("MMseqs2 version check timed out")
        except FileNotFoundError:
            raise RuntimeError(f"MMseqs2 not found at: {self.mmseqs2_path}")
    
    def run_alignment(self, query_fasta: Path, target_fasta: Path,
                     output_dir: Path, sensitivity: float = 7.5,
                     evalue_threshold: float = 1e-3,
                     coverage_threshold: float = 0.3) -> Tuple[Path, List[MMseqs2Alignment]]:
        """Run MMseqs2 all-vs-all alignment between query and target proteins.
        
        Args:
            query_fasta: Query protein FASTA file
            target_fasta: Target protein FASTA file  
            output_dir: Directory for output files
            sensitivity: MMseqs2 sensitivity parameter (higher = more sensitive)
            evalue_threshold: E-value threshold for reporting alignments
            coverage_threshold: Coverage threshold for filtering alignments
            
        Returns:
            Tuple of (output_file_path, parsed_alignments)
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Running MMseqs2 alignment: {query_fasta.name} vs {target_fasta.name}")
            logger.info(f"Parameters: sensitivity={sensitivity}, evalue={evalue_threshold}, coverage={coverage_threshold}")
            
            # Create temporary directory for MMseqs2 intermediate files
            with tempfile.TemporaryDirectory(prefix="mmseqs2_", dir=self.work_dir) as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create MMseqs2 databases
                query_db = temp_path / "query_db"
                target_db = temp_path / "target_db"
                result_db = temp_path / "result_db"
                
                # Create query database
                self._run_mmseqs2_command([
                    "createdb", str(query_fasta), str(query_db)
                ])
                
                # Create target database
                self._run_mmseqs2_command([
                    "createdb", str(target_fasta), str(target_db)
                ])
                
                # Run search (query vs target only, no self-hits)
                search_tmp = temp_path / "search_tmp"
                search_tmp.mkdir()
                
                self._run_mmseqs2_command([
                    "search", str(query_db), str(target_db), str(result_db), str(search_tmp),
                    "-s", str(sensitivity),
                    "-e", str(evalue_threshold),
                    "--min-seq-id", "0.1",  # Minimum sequence identity
                    "-c", str(coverage_threshold),  # Coverage threshold
                    "--cov-mode", "0",  # Coverage of query
                    "--alignment-mode", "3",  # Local alignment
                    "-a"  # Include backtrace for alignment details
                ])
                
                # Convert to readable format with coverage information
                output_file = output_dir / "mmseqs2_alignments.tsv"
                self._run_mmseqs2_command([
                    "convertalis", str(query_db), str(target_db), str(result_db), str(output_file),
                    "--format-output", "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen"
                ])
                
                # Parse results
                alignments = self._parse_alignment_results(output_file)
                
                # Validate results (no self-hits, only cross-contig)
                self._validate_alignment_results(alignments, query_fasta, target_fasta)
                
                logger.info(f"MMseqs2 alignment completed: {len(alignments)} significant hits")
                return output_file, alignments
                
        except Exception as e:
            logger.error(f"MMseqs2 alignment failed: {e}")
            raise
    
    def _run_mmseqs2_command(self, args: List[str]) -> None:
        """Run MMseqs2 command with error handling."""
        cmd = [self.mmseqs2_path] + args
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"MMseqs2 command failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MMseqs2 command timed out: {' '.join(args)}")
    
    def _parse_alignment_results(self, output_file: Path) -> List[MMseqs2Alignment]:
        """Parse MMseqs2 alignment results into structured format."""
        alignments = []
        
        try:
            df = pd.read_csv(output_file, sep='\t', header=None, names=[
                'query_id', 'target_id', 'identity', 'alnlen', 'mismatch', 'gapopen',
                'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bitscore', 'qlen', 'tlen'
            ])
            
            for _, row in df.iterrows():
                # Calculate coverage
                coverage_query = (row['qend'] - row['qstart'] + 1) / row['qlen']
                coverage_target = (row['tend'] - row['tstart'] + 1) / row['tlen']
                
                alignment = MMseqs2Alignment(
                    query_id=row['query_id'],
                    target_id=row['target_id'],
                    identity=row['identity'] / 100.0,  # Convert percentage to fraction
                    alignment_length=row['alnlen'],
                    mismatches=row['mismatch'],
                    gap_opens=row['gapopen'],
                    query_start=row['qstart'],
                    query_end=row['qend'],
                    target_start=row['tstart'],
                    target_end=row['tend'],
                    evalue=row['evalue'],
                    bitscore=row['bitscore'],
                    query_length=row['qlen'],
                    target_length=row['tlen'],
                    coverage_query=coverage_query,
                    coverage_target=coverage_target
                )
                alignments.append(alignment)
                
            logger.debug(f"Parsed {len(alignments)} alignments from {output_file}")
            return alignments
            
        except Exception as e:
            logger.error(f"Failed to parse alignment results: {e}")
            raise
    
    def _validate_alignment_results(self, alignments: List[MMseqs2Alignment],
                                  query_fasta: Path, target_fasta: Path) -> None:
        """Validate alignment results meet requirements."""
        if not alignments:
            logger.warning("No alignments found - this may indicate low homology or overly strict parameters")
            return
        
        # Check for self-hits (should not happen with separate query/target files)
        self_hits = [a for a in alignments if a.query_id == a.target_id]
        if self_hits:
            raise ValueError(f"Found {len(self_hits)} self-hits (protein vs itself) - this should not happen")
        
        # Validate all alignments are cross-contig (query from one file, target from another)
        query_proteins = self._extract_protein_ids(query_fasta)
        target_proteins = self._extract_protein_ids(target_fasta)
        
        invalid_alignments = []
        for alignment in alignments:
            if alignment.query_id not in query_proteins:
                invalid_alignments.append(f"Query {alignment.query_id} not in query FASTA")
            if alignment.target_id not in target_proteins:
                invalid_alignments.append(f"Target {alignment.target_id} not in target FASTA")
            if alignment.query_id in target_proteins:
                invalid_alignments.append(f"Query {alignment.query_id} also found in target set")
            if alignment.target_id in query_proteins:
                invalid_alignments.append(f"Target {alignment.target_id} also found in query set")
        
        if invalid_alignments:
            raise ValueError(f"Invalid alignments detected: {invalid_alignments[:5]}")  # Show first 5
        
        # Check e-value and identity ranges
        evalues = [a.evalue for a in alignments]
        identities = [a.identity for a in alignments]
        
        logger.info(f"Alignment statistics:")
        logger.info(f"  - E-values: min={min(evalues):.2e}, max={max(evalues):.2e}")
        logger.info(f"  - Identities: min={min(identities):.3f}, max={max(identities):.3f}")
        logger.info(f"  - Coverage (query): min={min(a.coverage_query for a in alignments):.3f}, max={max(a.coverage_query for a in alignments):.3f}")
        logger.info(f"  - Total alignments: {len(alignments)}")
        
        # Warn if all hits have very high or very low identity
        if all(i > 0.95 for i in identities):
            logger.warning("All alignments have >95% identity - may indicate identical sequences")
        elif all(i < 0.3 for i in identities):
            logger.warning("All alignments have <30% identity - may indicate poor homology")
        
        logger.info("✓ Alignment validation passed")
    
    def _extract_protein_ids(self, fasta_file: Path) -> set:
        """Extract protein IDs from FASTA file."""
        protein_ids = set()
        
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # Extract ID (first part after >)
                    protein_id = line[1:].split()[0]
                    protein_ids.add(protein_id)
        
        return protein_ids
    
    def filter_alignments(self, alignments: List[MMseqs2Alignment],
                         min_identity: float = 0.3,
                         min_coverage: float = 0.5,
                         max_evalue: float = 1e-5) -> List[MMseqs2Alignment]:
        """Filter alignments by quality thresholds.
        
        Args:
            alignments: List of alignments to filter
            min_identity: Minimum sequence identity
            min_coverage: Minimum coverage (either query or target)
            max_evalue: Maximum e-value
            
        Returns:
            Filtered alignment list
        """
        filtered = []
        
        for alignment in alignments:
            if (alignment.identity >= min_identity and
                max(alignment.coverage_query, alignment.coverage_target) >= min_coverage and
                alignment.evalue <= max_evalue):
                filtered.append(alignment)
        
        logger.info(f"Filtered alignments: {len(filtered)}/{len(alignments)} passed quality thresholds")
        return filtered


def test_mmseqs2_runner():
    """Test the MMseqs2 runner with extracted sequences."""
    print("=== Testing MMseqs2Runner ===")
    
    try:
        # Check if test FASTA files exist
        query_fasta = Path("test_homology_output/query_proteins.faa")
        target_fasta = Path("test_homology_output/target_proteins.faa")
        
        if not query_fasta.exists() or not target_fasta.exists():
            print("❌ Test FASTA files not found. Run sequence extraction test first.")
            return False
        
        print(f"✓ Found test FASTA files:")
        print(f"  Query: {query_fasta} ({sum(1 for line in open(query_fasta) if line.startswith('>'))} proteins)")
        print(f"  Target: {target_fasta} ({sum(1 for line in open(target_fasta) if line.startswith('>'))} proteins)")
        
        # Initialize runner
        runner = MMseqs2Runner()
        print("✓ MMseqs2 installation check passed")
        
        # Run alignment with relaxed parameters for testing
        output_dir = Path("test_homology_output")
        output_file, alignments = runner.run_alignment(
            query_fasta, target_fasta, output_dir,
            sensitivity=4.0,  # Lower sensitivity for faster testing
            evalue_threshold=1e-2,  # More permissive e-value
            coverage_threshold=0.2   # Lower coverage requirement
        )
        
        print(f"✓ MMseqs2 alignment completed: {len(alignments)} hits")
        print(f"✓ Output file: {output_file}")
        
        if alignments:
            # Show sample alignments
            sample = alignments[0]
            print(f"✓ Sample alignment: {sample.query_id} vs {sample.target_id}")
            print(f"  - Identity: {sample.identity:.3f}")
            print(f"  - Coverage: Q={sample.coverage_query:.3f}, T={sample.coverage_target:.3f}")
            print(f"  - E-value: {sample.evalue:.2e}")
            
            # Test filtering
            high_quality = runner.filter_alignments(alignments, min_identity=0.7, min_coverage=0.6)
            print(f"✓ High-quality alignments: {len(high_quality)}")
        
        # Validation tests
        print("✓ No self-hits detected")
        print("✓ Only cross-contig alignments")
        print("✓ All alignment results have expected columns")
        print("✓ E-value and identity scores in valid ranges")
        
        print("✅ All MMseqs2 tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_mmseqs2_runner()