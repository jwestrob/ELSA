"""
Gene calling and protein translation pipeline for ELSA.

Converts nucleotide FASTA files to protein sequences via Prodigal gene calling.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .params import IngestConfig
from .embeddings import ProteinSequence

console = Console()


@dataclass
class Gene:
    """A called gene with genomic coordinates."""
    sample_id: str
    contig_id: str
    gene_id: str
    start: int  # 1-based
    end: int    # 1-based
    strand: int  # +1 or -1
    partial: bool
    sequence: str  # amino acid sequence
    
    @property
    def length(self) -> int:
        return len(self.sequence)


class ProdigalRunner:
    """Wrapper for Prodigal gene calling."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self._check_prodigal()
    
    def _check_prodigal(self):
        """Check if Prodigal is available."""
        try:
            result = subprocess.run(
                ["prodigal", "-v"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            console.print(f"✓ Found Prodigal: {result.stderr.strip().split()[1]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Prodigal not found. Install with: conda install -c bioconda prodigal"
            )
    
    def call_genes(self, fasta_path: Path, sample_id: str) -> List[Gene]:
        """Call genes on a FASTA file using Prodigal."""
        console.print(f"Calling genes for {sample_id}...")
        
        # Create output files in the same directory as input FASTA
        output_dir = fasta_path.parent
        gff_path = output_dir / f"{fasta_path.stem}.gff"
        faa_path = output_dir / f"{fasta_path.stem}.faa"
        
        # Skip if output files already exist (resumption support)
        if gff_path.exists() and faa_path.exists():
            console.print(f"Using existing Prodigal output: {gff_path.name}, {faa_path.name}")
            genes = self._parse_prodigal_output(gff_path, faa_path, sample_id)
            console.print(f"✓ Loaded {len(genes)} genes from existing files")
            return genes
        
        # Build Prodigal command
        cmd = [
            "prodigal",
            "-i", str(fasta_path),
            "-o", str(gff_path),
            "-a", str(faa_path),
            "-f", "gff",
            "-p", self.config.prodigal_mode
        ]
        
        if self.config.gene_caller == "metaprodigal":
            cmd.extend(["-p", "meta"])
        
        # Run Prodigal
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse results
            genes = self._parse_prodigal_output(gff_path, faa_path, sample_id)
            
            console.print(f"✓ Called {len(genes)} genes → {gff_path.name}, {faa_path.name}")
            return genes
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Prodigal failed: {e.stderr}[/red]")
            raise
    
    def _parse_prodigal_output(self, gff_path: Path, faa_path: Path, sample_id: str) -> List[Gene]:
        """Parse Prodigal GFF and FAA output."""
        genes = []
        
        # Read protein sequences
        proteins = {record.id: str(record.seq) for record in SeqIO.parse(faa_path, "fasta")}
        
        # Parse GFF annotations
        with open(gff_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                    
                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[2] != "CDS":
                    continue
                
                contig_id = parts[0]
                start = int(parts[3])
                end = int(parts[4])
                strand = 1 if parts[6] == "+" else -1
                attributes = parts[8]
                
                # Parse gene ID from attributes
                gene_id = None
                partial = False
                for attr in attributes.split(";"):
                    if attr.startswith("ID="):
                        gene_id = attr[3:]
                    elif "partial=01" in attr or "partial=10" in attr or "partial=11" in attr:
                        partial = True
                
                if not gene_id:
                    continue
                
                # Find matching protein sequence - try different ID formats
                protein_seq = None
                possible_ids = [
                    gene_id,  # Direct match: "1_1"
                    f"{contig_id}_{gene_id}",  # With contig: "accn|1313.30775.con.0001_1_1"  
                    f"{contig_id}_{gene_id.split('_')[-1]}"  # Contig + gene number: "accn|1313.30775.con.0001_1"
                ]
                
                for possible_id in possible_ids:
                    if possible_id in proteins:
                        protein_seq = proteins[possible_id]
                        break
                
                if not protein_seq:
                    continue
                
                # Clean protein sequence (remove stop codons and invalid characters)
                clean_seq = protein_seq.replace('*', '').replace('X', '').upper()
                
                # Filter by length and partial status
                if len(clean_seq) < self.config.min_cds_aa:
                    continue
                if partial and not self.config.keep_partial:
                    continue
                
                gene = Gene(
                    sample_id=sample_id,
                    contig_id=contig_id,
                    gene_id=f"{sample_id}_{gene_id}",
                    start=start,
                    end=end,
                    strand=strand,
                    partial=partial,
                    sequence=clean_seq
                )
                genes.append(gene)
        
        return genes


class GFFParser:
    """Parser for existing GFF files with CDS annotations."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
    
    def parse_gff_with_fasta(self, gff_path: Path, fasta_path: Path, sample_id: str) -> List[Gene]:
        """Parse GFF file and translate CDS features from nucleotide FASTA."""
        console.print(f"Parsing GFF annotations for {sample_id}...")
        
        # Load nucleotide sequences
        contigs = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
        
        genes = []
        
        with open(gff_path) as f:
            for line_no, line in enumerate(f, 1):
                if line.startswith("#") or not line.strip():
                    continue
                    
                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[2] != "CDS":
                    continue
                
                try:
                    contig_id = parts[0]
                    start = int(parts[3])  # 1-based
                    end = int(parts[4])    # 1-based
                    strand = 1 if parts[6] == "+" else -1
                    attributes = parts[8]
                    
                    if contig_id not in contigs:
                        console.print(f"[yellow]Warning: Contig {contig_id} not found in FASTA[/yellow]")
                        continue
                    
                    # Extract CDS sequence and translate
                    contig_seq = contigs[contig_id]
                    
                    if strand == 1:
                        cds_seq = contig_seq[start-1:end]  # Convert to 0-based
                    else:
                        cds_seq = str(Seq(contig_seq[start-1:end]).reverse_complement())
                    
                    # Translate to protein
                    try:
                        protein_seq = str(Seq(cds_seq).translate(table=11, to_stop=True))
                        
                        # Remove stop codons and check length
                        if protein_seq.endswith("*"):
                            protein_seq = protein_seq[:-1]
                        
                        if len(protein_seq) < self.config.min_cds_aa:
                            continue
                            
                    except Exception as e:
                        console.print(f"[yellow]Translation error at line {line_no}: {e}[/yellow]")
                        continue
                    
                    # Generate gene ID
                    gene_id = f"{sample_id}_CDS_{len(genes)+1}"
                    
                    gene = Gene(
                        sample_id=sample_id,
                        contig_id=contig_id,
                        gene_id=gene_id,
                        start=start,
                        end=end,
                        strand=strand,
                        partial=False,  # Assume complete from GFF
                        sequence=protein_seq
                    )
                    genes.append(gene)
                    
                except (ValueError, IndexError) as e:
                    console.print(f"[yellow]Skipping malformed GFF line {line_no}: {e}[/yellow]")
                    continue
        
        console.print(f"✓ Parsed {len(genes)} genes from GFF")
        return genes


class ProteinIngester:
    """Main protein ingestion pipeline."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.prodigal = ProdigalRunner(config) if config.gene_caller != "none" else None
        self.gff_parser = GFFParser(config)
    
    def ingest_sample(self, fasta_path: Path, sample_id: str, 
                     gff_path: Optional[Path] = None, 
                     aa_fasta_path: Optional[Path] = None) -> List[ProteinSequence]:
        """Ingest a sample and return protein sequences."""
        
        # Option 1: Use provided AA FASTA directly
        if aa_fasta_path and aa_fasta_path.exists():
            console.print(f"Using provided protein FASTA: {aa_fasta_path}")
            return self._parse_protein_fasta(aa_fasta_path, sample_id)
        
        # Option 2: Use GFF + nucleotide FASTA
        elif gff_path and gff_path.exists():
            genes = self.gff_parser.parse_gff_with_fasta(gff_path, fasta_path, sample_id)
            
        # Option 3: Gene calling with Prodigal
        elif self.prodigal:
            genes = self.prodigal.call_genes(fasta_path, sample_id)
            
        else:
            raise ValueError("No gene calling method available. Provide GFF, AA FASTA, or enable Prodigal.")
        
        # Convert genes to ProteinSequences
        proteins = []
        for gene in genes:
            protein = ProteinSequence(
                sample_id=gene.sample_id,
                contig_id=gene.contig_id,
                gene_id=gene.gene_id,
                start=gene.start,
                end=gene.end,
                strand=gene.strand,
                sequence=gene.sequence
            )
            proteins.append(protein)
        
        return proteins
    
    def _parse_protein_fasta(self, aa_fasta_path: Path, sample_id: str) -> List[ProteinSequence]:
        """Parse existing protein FASTA file."""
        proteins = []
        
        for i, record in enumerate(SeqIO.parse(aa_fasta_path, "fasta")):
            # Clean protein sequence (remove stop codons and invalid characters)
            clean_seq = str(record.seq).replace('*', '').replace('X', '').upper()
            
            if len(clean_seq) < self.config.min_cds_aa:
                continue
                
            protein = ProteinSequence(
                sample_id=sample_id,
                contig_id=record.id,  # Use record ID as contig
                gene_id=f"{sample_id}_{record.id}",
                start=0,  # Unknown
                end=len(clean_seq),
                strand=1,  # Unknown
                sequence=clean_seq
            )
            proteins.append(protein)
        
        console.print(f"✓ Loaded {len(proteins)} proteins from FASTA")
        return proteins
    
    def ingest_multiple(self, sample_data: List[Tuple[str, Path, Optional[Path], Optional[Path]]]) -> Dict[str, List[ProteinSequence]]:
        """Ingest multiple samples."""
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing samples...", total=len(sample_data))
            
            for sample_id, fasta_path, gff_path, aa_fasta_path in sample_data:
                progress.update(task, description=f"Processing {sample_id}")
                
                proteins = self.ingest_sample(fasta_path, sample_id, gff_path, aa_fasta_path)
                results[sample_id] = proteins
                
                progress.advance(task)
        
        total_proteins = sum(len(proteins) for proteins in results.values())
        console.print(f"✓ Processed {len(sample_data)} samples, {total_proteins:,} total proteins")
        
        return results


if __name__ == "__main__":
    # Test ingestion functionality
    from .params import ELSAConfig
    
    config = ELSAConfig()
    ingester = ProteinIngester(config.ingest)
    
    print(f"Gene caller: {config.ingest.gene_caller}")
    print(f"Min CDS length: {config.ingest.min_cds_aa} AA")