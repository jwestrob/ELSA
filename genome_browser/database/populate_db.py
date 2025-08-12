#!/usr/bin/env python3
"""
Data ingestion pipeline for ELSA genome browser.
Populates SQLite database with genomes, genes, syntenic blocks, and PFAM annotations.
"""

import sqlite3
import pandas as pd
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio import SeqIO
try:
    from Bio.SeqUtils import GC
except ImportError:
    # For newer BioPython versions
    from Bio.SeqUtils import gc_fraction
    def GC(seq):
        return gc_fraction(seq) * 100
import argparse

logger = logging.getLogger(__name__)

class ELSADataIngester:
    """Ingest ELSA analysis data into SQLite database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def parse_gene_id(self, gene_id: str) -> Tuple[str, str, str]:
        """
        Parse gene ID to extract genome, contig, and gene components.
        
        Example: "accn|1313.30775.con.0001_1" -> ("1313.30775", "accn|1313.30775.con.0001", "1")
        """
        # Pattern for typical gene IDs from Prodigal
        match = re.match(r'(.+?)_(\d+)$', gene_id)
        if match:
            contig_part = match.group(1)
            gene_num = match.group(2)
            
            # Extract genome ID (look for pattern like "1313.30775")
            genome_match = re.search(r'(\w+\.\d+)', contig_part)
            if genome_match:
                genome_id = genome_match.group(1)
            else:
                # Fallback: use first part before first separator
                genome_id = contig_part.split('|')[0].split('.')[0]
            
            return genome_id, contig_part, gene_num
        
        # Fallback parsing
        parts = gene_id.split('|')
        if len(parts) >= 2:
            return parts[0], parts[1], parts[-1]
        
        return gene_id, gene_id, "1"
    
    def parse_locus_id(self, locus_id: str) -> Tuple[str, str]:
        """
        Parse locus ID to extract genome and contig.
        
        Example: "1313.30775:accn|1313.30775.con.0001" -> ("1313.30775", "accn|1313.30775.con.0001")
        """
        if ':' in locus_id:
            genome_id, contig_id = locus_id.split(':', 1)
            return genome_id, contig_id
        
        # Fallback: treat entire string as contig, extract genome from pattern
        genome_match = re.search(r'(\w+\.\d+)', locus_id)
        if genome_match:
            genome_id = genome_match.group(1)
        else:
            genome_id = locus_id.split('|')[0] if '|' in locus_id else locus_id
        
        return genome_id, locus_id
    
    def load_genome_files(self, sequences_dir: Path, annotations_dir: Optional[Path], 
                         proteins_dir: Path) -> Dict[str, Dict[str, Path]]:
        """
        Discover genome files from organized directory structure.
        
        Returns:
            Dict mapping genome_id -> {"fna": path, "gff": path, "faa": path}
        """
        genome_files = {}
        
        # Find all files by type
        fna_files = list(sequences_dir.glob("*.fna")) if sequences_dir and sequences_dir.exists() else []
        gff_files = list(annotations_dir.glob("*.gff")) if annotations_dir and annotations_dir.exists() else []
        faa_files = list(proteins_dir.glob("*.faa")) if proteins_dir and proteins_dir.exists() else []
        
        # Get all unique genome IDs from any file type
        genome_ids = set()
        for file_list in [fna_files, gff_files, faa_files]:
            genome_ids.update(f.stem for f in file_list)
        
        # Build file mapping for each genome
        for genome_id in genome_ids:
            fna_file = sequences_dir / f"{genome_id}.fna" if sequences_dir and sequences_dir.exists() else None
            gff_file = annotations_dir / f"{genome_id}.gff" if annotations_dir and annotations_dir.exists() else None
            faa_file = proteins_dir / f"{genome_id}.faa" if proteins_dir and proteins_dir.exists() else None
            
            genome_files[genome_id] = {
                "fna": fna_file if fna_file and fna_file.exists() else None,
                "gff": gff_file if gff_file and gff_file.exists() else None,
                "faa": faa_file if faa_file and faa_file.exists() else None
            }
        
        logger.info(f"Discovered {len(genome_files)} genomes from organized directories:")
        logger.info(f"  Sequences: {len(fna_files)} files")
        logger.info(f"  Annotations: {len(gff_files)} files") 
        logger.info(f"  Proteins: {len(faa_files)} files")
        return genome_files
    
    def ingest_genomes(self, genome_files: Dict[str, Dict[str, Path]], 
                      pfam_annotations: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        """
        Ingest genome data (sequences, genes, annotations) into database.
        
        Args:
            genome_files: Dict mapping genome_id -> file paths
            pfam_annotations: Optional PFAM annotations from annotation pipeline
        """
        cursor = self.conn.cursor()
        
        for genome_id, files in genome_files.items():
            logger.info(f"Processing genome: {genome_id}")
            
            # Get PFAM annotations for this genome
            genome_pfam = pfam_annotations.get(genome_id, {}) if pfam_annotations else {}
            
            # Process genome sequence file
            contigs = []
            total_genome_size = 0
            
            if files["fna"] and files["fna"].exists():
                for record in SeqIO.parse(files["fna"], "fasta"):
                    contig_id = record.id
                    contig_length = len(record.seq)
                    gc_content = GC(record.seq) / 100.0  # Convert percentage to fraction
                    
                    contigs.append({
                        "contig_id": contig_id,
                        "genome_id": genome_id,
                        "contig_name": record.description,
                        "length": contig_length,
                        "gc_content": gc_content,
                        "gene_count": 0  # Will be updated after gene processing
                    })
                    
                    total_genome_size += contig_length
            
            # Process protein FASTA file directly - extract all info from headers
            genes = []
            if files["faa"] and files["faa"].exists():
                genes = self.parse_protein_fasta(files["faa"], genome_id, genome_pfam)
            
            # Update contig gene counts
            contig_gene_counts = {}
            for gene in genes:
                contig_id = gene["contig_id"]
                contig_gene_counts[contig_id] = contig_gene_counts.get(contig_id, 0) + 1
            
            for contig in contigs:
                contig["gene_count"] = contig_gene_counts.get(contig["contig_id"], 0)
            
            # Insert genome record
            cursor.execute("""
                INSERT OR REPLACE INTO genomes 
                (genome_id, organism_name, total_contigs, total_genes, genome_size, 
                 file_path_fna, file_path_faa, file_path_gff)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                genome_id,
                f"Genome {genome_id}",  # Could be enhanced with metadata
                len(contigs),
                len(genes),
                total_genome_size,
                str(files["fna"]) if files["fna"] else None,
                str(files["faa"]) if files["faa"] else None,
                str(files["gff"]) if files["gff"] else None
            ))
            
            # Insert contigs
            for contig in contigs:
                cursor.execute("""
                    INSERT OR REPLACE INTO contigs 
                    (contig_id, genome_id, contig_name, length, gc_content, gene_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    contig["contig_id"], contig["genome_id"], contig["contig_name"],
                    contig["length"], contig["gc_content"], contig["gene_count"]
                ))
            
            # Insert genes
            for gene in genes:
                cursor.execute("""
                    INSERT OR REPLACE INTO genes
                    (gene_id, genome_id, contig_id, start_pos, end_pos, strand,
                     gene_length, protein_id, protein_sequence, pfam_domains, 
                     pfam_count, gc_content, partial_gene, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gene["gene_id"], gene["genome_id"], gene["contig_id"],
                    gene["start_pos"], gene["end_pos"], gene["strand"],
                    gene["gene_length"], gene["protein_id"], 
                    gene.get("protein_sequence", ""),
                    gene["pfam_domains"], gene["pfam_count"],
                    gene.get("gc_content", 0.0), gene.get("partial_gene", False),
                    gene.get("confidence_score", 0.0)
                ))
            
            logger.info(f"Ingested genome {genome_id}: {len(contigs)} contigs, {len(genes)} genes")
        
        self.conn.commit()
    
    def parse_protein_fasta(self, faa_file: Path, genome_id: str, 
                           pfam_annotations: Dict[str, str]) -> List[Dict]:
        """Parse protein FASTA file to extract gene information from headers."""
        genes = []
        
        for record in SeqIO.parse(faa_file, "fasta"):
            protein_id = record.id
            description = record.description
            protein_sequence = str(record.seq)
            
            # Parse Prodigal header format:
            # >accn|1313.30775.con.0001_1 # 1 # 270 # -1 # ID=1_1;partial=10;start_type=ATG;...
            
            # Extract coordinates and strand from description
            parts = description.split(' # ')
            if len(parts) >= 4:
                start_pos = int(parts[1])
                end_pos = int(parts[2])
                strand = int(parts[3])
            else:
                # Fallback if header format is different
                start_pos = 1
                end_pos = len(protein_sequence) * 3  # Approximate
                strand = 1
            
            # Extract contig from protein_id: "accn|1313.30775.con.0001_1" -> "accn|1313.30775.con.0001"
            contig_id = protein_id.rsplit('_', 1)[0] if '_' in protein_id else protein_id
            
            # Get PFAM domains for this protein
            pfam_domains = pfam_annotations.get(protein_id, "")
            pfam_count = len(pfam_domains.split(';')) if pfam_domains else 0
            
            gene = {
                "gene_id": protein_id,  # Use protein_id as gene_id for simplicity
                "genome_id": genome_id,
                "contig_id": contig_id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "strand": strand,
                "gene_length": end_pos - start_pos + 1,
                "protein_id": protein_id,
                "protein_sequence": protein_sequence,
                "pfam_domains": pfam_domains,
                "pfam_count": pfam_count,
                "partial_gene": 'partial=1' in description,
                "confidence_score": self._extract_confidence(description)
            }
            
            genes.append(gene)
        
        return genes
    
    def _extract_confidence(self, description: str) -> float:
        """Extract confidence score from Prodigal description."""
        import re
        match = re.search(r'conf=([0-9.]+)', description)
        return float(match.group(1)) if match else 0.0
    
    def _extract_window_index(self, window_id: str) -> Optional[int]:
        """Extract window index from window ID like 'sample_locus_123' -> 123."""
        try:
            # Window IDs end with '_<window_index>'
            parts = window_id.split('_')
            if len(parts) >= 2:
                return int(parts[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract window index from: {window_id}")
        return None
    
    def ingest_syntenic_blocks(self, blocks_file: Path, landscape_file: Optional[Path] = None) -> None:
        """Ingest syntenic blocks from CSV file with optional detailed window data from JSON."""
        logger.info(f"Loading syntenic blocks from {blocks_file}")
        
        df = pd.read_csv(blocks_file)
        cursor = self.conn.cursor()
        
        # Load detailed window data from landscape JSON if available
        detailed_blocks = {}
        if landscape_file and landscape_file.exists():
            logger.info(f"Loading detailed window data from {landscape_file}")
            with open(landscape_file, 'r') as f:
                landscape_data = json.load(f)
                # Index blocks by their position in the array (block_id)
                for i, block in enumerate(landscape_data.get('landscape', {}).get('blocks', [])):
                    detailed_blocks[i] = block
        
        for _, row in df.iterrows():
            # Parse locus IDs
            query_genome_id, query_contig_id = self.parse_locus_id(row['query_locus'])
            target_genome_id, target_contig_id = self.parse_locus_id(row['target_locus'])
            
            # Categorize block by size (now using gene windows, not bp)
            length = row['length']
            if length <= 3:
                block_type = 'small'
            elif length <= 10:
                block_type = 'medium'
            else:
                block_type = 'large'
            
            # Extract window information from detailed data
            block_id = row['block_id']
            query_window_start = query_window_end = None
            target_window_start = target_window_end = None
            query_windows_json = target_windows_json = None
            
            if block_id in detailed_blocks:
                detailed_block = detailed_blocks[block_id]
                
                # Extract window indices from window IDs
                query_windows = detailed_block.get('query_windows', [])
                target_windows = detailed_block.get('target_windows', [])
                
                if query_windows:
                    query_indices = [self._extract_window_index(w) for w in query_windows]
                    query_indices = [idx for idx in query_indices if idx is not None]
                    if query_indices:
                        query_window_start = min(query_indices)
                        query_window_end = max(query_indices)
                    query_windows_json = json.dumps(query_windows)
                
                if target_windows:
                    target_indices = [self._extract_window_index(w) for w in target_windows]
                    target_indices = [idx for idx in target_indices if idx is not None]
                    if target_indices:
                        target_window_start = min(target_indices)
                        target_window_end = max(target_indices)
                    target_windows_json = json.dumps(target_windows)
            
            cursor.execute("""
                INSERT INTO syntenic_blocks
                (block_id, query_locus, target_locus, query_genome_id, target_genome_id,
                 query_contig_id, target_contig_id, length, identity, score,
                 n_query_windows, n_target_windows, query_window_start, query_window_end,
                 target_window_start, target_window_end, query_windows_json, target_windows_json, block_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['block_id'], row['query_locus'], row['target_locus'],
                query_genome_id, target_genome_id, query_contig_id, target_contig_id,
                row['length'], row['identity'], row['score'],
                row['n_query_windows'], row['n_target_windows'],
                query_window_start, query_window_end, target_window_start, target_window_end,
                query_windows_json, target_windows_json, block_type
            ))
        
        self.conn.commit()
        logger.info(f"Ingested {len(df)} syntenic blocks with detailed window information")
    
    def ingest_clusters(self, clusters_file: Path) -> None:
        """Ingest cluster information from CSV file."""
        logger.info(f"Loading clusters from {clusters_file}")
        
        df = pd.read_csv(clusters_file)
        cursor = self.conn.cursor()
        
        for _, row in df.iterrows():
            # Categorize cluster
            size = row['size']
            diversity = row['diversity']
            
            if size > 1000:
                cluster_type = 'large'
            elif size > 100:
                cluster_type = 'medium'
            else:
                cluster_type = 'small'
            
            cursor.execute("""
                INSERT INTO clusters
                (cluster_id, size, consensus_length, consensus_score, diversity,
                 representative_query, representative_target, cluster_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['cluster_id'], row['size'], row['consensus_length'],
                row['consensus_score'], row['diversity'], row['representative_query'],
                row['representative_target'], cluster_type
            ))
        
        self.conn.commit()
        logger.info(f"Ingested {len(df)} clusters")
    
    def create_gene_block_mappings(self) -> None:
        """Create mappings between genes and syntenic blocks."""
        logger.info("Creating gene-block mappings...")
        
        cursor = self.conn.cursor()
        
        # This is a simplified mapping - in practice you'd need more sophisticated
        # overlap detection based on the actual syntenic block coordinates
        cursor.execute("""
            INSERT INTO gene_block_mappings (gene_id, block_id, block_role, relative_position)
            SELECT DISTINCT 
                g.gene_id,
                sb.block_id,
                'query' as block_role,
                0.5 as relative_position
            FROM genes g
            JOIN syntenic_blocks sb ON g.contig_id = sb.query_contig_id
            WHERE g.genome_id = sb.query_genome_id
        """)
        
        cursor.execute("""
            INSERT OR IGNORE INTO gene_block_mappings (gene_id, block_id, block_role, relative_position)
            SELECT DISTINCT 
                g.gene_id,
                sb.block_id,
                'target' as block_role,
                0.5 as relative_position
            FROM genes g
            JOIN syntenic_blocks sb ON g.contig_id = sb.target_contig_id
            WHERE g.genome_id = sb.target_genome_id
        """)
        
        self.conn.commit()
        logger.info("Gene-block mappings created")
    
    def generate_annotation_stats(self) -> None:
        """Generate annotation statistics for dashboard."""
        logger.info("Generating annotation statistics...")
        
        cursor = self.conn.cursor()
        
        # Get stats per genome
        cursor.execute("""
            SELECT 
                g.genome_id,
                COUNT(DISTINCT ge.gene_id) as total_genes,
                COUNT(DISTINCT CASE WHEN ge.pfam_domains != '' THEN ge.gene_id END) as annotated_genes,
                SUM(ge.pfam_count) as total_pfam_domains,
                COUNT(DISTINCT CASE WHEN ge.pfam_domains != '' THEN ge.pfam_domains END) as unique_domain_combinations,
                COUNT(DISTINCT gbm.gene_id) as syntenic_genes
            FROM genomes g
            LEFT JOIN genes ge ON g.genome_id = ge.genome_id
            LEFT JOIN gene_block_mappings gbm ON ge.gene_id = gbm.gene_id
            GROUP BY g.genome_id
        """)
        
        stats = cursor.fetchall()
        
        for stat in stats:
            genome_id, total_genes, annotated_genes, total_pfam, unique_combinations, syntenic_genes = stat
            non_syntenic_genes = total_genes - syntenic_genes if syntenic_genes else total_genes
            
            cursor.execute("""
                INSERT OR REPLACE INTO annotation_stats
                (genome_id, total_genes, annotated_genes, total_pfam_domains, 
                 unique_pfam_domains, syntenic_genes, non_syntenic_genes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                genome_id, total_genes, annotated_genes, total_pfam,
                unique_combinations, syntenic_genes, non_syntenic_genes
            ))
        
        self.conn.commit()
        logger.info("Annotation statistics generated")

def main():
    """Command line interface for data ingestion."""
    parser = argparse.ArgumentParser(description="Populate ELSA genome browser database")
    parser.add_argument("--db-path", type=Path, default="genome_browser.db",
                       help="Path to SQLite database")
    parser.add_argument("--genome-dir", type=Path,
                       help="Directory containing genome files (fna, gff, faa) - legacy format")
    parser.add_argument("--sequences-dir", type=Path,
                       help="Directory containing nucleotide sequences (.fna)")
    parser.add_argument("--proteins-dir", type=Path,
                       help="Directory containing proteins (.faa)")
    parser.add_argument("--blocks-file", type=Path, required=True,
                       help="Syntenic blocks CSV file")
    parser.add_argument("--clusters-file", type=Path, required=True,
                       help="Syntenic clusters CSV file")
    parser.add_argument("--pfam-results", type=Path,
                       help="PFAM annotation results JSON file")
    parser.add_argument("--landscape-file", type=Path,
                       help="Syntenic landscape JSON file with detailed window information")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handle directory structure
    if args.genome_dir:
        # Legacy format
        sequences_dir = args.genome_dir
        annotations_dir = args.genome_dir
        proteins_dir = args.genome_dir
        logger.info("Using legacy directory structure")
    else:
        # New organized format
        if not all([args.sequences_dir, args.proteins_dir]):
            logger.error("Either --genome-dir OR both (--sequences-dir, --proteins-dir) must be provided")
            sys.exit(1)
        
        sequences_dir = args.sequences_dir
        annotations_dir = getattr(args, 'annotations_dir', None)  # Optional
        proteins_dir = args.proteins_dir
        logger.info("Using organized directory structure")

    # Load PFAM annotations if provided
    pfam_annotations = None
    if args.pfam_results and args.pfam_results.exists():
        with open(args.pfam_results, 'r') as f:
            pfam_data = json.load(f)
            pfam_annotations = pfam_data.get("genome_annotations", {})
        logger.info(f"Loaded PFAM annotations for {len(pfam_annotations)} genomes")
    
    # Ingest data
    with ELSADataIngester(args.db_path) as ingester:
        # Discover and ingest genomes
        genome_files = ingester.load_genome_files(sequences_dir, annotations_dir, proteins_dir)
        ingester.ingest_genomes(genome_files, pfam_annotations)
        
        # Ingest syntenic analysis results
        ingester.ingest_syntenic_blocks(args.blocks_file, args.landscape_file)
        ingester.ingest_clusters(args.clusters_file)
        
        # Create mappings and statistics
        ingester.create_gene_block_mappings()
        ingester.generate_annotation_stats()
    
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()