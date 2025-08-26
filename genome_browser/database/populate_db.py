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
            
            # Get PFAM annotations for this genome by collecting from all sources
            genome_pfam = {}
            if pfam_annotations:
                # Method 1: Direct genome_id lookup (works for 1313.30775)
                if genome_id in pfam_annotations:
                    genome_pfam.update(pfam_annotations[genome_id])
                
                # Method 2: Collect from contig-based keys (works for other genomes)
                for pfam_key, pfam_proteins in pfam_annotations.items():
                    # Skip if we already got this from direct lookup
                    if pfam_key == genome_id:
                        continue
                    
                    # Check if any proteins in this pfam_key belong to our genome
                    # Look for proteins that match our genome pattern
                    for protein_id in pfam_proteins:
                        # Extract genome from protein_id using multiple strategies
                        protein_matches_genome = False
                        
                        # Strategy 1: Direct substring match (works for 1313.30775)
                        if genome_id in protein_id:
                            protein_matches_genome = True
                        
                        # Strategy 2: Extract genome from protein ID pattern like "accn|CAYEVI010000001_1"
                        # Match the base part: CAYEVI000000000 should match CAYEVI*
                        elif '|' in protein_id:
                            contig_part = protein_id.split('|')[1] if '|' in protein_id else protein_id
                            # Extract pattern like "CAYEVI010000001" from "CAYEVI010000001_1"
                            if '_' in contig_part:
                                contig_base = contig_part.split('_')[0]
                                # Check if genome_id shares a common prefix (handle CAYEVI000000000 vs CAYEVI010000001)
                                if len(genome_id) >= 6 and len(contig_base) >= 6:
                                    genome_prefix = genome_id[:6]  # Get "CAYEVI" part
                                    contig_prefix = contig_base[:6]  # Get "CAYEVI" part
                                    if genome_prefix == contig_prefix:
                                        protein_matches_genome = True
                        
                        if protein_matches_genome:
                            # This looks like it belongs to our genome, add all proteins from this key
                            genome_pfam.update(pfam_proteins)
                            break
            
            logger.info(f"DEBUG: Genome {genome_id} has {len(genome_pfam)} PFAM annotations")
            
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
                # Compute and attach primary PFAM token for fast consensus
                pf = gene.get("pfam_domains", "") or ""
                primary_pfam = pf.split(';', 1)[0].strip() if pf else ""
                cursor.execute("""
                    INSERT OR REPLACE INTO genes
                    (gene_id, genome_id, contig_id, start_pos, end_pos, strand,
                     gene_length, protein_id, protein_sequence, pfam_domains, 
                     pfam_count, primary_pfam, gc_content, partial_gene, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gene["gene_id"], gene["genome_id"], gene["contig_id"],
                    gene["start_pos"], gene["end_pos"], gene["strand"],
                    gene["gene_length"], gene["protein_id"], 
                    gene.get("protein_sequence", ""),
                    gene["pfam_domains"], gene["pfam_count"], primary_pfam,
                    gene.get("gc_content", 0.0), gene.get("partial_gene", False),
                    gene.get("confidence_score", 0.0)
                ))
            
            logger.info(f"Ingested genome {genome_id}: {len(contigs)} contigs, {len(genes)} genes")
        
        self.conn.commit()
    
    def parse_protein_fasta(self, faa_file: Path, genome_id: str, 
                           genome_pfam: Dict[str, str]) -> List[Dict]:
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
            pfam_domains = genome_pfam.get(protein_id, "")
            pfam_count = len(pfam_domains.split(';')) if pfam_domains else 0
            
            # DEBUG: Log PFAM annotation for first few proteins
            if len(genes) < 3:
                logger.info(f"DEBUG: Protein {protein_id} -> PFAM: '{pfam_domains}' (count: {pfam_count})")
            
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
        """Ingest syntenic blocks from CSV file with embedded window data.

        Clears existing syntenic_blocks/cluster_assignments to allow reruns without
        UNIQUE constraint failures.
        """
        logger.info(f"Loading syntenic blocks from {blocks_file}")
        # Clear previous data for reruns (respect FK order)
        self.conn.execute("DELETE FROM cluster_assignments")
        self.conn.execute("DELETE FROM gene_block_mappings")
        self.conn.execute("DELETE FROM syntenic_blocks")

        df = pd.read_csv(blocks_file)
        cursor = self.conn.cursor()
        
        # Check if the CSV has the new embedded window columns
        has_embedded_windows = all(col in df.columns for col in [
            'query_window_start', 'query_window_end', 'target_window_start', 'target_window_end',
            'query_windows_json', 'target_windows_json'
        ])
        
        if has_embedded_windows:
            logger.info("Using embedded window information from CSV")
        else:
            logger.info("CSV does not have embedded window info - using landscape file if available")
            
            # Load detailed window data from landscape JSON if available (legacy support)
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
            
            # Extract window information - prioritize embedded data over legacy JSON
            block_id = row['block_id']
            
            if has_embedded_windows:
                # Use embedded window information from CSV
                query_window_start = row.get('query_window_start')
                query_window_end = row.get('query_window_end')
                target_window_start = row.get('target_window_start')
                target_window_end = row.get('target_window_end')
                
                # Convert semicolon-separated format back to JSON for storage
                query_windows_str = row.get('query_windows_json', '')
                target_windows_str = row.get('target_windows_json', '')
                
                if query_windows_str:
                    query_windows_list = query_windows_str.split(';')
                    query_windows_json = json.dumps(query_windows_list)
                else:
                    query_windows_json = None
                
                if target_windows_str:
                    target_windows_list = target_windows_str.split(';')
                    target_windows_json = json.dumps(target_windows_list)
                else:
                    target_windows_json = None
            else:
                # Legacy: extract from landscape JSON
                query_window_start = query_window_end = None
                target_window_start = target_window_end = None
                query_windows_json = target_windows_json = None
                
                if 'detailed_blocks' in locals() and block_id in detailed_blocks:
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
                (block_id, cluster_id, query_locus, target_locus, query_genome_id, target_genome_id,
                 query_contig_id, target_contig_id, length, identity, score,
                 n_query_windows, n_target_windows, query_window_start, query_window_end,
                 target_window_start, target_window_end, query_windows_json, target_windows_json, block_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['block_id'], row.get('cluster_id', 0), row['query_locus'], row['target_locus'],
                query_genome_id, target_genome_id, query_contig_id, target_contig_id,
                row['length'], row['identity'], row['score'],
                row['n_query_windows'], row['n_target_windows'],
                query_window_start, query_window_end, target_window_start, target_window_end,
                query_windows_json, target_windows_json, block_type
            ))
        
        self.conn.commit()
        logger.info(f"Ingested {len(df)} syntenic blocks with window information")
    
    def ingest_clusters(self, clusters_file: Path) -> None:
        """Ingest cluster information from CSV file (legacy summary, optional).

        Clears existing clusters to avoid duplicate inserts on reruns.
        """
        logger.info(f"Loading clusters from {clusters_file}")
        # Clear previous cluster table to avoid duplicates
        self.conn.execute("DELETE FROM clusters")
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
        """Create precise mappings between genes and syntenic blocks using window information."""
        logger.info("Creating gene-block mappings using window boundaries...")
        
        cursor = self.conn.cursor()
        
        # Clear existing mappings
        cursor.execute("DELETE FROM gene_block_mappings")
        
        # Get all syntenic blocks with window information
        cursor.execute("""
            SELECT block_id, query_genome_id, target_genome_id,
                   query_contig_id, target_contig_id,
                   query_window_start, query_window_end,
                   target_window_start, target_window_end
            FROM syntenic_blocks
            WHERE query_window_start IS NOT NULL AND query_window_end IS NOT NULL
        """)
        blocks = cursor.fetchall()
        
        mappings = []
        blocks_without_windows = 0
        
        for block in blocks:
            (block_id, query_genome_id, target_genome_id,
             query_contig_id, target_contig_id,
             query_start, query_end, target_start, target_end) = block
            
            # Extract clean contig IDs (remove genome prefix if present)
            # Handle format like "1313.30775_accn|1313.30775.con.0001" -> "accn|1313.30775.con.0001"
            if '_' in query_contig_id and '|' in query_contig_id:
                query_clean_contig = query_contig_id.split('_', 1)[1]  # Take everything after first underscore
            else:
                query_clean_contig = query_contig_id
                
            if '_' in target_contig_id and '|' in target_contig_id:
                target_clean_contig = target_contig_id.split('_', 1)[1]  # Take everything after first underscore
            else:
                target_clean_contig = target_contig_id
            
            # Map query genes using window boundaries
            # Convert window indices to approximate gene indices:
            # Each window of size 5 genes (stride 1) spans 5 gene indices.
            if query_start is not None and query_end is not None:
                # Inclusive gene index range: [query_start, query_end + 4]
                gene_start_idx = int(query_start)
                gene_end_idx = int(query_end) + 4
                num_genes = max(1, gene_end_idx - gene_start_idx + 1)

                query_gene_query = """
                    SELECT gene_id FROM genes 
                    WHERE genome_id = ? AND contig_id = ?
                    ORDER BY start_pos
                    LIMIT ? OFFSET ?
                """
                cursor.execute(query_gene_query, (
                    query_genome_id, query_clean_contig,
                    num_genes, gene_start_idx
                ))
                query_genes = cursor.fetchall()

                for i, (gene_id,) in enumerate(query_genes):
                    relative_pos = i / max(1, len(query_genes) - 1) if len(query_genes) > 1 else 0.5
                    mappings.append((gene_id, block_id, 'query', relative_pos))
            
            # Map target genes using window boundaries  
            if target_start is not None and target_end is not None:
                gene_start_idx = int(target_start)
                gene_end_idx = int(target_end) + 4
                num_genes = max(1, gene_end_idx - gene_start_idx + 1)

                target_gene_query = """
                    SELECT gene_id FROM genes 
                    WHERE genome_id = ? AND contig_id = ?
                    ORDER BY start_pos
                    LIMIT ? OFFSET ?
                """
                cursor.execute(target_gene_query, (
                    target_genome_id, target_clean_contig,
                    num_genes, gene_start_idx
                ))
                target_genes = cursor.fetchall()

                for i, (gene_id,) in enumerate(target_genes):
                    relative_pos = i / max(1, len(target_genes) - 1) if len(target_genes) > 1 else 0.5
                    mappings.append((gene_id, block_id, 'target', relative_pos))
        
        # Handle blocks without window information (fallback to representative genes)
        cursor.execute("""
            SELECT COUNT(*) FROM syntenic_blocks 
            WHERE query_window_start IS NULL OR query_window_end IS NULL
        """)
        blocks_without_windows = cursor.fetchone()[0]
        
        if blocks_without_windows > 0:
            logger.warning(f"{blocks_without_windows} blocks lack window information, skipping precise mapping")
        
        # Insert all mappings in batch
        if mappings:
            cursor.executemany("""
                INSERT INTO gene_block_mappings (gene_id, block_id, block_role, relative_position)
                VALUES (?, ?, ?, ?)
            """, mappings)
        
        self.conn.commit()
        
        # Report mapping statistics
        cursor.execute("SELECT COUNT(DISTINCT gene_id) FROM gene_block_mappings")
        unique_genes = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM gene_block_mappings")
        total_mappings = cursor.fetchone()[0]
        
        logger.info(f"Gene-block mappings created: {unique_genes} unique genes, {total_mappings} total mappings")
        logger.info(f"Average mappings per gene: {total_mappings / max(1, unique_genes):.1f}")

    def compute_block_consensus(self, df_percentile_ban: float = 0.9) -> None:
        """Pre-compute pairwise PFAM consensus cassette per block and persist summary.

        Creates table block_consensus if needed with columns:
          - block_id INTEGER PRIMARY KEY
          - consensus_len INTEGER (count of 100% conserved tokens)
          - consensus_json TEXT (JSON payload with tokens/pairs)
        """
        # Import with robust package fallback when executed as a script
        try:
            from genome_browser.database.cluster_content import compute_block_pfam_consensus
        except Exception:
            try:
                from database.cluster_content import compute_block_pfam_consensus
            except Exception:
                # Fallback: load module directly from file path when run as a script
                import importlib.util as _ilu
                from pathlib import Path as _Path
                _mod_path = _Path(__file__).resolve().parent / 'cluster_content.py'
                if not _mod_path.exists():
                    raise ImportError("Unable to locate cluster_content.py for block consensus computation")
                _spec = _ilu.spec_from_file_location('gb_cluster_content', str(_mod_path))
                _mod = _ilu.module_from_spec(_spec)
                assert _spec and _spec.loader
                _spec.loader.exec_module(_mod)
                compute_block_pfam_consensus = getattr(_mod, 'compute_block_pfam_consensus')
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS block_consensus (
                block_id INTEGER PRIMARY KEY,
                consensus_len INTEGER,
                consensus_json TEXT,
                FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_block_consensus_len ON block_consensus(consensus_len)")
        # Clear previous
        cur.execute("DELETE FROM block_consensus")
        self.conn.commit()

        # Fast path: compute only consensus_len for all blocks via SQL (set-based)
        logger.info("Computing block-level PFAM consensus lengths (set-based SQL)…")
        cur.execute("DELETE FROM block_consensus")
        self.conn.commit()
        sql = """
        WITH q AS (
            SELECT gb.block_id, g.primary_pfam AS tok
            FROM gene_block_mappings gb
            JOIN genes g ON gb.gene_id = g.gene_id
            WHERE gb.block_role = 'query' AND g.primary_pfam IS NOT NULL AND g.primary_pfam != ''
            GROUP BY gb.block_id, tok
        ),
        t AS (
            SELECT gb.block_id, g.primary_pfam AS tok
            FROM gene_block_mappings gb
            JOIN genes g ON gb.gene_id = g.gene_id
            WHERE gb.block_role = 'target' AND g.primary_pfam IS NOT NULL AND g.primary_pfam != ''
            GROUP BY gb.block_id, tok
        ),
        cnt AS (
            SELECT q.block_id, COUNT(*) AS c
            FROM q JOIN t ON q.block_id = t.block_id AND q.tok = t.tok
            GROUP BY q.block_id
        )
        INSERT INTO block_consensus (block_id, consensus_len, consensus_json)
        SELECT sb.block_id, COALESCE(cnt.c, 0) AS consensus_len, NULL
        FROM syntenic_blocks sb
        LEFT JOIN cnt ON cnt.block_id = sb.block_id;
        """
        cur.executescript(sql)
        self.conn.commit()
        logger.info("Block consensus length table populated")
    
    def create_cluster_assignments(self,
                                  jaccard_tau: float = 0.60,
                                  mutual_k: int = 5,
                                  degree_cap: int = 15,
                                  df_max: int = 30) -> None:
        """Compute SRP mutual-Jaccard clusters (no PFAM, no size targets) and persist to DB.

        Uses window-level embeddings from windows.parquet to form SRP shingles per block,
        builds a sparse mutual-k Jaccard graph, runs community detection, and writes
        cluster_assignments + updates syntenic_blocks.cluster_id.
        """
        import pandas as _pd
        import numpy as _np
        from types import SimpleNamespace as _NS
        from pathlib import Path as _Path
        from elsa.analyze.cluster_mutual_jaccard import cluster_blocks_jaccard as _cluster

        logger.info("Creating cluster assignments via SRP mutual-Jaccard…")

        # If syntenic_blocks already carries non-zero cluster_id from CSV, just rebuild clusters summary
        existing = _pd.read_sql_query("SELECT COUNT(*) AS n FROM syntenic_blocks WHERE cluster_id > 0", self.conn)['n'].iloc[0]
        if existing and int(existing) > 0:
            logger.info("Detected existing cluster_id assignments (%d). Rebuilding clusters summary and skipping recluster.", int(existing))
            try:
                self.conn.execute("DELETE FROM clusters")
                agg = _pd.read_sql_query(
                    """
                    SELECT cluster_id, COUNT(*) AS size, CAST(AVG(length) AS INT) AS consensus_length, AVG(identity) AS consensus_score
                    FROM syntenic_blocks WHERE cluster_id > 0 GROUP BY cluster_id
                    """,
                    self.conn,
                )
                rep = _pd.read_sql_query(
                    "SELECT cluster_id, block_id, query_locus, target_locus, score FROM syntenic_blocks WHERE cluster_id > 0 AND score IS NOT NULL",
                    self.conn,
                )
                rep = rep.sort_values(['cluster_id','score'], ascending=[True, False]).groupby('cluster_id').head(1)
                merged = agg.merge(rep[['cluster_id','query_locus','target_locus']], on='cluster_id', how='left')
                merged = merged.fillna({'query_locus':'', 'target_locus':''})
                cur = self.conn.cursor()
                for row in merged.itertuples(index=False):
                    cur.execute(
                        """
                        INSERT INTO clusters
                        (cluster_id, size, consensus_length, consensus_score, diversity,
                         representative_query, representative_target, cluster_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(row.cluster_id), int(row.size), int(row.consensus_length or 0), float(row.consensus_score or 0.0), 0.0,
                            str(row.query_locus), str(row.target_locus), 'unknown'
                        )
                    )
                self.conn.commit()
                logger.info("Clusters summary rebuilt from existing assignments: %d rows", len(merged))
            except Exception as e:
                logger.warning("Failed to rebuild clusters summary from existing assignments: %s", e)
            return

        # 1) Load blocks from DB (with window JSON)
        df = _pd.read_sql_query(
            """
            SELECT block_id, query_windows_json, target_windows_json
            FROM syntenic_blocks
            """,
            self.conn,
        )
        blocks = []

        class _Block:
            __slots__ = ("id", "query_windows", "target_windows", "strand", "matches")
            def __init__(self, bid, qw, tw):
                self.id = int(bid)
                self.query_windows = list(qw)
                self.target_windows = list(tw)
                self.strand = 1
                self.matches = [
                    _NS(query_window_id=qq, target_window_id=tt) for qq, tt in zip(self.query_windows, self.target_windows)
                ]

        for row in df.itertuples(index=False):
            bid = int(row.block_id)
            qwins = [x for x in str(row.query_windows_json or '').split(';') if x]
            twins = [x for x in str(row.target_windows_json or '').split(';') if x]
            n = min(len(qwins), len(twins))
            if n <= 0:
                continue
            blocks.append(_Block(bid, qwins[:n], twins[:n]))

        logger.info("Loaded %d blocks with window info", len(blocks))
        if not blocks:
            logger.warning("No blocks found; skipping clustering")
            return

        # 2) Create window embedding lookup from windows.parquet
        def _find_windows_parquet() -> _Path | None:
            here = _Path(__file__).resolve()
            candidates = [
                here.parents[2] / 'elsa_index/shingles/windows.parquet',
                here.parents[1] / 'elsa_index/shingles/windows.parquet',
                _Path('elsa_index/shingles/windows.parquet').resolve(),
            ]
            for p in candidates:
                if p.exists():
                    return p
            return None

        win_pq = _find_windows_parquet()
        if not win_pq:
            logger.error("windows.parquet not found. Expected at elsa_index/shingles/windows.parquet")
            return
        win_df = _pd.read_parquet(win_pq)
        emb_cols = [c for c in win_df.columns if str(c).startswith('emb_')]
        if not emb_cols:
            logger.error("windows.parquet missing emb_* columns: %s", win_pq)
            return
        win_df = win_df[['sample_id', 'locus_id', 'window_idx'] + emb_cols].copy()
        win_df['wid'] = win_df['sample_id'].astype(str) + '_' + win_df['locus_id'].astype(str) + '_' + win_df['window_idx'].astype(str)
        mat = win_df[emb_cols].to_numpy(dtype=_np.float32)
        wid2i = {w: i for i, w in enumerate(win_df['wid'].tolist())}

        def lookup(wid: str):
            i = wid2i.get(str(wid))
            if i is None:
                return None
            return mat[i]

        # 3) Build clustering config
        cfg = _NS(
            jaccard_tau=float(jaccard_tau),
            mutual_k=int(mutual_k),
            df_max=int(df_max),
            use_weighted_jaccard=True,
            min_low_df_anchors=2,
            idf_mean_min=1.0,
            max_df_percentile=None,
            v_mad_max_genes=0.5,
            min_anchors=2,
            min_span_genes=4,
            enable_cassette_mode=True,
            cassette_max_len=4,
            degree_cap=int(degree_cap),
            k_core_min_degree=3,
            triangle_support_min=1,
            use_community_detection=True,
            community_method='greedy',
            srp_bits=256, srp_bands=32, srp_band_bits=8, srp_seed=1337,
            shingle_k=3,
            keep_singletons=False, sink_label=0,
            size_ratio_min=0.5, size_ratio_max=2.0,
        )

        # 4) Cluster and persist
        logger.info("Clustering %d blocks (SRP mutual-Jaccard)…", len(blocks))
        assignments = _cluster(blocks, lookup, cfg)
        from collections import Counter as _Counter
        ctr = _Counter([c for c in assignments.values() if c and c > 0])
        logger.info("Found %d non-sink clusters. Top sizes: %s", len(ctr), sorted(ctr.values(), reverse=True)[:10])

        cur = self.conn.cursor()
        cur.execute("DELETE FROM cluster_assignments")
        rows = [(int(bid), int(cid)) for bid, cid in assignments.items() if cid and cid > 0]
        if rows:
            cur.executemany("INSERT INTO cluster_assignments (block_id, cluster_id) VALUES (?, ?)", rows)
        # Reset and update
        cur.execute("UPDATE syntenic_blocks SET cluster_id = 0")
        cur.execute(
            """
            UPDATE syntenic_blocks
            SET cluster_id = (
                SELECT ca.cluster_id FROM cluster_assignments ca WHERE ca.block_id = syntenic_blocks.block_id
            )
            WHERE block_id IN (SELECT block_id FROM cluster_assignments)
            """
        )
        self.conn.commit()
        logger.info("Cluster assignments persisted: %d assigned", len(rows))

        # Rebuild clusters summary table from current assignments so the UI has consistent metadata
        logger.info("Rebuilding clusters summary table from assignments…")
        try:
            # Clear previous summary
            self.conn.execute("DELETE FROM clusters")
            # Aggregate per-cluster stats from syntenic_blocks
            agg_query = """
                SELECT cluster_id,
                       COUNT(*) AS size,
                       CAST(AVG(length) AS INT) AS consensus_length,
                       AVG(identity) AS consensus_score
                FROM syntenic_blocks
                WHERE cluster_id > 0
                GROUP BY cluster_id
            """
            clusters_df = _pd.read_sql_query(agg_query, self.conn)
            rep_df = _pd.read_sql_query(
                """
                SELECT sb.cluster_id, sb.block_id, sb.query_locus, sb.target_locus, sb.score
                FROM syntenic_blocks sb
                WHERE sb.cluster_id > 0 AND sb.score IS NOT NULL
            """, self.conn)
            rep_block = rep_df.sort_values(['cluster_id','score'], ascending=[True, False]).groupby('cluster_id').head(1)

            merged = clusters_df.merge(rep_block[['cluster_id','query_locus','target_locus']], on='cluster_id', how='left')
            merged['diversity'] = 0.0
            merged['cluster_type'] = 'unknown'
            merged = merged.fillna({'query_locus':'', 'target_locus':''})

            cur = self.conn.cursor()
            for row in merged.itertuples(index=False):
                cur.execute(
                    """
                    INSERT INTO clusters
                    (cluster_id, size, consensus_length, consensus_score, diversity,
                     representative_query, representative_target, cluster_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row.cluster_id), int(row.size), int(row.consensus_length or 0),
                        float(row.consensus_score or 0.0), float(row.diversity),
                        str(row.query_locus), str(row.target_locus), str(row.cluster_type)
                    )
                )
            self.conn.commit()
            logger.info("Clusters summary rebuilt: %d rows", len(merged))
        except Exception as e:
            logger.warning("Failed to rebuild clusters summary: %s", e)

        # If CSV already provided cluster_ids, we can skip re-clustering.
    
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
        logger.info(f"DEBUG: Loading PFAM results from: {args.pfam_results}")
        with open(args.pfam_results, 'r') as f:
            pfam_data = json.load(f)
            pfam_annotations = pfam_data.get("genome_annotations", {})
        logger.info(f"DEBUG: Loaded PFAM annotations for {len(pfam_annotations)} genomes")
        
        # DEBUG: Show sample data
        if pfam_annotations:
            first_genome = next(iter(pfam_annotations.keys()))
            sample_proteins = list(pfam_annotations[first_genome].keys())[:3]
            logger.info(f"DEBUG: Sample genome '{first_genome}' has {len(pfam_annotations[first_genome])} proteins")
            for protein_id in sample_proteins:
                domains = pfam_annotations[first_genome][protein_id]
                logger.info(f"DEBUG: Sample protein '{protein_id}': {domains}")
    else:
        logger.info(f"DEBUG: No PFAM results file provided or file doesn't exist: {args.pfam_results}")
    
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
        ingester.create_cluster_assignments()
        # Precompute block-level PFAM consensus (100% conserved tokens only)
        ingester.compute_block_consensus(df_percentile_ban=0.9)
        ingester.generate_annotation_stats()
    
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()
