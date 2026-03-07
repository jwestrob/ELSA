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
            
            # Insert contigs in batch
            if contigs:
                cursor.executemany("""
                    INSERT OR REPLACE INTO contigs
                    (contig_id, genome_id, contig_name, length, gc_content, gene_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [(
                    c["contig_id"], c["genome_id"], c["contig_name"],
                    c["length"], c["gc_content"], c["gene_count"]
                ) for c in contigs])
            
            # Insert genes in batch (much faster than individual inserts)
            gene_tuples = []
            for gene in genes:
                # Compute and attach primary PFAM token for fast consensus
                pf = gene.get("pfam_domains", "") or ""
                primary_pfam = pf.split(';', 1)[0].strip() if pf else ""
                gene_tuples.append((
                    gene["gene_id"], gene["genome_id"], gene["contig_id"],
                    gene["start_pos"], gene["end_pos"], gene["strand"],
                    gene["gene_length"], gene["protein_id"],
                    gene.get("protein_sequence", ""),
                    gene["pfam_domains"], gene["pfam_count"], primary_pfam,
                    gene.get("gc_content", 0.0), gene.get("partial_gene", False),
                    gene.get("confidence_score", 0.0)
                ))

            if gene_tuples:
                cursor.executemany("""
                    INSERT OR REPLACE INTO genes
                    (gene_id, genome_id, contig_id, start_pos, end_pos, strand,
                     gene_length, protein_id, protein_sequence, pfam_domains,
                     pfam_count, primary_pfam, gc_content, partial_gene, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, gene_tuples)
            
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
        # Clear previous data for reruns (disable FK checks temporarily)
        self.conn.execute("PRAGMA foreign_keys = OFF")
        self.conn.execute("DELETE FROM cluster_assignments")
        self.conn.execute("DELETE FROM gene_block_mappings")
        self.conn.execute("DELETE FROM syntenic_blocks")
        self.conn.execute("DELETE FROM clusters")
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys = ON")

        df = pd.read_csv(blocks_file)
        cursor = self.conn.cursor()

        # Detect v2 chain pipeline format (has query_genome/target_genome instead of query_locus/target_locus)
        is_v2 = 'query_genome' in df.columns and 'query_locus' not in df.columns

        if is_v2:
            logger.info("Detected v2 chain pipeline format — mapping to browser schema")

        # Check if the CSV has the new embedded window columns
        has_embedded_windows = all(col in df.columns for col in [
            'query_window_start', 'query_window_end', 'target_window_start', 'target_window_end',
            'query_windows_json', 'target_windows_json'
        ])

        if has_embedded_windows:
            logger.info("Using embedded window information from CSV")
        elif not is_v2:
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

        # Collect all block tuples for batch insert
        block_tuples = []
        for _, row in df.iterrows():
            block_id = row['block_id']

            if is_v2:
                # v2 chain pipeline: columns are query_genome, query_contig, query_start, query_end, etc.
                query_genome_id = str(row['query_genome'])
                target_genome_id = str(row['target_genome'])
                query_contig_id = str(row['query_contig'])
                target_contig_id = str(row['target_contig'])
                length = int(row.get('n_anchors', row.get('n_genes', 0)))
                raw_score = float(row.get('chain_score', 0.0))
                # Normalize: chain_score is cumulative similarity sum; divide by n_anchors for per-anchor avg
                identity = raw_score / max(1, length)
                score = raw_score
                query_locus = f"{query_genome_id}:{query_contig_id}:{row['query_start']}-{row['query_end']}"
                target_locus = f"{target_genome_id}:{target_contig_id}:{row['target_start']}-{row['target_end']}"
                n_query_windows = length
                n_target_windows = length
                # Use gene position indices as window boundaries for gene-block mapping
                query_window_start = int(row['query_start'])
                query_window_end = int(row['query_end'])
                target_window_start = int(row['target_start'])
                target_window_end = int(row['target_end'])
                # Store bp ranges and anchor gene IDs if available (v2 pipeline)
                _bp_meta = {}
                if pd.notna(row.get('query_start_bp', None)):
                    _bp_meta['query_start_bp'] = int(row['query_start_bp'])
                    _bp_meta['query_end_bp'] = int(row['query_end_bp'])
                    _bp_meta['target_start_bp'] = int(row['target_start_bp'])
                    _bp_meta['target_end_bp'] = int(row['target_end_bp'])
                query_windows_json = json.dumps(_bp_meta) if _bp_meta else None
                target_windows_json = None
            else:
                # Legacy format: parse combined locus IDs
                query_genome_id, query_contig_id = self.parse_locus_id(row['query_locus'])
                target_genome_id, target_contig_id = self.parse_locus_id(row['target_locus'])
                length = row['length']
                identity = row['identity']
                score = row['score']
                query_locus = row['query_locus']
                target_locus = row['target_locus']
                n_query_windows = row['n_query_windows']
                n_target_windows = row['n_target_windows']

                # Extract window information - prioritize embedded data over legacy JSON
                if has_embedded_windows:
                    query_window_start = row.get('query_window_start')
                    query_window_end = row.get('query_window_end')
                    target_window_start = row.get('target_window_start')
                    target_window_end = row.get('target_window_end')
                    query_windows_str = row.get('query_windows_json', '')
                    target_windows_str = row.get('target_windows_json', '')
                    query_windows_json = json.dumps(query_windows_str.split(';')) if query_windows_str else None
                    target_windows_json = json.dumps(target_windows_str.split(';')) if target_windows_str else None
                else:
                    query_window_start = query_window_end = None
                    target_window_start = target_window_end = None
                    query_windows_json = target_windows_json = None

                    if 'detailed_blocks' in locals() and block_id in detailed_blocks:
                        detailed_block = detailed_blocks[block_id]
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

            # Categorize block by size
            if length <= 3:
                block_type = 'small'
            elif length <= 10:
                block_type = 'medium'
            else:
                block_type = 'large'

            block_tuples.append((
                block_id, row.get('cluster_id', 0), query_locus, target_locus,
                query_genome_id, target_genome_id, query_contig_id, target_contig_id,
                length, identity, score,
                n_query_windows, n_target_windows,
                query_window_start, query_window_end, target_window_start, target_window_end,
                query_windows_json, target_windows_json, block_type
            ))

        # Batch insert all blocks
        if block_tuples:
            cursor.executemany("""
                INSERT INTO syntenic_blocks
                (block_id, cluster_id, query_locus, target_locus, query_genome_id, target_genome_id,
                 query_contig_id, target_contig_id, length, identity, score,
                 n_query_windows, n_target_windows, query_window_start, query_window_end,
                 target_window_start, target_window_end, query_windows_json, target_windows_json, block_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, block_tuples)

        self.conn.commit()
        logger.info(f"Ingested {len(df)} syntenic blocks with window information")
    
    def ingest_clusters(self, clusters_file: Path) -> None:
        """Ingest cluster information from CSV file.

        Supports both legacy format (consensus_length, consensus_score, diversity,
        representative_query, representative_target) and v2 chain pipeline format
        (genome_support, mean_chain_length, genes_json).

        Clears existing clusters to avoid duplicate inserts on reruns.
        """
        logger.info(f"Loading clusters from {clusters_file}")
        # Clear previous cluster table to avoid duplicates
        self.conn.execute("DELETE FROM clusters")
        df = pd.read_csv(clusters_file)
        cursor = self.conn.cursor()

        # Detect v2 format
        is_v2 = 'genome_support' in df.columns and 'consensus_length' not in df.columns

        # Collect tuples for batch insert
        cluster_tuples = []
        for _, row in df.iterrows():
            size = row['size']
            if size > 1000:
                cluster_type = 'large'
            elif size > 100:
                cluster_type = 'medium'
            else:
                cluster_type = 'small'

            if is_v2:
                consensus_length = int(row.get('mean_chain_length', 0))
                consensus_score = 0.0
                diversity = float(row.get('genome_support', 0))
                representative_query = ''
                representative_target = ''
            else:
                consensus_length = row['consensus_length']
                consensus_score = row['consensus_score']
                diversity = row['diversity']
                representative_query = row['representative_query']
                representative_target = row['representative_target']

            cluster_tuples.append((
                row['cluster_id'], size, consensus_length,
                consensus_score, diversity, representative_query,
                representative_target, cluster_type
            ))

        # Filter to only clusters that have corresponding syntenic_blocks entries
        # This prevents orphan clusters that would cause zero stats in the UI
        existing_cluster_ids = set(
            row[0] for row in self.conn.execute(
                "SELECT DISTINCT cluster_id FROM syntenic_blocks WHERE cluster_id > 0"
            )
        )
        original_count = len(cluster_tuples)
        cluster_tuples = [t for t in cluster_tuples if t[0] in existing_cluster_ids]
        filtered_count = original_count - len(cluster_tuples)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} clusters with no syntenic_blocks entries")

        # Batch insert all clusters
        if cluster_tuples:
            cursor.executemany("""
                INSERT INTO clusters
                (cluster_id, size, consensus_length, consensus_score, diversity,
                 representative_query, representative_target, cluster_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, cluster_tuples)

        self.conn.commit()
        logger.info(f"Ingested {len(cluster_tuples)} clusters (from {original_count} in CSV)")
    
    def create_gene_block_mappings(self) -> None:
        """Create precise mappings between genes and syntenic blocks.

        Preferred path: use bp-range metadata (query_start_bp / query_end_bp)
        stored in query_windows_json by the v2 chain pipeline. This finds DB
        genes whose coordinates overlap the anchor bp range, avoiding the
        gene-set mismatch between the pipeline parquet and the browser DB.

        Fallback: ordinal OFFSET/LIMIT on genes ordered by start_pos (legacy).
        """
        logger.info("Creating gene-block mappings...")

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM gene_block_mappings")

        cursor.execute("""
            SELECT block_id, query_genome_id, target_genome_id,
                   query_contig_id, target_contig_id,
                   query_window_start, query_window_end,
                   target_window_start, target_window_end,
                   query_windows_json, target_windows_json
            FROM syntenic_blocks
            WHERE query_window_start IS NOT NULL AND query_window_end IS NOT NULL
        """)
        blocks = cursor.fetchall()

        mappings = []
        blocks_without_windows = 0
        bp_mapped = 0

        cursor.execute("SELECT gene_id FROM genes")
        gene_id_set = {row[0] for row in cursor.fetchall()}

        def _normalize_gene_id(candidate: str) -> Optional[str]:
            if candidate in gene_id_set:
                return candidate
            parts = candidate.split('_')
            for idx in range(1, len(parts)):
                trimmed = '_'.join(parts[idx:])
                if trimmed in gene_id_set:
                    return trimmed
            return None

        def _genes_from_windows(json_blob: Optional[str]) -> list:
            if not json_blob:
                return []
            try:
                import json as _json
                entries = _json.loads(json_blob)
            except Exception:
                return []
            # New format: bp_meta dict — not a gene list
            if isinstance(entries, dict):
                return []
            genes = []
            for entry in entries:
                if isinstance(entry, str):
                    genes.extend([tok.strip() for tok in entry.split(',') if tok.strip()])
            seen = set()
            ordered = []
            for gid in genes:
                if gid not in seen:
                    seen.add(gid)
                    ordered.append(gid)
            return ordered

        def _parse_bp_meta(json_blob: Optional[str]) -> Optional[dict]:
            """Parse bp-range metadata from query_windows_json."""
            if not json_blob:
                return None
            try:
                import json as _json
                data = _json.loads(json_blob)
                if isinstance(data, dict) and 'query_start_bp' in data:
                    return data
            except Exception:
                pass
            return None

        def _genes_by_bp_range(genome_id, contig_id, start_bp, end_bp, role, block_id):
            """Find genes overlapping a bp range and add to mappings."""
            cursor.execute("""
                SELECT gene_id, start_pos FROM genes
                WHERE genome_id = ? AND contig_id = ?
                  AND end_pos >= ? AND start_pos <= ?
                ORDER BY start_pos
            """, (genome_id, contig_id, int(start_bp), int(end_bp)))
            genes = cursor.fetchall()
            for i, (gene_id, _) in enumerate(genes):
                relative_pos = i / max(1, len(genes) - 1) if len(genes) > 1 else 0.5
                mappings.append((gene_id, block_id, role, relative_pos))
            return len(genes)

        for block in blocks:
            (block_id, query_genome_id, target_genome_id,
             query_contig_id, target_contig_id,
             query_start, query_end, target_start, target_end,
             query_windows_json, target_windows_json) = block

            # Clean contig IDs
            if '_' in query_contig_id and '|' in query_contig_id:
                query_clean_contig = query_contig_id.split('_', 1)[1]
            else:
                query_clean_contig = query_contig_id
            if '_' in target_contig_id and '|' in target_contig_id:
                target_clean_contig = target_contig_id.split('_', 1)[1]
            else:
                target_clean_contig = target_contig_id

            # Try bp-range mapping first (v2 chain pipeline with coordinates)
            bp_meta = _parse_bp_meta(query_windows_json)
            if bp_meta:
                _genes_by_bp_range(query_genome_id, query_clean_contig,
                                   bp_meta['query_start_bp'], bp_meta['query_end_bp'],
                                   'query', block_id)
                _genes_by_bp_range(target_genome_id, target_clean_contig,
                                   bp_meta['target_start_bp'], bp_meta['target_end_bp'],
                                   'target', block_id)
                bp_mapped += 1
                continue

            # Legacy path: explicit gene lists from windows JSON
            query_gene_ids = _genes_from_windows(query_windows_json)
            if query_gene_ids:
                for i, gene_id in enumerate(query_gene_ids):
                    lookup_id = _normalize_gene_id(gene_id)
                    if not lookup_id:
                        continue
                    relative_pos = i / max(1, len(query_gene_ids) - 1) if len(query_gene_ids) > 1 else 0.5
                    mappings.append((lookup_id, block_id, 'query', relative_pos))
            elif query_start is not None and query_end is not None:
                gene_start_idx = int(query_start)
                gene_end_idx = int(query_end)
                num_genes = max(1, gene_end_idx - gene_start_idx + 1)
                cursor.execute("""
                    SELECT gene_id FROM genes
                    WHERE genome_id = ? AND contig_id = ?
                    ORDER BY start_pos LIMIT ? OFFSET ?
                """, (query_genome_id, query_clean_contig, num_genes, gene_start_idx))
                for i, (gene_id,) in enumerate(cursor.fetchall()):
                    relative_pos = i / max(1, num_genes - 1) if num_genes > 1 else 0.5
                    mappings.append((gene_id, block_id, 'query', relative_pos))
            else:
                blocks_without_windows += 1

            target_gene_ids = _genes_from_windows(target_windows_json)
            if target_gene_ids:
                for i, gene_id in enumerate(target_gene_ids):
                    lookup_id = _normalize_gene_id(gene_id)
                    if not lookup_id:
                        continue
                    relative_pos = i / max(1, len(target_gene_ids) - 1) if len(target_gene_ids) > 1 else 0.5
                    mappings.append((lookup_id, block_id, 'target', relative_pos))
            elif target_start is not None and target_end is not None:
                gene_start_idx = int(target_start)
                gene_end_idx = int(target_end)
                num_genes = max(1, gene_end_idx - gene_start_idx + 1)
                cursor.execute("""
                    SELECT gene_id FROM genes
                    WHERE genome_id = ? AND contig_id = ?
                    ORDER BY start_pos LIMIT ? OFFSET ?
                """, (target_genome_id, target_clean_contig, num_genes, gene_start_idx))
                for i, (gene_id,) in enumerate(cursor.fetchall()):
                    relative_pos = i / max(1, num_genes - 1) if num_genes > 1 else 0.5
                    mappings.append((gene_id, block_id, 'target', relative_pos))
            else:
                blocks_without_windows += 1

        if blocks_without_windows > 0:
            logger.warning(f"{blocks_without_windows} blocks lacked usable window information for gene mapping")
        if bp_mapped > 0:
            logger.info(f"{bp_mapped}/{len(blocks)} blocks mapped via bp-range coordinates")
        
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

    def compute_cluster_consensus(self, min_core_coverage: float = 0.6, df_percentile_ban: float = 0.9) -> None:
        """Pre-compute PFAM consensus cassette per cluster and persist summary for fast UI.

        Creates/updates table cluster_consensus with:
          - cluster_id INTEGER PRIMARY KEY
          - consensus_json TEXT
          - agree_frac REAL
          - core_tokens INTEGER
        """
        # Robust import for compute_cluster_pfam_consensus
        try:
            from genome_browser.database.cluster_content import compute_cluster_pfam_consensus
        except Exception:
            try:
                from database.cluster_content import compute_cluster_pfam_consensus
            except Exception:
                import importlib.util as _ilu
                from pathlib import Path as _Path
                _mod_path = _Path(__file__).resolve().parent / 'cluster_content.py'
                if not _mod_path.exists():
                    raise ImportError("Unable to locate cluster_content.py for cluster consensus computation")
                _spec = _ilu.spec_from_file_location('gb_cluster_content', str(_mod_path))
                _mod = _ilu.module_from_spec(_spec)
                assert _spec and _spec.loader
                _spec.loader.exec_module(_mod)
                compute_cluster_pfam_consensus = getattr(_mod, 'compute_cluster_pfam_consensus')

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_consensus (
                cluster_id INTEGER PRIMARY KEY,
                consensus_json TEXT,
                agree_frac REAL,
                core_tokens INTEGER,
                FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cluster_consensus_core ON cluster_consensus(core_tokens)")
        cur.execute("DELETE FROM cluster_consensus")
        self.conn.commit()

        cluster_ids = pd.read_sql_query("SELECT cluster_id FROM clusters ORDER BY cluster_id", self.conn)['cluster_id'].astype(int).tolist()
        logger.info("Computing cluster-level PFAM consensus for %d clusters…", len(cluster_ids))
        import json as _json
        rows = []
        for idx, cid in enumerate(cluster_ids, 1):
            try:
                payload = compute_cluster_pfam_consensus(self.conn, int(cid), float(min_core_coverage), float(df_percentile_ban), 0)
                tokens = payload.get('consensus', []) if isinstance(payload, dict) else []
                pairs = payload.get('pairs', []) if isinstance(payload, dict) else []
                supported = [p for p in pairs if p.get('same_frac') is not None and int(p.get('support', 0)) >= 3]
                if supported:
                    import statistics as _st
                    agree = _st.mean([float(p['same_frac']) for p in supported])
                else:
                    agree = 0.0
                rows.append((int(cid), _json.dumps(payload), float(agree), int(len(tokens))))
            except Exception as e:
                logger.debug("cluster %s consensus failed: %s", cid, e)
                rows.append((int(cid), '{}', 0.0, 0))
            if idx % 50 == 0:
                logger.info("Cluster consensus progress: %d/%d (%.1f%%)", idx, len(cluster_ids), 100.0*idx/len(cluster_ids))
        if rows:
            cur.executemany("INSERT INTO cluster_consensus (cluster_id, consensus_json, agree_frac, core_tokens) VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()
        logger.info("Cluster consensus table populated: %d rows", len(rows))
    
    def create_cluster_assignments(self, **kwargs) -> None:
        """Rebuild cluster summary from existing cluster_id assignments in syntenic_blocks.

        Legacy SRP mutual-Jaccard clustering was removed in ELSA v2.
        This method now only rebuilds the clusters summary table from
        cluster_id values already present in syntenic_blocks (loaded from CSV).
        """
        import pandas as _pd

        logger.info("Rebuilding cluster summary from existing assignments...")

        existing = _pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM syntenic_blocks WHERE cluster_id > 0", self.conn
        )['n'].iloc[0]

        if not existing or int(existing) == 0:
            logger.warning("No existing cluster_id assignments found; skipping cluster summary rebuild")
            return

        try:
            self.conn.execute("DELETE FROM clusters")
            agg = _pd.read_sql_query(
                """
                SELECT cluster_id, COUNT(*) AS size, CAST(AVG(length) AS INT) AS consensus_length,
                       AVG(identity) AS consensus_score
                FROM syntenic_blocks WHERE cluster_id > 0 GROUP BY cluster_id
                """,
                self.conn,
            )
            rep = _pd.read_sql_query(
                "SELECT cluster_id, block_id, query_locus, target_locus, score "
                "FROM syntenic_blocks WHERE cluster_id > 0 AND score IS NOT NULL",
                self.conn,
            )
            rep = rep.sort_values(['cluster_id', 'score'], ascending=[True, False]).groupby('cluster_id').head(1)
            merged = agg.merge(rep[['cluster_id', 'query_locus', 'target_locus']], on='cluster_id', how='left')
            merged = merged.fillna({'query_locus': '', 'target_locus': ''})
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
                        float(row.consensus_score or 0.0), 0.0,
                        str(row.query_locus), str(row.target_locus), 'unknown'
                    )
                )
            self.conn.commit()
            logger.info("Clusters summary rebuilt: %d rows", len(merged))
        except Exception as e:
            logger.warning("Failed to rebuild clusters summary: %s", e)
    
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
        # Precompute cluster-level consensus for fast explorer rendering
        ingester.compute_cluster_consensus(min_core_coverage=0.6, df_percentile_ban=0.9)
        ingester.generate_annotation_stats()
    
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()
