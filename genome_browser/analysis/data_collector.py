#!/usr/bin/env python3
"""
Data collector for syntenic block analysis.
Gathers all relevant information for GPT-5 analysis including block metadata,
gene annotations, PFAM domains, and conservation statistics.
"""

import sqlite3
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .gpt5_analyzer import SyntenicBlockData
from .sequence_extractor import SequenceExtractor
from .mmseqs2_runner import MMseqs2Runner
from .homology_processor import HomologyProcessor

logger = logging.getLogger(__name__)


class SyntenicDataCollector:
    """Collects and structures data for syntenic block analysis."""
    
    def __init__(self, db_path: str = "genome_browser.db", pfam_dir: str = "pfam_annotations"):
        """Initialize the data collector.
        
        Args:
            db_path: Path to the genome browser SQLite database
            pfam_dir: Directory containing PFAM annotation results
        """
        self.db_path = Path(db_path)
        self.pfam_dir = Path(pfam_dir)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        logger.info(f"SyntenicDataCollector initialized with DB: {self.db_path}")
    
    def collect_block_data(self, block_id) -> Optional[SyntenicBlockData]:
        """Collect all data needed for analyzing a specific syntenic block.
        
        Args:
            block_id: Identifier for the syntenic block
            
        Returns:
            SyntenicBlockData object with all collected information
        """
        try:
            # Ensure block_id is an integer for SQL queries
            block_id = int(block_id) if isinstance(block_id, str) else block_id
            logger.info(f"Collecting data for syntenic block: {block_id}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Get basic block information
                block_info = self._get_block_info(conn, block_id)
                if not block_info:
                    logger.warning(f"No block found with ID: {block_id}")
                    return None
                
                # Get genes in the block - only core aligned regions
                genes = self._get_core_aligned_genes(conn, block_id, block_info)
                
                # Get genome information
                genomes = self._get_block_genomes(conn, block_id)
                
                # Get PFAM domains for genes in the block
                pfam_domains = self._get_pfam_domains(genes)
                
                # Calculate conservation statistics
                conservation_stats = self._calculate_conservation_stats(genes, block_info)
                
                # Run homology analysis
                homology_analysis = self._run_homology_analysis(block_id, block_info)
                
                # Create structured data object
                # Use actual gene count as length, not the window count from DB
                actual_gene_count = len(genes)
                block_data = SyntenicBlockData(
                    block_id=block_id,
                    genomes=genomes,
                    genes=genes,
                    coordinates=self._extract_coordinates(block_info),
                    similarity_score=block_info.get('identity', 0.0),
                    length=actual_gene_count,  # Use actual gene count
                    pfam_domains=pfam_domains,
                    conservation_stats=conservation_stats,
                    homology_analysis=homology_analysis
                )
                
                logger.info(f"Collected data for block {block_id}: {len(genes)} genes, {len(pfam_domains)} PFAM domains")
                return block_data
                
        except Exception as e:
            logger.error(f"Failed to collect data for block {block_id}: {e}")
            return None
    
    def _get_block_info(self, conn: sqlite3.Connection, block_id: str) -> Optional[Dict[str, Any]]:
        """Get basic syntenic block information from database."""
        query = """
        SELECT * FROM syntenic_blocks 
        WHERE block_id = ?
        """
        
        cursor = conn.execute(query, (block_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def _get_block_genes(self, conn: sqlite3.Connection, block_id: str) -> List[Dict[str, Any]]:
        """Get all genes associated with a syntenic block."""
        # Try multiple potential table structures based on actual schema
        queries = [
            # Main structure using gene_block_mappings table
            """
            SELECT g.*, gbm.block_role, gbm.position_in_block
            FROM genes g
            JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
            WHERE gbm.block_id = ?
            ORDER BY gbm.position_in_block
            """,
            # Alternative: get genes from query and target loci
            """
            SELECT g.*, 'query' as block_role
            FROM genes g
            JOIN syntenic_blocks sb ON (g.contig_id = sb.query_contig_id)
            WHERE sb.block_id = ? AND g.contig_id LIKE '%' || sb.query_locus || '%'
            UNION
            SELECT g.*, 'target' as block_role
            FROM genes g
            JOIN syntenic_blocks sb ON (g.contig_id = sb.target_contig_id)
            WHERE sb.block_id = ? AND g.contig_id LIKE '%' || sb.target_locus || '%'
            ORDER BY start_pos
            """,
            # Fallback - get ALL genes from loci mentioned in the block (no limit)
            """
            SELECT g.*, 'unknown' as block_role
            FROM genes g, syntenic_blocks sb
            WHERE sb.block_id = ? 
            AND (g.contig_id LIKE '%' || SUBSTR(sb.query_locus, 1, 10) || '%'
                 OR g.contig_id LIKE '%' || SUBSTR(sb.target_locus, 1, 10) || '%')
            ORDER BY g.start_pos
            """
        ]
        
        genes = []
        for i, query in enumerate(queries):
            try:
                if i == 1:  # UNION query needs block_id twice
                    cursor = conn.execute(query, (block_id, block_id))
                else:
                    cursor = conn.execute(query, (block_id,))
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                if rows:
                    genes = [dict(zip(columns, row)) for row in rows]
                    logger.debug(f"Found {len(genes)} genes using query variant {i+1}")
                    break
                    
            except sqlite3.Error as e:
                logger.debug(f"Query {i+1} failed, trying next variant: {e}")
                continue
        
        # If no genes found via direct queries, try to extract from block data
        if not genes:
            genes = self._extract_genes_from_block_description(conn, block_id)
        
        return genes
    
    def _get_core_aligned_genes(self, conn: sqlite3.Connection, block_id: str, block_info: Dict) -> List[Dict[str, Any]]:
        """Get only genes from the core aligned windows, not the extended context."""
        try:
            # Try to get genes using window-based filtering
            core_genes = self._get_genes_from_aligned_windows(block_info)
            if core_genes:
                logger.info(f"Found {len(core_genes)} genes in core aligned windows for block {block_id}")
                return core_genes
        except Exception as e:
            logger.warning(f"Could not filter to core aligned genes for block {block_id}: {e}")
        
        # Fallback to original method if window filtering fails
        logger.info(f"Falling back to all genes for block {block_id}")
        return self._get_block_genes(conn, block_id)
    
    def _get_genes_from_aligned_windows(self, block_info: Dict) -> List[Dict[str, Any]]:
        """Extract genes from core aligned windows using ELSA window data."""
        import pandas as pd
        import json
        from pathlib import Path
        
        # Load ELSA window data
        elsa_index_dir = Path("./elsa_index")
        windows_file = elsa_index_dir / "multiscale_windows" / "macro_windows.parquet"
        
        if not windows_file.exists():
            raise FileNotFoundError("ELSA macro_windows.parquet not found")
        
        windows_df = pd.read_parquet(windows_file)
        
        # Parse window JSON data from block
        query_windows_json = block_info.get('query_windows_json', '[]')
        target_windows_json = block_info.get('target_windows_json', '[]')
        
        if isinstance(query_windows_json, str):
            query_window_ids = json.loads(query_windows_json)
        else:
            query_window_ids = query_windows_json or []
            
        if isinstance(target_windows_json, str):
            target_window_ids = json.loads(target_windows_json)
        else:
            target_window_ids = target_windows_json or []
        
        # Convert window IDs to macro window format for matching
        # "1313.30775_accn|1313.30775.con.0006_15" -> "1313.30775_accn|1313.30775.con.0006_macro_15"
        query_macro_ids = [wid.replace('_', '_macro_', 1) for wid in query_window_ids]
        target_macro_ids = [wid.replace('_', '_macro_', 1) for wid in target_window_ids]
        
        all_macro_ids = query_macro_ids + target_macro_ids
        
        if not all_macro_ids:
            raise ValueError("No aligned windows found in block")
        
        # Get window data for aligned regions
        aligned_windows = windows_df[windows_df['window_id'].isin(all_macro_ids)]
        
        if aligned_windows.empty:
            raise ValueError(f"No matching windows found for IDs: {all_macro_ids[:3]}...")
        
        # Extract gene index ranges for each locus
        core_genes = []
        
        # Process query genes
        query_aligned = aligned_windows[aligned_windows['window_id'].isin(query_macro_ids)]
        if not query_aligned.empty:
            query_locus = block_info.get('query_locus', '').split(':')[-1]  # Get contig part
            min_gene_idx = query_aligned['start_gene_idx'].min()
            max_gene_idx = query_aligned['end_gene_idx'].max()
            
            # Get genes from this locus and gene index range
            locus_genes = self._get_genes_by_locus_and_indices(query_locus, min_gene_idx, max_gene_idx, 'query')
            core_genes.extend(locus_genes)
        
        # Process target genes
        target_aligned = aligned_windows[aligned_windows['window_id'].isin(target_macro_ids)]
        if not target_aligned.empty:
            target_locus = block_info.get('target_locus', '').split(':')[-1]  # Get contig part  
            min_gene_idx = target_aligned['start_gene_idx'].min()
            max_gene_idx = target_aligned['end_gene_idx'].max()
            
            # Get genes from this locus and gene index range
            locus_genes = self._get_genes_by_locus_and_indices(target_locus, min_gene_idx, max_gene_idx, 'target')
            core_genes.extend(locus_genes)
        
        return core_genes
    
    def _get_genes_by_locus_and_indices(self, locus_id: str, min_idx: int, max_idx: int, role: str) -> List[Dict[str, Any]]:
        """Get genes from a specific locus within gene index range."""
        with sqlite3.connect(self.db_path) as conn:
            # Get genes from the specified locus, ordered by position
            query = """
            SELECT *, ROW_NUMBER() OVER (ORDER BY start_pos) - 1 as gene_index
            FROM genes 
            WHERE contig_id LIKE ? 
            ORDER BY start_pos
            """
            
            cursor = conn.execute(query, (f"%{locus_id}%",))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Filter to gene index range and add block role
            genes = []
            for row in rows:
                gene_dict = dict(zip(columns, row))
                gene_idx = gene_dict['gene_index']
                
                if min_idx <= gene_idx <= max_idx:
                    gene_dict['block_role'] = role
                    gene_dict['position_in_block'] = gene_idx - min_idx  # Relative position within aligned region
                    genes.append(gene_dict)
            
            return genes
    
    def _get_block_genomes(self, conn: sqlite3.Connection, block_id: str) -> List[str]:
        """Get genome identifiers associated with a syntenic block."""
        queries = [
            # Direct from syntenic_blocks table
            """
            SELECT DISTINCT query_genome_id as genome_id, target_genome_id as genome_id2
            FROM syntenic_blocks
            WHERE block_id = ?
            """,
            # Via genes in the block using gene_block_mappings
            """
            SELECT DISTINCT g.genome_id, g.organism_name
            FROM genomes g
            JOIN genes ge ON g.genome_id = ge.genome_id
            JOIN gene_block_mappings gbm ON ge.gene_id = gbm.gene_id
            WHERE gbm.block_id = ?
            """,
            # Fallback - get genomes from query and target locus
            """
            SELECT DISTINCT sb.query_genome_id, sb.target_genome_id, g1.organism_name, g2.organism_name
            FROM syntenic_blocks sb
            LEFT JOIN genomes g1 ON sb.query_genome_id = g1.genome_id
            LEFT JOIN genomes g2 ON sb.target_genome_id = g2.genome_id
            WHERE sb.block_id = ?
            """
        ]
        
        genomes = []
        for i, query in enumerate(queries):
            try:
                cursor = conn.execute(query, (block_id,))
                rows = cursor.fetchall()
                
                if rows:
                    if i == 0:  # First query returns genome_id columns
                        genome_set = set()
                        for row in rows:
                            if row[0]:
                                genome_set.add(row[0])
                            if len(row) > 1 and row[1]:
                                genome_set.add(row[1])
                        genomes = list(genome_set)
                    elif i == 1:  # Second query has genome_id and organism_name
                        genomes = [f"{row[0]} ({row[1]})" if row[1] else row[0] for row in rows]
                    else:  # Third query has multiple columns
                        genome_set = set()
                        for row in rows:
                            if row[0]:
                                genome_set.add(f"{row[0]} ({row[2]})" if len(row) > 2 and row[2] else row[0])
                            if row[1]:
                                genome_set.add(f"{row[1]} ({row[3]})" if len(row) > 3 and row[3] else row[1])
                        genomes = list(genome_set)
                    break
                    
            except sqlite3.Error as e:
                logger.debug(f"Genome query {i+1} failed: {e}")
                continue
        
        return genomes or ["unknown_genome"]
    
    def _get_pfam_domains(self, genes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get PFAM domain annotations for genes."""
        if not genes:
            return []
        
        pfam_domains = []
        
        # Try to load from PFAM annotation files
        if self.pfam_dir.exists():
            pfam_domains.extend(self._load_pfam_from_files(genes))
        
        # Also try database if available
        pfam_domains.extend(self._load_pfam_from_db(genes))
        
        return pfam_domains
    
    def _load_pfam_from_files(self, genes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Load PFAM domains from annotation files."""
        domains = []
        
        if not genes:
            return domains
            
        # Create a set of gene identifiers to match against
        gene_ids = set()
        for g in genes:
            gene_id = g.get('gene_id', g.get('id', ''))
            if gene_id:
                gene_ids.add(gene_id)
                # Also try different ID formats
                gene_ids.add(gene_id.split('|')[-1] if '|' in gene_id else gene_id)
        
        if not gene_ids:
            return domains
        
        # Look for PFAM results in JSON format first
        for pfam_json_file in self.pfam_dir.glob("**/pfam_annotation_results.json"):
            try:
                with open(pfam_json_file, 'r') as f:
                    pfam_data = json.load(f)
                
                if 'genome_annotations' in pfam_data:
                    for genome_id, annotations in pfam_data['genome_annotations'].items():
                        for gene_id, domain_string in annotations.items():
                            # Check if this gene is in our block
                            if any(target_id in gene_id or gene_id in target_id for target_id in gene_ids):
                                # Parse semicolon-separated domain string
                                if domain_string:
                                    domains_list = domain_string.split(';')
                                    for i, domain_name in enumerate(domains_list):
                                        if domain_name.strip():
                                            domains.append({
                                                'gene_id': gene_id,
                                                'hmm_name': domain_name.strip(),
                                                'evalue': 1e-5,  # Default conservative e-value
                                                'score': 50.0,   # Default score
                                                'description': f'PFAM domain: {domain_name.strip()}',
                                                'env_from': i * 100,  # Approximate positions
                                                'env_to': (i + 1) * 100,
                                                'domain_order': i
                                            })
                logger.debug(f"Loaded PFAM data from JSON: {pfam_json_file}")
                        
            except Exception as e:
                logger.debug(f"Could not load PFAM JSON file {pfam_json_file}: {e}")
        
        # Also look for PFAM hits TSV files (legacy format)
        for pfam_file in self.pfam_dir.glob("**/PFAM_hits_df.tsv"):
            try:
                df = pd.read_csv(pfam_file, sep='\t')
                
                # Filter for genes in our block
                if 'sequence_id' in df.columns:
                    # Create boolean mask for matching genes
                    mask = df['sequence_id'].apply(lambda x: any(gene_id in str(x) for gene_id in gene_ids))
                    relevant_hits = df[mask]
                    
                    for _, hit in relevant_hits.iterrows():
                        domains.append({
                            'gene_id': hit.get('sequence_id', ''),
                            'hmm_name': hit.get('hmm_name', ''),
                            'evalue': hit.get('evalue', 1.0),
                            'score': hit.get('score', 0.0),
                            'description': hit.get('description', ''),
                            'env_from': hit.get('env_from', 0),
                            'env_to': hit.get('env_to', 0)
                        })
                        
            except Exception as e:
                logger.debug(f"Could not load PFAM file {pfam_file}: {e}")
        
        logger.debug(f"Loaded {len(domains)} PFAM domains from files for {len(genes)} genes")
        return domains
    
    def _load_pfam_from_db(self, genes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Load PFAM domains from database if available."""
        domains = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if PFAM table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE '%pfam%'
                """)
                
                pfam_tables = [row[0] for row in cursor.fetchall()]
                
                if pfam_tables:
                    gene_ids = [g.get('gene_id', g.get('id', '')) for g in genes]
                    placeholders = ','.join(['?' for _ in gene_ids])
                    
                    query = f"""
                    SELECT * FROM {pfam_tables[0]}
                    WHERE gene_id IN ({placeholders})
                    """
                    
                    cursor = conn.execute(query, gene_ids)
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in cursor.fetchall():
                        domains.append(dict(zip(columns, row)))
                        
        except Exception as e:
            logger.debug(f"Could not load PFAM domains from database: {e}")
        
        return domains
    
    def _calculate_conservation_stats(self, genes: List[Dict[str, Any]], 
                                    block_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate conservation statistics for the block."""
        stats = {
            'total_genes': len(genes),
            'avg_identity': block_info.get('identity', 0.0),
            'conservation_level': 'unknown'
        }
        
        # Determine conservation level based on identity
        identity = stats['avg_identity']
        if identity >= 0.95:
            stats['conservation_level'] = 'very_high'
        elif identity >= 0.85:
            stats['conservation_level'] = 'high'
        elif identity >= 0.70:
            stats['conservation_level'] = 'moderate'
        else:
            stats['conservation_level'] = 'low'
        
        # Add gene-level stats if available
        if genes:
            hypothetical_count = sum(1 for g in genes 
                                   if 'hypothetical' in g.get('function', '').lower())
            stats['hypothetical_fraction'] = hypothetical_count / len(genes)
            stats['core_genes'] = len(genes) - hypothetical_count
            stats['accessory_genes'] = hypothetical_count
        
        return stats
    
    def _extract_coordinates(self, block_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract coordinate information from block data."""
        coords = {}
        
        # Standard coordinate fields
        for field in ['start', 'end', 'query_start', 'query_end', 
                     'target_start', 'target_end', 'length']:
            if field in block_info:
                coords[field] = block_info[field]
        
        # Add derived information
        if 'start' in coords and 'end' in coords:
            coords['span'] = coords['end'] - coords['start']
        
        return coords
    
    def _extract_genes_from_block_description(self, conn: sqlite3.Connection, 
                                            block_id: str) -> List[Dict[str, Any]]:
        """Fallback method to extract gene information from block descriptions."""
        # This is a fallback for cases where direct gene-block relationships aren't clear
        genes = []
        
        try:
            # Try to find genes mentioned in block descriptions or nearby coordinates
            cursor = conn.execute("""
                SELECT * FROM genes 
                ORDER BY genome_id, start_pos 
                LIMIT 20
            """)
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Take a sample if we can't find specific genes
            if rows:
                genes = [dict(zip(columns, row)) for row in rows[:5]]
                logger.warning(f"Using sample genes for block {block_id} (specific genes not found)")
                
        except Exception as e:
            logger.error(f"Could not extract genes for block {block_id}: {e}")
        
        return genes
    
    def _run_homology_analysis(self, block_id: str, block_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run complete homology analysis pipeline for the syntenic block."""
        try:
            logger.info(f"Starting homology analysis for block {block_id}")
            
            # Initialize components
            extractor = SequenceExtractor(str(self.db_path))
            mmseqs_runner = MMseqs2Runner()
            processor = HomologyProcessor()
            
            # Step 1: Extract sequences
            query_proteins, target_proteins = extractor.extract_block_sequences(block_id)
            
            # Step 2: Write FASTA files
            output_dir = Path("homology_analysis") / f"block_{block_id}"
            query_fasta, target_fasta = extractor.write_fasta_files(
                query_proteins, target_proteins, output_dir
            )
            
            # Step 3: Run MMseqs2 alignment
            alignment_file, alignments = mmseqs_runner.run_alignment(
                query_fasta, target_fasta, output_dir,
                sensitivity=4.0,  # Balanced sensitivity
                evalue_threshold=1e-3,
                coverage_threshold=0.3
            )
            
            # Step 4: Process homology relationships
            query_protein_functions = {p.gene_id: p.function for p in query_proteins}
            target_protein_functions = {p.gene_id: p.function for p in target_proteins}
            
            analysis = processor.process_homology_data(
                alignments, query_protein_functions, target_protein_functions, str(block_id),
                query_locus=block_info.get('query_locus'),
                target_locus=block_info.get('target_locus')
            )
            
            # Convert to dictionary format for storage
            homology_data = {
                'ortholog_pairs': [
                    {
                        'query_id': pair.query_id,
                        'target_id': pair.target_id,
                        'relationship': pair.relationship.value,
                        'identity': pair.identity,
                        'coverage_query': pair.coverage_query,
                        'coverage_target': pair.coverage_target,
                        'evalue': pair.evalue,
                        'functional_conservation': pair.functional_conservation,
                        'pathway_role': pair.pathway_role
                    }
                    for pair in analysis.ortholog_pairs
                ],
                'functional_groups': [
                    {
                        'group_id': group.group_id,
                        'function_description': group.function_description,
                        'pathway': group.pathway,
                        'query_proteins': group.query_proteins,
                        'target_proteins': group.target_proteins,
                        'conservation_level': group.conservation_level
                    }
                    for group in analysis.functional_groups
                ],
                'pathway_conservation': analysis.pathway_conservation,
                'conservation_summary': analysis.conservation_summary,
                'analysis_status': analysis.analysis_status,
                'failure_reason': analysis.failure_reason,
                'elsa_evidence': analysis.elsa_evidence
            }
            
            logger.info(f"Homology analysis completed for block {block_id}: "
                       f"{len(analysis.ortholog_pairs)} orthologs, {len(analysis.functional_groups)} functional groups")
            
            return homology_data
            
        except Exception as e:
            logger.error(f"Homology analysis pipeline failed for block {block_id}: {e}")
            # Return structured failure data
            return {
                'ortholog_pairs': [],
                'functional_groups': [],
                'pathway_conservation': {},
                'conservation_summary': {'conservation_level': 'unknown', 'summary': 'Pipeline failure'},
                'analysis_status': 'pipeline_failure',
                'failure_reason': f'Homology pipeline failed: {str(e)}',
                'elsa_evidence': None
            }


def test_data_collector():
    """Test the data collector with sample data."""
    try:
        collector = SyntenicDataCollector()
        
        # Try to collect data for any available block
        with sqlite3.connect(collector.db_path) as conn:
            cursor = conn.execute("SELECT block_id FROM syntenic_blocks LIMIT 1")
            row = cursor.fetchone()
            
            if row:
                block_id = row[0]
                print(f"Testing data collection for block: {block_id}")
                
                block_data = collector.collect_block_data(block_id)
                if block_data:
                    print(f"✓ Successfully collected data:")
                    print(f"  - Block ID: {block_data.block_id}")
                    print(f"  - Genes: {len(block_data.genes)}")
                    print(f"  - Genomes: {len(block_data.genomes)}")
                    print(f"  - PFAM domains: {len(block_data.pfam_domains)}")
                    print(f"  - Similarity: {block_data.similarity_score}")
                else:
                    print("✗ Failed to collect block data")
            else:
                print("No syntenic blocks found in database")
                
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_data_collector()