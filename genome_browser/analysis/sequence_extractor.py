#!/usr/bin/env python3
"""
Sequence extractor for syntenic block homology analysis.
Extracts protein sequences from syntenic blocks for MMseqs2 input.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProteinSequence:
    """Container for protein sequence data."""
    gene_id: str
    protein_sequence: str
    contig_id: str
    genome_id: str
    start_pos: int
    end_pos: int
    strand: str
    function: str


class SequenceExtractor:
    """Extract protein sequences from syntenic blocks for homology analysis."""
    
    def __init__(self, db_path: str = "genome_browser.db"):
        """Initialize sequence extractor.
        
        Args:
            db_path: Path to genome browser database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        logger.info(f"SequenceExtractor initialized with DB: {self.db_path}")
    
    def extract_block_sequences(self, block_id: int) -> Tuple[List[ProteinSequence], List[ProteinSequence]]:
        """Extract protein sequences for a syntenic block.
        
        Args:
            block_id: Syntenic block identifier
            
        Returns:
            Tuple of (query_proteins, target_proteins)
        """
        try:
            block_id = int(block_id)
            logger.info(f"Extracting sequences for syntenic block: {block_id}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Get block information
                block_info = self._get_block_info(conn, block_id)
                if not block_info:
                    raise ValueError(f"Block {block_id} not found")
                
                # Get genes for query and target contigs
                query_contig = block_info['query_contig_id']
                target_contig = block_info['target_contig_id']
                query_proteins = self._get_contig_proteins(conn, query_contig, 'query')
                target_proteins = self._get_contig_proteins(conn, target_contig, 'target')
                
                logger.info(f"Extracted {len(query_proteins)} query proteins, {len(target_proteins)} target proteins")
                
                # Validation
                self._validate_protein_sets(query_proteins, target_proteins, query_contig, target_contig)
                
                return query_proteins, target_proteins
                
        except Exception as e:
            logger.error(f"Failed to extract sequences for block {block_id}: {e}")
            raise
    
    def _get_block_info(self, conn: sqlite3.Connection, block_id: int) -> Optional[Dict]:
        """Get syntenic block metadata."""
        query = """
        SELECT block_id, query_locus, target_locus, query_genome_id, target_genome_id,
               query_contig_id, target_contig_id
        FROM syntenic_blocks 
        WHERE block_id = ?
        """
        
        cursor = conn.execute(query, (block_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def _get_contig_proteins(self, conn: sqlite3.Connection, contig_id: str, contig_type: str) -> List[ProteinSequence]:
        """Get protein sequences for a specific contig."""
        try:
            query = """
            SELECT gene_id, protein_sequence, contig_id, genome_id, start_pos, end_pos, strand, pfam_domains
            FROM genes 
            WHERE contig_id = ? AND protein_sequence IS NOT NULL AND LENGTH(protein_sequence) > 0
            ORDER BY start_pos
            """
            
            cursor = conn.execute(query, (contig_id,))
            rows = cursor.fetchall()
            
            if rows:
                columns = [desc[0] for desc in cursor.description]
                proteins = []
                for row in rows:
                    data = dict(zip(columns, row))
                    proteins.append(ProteinSequence(
                        gene_id=data['gene_id'],
                        protein_sequence=data['protein_sequence'],
                        contig_id=data['contig_id'],
                        genome_id=data['genome_id'],
                        start_pos=data['start_pos'],
                        end_pos=data['end_pos'],
                        strand=str(data['strand']),
                        function=data['pfam_domains'] or 'hypothetical protein'
                    ))
                
                logger.debug(f"Found {len(proteins)} proteins for contig {contig_id}")
                return proteins
            else:
                logger.warning(f"No proteins found for contig {contig_id}")
                return []
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get proteins for contig {contig_id}: {e}")
            return []
    
    def _get_locus_proteins(self, conn: sqlite3.Connection, locus: str, locus_type: str) -> List[ProteinSequence]:
        """Get protein sequences for a specific locus."""
        # Strategy: Find genes in the genomic region corresponding to this locus
        # We'll use multiple approaches to find relevant genes
        
        proteins = []
        
        # Approach 1: Direct locus match in gene tables
        proteins.extend(self._get_proteins_by_locus_direct(conn, locus, locus_type))
        
        # Approach 2: If no direct matches, try contig-based search
        if not proteins:
            proteins.extend(self._get_proteins_by_contig_region(conn, locus, locus_type))
        
        # Approach 3: If still no matches, try genome-based search with limits
        if not proteins:
            proteins.extend(self._get_proteins_by_genome_sampling(conn, locus, locus_type))
        
        return proteins
    
    def _get_proteins_by_locus_direct(self, conn: sqlite3.Connection, locus: str, locus_type: str) -> List[ProteinSequence]:
        """Try to find proteins directly associated with the locus."""
        queries = [
            # Try exact locus match
            """
            SELECT gene_id, protein_sequence, contig_id, genome_id, start_pos, end_pos, strand, pfam_domains
            FROM genes 
            WHERE contig_id = ? AND protein_sequence IS NOT NULL AND LENGTH(protein_sequence) > 0
            ORDER BY start_pos
            """,
            # Try locus as part of contig name
            """
            SELECT gene_id, protein_sequence, contig_id, genome_id, start_pos, end_pos, strand, pfam_domains
            FROM genes 
            WHERE contig_id LIKE ? AND protein_sequence IS NOT NULL AND LENGTH(protein_sequence) > 0
            ORDER BY start_pos
            LIMIT 20
            """
        ]
        
        for i, query in enumerate(queries):
            try:
                if i == 0:
                    cursor = conn.execute(query, (locus,))
                else:
                    cursor = conn.execute(query, (f"%{locus}%",))
                
                rows = cursor.fetchall()
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    proteins = []
                    for row in rows:
                        data = dict(zip(columns, row))
                        proteins.append(ProteinSequence(
                            gene_id=data['gene_id'],
                            protein_sequence=data['protein_sequence'],
                            contig_id=data['contig_id'],
                            genome_id=data['genome_id'],
                            start_pos=data['start_pos'],
                            end_pos=data['end_pos'],
                            strand=data['strand'],
                            function=data['pfam_domains'] or 'hypothetical protein'
                        ))
                    
                    logger.debug(f"Found {len(proteins)} proteins for {locus} using direct query {i+1}")
                    return proteins
                    
            except sqlite3.Error as e:
                logger.debug(f"Direct query {i+1} failed for {locus}: {e}")
                continue
        
        return []
    
    def _get_proteins_by_contig_region(self, conn: sqlite3.Connection, locus: str, locus_type: str) -> List[ProteinSequence]:
        """Try to find proteins by searching for similar contig names."""
        # Extract potential contig identifier from locus
        contig_parts = locus.replace('.', '_').split('_')
        
        for part in contig_parts:
            if len(part) > 5:  # Skip very short parts
                try:
                    query = """
                    SELECT gene_id, protein_sequence, contig_id, genome_id, start_pos, end_pos, strand, pfam_domains
                    FROM genes 
                    WHERE contig_id LIKE ? AND protein_sequence IS NOT NULL AND LENGTH(protein_sequence) > 0
                    ORDER BY start_pos
                    LIMIT 15
                    """
                    
                    cursor = conn.execute(query, (f"%{part}%",))
                    rows = cursor.fetchall()
                    
                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        proteins = []
                        for row in rows:
                            data = dict(zip(columns, row))
                            proteins.append(ProteinSequence(
                                gene_id=data['gene_id'],
                                protein_sequence=data['protein_sequence'],
                                contig_id=data['contig_id'],
                                genome_id=data['genome_id'],
                                start_pos=data['start_pos'],
                                end_pos=data['end_pos'],
                                strand=data['strand'],
                                function=data['pfam_domains'] or 'hypothetical protein'
                            ))
                        
                        logger.debug(f"Found {len(proteins)} proteins for {locus} using contig search with '{part}'")
                        return proteins
                        
                except sqlite3.Error as e:
                    logger.debug(f"Contig search failed for {part}: {e}")
                    continue
        
        return []
    
    def _get_proteins_by_genome_sampling(self, conn: sqlite3.Connection, locus: str, locus_type: str) -> List[ProteinSequence]:
        """Fallback: get sample proteins from any genome as a test."""
        try:
            query = """
            SELECT gene_id, protein_sequence, contig_id, genome_id, start_pos, end_pos, strand, pfam_domains
            FROM genes 
            WHERE protein_sequence IS NOT NULL AND LENGTH(protein_sequence) > 0
            ORDER BY RANDOM()
            LIMIT 10
            """
            
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            
            if rows:
                columns = [desc[0] for desc in cursor.description]
                proteins = []
                for row in rows:
                    data = dict(zip(columns, row))
                    proteins.append(ProteinSequence(
                        gene_id=f"sample_{data['gene_id']}_{locus_type}",  # Mark as sample
                        protein_sequence=data['protein_sequence'],
                        contig_id=data['contig_id'],
                        genome_id=data['genome_id'],
                        start_pos=data['start_pos'],
                        end_pos=data['end_pos'],
                        strand=data['strand'],
                        function=data['function'] or 'hypothetical protein'
                    ))
                
                logger.warning(f"Using {len(proteins)} sample proteins for {locus} (fallback method)")
                return proteins
        
        except sqlite3.Error as e:
            logger.error(f"Sample protein query failed: {e}")
        
        return []
    
    def _validate_protein_sets(self, query_proteins: List[ProteinSequence], 
                             target_proteins: List[ProteinSequence],
                             query_contig: str, target_contig: str) -> None:
        """Validate extracted protein sets."""
        
        # Check we have proteins
        if not query_proteins:
            raise ValueError(f"No query proteins found for contig: {query_contig}")
        if not target_proteins:
            raise ValueError(f"No target proteins found for contig: {target_contig}")
        
        # Check no overlap between sets (different contigs/genomes)
        query_gene_ids = {p.gene_id for p in query_proteins}
        target_gene_ids = {p.gene_id for p in target_proteins}
        
        overlap = query_gene_ids & target_gene_ids
        if overlap:
            raise ValueError(f"Gene ID overlap detected: {overlap}")
        
        # Validate protein sequences
        for protein_set, set_name in [(query_proteins, "query"), (target_proteins, "target")]:
            for protein in protein_set:
                if not protein.protein_sequence:
                    raise ValueError(f"Empty protein sequence for {protein.gene_id} in {set_name} set")
                
                # Check for valid protein characters
                valid_chars = set("ACDEFGHIKLMNPQRSTVWY*X-")
                seq_chars = set(protein.protein_sequence.upper())
                invalid_chars = seq_chars - valid_chars
                if invalid_chars:
                    raise ValueError(f"Invalid protein characters in {protein.gene_id}: {invalid_chars}")
        
        logger.info(f"Validation passed: {len(query_proteins)} query, {len(target_proteins)} target proteins")
    
    def write_fasta_files(self, query_proteins: List[ProteinSequence], 
                         target_proteins: List[ProteinSequence],
                         output_dir: Path) -> Tuple[Path, Path]:
        """Write protein sequences to FASTA files.
        
        Args:
            query_proteins: Query protein sequences
            target_proteins: Target protein sequences  
            output_dir: Directory for output files
            
        Returns:
            Tuple of (query_fasta_path, target_fasta_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        query_fasta = output_dir / "query_proteins.faa"
        target_fasta = output_dir / "target_proteins.faa"
        
        # Write query proteins
        with open(query_fasta, 'w') as f:
            for protein in query_proteins:
                header = f">{protein.gene_id} {protein.function} [{protein.genome_id}:{protein.contig_id}]"
                f.write(f"{header}\n")
                # Write sequence in 80-character lines
                seq = protein.protein_sequence
                for i in range(0, len(seq), 80):
                    f.write(f"{seq[i:i+80]}\n")
        
        # Write target proteins  
        with open(target_fasta, 'w') as f:
            for protein in target_proteins:
                header = f">{protein.gene_id} {protein.function} [{protein.genome_id}:{protein.contig_id}]"
                f.write(f"{header}\n")
                # Write sequence in 80-character lines
                seq = protein.protein_sequence
                for i in range(0, len(seq), 80):
                    f.write(f"{seq[i:i+80]}\n")
        
        logger.info(f"Wrote FASTA files: {query_fasta} ({len(query_proteins)} seqs), "
                   f"{target_fasta} ({len(target_proteins)} seqs)")
        
        return query_fasta, target_fasta


def test_sequence_extractor():
    """Test the sequence extractor with real data."""
    print("=== Testing SequenceExtractor ===")
    
    try:
        extractor = SequenceExtractor()
        
        # Test with known block
        block_id = 0
        print(f"Testing extraction for block {block_id}")
        
        query_proteins, target_proteins = extractor.extract_block_sequences(block_id)
        
        print(f"✓ Extracted {len(query_proteins)} query proteins")
        print(f"✓ Extracted {len(target_proteins)} target proteins")
        
        # Test FASTA writing
        output_dir = Path("test_homology_output")
        query_fasta, target_fasta = extractor.write_fasta_files(
            query_proteins, target_proteins, output_dir
        )
        
        print(f"✓ Wrote FASTA files:")
        print(f"  Query: {query_fasta}")
        print(f"  Target: {target_fasta}")
        
        # Validation tests
        print("✓ No gene ID overlap between sets")
        print("✓ All sequences have valid protein characters")
        print("✓ All sequences are non-empty")
        
        # Show sample data
        if query_proteins:
            sample = query_proteins[0]
            print(f"✓ Sample query protein: {sample.gene_id} ({len(sample.protein_sequence)} aa)")
        
        if target_proteins:
            sample = target_proteins[0]
            print(f"✓ Sample target protein: {sample.gene_id} ({len(sample.protein_sequence)} aa)")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sequence_extractor()