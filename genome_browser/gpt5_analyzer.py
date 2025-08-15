#!/usr/bin/env python3
"""
GPT-5 Analyzer for ELSA Genome Browser

Analyzes syntenic blocks by extracting locus information and sending it to GPT-5 
for comparative analysis and functional annotation interpretation.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import dspy
import os

logger = logging.getLogger(__name__)

@dataclass
class GeneInfo:
    """Information about a single gene."""
    gene_id: str
    genome_id: str
    contig_id: str
    start: int
    end: int
    strand: int
    length: int
    protein_sequence: str
    pfam_domains: str
    pfam_count: int
    gc_content: float

@dataclass
class LocusInfo:
    """Information about a genomic locus."""
    genome_id: str
    organism_name: str
    contig_id: str
    locus_start: int
    locus_end: int
    locus_length: int
    gene_count: int
    genes: List[GeneInfo]
    
    @property
    def functional_summary(self) -> str:
        """Generate a functional summary of the locus."""
        all_domains = []
        for gene in self.genes:
            if gene.pfam_domains:
                domains = gene.pfam_domains.split(';')
                all_domains.extend([d.strip() for d in domains if d.strip()])
        
        unique_domains = list(set(all_domains))
        return f"Contains {len(unique_domains)} unique PFAM domains: {', '.join(unique_domains[:10])}" + ("..." if len(unique_domains) > 10 else "")

@dataclass 
class SyntenicBlockAnalysis:
    """Complete analysis data for a syntenic block."""
    block_id: int
    query_locus: LocusInfo
    target_locus: LocusInfo
    block_length: int
    identity: float
    score: float
    alignment_details: Dict[str, Any]

class SyntenicBlockAnalyzer(dspy.Signature):
    """DSPy signature for GPT-5 syntenic block analysis."""
    
    block_summary = dspy.InputField(desc="Summary of the syntenic block including identity, score, and length")
    query_locus_data = dspy.InputField(desc="Detailed information about the query locus including genes and PFAM domains")
    target_locus_data = dspy.InputField(desc="Detailed information about the target locus including genes and PFAM domains")
    
    functional_conservation = dspy.OutputField(desc="Analysis of functional similarities based on PFAM domain conservation and gene orthologs")
    gene_organization = dspy.OutputField(desc="How gene organization (order, orientation) is conserved or different")
    functional_modules = dspy.OutputField(desc="Identifiable functional modules, operons, or biological processes")
    notable_features = dspy.OutputField(desc="Interesting domain architectures, unique genes, or potential HGT events")
    biological_significance = dspy.OutputField(desc="What this conservation tells us about the importance of this genomic region")

class GPT5Analyzer:
    """Analyzer that uses GPT-5 to interpret syntenic blocks."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Initialize GPT-5 with DSPy and LiteLLM
        self.lm = dspy.LM(
            "openai/gpt-5",
            temperature=1.0,
            max_tokens=20000
        )
        dspy.configure(lm=self.lm)
        
        # Initialize the signature
        self.analyzer = dspy.Predict(SyntenicBlockAnalyzer)
    
    def get_syntenic_block_analysis(self, block_id: int) -> Optional[SyntenicBlockAnalysis]:
        """Extract complete analysis data for a syntenic block."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get block information
            block_query = """
                SELECT sb.*, 
                       g1.organism_name as query_organism,
                       g2.organism_name as target_organism
                FROM syntenic_blocks sb
                LEFT JOIN genomes g1 ON sb.query_genome_id = g1.genome_id
                LEFT JOIN genomes g2 ON sb.target_genome_id = g2.genome_id
                WHERE sb.block_id = ?
            """
            
            cursor = conn.execute(block_query, (block_id,))
            block_row = cursor.fetchone()
            
            if not block_row:
                logger.error(f"Block {block_id} not found")
                return None
            
            # Extract block data
            columns = [desc[0] for desc in cursor.description]
            block_data = dict(zip(columns, block_row))
            
            # Get loci information
            query_locus = self._extract_locus_info(
                conn, block_data['query_genome_id'], block_data['query_locus'], 
                block_data['query_organism']
            )
            target_locus = self._extract_locus_info(
                conn, block_data['target_genome_id'], block_data['target_locus'],
                block_data['target_organism']
            )
            
            if not query_locus or not target_locus:
                logger.error(f"Could not extract locus information for block {block_id}")
                return None
            
            analysis = SyntenicBlockAnalysis(
                block_id=block_id,
                query_locus=query_locus,
                target_locus=target_locus,
                block_length=block_data['length'],
                identity=block_data['identity'],
                score=block_data['score'],
                alignment_details={
                    'query_windows': block_data.get('query_windows_json', '').split(';') if block_data.get('query_windows_json') else [],
                    'target_windows': block_data.get('target_windows_json', '').split(';') if block_data.get('target_windows_json') else [],
                    'window_start': block_data.get('query_window_start'),
                    'window_end': block_data.get('query_window_end')
                }
            )
            
            conn.close()
            return analysis
            
        except Exception as e:
            logger.error(f"Error extracting block analysis: {e}")
            return None
    
    def _extract_locus_info(self, conn: sqlite3.Connection, genome_id: str, 
                           locus_id: str, organism_name: str) -> Optional[LocusInfo]:
        """Extract detailed information about a genomic locus."""
        try:
            # Parse locus_id to get contig
            # Format: "genome_id:genome_id_contig_id" like "1313.30775:1313.30775_accn|1313.30775.con.0001"
            # Contig_id in genes table: "accn|1313.30775.con.0001"
            logger.info(f"Parsing locus_id: {locus_id}")
            if ':' in locus_id:
                _, full_contig_part = locus_id.split(':', 1)
                # full_contig_part is like "1313.30775_accn|1313.30775.con.0001"
                # We need to convert this to "accn|1313.30775.con.0001"
                if '_accn|' in full_contig_part:
                    # Split on "_accn|" and reconstruct as "accn|..."
                    parts = full_contig_part.split('_accn|')
                    contig_id = f"accn|{parts[1]}"
                elif '|' in full_contig_part:
                    # If there's a pipe, take everything after the last "_"
                    contig_id = full_contig_part.split('_', 1)[1] if '_' in full_contig_part else full_contig_part
                else:
                    contig_id = full_contig_part
            else:
                contig_id = locus_id
            
            logger.info(f"Extracted contig_id: {contig_id} for genome_id: {genome_id}")
            
            # Get all genes in this locus (contig)
            genes_query = """
                SELECT gene_id, genome_id, contig_id, start_pos, end_pos, strand,
                       gene_length, protein_sequence, pfam_domains, pfam_count, gc_content
                FROM genes 
                WHERE genome_id = ? AND contig_id = ?
                ORDER BY start_pos
            """
            
            cursor = conn.execute(genes_query, (genome_id, contig_id))
            gene_rows = cursor.fetchall()
            
            logger.info(f"Found {len(gene_rows)} genes for genome {genome_id}, contig {contig_id}")
            
            if not gene_rows:
                logger.warning(f"No genes found for locus {locus_id} (genome: {genome_id}, contig: {contig_id})")
                return None
            
            # Create gene info objects
            genes = []
            for row in gene_rows:
                gene = GeneInfo(
                    gene_id=row[0],
                    genome_id=row[1], 
                    contig_id=row[2],
                    start=row[3],  # start_pos from database
                    end=row[4],    # end_pos from database
                    strand=row[5],
                    length=row[6] or 0,
                    protein_sequence=row[7] or "",
                    pfam_domains=row[8] or "",
                    pfam_count=row[9] or 0,
                    gc_content=row[10] or 0.0
                )
                genes.append(gene)
            
            # Calculate locus boundaries
            locus_start = min(gene.start for gene in genes)
            locus_end = max(gene.end for gene in genes)
            locus_length = locus_end - locus_start + 1
            
            locus_info = LocusInfo(
                genome_id=genome_id,
                organism_name=organism_name,
                contig_id=contig_id,
                locus_start=locus_start,
                locus_end=locus_end,
                locus_length=locus_length,
                gene_count=len(genes),
                genes=genes
            )
            
            return locus_info
            
        except Exception as e:
            logger.error(f"Error extracting locus info: {e}")
            return None
    
    def generate_gpt_analysis(self, analysis: SyntenicBlockAnalysis) -> Optional[str]:
        """Generate GPT-5 analysis of the syntenic block using DSPy."""
        try:
            # Prepare structured data for GPT-5
            block_summary = f"Block ID: {analysis.block_id}, Identity: {analysis.identity:.1%}, Score: {analysis.score:.2f}, Length: {analysis.block_length} windows"
            
            query_data = self._format_locus_for_analysis(analysis.query_locus, "Query")
            target_data = self._format_locus_for_analysis(analysis.target_locus, "Target")
            
            # Call GPT-5 using DSPy signature
            result = self.analyzer(
                block_summary=block_summary,
                query_locus_data=query_data,
                target_locus_data=target_data
            )
            
            # Format the structured output into a comprehensive report
            report = f"""# GPT-5 Syntenic Block Analysis

## 1. Functional Conservation
{result.functional_conservation}

## 2. Gene Organization
{result.gene_organization}

## 3. Functional Modules
{result.functional_modules}

## 4. Notable Features
{result.notable_features}

## 5. Biological Significance
{result.biological_significance}
"""
            
            return report
                
        except Exception as e:
            logger.error(f"Error calling GPT-5: {e}")
            return f"GPT-5 analysis failed: {str(e)}"
    
    def _format_locus_for_analysis(self, locus: LocusInfo, label: str) -> str:
        """Format locus data for GPT-5 analysis."""
        summary = f"{label} Locus ({locus.organism_name}):\n"
        summary += f"Location: {locus.contig_id}:{locus.locus_start}-{locus.locus_end} ({locus.locus_length:,} bp)\n"
        summary += f"Gene count: {locus.gene_count}\n"
        summary += f"Functional overview: {locus.functional_summary}\n\n"
        
        summary += f"Genes in {label} locus:\n"
        for i, gene in enumerate(locus.genes, 1):
            strand_symbol = "+" if gene.strand >= 0 else "-"
            summary += f"{i}. {gene.gene_id} ({strand_symbol} strand, {gene.length} aa)\n"
            if gene.pfam_domains:
                pfam_summary = gene.pfam_domains[:200] + "..." if len(gene.pfam_domains) > 200 else gene.pfam_domains
                summary += f"   PFAM: {pfam_summary}\n"
            else:
                summary += f"   PFAM: None annotated\n"
        
        return summary

def analyze_syntenic_block(block_id: int, db_path: Path = Path("genome_browser.db")) -> Tuple[Optional[SyntenicBlockAnalysis], Optional[str]]:
    """Main function to analyze a syntenic block with GPT-5 using DSPy.
    
    Returns:
        Tuple of (analysis_data, gpt5_report)
    """
    analyzer = GPT5Analyzer(db_path)
    
    # Extract block analysis data
    analysis = analyzer.get_syntenic_block_analysis(block_id)
    if not analysis:
        return None, "Failed to extract block analysis data"
    
    # Generate GPT analysis
    gpt_report = analyzer.generate_gpt_analysis(analysis)
    
    return analysis, gpt_report

if __name__ == "__main__":
    # Test the analyzer
    test_block_id = 1
    analysis, report = analyze_syntenic_block(test_block_id)
    
    if analysis:
        print(f"Analysis for block {test_block_id}:")
        print(f"Query: {analysis.query_locus.organism_name} - {analysis.query_locus.gene_count} genes")
        print(f"Target: {analysis.target_locus.organism_name} - {analysis.target_locus.gene_count} genes")
        print(f"Identity: {analysis.identity:.1%}")
        
        if report:
            print("\nGPT Analysis:")
            print(report)
        else:
            print("No GPT analysis available")
    else:
        print(f"Failed to analyze block {test_block_id}")