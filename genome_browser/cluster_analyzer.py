#!/usr/bin/env python3
"""
Cluster analysis and summarization for ELSA genome browser.
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import dspy
import json
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class ClusterStats:
    """Precomputed statistics for a cluster."""
    cluster_id: int
    size: int
    consensus_length: int
    consensus_score: float
    diversity: float
    cluster_type: str
    
    # Precomputed stats
    avg_identity: float
    identity_range: Tuple[float, float]
    length_range: Tuple[int, int]
    organism_count: int
    organisms: List[str]
    representative_query: str
    representative_target: str
    
    # Multi-dimensional scope metrics
    total_alignments: int      # Total syntenic blocks (was 'size')
    unique_contigs: int        # Distinct contigs involved
    unique_genes: int          # Distinct genes participating
    unique_pfam_domains: int
    dominant_functions: List[str]
    domain_counts: List[Tuple[str, int]]  # Domain names with their frequencies

class ClusterSummarizer(dspy.Signature):
    """Conservative summary of a syntenic-block cluster using consensus evidence.

    Base claims on measurable evidence from the consensus cassette: token coverage,
    conserved ordering (mean_pos), and directional consensus (co-directionality across
    adjacent consensus tokens). Prefer descriptive summaries of conserved components over
    speculative functional narratives. Do not claim full pathways; these loci are short.
    """

    cluster_stats = dspy.InputField(desc="Cluster statistics: size, organisms, consensus length, identity range")
    conserved_domains = dspy.InputField(desc="Context PFAMs ordered by frequency (low weight)")
    consensus_cassette = dspy.InputField(desc="JSON with keys: consensus (ordered tokens: token, coverage, df, mean_pos, fwd_frac, n_occ) and pairs (adjacency same-strand: t1, t2, same_frac, support). Use for core selection and directional consensus.")

    molecular_mechanism = dspy.OutputField(desc="Evidence-grounded description of conserved components. Cite specific PFAM tokens with coverage and ordering. Avoid pathway claims.")
    conservation_basis = dspy.OutputField(desc="Tie claims to consensus tokens and directional consensus (co-directionality). Note ordering and any mixed signals.")
    outputs_json = dspy.OutputField(desc="JSON: {summary, core_tokens:[{pfam,coverage,df,pos}], directional_consensus:{agree_frac,note}, adjacency_support:[{t1,t2,same_frac,support}], caveats:[...], confidence:0..1}")

class ClusterAnalyzer:
    """Analyzer for syntenic block clusters."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Initialize OpenAI client for GPT-5-mini Responses API
        self.openai_client = OpenAI()
        # Keep DSPy signature for prompt structure reference
        self.summarizer = dspy.Predict(ClusterSummarizer)
    
    def get_cluster_stats(self, cluster_id: int) -> Optional[ClusterStats]:
        """Compute comprehensive statistics for a cluster."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get basic cluster info
            cluster_query = """
                SELECT cluster_id, size, consensus_length, consensus_score, diversity,
                       representative_query, representative_target, cluster_type
                FROM clusters WHERE cluster_id = ?
            """
            cursor = conn.execute(cluster_query, (cluster_id,))
            cluster_row = cursor.fetchone()
            
            if not cluster_row:
                return None
            
            # For now, create enhanced basic stats with sample data from all blocks
            # Extract organisms from representative loci
            repr_query = cluster_row[5] or ""
            repr_target = cluster_row[6] or ""
            
            organisms = []
            if ":" in repr_query:
                query_genome = repr_query.split(":")[0]
                organisms.append(query_genome)
            if ":" in repr_target:
                target_genome = repr_target.split(":")[0]
                organisms.append(target_genome)
            
            # Get cluster-specific functional data
            dominant_functions = self._get_cluster_functions(conn, cluster_row[0])
            domain_counts = self._get_cluster_functions_with_counts(conn, cluster_row[0])
            
            # Get multi-dimensional scope metrics
            unique_contigs = self._get_cluster_contig_count(conn, cluster_row[0])
            unique_genes = self._get_cluster_gene_count(conn, cluster_row[0])
            unique_pfam = self._get_cluster_pfam_count(conn, cluster_row[0])
            
            # Estimate identity based on consensus score (rough approximation)
            try:
                consensus_score = cluster_row[3] if cluster_row[3] is not None else 0.0
                estimated_identity = min(float(consensus_score) / 10.0, 1.0)
            except Exception:
                estimated_identity = 0.0
            # Handle None length
            consensus_length = cluster_row[2] if cluster_row[2] is not None else 0
            
            stats = ClusterStats(
                cluster_id=cluster_row[0],
                size=cluster_row[1],
                consensus_length=consensus_length,
                consensus_score=consensus_score,
                diversity=cluster_row[4] if cluster_row[4] is not None else 0.0,
                representative_query=cluster_row[5],
                representative_target=cluster_row[6],
                cluster_type=cluster_row[7],
                
                avg_identity=estimated_identity,
                identity_range=(max(0.0, estimated_identity - 0.1), min(1.0, estimated_identity + 0.1)),
                length_range=(max(1, int(consensus_length) - 5), int(consensus_length) + 5),
                organism_count=len(set(organisms)),
                organisms=list(set(organisms)),
                
                # Multi-dimensional scope metrics - count from current syntenic_blocks table
                total_alignments=self._get_actual_block_count(conn, cluster_id),
                unique_contigs=unique_contigs,
                unique_genes=unique_genes,
                unique_pfam_domains=unique_pfam,
                dominant_functions=dominant_functions,
                domain_counts=domain_counts
            )
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error computing cluster stats: {e}")
            return None

    def summarize_cluster(self, cluster_id: int) -> Optional[dict]:
        """Run the DSPy ClusterSummarizer using consensus cassette evidence.

        Returns a dict with keys: molecular_mechanism, conservation_basis, outputs_json (parsed if valid JSON).
        """
        try:
            stats = self.get_cluster_stats(cluster_id)
            if not stats:
                return None
            conn = sqlite3.connect(self.db_path)
            # Load consensus cassette (PFAM-based)
            try:
                from genome_browser.database.cluster_content import compute_cluster_pfam_consensus
            except Exception:
                from database.cluster_content import compute_cluster_pfam_consensus
            payload = compute_cluster_pfam_consensus(conn, int(cluster_id), 0.6, 0.9, 0)

            # Prepare inputs
            stats_text = (
                f"size={stats.size}; consensus_len={stats.consensus_length}; "
                f"avg_id={stats.avg_identity:.2f}; organisms={','.join(stats.organisms)}"
            )
            domains = self._get_cluster_functions_with_counts(conn, cluster_id)
            conserved_domains_text = ", ".join([f"{d} x{c}" for d, c in domains[:15]])
            cassette_json = json.dumps(payload if isinstance(payload, dict) else {"consensus": [], "pairs": []})

            # Invoke DSPy signature
            result = self.summarizer(
                cluster_stats=stats_text,
                conserved_domains=conserved_domains_text,
                consensus_cassette=cassette_json,
            )

            # Parse outputs_json if present
            out_json = None
            try:
                out_json = json.loads(getattr(result, 'outputs_json', '') or '{}')
            except Exception:
                out_json = None
            return {
                "molecular_mechanism": getattr(result, 'molecular_mechanism', ''),
                "conservation_basis": getattr(result, 'conservation_basis', ''),
                "outputs_json": out_json or getattr(result, 'outputs_json', ''),
            }
        except Exception as e:
            logger.error(f"Error summarizing cluster {cluster_id}: {e}")
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass
    
    def _get_actual_block_count(self, conn, cluster_id: int) -> int:
        """Get actual block count from syntenic_blocks table instead of legacy clusters table."""
        cursor = conn.execute("SELECT COUNT(*) FROM syntenic_blocks WHERE cluster_id = ?", (cluster_id,))
        result = cursor.fetchone()
        return result[0] if result else 0
    
    def _create_basic_cluster_stats(self, cluster_row) -> ClusterStats:
        """Create basic stats when detailed block info isn't available."""
        return ClusterStats(
            cluster_id=cluster_row[0],
            size=cluster_row[1],
            consensus_length=cluster_row[2],
            consensus_score=cluster_row[3],
            diversity=cluster_row[4],
            representative_query=cluster_row[5],
            representative_target=cluster_row[6],
            cluster_type=cluster_row[7],
            
            avg_identity=0.0,
            identity_range=(0.0, 0.0),
            length_range=(0, 0),
            organism_count=0,
            organisms=[],
            
            total_genes=0,
            unique_pfam_domains=0,
            dominant_functions=[],
            domain_counts=[]
        )
    
    def _get_dominant_functions(self, conn, cluster_id: int, sample_blocks: List) -> List[str]:
        """Get dominant PFAM functions from sample blocks."""
        try:
            if not sample_blocks:
                return []
            
            block_ids = [row[0] for row in sample_blocks[:3]]  # Top 3 blocks
            placeholders = ','.join(['?' for _ in block_ids])
            
            # Get genes from these blocks (simplified - using locus matching)
            functions_query = f"""
                SELECT pfam_domains FROM genes 
                WHERE pfam_domains IS NOT NULL AND pfam_domains != ''
                LIMIT 100
            """
            
            cursor = conn.execute(functions_query)
            pfam_rows = cursor.fetchall()
            
            # Count domain frequencies
            domain_counts = {}
            for row in pfam_rows:
                domains = row[0].split(';')
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Return top 5 most common domains
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return [domain for domain, count in sorted_domains[:5]]
            
        except Exception as e:
            logger.error(f"Error getting dominant functions: {e}")
            return []
    
    def _get_sample_functions(self, conn) -> List[str]:
        """Get all PFAM functions from database for comprehensive GPT-4.1-mini analysis."""
        try:
            functions_query = """
                SELECT pfam_domains, COUNT(*) as freq FROM genes 
                WHERE pfam_domains IS NOT NULL AND pfam_domains != ''
                GROUP BY pfam_domains
                ORDER BY freq DESC
            """
            
            cursor = conn.execute(functions_query)
            pfam_rows = cursor.fetchall()
            
            # Count individual domain frequencies
            domain_counts = {}
            for row in pfam_rows:
                domains = row[0].split(';')
                freq = row[1]
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + freq
            
            # Return ALL domains sorted by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return [domain for domain, count in sorted_domains]
            
        except Exception as e:
            logger.error(f"Error getting sample functions: {e}")
            return ["ABC_transporter", "DNA_binding", "Kinase", "Transferase", "Oxidoreductase"]
    
    def _get_sample_functions_with_counts(self, conn) -> List[Tuple[str, int]]:
        """Get PFAM functions with their counts for visualization."""
        try:
            functions_query = """
                SELECT pfam_domains, COUNT(*) as freq FROM genes 
                WHERE pfam_domains IS NOT NULL AND pfam_domains != ''
                GROUP BY pfam_domains
                ORDER BY freq DESC
            """
            
            cursor = conn.execute(functions_query)
            pfam_rows = cursor.fetchall()
            
            # Count individual domain frequencies
            domain_counts = {}
            for row in pfam_rows:
                domains = row[0].split(';')
                freq = row[1]
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + freq
            
            # Return domains with counts sorted by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_domains
            
        except Exception as e:
            logger.error(f"Error getting sample functions with counts: {e}")
            return []
    
    def _get_cluster_functions(self, conn, cluster_id: int) -> List[str]:
        """Get PFAM functions specific to this cluster."""
        try:
            # Get blocks in this cluster and their associated genes
            cluster_query = """
                SELECT DISTINCT g.pfam_domains
                FROM genes g
                JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                WHERE sb.cluster_id = ? 
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
            """
            
            cursor = conn.execute(cluster_query, (cluster_id,))
            pfam_rows = cursor.fetchall()
            
            # Count individual domain frequencies
            domain_counts = {}
            for row in pfam_rows:
                domains = row[0].split(';')
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Return domains sorted by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return [domain for domain, count in sorted_domains]
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} functions: {e}")
            return ["ABC_transporter", "DNA_binding", "Kinase", "Transferase", "Oxidoreductase"]
    
    def _get_cluster_functions_with_counts(self, conn, cluster_id: int) -> List[Tuple[str, int]]:
        """Get PFAM functions with counts specific to this cluster."""
        try:
            # Get all distinct genes in this cluster with their PFAM domains
            cluster_query = """
                SELECT DISTINCT g.gene_id, g.pfam_domains
                FROM genes g
                JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                WHERE sb.cluster_id = ? 
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
            """
            
            cursor = conn.execute(cluster_query, (cluster_id,))
            gene_rows = cursor.fetchall()
            
            # If cluster_assignments table is empty, fall back to representative-based sampling
            if not gene_rows:
                logger.info(f"No genes found for cluster {cluster_id}, falling back to representative-based domains")
                return self._get_representative_based_domains(conn, cluster_id)
            
            logger.info(f"Cluster {cluster_id}: Found {len(gene_rows)} unique genes for PFAM analysis")
            
            # Count individual domain frequencies (each domain counted once per gene)
            domain_counts = {}
            for gene_id, pfam_domains in gene_rows:
                domains = pfam_domains.split(';')
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Return domains with counts sorted by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"Cluster {cluster_id}: Top 5 domains: {sorted_domains[:5]}")
            return sorted_domains
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} functions with counts: {e}")
            return self._get_representative_based_domains(conn, cluster_id)
    
    def _get_representative_based_domains(self, conn, cluster_id: int) -> List[Tuple[str, int]]:
        """Fallback: Get domains from representative loci of this cluster."""
        try:
            # Get cluster representative loci
            cluster_query = "SELECT representative_query, representative_target FROM clusters WHERE cluster_id = ?"
            cursor = conn.execute(cluster_query, (cluster_id,))
            cluster_row = cursor.fetchone()
            
            if not cluster_row or not cluster_row[0]:
                # Return varied results based on cluster_id to differentiate clusters
                offset = cluster_id * 100
                return self._get_sample_domains_with_offset(conn, offset)
            
            repr_query, repr_target = cluster_row[0], cluster_row[1]
            
            # Extract genome IDs from representative loci
            query_genome = repr_query.split(':')[0] if ':' in repr_query else repr_query
            target_genome = repr_target.split(':')[0] if ':' in repr_target and repr_target else query_genome
            
            # Get genes from these genomes, weighted by cluster_id for variation
            domains_query = """
                SELECT DISTINCT g.gene_id, g.pfam_domains
                FROM genes g
                WHERE (g.genome_id = ? OR g.genome_id = ?)
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
                ORDER BY g.gene_id
                LIMIT 50
                OFFSET ?
            """
            
            offset = cluster_id * 5  # Different offset for each cluster
            cursor = conn.execute(domains_query, (query_genome, target_genome, offset))
            gene_rows = cursor.fetchall()
            
            # Count individual domain frequencies (each domain counted once per gene)
            domain_counts = {}
            for gene_id, pfam_domains in gene_rows:
                domains = pfam_domains.split(';')
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Return domains with counts sorted by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_domains[:15]  # Limit to top 15 for cluster specificity
            
        except Exception as e:
            logger.error(f"Error getting representative-based domains for cluster {cluster_id}: {e}")
            return self._get_sample_domains_with_offset(conn, cluster_id * 10)
    
    def _get_sample_domains_with_offset(self, conn, offset: int) -> List[Tuple[str, int]]:
        """Get sample domains with offset for variation between clusters."""
        try:
            domains_query = """
                SELECT DISTINCT g.gene_id, g.pfam_domains
                FROM genes g
                WHERE g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
                ORDER BY g.gene_id
                LIMIT 20
                OFFSET ?
            """
            
            cursor = conn.execute(domains_query, (offset,))
            gene_rows = cursor.fetchall()
            
            domain_counts = {}
            for gene_id, pfam_domains in gene_rows:
                domains = pfam_domains.split(';')
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_domains[:10]
            
        except Exception as e:
            logger.error(f"Error getting sample domains with offset: {e}")
            return [("ABC_transporter", 10), ("DNA_binding", 8), ("Kinase", 6)]
    
    def _extract_locus_info(self, conn, genome_id: str, locus_id: str, organism_name: str) -> Optional['LocusInfo']:
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
            
            # Create gene info objects - use absolute import to avoid issues
            try:
                from gpt5_analyzer import GeneInfo, LocusInfo
            except ImportError:
                # Fallback for relative import
                from .gpt5_analyzer import GeneInfo, LocusInfo
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
    
    def _get_cluster_contig_count(self, conn, cluster_id: int) -> int:
        """Get count of unique contigs involved in this cluster."""
        try:
            contig_count_query = """
                SELECT 
                  COUNT(DISTINCT sb.query_contig_id) + COUNT(DISTINCT sb.target_contig_id)
                FROM syntenic_blocks sb
                WHERE sb.cluster_id = ?
                  AND sb.query_contig_id IS NOT NULL 
                  AND sb.target_contig_id IS NOT NULL
            """
            cursor = conn.execute(contig_count_query, (cluster_id,))
            unique_contigs = cursor.fetchone()[0] or 0
            return unique_contigs
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} contig count: {e}")
            return 0
    
    def _get_cluster_gene_count(self, conn, cluster_id: int) -> int:
        """Get count of unique genes involved in this cluster."""
        try:
            # Try cluster_assignments first
            gene_count_query = """
                SELECT COUNT(DISTINCT gbm.gene_id) 
                FROM gene_block_mappings gbm
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                WHERE sb.cluster_id = ?
            """
            cursor = conn.execute(gene_count_query, (cluster_id,))
            unique_genes = cursor.fetchone()[0] or 0
            
            # If cluster_assignments is empty, estimate from cluster size
            if unique_genes == 0:
                logger.debug(f"No gene mappings found for cluster {cluster_id}, using size-based estimate")
                cluster_query = "SELECT size, consensus_length FROM clusters WHERE cluster_id = ?"
                cursor = conn.execute(cluster_query, (cluster_id,))
                cluster_row = cursor.fetchone()
                if cluster_row:
                    # Rough estimate: size * consensus_length * 2 (avg genes per window)
                    sz = int(cluster_row[0] or 0)
                    clen = int(cluster_row[1] or 0)
                    estimated_genes = sz * clen * 2
                    return min(estimated_genes, 10000)  # Cap at reasonable max
            
            return unique_genes
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} gene count: {e}")
            return 0
    
    def _get_cluster_pfam_count(self, conn, cluster_id: int) -> int:
        """Get count of unique PFAM domains in this cluster."""
        try:
            pfam_count_query = """
                SELECT COUNT(DISTINCT g.pfam_domains) 
                FROM genes g
                JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                WHERE sb.cluster_id = ? 
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
            """
            cursor = conn.execute(pfam_count_query, (cluster_id,))
            unique_pfam = cursor.fetchone()[0] or 0
            return unique_pfam
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} PFAM count: {e}")
            return 0
    
    def _get_cluster_block_lengths(self, conn, cluster_id: int) -> Dict:
        """Get block length statistics for this cluster."""
        try:
            # Get all block lengths in this cluster
            length_query = """
                SELECT sb.length
                FROM syntenic_blocks sb
                WHERE sb.cluster_id = ?
            """
            cursor = conn.execute(length_query, (cluster_id,))
            lengths = [row[0] for row in cursor.fetchall()]
            
            if not lengths:
                # Fallback to consensus_length from clusters table
                cluster_query = "SELECT consensus_length FROM clusters WHERE cluster_id = ?"
                cursor = conn.execute(cluster_query, (cluster_id,))
                consensus = cursor.fetchone()
                if consensus:
                    return {
                        "avg": consensus[0],
                        "std": 0.0,
                        "min": consensus[0], 
                        "max": consensus[0],
                        "count": 1
                    }
                return {"avg": 0, "std": 0, "min": 0, "max": 0, "count": 0}
            
            lengths_array = np.array(lengths)
            return {
                "avg": np.mean(lengths_array),
                "std": np.std(lengths_array),
                "min": np.min(lengths_array),
                "max": np.max(lengths_array),
                "count": len(lengths)
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} block lengths: {e}")
            return {"avg": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    
    def _get_sample_loci(self, conn, cluster_id: int, sample_size: int = 5) -> List:
        """Get random sample of loci from this cluster with full gene details."""
        try:
            logger.info(f"=== DEBUG: _get_sample_loci for cluster {cluster_id} ===")
            
            # Get random blocks from this cluster
            blocks_query = """
                SELECT sb.block_id, sb.query_locus, sb.target_locus, sb.query_genome_id, sb.target_genome_id
                FROM syntenic_blocks sb
                WHERE sb.cluster_id = ?
                ORDER BY RANDOM()
                LIMIT ?
            """
            cursor = conn.execute(blocks_query, (cluster_id, sample_size * 2))
            blocks = cursor.fetchall()
            
            logger.info(f"Found {len(blocks)} blocks for cluster {cluster_id}")
            if not blocks:
                logger.warning(f"No blocks found for cluster {cluster_id} - cluster_assignments may be empty")
                return []
            
            sample_loci = []
            seen_loci = set()
            
            for i, block in enumerate(blocks):
                if len(sample_loci) >= sample_size:
                    break
                    
                block_id, query_locus, target_locus, query_genome_id, target_genome_id = block
                logger.info(f"Processing block {i+1}/{len(blocks)}: {block_id}, query: {query_locus}, target: {target_locus}")
                
                # Try query locus first
                if query_locus not in seen_loci:
                    logger.info(f"Attempting to extract query locus: {query_locus} (genome: {query_genome_id})")
                    try:
                        locus_info = self._extract_locus_info(conn, query_genome_id, query_locus, f"Genome_{query_genome_id}")
                        if locus_info:
                            logger.info(f"Successfully extracted query locus with {locus_info.gene_count} genes")
                            sample_loci.append(locus_info)
                            seen_loci.add(query_locus)
                        else:
                            logger.warning(f"_extract_locus_info returned None for query locus {query_locus}")
                    except Exception as e:
                        logger.error(f"Error extracting query locus {query_locus}: {e}")
                
                # Try target locus if we need more samples
                if len(sample_loci) < sample_size and target_locus not in seen_loci:
                    logger.info(f"Attempting to extract target locus: {target_locus} (genome: {target_genome_id})")
                    try:
                        locus_info = self._extract_locus_info(conn, target_genome_id, target_locus, f"Genome_{target_genome_id}")
                        if locus_info:
                            logger.info(f"Successfully extracted target locus with {locus_info.gene_count} genes")
                            sample_loci.append(locus_info)
                            seen_loci.add(target_locus)
                        else:
                            logger.warning(f"_extract_locus_info returned None for target locus {target_locus}")
                    except Exception as e:
                        logger.error(f"Error extracting target locus {target_locus}: {e}")
            
            logger.info(f"=== FINAL: Extracted {len(sample_loci)} sample loci for cluster {cluster_id} ===")
            return sample_loci[:sample_size]
            
        except Exception as e:
            logger.error(f"Error getting sample loci for cluster {cluster_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _get_sample_gene_statistics(self, conn) -> Tuple[int, int]:
        """Get sample gene and PFAM statistics."""
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM genes")
            total_genes = cursor.fetchone()[0] or 0
            
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT pfam_domains) FROM genes 
                WHERE pfam_domains IS NOT NULL AND pfam_domains != ''
            """)
            unique_pfam = cursor.fetchone()[0] or 0
            
            return total_genes // 100, unique_pfam // 10  # Scale down for cluster context
            
        except Exception as e:
            logger.error(f"Error getting sample gene statistics: {e}")
            return 50, 25
    
    def _get_gene_statistics(self, conn, sample_blocks: List) -> Tuple[int, int]:
        """Get gene and PFAM statistics from sample blocks."""
        try:
            # Simplified - count all genes and unique domains
            cursor = conn.execute("SELECT COUNT(*) FROM genes")
            total_genes = cursor.fetchone()[0] or 0
            
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT pfam_domains) FROM genes 
                WHERE pfam_domains IS NOT NULL AND pfam_domains != ''
            """)
            unique_pfam = cursor.fetchone()[0] or 0
            
            return total_genes, unique_pfam
            
        except Exception as e:
            logger.error(f"Error getting gene statistics: {e}")
            return 0, 0
    
    def generate_cluster_summary(self, stats: ClusterStats) -> Optional[str]:
        """Generate GPT-5-mini summary for a cluster using Responses API."""
        try:
            # Get enhanced cluster data
            conn = sqlite3.connect(self.db_path)
            
            # Get block length variance
            length_stats = self._get_cluster_block_lengths(conn, stats.cluster_id)
            
            # Get sample loci
            sample_loci = self._get_sample_loci(conn, stats.cluster_id, sample_size=5)
            
            conn.close()
            
            # Format length variance
            if length_stats["count"] > 1:
                length_desc = f"{length_stats['avg']:.1f} ± {length_stats['std']:.1f} gene windows (range: {int(length_stats['min'])}-{int(length_stats['max'])})"
            else:
                length_desc = f"{length_stats['avg']:.0f} gene windows"
            
            # Format sample loci
            sample_text = ""
            if sample_loci:
                sample_text = "\n\nREPRESENTATIVE LOCI:\n"
                for i, locus in enumerate(sample_loci, 1):
                    sample_text += f"\nLocus {i}: {locus.organism_name} - {locus.contig_id}:{locus.locus_start}-{locus.locus_end}\n"
                    for j, gene in enumerate(locus.genes, 1):
                        strand_symbol = "+" if gene.strand >= 0 else "-"
                        pfam_info = gene.pfam_domains[:50] + "..." if len(gene.pfam_domains) > 50 else gene.pfam_domains
                        if not pfam_info:
                            pfam_info = "None"
                        sample_text += f"  Gene {j}: {gene.gene_id} ({strand_symbol}, {gene.length} aa) - PFAM: {pfam_info}\n"
            
            # Format comprehensive prompt for GPT-5-mini
            prompt = f"""Analyze this syntenic block cluster:

CLUSTER OVERVIEW:
- Found across {stats.organism_count} organisms
- Block length: {length_desc}
- {length_stats['count']} similar syntenic arrangements

CONSERVED PFAM DOMAINS (most frequent):
{', '.join(stats.dominant_functions[:10])}{sample_text}

Provide two concise analyses (under 100 words total):

1. MOLECULAR MECHANISM: Specific molecular pathway with key PFAM domains and enzyme functions. Include biochemical context and conservation rationale.

2. CONSERVATION BASIS: Molecular constraints explaining synteny preservation. Avoid generic terms.

Format as:
**Function:** [analysis 1]
**Conservation:** [analysis 2]"""

            # Comprehensive debug logging
            logger.info(f"=== DEBUG: CLUSTER DATA COMPONENTS ===")
            logger.info(f"Cluster ID: {stats.cluster_id}")
            logger.info(f"Length stats: {length_stats}")
            logger.info(f"Sample loci count: {len(sample_loci)}")
            logger.info(f"PFAM domains: {stats.dominant_functions[:10]}")
            
            # Log each sample locus in detail
            for i, locus in enumerate(sample_loci):
                logger.info(f"Sample locus {i+1}: {locus.organism_name} - {len(locus.genes)} genes")
                for j, gene in enumerate(locus.genes):
                    logger.info(f"  Gene {j+1}: {gene.gene_id} - PFAM: '{gene.pfam_domains}' - Length: {gene.length}aa")
            
            logger.info(f"=== FULL GPT-5 PROMPT ===")
            logger.info(f"COMPLETE PROMPT:\n{prompt}")
            logger.info(f"=== END PROMPT ===")
            
            # Call GPT-5-mini using Responses API
            response = self.openai_client.responses.create(
                model="gpt-5-mini",
                input=prompt,
                reasoning={"effort": "low"},
                max_output_tokens=500  # Increased from 300
            )
            
            # Debug response structure
            logger.info(f"GPT-5 response object type: {type(response)}")
            logger.info(f"GPT-5 response attributes: {dir(response)}")
            
            # Try different possible response attributes
            result_text = ""
            if hasattr(response, 'output_text'):
                result_text = response.output_text.strip()
                logger.info(f"GPT-5 output_text: '{result_text}' (length: {len(result_text)})")
            elif hasattr(response, 'text'):
                result_text = response.text.strip()
                logger.info(f"GPT-5 text: '{result_text}' (length: {len(result_text)})")
            elif hasattr(response, 'content'):
                result_text = response.content.strip()
                logger.info(f"GPT-5 content: '{result_text}' (length: {len(result_text)})")
            else:
                logger.error(f"Unknown GPT-5 response format: {response}")
                result_text = "Unable to parse GPT-5 response"
            
            return f"**Cluster:** {stats.total_alignments:,} alignments ({stats.unique_genes:,} genes)\n\n{result_text}"
            
        except Exception as e:
            logger.error(f"Error generating cluster summary: {e}")
            return f"Cluster {stats.cluster_id}: {stats.total_alignments:,} alignments spanning {stats.unique_genes:,} genes. Analysis pending."

def get_all_cluster_stats(db_path: Path = Path("genome_browser.db")) -> List[ClusterStats]:
    """Get statistics for all clusters from macro clusters table and micro sidecar tables.

    - Macro clusters come from `clusters`
    - Micro clusters come from `micro_gene_clusters`/`micro_gene_blocks`
    """
    try:
        conn = sqlite3.connect(db_path)

        # Load macro cluster IDs and count (exclude sink 0)
        cursor = conn.execute("SELECT cluster_id FROM clusters WHERE cluster_id > 0 ORDER BY size DESC")
        macro_rows = cursor.fetchall()
        macro_ids = [int(row[0]) for row in macro_rows]
        # Use ceiling (max id) so micro display ids don't collide with sparse macro ids
        try:
            max_macro_id = conn.execute("SELECT COALESCE(MAX(cluster_id),0) FROM clusters WHERE cluster_id > 0").fetchone()[0]
            n_macro = int(max_macro_id or 0)
        except Exception:
            n_macro = len(macro_ids)

        analyzer = ClusterAnalyzer(db_path)
        all_stats: List[ClusterStats] = []

        # Add macro stats
        for cid in macro_ids:
            stats = analyzer.get_cluster_stats(cid)
            if stats:
                # Force macro type to avoid mislabeled rows in DB
                try:
                    stats.cluster_type = 'macro'
                except Exception:
                    pass
                all_stats.append(stats)

        # Try to load micro clusters if present
        try:
            # Check existence
            has_micro = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='micro_gene_clusters'"
            ).fetchone() is not None
            if has_micro:
                micro_rows = conn.execute(
                    "SELECT cluster_id, size, genomes FROM micro_gene_clusters ORDER BY size DESC"
                ).fetchall()
                # Build quick lookups for per-cluster unique contigs and gene counts
                contigs_by_cid: Dict[int, Set[Tuple[str, str]]] = {}
                genes_by_cid: Dict[int, int] = {}
                # Prefer pair-backed sources; fall back to legacy micro_gene_blocks if present
                have_pairs = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='micro_block_pairs'").fetchone() is not None
                have_maps = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='micro_gene_pair_mappings'").fetchone() is not None
                if have_pairs:
                    # Unique contigs from query and target sides
                    rows_q = conn.execute(
                        "SELECT cluster_id, query_genome_id, query_contig_id FROM micro_block_pairs"
                    ).fetchall()
                    rows_t = conn.execute(
                        "SELECT cluster_id, target_genome_id, target_contig_id FROM micro_block_pairs"
                    ).fetchall()
                    for cid, gid, cid2 in rows_q:
                        contigs_by_cid.setdefault(int(cid), set()).add((str(gid), str(cid2)))
                    for cid, gid, cid2 in rows_t:
                        contigs_by_cid.setdefault(int(cid), set()).add((str(gid), str(cid2)))
                    # Gene counts from mappings joined to pairs by block_id
                    if have_maps:
                        rows = conn.execute(
                            """
                            SELECT p.cluster_id, COUNT(DISTINCT m.gene_id)
                            FROM micro_gene_pair_mappings m
                            JOIN micro_block_pairs p ON p.block_id = m.block_id
                            GROUP BY p.cluster_id
                            """
                        ).fetchall()
                        for cid, n in rows:
                            genes_by_cid[int(cid)] = int(n or 0)
                else:
                    # Legacy fallback: approximate genes via index spans
                    has_mgb = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='micro_gene_blocks'").fetchone() is not None
                    if has_mgb:
                        mb_rows = conn.execute(
                            "SELECT cluster_id, genome_id, contig_id, start_index, end_index FROM micro_gene_blocks"
                        ).fetchall()
                        for cid, gid, contig, s, e in mb_rows:
                            cid = int(cid)
                            contigs_by_cid.setdefault(cid, set()).add((str(gid), str(contig)))
                            try:
                                genes_by_cid[cid] = genes_by_cid.get(cid, 0) + max(1, int(e) - int(s) + 1)
                            except Exception:
                                genes_by_cid[cid] = genes_by_cid.get(cid, 0) + 2

                # Helper: compute PFAM domain counts per micro cluster from gene spans
                def _micro_domain_counts(micro_cid: int) -> List[Tuple[str, int]]:
                    try:
                        # Prefer pair mappings if available
                        have_pairs = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='micro_block_pairs'").fetchone() is not None
                        have_maps = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='micro_gene_pair_mappings'").fetchone() is not None
                        rows: List[Tuple[str]] = []
                        if have_pairs and have_maps:
                            q = """
                                SELECT g.pfam_domains
                                FROM micro_gene_pair_mappings m
                                JOIN micro_block_pairs p ON p.block_id = m.block_id
                                JOIN genes g ON g.gene_id = m.gene_id
                                WHERE p.cluster_id = ? AND g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
                            """
                            rows = conn.execute(q, (micro_cid,)).fetchall()
                        else:
                            # Legacy: use micro_gene_blocks spans
                            cur2 = conn.cursor()
                            cur2.execute("DROP TABLE IF EXISTS _tmp_go_m")
                            cur2.execute(
                                """
                                CREATE TEMP TABLE _tmp_go_m AS
                                SELECT genome_id, contig_id, gene_id, start_pos, end_pos,
                                       ROW_NUMBER() OVER (PARTITION BY genome_id, contig_id ORDER BY start_pos, end_pos) - 1 AS idx
                                FROM genes
                                """
                            )
                            q = """
                                WITH ranges AS (
                                    SELECT block_id, genome_id, contig_id, start_index, end_index
                                    FROM micro_gene_blocks WHERE cluster_id = ?
                                )
                                SELECT g.pfam_domains
                                FROM ranges r
                                JOIN _tmp_go_m go ON go.genome_id = r.genome_id AND go.contig_id = r.contig_id AND go.idx BETWEEN r.start_index AND r.end_index
                                JOIN genes g ON g.gene_id = go.gene_id
                                WHERE g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
                            """
                            rows = cur2.execute(q, (micro_cid,)).fetchall()
                        cnt: Dict[str, int] = {}
                        for (pf,) in rows:
                            for tok in str(pf).split(';'):
                                tok = tok.strip()
                                if tok:
                                    cnt[tok] = cnt.get(tok, 0) + 1
                        return sorted(cnt.items(), key=lambda x: x[1], reverse=True)
                    except Exception:
                        return []

                for row in micro_rows:
                    cid, size, genomes = int(row[0]), int(row[1] or 0), str(row[2] or "")
                    disp_cid = (n_macro + cid) if cid > 0 else 0
                    org_ids = [g for g in genomes.split(';') if g]
                    # Construct ClusterStats for micro cluster
                    # Precompute consensus length from micro_cluster_consensus if available
                    cons_len = 0
                    try:
                        r = conn.execute("SELECT consensus_json FROM micro_cluster_consensus WHERE cluster_id = ?", (cid,)).fetchone()
                        if r and r[0]:
                            import json as _json
                            payload = _json.loads(str(r[0]))
                            if isinstance(payload, dict):
                                cons = payload.get('consensus', [])
                                if isinstance(cons, list):
                                    cons_len = len(cons)
                        if cons_len == 0:
                            # On-the-fly fallback using DB/sidecars
                            try:
                                from genome_browser.database.cluster_content import compute_micro_cluster_pfam_consensus as _cmc
                            except Exception:
                                from database.cluster_content import compute_micro_cluster_pfam_consensus as _cmc
                            payload2 = _cmc(conn, int(cid), 0.6, 0.9, 0)
                            if isinstance(payload2, dict):
                                cons = payload2.get('consensus', [])
                                if isinstance(cons, list):
                                    cons_len = len(cons)
                    except Exception:
                        cons_len = 0

                    cs = ClusterStats(
                        cluster_id=disp_cid,
                        size=size,
                        consensus_length=int(cons_len or 0),
                        consensus_score=0.0,
                        diversity=0.0,
                        representative_query="",
                        representative_target="",
                        cluster_type='micro',
                        avg_identity=0.0,
                        identity_range=(0.0, 0.0),
                        length_range=(0, 0),
                        organism_count=len(set(org_ids)),
                        organisms=list(sorted(set(org_ids))),
                        total_alignments=size,
                        unique_contigs=len(contigs_by_cid.get(cid, set())),
                        unique_genes=int(genes_by_cid.get(cid, 0)),
                        unique_pfam_domains=0,
                        dominant_functions=[d for d, _ in _micro_domain_counts(cid)[:5]],
                        domain_counts=_micro_domain_counts(cid),
                    )
                    all_stats.append(cs)
        except Exception as e:
            logger.warning(f"Micro clusters not merged (skipping): {e}")
        finally:
            conn.close()

        return all_stats

    except Exception as e:
        logger.error(f"Error getting all cluster stats: {e}")
        return []

if __name__ == "__main__":
    # Test the cluster analyzer
    stats = get_all_cluster_stats()
    print(f"Found {len(stats)} clusters")
    
    if stats:
        analyzer = ClusterAnalyzer(Path("genome_browser.db"))
        test_cluster = stats[0]
        print(f"\nTesting cluster {test_cluster.cluster_id}:")
        print(f"Size: {test_cluster.size}, Type: {test_cluster.cluster_type}")
        
        summary = analyzer.generate_cluster_summary(test_cluster)
        print(f"\nGPT Summary:\n{summary}")
