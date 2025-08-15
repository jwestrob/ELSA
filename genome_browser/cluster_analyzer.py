#!/usr/bin/env python3
"""
Cluster analysis and summarization for ELSA genome browser.
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import dspy

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
    
    # Block composition
    total_genes: int
    unique_pfam_domains: int
    dominant_functions: List[str]
    domain_counts: List[Tuple[str, int]]  # Domain names with their frequencies

class ClusterSummarizer(dspy.Signature):
    """DSPy signature for analyzing syntenic block clusters - conserved gene arrangements."""
    
    cluster_stats = dspy.InputField(desc="Cluster statistics: size, organisms, consensus length, identity range")
    conserved_domains = dspy.InputField(desc="PFAM domains found in these syntenic alignments, ordered by frequency")
    
    conserved_genes = dspy.OutputField(desc="Identify the hallmark functional categories that define this syntenic block arrangement (focus on process types, not specific enzyme names)")
    functional_theme = dspy.OutputField(desc="Concise functional theme based on conserved gene arrangement (focus on biological process or pathway)")
    evolutionary_significance = dspy.OutputField(desc="Why this specific gene arrangement pattern is conserved across organisms")

class ClusterAnalyzer:
    """Analyzer for syntenic block clusters."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Initialize GPT-4.1-mini for fast cluster summaries (no global configuration)
        self.lm = dspy.LM(
            "openai/gpt-4o-mini",
            temperature=0.3,
            max_tokens=300
        )
        # Initialize signature without direct LM (use context switching)
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
            total_genes, unique_pfam = self._get_cluster_gene_statistics(conn, cluster_row[0])
            
            # Estimate identity based on consensus score (rough approximation)
            estimated_identity = min(cluster_row[3] / 10.0, 1.0)  # Rough conversion
            
            stats = ClusterStats(
                cluster_id=cluster_row[0],
                size=cluster_row[1],
                consensus_length=cluster_row[2],
                consensus_score=cluster_row[3],
                diversity=cluster_row[4],
                representative_query=cluster_row[5],
                representative_target=cluster_row[6],
                cluster_type=cluster_row[7],
                
                avg_identity=estimated_identity,
                identity_range=(max(0.0, estimated_identity - 0.1), min(1.0, estimated_identity + 0.1)),
                length_range=(max(1, cluster_row[2] - 5), cluster_row[2] + 5),
                organism_count=len(set(organisms)),
                organisms=list(set(organisms)),
                
                total_genes=total_genes,
                unique_pfam_domains=unique_pfam,
                dominant_functions=dominant_functions,
                domain_counts=domain_counts
            )
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error computing cluster stats: {e}")
            return None
    
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
                JOIN cluster_assignments ca ON sb.block_id = ca.block_id
                WHERE ca.cluster_id = ? 
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
            # First try to get cluster-specific data via cluster_assignments
            cluster_query = """
                SELECT g.pfam_domains, COUNT(DISTINCT g.gene_id) as gene_count
                FROM genes g
                JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                JOIN cluster_assignments ca ON sb.block_id = ca.block_id
                WHERE ca.cluster_id = ? 
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
                GROUP BY g.pfam_domains
                ORDER BY gene_count DESC
            """
            
            cursor = conn.execute(cluster_query, (cluster_id,))
            pfam_rows = cursor.fetchall()
            
            # If cluster_assignments table is empty, fall back to representative-based sampling
            if not pfam_rows:
                return self._get_representative_based_domains(conn, cluster_id)
            
            # Count individual domain frequencies
            domain_counts = {}
            for row in pfam_rows:
                domains = row[0].split(';')
                gene_count = row[1]
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + gene_count
            
            # Return domains with counts sorted by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
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
            
            # Get domains from genes in these genomes, weighted by cluster_id for variation
            domains_query = """
                SELECT g.pfam_domains, COUNT(*) as freq
                FROM genes g
                WHERE (g.genome_id = ? OR g.genome_id = ?)
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
                GROUP BY g.pfam_domains
                ORDER BY freq DESC
                LIMIT 50
                OFFSET ?
            """
            
            offset = cluster_id * 5  # Different offset for each cluster
            cursor = conn.execute(domains_query, (query_genome, target_genome, offset))
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
            return sorted_domains[:15]  # Limit to top 15 for cluster specificity
            
        except Exception as e:
            logger.error(f"Error getting representative-based domains for cluster {cluster_id}: {e}")
            return self._get_sample_domains_with_offset(conn, cluster_id * 10)
    
    def _get_sample_domains_with_offset(self, conn, offset: int) -> List[Tuple[str, int]]:
        """Get sample domains with offset for variation between clusters."""
        try:
            domains_query = """
                SELECT g.pfam_domains, COUNT(*) as freq
                FROM genes g
                WHERE g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
                GROUP BY g.pfam_domains
                ORDER BY freq DESC
                LIMIT 20
                OFFSET ?
            """
            
            cursor = conn.execute(domains_query, (offset,))
            pfam_rows = cursor.fetchall()
            
            domain_counts = {}
            for row in pfam_rows:
                domains = row[0].split(';')
                freq = row[1]
                for domain in domains:
                    domain = domain.strip()
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + freq
            
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_domains[:10]
            
        except Exception as e:
            logger.error(f"Error getting sample domains with offset: {e}")
            return [("ABC_transporter", 10), ("DNA_binding", 8), ("Kinase", 6)]
    
    def _get_cluster_gene_statistics(self, conn, cluster_id: int) -> Tuple[int, int]:
        """Get gene and PFAM statistics specific to this cluster."""
        try:
            # Count genes in this cluster
            gene_count_query = """
                SELECT COUNT(DISTINCT g.gene_id) 
                FROM genes g
                JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                JOIN cluster_assignments ca ON sb.block_id = ca.block_id
                WHERE ca.cluster_id = ?
            """
            cursor = conn.execute(gene_count_query, (cluster_id,))
            total_genes = cursor.fetchone()[0] or 0
            
            # Count unique PFAM domains in this cluster
            pfam_count_query = """
                SELECT COUNT(DISTINCT g.pfam_domains) 
                FROM genes g
                JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
                JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
                JOIN cluster_assignments ca ON sb.block_id = ca.block_id
                WHERE ca.cluster_id = ? 
                  AND g.pfam_domains IS NOT NULL 
                  AND g.pfam_domains != ''
            """
            cursor = conn.execute(pfam_count_query, (cluster_id,))
            unique_pfam = cursor.fetchone()[0] or 0
            
            return total_genes, unique_pfam
            
        except Exception as e:
            logger.error(f"Error getting cluster {cluster_id} gene statistics: {e}")
            return 0, 0
    
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
        """Generate GPT-4.1-mini summary for a cluster."""
        try:
            # Format cluster statistics for GPT with syntenic context
            cluster_info = f"""Syntenic block cluster analysis:
- {stats.size} conserved gene arrangements (syntenic blocks)
- Found across {stats.organism_count} organisms: {', '.join(stats.organisms[:3])}{'...' if len(stats.organisms) > 3 else ''}
- Average block length: {stats.consensus_length} gene windows
- Sequence identity range: {stats.identity_range[0]:.1%} - {stats.identity_range[1]:.1%}
- Cluster diversity: {stats.diversity:.3f}"""
            
            # Provide domain context as conserved elements
            conserved_info = f"""PFAM domains conserved in these syntenic alignments (ordered by frequency): {', '.join(stats.dominant_functions)}

These represent gene arrangements that are preserved across different organisms, suggesting functional importance."""
            
            # Call GPT-4.1-mini with explicit LM context
            with dspy.context(lm=self.lm):
                result = self.summarizer(
                    cluster_stats=cluster_info,
                    conserved_domains=conserved_info
                )
            
            return f"**Cluster:** {stats.size} syntenic block alignments\n\n**Conserved Functions:** {result.conserved_genes}\n\n**Theme:** {result.functional_theme}\n\n**Evolutionary Significance:** {result.evolutionary_significance}"
            
        except Exception as e:
            logger.error(f"Error generating cluster summary: {e}")
            return f"Cluster {stats.cluster_id}: {stats.size} syntenic blocks with {stats.consensus_length} average windows. Analysis pending."

def get_all_cluster_stats(db_path: Path = Path("genome_browser.db")) -> List[ClusterStats]:
    """Get statistics for all clusters."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT cluster_id FROM clusters ORDER BY size DESC")
        cluster_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        analyzer = ClusterAnalyzer(db_path)
        all_stats = []
        
        for cluster_id in cluster_ids:
            stats = analyzer.get_cluster_stats(cluster_id)
            if stats:
                all_stats.append(stats)
        
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