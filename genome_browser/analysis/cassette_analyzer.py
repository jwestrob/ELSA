#!/usr/bin/env python3
"""
Conserved Cassette Analysis for identifying functionally similar genomic regions.
Detects regions with shared PFAM domain architecture between genomic loci.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import re

@dataclass
class ConservedCassette:
    """Represents a conserved functional cassette between two genomic regions."""
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    shared_domains: List[str]
    domain_conservation_score: float
    synteny_score: float
    query_genes: List[str]
    target_genes: List[str]
    cassette_type: str  # 'perfect', 'rearranged', 'partial'

class CassetteAnalyzer:
    """Analyzes conserved functional cassettes between genomic loci."""
    
    def __init__(self):
        self.min_cassette_size = 2  # Minimum genes in a cassette (must be at least 2)
        self.require_exact_domain_match = True  # Only exact domain architecture matches
        self.require_synteny = True  # Require adjacent homologous proteins
    
    def find_conserved_cassettes(self, query_genes: pd.DataFrame, 
                                target_genes: pd.DataFrame) -> List[ConservedCassette]:
        """
        Find conserved functional cassettes between two genomic loci.
        
        Args:
            query_genes: DataFrame with query locus gene information
            target_genes: DataFrame with target locus gene information
            
        Returns:
            List of ConservedCassette objects representing functional similarities
        """
        if query_genes.empty or target_genes.empty:
            return []
        
        # Extract domain profiles for both loci
        query_profile = self._extract_domain_profile(query_genes)
        target_profile = self._extract_domain_profile(target_genes)
        
        if not query_profile or not target_profile:
            return []
        
        # Find conserved cassettes using sliding windows
        cassettes = []
        
        # Try different window sizes
        for window_size in range(self.min_cassette_size, min(len(query_profile), len(target_profile)) + 1):
            cassettes.extend(self._find_cassettes_by_window(
                query_genes, target_genes, query_profile, target_profile, window_size
            ))
        
        # Merge overlapping cassettes and rank by quality
        merged_cassettes = self._merge_overlapping_cassettes(cassettes)
        return sorted(merged_cassettes, key=lambda c: c.domain_conservation_score, reverse=True)
    
    def _extract_domain_profile(self, genes_df: pd.DataFrame) -> List[Dict]:
        """Extract ordered domain profile from genes."""
        profile = []
        
        for _, gene in genes_df.iterrows():
            gene_domains = []
            if pd.notna(gene['pfam_domains']) and gene['pfam_domains']:
                domains = gene['pfam_domains'].split(';')
                gene_domains = [d.strip() for d in domains if d.strip()]
            
            profile.append({
                'gene_id': gene['gene_id'],
                'start_pos': gene['start_pos'],
                'end_pos': gene['end_pos'],
                'strand': gene['strand'],
                'domains': gene_domains,
                'domain_set': set(gene_domains)
            })
        
        return profile
    
    def _find_cassettes_by_window(self, query_genes: pd.DataFrame, target_genes: pd.DataFrame,
                                 query_profile: List[Dict], target_profile: List[Dict],
                                 window_size: int) -> List[ConservedCassette]:
        """Find cassettes using sliding window approach."""
        cassettes = []
        
        # Slide window across query locus
        for q_start in range(len(query_profile) - window_size + 1):
            q_end = q_start + window_size
            query_window = query_profile[q_start:q_end]
            
            # Slide window across target locus
            for t_start in range(len(target_profile) - window_size + 1):
                t_end = t_start + window_size
                target_window = target_profile[t_start:t_end]
                
                # Analyze domain conservation between windows
                cassette = self._analyze_window_pair(query_window, target_window)
                if cassette and cassette.domain_conservation_score >= 0.95:  # Very strict threshold for exact matches
                    cassettes.append(cassette)
        
        return cassettes
    
    def _analyze_window_pair(self, query_window: List[Dict], 
                            target_window: List[Dict]) -> Optional[ConservedCassette]:
        """Analyze a pair of genomic windows for exact domain architecture conservation."""
        
        # Must have at least min_cassette_size genes
        if len(query_window) < self.min_cassette_size or len(target_window) < self.min_cassette_size:
            return None
        
        # Check if windows have same number of genes (required for synteny)
        if self.require_synteny and len(query_window) != len(target_window):
            return None
        
        # Extract ordered domain architectures for each gene
        query_architectures = []
        target_architectures = []
        
        for gene in query_window:
            if not gene['domains']:  # Skip genes without domains
                return None
            query_architectures.append(tuple(gene['domains']))  # Preserve order as tuple
        
        for gene in target_window:
            if not gene['domains']:  # Skip genes without domains
                return None
            target_architectures.append(tuple(gene['domains']))
        
        # For exact matching, require identical domain architectures in order
        if self.require_exact_domain_match:
            # Check if domain architectures match exactly (gene by gene)
            exact_matches = 0
            for i in range(min(len(query_architectures), len(target_architectures))):
                if query_architectures[i] == target_architectures[i]:
                    exact_matches += 1
            
            # Require all genes to have exact domain architecture matches
            if exact_matches < len(query_architectures) or exact_matches < len(target_architectures):
                return None
        
        # Collect all shared domains
        query_all_domains = []
        target_all_domains = []
        for arch in query_architectures:
            query_all_domains.extend(arch)
        for arch in target_architectures:
            target_all_domains.extend(arch)
            
        shared_domains = set(query_all_domains).intersection(set(target_all_domains))
        
        if not shared_domains:
            return None
        
        # Calculate conservation scores
        domain_conservation_score = self._calculate_exact_conservation(
            query_architectures, target_architectures
        )
        
        # High synteny score for exact matches
        synteny_score = 1.0 if self.require_exact_domain_match else 0.5
        
        # Classify as perfect since we only allow exact matches
        cassette_type = 'perfect'
        
        # Extract genomic coordinates
        query_start = min(gene['start_pos'] for gene in query_window)
        query_end = max(gene['end_pos'] for gene in query_window)
        target_start = min(gene['start_pos'] for gene in target_window)
        target_end = max(gene['end_pos'] for gene in target_window)
        
        query_gene_ids = [gene['gene_id'] for gene in query_window]
        target_gene_ids = [gene['gene_id'] for gene in target_window]
        
        return ConservedCassette(
            query_start=query_start,
            query_end=query_end,
            target_start=target_start,
            target_end=target_end,
            shared_domains=list(shared_domains),
            domain_conservation_score=domain_conservation_score,
            synteny_score=synteny_score,
            query_genes=query_gene_ids,
            target_genes=target_gene_ids,
            cassette_type=cassette_type
        )
    
    def _calculate_exact_conservation(self, query_architectures: List[Tuple], 
                                    target_architectures: List[Tuple]) -> float:
        """Calculate conservation score for exact domain architecture matches."""
        
        if len(query_architectures) != len(target_architectures):
            return 0.0
        
        exact_matches = 0
        for q_arch, t_arch in zip(query_architectures, target_architectures):
            if q_arch == t_arch:  # Exact domain architecture match
                exact_matches += 1
        
        # Return percentage of genes with exact matches
        return exact_matches / len(query_architectures)
    
    def _calculate_domain_conservation(self, query_domains: List[str], 
                                     target_domains: List[str], 
                                     shared_domains: Set[str]) -> float:
        """Calculate domain conservation score between two windows."""
        
        # Jaccard similarity of domain sets
        query_set = set(query_domains)
        target_set = set(target_domains)
        union_size = len(query_set.union(target_set))
        jaccard = len(shared_domains) / union_size if union_size > 0 else 0
        
        # Weighted by domain frequency conservation
        query_counts = Counter(query_domains)
        target_counts = Counter(target_domains)
        
        frequency_conservation = 0
        for domain in shared_domains:
            q_freq = query_counts[domain] / len(query_domains)
            t_freq = target_counts[domain] / len(target_domains)
            frequency_conservation += 1 - abs(q_freq - t_freq)
        
        frequency_conservation /= len(shared_domains) if shared_domains else 1
        
        # Combined score
        return 0.6 * jaccard + 0.4 * frequency_conservation
    
    def _calculate_synteny_score(self, query_window: List[Dict], 
                               target_window: List[Dict], 
                               shared_domains: Set[str]) -> float:
        """Calculate synteny preservation score."""
        
        # Check if genes with shared domains maintain relative order
        query_shared_positions = []
        target_shared_positions = []
        
        for i, gene in enumerate(query_window):
            if gene['domain_set'].intersection(shared_domains):
                query_shared_positions.append(i)
        
        for i, gene in enumerate(target_window):
            if gene['domain_set'].intersection(shared_domains):
                target_shared_positions.append(i)
        
        if len(query_shared_positions) < 2 or len(target_shared_positions) < 2:
            return 0.5  # Default score for single gene matches
        
        # Calculate rank correlation (simplified Spearman's rho)
        if len(query_shared_positions) == len(target_shared_positions):
            # Check if relative order is preserved
            query_order = [0] * len(query_shared_positions)
            target_order = [0] * len(target_shared_positions)
            
            for i in range(len(query_shared_positions)):
                query_order[i] = i
                target_order[i] = i
            
            # Simple order preservation check
            inversions = 0
            total_pairs = len(query_order) * (len(query_order) - 1) // 2
            
            for i in range(len(query_order)):
                for j in range(i + 1, len(query_order)):
                    if (query_order[i] < query_order[j]) != (target_order[i] < target_order[j]):
                        inversions += 1
            
            return 1 - (inversions / total_pairs) if total_pairs > 0 else 1
        
        return 0.3  # Lower score for different numbers of shared genes
    
    def _classify_cassette_type(self, query_window: List[Dict], 
                              target_window: List[Dict], 
                              shared_domains: Set[str]) -> str:
        """Classify the type of conserved cassette."""
        
        query_domain_order = []
        target_domain_order = []
        
        for gene in query_window:
            for domain in gene['domains']:
                if domain in shared_domains:
                    query_domain_order.append(domain)
        
        for gene in target_window:
            for domain in gene['domains']:
                if domain in shared_domains:
                    target_domain_order.append(domain)
        
        # Perfect: same domains in same order
        if query_domain_order == target_domain_order:
            return 'perfect'
        
        # Rearranged: same domains, different order
        elif set(query_domain_order) == set(target_domain_order):
            return 'rearranged'
        
        # Partial: some shared domains
        else:
            return 'partial'
    
    def _merge_overlapping_cassettes(self, cassettes: List[ConservedCassette]) -> List[ConservedCassette]:
        """Merge overlapping cassettes and keep the best ones."""
        if not cassettes:
            return []
        
        # Sort by conservation score
        sorted_cassettes = sorted(cassettes, key=lambda c: c.domain_conservation_score, reverse=True)
        merged = []
        
        for cassette in sorted_cassettes:
            # Check if this cassette overlaps significantly with any already merged
            overlaps = False
            for merged_cassette in merged:
                if self._cassettes_overlap(cassette, merged_cassette):
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(cassette)
        
        return merged
    
    def _cassettes_overlap(self, c1: ConservedCassette, c2: ConservedCassette, 
                          overlap_threshold: float = 0.5) -> bool:
        """Check if two cassettes overlap significantly."""
        
        # Check query region overlap
        q1_len = c1.query_end - c1.query_start
        q2_len = c2.query_end - c2.query_start
        q_overlap = max(0, min(c1.query_end, c2.query_end) - max(c1.query_start, c2.query_start))
        q_overlap_ratio = q_overlap / min(q1_len, q2_len) if min(q1_len, q2_len) > 0 else 0
        
        # Check target region overlap
        t1_len = c1.target_end - c1.target_start
        t2_len = c2.target_end - c2.target_start
        t_overlap = max(0, min(c1.target_end, c2.target_end) - max(c1.target_start, c2.target_start))
        t_overlap_ratio = t_overlap / min(t1_len, t2_len) if min(t1_len, t2_len) > 0 else 0
        
        return q_overlap_ratio > overlap_threshold and t_overlap_ratio > overlap_threshold

def analyze_conserved_cassettes(query_genes: pd.DataFrame, 
                              target_genes: pd.DataFrame) -> Dict:
    """
    Main function to analyze conserved cassettes between two genomic loci.
    
    Returns:
        Dictionary with cassette analysis results
    """
    analyzer = CassetteAnalyzer()
    cassettes = analyzer.find_conserved_cassettes(query_genes, target_genes)
    
    # Summarize results
    if not cassettes:
        return {
            'cassettes': [],
            'total_cassettes': 0,
            'conservation_summary': 'No conserved functional cassettes detected'
        }
    
    # Calculate summary statistics
    avg_conservation = np.mean([c.domain_conservation_score for c in cassettes])
    total_shared_domains = set()
    for cassette in cassettes:
        total_shared_domains.update(cassette.shared_domains)
    
    cassette_types = Counter([c.cassette_type for c in cassettes])
    
    summary = f"Found {len(cassettes)} conserved cassettes with {len(total_shared_domains)} unique shared domains. "
    summary += f"Average conservation: {avg_conservation:.1%}. "
    summary += f"Types: {dict(cassette_types)}"
    
    return {
        'cassettes': cassettes,
        'total_cassettes': len(cassettes),
        'shared_domains': list(total_shared_domains),
        'conservation_summary': summary,
        'cassette_types': dict(cassette_types),
        'average_conservation': avg_conservation
    }