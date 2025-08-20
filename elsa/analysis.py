"""
ELSA comprehensive syntenic block analysis.

Scans entire dataset to find all syntenic blocks, performs clustering,
and builds a complete catalog of syntenic relationships.
"""

import numpy as np
import math
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
import itertools
from collections import defaultdict
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .params import ELSAConfig
from .manifest import ELSAManifest
from .search import SearchEngine, SyntenicBlock, WindowMatch
from .analyze.cluster_mutual_jaccard import cluster_blocks_jaccard

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class LocusInfo:
    """Information about a genomic locus."""
    sample_id: str
    locus_id: str
    n_windows: int
    window_ids: List[str]
    embeddings: np.ndarray  # Shape: (n_windows, embedding_dim)


@dataclass
class SyntenicCluster:
    """A cluster of similar syntenic blocks."""
    cluster_id: int
    blocks: List[SyntenicBlock]
    consensus_length: int
    consensus_score: float
    representative_block: SyntenicBlock
    diversity: float  # Measure of within-cluster variation


@dataclass
class SyntenicLandscape:
    """Complete syntenic landscape of the dataset."""
    total_loci: int
    total_blocks: int
    total_clusters: int
    blocks: List[SyntenicBlock]
    clusters: List[SyntenicCluster]
    statistics: Dict[str, Any]


class LocusScanner:
    """Scans dataset to identify all loci and their windows."""
    
    def __init__(self, manifest: ELSAManifest):
        self.manifest = manifest
        self.windows_df = None
        
    def load_data(self):
        """Load window data for scanning."""
        windows_path = Path(self.manifest.data['artifacts']['windows']['path'])
        self.windows_df = pd.read_parquet(windows_path)
        console.print(f"Loaded {len(self.windows_df):,} windows from {self.windows_df['sample_id'].nunique()} samples")
    
    def extract_all_loci(self, min_windows: int = 3) -> List[LocusInfo]:
        """Extract all loci with sufficient windows for analysis."""
        if self.windows_df is None:
            self.load_data()
        
        loci = []
        emb_cols = [col for col in self.windows_df.columns if col.startswith('emb_')]
        
        # Group by locus
        for (sample_id, locus_id), group in self.windows_df.groupby(['sample_id', 'locus_id']):
            if len(group) >= min_windows:
                window_ids = [f"{row['sample_id']}_{row['locus_id']}_{row['window_idx']}" 
                             for _, row in group.iterrows()]
                
                # Extract embeddings matrix
                embeddings = np.array([[row[col] for col in emb_cols] for _, row in group.iterrows()])
                
                locus_info = LocusInfo(
                    sample_id=sample_id,
                    locus_id=locus_id,
                    n_windows=len(group),
                    window_ids=window_ids,
                    embeddings=embeddings
                )
                loci.append(locus_info)
        
        console.print(f"Identified {len(loci)} loci with ≥{min_windows} windows")
        return loci


class AllVsAllComparator:
    """Performs all-vs-all locus comparisons to find syntenic blocks."""
    
    def __init__(self, config: ELSAConfig):
        self.config = config
        
    def compare_loci(self, locus_a: LocusInfo, locus_b: LocusInfo) -> Optional[SyntenicBlock]:
        """Compare two loci to find syntenic blocks with cassette mode improvements."""
        if locus_a.sample_id == locus_b.sample_id and locus_a.locus_id == locus_b.locus_id:
            return None  # Skip self-comparison
        
        # Compute pairwise similarities between windows
        similarities = cosine_similarity(locus_a.embeddings, locus_b.embeddings)
        
        # Apply two-key anchor filtering (cassette mode)
        cassette_config = getattr(self.config, 'cassette_mode', None)
        if cassette_config and cassette_config.enable:
            anchor_matches = self._apply_two_key_anchor_filtering(
                similarities, locus_a, locus_b, cassette_config.anchors
            )
        else:
            # Fallback to original threshold-based filtering
            similarity_threshold = 0.8  
            anchor_matches = []
            for i, j in np.argwhere(similarities > similarity_threshold):
                match = WindowMatch(
                    query_window_id=locus_a.window_ids[i],
                    target_window_id=locus_b.window_ids[j],
                    similarity_score=similarities[i, j],
                    method='cosine'
                )
                anchor_matches.append((i, j, match))
        
        if len(anchor_matches) < 2:
            return None  # Not enough matches
        
        # Apply improved chaining with local gain (cassette mode)
        if cassette_config and cassette_config.enable:
            filtered_matches = self._apply_local_gain_chaining(
                anchor_matches, cassette_config.chain
            )
        else:
            # Fallback to original positional conservation filtering
            filtered_matches = self._filter_positional_conservation(anchor_matches)
        
        if len(filtered_matches) >= 2:  # Require at least 2 conserved matches
            matches = [match for _, _, match in filtered_matches]
            
            # Calculate gene order conservation score
            gene_order_score = self._calculate_gene_order_score(filtered_matches)
            
            # Only accept blocks with reasonable gene order conservation
            if gene_order_score >= 0.3:  # Configurable threshold
                # Additional strictness: enforce min anchors and span in windows using clustering thresholds
                try:
                    min_anchors = int(getattr(self.config.analyze.clustering, 'min_anchors', 2))
                    min_span_genes = int(getattr(self.config.analyze.clustering, 'min_span_genes', 0))
                except Exception:
                    min_anchors = 2
                    min_span_genes = 0
                # Determine minimum required consecutive windows to approximate span in genes
                # Using shingle window size n as proxy: need ceil(min_span_genes / n) windows on both sides
                try:
                    shingle_n = int(getattr(self.config.shingles, 'n', 2))
                except Exception:
                    shingle_n = 2
                min_wins_span = math.ceil(min_span_genes / max(1, shingle_n)) if min_span_genes > 0 else 0

                # Compute spans in window indices for query and target
                q_positions = [q for (q, _t, _m) in filtered_matches]
                t_positions = [t for (_q, t, _m) in filtered_matches]
                q_span_wins = (max(q_positions) - min(q_positions) + 1) if q_positions else 0
                t_span_wins = (max(t_positions) - min(t_positions) + 1) if t_positions else 0

                if len(filtered_matches) < max(2, min_anchors):
                    return None
                if min_wins_span > 0 and (q_span_wins < min_wins_span or t_span_wins < min_wins_span):
                    return None
                # Create syntenic block
                block = SyntenicBlock(
                    query_locus=f"{locus_a.sample_id}:{locus_a.locus_id}",
                    target_locus=f"{locus_b.sample_id}:{locus_b.locus_id}",
                    query_windows=[m.query_window_id for m in matches],
                    target_windows=[m.target_window_id for m in matches],
                    matches=matches,
                    chain_score=sum(m.similarity_score for m in matches) * gene_order_score,  # Weight by gene order
                    alignment_length=len(matches),
                    identity=sum(m.similarity_score for m in matches) / len(matches)
                )
                return block
        
        return None
    
    def _filter_positional_conservation(self, matches, max_offset=3):
        """Filter matches to require positional conservation."""
        if len(matches) < 2:
            return matches
        
        # Group matches by their position offset
        offset_groups = {}
        for query_pos, target_pos, match in matches:
            offset = target_pos - query_pos
            if offset not in offset_groups:
                offset_groups[offset] = []
            offset_groups[offset].append((query_pos, target_pos, match))
        
        # Find the largest group of matches with consistent offset
        best_group = []
        best_size = 0
        
        for offset, group in offset_groups.items():
            if abs(offset) <= max_offset:  # Only consider reasonable offsets
                # Check for consecutive windows within this offset group
                consecutive_matches = self._find_consecutive_matches(group)
                if len(consecutive_matches) > best_size:
                    best_group = consecutive_matches
                    best_size = len(consecutive_matches)
        
        return best_group
    
    def _find_consecutive_matches(self, matches, min_consecutive=2):
        """Find the longest consecutive sequence of window matches."""
        if len(matches) < min_consecutive:
            return matches
        
        # Sort matches by query position
        sorted_matches = sorted(matches, key=lambda x: x[0])
        
        # Find longest consecutive sequence
        best_sequence = []
        current_sequence = [sorted_matches[0]]
        
        for i in range(1, len(sorted_matches)):
            prev_query, prev_target, _ = sorted_matches[i-1]
            curr_query, curr_target, _ = sorted_matches[i]
            
            # Check if windows are consecutive and have consistent offset
            if (curr_query == prev_query + 1 and 
                curr_target == prev_target + 1):
                current_sequence.append(sorted_matches[i])
            else:
                # Sequence broken, check if it's the best so far
                if len(current_sequence) > len(best_sequence):
                    best_sequence = current_sequence[:]
                current_sequence = [sorted_matches[i]]
        
        # Check final sequence
        if len(current_sequence) > len(best_sequence):
            best_sequence = current_sequence
        
        # Return the best sequence if it meets minimum length
        return best_sequence if len(best_sequence) >= min_consecutive else []
    
    def _calculate_gene_order_score(self, matches):
        """Calculate gene order conservation score (0-1)."""
        if len(matches) < 2:
            return 1.0
        
        # Calculate position offsets
        offsets = [target_pos - query_pos for query_pos, target_pos, _ in matches]
        
        # Perfect conservation = all offsets identical
        offset_std = np.std(offsets)
        
        # Convert to score (lower std = higher score)
        # Use exponential decay: score = exp(-std)
        gene_order_score = np.exp(-offset_std)
        
        return min(gene_order_score, 1.0)
    
    def _apply_two_key_anchor_filtering(self, similarities, locus_a, locus_b, anchor_config):
        """Apply two-key anchor filtering: cosine + Jaccard + reciprocal top-k."""
        anchor_matches = []
        
        # Step 1: Apply cosine similarity threshold
        cosine_candidates = []
        for i, j in np.argwhere(similarities >= anchor_config.cosine_min):
            cosine_candidates.append((i, j, similarities[i, j]))
        
        if len(cosine_candidates) == 0:
            return []
        
        # Step 2: Calculate Jaccard similarity for cosine candidates
        # For embeddings, approximate Jaccard using cosine similarity conversion
        # Jaccard ≈ cosine / (2 - cosine) for normalized vectors
        jaccard_matches = []
        for i, j, cosine_sim in cosine_candidates:
            # Convert cosine to approximate Jaccard similarity
            jaccard_approx = cosine_sim / (2 - cosine_sim) if cosine_sim < 2 else 1.0
            
            if jaccard_approx >= anchor_config.jaccard_min:
                match = WindowMatch(
                    query_window_id=locus_a.window_ids[i],
                    target_window_id=locus_b.window_ids[j],
                    similarity_score=cosine_sim,  # Use cosine as primary score
                    method='two_key'
                )
                jaccard_matches.append((i, j, match, jaccard_approx))
        
        if len(jaccard_matches) == 0:
            return []
        
        # Step 3: Apply reciprocal top-k filtering
        k = anchor_config.reciprocal_topk
        reciprocal_matches = []
        
        # For each query window, find top-k target matches
        query_tops = {}
        for i, j, match, jaccard in jaccard_matches:
            if i not in query_tops:
                query_tops[i] = []
            query_tops[i].append((j, match, jaccard))
        
        # Keep only top-k for each query window
        for i in query_tops:
            query_tops[i] = sorted(query_tops[i], 
                                 key=lambda x: x[1].similarity_score, reverse=True)[:k]
        
        # For each target window, find top-k query matches  
        target_tops = {}
        for i in query_tops:
            for j, match, jaccard in query_tops[i]:
                if j not in target_tops:
                    target_tops[j] = []
                target_tops[j].append((i, match, jaccard))
        
        for j in target_tops:
            target_tops[j] = sorted(target_tops[j], 
                                  key=lambda x: x[1].similarity_score, reverse=True)[:k]
        
        # Find reciprocal matches (mutual top-k)
        for i in query_tops:
            for j, match, jaccard in query_tops[i]:
                # Check if (i,j) is also in target's top-k
                if j in target_tops:
                    for i_back, match_back, _ in target_tops[j]:
                        if i_back == i:  # Reciprocal match found
                            reciprocal_matches.append((i, j, match))
                            break
        
        return reciprocal_matches
    
    def _apply_local_gain_chaining(self, anchor_matches, chain_config):
        """Apply local gain chaining with position/gap constraints and density floor."""
        if len(anchor_matches) < 2:
            return anchor_matches
        
        # Sort matches by query position for chaining
        sorted_matches = sorted(anchor_matches, key=lambda x: x[0])  # x[0] is query_pos
        
        # Apply local gain dynamic programming
        best_chain = []
        current_chain = [sorted_matches[0]]
        
        for i in range(1, len(sorted_matches)):
            prev_query, prev_target, prev_match = sorted_matches[i-1]
            curr_query, curr_target, curr_match = sorted_matches[i]
            
            # Calculate position delta and gap
            delta_pos = abs(curr_target - prev_target - (curr_query - prev_query))
            gap_genes = curr_query - prev_query - 1
            
            # Check position and gap constraints
            if (delta_pos <= chain_config.pos_band_genes and 
                gap_genes <= chain_config.max_gap_genes):
                
                # Calculate local gain: ΔS = cosine + λ·jaccard - α·|Δpos| - β·gap_genes  
                # For simplicity, use similarity as combined cosine+jaccard score
                lambda_jaccard = getattr(chain_config, 'lambda_jaccard', 0.5)
                alpha = 0.1  # Position penalty weight
                beta = 0.05  # Gap penalty weight
                
                local_gain = (curr_match.similarity_score + 
                            lambda_jaccard * curr_match.similarity_score -  # Approximate Jaccard
                            alpha * delta_pos - 
                            beta * gap_genes)
                
                # Accept step if local gain exceeds threshold
                if local_gain >= chain_config.delta_min:
                    current_chain.append(sorted_matches[i])
                else:
                    # Local gain too low, evaluate current chain
                    if self._check_density_floor(current_chain, chain_config):
                        if len(current_chain) > len(best_chain):
                            best_chain = current_chain[:]
                    # Start new chain
                    current_chain = [sorted_matches[i]]
            else:
                # Constraints violated, evaluate current chain
                if self._check_density_floor(current_chain, chain_config):
                    if len(current_chain) > len(best_chain):
                        best_chain = current_chain[:]
                # Start new chain
                current_chain = [sorted_matches[i]]
        
        # Evaluate final chain
        if self._check_density_floor(current_chain, chain_config):
            if len(current_chain) > len(best_chain):
                best_chain = current_chain
        
        return best_chain if len(best_chain) >= 2 else []
    
    def _check_density_floor(self, chain_matches, chain_config):
        """Check if chain meets density floor requirement."""
        if len(chain_matches) < 2:
            return False
        
        # Calculate density over last N genes
        window_genes = chain_config.density_window_genes
        min_density = chain_config.density_min_anchors_per_gene
        
        # For simplicity, use chain length as gene span approximation
        gene_span = len(chain_matches) * 2  # Assuming 2 genes per window
        anchors_per_gene = len(chain_matches) / max(gene_span, 1)
        
        return anchors_per_gene >= min_density
    
    def find_all_blocks(self, loci: List[LocusInfo]) -> List[SyntenicBlock]:
        """Find all syntenic blocks via all-vs-all comparison."""
        blocks = []
        total_comparisons = len(loci) * (len(loci) - 1) // 2
        
        console.print(f"\n[bold]Performing all-vs-all locus comparisons...[/bold]")
        console.print(f"Total comparisons: {total_comparisons:,}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Comparing loci...", total=total_comparisons)
            
            for i, locus_a in enumerate(loci):
                for j, locus_b in enumerate(loci[i+1:], i+1):
                    block = self.compare_loci(locus_a, locus_b)
                    if block:
                        blocks.append(block)
                    
                    progress.advance(task)
        
        console.print(f"Found {len(blocks):,} syntenic blocks")
        return blocks


class SyntenicClusterer:
    """Clusters syntenic blocks by similarity."""
    
    def __init__(self, config: ELSAConfig):
        self.config = config
    
    def extract_block_features(self, blocks: List[SyntenicBlock]) -> np.ndarray:
        """Extract feature vectors for clustering (legacy method)."""
        features = []
        
        for block in blocks:
            # Simple features: length, identity, normalized score
            feature_vector = [
                block.alignment_length,
                block.identity,
                block.chain_score / block.alignment_length,  # Normalized score
                len(block.query_windows),
                len(block.target_windows)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def cluster_blocks(self, blocks: List[SyntenicBlock], window_embed_lookup: Callable = None) -> Tuple[List[SyntenicCluster], Dict[int, int]]:
        """Cluster syntenic blocks using configurable method."""
        if len(blocks) < 2:
            return [], {}
        
        clustering_config = getattr(self.config, 'analyze', None)
        if clustering_config:
            clustering_config = getattr(clustering_config, 'clustering', None)
        
        # Get clustering method from config
        method = getattr(clustering_config, 'method', 'mutual_jaccard') if clustering_config else 'mutual_jaccard'
        
        if method == 'mutual_jaccard' and window_embed_lookup is not None:
            console.print(f"\n[bold]Clustering {len(blocks)} syntenic blocks with Mutual-k Jaccard...[/bold]")
            
            # Use new Mutual-k Jaccard clustering
            cluster_assignments = cluster_blocks_jaccard(blocks, window_embed_lookup, clustering_config)
            
            # Convert to legacy cluster format for compatibility
            clusters = self._convert_assignments_to_clusters(blocks, cluster_assignments)
            
            console.print(f"Found {len(clusters)} non-sink clusters")
            return clusters, cluster_assignments
            
        elif method == 'dbscan' or window_embed_lookup is None:
            console.print(f"\n[bold]Clustering {len(blocks)} syntenic blocks with DBSCAN (legacy)...[/bold]")
            
            # Fall back to legacy DBSCAN clustering
            clusters = self._cluster_blocks_dbscan(blocks)
            
            # Create cluster assignments (no sink cluster for DBSCAN)
            cluster_assignments = {}
            for i, block in enumerate(blocks):
                # Find which cluster this block belongs to
                for cluster in clusters:
                    if block in cluster.blocks:
                        cluster_assignments[i] = cluster.cluster_id + 1  # Offset by 1 to avoid sink
                        break
                else:
                    cluster_assignments[i] = -1  # DBSCAN noise
            
            return clusters, cluster_assignments
            
        else:
            console.print(f"\n[bold]Clustering disabled[/bold]")
            return [], {}
    
    def _cluster_blocks_dbscan(self, blocks: List[SyntenicBlock]) -> List[SyntenicCluster]:
        """Legacy DBSCAN clustering method."""
        # Extract features
        features = self.extract_block_features(blocks)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster with DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Group blocks by cluster
        clusters = []
        cluster_groups = defaultdict(list)
        
        for block, label in zip(blocks, cluster_labels):
            if label != -1:  # Ignore noise points
                cluster_groups[label].append(block)
        
        for cluster_id, cluster_blocks in cluster_groups.items():
            # Find representative block (highest scoring)
            representative = max(cluster_blocks, key=lambda b: b.chain_score)
            
            # Calculate consensus metrics
            consensus_length = int(np.mean([b.alignment_length for b in cluster_blocks]))
            consensus_score = np.mean([b.chain_score for b in cluster_blocks])
            
            # Calculate diversity (coefficient of variation of scores)
            scores = [b.chain_score for b in cluster_blocks]
            diversity = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            
            cluster = SyntenicCluster(
                cluster_id=cluster_id,
                blocks=cluster_blocks,
                consensus_length=consensus_length,
                consensus_score=consensus_score,
                representative_block=representative,
                diversity=diversity
            )
            clusters.append(cluster)
        
        return clusters
    
    def _convert_assignments_to_clusters(self, blocks: List[SyntenicBlock], cluster_assignments: Dict[int, int]) -> List[SyntenicCluster]:
        """Convert cluster assignments to SyntenicCluster objects."""
        # Group blocks by cluster ID (excluding sink cluster 0)
        cluster_groups = defaultdict(list)
        
        for i, block in enumerate(blocks):
            cluster_id = cluster_assignments.get(i, 0)
            if cluster_id != 0:  # Skip sink cluster
                cluster_groups[cluster_id].append(block)
        
        clusters = []
        for cluster_id, cluster_blocks in cluster_groups.items():
            # Find representative block (highest scoring)
            representative = max(cluster_blocks, key=lambda b: b.chain_score)
            
            # Calculate consensus metrics
            consensus_length = int(np.mean([b.alignment_length for b in cluster_blocks]))
            consensus_score = np.mean([b.chain_score for b in cluster_blocks])
            
            # Calculate diversity (coefficient of variation of scores)
            scores = [b.chain_score for b in cluster_blocks]
            diversity = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            
            cluster = SyntenicCluster(
                cluster_id=cluster_id,
                blocks=cluster_blocks,
                consensus_length=consensus_length,
                consensus_score=consensus_score,
                representative_block=representative,
                diversity=diversity
            )
            clusters.append(cluster)
        
        return clusters


class SyntenicAnalyzer:
    """Main analyzer coordinating comprehensive syntenic analysis."""
    
    def __init__(self, config: ELSAConfig, manifest: ELSAManifest):
        self.config = config
        self.manifest = manifest
        self.locus_scanner = LocusScanner(manifest)
        self.comparator = AllVsAllComparator(config)
        self.clusterer = SyntenicClusterer(config)
    
    def analyze(self, min_windows: int = 3, min_similarity: float = 0.7) -> SyntenicLandscape:
        """Perform comprehensive syntenic block analysis."""
        console.print("[bold blue]ELSA Comprehensive Syntenic Analysis[/bold blue]")
        
        # Step 1: Extract all loci
        console.print("\n[bold]Step 1: Scanning dataset for loci...[/bold]")
        loci = self.locus_scanner.extract_all_loci(min_windows=min_windows)
        
        # Step 2: Find all syntenic blocks
        console.print("\n[bold]Step 2: Finding syntenic blocks...[/bold]")
        blocks = self.comparator.find_all_blocks(loci)
        
        # Step 3: Cluster blocks
        console.print("\n[bold]Step 3: Clustering syntenic blocks...[/bold]")
        
        # Create window embedding lookup from loaded data
        window_embed_lookup = self._create_window_embed_lookup()
        
        # Cluster with assignments
        clusters, cluster_assignments = self.clusterer.cluster_blocks(blocks, window_embed_lookup)
        
        # Step 4: Generate statistics
        console.print("\n[bold]Step 4: Computing statistics...[/bold]")
        statistics = self._compute_statistics(loci, blocks, clusters)
        
        # Create landscape summary with cluster assignments
        landscape = SyntenicLandscape(
            total_loci=len(loci),
            total_blocks=len(blocks),
            total_clusters=len(clusters),
            blocks=blocks,
            clusters=clusters,
            statistics=statistics
        )
        
        # Store cluster assignments for save_results
        landscape.cluster_assignments = cluster_assignments
        
        return landscape
    
    def _compute_statistics(self, loci: List[LocusInfo], blocks: List[SyntenicBlock], 
                          clusters: List[SyntenicCluster]) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        # Sample-level statistics
        samples = set(locus.sample_id for locus in loci)
        sample_stats = {}
        
        for sample in samples:
            sample_loci = [l for l in loci if l.sample_id == sample]
            sample_blocks = [b for b in blocks if sample in b.query_locus or sample in b.target_locus]
            
            sample_stats[sample] = {
                'loci': len(sample_loci),
                'total_windows': sum(l.n_windows for l in sample_loci),
                'blocks_involving': len(sample_blocks)
            }
        
        # Block statistics
        if blocks:
            block_lengths = [b.alignment_length for b in blocks]
            block_identities = [b.identity for b in blocks]
            block_scores = [b.chain_score for b in blocks]
        else:
            block_lengths = block_identities = block_scores = [0]
        
        # Cluster statistics
        if clusters:
            cluster_sizes = [len(c.blocks) for c in clusters]
            cluster_diversities = [c.diversity for c in clusters]
        else:
            cluster_sizes = cluster_diversities = [0]
        
        statistics = {
            'samples': {
                'total_samples': len(samples),
                'sample_breakdown': sample_stats
            },
            'loci': {
                'total_loci': len(loci),
                'mean_windows_per_locus': np.mean([l.n_windows for l in loci]),
                'median_windows_per_locus': np.median([l.n_windows for l in loci])
            },
            'blocks': {
                'total_blocks': len(blocks),
                'mean_length': np.mean(block_lengths),
                'median_length': np.median(block_lengths),
                'mean_identity': np.mean(block_identities),
                'median_identity': np.median(block_identities),
                'length_distribution': {
                    'min': int(np.min(block_lengths)),
                    'max': int(np.max(block_lengths)),
                    'q25': int(np.percentile(block_lengths, 25)),
                    'q75': int(np.percentile(block_lengths, 75))
                }
            },
            'clusters': {
                'total_clusters': len(clusters),
                'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0,
                'mean_diversity': np.mean(cluster_diversities) if cluster_diversities else 0
            }
        }
        
        return statistics
    
    def _create_window_embed_lookup(self) -> Callable[[str], Optional[np.ndarray]]:
        """Create a window embedding lookup function from loaded window data."""
        try:
            # Load window embeddings from manifest
            if not self.manifest.has_artifact('windows'):
                console.print("[yellow]No window embeddings found - clustering will use DBSCAN fallback[/yellow]")
                return None
                
            windows_path = Path(self.manifest.data['artifacts']['windows']['path'])
            windows_df = pd.read_parquet(windows_path)
            
            # Extract embedding columns
            emb_cols = [col for col in windows_df.columns if col.startswith('emb_')]
            
            if not emb_cols:
                console.print("[yellow]No embedding columns found - clustering will use DBSCAN fallback[/yellow]")
                return None
            
            # Create window ID to embedding mapping
            window_embeddings = {}
            for _, row in windows_df.iterrows():
                window_id = f"{row['sample_id']}_{row['locus_id']}_{row['window_idx']}"
                embedding = np.array([row[col] for col in emb_cols])
                window_embeddings[window_id] = embedding
            
            console.print(f"Loaded {len(window_embeddings)} window embeddings for clustering")
            
            def lookup_func(window_id: str) -> Optional[np.ndarray]:
                return window_embeddings.get(window_id)
            
            return lookup_func
            
        except Exception as e:
            logger.warning(f"Failed to create window embedding lookup: {e}")
            console.print("[yellow]Failed to load window embeddings - clustering will use DBSCAN fallback[/yellow]")
            return None
    
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
    
    def save_results(self, landscape: SyntenicLandscape, output_dir: Path):
        """Save comprehensive analysis results."""
        output_dir.mkdir(exist_ok=True)
        
        # Skip saving the massive JSON file - it's too big and slow
        # We'll extract window details directly in the database population step
        console.print("[yellow]Skipping syntenic_landscape.json (too large for practical use)[/yellow]")
        
        # Save blocks table with embedded window information
        if landscape.blocks:
            blocks_data = []
            for i, block in enumerate(landscape.blocks):
                # Deduplicate and preserve order of window IDs to avoid repeated entries
                def _dedupe_preserve_order(seq):
                    seen = set()
                    out = []
                    for x in seq:
                        if x not in seen:
                            seen.add(x)
                            out.append(x)
                    return out

                query_windows_dedup = _dedupe_preserve_order(block.query_windows)
                target_windows_dedup = _dedupe_preserve_order(block.target_windows)

                # Extract window indices from deduplicated window IDs
                query_indices = [self._extract_window_index(w) for w in query_windows_dedup]
                target_indices = [self._extract_window_index(w) for w in target_windows_dedup]
                
                # Filter out None values and calculate ranges
                query_indices = [idx for idx in query_indices if idx is not None]
                target_indices = [idx for idx in target_indices if idx is not None]
                
                query_start = min(query_indices) if query_indices else None
                query_end = max(query_indices) if query_indices else None
                target_start = min(target_indices) if target_indices else None
                target_end = max(target_indices) if target_indices else None
                
                # Get cluster assignment for this block
                cluster_assignments = getattr(landscape, 'cluster_assignments', {})
                cluster_id = cluster_assignments.get(i, 0)  # Default to sink (0) if no assignment
                
                blocks_data.append({
                    'block_id': i,
                    'cluster_id': cluster_id,
                    'query_locus': block.query_locus,
                    'target_locus': block.target_locus,
                    'length': block.alignment_length,  # This is gene-window count, not genomic bp
                    'identity': block.identity,
                    'score': block.chain_score,
                    'n_query_windows': len(query_windows_dedup),  # Use deduplicated counts
                    'n_target_windows': len(target_windows_dedup),
                    'query_window_start': query_start,
                    'query_window_end': query_end,
                    'target_window_start': target_start,
                    'target_window_end': target_end,
                    'query_windows_json': ';'.join(query_windows_dedup),  # Use semicolon instead of JSON
                    'target_windows_json': ';'.join(target_windows_dedup)
                })
            
            blocks_df = pd.DataFrame(blocks_data)
            blocks_df.to_csv(output_dir / 'syntenic_blocks.csv', index=False)
        
        # Save clusters table
        if landscape.clusters:
            clusters_data = []
            for cluster in landscape.clusters:
                clusters_data.append({
                    'cluster_id': cluster.cluster_id,
                    'size': len(cluster.blocks),
                    'consensus_length': cluster.consensus_length,
                    'consensus_score': cluster.consensus_score,
                    'diversity': cluster.diversity,
                    'representative_query': cluster.representative_block.query_locus,
                    'representative_target': cluster.representative_block.target_locus
                })
            
            clusters_df = pd.DataFrame(clusters_data)
            clusters_df.to_csv(output_dir / 'syntenic_clusters.csv', index=False)
        
        console.print(f"✓ Results saved to: {output_dir}")
        console.print(f"  - syntenic_blocks.csv")
        console.print(f"  - syntenic_clusters.csv")


if __name__ == "__main__":
    # Test analysis functionality
    print("ELSA Comprehensive Syntenic Analysis")
