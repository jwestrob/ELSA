"""
ELSA comprehensive syntenic block analysis.

Scans entire dataset to find all syntenic blocks, performs clustering,
and builds a complete catalog of syntenic relationships.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
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
        """Compare two loci to find syntenic blocks."""
        if locus_a.sample_id == locus_b.sample_id and locus_a.locus_id == locus_b.locus_id:
            return None  # Skip self-comparison
        
        # Compute pairwise similarities between windows
        similarities = cosine_similarity(locus_a.embeddings, locus_b.embeddings)
        
        # Find high-similarity window pairs
        similarity_threshold = 0.7  # TODO: Make configurable
        matches = []
        
        for i, j in np.argwhere(similarities > similarity_threshold):
            match = WindowMatch(
                query_window_id=locus_a.window_ids[i],
                target_window_id=locus_b.window_ids[j],
                similarity_score=similarities[i, j],
                method='cosine'
            )
            matches.append(match)
        
        if len(matches) >= 2:  # Require at least 2 matching windows
            # Create syntenic block
            block = SyntenicBlock(
                query_locus=f"{locus_a.sample_id}:{locus_a.locus_id}",
                target_locus=f"{locus_b.sample_id}:{locus_b.locus_id}",
                query_windows=[m.query_window_id for m in matches],
                target_windows=[m.target_window_id for m in matches],
                matches=matches,
                chain_score=sum(m.similarity_score for m in matches),
                alignment_length=len(matches),
                identity=sum(m.similarity_score for m in matches) / len(matches)
            )
            return block
        
        return None
    
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
        """Extract feature vectors for clustering."""
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
    
    def cluster_blocks(self, blocks: List[SyntenicBlock]) -> List[SyntenicCluster]:
        """Cluster syntenic blocks using DBSCAN."""
        if len(blocks) < 2:
            return []
        
        console.print(f"\n[bold]Clustering {len(blocks)} syntenic blocks...[/bold]")
        
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
        
        console.print(f"Found {len(clusters)} clusters")
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
        clusters = self.clusterer.cluster_blocks(blocks)
        
        # Step 4: Generate statistics
        console.print("\n[bold]Step 4: Computing statistics...[/bold]")
        statistics = self._compute_statistics(loci, blocks, clusters)
        
        # Create landscape summary
        landscape = SyntenicLandscape(
            total_loci=len(loci),
            total_blocks=len(blocks),
            total_clusters=len(clusters),
            blocks=blocks,
            clusters=clusters,
            statistics=statistics
        )
        
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
    
    def save_results(self, landscape: SyntenicLandscape, output_dir: Path):
        """Save comprehensive analysis results."""
        output_dir.mkdir(exist_ok=True)
        
        # Save main results
        results = {
            'landscape': asdict(landscape),
            'metadata': {
                'version': '0.1.0',
                'config': self.config.dict()
            }
        }
        
        results_file = output_dir / 'syntenic_landscape.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save blocks table
        if landscape.blocks:
            blocks_data = []
            for i, block in enumerate(landscape.blocks):
                blocks_data.append({
                    'block_id': i,
                    'query_locus': block.query_locus,
                    'target_locus': block.target_locus,
                    'length': block.alignment_length,  # This is gene-window count, not genomic bp
                    'identity': block.identity,
                    'score': block.chain_score,
                    'n_query_windows': len(block.query_windows),  # Number of matching query windows
                    'n_target_windows': len(block.target_windows)  # Number of matching target windows
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
        console.print(f"  - {results_file}")
        console.print(f"  - syntenic_blocks.csv")
        console.print(f"  - syntenic_clusters.csv")


if __name__ == "__main__":
    # Test analysis functionality
    print("ELSA Comprehensive Syntenic Analysis")