"""
ELSA search system for finding syntenic blocks.

Implements query parsing, index searching, result ranking, and collinear chaining
to find syntenic blocks similar to a query locus.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import re
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .params import ELSAConfig
from .manifest import ELSAManifest
from .indexing import MinHashLSH, SignedRandomProjection

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class QueryLocus:
    """A query locus specification."""
    sample_id: str
    contig_id: str
    start: Optional[int] = None
    end: Optional[int] = None
    
    @classmethod
    def parse(cls, locus_str: str) -> 'QueryLocus':
        """Parse locus string like 'sample:contig:start-end' or 'sample:contig'."""
        parts = locus_str.split(':')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid locus format: {locus_str}. Expected 'sample:contig' or 'sample:contig:start-end'")
        
        sample_id = parts[0]
        contig_id = parts[1]
        
        start, end = None, None
        if len(parts) == 3:
            # Parse start-end range
            range_part = parts[2]
            if '-' in range_part:
                start_str, end_str = range_part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
            else:
                start = int(range_part)
                end = start
        
        return cls(sample_id=sample_id, contig_id=contig_id, start=start, end=end)


@dataclass
class WindowMatch:
    """A match between query and target windows."""
    query_window_id: str
    target_window_id: str
    similarity_score: float
    method: str  # 'discrete' or 'continuous'


@dataclass 
class SyntenicBlock:
    """A syntenic block found by collinear chaining."""
    query_locus: str
    target_locus: str
    query_windows: List[str]
    target_windows: List[str]
    matches: List[WindowMatch]
    chain_score: float
    alignment_length: int
    identity: float


class IndexLoader:
    """Load and manage search indexes."""
    
    def __init__(self, manifest: ELSAManifest):
        self.manifest = manifest
        self.discrete_index = None
        self.continuous_index = None
        
    def load_indexes(self):
        """Load discrete and continuous indexes from disk."""
        console.print("Loading search indexes...")
        
        # Load discrete index
        discrete_path = Path(self.manifest.data['artifacts']['indexes/discrete_index.json']['file_path'])
        with open(discrete_path) as f:
            discrete_data = json.load(f)
        
        # Reconstruct discrete index
        self.discrete_index = MinHashLSH.__new__(MinHashLSH)
        self.discrete_index.num_hashes = discrete_data['config']['num_hashes']
        self.discrete_index.bands = discrete_data['config']['bands']
        self.discrete_index.rows = discrete_data['config']['rows']
        self.discrete_index.hash_coeffs_a = np.array(discrete_data['hash_coeffs_a'])
        self.discrete_index.hash_coeffs_b = np.array(discrete_data['hash_coeffs_b'])
        self.discrete_index.buckets = discrete_data['buckets']
        
        # Load continuous index
        continuous_path = Path(self.manifest.data['artifacts']['indexes/continuous_index.json']['file_path'])
        with open(continuous_path) as f:
            continuous_data = json.load(f)
        
        # Reconstruct continuous index
        self.continuous_index = SignedRandomProjection.__new__(SignedRandomProjection)
        self.continuous_index.embedding_dim = continuous_data['config']['embedding_dim']
        self.continuous_index.num_bits = continuous_data['config']['num_bits']
        self.continuous_index.projection_matrix = np.array(continuous_data['projection_matrix'])
        self.continuous_index.index = continuous_data['index']
        
        console.print(f"✓ Loaded discrete index: {len(self.discrete_index.buckets)} bands")
        console.print(f"✓ Loaded continuous index: {len(self.continuous_index.index)} patterns")


class QueryProcessor:
    """Process query loci and extract windows."""
    
    def __init__(self, manifest: ELSAManifest):
        self.manifest = manifest
        self.windows_df = None
        
    def load_windows(self):
        """Load window data for query processing."""
        windows_path = Path(self.manifest.data['artifacts']['windows']['path'])
        self.windows_df = pd.read_parquet(windows_path)
        console.print(f"Loaded {len(self.windows_df):,} windows for search")
        
    def find_query_windows(self, query_locus: QueryLocus) -> List[Tuple[str, np.ndarray]]:
        """Find windows that match the query locus specification."""
        if self.windows_df is None:
            self.load_windows()
        
        # Filter by sample and locus
        mask = (self.windows_df['sample_id'] == query_locus.sample_id)
        
        if query_locus.contig_id:
            # Check if contig_id is contained in locus_id
            mask = mask & (self.windows_df['locus_id'].str.contains(query_locus.contig_id, regex=False))
        
        matching_windows = self.windows_df[mask]
        
        if len(matching_windows) == 0:
            raise ValueError(f"No windows found for query locus: {query_locus}")
        
        # Extract embeddings
        emb_cols = [col for col in self.windows_df.columns if col.startswith('emb_')]
        
        query_windows = []
        for _, row in matching_windows.iterrows():
            window_id = f"{row['sample_id']}_{row['locus_id']}_{row['window_idx']}"
            embedding = np.array([row[col] for col in emb_cols])
            query_windows.append((window_id, embedding))
        
        console.print(f"Found {len(query_windows)} query windows")
        return query_windows


class SimilaritySearcher:
    """Search indexes for similar windows."""
    
    def __init__(self, index_loader: IndexLoader):
        self.index_loader = index_loader
        
    def search_discrete(self, query_windows: List[Tuple[str, np.ndarray]], 
                       max_candidates: int = 1000) -> List[WindowMatch]:
        """Search discrete index for similar windows."""
        matches = []
        
        for query_window_id, embedding in query_windows:
            candidates = self.index_loader.discrete_index.query(embedding, max_candidates)
            
            for candidate_id in candidates:
                match = WindowMatch(
                    query_window_id=query_window_id,
                    target_window_id=candidate_id,
                    similarity_score=1.0,  # LSH doesn't provide similarity scores
                    method='discrete'
                )
                matches.append(match)
        
        return matches
    
    def search_continuous(self, query_windows: List[Tuple[str, np.ndarray]], 
                         max_hamming_dist: int = 10) -> List[WindowMatch]:
        """Search continuous index for similar windows."""
        matches = []
        
        for query_window_id, embedding in query_windows:
            candidates = self.index_loader.continuous_index.query(embedding, max_hamming_dist)
            
            for candidate_id, hamming_dist in candidates:
                # Convert Hamming distance to similarity score (0-1)
                similarity = 1.0 - (hamming_dist / self.index_loader.continuous_index.num_bits)
                
                match = WindowMatch(
                    query_window_id=query_window_id,
                    target_window_id=candidate_id,
                    similarity_score=similarity,
                    method='continuous'
                )
                matches.append(match)
        
        return matches


class CollinearChainer:
    """Chain individual window matches into syntenic blocks."""
    
    def __init__(self, config: ELSAConfig):
        self.config = config
        
    def chain_matches(self, matches: List[WindowMatch]) -> List[SyntenicBlock]:
        """Apply collinear chaining to find syntenic blocks."""
        # NOTE: This simplified implementation is being deprecated.
        # For production use, the sophisticated chaining logic in analysis.py
        # should be used with _filter_positional_conservation() and local gain.
        
        # Group matches by target locus for basic chaining
        target_groups = {}
        for match in matches:
            # Extract target locus from window ID 
            target_parts = match.target_window_id.split('_')
            if len(target_parts) >= 2:
                target_locus = f"{target_parts[0]}_{target_parts[1]}"
            else:
                target_locus = target_parts[0]
                
            if target_locus not in target_groups:
                target_groups[target_locus] = []
            target_groups[target_locus].append(match)
        
        blocks = []
        for target_locus, locus_matches in target_groups.items():
            if len(locus_matches) >= 2:  # Require at least 2 windows for a block
                block = SyntenicBlock(
                    query_locus="query",  # TODO: Extract from matches
                    target_locus=target_locus,
                    query_windows=[m.query_window_id for m in locus_matches],
                    target_windows=[m.target_window_id for m in locus_matches],
                    matches=locus_matches,
                    chain_score=sum(m.similarity_score for m in locus_matches),
                    alignment_length=len(locus_matches),
                    identity=sum(m.similarity_score for m in locus_matches) / len(locus_matches)
                )
                blocks.append(block)
        
        # Sort by chain score (best first)
        blocks.sort(key=lambda b: b.chain_score, reverse=True)
        return blocks


class SearchEngine:
    """Main search engine coordinating all components."""
    
    def __init__(self, config: ELSAConfig, manifest: ELSAManifest):
        self.config = config
        self.manifest = manifest
        self.index_loader = IndexLoader(manifest)
        self.query_processor = QueryProcessor(manifest)
        self.similarity_searcher = SimilaritySearcher(self.index_loader)
        self.collinear_chainer = CollinearChainer(config)
        
    def search(self, query_locus_str: str, max_results: int = 50) -> List[SyntenicBlock]:
        """Execute complete search pipeline."""
        console.print(f"[bold blue]ELSA Search Pipeline[/bold blue]")
        
        # Parse query
        query_locus = QueryLocus.parse(query_locus_str)
        console.print(f"Query: {query_locus.sample_id}:{query_locus.contig_id}")
        
        # Load indexes
        self.index_loader.load_indexes()
        
        # Find query windows
        query_windows = self.query_processor.find_query_windows(query_locus)
        
        # Search indexes
        console.print("\n[bold]Searching indexes...[/bold]")
        discrete_matches = self.similarity_searcher.search_discrete(query_windows)
        continuous_matches = self.similarity_searcher.search_continuous(query_windows)
        
        all_matches = discrete_matches + continuous_matches
        console.print(f"Found {len(all_matches)} total window matches")
        
        # Chain into syntenic blocks
        console.print("\n[bold]Chaining into syntenic blocks...[/bold]")
        blocks = self.collinear_chainer.chain_matches(all_matches)
        console.print(f"Found {len(blocks)} syntenic blocks")
        
        return blocks[:max_results]


if __name__ == "__main__":
    # Test search functionality
    print("ELSA Search System")
    
    # Test query parsing
    query = QueryLocus.parse("sample1:contig_1:1000-2000")
    print(f"Parsed query: {query}")