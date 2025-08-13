"""
ELSA indexing system for discrete and continuous similarity search.

Implements MinHash LSH for discrete indexing and signed random projection 
for continuous indexing based on projected protein embeddings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from joblib import Parallel, delayed

from .params import ELSAConfig, DiscreteConfig, ContinuousConfig
from .manifest import ELSAManifest

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics for built indexes."""
    total_windows: int
    discrete_buckets: int
    continuous_bits: int
    build_time_seconds: float
    memory_mb: float


class MinHashLSH:
    """MinHash LSH indexing for discrete similarity search."""
    
    def __init__(self, config: DiscreteConfig):
        self.config = config
        self.num_hashes = config.minhash_hashes
        self.bands, self.rows = config.bands_rows
        self.buckets = {}  # band_idx -> bucket_hash -> [window_ids]
        
        # Generate random hash functions
        np.random.seed(42)  # Reproducible hashes
        self.hash_coeffs_a = np.random.randint(1, 2**32 - 1, size=self.num_hashes, dtype=np.uint64)
        self.hash_coeffs_b = np.random.randint(0, 2**32 - 1, size=self.num_hashes, dtype=np.uint64)
        
    def _hash_function(self, x: int, a: int, b: int) -> int:
        """Single hash function for MinHash."""
        # Use Python's unlimited precision integers to prevent overflow
        PRIME = 2**32 - 1
        return ((a * x + b) % PRIME)
    
    def _compute_minhash(self, features: Set[int]) -> np.ndarray:
        """Compute MinHash signature for a set of features."""
        if not features:
            return np.full(self.num_hashes, 2**32 - 1, dtype=np.uint64)
        
        signatures = np.full(self.num_hashes, 2**32 - 1, dtype=np.uint64)
        
        for feature in features:
            for i in range(self.num_hashes):
                hash_val = self._hash_function(feature, self.hash_coeffs_a[i], self.hash_coeffs_b[i])
                signatures[i] = min(signatures[i], hash_val)
        
        return signatures
    
    def _features_from_embedding(self, embedding: np.ndarray) -> Set[int]:
        """Convert embedding to discrete features using quantization."""
        # Simple quantization: discretize each dimension
        quantized = np.round(embedding * 1000).astype(np.int32)
        # Create features from quantized values and their positions
        features = set()
        for i, val in enumerate(quantized):
            features.add(hash((i, val)))
        return features
    
    def add_window(self, window_id: str, embedding: np.ndarray):
        """Add a window to the LSH index."""
        # Convert embedding to discrete features
        features = self._features_from_embedding(embedding)
        
        # Compute MinHash signature
        signature = self._compute_minhash(features)
        
        # Add to LSH bands
        for band_idx in range(self.bands):
            start_idx = band_idx * self.rows
            end_idx = start_idx + self.rows
            band_signature = tuple(signature[start_idx:end_idx])
            
            # Hash the band signature
            band_hash = hashlib.md5(str(band_signature).encode()).hexdigest()
            
            # Add to bucket
            if band_idx not in self.buckets:
                self.buckets[band_idx] = {}
            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            
            self.buckets[band_idx][band_hash].append(window_id)
    
    def query(self, embedding: np.ndarray, max_candidates: int = 1000) -> List[str]:
        """Query for similar windows."""
        features = self._features_from_embedding(embedding)
        signature = self._compute_minhash(features)
        
        candidates = set()
        
        for band_idx in range(self.bands):
            start_idx = band_idx * self.rows
            end_idx = start_idx + self.rows
            band_signature = tuple(signature[start_idx:end_idx])
            band_hash = hashlib.md5(str(band_signature).encode()).hexdigest()
            
            if band_idx in self.buckets and band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])
                
                if len(candidates) >= max_candidates:
                    break
        
        return list(candidates)[:max_candidates]
    
    def save(self, output_path: Path):
        """Save index to disk."""
        index_data = {
            'config': {
                'num_hashes': self.num_hashes,
                'bands': self.bands,
                'rows': self.rows
            },
            'hash_coeffs_a': self.hash_coeffs_a.tolist(),
            'hash_coeffs_b': self.hash_coeffs_b.tolist(),
            'buckets': self.buckets
        }
        
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)


class SignedRandomProjection:
    """Signed random projection for continuous similarity search."""
    
    def __init__(self, config: ContinuousConfig, embedding_dim: int):
        self.config = config
        self.embedding_dim = embedding_dim
        self.num_bits = config.srp_bits
        
        # Generate random projection matrix
        np.random.seed(config.srp_seed)
        self.projection_matrix = np.random.randn(embedding_dim, self.num_bits)
        
        self.index = {}  # bit_pattern -> [window_ids]
    
    def _embed_to_bits(self, embedding: np.ndarray) -> str:
        """Convert embedding to binary hash."""
        # Project to lower dimension
        projected = np.dot(embedding, self.projection_matrix)
        
        # Convert to binary string
        bits = ''.join(['1' if x > 0 else '0' for x in projected])
        return bits
    
    def add_window(self, window_id: str, embedding: np.ndarray):
        """Add window to continuous index."""
        bit_pattern = self._embed_to_bits(embedding)
        
        if bit_pattern not in self.index:
            self.index[bit_pattern] = []
        self.index[bit_pattern].append(window_id)
    
    def query(self, embedding: np.ndarray, max_hamming_dist: int = 10) -> List[Tuple[str, int]]:
        """Query for similar windows with Hamming distance."""
        query_bits = self._embed_to_bits(embedding)
        results = []
        
        for pattern, window_ids in self.index.items():
            # Compute Hamming distance
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(query_bits, pattern))
            
            if hamming_dist <= max_hamming_dist:
                for window_id in window_ids:
                    results.append((window_id, hamming_dist))
        
        # Sort by Hamming distance
        results.sort(key=lambda x: x[1])
        return results
    
    def save(self, output_path: Path):
        """Save index to disk."""
        index_data = {
            'config': {
                'embedding_dim': self.embedding_dim,
                'num_bits': self.num_bits,
                'seed': self.config.srp_seed
            },
            'projection_matrix': self.projection_matrix.tolist(),
            'index': self.index
        }
        
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)


def process_window_batch(batch_data: List[Tuple[str, np.ndarray]], 
                         discrete_config: DiscreteConfig, 
                         continuous_config: ContinuousConfig,
                         embedding_dim: int) -> Tuple[Dict, Dict]:
    """Process a batch of windows for indexing (for parallel processing)."""
    # Create local indexes for this batch
    discrete_index = MinHashLSH(discrete_config)
    continuous_index = SignedRandomProjection(continuous_config, embedding_dim)
    
    for window_id, embedding in batch_data:
        discrete_index.add_window(window_id, embedding)
        continuous_index.add_window(window_id, embedding)
    
    return discrete_index.buckets, continuous_index.index


class IndexBuilder:
    """Main index building pipeline."""
    
    def __init__(self, config: ELSAConfig, manifest: ELSAManifest):
        self.config = config
        self.manifest = manifest
        self.console = console
        
        # Determine number of parallel jobs
        if self.config.system.jobs == "auto":
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = self.config.system.jobs
        
        console.print(f"Using {self.n_jobs} parallel jobs for indexing")
    
    def build_indexes(self, resume: bool = False):
        """Build discrete and continuous indexes from shingles."""
        console.print("[bold blue]Building ELSA Indexes[/bold blue]")
        
        # Check if weighted sketching is enabled
        use_weighted_sketching = (
            hasattr(self.config, 'phase2') and 
            self.config.phase2.enable and 
            self.config.phase2.weighted_sketch
        )
        
        # Load PFAM annotations if weighted sketching is enabled
        pfam_annotations = {}
        if use_weighted_sketching:
            console.print("Loading PFAM annotations for weighted sketching...")
            from .pfam_annotation import PfamAnnotator
            
            pfam_dir = Path(self.manifest.work_dir) / "pfam_annotations"
            if pfam_dir.exists():
                annotator = PfamAnnotator()
                pfam_annotations = annotator.load_pfam_annotations(pfam_dir)
                console.print(f"✓ Loaded PFAM annotations for {len(pfam_annotations)} samples")
            else:
                console.print("⚠️  No PFAM annotations found, weighted sketching will use uniform weights")
        
        # Load shingles  
        shingles_path = Path(self.manifest.data['artifacts']['windows']['path'])
        if not shingles_path.exists():
            raise FileNotFoundError(f"Shingles not found: {shingles_path}")
        
        console.print(f"Loading shingles from: {shingles_path}")
        shingles_df = pd.read_parquet(shingles_path)
        
        if use_weighted_sketching:
            console.print(f"Found {len(shingles_df):,} windows to index with weighted sketching")
        else:
            console.print(f"Found {len(shingles_df):,} windows to index")
        
        # Get embedding columns (emb_000, emb_001, etc.)
        emb_cols = [col for col in shingles_df.columns if col.startswith('emb_')]
        embedding_dim = len(emb_cols)
        console.print(f"Window embedding dimension: {embedding_dim}")
        
        # Prepare data for parallel processing
        console.print("\n[bold]Preparing data for parallel indexing...[/bold]")
        window_data = []
        for idx, row in shingles_df.iterrows():
            window_id = f"{row['sample_id']}_{row['locus_id']}_{row['window_idx']}"
            embedding = np.array([row[col] for col in emb_cols])
            window_data.append((window_id, embedding))
        
        # Split into batches for parallel processing
        batch_size = max(1, len(window_data) // (self.n_jobs * 2))  # 2x jobs for better load balancing
        batches = [window_data[i:i + batch_size] for i in range(0, len(window_data), batch_size)]
        
        console.print(f"Processing {len(window_data):,} windows in {len(batches)} batches (batch size: {batch_size})")
        
        # Process batches in parallel
        console.print("\n[bold]Building indexes in parallel...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing batches...", total=len(batches))
            
            # Use joblib for parallel processing
            batch_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(process_window_batch)(
                    batch, self.config.discrete, self.config.continuous, embedding_dim
                ) for batch in batches
            )
            
            progress.advance(task, len(batches))
        
        # Merge results from all batches
        console.print("\n[bold]Merging parallel results...[/bold]")
        discrete_index = MinHashLSH(self.config.discrete)
        continuous_index = SignedRandomProjection(self.config.continuous, embedding_dim)
        
        # Merge discrete index buckets
        for discrete_buckets, _ in batch_results:
            for band_idx, band_buckets in discrete_buckets.items():
                if band_idx not in discrete_index.buckets:
                    discrete_index.buckets[band_idx] = {}
                for bucket_hash, window_ids in band_buckets.items():
                    if bucket_hash not in discrete_index.buckets[band_idx]:
                        discrete_index.buckets[band_idx][bucket_hash] = []
                    discrete_index.buckets[band_idx][bucket_hash].extend(window_ids)
        
        # Merge continuous index patterns
        for _, continuous_patterns in batch_results:
            for pattern, window_ids in continuous_patterns.items():
                if pattern not in continuous_index.index:
                    continuous_index.index[pattern] = []
                continuous_index.index[pattern].extend(window_ids)
        
        # Save indexes
        index_dir = self.manifest.artifact_path('indexes')
        index_dir.mkdir(exist_ok=True)
        
        discrete_path = index_dir / 'discrete_index.json'
        continuous_path = index_dir / 'continuous_index.json'
        
        console.print(f"\n[bold]Saving indexes...[/bold]")
        discrete_index.save(discrete_path)
        continuous_index.save(continuous_path)
        
        # Create index manifest
        index_manifest = {
            'version': '0.1.0',
            'created_at': pd.Timestamp.now().isoformat(),
            'config': {
                'discrete': self.config.discrete.dict(),
                'continuous': self.config.continuous.dict()
            },
            'statistics': {
                'total_windows': len(shingles_df),
                'embedding_dim': embedding_dim,
                'discrete_buckets': sum(len(buckets) for buckets in discrete_index.buckets.values()),
                'continuous_patterns': len(continuous_index.index),
                'discrete_index_file': str(discrete_path),
                'continuous_index_file': str(continuous_path)
            }
        }
        
        manifest_path = index_dir / 'index_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(index_manifest, f, indent=2)
        
        # Update main manifest
        self.manifest.add_artifact('indexes/discrete_index.json', {
            'type': 'discrete_index',
            'file_path': str(discrete_path),
            'num_buckets': index_manifest['statistics']['discrete_buckets']
        })
        
        self.manifest.add_artifact('indexes/continuous_index.json', {
            'type': 'continuous_index', 
            'file_path': str(continuous_path),
            'num_patterns': index_manifest['statistics']['continuous_patterns']
        })
        
        self.manifest.add_artifact('indexes/index_manifest.json', {
            'type': 'index_manifest',
            'file_path': str(manifest_path)
        })
        
        console.print(f"✓ Discrete index: {index_manifest['statistics']['discrete_buckets']:,} buckets")
        console.print(f"✓ Continuous index: {index_manifest['statistics']['continuous_patterns']:,} patterns")
        console.print(f"✓ Index manifest: {manifest_path}")


if __name__ == "__main__":
    # Test indexing functionality
    from .params import ELSAConfig
    
    config = ELSAConfig()
    print(f"Discrete config: {config.discrete}")
    print(f"Continuous config: {config.continuous}")