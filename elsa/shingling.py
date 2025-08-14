"""
Shingling system for ELSA - creates order-aware window embeddings.

Generates sliding windows over gene sequences with positional encoding.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Iterator, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from .params import ShingleConfig, ELSAConfig
from .projection import ProjectedProtein
from .manifest import ELSAManifest

console = Console()


@dataclass
class WindowEmbedding:
    """A window embedding with positional and strand information."""
    sample_id: str
    locus_id: str  # contig_id for genomic loci
    window_idx: int  # j in the spec
    strand_sign: int  # sigma: +1 or -1
    embedding: np.ndarray  # E_w: window embedding
    gene_ids: List[str]  # genes in this window
    gene_positions: List[int]  # relative positions within window


class ShingleSystem:
    """Creates order-aware window embeddings from projected proteins."""
    
    def __init__(self, config: ShingleConfig, work_dir: Path, manifest: ELSAManifest, full_config: ELSAConfig = None):
        self.config = config
        self.full_config = full_config  # For accessing phase2 settings
        self.work_dir = Path(work_dir)
        self.manifest = manifest
        
        # Create directories
        self.shingles_dir = self.work_dir / "shingles"
        self.shingles_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.windows_parquet_path = self.shingles_dir / "windows.parquet"
        
        # Precompute positional encodings
        self.pos_encodings = self._create_positional_encodings()
    
    def _create_positional_encodings(self) -> np.ndarray:
        """Create positional encodings for window positions."""
        # Simple learnable positional encoding (can be made more sophisticated)
        # For now, use sine/cosine positional encoding like in transformers
        
        pos_dim = self.config.pos_dim
        max_len = self.config.n  # Maximum window size
        
        pos_encoding = np.zeros((max_len, pos_dim))
        
        for pos in range(max_len):
            for i in range(0, pos_dim, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / pos_dim)))
                if i + 1 < pos_dim:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / pos_dim)))
        
        return pos_encoding
    
    def _group_proteins_by_locus(self, proteins: List[ProjectedProtein]) -> Dict[str, List[ProjectedProtein]]:
        """Group proteins by genomic locus (sample + contig)."""
        loci = defaultdict(list)
        
        for protein in proteins:
            locus_id = f"{protein.sample_id}_{protein.contig_id}"
            loci[locus_id].append(protein)
        
        # Sort proteins within each locus by genomic position
        for locus_id in loci:
            loci[locus_id].sort(key=lambda p: (p.start, p.end))
        
        return dict(loci)
    
    def _get_window_weights(self) -> np.ndarray:
        """Get weights for combining genes within a window."""
        n = self.config.n
        
        if self.config.weights == "uniform":
            return np.ones(n) / n
        
        elif self.config.weights == "triangular":
            # Higher weight for center genes
            weights = np.array([1.0 - abs(i - (n - 1) / 2) / ((n - 1) / 2 + 1e-6) for i in range(n)])
            return weights / weights.sum()
        
        elif self.config.weights == "gaussian":
            # Gaussian centered on middle
            center = (n - 1) / 2
            sigma = n / 6  # 3 sigma covers the window
            weights = np.exp(-0.5 * ((np.arange(n) - center) / sigma) ** 2)
            return weights / weights.sum()
        
        else:
            raise ValueError(f"Unknown weighting scheme: {self.config.weights}")
    
    def _create_window_embedding(self, genes: List[ProjectedProtein], 
                               window_idx: int) -> Tuple[np.ndarray, int]:
        """Create a single window embedding from a list of genes."""
        n_genes = len(genes)
        if n_genes == 0:
            return None, 1
        
        # Get embedding dimension
        embed_dim = genes[0].embedding.shape[0]
        
        # Determine strand sign (majority vote)
        strands = [gene.strand for gene in genes]
        strand_sign = 1 if sum(strands) >= 0 else -1
        
        # Get window weights
        weights = self._get_window_weights()
        
        # Create window embedding by weighted combination
        window_embedding = np.zeros(embed_dim + self.config.pos_dim + 1)  # +1 for strand
        
        # Add gene embeddings with weights
        for i, gene in enumerate(genes):
            if i < len(weights):
                # Gene embedding
                window_embedding[:embed_dim] += weights[i] * gene.embedding
                
                # Positional encoding
                if i < self.config.n:
                    window_embedding[embed_dim:embed_dim+self.config.pos_dim] += weights[i] * self.pos_encodings[i]
        
        # Add strand information
        if self.config.strand_flag == "signed":
            window_embedding[-1] = strand_sign
        elif self.config.strand_flag == "onehot":
            # Could expand to 2D one-hot, but for now use signed
            window_embedding[-1] = strand_sign
        
        return window_embedding, strand_sign
    
    def _create_locus_windows(self, locus_id: str, 
                            proteins: List[ProjectedProtein]) -> Iterator[WindowEmbedding]:
        """Create sliding windows for a single genomic locus."""
        if len(proteins) < self.config.n:
            # Skip loci with too few genes
            return
        
        # Create sliding windows
        for i in range(0, len(proteins) - self.config.n + 1, self.config.stride):
            window_genes = proteins[i:i + self.config.n]
            
            # Create window embedding
            window_emb, strand_sign = self._create_window_embedding(window_genes, i)
            if window_emb is None:
                continue
            
            # Extract metadata
            gene_ids = [gene.gene_id for gene in window_genes]
            gene_positions = list(range(len(window_genes)))
            
            yield WindowEmbedding(
                sample_id=window_genes[0].sample_id,
                locus_id=locus_id,
                window_idx=i,
                strand_sign=strand_sign,
                embedding=window_emb.astype(np.float16),  # Save memory
                gene_ids=gene_ids,
                gene_positions=gene_positions
            )
    
    def create_windows(self, proteins: List[ProjectedProtein]) -> List[WindowEmbedding]:
        """Create all window embeddings from projected proteins."""
        console.print(f"[bold blue]Creating shingle windows[/bold blue]")
        console.print(f"Window size: {self.config.n}, Stride: {self.config.stride}")
        console.print(f"Weighting: {self.config.weights}, Strand: {self.config.strand_flag}")
        
        # Group proteins by locus
        loci = self._group_proteins_by_locus(proteins)
        console.print(f"Processing {len(loci)} genomic loci...")
        
        all_windows = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Creating windows...", total=len(loci))
            
            for locus_id, locus_proteins in loci.items():
                progress.update(task, description=f"Processing {locus_id}")
                
                # Create windows for this locus
                locus_windows = list(self._create_locus_windows(locus_id, locus_proteins))
                all_windows.extend(locus_windows)
                
                progress.advance(task)
        
        console.print(f"✓ Created {len(all_windows):,} windows from {sum(len(p) for p in loci.values()):,} genes")
        return all_windows
    
    def save_windows_parquet(self, windows: List[WindowEmbedding]) -> None:
        """Save windows to Parquet format."""
        console.print("Saving windows to Parquet...")
        
        if not windows:
            console.print("[yellow]No windows to save[/yellow]")
            return
        
        # Prepare data for DataFrame
        embedding_dim = windows[0].embedding.shape[0]
        
        data = {
            'sample_id': [w.sample_id for w in windows],
            'locus_id': [w.locus_id for w in windows],
            'window_idx': [w.window_idx for w in windows],
            'strand_sign': [w.strand_sign for w in windows],
            'n_genes': [len(w.gene_ids) for w in windows],
        }
        
        # Add embedding columns
        embedding_matrix = np.array([w.embedding for w in windows])
        for i in range(embedding_dim):
            data[f'emb_{i:03d}'] = embedding_matrix[:, i]
        
        # Store gene IDs as JSON strings (for now - could be optimized)
        data['gene_ids'] = [','.join(w.gene_ids) for w in windows]
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_parquet(self.windows_parquet_path, compression='snappy', index=False)
        
        # Register artifact
        n_loci = len(set(w.locus_id for w in windows))
        n_samples = len(set(w.sample_id for w in windows))
        
        self.manifest.register_artifact(
            "windows", self.windows_parquet_path, "shingling",
            metadata={
                "n_windows": len(windows),
                "n_loci": n_loci,
                "n_samples": n_samples,
                "window_size": self.config.n,
                "stride": self.config.stride,
                "embedding_dim": embedding_dim,
                "weighting": self.config.weights,
                "strand_flag": self.config.strand_flag
            }
        )
        
        file_size_mb = self.windows_parquet_path.stat().st_size / (1024 * 1024)
        console.print(f"✓ Saved {len(windows):,} windows ({file_size_mb:.1f} MB)")
    
    def load_windows_parquet(self) -> pd.DataFrame:
        """Load windows from Parquet if available."""
        if self.windows_parquet_path.exists():
            console.print("Loading existing windows.parquet...")
            df = pd.read_parquet(self.windows_parquet_path)
            console.print(f"✓ Loaded {len(df):,} windows")
            return df
        return None
    
    def process_proteins(self, proteins: List[ProjectedProtein]) -> List[WindowEmbedding]:
        """Complete shingling pipeline: create windows → save."""
        
        # Check if windows already exist
        existing_windows = self.load_windows_parquet()
        if existing_windows is not None:
            console.print("Using existing windows")
            # Could convert back to WindowEmbedding objects if needed
            return []
        
        # Check if multi-scale windowing is enabled
        use_multiscale = (
            self.full_config and
            hasattr(self.full_config, 'phase2') and
            self.full_config.phase2 and 
            self.full_config.phase2.enable and 
            self.full_config.phase2.multiscale
        )
        
        if use_multiscale:
            console.print("🔄 Multi-scale windowing enabled")
            return self._process_multiscale_windows(proteins)
        else:
            # Standard single-scale windowing
            windows = self.create_windows(proteins)
            self.save_windows_parquet(windows)
            return windows
    
    def _process_multiscale_windows(self, proteins: List[ProjectedProtein]) -> List[WindowEmbedding]:
        """Process proteins using multi-scale windowing approach."""
        from .windowing import MultiScaleWindowGenerator
        
        # Create multi-scale window generator
        generator = MultiScaleWindowGenerator(self.full_config)
        
        # Generate multi-scale windows
        multiscale_windows, mappings = generator.generate_multiscale_windows(proteins)
        
        # Save multi-scale windows to separate directory
        multiscale_dir = self.work_dir / "multiscale_windows"
        generator.save_multiscale_windows(multiscale_windows, mappings, multiscale_dir)
        
        # Convert to standard WindowEmbedding format for compatibility
        # For now, we'll use the micro windows as the primary windows
        micro_windows = [w for w in multiscale_windows if w.scale == 'micro']
        
        standard_windows = []
        for i, window in enumerate(micro_windows):
            # Create gene IDs from window range (simplified for now)
            gene_ids = [f"gene_{j}" for j in range(window.start_gene_idx, window.end_gene_idx + 1)]
            # Create relative positions within window
            gene_positions = list(range(len(gene_ids)))
            
            standard_window = WindowEmbedding(
                sample_id=window.sample_id,
                locus_id=window.locus_id,
                window_idx=window.window_idx,
                embedding=window.embedding,
                gene_ids=gene_ids,
                gene_positions=gene_positions,
                strand_sign=1 if window.strand_composition.get('+', 0) >= window.strand_composition.get('-', 0) else -1
            )
            standard_windows.append(standard_window)
        
        # Save in standard format for downstream compatibility
        self.save_windows_parquet(standard_windows)
        
        console.print(f"✓ Multi-scale windowing complete: {len([w for w in multiscale_windows if w.scale == 'macro'])} macro, {len(micro_windows)} micro windows")
        console.print(f"✓ Using {len(standard_windows)} micro windows for indexing")
        
        return standard_windows
    
    def get_window_stats(self, windows: List[WindowEmbedding]) -> Dict[str, Any]:
        """Get statistics about windows."""
        if not windows:
            return {}
        
        # Collect statistics
        loci_counts = defaultdict(int)
        sample_counts = defaultdict(int)
        strand_counts = {1: 0, -1: 0}
        
        for window in windows:
            loci_counts[window.locus_id] += 1
            sample_counts[window.sample_id] += 1
            strand_counts[window.strand_sign] += 1
        
        return {
            "total_windows": len(windows),
            "n_loci": len(loci_counts),
            "n_samples": len(sample_counts),
            "avg_windows_per_locus": np.mean(list(loci_counts.values())),
            "strand_distribution": dict(strand_counts),
            "embedding_dim": windows[0].embedding.shape[0] if windows else 0
        }


if __name__ == "__main__":
    # Test shingling system
    from .params import ELSAConfig
    from .manifest import ELSAManifest
    
    config = ELSAConfig()
    manifest = ELSAManifest(Path("./test_elsa_index"))
    
    shingle_system = ShingleSystem(config.shingles, Path("./test_elsa_index"), manifest)
    
    print(f"Window size: {config.shingles.n}")
    print(f"Stride: {config.shingles.stride}")
    print(f"Positional dim: {config.shingles.pos_dim}")
    print(f"Weights: {config.shingles.weights}")
    print(f"Strand flag: {config.shingles.strand_flag}")