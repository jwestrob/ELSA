"""
PCA projection and data storage for ELSA embeddings.

Handles dimensionality reduction, normalization, and efficient storage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .embeddings import ProteinEmbedding
from .params import PLMConfig
from .manifest import ELSAManifest

console = Console()


@dataclass
class ProjectedProtein:
    """A protein with projected embedding."""
    sample_id: str
    contig_id: str  
    gene_id: str
    start: int
    end: int
    strand: int
    embedding: np.ndarray  # Projected to target dimension
    original_length: int


class ProjectionSystem:
    """Handles PCA projection and storage of protein embeddings."""
    
    def __init__(self, config: PLMConfig, work_dir: Path, manifest: ELSAManifest):
        self.config = config
        self.work_dir = Path(work_dir)
        self.manifest = manifest
        
        # Create directories
        self.ingest_dir = self.work_dir / "ingest"
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.pca_model_path = self.ingest_dir / "pca_model.pkl"
        self.scaler_path = self.ingest_dir / "scaler.pkl" 
        self.genes_parquet_path = self.ingest_dir / "genes.parquet"
        
        # Models
        self.pca_model = None
        self.scaler = None
    
    def fit_projection(self, embeddings: List[ProteinEmbedding], 
                      subsample_size: int = 50000) -> None:
        """Fit PCA projection on a subsample of embeddings."""
        console.print(f"[bold blue]Fitting PCA projection to {self.config.project_to_D}D[/bold blue]")
        
        if not embeddings:
            raise ValueError("No embeddings provided for PCA fitting")
        
        # Collect embedding data
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        console.print(f"Input embeddings: {embedding_matrix.shape}")
        
        # Subsample for PCA fitting if needed
        if len(embeddings) > subsample_size:
            console.print(f"Subsampling {subsample_size} embeddings for PCA fitting...")
            indices = np.random.choice(len(embeddings), subsample_size, replace=False)
            fit_matrix = embedding_matrix[indices]
        else:
            fit_matrix = embedding_matrix
        
        # Optional standardization (usually not needed for PLM embeddings)
        if hasattr(self.config, 'standardize') and self.config.standardize:
            console.print("Fitting standardization scaler...")
            self.scaler = StandardScaler()
            fit_matrix = self.scaler.fit_transform(fit_matrix)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Fit PCA
        console.print(f"Fitting PCA: {fit_matrix.shape[1]}D → {self.config.project_to_D}D...")
        self.pca_model = PCA(
            n_components=self.config.project_to_D,
            random_state=42  # For reproducibility
        )
        self.pca_model.fit(fit_matrix)
        
        # Save PCA model
        with open(self.pca_model_path, 'wb') as f:
            pickle.dump(self.pca_model, f)
        
        # Register artifacts
        self.manifest.register_artifact(
            "pca_model", self.pca_model_path, "projection",
            metadata={
                "input_dim": fit_matrix.shape[1], 
                "output_dim": self.config.project_to_D,
                "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
                "total_explained_variance": float(self.pca_model.explained_variance_ratio_.sum())
            }
        )
        
        if self.scaler:
            self.manifest.register_artifact(
                "scaler", self.scaler_path, "projection",
                metadata={"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist()}
            )
        
        explained_var = self.pca_model.explained_variance_ratio_.sum()
        console.print(f"✓ PCA fitted - explained variance: {explained_var:.3f}")
    
    def load_projection(self) -> bool:
        """Load existing PCA model if available."""
        if self.pca_model_path.exists():
            console.print("Loading existing PCA model...")
            with open(self.pca_model_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            
            if self.scaler_path.exists():
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            return True
        return False
    
    def project_embeddings(self, embeddings: List[ProteinEmbedding]) -> List[ProjectedProtein]:
        """Project embeddings to target dimensionality."""
        if self.pca_model is None:
            raise ValueError("PCA model not fitted or loaded")
        
        console.print(f"Projecting {len(embeddings):,} embeddings...")
        
        # Create embedding matrix
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Apply standardization if fitted
        if self.scaler is not None:
            embedding_matrix = self.scaler.transform(embedding_matrix)
        
        # Apply PCA projection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Applying PCA transformation...", total=None)
            
            projected_matrix = self.pca_model.transform(embedding_matrix)
            
            progress.update(task, description="✓ PCA transformation complete")
        
        # Apply L2 normalization if configured
        if self.config.l2_normalize:
            norms = np.linalg.norm(projected_matrix, axis=1, keepdims=True)
            projected_matrix = projected_matrix / (norms + 1e-8)  # Avoid division by zero
        
        # Create projected proteins
        projected_proteins = []
        for emb, projected_emb in zip(embeddings, projected_matrix):
            # Note: We need to get gene coordinates from somewhere
            # For now, using placeholders - this should be improved
            projected = ProjectedProtein(
                sample_id=emb.sample_id,
                contig_id="unknown",  # TODO: get from gene metadata
                gene_id=emb.gene_id,
                start=0,  # TODO: get from gene metadata
                end=0,    # TODO: get from gene metadata  
                strand=1, # TODO: get from gene metadata
                embedding=projected_emb.astype(np.float16),  # Save space
                original_length=emb.sequence_length
            )
            projected_proteins.append(projected)
        
        console.print(f"✓ Projected to {projected_matrix.shape[1]}D")
        return projected_proteins
    
    def save_genes_parquet(self, projected_proteins: List[ProjectedProtein]) -> None:
        """Save projected proteins to Parquet format."""
        console.print("Saving genes to Parquet...")
        
        # Prepare data for DataFrame
        data = {
            'sample_id': [p.sample_id for p in projected_proteins],
            'contig_id': [p.contig_id for p in projected_proteins], 
            'gene_id': [p.gene_id for p in projected_proteins],
            'start': [p.start for p in projected_proteins],
            'end': [p.end for p in projected_proteins],
            'strand': [p.strand for p in projected_proteins],
            'original_length': [p.original_length for p in projected_proteins]
        }
        
        # Add embedding columns
        embedding_dim = projected_proteins[0].embedding.shape[0]
        embedding_matrix = np.array([p.embedding for p in projected_proteins])
        
        for i in range(embedding_dim):
            data[f'emb_{i:03d}'] = embedding_matrix[:, i]
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_parquet(self.genes_parquet_path, compression='snappy', index=False)
        
        # Register artifact
        self.manifest.register_artifact(
            "genes", self.genes_parquet_path, "projection",
            metadata={
                "n_genes": len(projected_proteins),
                "embedding_dim": embedding_dim,
                "total_aa": sum(p.original_length for p in projected_proteins),
                "samples": list(set(p.sample_id for p in projected_proteins))
            }
        )
        
        file_size_mb = self.genes_parquet_path.stat().st_size / (1024 * 1024)
        console.print(f"✓ Saved {len(projected_proteins):,} genes ({file_size_mb:.1f} MB)")
    
    def load_genes_parquet(self) -> Optional[pd.DataFrame]:
        """Load genes from Parquet if available."""
        if self.genes_parquet_path.exists():
            console.print("Loading existing genes.parquet...")
            df = pd.read_parquet(self.genes_parquet_path)
            console.print(f"✓ Loaded {len(df):,} genes")
            return df
        return None
    
    def process_embeddings(self, embeddings: List[ProteinEmbedding], 
                          protein_metadata: Dict[str, Dict] = None) -> List[ProjectedProtein]:
        """Complete projection pipeline: fit → project → save."""
        
        # Try to load existing model
        if not self.load_projection():
            # Fit new projection
            self.fit_projection(embeddings)
        
        # Project embeddings
        projected_proteins = self.project_embeddings(embeddings)
        
        # Add metadata if provided
        if protein_metadata:
            for projected in projected_proteins:
                if projected.gene_id in protein_metadata:
                    metadata = protein_metadata[projected.gene_id]
                    projected.contig_id = metadata.get('contig_id', projected.contig_id)
                    projected.start = metadata.get('start', projected.start)
                    projected.end = metadata.get('end', projected.end)
                    projected.strand = metadata.get('strand', projected.strand)
        
        # Save to Parquet
        self.save_genes_parquet(projected_proteins)
        
        return projected_proteins
    
    def get_projection_stats(self) -> Dict[str, Any]:
        """Get statistics about the projection."""
        if self.pca_model is None:
            return {}
        
        return {
            "input_dim": self.pca_model.n_features_in_,
            "output_dim": self.pca_model.n_components_,
            "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(self.pca_model.explained_variance_ratio_.sum()),
            "singular_values": self.pca_model.singular_values_.tolist()
        }


if __name__ == "__main__":
    # Test projection system
    from .params import ELSAConfig
    from .manifest import ELSAManifest
    
    config = ELSAConfig()
    manifest = ELSAManifest(Path("./test_elsa_index"))
    
    projection_system = ProjectionSystem(config.plm, Path("./test_elsa_index"), manifest)
    print(f"Target dimension: {config.plm.project_to_D}")
    print(f"L2 normalize: {config.plm.l2_normalize}")