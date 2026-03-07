"""
PCA projection and data storage for ELSA embeddings.

Handles dimensionality reduction, normalization, and efficient storage.

PROJECTION WORKFLOW
-------------------
Normal usage: `elsa embed` runs the complete pipeline:
  1. Gene calling → 2. PLM embedding → 3. PCA projection → 4. Shingling

For combining separately-embedded datasets (e.g., cross-species comparison):
  1. Run `elsa embed --save-raw` on each dataset separately
  2. Merge raw embedding parquets manually (concat genes_raw.parquet files)
  3. Run `elsa project` on combined raw embeddings to fit unified PCA

This ensures all datasets share the same PCA projection space.
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
        self.raw_embeddings_path = self.ingest_dir / "genes_raw.parquet"

        # Models
        self.pca_model = None
        self.scaler = None
    
    @property
    def skip_pca(self) -> bool:
        """Return True if PCA should be skipped (project_to_D == 0)."""
        return self.config.project_to_D == 0

    def fit_projection(self, embeddings: List[ProteinEmbedding],
                      subsample_size: int = 50000) -> None:
        """Fit PCA projection on a subsample of embeddings."""
        if self.skip_pca:
            console.print("[bold blue]Skipping PCA (project_to_D=0) — using raw embeddings[/bold blue]")
            return

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
        """Load existing PCA model if available.

        Priority:
        1. Frozen PCA from config (frozen_pca_path) - pre-fitted on e.g. UniRef50
        2. Local PCA model from work_dir (fitted on this dataset)
        """
        # Check for frozen (pre-fitted) PCA model first
        if self.config.frozen_pca_path:
            frozen_path = Path(self.config.frozen_pca_path)
            if frozen_path.exists():
                console.print(f"[bold green]Loading frozen PCA model: {frozen_path}[/bold green]")
                with open(frozen_path, 'rb') as f:
                    self.pca_model = pickle.load(f)
                console.print(
                    f"  Frozen PCA: {self.pca_model.n_features_in_}D -> "
                    f"{self.pca_model.n_components_}D, "
                    f"explained variance: {self.pca_model.explained_variance_ratio_.sum():.3f}"
                )
                # Also check for a companion scaler
                frozen_scaler = frozen_path.parent / "scaler.pkl"
                if frozen_scaler.exists():
                    with open(frozen_scaler, 'rb') as f:
                        self.scaler = pickle.load(f)
                return True
            else:
                console.print(f"[yellow]Warning: frozen_pca_path not found: {frozen_path}[/yellow]")
                console.print("[yellow]Falling back to per-dataset PCA fitting[/yellow]")

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
        """Project embeddings to target dimensionality (or pass through if PCA skipped)."""
        if not self.skip_pca and self.pca_model is None:
            raise ValueError("PCA model not fitted or loaded")

        # Create embedding matrix
        embedding_matrix = np.array([emb.embedding for emb in embeddings])

        if self.skip_pca:
            console.print(f"Using raw {embedding_matrix.shape[1]}D embeddings (no PCA)...")
            projected_matrix = embedding_matrix
        else:
            console.print(f"Projecting {len(embeddings):,} embeddings...")

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
            projected_matrix = projected_matrix / (norms + 1e-8)

        # Create projected proteins
        projected_proteins = []
        for emb, projected_emb in zip(embeddings, projected_matrix):
            projected = ProjectedProtein(
                sample_id=emb.sample_id,
                contig_id="unknown",
                gene_id=emb.gene_id,
                start=0,
                end=0,
                strand=1,
                embedding=projected_emb.astype(np.float16),
                original_length=emb.sequence_length
            )
            projected_proteins.append(projected)

        console.print(f"✓ Output dimension: {projected_matrix.shape[1]}D")
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

    def save_raw_embeddings(
        self,
        embeddings: List[ProteinEmbedding],
        protein_metadata: Dict[str, Dict] = None,
    ) -> Path:
        """Save raw (unprojected) embeddings to Parquet for later projection.

        Use this when you need to:
        - Combine multiple datasets into a shared PCA space
        - Re-project with different target dimensions
        - Keep original embeddings for debugging

        Args:
            embeddings: List of raw ProteinEmbedding objects (not yet projected)
            protein_metadata: Optional dict mapping gene_id -> {contig_id, start, end, strand}

        Returns:
            Path to the saved raw parquet file
        """
        console.print(f"Saving {len(embeddings):,} raw embeddings to Parquet...")

        # Prepare data for DataFrame
        data = {
            "sample_id": [emb.sample_id for emb in embeddings],
            "gene_id": [emb.gene_id for emb in embeddings],
            "sequence_length": [emb.sequence_length for emb in embeddings],
        }

        # Add metadata if provided
        if protein_metadata:
            data["contig_id"] = [
                protein_metadata.get(emb.gene_id, {}).get("contig_id", "unknown")
                for emb in embeddings
            ]
            data["start"] = [
                protein_metadata.get(emb.gene_id, {}).get("start", 0)
                for emb in embeddings
            ]
            data["end"] = [
                protein_metadata.get(emb.gene_id, {}).get("end", 0)
                for emb in embeddings
            ]
            data["strand"] = [
                protein_metadata.get(emb.gene_id, {}).get("strand", 1)
                for emb in embeddings
            ]
        else:
            data["contig_id"] = ["unknown"] * len(embeddings)
            data["start"] = [0] * len(embeddings)
            data["end"] = [0] * len(embeddings)
            data["strand"] = [1] * len(embeddings)

        # Add raw embedding columns (full dimension, e.g., 480 for ESM2-t12)
        embedding_dim = embeddings[0].embedding.shape[0]
        embedding_matrix = np.array([emb.embedding for emb in embeddings])

        # Store as float32 for raw (not float16 like projected)
        for i in range(embedding_dim):
            data[f"raw_{i:04d}"] = embedding_matrix[:, i].astype(np.float32)

        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_parquet(self.raw_embeddings_path, compression="snappy", index=False)

        # Register artifact
        self.manifest.register_artifact(
            "genes_raw",
            self.raw_embeddings_path,
            "embedding",
            metadata={
                "n_genes": len(embeddings),
                "embedding_dim": embedding_dim,
                "total_aa": sum(emb.sequence_length for emb in embeddings),
                "samples": list(set(emb.sample_id for emb in embeddings)),
            },
        )

        file_size_mb = self.raw_embeddings_path.stat().st_size / (1024 * 1024)
        console.print(f"✓ Saved {len(embeddings):,} raw embeddings ({file_size_mb:.1f} MB)")
        console.print(f"  Raw embedding dimension: {embedding_dim}")
        console.print(f"  File: {self.raw_embeddings_path}")

        return self.raw_embeddings_path

    def load_raw_embeddings(self, path: Path = None) -> Tuple[List[ProteinEmbedding], Dict[str, Dict]]:
        """Load raw embeddings from Parquet file.

        Args:
            path: Path to raw parquet file. If None, uses default location.

        Returns:
            Tuple of (list of ProteinEmbedding, dict of protein_metadata)
        """
        raw_path = path or self.raw_embeddings_path
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw embeddings not found: {raw_path}")

        console.print(f"Loading raw embeddings from {raw_path}...")
        df = pd.read_parquet(raw_path)

        # Extract embedding columns
        raw_cols = sorted([c for c in df.columns if c.startswith("raw_")])
        if not raw_cols:
            raise ValueError("No raw embedding columns found in parquet")

        embedding_matrix = df[raw_cols].values

        # Build ProteinEmbedding objects
        embeddings = []
        protein_metadata = {}

        for i, row in df.iterrows():
            emb = ProteinEmbedding(
                sample_id=row["sample_id"],
                gene_id=row["gene_id"],
                embedding=embedding_matrix[i],
                sequence_length=row["sequence_length"],
            )
            embeddings.append(emb)

            protein_metadata[row["gene_id"]] = {
                "contig_id": row.get("contig_id", "unknown"),
                "start": row.get("start", 0),
                "end": row.get("end", 0),
                "strand": row.get("strand", 1),
            }

        console.print(f"✓ Loaded {len(embeddings):,} raw embeddings (dim={embedding_matrix.shape[1]})")
        return embeddings, protein_metadata

    def process_embeddings(self, embeddings: List[ProteinEmbedding],
                          protein_metadata: Dict[str, Dict] = None) -> List[ProjectedProtein]:
        """Complete projection pipeline: fit → project → save."""

        if self.skip_pca:
            # No PCA needed — go straight to projection (passthrough + L2 norm)
            self.fit_projection(embeddings)  # logs skip message
        elif not self.load_projection():
            # Fit new projection
            self.fit_projection(embeddings)

        # Project embeddings (or pass through if skip_pca)
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