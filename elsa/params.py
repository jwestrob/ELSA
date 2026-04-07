"""
Configuration system for ELSA v2 with strict validation.
"""

from pathlib import Path
from typing import Optional, Union, Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataConfig(BaseModel):
    """Data input/output configuration."""
    model_config = ConfigDict(extra="forbid")
    work_dir: Path = Field(default="./elsa_index", description="Working directory for artifacts")
    allow_overwrite: bool = Field(default=False, description="Allow overwriting existing artifacts")


class IngestConfig(BaseModel):
    """Gene calling and ingestion parameters."""
    model_config = ConfigDict(extra="forbid")
    gene_caller: Literal["prodigal", "metaprodigal", "none"] = Field(default="prodigal")
    prodigal_mode: Literal["single", "meta"] = Field(default="single")
    min_cds_aa: int = Field(default=60, description="Minimum CDS length in amino acids (0 = no filter)")
    keep_partial: bool = Field(default=False, description="Keep partial genes at contig ends")
    run_pfam: bool = Field(default=False, description="Run PFAM annotation with astra (requires astra to be installed)")


class PLMConfig(BaseModel):
    """Protein language model configuration."""
    model_config = ConfigDict(extra="forbid")
    model: Literal["esm2_t33", "esm2_t12", "prot_t5", "glm2_650m", "glm2_150m"] = Field(default="esm2_t12")
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(default="auto")
    batch_amino_acids: int = Field(default=16000, description="Approximate per-batch residue budget")
    fp16: bool = Field(default=True, description="Use half precision")
    project_to_D: int = Field(default=0, description="PCA target dimension (0 = skip PCA, use raw embeddings)")
    l2_normalize: bool = Field(default=True, description="L2 normalize embeddings")
    frozen_pca_path: Optional[str] = Field(default=None, description="Path to pre-fitted PCA model (e.g. from UniRef50). Skips per-dataset PCA fitting.")

    @field_validator("batch_amino_acids")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1000 or v > 100000:
            raise ValueError("batch_amino_acids should be between 1000 and 100000")
        return v


class SystemConfig(BaseModel):
    """System resource configuration."""
    model_config = ConfigDict(extra="forbid")
    jobs: Union[int, Literal["auto"]] = Field(default="auto", description="Number of parallel jobs")
    mmap: bool = Field(default=True, description="Use memory mapping")
    rng_seed: int = Field(default=17, description="Global random seed")


def resolve_jobs(jobs: Union[int, str]) -> int:
    """Resolve ``system.jobs`` to a concrete thread count.

    Returns the configured integer, or ``os.cpu_count()`` when *jobs* is
    ``"auto"``.  Always returns at least 1.
    """
    import os
    if isinstance(jobs, int):
        return max(1, jobs)
    return os.cpu_count() or 4


class ChainConfig(BaseModel):
    """Gene-level anchor chaining pipeline configuration."""
    model_config = ConfigDict(extra="forbid")
    # Index backend
    index_backend: Literal["auto", "hnsw", "faiss_ivfflat", "faiss_ivfpq", "faiss_ivfsq", "faiss_flat", "sklearn"] = Field(
        default="faiss_ivfflat", description="ANN index backend"
    )
    faiss_nprobe: int = Field(default=32, description="IVF clusters to search (higher = better recall, slower)")

    # HNSW parameters
    hnsw_k: int = Field(default=50, description="Number of neighbors to retrieve per gene")
    hnsw_m: int = Field(default=32, description="HNSW M parameter (connections per node)")
    hnsw_ef_construction: int = Field(default=200, description="HNSW build quality parameter")
    hnsw_ef_search: int = Field(default=128, description="HNSW query quality parameter")

    # Similarity and chaining
    similarity_threshold: float = Field(default=0.85, description="Minimum cosine similarity for anchor")
    max_gap_genes: int = Field(default=2, description="Maximum gap between chain members (genes)")
    min_chain_size: int = Field(default=2, description="Minimum anchors per chain")

    # Gap penalty (Phase 2 feature)
    gap_penalty_scale: float = Field(default=0.0, description="Concave gap penalty scale (0 = disabled)")

    # Clustering
    jaccard_tau: float = Field(default=0.3, description="Minimum Jaccard similarity for overlap clustering")
    mutual_k: int = Field(default=5, description="Mutual top-k parameter for clustering")
    df_max: int = Field(default=500, description="Maximum document frequency")
    min_genome_support: int = Field(default=2, description="Minimum genomes per cluster")

    @field_validator("similarity_threshold", "jaccard_tau")
    @classmethod
    def validate_thresholds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    @field_validator("hnsw_m")
    @classmethod
    def validate_hnsw_m(cls, v):
        if v < 2 or v > 100:
            raise ValueError("hnsw_m should be between 2 and 100")
        return v


class ELSAConfig(BaseModel):
    """Complete ELSA v2 configuration."""
    model_config = ConfigDict(extra="forbid")
    data: DataConfig = Field(default_factory=DataConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    plm: PLMConfig = Field(default_factory=PLMConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    chain: ChainConfig = Field(default_factory=ChainConfig)


def load_config(config_path: Union[str, Path]) -> ELSAConfig:
    """Load and validate configuration from YAML file.

    Ignores legacy config sections that no longer exist in v2
    (shingles, discrete, continuous, score, chain (old DTW),
    dtw, window, phase2, cassette_mode, analyze).
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Map legacy config locations to new structure
    # Old: analyze.micro_chain.* -> New: chain.*
    if "analyze" in data and "micro_chain" in data.get("analyze", {}):
        if "chain" not in data:
            data["chain"] = data["analyze"]["micro_chain"]

    # Filter to only known top-level keys
    known_keys = {"data", "ingest", "plm", "system", "chain"}
    filtered = {k: v for k, v in data.items() if k in known_keys}

    return ELSAConfig(**filtered)


def create_default_config(output_path: Union[str, Path]) -> None:
    """Create a default configuration file with comments."""
    cfg = ELSAConfig()
    output_path = Path(output_path)

    template = f"""\
# =============================================================================
# ELSA Configuration
# Embedding Locus Search and Alignment — syntenic block discovery
# Full documentation: https://github.com/jwestrob/ELSA
# =============================================================================

# ---- Data paths ----
data:
  # Directory where ELSA stores all intermediate and output artifacts
  work_dir: {cfg.data.work_dir}
  # Allow overwriting existing artifacts (set true to re-run without cleaning)
  allow_overwrite: {str(cfg.data.allow_overwrite).lower()}

# ---- Gene calling ----
ingest:
  # Gene caller: "prodigal" (isolates), "metaprodigal" (metagenomes), or "none" (skip)
  gene_caller: {cfg.ingest.gene_caller}
  # Prodigal mode: "single" for isolate genomes, "meta" for metagenomes
  prodigal_mode: {cfg.ingest.prodigal_mode}
  # Minimum protein length in amino acids (filters short spurious ORFs)
  min_cds_aa: {cfg.ingest.min_cds_aa}
  # Keep partial genes at contig ends
  keep_partial: {str(cfg.ingest.keep_partial).lower()}
  # Run PFAM domain annotation (requires astra; set true if installed)
  run_pfam: {str(cfg.ingest.run_pfam).lower()}

# ---- Protein language model ----
plm:
  # Model: esm2_t12 (fast, 480D), esm2_t33 (accurate, 1280D),
  #        prot_t5 (1024D), glm2_650m, glm2_150m
  # esm2_t12 recommended — chaining algorithm dominates performance
  model: {cfg.plm.model}
  # Device: "auto" detects GPU (MPS on Mac, CUDA on Linux), falls back to CPU
  device: {cfg.plm.device}
  # Use half precision (faster, lower memory; disable if you see NaN errors)
  fp16: {str(cfg.plm.fp16).lower()}
  # L2-normalize embeddings (required for cosine similarity via dot product)
  l2_normalize: {str(cfg.plm.l2_normalize).lower()}
  # PCA target dimension: 0 = use raw embeddings (recommended)
  project_to_D: {cfg.plm.project_to_D}

  # --- Advanced PLM settings ---
  # Approximate amino acids per batch (tune down if you hit OOM)
  # batch_amino_acids: {cfg.plm.batch_amino_acids}
  # Path to pre-fitted PCA model (e.g. from UniRef50); skips per-dataset PCA
  # frozen_pca_path: null

# ---- Chaining parameters ----
chain:
  # Minimum cosine similarity between gene embeddings to call an anchor pair
  similarity_threshold: {cfg.chain.similarity_threshold}
  # Maximum gene-position gap allowed within a chain (tighter = more precise blocks)
  max_gap_genes: {cfg.chain.max_gap_genes}
  # Minimum number of anchor genes to keep a chain as a syntenic block
  min_chain_size: {cfg.chain.min_chain_size}
  # Minimum number of genomes a cluster must span to be reported
  min_genome_support: {cfg.chain.min_genome_support}

  # --- Advanced chaining settings ---
  # Concave gap penalty scale (0 = hard cutoff; >0 = minimap2-style penalty)
  # gap_penalty_scale: {cfg.chain.gap_penalty_scale}
  # Jaccard overlap threshold for merging blocks into clusters
  # jaccard_tau: {cfg.chain.jaccard_tau}
  # Number of nearest neighbors to retrieve per gene
  # hnsw_k: {cfg.chain.hnsw_k}
  # ANN index backend: auto, faiss_ivfflat (default), hnsw, faiss_flat, sklearn
  # index_backend: {cfg.chain.index_backend}
  # IVF clusters to probe at query time (higher = better recall, slower)
  # faiss_nprobe: {cfg.chain.faiss_nprobe}
  # HNSW graph connectivity (higher = better recall, more memory)
  # hnsw_m: {cfg.chain.hnsw_m}
  # HNSW index build quality
  # hnsw_ef_construction: {cfg.chain.hnsw_ef_construction}
  # HNSW query quality
  # hnsw_ef_search: {cfg.chain.hnsw_ef_search}
  # Mutual top-k for clustering
  # mutual_k: {cfg.chain.mutual_k}
  # Maximum document frequency for clustering
  # df_max: {cfg.chain.df_max}

# ---- System ----
# Thread count is set via the CLI flag --jobs / -j, not in this file.
system:
  # Random seed for reproducibility
  rng_seed: {cfg.system.rng_seed}
  # Use memory-mapped arrays for large datasets
  # mmap: true
"""

    with open(output_path, "w") as f:
        f.write(template)

    print(f"Created default configuration at: {output_path}")
