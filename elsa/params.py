"""
Configuration system for ELSA v2 with strict validation.
"""

from pathlib import Path
from typing import Optional, Union, Literal
import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data input/output configuration."""
    work_dir: Path = Field(default="./elsa_index", description="Working directory for artifacts")
    allow_overwrite: bool = Field(default=False, description="Allow overwriting existing artifacts")


class IngestConfig(BaseModel):
    """Gene calling and ingestion parameters."""
    gene_caller: Literal["prodigal", "metaprodigal", "none"] = Field(default="prodigal")
    prodigal_mode: Literal["single", "meta"] = Field(default="single")
    min_cds_aa: int = Field(default=0, description="Minimum CDS length in amino acids (0 = no filter)")
    keep_partial: bool = Field(default=False, description="Keep partial genes at contig ends")
    run_pfam: bool = Field(default=True, description="Run PFAM annotation with astra")


class PLMConfig(BaseModel):
    """Protein language model configuration."""
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
    jobs: Union[int, Literal["auto"]] = Field(default="auto", description="Number of parallel jobs")
    mmap: bool = Field(default=True, description="Use memory mapping")
    rng_seed: int = Field(default=17, description="Global random seed")


class ChainConfig(BaseModel):
    """Gene-level anchor chaining pipeline configuration."""
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
    """Create a default configuration file."""
    config = ELSAConfig()
    output_path = Path(output_path)

    config_dict = config.model_dump(mode="python")
    # Convert Path objects to strings for YAML
    if "data" in config_dict and "work_dir" in config_dict["data"]:
        config_dict["data"]["work_dir"] = str(config_dict["data"]["work_dir"])

    with open(output_path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)

    print(f"Created default configuration at: {output_path}")
