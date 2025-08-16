"""
Configuration system for ELSA with strict validation.
"""

from pathlib import Path
from typing import List, Optional, Union, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Data input/output configuration."""
    work_dir: Path = Field(default="./elsa_index", description="Working directory for artifacts")
    allow_overwrite: bool = Field(default=False, description="Allow overwriting existing artifacts")


class IngestConfig(BaseModel):
    """Gene calling and ingestion parameters."""
    gene_caller: Literal["prodigal", "metaprodigal", "none"] = Field(default="prodigal")
    prodigal_mode: Literal["single", "meta"] = Field(default="single")
    min_cds_aa: int = Field(default=60, description="Minimum CDS length in amino acids")
    keep_partial: bool = Field(default=False, description="Keep partial genes at contig ends")
    
    # PFAM annotation settings
    run_pfam: bool = Field(default=True, description="Run PFAM annotation with astra")


class PLMConfig(BaseModel):
    """Protein language model configuration."""
    model: Literal["esm2_t33", "esm2_t12", "prot_t5"] = Field(default="esm2_t12")
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(default="auto")
    batch_amino_acids: int = Field(default=16000, description="Approximate per-batch residue budget")
    fp16: bool = Field(default=True, description="Use half precision")
    project_to_D: int = Field(default=256, description="PCA target dimension")
    l2_normalize: bool = Field(default=True, description="L2 normalize embeddings")
    
    @field_validator("batch_amino_acids")
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1000 or v > 100000:
            raise ValueError("batch_amino_acids should be between 1000 and 100000")
        return v


class ShingleConfig(BaseModel):
    """Window shingling parameters."""
    n: int = Field(default=5, description="Window size (number of genes)")
    stride: int = Field(default=1, description="Window stride")
    pos_dim: int = Field(default=16, description="Positional encoding dimension")
    weights: Literal["triangular", "uniform", "gaussian"] = Field(default="triangular")
    strand_flag: Literal["signed", "onehot"] = Field(default="signed")


class DiscreteConfig(BaseModel):
    """Discrete indexing (MinHash LSH) parameters."""
    K: int = Field(default=4096, description="Codebook centroids")
    minhash_hashes: int = Field(default=192, description="Number of MinHash functions")
    bands_rows: List[int] = Field(default=[24, 8], description="LSH bands × rows")
    emit_skipgram: bool = Field(default=True, description="Include skip-grams")
    idf_min_df: int = Field(default=5, description="Minimum document frequency")
    idf_max_df_frac: float = Field(default=0.05, description="Maximum document frequency fraction")
    
    @field_validator("bands_rows")
    @classmethod
    def validate_bands_rows(cls, v):
        if len(v) != 2 or v[0] * v[1] <= 0:
            raise ValueError("bands_rows must be [bands, rows] with positive integers")
        return v


class ContinuousConfig(BaseModel):
    """Continuous indexing (SRP) parameters."""
    srp_bits: int = Field(default=256, description="Signed random projection bits")
    srp_seed: int = Field(default=13, description="SRP random seed")
    hnsw_enable: bool = Field(default=False, description="Enable HNSW approximate search")


class ChainConfig(BaseModel):
    """Collinear chaining parameters."""
    offset_band: int = Field(default=10, description="Allowed window drift")
    gap_open: float = Field(default=2.0, description="Gap opening penalty")
    gap_extend: float = Field(default=0.5, description="Gap extension penalty")
    slope_penalty: float = Field(default=0.05, description="Slope deviation penalty")


class DTWConfig(BaseModel):
    """Dynamic time warping refinement."""
    enable: bool = Field(default=True, description="Enable DTW refinement")
    band: int = Field(default=10, description="DTW band width")


class ScoreConfig(BaseModel):
    """Scoring parameters."""
    alpha: float = Field(default=1.0, description="Anchor strength weight")
    beta: float = Field(default=0.1, description="LIS length weight")
    gamma: float = Field(default=0.2, description="Gap penalty weight")
    delta: float = Field(default=0.1, description="Offset variance penalty weight")
    fdr_target: float = Field(default=0.01, description="False discovery rate target")
    
    @field_validator("fdr_target")
    @classmethod
    def validate_fdr(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError("fdr_target must be between 0 and 1")
        return v


class SystemConfig(BaseModel):
    """System resource configuration."""
    jobs: Union[int, Literal["auto"]] = Field(default="auto", description="Number of parallel jobs")
    mmap: bool = Field(default=True, description="Use memory mapping")
    rng_seed: int = Field(default=17, description="Global random seed")


class WindowConfig(BaseModel):
    """Multi-scale windowing configuration."""
    micro: 'MicroWindowConfig' = Field(default_factory=lambda: MicroWindowConfig())
    macro: 'MacroWindowConfig' = Field(default_factory=lambda: MacroWindowConfig())
    adaptive: 'AdaptiveWindowConfig' = Field(default_factory=lambda: AdaptiveWindowConfig())


class MicroWindowConfig(BaseModel):
    """Micro window configuration."""
    size: int = Field(default=3, description="Genes per micro window")
    stride: int = Field(default=1, description="Micro window stride")


class MacroWindowConfig(BaseModel):
    """Macro window configuration."""
    size: int = Field(default=4, description="Micro windows per macro window")
    stride: int = Field(default=2, description="Macro window stride")


class AdaptiveWindowConfig(BaseModel):
    """Adaptive windowing configuration."""
    enable: bool = Field(default=False, description="Enable adaptive windowing")


class Phase2Config(BaseModel):
    """Phase-2 feature flags."""
    enable: bool = Field(default=True, description="Enable phase-2 features")
    weighted_sketch: bool = Field(default=True, description="Use weighted sketching")
    multiscale: bool = Field(default=True, description="Enable multiscale windowing")
    flip_dp: bool = Field(default=False, description="Flip dynamic programming")
    calibration: bool = Field(default=True, description="Enable FDR calibration")
    hnsw: bool = Field(default=False, description="Enable HNSW indexing")


class CassetteModeConfig(BaseModel):
    """Cassette mode configuration for fine-grained syntenic blocks."""
    enable: bool = Field(default=False, description="Enable cassette mode")
    anchors: 'CassetteAnchorsConfig' = Field(default_factory=lambda: CassetteAnchorsConfig())
    chain: 'CassetteChainConfig' = Field(default_factory=lambda: CassetteChainConfig())
    segmenter: 'CassetteSegmenterConfig' = Field(default_factory=lambda: CassetteSegmenterConfig())


class CassetteAnchorsConfig(BaseModel):
    """Cassette mode anchor filtering configuration."""
    cosine_min: float = Field(default=0.91, description="Minimum cosine similarity threshold")
    jaccard_min: float = Field(default=0.30, description="Minimum Jaccard similarity threshold")
    reciprocal_topk: int = Field(default=2, description="Reciprocal top-k filtering")
    blacklist_top_pct: float = Field(default=1.0, description="Blacklist top collision windows percentage")
    lambda_jaccard: float = Field(default=0.5, description="Jaccard weight in chain scoring")
    
    @field_validator("cosine_min", "jaccard_min")
    @classmethod
    def validate_similarity_thresholds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity thresholds must be between 0 and 1")
        return v


class CassetteChainConfig(BaseModel):
    """Cassette mode chaining configuration."""
    delta_min: float = Field(default=0.02, description="Minimum local gain per step")
    density_window_genes: int = Field(default=10, description="Window size for density calculation")
    density_min_anchors_per_gene: float = Field(default=0.3, description="Minimum anchor density")
    pos_band_genes: int = Field(default=1, description="Position drift tolerance in genes")
    max_gap_genes: int = Field(default=1, description="Maximum gap size in genes")
    
    @field_validator("delta_min")
    @classmethod
    def validate_delta_min(cls, v):
        if v <= 0:
            raise ValueError("delta_min must be positive")
        return v


class CassetteSegmenterConfig(BaseModel):
    """Cassette mode segmentation method configuration."""
    method: Literal["chain", "ransac"] = Field(default="chain", description="Segmentation method")


class AnalyzeConfig(BaseModel):
    """Analysis stage configuration."""
    clustering: 'ClusteringConfig' = Field(default_factory=lambda: ClusteringConfig())


class ClusteringConfig(BaseModel):
    """Clustering configuration for syntenic block analysis."""
    method: Literal["mutual_jaccard", "dbscan", "disabled"] = Field(default="mutual_jaccard", description="Clustering method")
    sink_label: int = Field(default=0, description="Label for sink cluster (non-robust blocks)")
    keep_singletons: bool = Field(default=False, description="Keep robust singleton blocks (don't send to sink)")
    
    # Robustness gate (per block)
    min_anchors: int = Field(default=4, description="Minimum alignment length (number of anchors)")
    min_span_genes: int = Field(default=8, description="Minimum span in genes on both query and target")
    v_mad_max_genes: int = Field(default=1, description="Maximum MAD of diagonal offset (genes)")
    
    # SRP + shingling
    srp_bits: int = Field(default=256, description="Total SRP projection bits")
    srp_bands: int = Field(default=32, description="Number of SRP bands")
    srp_band_bits: int = Field(default=8, description="Bits per SRP band")
    srp_seed: int = Field(default=1337, description="SRP random seed for determinism")
    shingle_k: int = Field(default=3, description="k-gram shingle size")
    
    # Graph construction
    jaccard_tau: float = Field(default=0.5, description="Minimum Jaccard similarity threshold")
    mutual_k: int = Field(default=3, description="Mutual top-k parameter")
    df_max: int = Field(default=200, description="Maximum document frequency for shingles")
    
    # Prefilters
    size_ratio_min: float = Field(default=0.5, description="Minimum size ratio for candidates")
    size_ratio_max: float = Field(default=2.0, description="Maximum size ratio for candidates")
    
    @model_validator(mode='after')
    def validate_srp_params(self):
        if self.srp_bands * self.srp_band_bits != self.srp_bits:
            raise ValueError(f"srp_bands * srp_band_bits ({self.srp_bands} * {self.srp_band_bits}) must equal srp_bits ({self.srp_bits})")
        return self
    
    @field_validator("jaccard_tau")
    @classmethod
    def validate_jaccard_tau(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("jaccard_tau must be between 0 and 1")
        return v
    
    @field_validator("size_ratio_min", "size_ratio_max")
    @classmethod
    def validate_size_ratios(cls, v):
        if v <= 0:
            raise ValueError("Size ratios must be positive")
        return v


class ELSAConfig(BaseModel):
    """Complete ELSA configuration."""
    data: DataConfig = Field(default_factory=DataConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    plm: PLMConfig = Field(default_factory=PLMConfig)
    shingles: ShingleConfig = Field(default_factory=ShingleConfig)
    discrete: DiscreteConfig = Field(default_factory=DiscreteConfig)
    continuous: ContinuousConfig = Field(default_factory=ContinuousConfig)
    chain: ChainConfig = Field(default_factory=ChainConfig)
    dtw: DTWConfig = Field(default_factory=DTWConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    phase2: Phase2Config = Field(default_factory=Phase2Config)
    cassette_mode: CassetteModeConfig = Field(default_factory=CassetteModeConfig)
    analyze: AnalyzeConfig = Field(default_factory=AnalyzeConfig)
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Cross-field validation."""
        discrete = self.discrete
        if discrete:
            bands, rows = discrete.bands_rows
            if bands * rows != discrete.minhash_hashes:
                raise ValueError(f"bands × rows ({bands} × {rows}) must equal minhash_hashes ({discrete.minhash_hashes})")
        return self


def load_config(config_path: Union[str, Path]) -> ELSAConfig:
    """Load and validate configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    return ELSAConfig(**data)


def create_default_config(output_path: Union[str, Path]) -> None:
    """Create a default configuration file."""
    config = ELSAConfig()
    output_path = Path(output_path)
    
    # Convert to dict and then to YAML
    config_dict = config.dict()
    
    with open(output_path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Created default configuration at: {output_path}")


if __name__ == "__main__":
    # Test configuration validation
    config = ELSAConfig()
    print("Default configuration validates successfully!")
    
    # Create example config file
    create_default_config("elsa.config.yaml")