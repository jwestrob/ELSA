"""
Configuration system for ELSA with strict validation.
"""

from pathlib import Path
from typing import List, Optional, Union, Literal
import yaml
from pydantic import BaseModel, Field, validator, model_validator


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


class PLMConfig(BaseModel):
    """Protein language model configuration."""
    model: Literal["esm2_t33", "esm2_t12", "prot_t5"] = Field(default="esm2_t12")
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(default="auto")
    batch_amino_acids: int = Field(default=16000, description="Approximate per-batch residue budget")
    fp16: bool = Field(default=True, description="Use half precision")
    project_to_D: int = Field(default=256, description="PCA target dimension")
    l2_normalize: bool = Field(default=True, description="L2 normalize embeddings")
    
    @validator("batch_amino_acids")
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
    
    @validator("bands_rows")
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
    # Phase-2 chaining parameters
    alpha: float = Field(default=0.1, description="Position deviation penalty")
    beta: float = Field(default=1.0, description="Gap penalty")
    gamma: float = Field(default=2.0, description="Strand flip penalty")


class DTWConfig(BaseModel):
    """Dynamic time warping refinement."""
    enable: bool = Field(default=True, description="Enable DTW refinement")
    band: int = Field(default=10, description="DTW band width")
    # Phase-2 DTW refinement
    refine_enable: bool = Field(default=False, description="Bounded DTW on final chain only")
    refine_band: int = Field(default=2, description="DTW band width in genes")


class ScoreConfig(BaseModel):
    """Scoring parameters."""
    alpha: float = Field(default=1.0, description="Anchor strength weight")
    beta: float = Field(default=0.1, description="LIS length weight")
    gamma: float = Field(default=0.2, description="Gap penalty weight")
    delta: float = Field(default=0.1, description="Offset variance penalty weight")
    fdr_target: float = Field(default=0.01, description="False discovery rate target")
    
    @validator("fdr_target")
    def validate_fdr(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError("fdr_target must be between 0 and 1")
        return v


class SystemConfig(BaseModel):
    """System resource configuration."""
    jobs: Union[int, Literal["auto"]] = Field(default="auto", description="Number of parallel jobs")
    mmap: bool = Field(default=True, description="Use memory mapping")
    rng_seed: int = Field(default=17, description="Global random seed")


# Phase-2 Configuration Classes

class Phase2Config(BaseModel):
    """Phase-2 feature flags."""
    enable: bool = Field(default=False, description="Master switch for all Phase-2 features")
    weighted_sketch: bool = Field(default=False, description="Use weighted MinHash with IDF/MGE masking")
    multiscale: bool = Field(default=False, description="Enable macro→micro windowing")
    flip_dp: bool = Field(default=False, description="Use flip-aware affine-gap chaining")
    calibration: bool = Field(default=False, description="Enable null models and FDR control")
    hnsw: bool = Field(default=False, description="Use HNSW dense retrieval")


class SketchConfig(BaseModel):
    """Weighted sketching configuration."""
    type: Literal["minhash", "weighted_minhash"] = Field(default="minhash")
    bits: Literal[64, 2, 1] = Field(default=64, description="Compression level")
    size: int = Field(default=96, description="Sketch size")
    idf: dict = Field(default_factory=lambda: {"max": 10.0}, description="IDF parameters")


class MGEMaskConfig(BaseModel):
    """MGE masking configuration."""
    path: Optional[str] = Field(default=None, description="YAML list of PFAM accessions to mask")


class WindowConfig(BaseModel):
    """Multi-scale windowing configuration."""
    micro: dict = Field(default_factory=lambda: {"size": 5, "stride": 1})
    macro: dict = Field(default_factory=lambda: {"size": 12, "stride": 3})
    adaptive: dict = Field(default_factory=lambda: {"enable": False})


class HNSWConfig(BaseModel):
    """HNSW configuration."""
    M: int = Field(default=16, description="Max connections per element")
    efConstruction: int = Field(default=200, description="Construction parameter")
    efSearch: int = Field(default=50, description="Search parameter")


class RetrievalConfig(BaseModel):
    """Retrieval method configuration."""
    dense: Literal["srp", "hnsw"] = Field(default="srp")


class CalibConfig(BaseModel):
    """Calibration configuration."""
    null: dict = Field(default_factory=lambda: {"iters": 100})
    target_fdr: float = Field(default=0.05, description="FDR threshold")
    
    @validator("target_fdr")
    def validate_target_fdr(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError("target_fdr must be between 0 and 1")
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
    
    # Phase-2 configurations (optional, backwards compatible)
    phase2: Optional[Phase2Config] = Field(default_factory=Phase2Config)
    sketch: Optional[SketchConfig] = Field(default_factory=SketchConfig)
    mge_mask: Optional[MGEMaskConfig] = Field(default_factory=MGEMaskConfig)
    window: Optional[WindowConfig] = Field(default_factory=WindowConfig)
    hnsw: Optional[HNSWConfig] = Field(default_factory=HNSWConfig)
    retrieval: Optional[RetrievalConfig] = Field(default_factory=RetrievalConfig)
    calib: Optional[CalibConfig] = Field(default_factory=CalibConfig)
    
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