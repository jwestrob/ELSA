"""
Data manifest and artifact registry for ELSA.

Tracks configuration hashes, model parameters, and dataset statistics for reproducibility.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from .params import ELSAConfig


@dataclass
class ArtifactInfo:
    """Information about a generated artifact."""
    name: str
    path: str
    created: str
    size_bytes: int
    checksum: str
    stage: str
    metadata: Dict[str, Any] = None


@dataclass 
class DatasetStats:
    """Statistics about processed dataset."""
    n_samples: int
    n_contigs: int
    n_genes: int
    total_aa: int
    mean_gene_length: float
    contig_n50: int


@dataclass
class ModelInfo:
    """Information about trained models."""
    model_type: str
    parameters: Dict[str, Any]
    training_samples: int
    checksum: str


class ELSAManifest:
    """Registry for ELSA artifacts and metadata."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.manifest_path = self.work_dir / "MANIFEST.json"
        self.data = self._load_or_create()
    
    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        else:
            return {
                "version": "0.1.0",
                "created": datetime.now().isoformat(),
                "config_hash": None,
                "artifacts": {},
                "models": {},
                "dataset_stats": None,
                "checksums": {}
            }
    
    def save(self):
        """Save manifest to disk."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def set_config(self, config: ELSAConfig):
        """Register configuration and compute hash."""
        config_dict = config.dict()
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        serializable_config = convert_paths(config_dict)
        config_str = json.dumps(serializable_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        self.data["config_hash"] = config_hash
        self.data["config"] = serializable_config
        self.data["updated"] = datetime.now().isoformat()
        self.save()
    
    def register_artifact(self, name: str, path: Path, stage: str, metadata: Optional[Dict] = None):
        """Register an artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        
        checksum = self._compute_checksum(path)
        artifact = ArtifactInfo(
            name=name,
            path=str(path),
            created=datetime.now().isoformat(),
            size_bytes=path.stat().st_size,
            checksum=checksum,
            stage=stage,
            metadata=metadata or {}
        )
        
        self.data["artifacts"][name] = asdict(artifact)
        self.data["checksums"][name] = checksum
        self.save()
    
    def register_model(self, model_type: str, parameters: Dict[str, Any], 
                      training_samples: int, model_path: Path):
        """Register a trained model."""
        checksum = self._compute_checksum(model_path)
        model_info = ModelInfo(
            model_type=model_type,
            parameters=parameters,
            training_samples=training_samples,
            checksum=checksum
        )
        
        self.data["models"][model_type] = asdict(model_info)
        self.save()
    
    def set_dataset_stats(self, stats: DatasetStats):
        """Set dataset statistics."""
        self.data["dataset_stats"] = asdict(stats)
        self.save()
    
    def get_artifact_path(self, name: str) -> Optional[Path]:
        """Get path to named artifact."""
        if name in self.data["artifacts"]:
            return Path(self.data["artifacts"][name]["path"])
        return None
    
    def verify_checksums(self) -> Dict[str, bool]:
        """Verify all artifact checksums."""
        results = {}
        for name, artifact in self.data["artifacts"].items():
            path = Path(artifact["path"])
            if path.exists():
                current_checksum = self._compute_checksum(path)
                results[name] = current_checksum == artifact["checksum"]
            else:
                results[name] = False
        return results
    
    def get_stage_artifacts(self, stage: str) -> List[str]:
        """Get all artifacts from a specific stage."""
        return [name for name, artifact in self.data["artifacts"].items() 
                if artifact["stage"] == stage]
    
    def is_stage_complete(self, stage: str, required_artifacts: List[str]) -> bool:
        """Check if a stage is complete based on required artifacts."""
        stage_artifacts = set(self.get_stage_artifacts(stage))
        return all(artifact in stage_artifacts for artifact in required_artifacts)
    
    def has_artifact(self, name: str) -> bool:
        """Check if an artifact exists."""
        return name in self.data.get("artifacts", {})
    
    def artifact_path(self, name: str) -> Path:
        """Get the path to an artifact."""
        if name in self.data.get("artifacts", {}):
            return Path(self.data["artifacts"][name]["path"])
        else:
            # For relative paths, assume they're in work_dir
            return self.work_dir / name
    
    def add_artifact(self, name: str, metadata: Dict):
        """Add artifact metadata to manifest."""
        if "artifacts" not in self.data:
            self.data["artifacts"] = {}
        
        self.data["artifacts"][name] = {
            "created": datetime.now().isoformat(),
            **metadata
        }
        self.save()
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    
    def summary(self) -> str:
        """Generate a summary string of the manifest."""
        lines = [
            f"ELSA Manifest (Config: {self.data.get('config_hash', 'None')})",
            f"Created: {self.data.get('created', 'Unknown')}",
            f"Updated: {self.data.get('updated', 'Never')}",
            ""
        ]
        
        if self.data["dataset_stats"]:
            stats = self.data["dataset_stats"]
            lines.extend([
                "Dataset Statistics:",
                f"  Samples: {stats['n_samples']}",
                f"  Contigs: {stats['n_contigs']}",
                f"  Genes: {stats['n_genes']:,}",
                f"  Total AA: {stats['total_aa']:,}",
                ""
            ])
        
        if self.data["artifacts"]:
            lines.append("Artifacts:")
            for name, artifact in self.data["artifacts"].items():
                size_mb = artifact["size_bytes"] / (1024 * 1024)
                lines.append(f"  {name}: {size_mb:.1f}MB ({artifact['stage']})")
            lines.append("")
        
        if self.data["models"]:
            lines.append("Models:")
            for model_type, model_info in self.data["models"].items():
                lines.append(f"  {model_type}: {model_info['training_samples']:,} samples")
        
        return "\n".join(lines)


def validate_fasta_files(sample_data: List[tuple]) -> None:
    """Validate FASTA files exist and sample IDs are unique."""
    sample_ids = [sample_id for sample_id, _ in sample_data]
    
    # Check for duplicate sample IDs
    if len(set(sample_ids)) != len(sample_ids):
        dupes = [sid for sid in set(sample_ids) if sample_ids.count(sid) > 1]
        raise ValueError(f"Duplicate sample IDs: {dupes}")
    
    # Validate file paths exist
    missing_files = []
    for sample_id, fasta_path in sample_data:
        if not Path(fasta_path).exists():
            missing_files.append(str(fasta_path))
    
    if missing_files:
        raise FileNotFoundError(f"Missing FASTA files: {missing_files}")


if __name__ == "__main__":
    # Test manifest functionality
    from .params import ELSAConfig
    
    config = ELSAConfig()
    manifest = ELSAManifest(Path("./test_elsa_index"))
    manifest.set_config(config)
    
    print(manifest.summary())