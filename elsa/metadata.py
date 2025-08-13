#!/usr/bin/env python3
"""
ELSA run metadata tracking for Phase-2.

Captures configuration, versions, seeds, and stage execution for reproducibility.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import platform
import sys
from datetime import datetime

import numpy as np

from .params import ELSAConfig


@dataclass
class StageMetadata:
    """Metadata for a single pipeline stage."""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    input_files: List[str] = None
    output_files: List[str] = None
    parameters: Dict[str, Any] = None
    feature_flags: Dict[str, bool] = None
    memory_peak_mb: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    @property
    def status(self) -> str:
        """Get stage execution status."""
        if self.error_message:
            return "FAILED"
        elif self.end_time is not None:
            return "COMPLETED"
        else:
            return "RUNNING"


@dataclass
class RunMetadata:
    """Complete metadata for an ELSA analysis run."""
    run_id: str
    start_time: float
    elsa_version: str
    python_version: str
    platform: str
    hostname: str
    
    # Configuration snapshot
    config_snapshot: Dict[str, Any]
    phase2_enabled: bool
    active_feature_flags: Dict[str, bool]
    
    # Random seeds for reproducibility
    global_seed: int
    stage_seeds: Dict[str, int]
    
    # Stage execution tracking
    stages: List[StageMetadata]
    
    # Final results summary
    end_time: Optional[float] = None
    total_loci: Optional[int] = None
    total_blocks: Optional[int] = None
    total_clusters: Optional[int] = None
    success: bool = True
    
    def add_stage(self, stage: StageMetadata) -> None:
        """Add a completed stage to the run."""
        self.stages.append(stage)
    
    def get_stage(self, stage_name: str) -> Optional[StageMetadata]:
        """Get metadata for a specific stage."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage
        return None
    
    @property
    def total_duration_seconds(self) -> Optional[float]:
        """Calculate total run duration."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunMetadata':
        """Create from dictionary (JSON deserialization)."""
        # Convert stage data back to StageMetadata objects
        stages = [StageMetadata(**stage_data) for stage_data in data.get('stages', [])]
        data['stages'] = stages
        return cls(**data)


class MetadataTracker:
    """Tracks and persists run metadata throughout ELSA execution."""
    
    def __init__(self, config: ELSAConfig, work_dir: Path):
        self.config = config
        self.work_dir = Path(work_dir)
        self.metadata_file = self.work_dir / "run_metadata.json"
        self.run_metadata: Optional[RunMetadata] = None
    
    def start_run(self, run_id: Optional[str] = None) -> str:
        """Initialize a new run and return the run ID."""
        if run_id is None:
            run_id = f"elsa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract Phase-2 feature flags
        phase2_config = getattr(self.config, 'phase2', None)
        phase2_enabled = phase2_config.enable if phase2_config else False
        
        active_flags = {}
        if phase2_enabled and phase2_config:
            active_flags = {
                'weighted_sketch': phase2_config.weighted_sketch,
                'multiscale': phase2_config.multiscale,
                'flip_dp': phase2_config.flip_dp,
                'calibration': phase2_config.calibration,
                'hnsw': phase2_config.hnsw,
            }
        
        # Generate deterministic stage seeds from global seed
        global_seed = getattr(self.config.system, 'rng_seed', 17)
        rng = np.random.RandomState(global_seed)
        stage_seeds = {
            'ingest': int(rng.randint(0, 2**31)),
            'embed': int(rng.randint(0, 2**31)),
            'project': int(rng.randint(0, 2**31)),
            'shingle': int(rng.randint(0, 2**31)),
            'index': int(rng.randint(0, 2**31)),
            'search': int(rng.randint(0, 2**31)),
            'chain': int(rng.randint(0, 2**31)),
            'calibrate': int(rng.randint(0, 2**31)),
        }
        
        self.run_metadata = RunMetadata(
            run_id=run_id,
            start_time=time.time(),
            elsa_version="0.2.0-phase2",  # TODO: get from package metadata
            python_version=sys.version,
            platform=platform.platform(),
            hostname=platform.node(),
            config_snapshot=self.config.model_dump(),
            phase2_enabled=phase2_enabled,
            active_feature_flags=active_flags,
            global_seed=global_seed,
            stage_seeds=stage_seeds,
            stages=[]
        )
        
        self._save_metadata()
        return run_id
    
    def start_stage(self, stage_name: str, input_files: List[str] = None,
                   parameters: Dict[str, Any] = None) -> StageMetadata:
        """Start tracking a new stage."""
        if not self.run_metadata:
            raise RuntimeError("Must call start_run() before start_stage()")
        
        # Get stage-specific feature flags
        feature_flags = {}
        if self.run_metadata.phase2_enabled:
            feature_flags = self.run_metadata.active_feature_flags.copy()
        
        stage = StageMetadata(
            stage_name=stage_name,
            start_time=time.time(),
            input_files=input_files or [],
            parameters=parameters or {},
            feature_flags=feature_flags
        )
        
        return stage
    
    def end_stage(self, stage: StageMetadata, output_files: List[str] = None,
                 error_message: Optional[str] = None) -> None:
        """Complete a stage and add it to the run."""
        stage.end_time = time.time()
        stage.output_files = output_files or []
        stage.error_message = error_message
        
        if not self.run_metadata:
            raise RuntimeError("No active run to add stage to")
        
        self.run_metadata.add_stage(stage)
        self._save_metadata()
    
    def end_run(self, total_loci: int = 0, total_blocks: int = 0, 
               total_clusters: int = 0, success: bool = True) -> None:
        """Complete the run with final statistics."""
        if not self.run_metadata:
            raise RuntimeError("No active run to end")
        
        self.run_metadata.end_time = time.time()
        self.run_metadata.total_loci = total_loci
        self.run_metadata.total_blocks = total_blocks
        self.run_metadata.total_clusters = total_clusters
        self.run_metadata.success = success
        
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Persist metadata to JSON file."""
        if not self.run_metadata:
            return
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.run_metadata.to_dict(), f, indent=2, default=str)
    
    def load_metadata(self) -> Optional[RunMetadata]:
        """Load existing metadata from file."""
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)
        
        self.run_metadata = RunMetadata.from_dict(data)
        return self.run_metadata
    
    def get_stage_seed(self, stage_name: str) -> int:
        """Get the deterministic seed for a specific stage."""
        if not self.run_metadata:
            raise RuntimeError("No active run")
        
        return self.run_metadata.stage_seeds.get(stage_name, 
                                                self.run_metadata.global_seed)


def create_metadata_tracker(config: ELSAConfig, work_dir: Path) -> MetadataTracker:
    """Factory function to create a metadata tracker."""
    return MetadataTracker(config, work_dir)