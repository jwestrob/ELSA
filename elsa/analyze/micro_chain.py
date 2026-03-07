"""
Backward-compatibility shim for micro_chain module.

All implementations have moved to:
  - elsa.analyze.pipeline (run_chain_pipeline, ChainSummary, ChainConfig)
  - elsa.cluster (cluster_blocks_by_overlap)
"""

from elsa.analyze.pipeline import run_chain_pipeline as run_micro_chain_pipeline
from elsa.analyze.pipeline import ChainConfig as MicroChainConfig
from elsa.analyze.pipeline import ChainSummary as MicroChainSummary

__all__ = [
    'run_micro_chain_pipeline',
    'MicroChainConfig',
    'MicroChainSummary',
]
