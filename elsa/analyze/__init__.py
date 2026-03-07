"""
ELSA analyze module for syntenic block analysis via gene-level anchor chaining.
"""

from .pipeline import run_chain_pipeline, ChainSummary, ChainConfig
# Backward compat aliases
from .micro_chain import run_micro_chain_pipeline, MicroChainConfig, MicroChainSummary

__all__ = [
    'run_chain_pipeline',
    'ChainSummary',
    'ChainConfig',
    'run_micro_chain_pipeline',
    'MicroChainConfig',
    'MicroChainSummary',
]
