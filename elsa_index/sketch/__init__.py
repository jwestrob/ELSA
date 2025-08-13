"""
ELSA Phase-2 weighted sketching module.

Implements weighted MinHash with IDF weighting and MGE masking.
"""

from .weighted_minhash import WeightedMinHashSketch, DOPH
from .idf_stats import IDFStatistics, compute_idf_weights
from .mge_mask import MGEMask, load_mge_mask

__all__ = [
    'WeightedMinHashSketch',
    'DOPH', 
    'IDFStatistics',
    'compute_idf_weights',
    'MGEMask',
    'load_mge_mask'
]