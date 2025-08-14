"""
ELSA windowing module for multi-scale window generation.
"""

from .multiscale import (
    MultiScaleWindowGenerator,
    MultiScaleSearchEngine,
    MultiScaleWindow,
    WindowMapping
)

__all__ = [
    'MultiScaleWindowGenerator',
    'MultiScaleSearchEngine', 
    'MultiScaleWindow',
    'WindowMapping'
]