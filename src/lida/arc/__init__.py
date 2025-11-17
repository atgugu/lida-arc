"""
ARC-AGI components for LIDA cognitive architecture.

This package implements hybrid bootstrapping for learning transformations
from demonstrations without a pre-defined DSL.
"""

from .environment import ARCTask, GridPair, ARCEnvironment
from .perception import GridObject, ObjectExtractor, GridAnalyzer

__all__ = [
    'ARCTask',
    'GridPair',
    'ARCEnvironment',
    'GridObject',
    'ObjectExtractor',
    'GridAnalyzer',
]
