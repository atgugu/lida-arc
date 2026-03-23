"""Self-play puzzle generation by composing grid primitives."""

from .core import PuzzleGenerator, GeneratedPuzzle
from .grids import GridGenerator
from .sampler import TransformationSampler
from .validator import PuzzleValidator
from .stats import GenerationStats

__all__ = [
    "PuzzleGenerator",
    "GeneratedPuzzle",
    "GridGenerator",
    "TransformationSampler",
    "PuzzleValidator",
    "GenerationStats",
]
