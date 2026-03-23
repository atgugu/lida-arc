"""Neural guidance networks for hybrid ARC solving.

Train lightweight networks on self-play puzzles to guide symbolic search:
- **TaskClassifier** categorises puzzles (geometric, color, tiling, ...).
- **PrimitivePredictor** scores which primitives are relevant.
- **CompositionPredictor** suggests the next primitive in a sequence.
"""

from .encoding import encode_grid, encode_grid_pair, decode_grid
from .labels import (
    TASK_CATEGORIES,
    extract_task_label,
    extract_primitive_labels,
    extract_composition_sequences,
)
from .dataset import SelfPlayDataset, CompositionDataset, create_dataloaders

__all__ = [
    "encode_grid",
    "encode_grid_pair",
    "decode_grid",
    "TASK_CATEGORIES",
    "extract_task_label",
    "extract_primitive_labels",
    "extract_composition_sequences",
    "SelfPlayDataset",
    "CompositionDataset",
    "create_dataloaders",
]
