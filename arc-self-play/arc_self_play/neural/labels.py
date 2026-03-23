"""Extract supervised labels from self-play puzzles.

Because we generate puzzles *and* know the ground-truth transformation, every
puzzle provides perfect supervision for three tasks:

1. **Task classification** -- what family does the transformation belong to?
2. **Primitive relevance** -- which of the 9 primitives appear in the
   transformation?
3. **Composition sequence** -- given a partial application, what primitive
   comes next?
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

TASK_CATEGORIES = [
    "geometric",       # rotate, reflect
    "color_mapping",   # recolor
    "tiling",          # tile
    "cropping",        # auto_crop
    "composition",     # multi-operation (3+)
]

TASK_TO_IDX = {c: i for i, c in enumerate(TASK_CATEGORIES)}


def extract_task_label(transformation: List[Tuple[str, Dict]]) -> str:
    """Classify a transformation into one of ``TASK_CATEGORIES``."""
    if not transformation:
        return "composition"
    ops = [name for name, _ in transformation]
    if len(ops) >= 3:
        return "composition"
    if "tile" in ops:
        return "tiling"
    if "recolor" in ops:
        return "color_mapping"
    if "auto_crop" in ops:
        return "cropping"
    geo = {"rotate_90", "rotate_180", "rotate_270",
           "reflect_horizontal", "reflect_vertical", "reflect_diagonal"}
    if any(o in geo for o in ops):
        return "geometric"
    return "composition"


def extract_task_index(transformation: List[Tuple[str, Dict]]) -> int:
    return TASK_TO_IDX[extract_task_label(transformation)]


def extract_primitive_labels(
    transformation: List[Tuple[str, Dict]],
    all_primitives: List[str],
) -> np.ndarray:
    """Binary vector marking which primitives are used."""
    used = {name for name, _ in transformation}
    return np.array([1.0 if p in used else 0.0 for p in all_primitives], dtype=np.float32)


def extract_composition_sequences(
    puzzle: Dict[str, Any],
    prim_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Break a transformation into step-by-step training examples.

    For ``[A, B, C]`` this yields three examples per demo pair::

        (grid_0, [])      -> A
        (grid_0, [A])     -> B
        (grid_0, [A, B])  -> C
    """
    transform = puzzle["transformation"]
    examples: List[Dict[str, Any]] = []

    for inp, out in puzzle.get("train", []):
        seq: List[int] = []
        for op_name, _params in transform:
            if op_name not in prim_to_idx:
                continue
            examples.append({
                "input_grid": inp,
                "output_grid": out,
                "primitive_sequence": seq.copy(),
                "next_primitive": prim_to_idx[op_name],
                "sequence_length": len(seq),
            })
            seq.append(prim_to_idx[op_name])

    return examples
