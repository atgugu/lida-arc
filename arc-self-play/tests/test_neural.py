"""Tests for neural encoding and label extraction."""

import json
import tempfile
from pathlib import Path

import torch

from arc_self_play.neural.encoding import encode_grid, encode_grid_pair, decode_grid
from arc_self_play.neural.labels import (
    extract_task_label,
    extract_primitive_labels,
    extract_composition_sequences,
)
from arc_self_play.generator import PuzzleGenerator


def test_encode_grid_shape():
    grid = [[1, 2, 3], [4, 5, 6]]
    t = encode_grid(grid)
    assert t.shape == (10, 30, 30)
    assert t.dtype == torch.float32


def test_encode_grid_onehot():
    grid = [[0, 1], [2, 3]]
    t = encode_grid(grid, size=4)
    assert t[0, 0, 0] == 1.0  # colour 0 at (0,0)
    assert t[1, 0, 1] == 1.0  # colour 1 at (0,1)
    assert t[2, 1, 0] == 1.0  # colour 2 at (1,0)
    assert t[3, 1, 1] == 1.0  # colour 3 at (1,1)


def test_encode_grid_pair_shape():
    t = encode_grid_pair([[1]], [[2]])
    assert t.shape == (2, 10, 30, 30)


def test_task_label_geometric():
    assert extract_task_label([("rotate_90", {})]) == "geometric"
    assert extract_task_label([("reflect_vertical", {})]) == "geometric"


def test_task_label_composition():
    ops = [("rotate_90", {}), ("reflect_vertical", {}), ("auto_crop", {})]
    assert extract_task_label(ops) == "composition"


def test_task_label_tiling():
    assert extract_task_label([("tile", {})]) == "tiling"


def test_primitive_labels():
    prims = ["rotate_90", "rotate_180", "reflect_horizontal", "auto_crop", "recolor"]
    transform = [("rotate_90", {}), ("auto_crop", {})]
    labels = extract_primitive_labels(transform, prims)
    assert labels.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_composition_sequences():
    prim_to_idx = {"rotate_90": 0, "reflect_horizontal": 1, "auto_crop": 2}
    puzzle = {
        "train": [([[1, 2]], [[3, 4]])],
        "transformation": [("rotate_90", {}), ("reflect_horizontal", {})],
    }
    seqs = extract_composition_sequences(puzzle, prim_to_idx)
    assert len(seqs) == 2
    assert seqs[0]["primitive_sequence"] == []
    assert seqs[0]["next_primitive"] == 0
    assert seqs[1]["primitive_sequence"] == [0]
    assert seqs[1]["next_primitive"] == 1


def test_dataset_from_generated():
    """Round-trip: generate puzzles -> save -> load as dataset."""
    from arc_self_play.neural.dataset import SelfPlayDataset
    from arc_self_play.primitives import PrimitiveLibrary

    gen = PuzzleGenerator(seed=99)
    puzzles = gen.generate_batch(10)
    assert len(puzzles) >= 5

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([p.to_dict() for p in puzzles], f)
        path = f.name

    lib = PrimitiveLibrary()
    ds = SelfPlayDataset(path, lib.get_all_names())
    assert len(ds) > 0

    item = ds[0]
    assert item["grid_pair"].shape == (2, 10, 30, 30)
    assert item["task_index"].dtype == torch.long
    assert item["primitive_labels"].shape == (len(lib),)
