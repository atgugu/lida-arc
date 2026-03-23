"""Tests for grid transformation primitives."""

from arc_self_play.primitives import operations as ops
from arc_self_play.primitives import PrimitiveLibrary


def test_rotate_90():
    grid = [[1, 2], [3, 4]]
    assert ops.rotate_90(grid) == [[3, 1], [4, 2]]


def test_rotate_180():
    grid = [[1, 2], [3, 4]]
    assert ops.rotate_180(grid) == [[4, 3], [2, 1]]


def test_rotate_270():
    grid = [[1, 2], [3, 4]]
    assert ops.rotate_270(grid) == [[2, 4], [1, 3]]


def test_reflect_horizontal():
    grid = [[1, 2], [3, 4]]
    assert ops.reflect_horizontal(grid) == [[3, 4], [1, 2]]


def test_reflect_vertical():
    grid = [[1, 2], [3, 4]]
    assert ops.reflect_vertical(grid) == [[2, 1], [4, 3]]


def test_reflect_diagonal():
    grid = [[1, 2], [3, 4]]
    assert ops.reflect_diagonal(grid) == [[1, 3], [2, 4]]


def test_auto_crop():
    grid = [
        [0, 0, 0],
        [0, 1, 2],
        [0, 3, 4],
    ]
    assert ops.auto_crop(grid) == [[1, 2], [3, 4]]


def test_tile():
    grid = [[1, 2]]
    assert ops.tile(grid, 2, 2) == [[1, 2, 1, 2], [1, 2, 1, 2]]


def test_recolor():
    grid = [[1, 2], [3, 0]]
    cmap = {0: 0, 1: 5, 2: 6, 3: 7}
    assert ops.recolor(grid, cmap) == [[5, 6], [7, 0]]


def test_library_has_all():
    lib = PrimitiveLibrary()
    assert len(lib) == 9
    for name in ["rotate_90", "rotate_180", "rotate_270",
                 "reflect_horizontal", "reflect_vertical", "reflect_diagonal",
                 "auto_crop", "tile", "recolor"]:
        assert name in lib
