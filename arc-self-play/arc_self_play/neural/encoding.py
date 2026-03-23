"""Encode ARC grids as tensors for neural networks.

Each grid cell holds a colour 0-9. We one-hot encode across 10 channels,
producing a ``(10, H, W)`` float tensor, zero-padded to a fixed spatial size.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

Grid = List[List[int]]

MAX_SIZE = 30  # ARC grids are at most 30x30


def encode_grid(grid: Grid, size: int = MAX_SIZE) -> torch.Tensor:
    """One-hot encode a grid.

    Args:
        grid: 2-D list of ints 0-9.
        size: Spatial dimension to pad/crop to.

    Returns:
        Float tensor of shape ``(10, size, size)``.
    """
    if not grid or not grid[0]:
        return torch.zeros(10, size, size)

    h, w = min(len(grid), size), min(len(grid[0]), size)
    buf = np.zeros((10, size, size), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if 0 <= v <= 9:
                buf[v, r, c] = 1.0

    return torch.from_numpy(buf)


def encode_grid_pair(inp: Grid, out: Grid, size: int = MAX_SIZE) -> torch.Tensor:
    """Encode an input-output pair.

    Returns:
        Float tensor of shape ``(2, 10, size, size)``.
    """
    return torch.stack([encode_grid(inp, size), encode_grid(out, size)])


def decode_grid(tensor: torch.Tensor) -> Grid:
    """Argmax-decode a ``(10, H, W)`` tensor back to a grid."""
    colors = tensor.argmax(dim=0).numpy()
    return colors.tolist()
