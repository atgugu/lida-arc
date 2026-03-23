"""Pure-function grid transformation operations.

Each function takes a grid (list of lists of ints 0-9) and returns a new grid.
No side effects; all operations are deterministic.
"""

from typing import Dict, List

Grid = List[List[int]]


# ---------------------------------------------------------------------------
# Geometric operations
# ---------------------------------------------------------------------------

def rotate_90(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    if not grid or not grid[0]:
        return grid
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid) -> Grid:
    """Rotate grid 270 degrees clockwise (= 90 degrees counter-clockwise)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    return [[grid[r][w - 1 - c] for r in range(h)] for c in range(w)]


def reflect_horizontal(grid: Grid) -> Grid:
    """Reflect grid across the horizontal axis (flip top-bottom)."""
    if not grid:
        return grid
    return grid[::-1]


def reflect_vertical(grid: Grid) -> Grid:
    """Reflect grid across the vertical axis (flip left-right)."""
    if not grid:
        return grid
    return [row[::-1] for row in grid]


def reflect_diagonal(grid: Grid) -> Grid:
    """Reflect grid across the main diagonal (transpose)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(h)] for c in range(w)]


# ---------------------------------------------------------------------------
# Spatial operations
# ---------------------------------------------------------------------------

def auto_crop(grid: Grid, background: int = 0) -> Grid:
    """Crop grid to the minimal bounding box of non-background pixels."""
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    min_r, max_r = h, -1
    min_c, max_c = w, -1

    for r in range(h):
        for c in range(w):
            if grid[r][c] != background:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    if max_r < 0:
        return grid

    return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]


def tile(pattern: Grid, repeat_v: int, repeat_h: int) -> Grid:
    """Tile a pattern by repeating it vertically and horizontally."""
    if not pattern or not pattern[0]:
        return pattern
    tiled = []
    for _ in range(repeat_v):
        for row in pattern:
            tiled.append(row * repeat_h)
    return tiled


# ---------------------------------------------------------------------------
# Color operations
# ---------------------------------------------------------------------------

def recolor(grid: Grid, color_map: Dict[int, int]) -> Grid:
    """Apply a color mapping to every cell in the grid."""
    return [[color_map.get(v, v) for v in row] for row in grid]
