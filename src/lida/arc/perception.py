"""
Grid perception: object extraction, feature analysis, and pattern detection.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


@dataclass
class GridObject:
    """Represents a connected component (object) extracted from a grid."""

    object_id: str
    pixels: FrozenSet[Tuple[int, int]]  # Immutable set of (row, col) coordinates
    color: int  # 0-9
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Center of mass of the object."""
        if not self.pixels:
            return (0.0, 0.0)
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (sum(rows) / len(rows), sum(cols) / len(cols))

    @property
    def size(self) -> int:
        """Number of pixels in the object."""
        return len(self.pixels)

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bounding_box[3] - self.bounding_box[1] + 1

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bounding_box[2] - self.bounding_box[0] + 1

    @property
    def area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height

    @property
    def density(self) -> float:
        """Ratio of pixels to bounding box area."""
        area = self.area
        return self.size / area if area > 0 else 0.0

    def get_shape_signature(self) -> str:
        """Canonical representation of shape (normalized to origin)."""
        if not self.pixels:
            return "empty"

        min_r, min_c, _, _ = self.bounding_box
        # Normalize to origin
        normalized = frozenset((r - min_r, c - min_c) for r, c in self.pixels)
        return str(hash(normalized))

    def has_symmetry_vertical(self) -> bool:
        """Check vertical axis symmetry (reflection across vertical center line)."""
        if not self.pixels:
            return False

        min_r, min_c, max_r, max_c = self.bounding_box
        center_col = (min_c + max_c) / 2.0

        for r, c in self.pixels:
            # Mirror position
            mirrored_c = 2 * center_col - c
            mirrored_pos = (r, int(round(mirrored_c)))

            if mirrored_pos not in self.pixels:
                return False

        return True

    def has_symmetry_horizontal(self) -> bool:
        """Check horizontal axis symmetry (reflection across horizontal center line)."""
        if not self.pixels:
            return False

        min_r, min_c, max_r, max_c = self.bounding_box
        center_row = (min_r + max_r) / 2.0

        for r, c in self.pixels:
            # Mirror position
            mirrored_r = 2 * center_row - r
            mirrored_pos = (int(round(mirrored_r)), c)

            if mirrored_pos not in self.pixels:
                return False

        return True

    def has_symmetry_diagonal(self) -> bool:
        """Check diagonal symmetry (reflection across main diagonal)."""
        if not self.pixels:
            return False

        min_r, min_c, _, _ = self.bounding_box

        for r, c in self.pixels:
            # Transpose relative to bounding box
            rel_r, rel_c = r - min_r, c - min_c
            mirrored_pos = (min_r + rel_c, min_c + rel_r)

            if mirrored_pos not in self.pixels:
                return False

        return True

    def to_features(self) -> Dict[str, float]:
        """Convert object properties to feature dictionary."""
        return {
            'size': float(self.size),
            'width': float(self.width),
            'height': float(self.height),
            'density': self.density,
            'centroid_row': self.centroid[0],
            'centroid_col': self.centroid[1],
            'has_symmetry_v': 1.0 if self.has_symmetry_vertical() else 0.0,
            'has_symmetry_h': 1.0 if self.has_symmetry_horizontal() else 0.0,
            'has_symmetry_d': 1.0 if self.has_symmetry_diagonal() else 0.0,
            'color': float(self.color),
            'aspect_ratio': self.width / self.height if self.height > 0 else 0.0,
        }


class ObjectExtractor:
    """Extract connected components (objects) from grids."""

    def __init__(self, background_color: int = 0, connectivity: int = 4):
        """
        Args:
            background_color: Color to treat as background (not extracted as objects)
            connectivity: 4 or 8 (4-connected or 8-connected components)
        """
        self.background_color = background_color
        self.connectivity = connectivity

    def extract_objects(self, grid: List[List[int]]) -> List[GridObject]:
        """Extract all objects from grid using connected component analysis.

        Args:
            grid: 2D grid of integers

        Returns:
            List of GridObject instances
        """
        if not grid or not grid[0]:
            return []

        height = len(grid)
        width = len(grid[0])
        visited = set()
        objects = []

        for r in range(height):
            for c in range(width):
                if (r, c) not in visited and grid[r][c] != self.background_color:
                    # Start flood fill
                    obj_pixels = self._flood_fill(grid, r, c, visited, height, width)
                    if obj_pixels:
                        obj = self._create_object(obj_pixels, grid[r][c])
                        objects.append(obj)

        return objects

    def _flood_fill(
        self,
        grid: List[List[int]],
        start_r: int,
        start_c: int,
        visited: Set[Tuple[int, int]],
        height: int,
        width: int
    ) -> Set[Tuple[int, int]]:
        """BFS flood fill to find connected component."""
        target_color = grid[start_r][start_c]
        queue = [(start_r, start_c)]
        pixels = set()

        while queue:
            r, c = queue.pop(0)

            if (r, c) in visited:
                continue
            if r < 0 or r >= height or c < 0 or c >= width:
                continue
            if grid[r][c] != target_color:
                continue

            visited.add((r, c))
            pixels.add((r, c))

            # Add neighbors based on connectivity
            if self.connectivity == 4:
                neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
            else:  # 8-connected
                neighbors = [
                    (r+1, c), (r-1, c), (r, c+1), (r, c-1),
                    (r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)
                ]

            queue.extend(neighbors)

        return pixels

    def _create_object(self, pixels: Set[Tuple[int, int]], color: int) -> GridObject:
        """Create GridObject from pixel set."""
        if not pixels:
            raise ValueError("Cannot create object from empty pixel set")

        min_r = min(p[0] for p in pixels)
        max_r = max(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        max_c = max(p[1] for p in pixels)

        # Create deterministic ID from frozenset hash
        frozen_pixels = frozenset(pixels)
        obj_id = f"obj_{hash(frozen_pixels) & 0x7FFFFFFF:08x}"  # Positive hex

        return GridObject(
            object_id=obj_id,
            pixels=frozen_pixels,
            color=color,
            bounding_box=(min_r, min_c, max_r, max_c)
        )


class GridAnalyzer:
    """Analyze grid-level properties and patterns."""

    def __init__(self):
        self.object_extractor = ObjectExtractor()

    def compute_color_histogram(self, grid: List[List[int]]) -> Dict[int, int]:
        """Count pixels of each color."""
        hist = defaultdict(int)
        for row in grid:
            for val in row:
                hist[val] += 1
        return dict(hist)

    def detect_grid_symmetry(self, grid: List[List[int]]) -> Dict[str, bool]:
        """Check if entire grid has symmetry."""
        return {
            'vertical': self._check_vertical_symmetry(grid),
            'horizontal': self._check_horizontal_symmetry(grid),
            'rotational_180': self._check_rotational_180(grid),
            'rotational_90': self._check_rotational_90(grid),
        }

    def _check_vertical_symmetry(self, grid: List[List[int]]) -> bool:
        """Check if grid is symmetric across vertical center line."""
        height = len(grid)
        width = len(grid[0]) if grid else 0

        for r in range(height):
            for c in range(width // 2):
                if grid[r][c] != grid[r][width - 1 - c]:
                    return False
        return True

    def _check_horizontal_symmetry(self, grid: List[List[int]]) -> bool:
        """Check if grid is symmetric across horizontal center line."""
        height = len(grid)
        width = len(grid[0]) if grid else 0

        for r in range(height // 2):
            for c in range(width):
                if grid[r][c] != grid[height - 1 - r][c]:
                    return False
        return True

    def _check_rotational_180(self, grid: List[List[int]]) -> bool:
        """Check if grid is symmetric under 180-degree rotation."""
        height = len(grid)
        width = len(grid[0]) if grid else 0

        for r in range(height):
            for c in range(width):
                if grid[r][c] != grid[height - 1 - r][width - 1 - c]:
                    return False
        return True

    def _check_rotational_90(self, grid: List[List[int]]) -> bool:
        """Check if grid is symmetric under 90-degree rotation (must be square)."""
        height = len(grid)
        width = len(grid[0]) if grid else 0

        if height != width:
            return False

        n = height
        for r in range(n):
            for c in range(n):
                # Check if position (r,c) maps correctly under 90° rotation
                if grid[r][c] != grid[c][n - 1 - r]:
                    return False
        return True

    def compute_grid_features(self, grid: List[List[int]]) -> Dict[str, float]:
        """Extract high-level features from grid."""
        if not grid or not grid[0]:
            return {}

        objects = self.object_extractor.extract_objects(grid)
        color_hist = self.compute_color_histogram(grid)
        symmetry = self.detect_grid_symmetry(grid)

        height = len(grid)
        width = len(grid[0])

        return {
            'n_objects': float(len(objects)),
            'n_colors': float(len([c for c, count in color_hist.items() if count > 0])),
            'grid_height': float(height),
            'grid_width': float(width),
            'grid_size': float(height * width),
            'max_object_size': float(max([obj.size for obj in objects], default=0)),
            'min_object_size': float(min([obj.size for obj in objects], default=0)),
            'avg_object_size': float(sum([obj.size for obj in objects]) / len(objects) if objects else 0),
            'has_symmetry_v': 1.0 if symmetry['vertical'] else 0.0,
            'has_symmetry_h': 1.0 if symmetry['horizontal'] else 0.0,
            'has_symmetry_rot90': 1.0 if symmetry['rotational_90'] else 0.0,
            'has_symmetry_rot180': 1.0 if symmetry['rotational_180'] else 0.0,
            'color_diversity': len(color_hist) / 10.0,  # Normalized (max 10 colors)
            'background_ratio': color_hist.get(0, 0) / (height * width) if (height * width) > 0 else 0.0,
        }

    def find_repeating_pattern(self, grid: List[List[int]]) -> Optional[Dict[str, Any]]:
        """Detect if grid is a tiled repetition of a smaller pattern."""
        height = len(grid)
        width = len(grid[0]) if grid else 0

        if not grid or height == 0 or width == 0:
            return None

        # Try different tile sizes
        for tile_h in range(1, height // 2 + 1):
            for tile_w in range(1, width // 2 + 1):
                if height % tile_h == 0 and width % tile_w == 0:
                    if self._is_tiled(grid, tile_h, tile_w):
                        return {
                            'type': 'tiling',
                            'tile_height': tile_h,
                            'tile_width': tile_w,
                            'repetitions_v': height // tile_h,
                            'repetitions_h': width // tile_w,
                        }

        return None

    def _is_tiled(self, grid: List[List[int]], tile_h: int, tile_w: int) -> bool:
        """Check if grid is a perfect tiling of tile_h x tile_w pattern."""
        # Extract base tile
        base_tile = [row[:tile_w] for row in grid[:tile_h]]

        # Check if entire grid is repetitions of base tile
        height = len(grid)
        width = len(grid[0])

        for r in range(0, height, tile_h):
            for c in range(0, width, tile_w):
                # Extract current tile
                for tr in range(tile_h):
                    for tc in range(tile_w):
                        if grid[r + tr][c + tc] != base_tile[tr][tc]:
                            return False

        return True

    def compare_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Compare two grids and return differences."""
        # Check if shapes match
        if len(grid1) != len(grid2):
            return {
                'size_changed': True,
                'shape1': (len(grid1), len(grid1[0]) if grid1 else 0),
                'shape2': (len(grid2), len(grid2[0]) if grid2 else 0),
            }

        if not grid1 or not grid2:
            return {'size_changed': True}

        if len(grid1[0]) != len(grid2[0]):
            return {
                'size_changed': True,
                'shape1': (len(grid1), len(grid1[0])),
                'shape2': (len(grid2), len(grid2[0])),
            }

        height = len(grid1)
        width = len(grid1[0])

        # Count differences
        differences = []
        for r in range(height):
            for c in range(width):
                if grid1[r][c] != grid2[r][c]:
                    differences.append({
                        'position': (r, c),
                        'from_color': grid1[r][c],
                        'to_color': grid2[r][c],
                    })

        total_cells = height * width

        return {
            'size_changed': False,
            'n_differences': len(differences),
            'differences': differences,
            'percent_changed': len(differences) / total_cells if total_cells > 0 else 0.0,
            'is_identical': len(differences) == 0,
        }
