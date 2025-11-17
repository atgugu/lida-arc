"""
Cognitive primitives: minimal set of basic perceptual and manipulation operations.

These primitives form the foundation for learning higher-level transformations
through demonstration analysis and category induction.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy

from .perception import GridObject, ObjectExtractor, GridAnalyzer


class Primitive(ABC):
    """Base class for cognitive primitives."""

    def __init__(self, name: str, category: str):
        """
        Args:
            name: Unique identifier for the primitive
            category: Category of primitive (perceptual, manipulation, spatial, logical)
        """
        self.name = name
        self.category = category

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the primitive operation."""
        pass

    def get_features(self) -> Dict[str, float]:
        """Return feature vector for PAM encoding.

        Subclasses can override to provide more specific features.
        """
        return {
            self.category: 1.0,
            f'primitive_{self.name}': 1.0
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# PERCEPTUAL PRIMITIVES (5)
# =============================================================================

class DetectObjectsPrimitive(Primitive):
    """Extract connected components from a grid."""

    def __init__(self, background_color: int = 0, connectivity: int = 4):
        super().__init__('detect_objects', 'perceptual')
        self.extractor = ObjectExtractor(background_color, connectivity)

    def execute(self, grid: List[List[int]]) -> List[GridObject]:
        """Extract all objects from grid.

        Args:
            grid: Input grid

        Returns:
            List of GridObject instances
        """
        return self.extractor.extract_objects(grid)

    def get_features(self) -> Dict[str, float]:
        return {
            'perceptual': 1.0,
            'object_based': 1.0,
            'analysis': 1.0,
        }


class CompareGridsPrimitive(Primitive):
    """Compare two grids and identify differences."""

    def __init__(self):
        super().__init__('compare_grids', 'perceptual')
        self.analyzer = GridAnalyzer()

    def execute(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Find differences between two grids.

        Args:
            grid1: First grid
            grid2: Second grid

        Returns:
            Dictionary with comparison results
        """
        return self.analyzer.compare_grids(grid1, grid2)

    def get_features(self) -> Dict[str, float]:
        return {
            'perceptual': 1.0,
            'comparison': 1.0,
            'analysis': 1.0,
        }


class FindPatternPrimitive(Primitive):
    """Detect repeating patterns in a grid."""

    def __init__(self):
        super().__init__('find_pattern', 'perceptual')
        self.analyzer = GridAnalyzer()

    def execute(self, grid: List[List[int]]) -> Optional[Dict[str, Any]]:
        """Detect if grid contains a repeating pattern.

        Args:
            grid: Input grid

        Returns:
            Pattern information if found, None otherwise
        """
        return self.analyzer.find_repeating_pattern(grid)

    def get_features(self) -> Dict[str, float]:
        return {
            'perceptual': 1.0,
            'pattern_detection': 1.0,
            'analysis': 1.0,
        }


class MatchObjectsPrimitive(Primitive):
    """Find correspondence between two sets of objects."""

    def __init__(self):
        super().__init__('match_objects', 'perceptual')

    def execute(self, objects1: List[GridObject], objects2: List[GridObject]) -> Dict[str, Any]:
        """Find best matching between two object sets.

        Args:
            objects1: First set of objects
            objects2: Second set of objects

        Returns:
            Dictionary with matches and unmatched objects
        """
        matches = []
        unmatched1 = set(range(len(objects1)))
        unmatched2 = set(range(len(objects2)))

        # Greedy matching based on similarity
        for i, obj1 in enumerate(objects1):
            best_match = None
            best_score = -1.0

            for j, obj2 in enumerate(objects2):
                if j not in unmatched2:
                    continue

                score = self._compute_similarity(obj1, obj2)
                if score > best_score:
                    best_score = score
                    best_match = j

            if best_match is not None and best_score > 0.3:
                matches.append((i, best_match, best_score))
                unmatched1.discard(i)
                unmatched2.discard(best_match)

        return {
            'matches': matches,
            'unmatched1': list(unmatched1),
            'unmatched2': list(unmatched2),
        }

    def _compute_similarity(self, obj1: GridObject, obj2: GridObject) -> float:
        """Compute similarity score between two objects."""
        score = 0.0

        # Color match
        if obj1.color == obj2.color:
            score += 0.5

        # Size similarity
        if obj1.size > 0 and obj2.size > 0:
            size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
            score += 0.3 * size_ratio

        # Shape similarity
        if obj1.get_shape_signature() == obj2.get_shape_signature():
            score += 0.2

        return score

    def get_features(self) -> Dict[str, float]:
        return {
            'perceptual': 1.0,
            'correspondence': 1.0,
            'matching': 1.0,
        }


class AnalyzeFeaturesPrimitive(Primitive):
    """Extract high-level features from a grid."""

    def __init__(self):
        super().__init__('analyze_features', 'perceptual')
        self.analyzer = GridAnalyzer()

    def execute(self, grid: List[List[int]]) -> Dict[str, float]:
        """Compute grid features.

        Args:
            grid: Input grid

        Returns:
            Dictionary of feature name -> value
        """
        return self.analyzer.compute_grid_features(grid)

    def get_features(self) -> Dict[str, float]:
        return {
            'perceptual': 1.0,
            'analysis': 1.0,
            'feature_extraction': 1.0,
        }


# =============================================================================
# MANIPULATION PRIMITIVES (8)
# =============================================================================

class RecolorPrimitive(Primitive):
    """Apply color mapping to a grid."""

    def __init__(self):
        super().__init__('recolor', 'manipulation')

    def execute(self, grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
        """Apply color mapping.

        Args:
            grid: Input grid
            color_map: Dictionary mapping from_color -> to_color

        Returns:
            New grid with colors mapped
        """
        new_grid = []
        for row in grid:
            new_row = [color_map.get(val, val) for val in row]
            new_grid.append(new_row)
        return new_grid

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'color_based': 1.0,
            'transformation': 1.0,
        }


class RotateGridPrimitive(Primitive):
    """Rotate grid by specified angle."""

    def __init__(self, angle: int):
        """
        Args:
            angle: Rotation angle in degrees (90, 180, or 270)
        """
        super().__init__(f'rotate_{angle}', 'manipulation')
        self.angle = angle

    def execute(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid.

        Args:
            grid: Input grid

        Returns:
            Rotated grid
        """
        if self.angle == 90:
            return self._rotate_90_cw(grid)
        elif self.angle == 180:
            return self._rotate_180(grid)
        elif self.angle == 270:
            return self._rotate_90_ccw(grid)
        return grid

    def _rotate_90_cw(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate 90 degrees clockwise."""
        if not grid or not grid[0]:
            return grid
        height = len(grid)
        width = len(grid[0])
        return [[grid[height-1-r][c] for r in range(height)] for c in range(width)]

    def _rotate_180(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate 180 degrees."""
        if not grid or not grid[0]:
            return grid
        return [row[::-1] for row in grid[::-1]]

    def _rotate_90_ccw(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate 90 degrees counter-clockwise."""
        if not grid or not grid[0]:
            return grid
        height = len(grid)
        width = len(grid[0])
        return [[grid[r][width-1-c] for r in range(height)] for c in range(width)]

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'geometric': 1.0,
            'rotation': 1.0,
            f'angle_{self.angle}': 1.0,
        }


class ReflectGridPrimitive(Primitive):
    """Reflect grid across an axis."""

    def __init__(self, axis: str):
        """
        Args:
            axis: 'horizontal', 'vertical', or 'diagonal'
        """
        super().__init__(f'reflect_{axis}', 'manipulation')
        self.axis = axis

    def execute(self, grid: List[List[int]]) -> List[List[int]]:
        """Reflect grid.

        Args:
            grid: Input grid

        Returns:
            Reflected grid
        """
        if not grid or not grid[0]:
            return grid

        if self.axis == 'horizontal':
            return grid[::-1]
        elif self.axis == 'vertical':
            return [row[::-1] for row in grid]
        elif self.axis == 'diagonal':
            # Transpose
            height = len(grid)
            width = len(grid[0])
            return [[grid[r][c] for r in range(height)] for c in range(width)]
        return grid

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'geometric': 1.0,
            'reflection': 1.0,
            f'axis_{self.axis}': 1.0,
        }


class CropGridPrimitive(Primitive):
    """Crop grid to bounding box."""

    def __init__(self):
        super().__init__('crop', 'manipulation')

    def execute(self, grid: List[List[int]],
                min_r: int, min_c: int, max_r: int, max_c: int) -> List[List[int]]:
        """Crop grid to specified region.

        Args:
            grid: Input grid
            min_r, min_c, max_r, max_c: Bounding box coordinates

        Returns:
            Cropped grid
        """
        if not grid or not grid[0]:
            return grid

        height = len(grid)
        width = len(grid[0])

        # Clamp coordinates
        min_r = max(0, min(min_r, height-1))
        max_r = max(0, min(max_r, height-1))
        min_c = max(0, min(min_c, width-1))
        max_c = max(0, min(max_c, width-1))

        cropped = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(grid[r][c])
            cropped.append(row)

        return cropped

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'spatial': 1.0,
            'cropping': 1.0,
        }


class ExtendGridPrimitive(Primitive):
    """Extend grid with padding."""

    def __init__(self):
        super().__init__('extend', 'manipulation')

    def execute(self, grid: List[List[int]],
                top: int = 0, bottom: int = 0, left: int = 0, right: int = 0,
                fill_value: int = 0) -> List[List[int]]:
        """Extend grid with padding.

        Args:
            grid: Input grid
            top, bottom, left, right: Padding amounts
            fill_value: Value to fill padding with

        Returns:
            Extended grid
        """
        if not grid or not grid[0]:
            return grid

        width = len(grid[0])

        # Add top padding
        extended = [[fill_value] * (left + width + right) for _ in range(top)]

        # Add middle rows with left and right padding
        for row in grid:
            padded_row = [fill_value] * left + row + [fill_value] * right
            extended.append(padded_row)

        # Add bottom padding
        extended.extend([[fill_value] * (left + width + right) for _ in range(bottom)])

        return extended

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'spatial': 1.0,
            'extending': 1.0,
        }


class TileGridPrimitive(Primitive):
    """Tile a pattern across a grid."""

    def __init__(self):
        super().__init__('tile', 'manipulation')

    def execute(self, pattern: List[List[int]], repeat_v: int, repeat_h: int) -> List[List[int]]:
        """Tile pattern.

        Args:
            pattern: Pattern to tile
            repeat_v: Vertical repetitions
            repeat_h: Horizontal repetitions

        Returns:
            Tiled grid
        """
        if not pattern or not pattern[0]:
            return pattern

        tiled = []
        for _ in range(repeat_v):
            for row in pattern:
                tiled_row = row * repeat_h
                tiled.append(tiled_row)

        return tiled

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'spatial': 1.0,
            'tiling': 1.0,
            'pattern_based': 1.0,
        }


class OverlayGridsPrimitive(Primitive):
    """Overlay two grids with specified mode."""

    def __init__(self):
        super().__init__('overlay', 'manipulation')

    def execute(self, grid1: List[List[int]], grid2: List[List[int]],
                mode: str = 'replace') -> List[List[int]]:
        """Overlay two grids.

        Args:
            grid1: Base grid
            grid2: Overlay grid
            mode: 'replace' (grid2 overwrites grid1 where non-zero),
                  'add' (sum values), 'max' (max value), 'min' (min value)

        Returns:
            Overlaid grid
        """
        if not grid1 or not grid1[0]:
            return grid2 if grid2 else grid1

        if not grid2 or not grid2[0]:
            return grid1

        height = min(len(grid1), len(grid2))
        width = min(len(grid1[0]), len(grid2[0]))

        result = []
        for r in range(height):
            row = []
            for c in range(width):
                val1 = grid1[r][c] if r < len(grid1) and c < len(grid1[0]) else 0
                val2 = grid2[r][c] if r < len(grid2) and c < len(grid2[0]) else 0

                if mode == 'replace':
                    row.append(val2 if val2 != 0 else val1)
                elif mode == 'add':
                    row.append((val1 + val2) % 10)  # Keep in 0-9 range
                elif mode == 'max':
                    row.append(max(val1, val2))
                elif mode == 'min':
                    row.append(min(val1, val2))
                else:
                    row.append(val1)

            result.append(row)

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'spatial': 1.0,
            'composition': 1.0,
        }


class FillBackgroundPrimitive(Primitive):
    """Fill all background (0) cells with specified color."""

    def __init__(self):
        super().__init__('fill_background', 'manipulation')

    def execute(self, grid: List[List[int]], fill_color: int) -> List[List[int]]:
        """Fill background.

        Args:
            grid: Input grid
            fill_color: Color to fill background with

        Returns:
            Grid with background filled
        """
        return [[fill_color if cell == 0 else cell for cell in row] for row in grid]

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'color_based': 1.0,
            'filling': 1.0,
        }


# =============================================================================
# PRIMITIVE LIBRARY
# =============================================================================

class PrimitiveLibrary:
    """Central registry of all cognitive primitives."""

    def __init__(self):
        self.primitives: Dict[str, Primitive] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all default primitives."""
        # Perceptual (5)
        self.register(DetectObjectsPrimitive())
        self.register(CompareGridsPrimitive())
        self.register(FindPatternPrimitive())
        self.register(MatchObjectsPrimitive())
        self.register(AnalyzeFeaturesPrimitive())

        # Manipulation - Geometric (6)
        for angle in [90, 180, 270]:
            self.register(RotateGridPrimitive(angle))
        for axis in ['horizontal', 'vertical', 'diagonal']:
            self.register(ReflectGridPrimitive(axis))

        # Manipulation - Spatial (5)
        self.register(CropGridPrimitive())
        self.register(ExtendGridPrimitive())
        self.register(TileGridPrimitive())
        self.register(OverlayGridsPrimitive())

        # Manipulation - Color (2)
        self.register(RecolorPrimitive())
        self.register(FillBackgroundPrimitive())

    def register(self, primitive: Primitive):
        """Add a primitive to the library."""
        self.primitives[primitive.name] = primitive

    def get(self, name: str) -> Optional[Primitive]:
        """Retrieve a primitive by name."""
        return self.primitives.get(name)

    def get_all_by_category(self, category: str) -> List[Primitive]:
        """Get all primitives of a given category."""
        return [p for p in self.primitives.values() if p.category == category]

    def get_all_names(self) -> List[str]:
        """Get names of all registered primitives."""
        return list(self.primitives.keys())

    def get_features_for_pam(self) -> Dict[str, Dict[str, float]]:
        """Get feature vectors for all primitives for PAM seeding."""
        return {name: prim.get_features() for name, prim in self.primitives.items()}

    def __len__(self) -> int:
        return len(self.primitives)

    def __contains__(self, name: str) -> bool:
        return name in self.primitives
