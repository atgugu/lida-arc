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
# OBJECT-BASED PRIMITIVES
# =============================================================================

class RenderObjectsPrimitive(Primitive):
    """Convert list of GridObjects back to a grid."""

    def __init__(self):
        super().__init__('render_objects', 'object_manipulation')

    def execute(self, objects: List[GridObject],
                width: int, height: int,
                background_color: int = 0) -> List[List[int]]:
        """Render objects onto a grid.

        Args:
            objects: List of GridObjects to render
            width: Grid width
            height: Grid height
            background_color: Background fill color

        Returns:
            Rendered grid
        """
        # Create blank grid
        grid = [[background_color for _ in range(width)] for _ in range(height)]

        # Render each object
        for obj in objects:
            for r, c in obj.pixels:
                if 0 <= r < height and 0 <= c < width:
                    grid[r][c] = obj.color

        return grid

    def get_features(self) -> Dict[str, float]:
        return {
            'object_manipulation': 1.0,
            'rendering': 1.0,
            'synthesis': 1.0,
        }


class MoveObjectPrimitive(Primitive):
    """Translate an object to a new position."""

    def __init__(self):
        super().__init__('move_object', 'object_manipulation')

    def execute(self, obj: GridObject, delta_r: int, delta_c: int) -> GridObject:
        """Move object by delta.

        Args:
            obj: Object to move
            delta_r: Row offset
            delta_c: Column offset

        Returns:
            New GridObject at translated position
        """
        new_pixels = frozenset((r + delta_r, c + delta_c) for r, c in obj.pixels)
        min_r, min_c, max_r, max_c = obj.bounding_box
        new_bbox = (min_r + delta_r, min_c + delta_c,
                    max_r + delta_r, max_c + delta_c)

        return GridObject(
            object_id=f"{obj.object_id}_moved",
            pixels=new_pixels,
            color=obj.color,
            bounding_box=new_bbox
        )

    def get_features(self) -> Dict[str, float]:
        return {
            'object_manipulation': 1.0,
            'spatial': 1.0,
            'translation': 1.0,
        }


class ScaleObjectPrimitive(Primitive):
    """Scale an object by a factor."""

    def __init__(self):
        super().__init__('scale_object', 'object_manipulation')

    def execute(self, obj: GridObject, scale_factor: float) -> GridObject:
        """Scale object.

        Args:
            obj: Object to scale
            scale_factor: Scaling factor (e.g., 2.0 for 2x, 0.5 for half)

        Returns:
            New GridObject at scaled size
        """
        if scale_factor == 1.0:
            return obj

        min_r, min_c, _, _ = obj.bounding_box
        centroid_r, centroid_c = obj.centroid

        # Scale relative to centroid
        new_pixels = set()
        for r, c in obj.pixels:
            rel_r = (r - centroid_r) * scale_factor
            rel_c = (c - centroid_c) * scale_factor
            new_r = int(round(centroid_r + rel_r))
            new_c = int(round(centroid_c + rel_c))
            new_pixels.add((new_r, new_c))

        if not new_pixels:
            return obj

        new_pixels = frozenset(new_pixels)

        # Compute new bounding box
        rows = [p[0] for p in new_pixels]
        cols = [p[1] for p in new_pixels]
        new_bbox = (min(rows), min(cols), max(rows), max(cols))

        return GridObject(
            object_id=f"{obj.object_id}_scaled",
            pixels=new_pixels,
            color=obj.color,
            bounding_box=new_bbox
        )

    def get_features(self) -> Dict[str, float]:
        return {
            'object_manipulation': 1.0,
            'geometric': 1.0,
            'scaling': 1.0,
        }


class ReplicateObjectPrimitive(Primitive):
    """Create N copies of an object."""

    def __init__(self):
        super().__init__('replicate_object', 'object_manipulation')

    def execute(self, obj: GridObject, count: int,
                spacing_r: int = 0, spacing_c: int = 0) -> List[GridObject]:
        """Replicate object with spacing.

        Args:
            obj: Object to replicate
            count: Number of copies (including original)
            spacing_r: Row spacing between copies
            spacing_c: Column spacing between copies

        Returns:
            List of GridObjects (original + copies)
        """
        copies = [obj]

        for i in range(1, count):
            delta_r = i * (obj.height + spacing_r)
            delta_c = i * (obj.width + spacing_c)

            new_pixels = frozenset((r + delta_r, c + delta_c) for r, c in obj.pixels)
            min_r, min_c, max_r, max_c = obj.bounding_box
            new_bbox = (min_r + delta_r, min_c + delta_c,
                        max_r + delta_r, max_c + delta_c)

            copy = GridObject(
                object_id=f"{obj.object_id}_copy{i}",
                pixels=new_pixels,
                color=obj.color,
                bounding_box=new_bbox
            )
            copies.append(copy)

        return copies

    def get_features(self) -> Dict[str, float]:
        return {
            'object_manipulation': 1.0,
            'replication': 1.0,
            'pattern_generation': 1.0,
        }


class RecolorObjectPrimitive(Primitive):
    """Change an object's color."""

    def __init__(self):
        super().__init__('recolor_object', 'object_manipulation')

    def execute(self, obj: GridObject, new_color: int) -> GridObject:
        """Change object color.

        Args:
            obj: Object to recolor
            new_color: New color value (0-9)

        Returns:
            New GridObject with updated color
        """
        return GridObject(
            object_id=f"{obj.object_id}_recolored",
            pixels=obj.pixels,
            color=new_color,
            bounding_box=obj.bounding_box
        )

    def get_features(self) -> Dict[str, float]:
        return {
            'object_manipulation': 1.0,
            'color_based': 1.0,
            'transformation': 1.0,
        }


# =============================================================================
# SIZE/SHAPE CHANGE PRIMITIVES
# =============================================================================

class ScaleGridPrimitive(Primitive):
    """Scale entire grid up or down."""

    def __init__(self):
        super().__init__('scale_grid', 'manipulation')

    def execute(self, grid: List[List[int]], scale_factor: float) -> List[List[int]]:
        """Scale grid by factor.

        Args:
            grid: Input grid
            scale_factor: Scaling factor (e.g., 2.0 for 2x, 0.5 for half)

        Returns:
            Scaled grid
        """
        if not grid or not grid[0] or scale_factor == 1.0:
            return grid

        old_height = len(grid)
        old_width = len(grid[0])

        new_height = max(1, int(round(old_height * scale_factor)))
        new_width = max(1, int(round(old_width * scale_factor)))

        # Nearest-neighbor sampling
        scaled = []
        for r in range(new_height):
            row = []
            src_r = min(int(r / scale_factor), old_height - 1)
            for c in range(new_width):
                src_c = min(int(c / scale_factor), old_width - 1)
                row.append(grid[src_r][src_c])
            scaled.append(row)

        return scaled

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'geometric': 1.0,
            'scaling': 1.0,
            'size_change': 1.0,
        }


class AutoCropPrimitive(Primitive):
    """Crop grid to minimal bounding box containing non-background pixels."""

    def __init__(self, background_color: int = 0):
        super().__init__('auto_crop', 'manipulation')
        self.background_color = background_color

    def execute(self, grid: List[List[int]]) -> List[List[int]]:
        """Auto-crop to content.

        Args:
            grid: Input grid

        Returns:
            Cropped grid containing all non-background pixels
        """
        if not grid or not grid[0]:
            return grid

        height = len(grid)
        width = len(grid[0])

        # Find bounding box of non-background
        min_r, max_r = height, -1
        min_c, max_c = width, -1

        for r in range(height):
            for c in range(width):
                if grid[r][c] != self.background_color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        # If no content found, return original
        if max_r < 0:
            return grid

        # Crop to bounding box
        cropped = []
        for r in range(min_r, max_r + 1):
            row = [grid[r][c] for c in range(min_c, max_c + 1)]
            cropped.append(row)

        return cropped

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'spatial': 1.0,
            'cropping': 1.0,
            'size_change': 1.0,
        }


class ResizeToTargetPrimitive(Primitive):
    """Resize grid to match target dimensions."""

    def __init__(self):
        super().__init__('resize_to_target', 'manipulation')

    def execute(self, grid: List[List[int]],
                target_height: int, target_width: int,
                fill_color: int = 0,
                align: str = 'center') -> List[List[int]]:
        """Resize grid to target dimensions.

        Args:
            grid: Input grid
            target_height: Target height
            target_width: Target width
            fill_color: Color for padding
            align: 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'

        Returns:
            Resized grid
        """
        if not grid or not grid[0]:
            return [[fill_color] * target_width for _ in range(target_height)]

        current_height = len(grid)
        current_width = len(grid[0])

        # If same size, return copy
        if current_height == target_height and current_width == target_width:
            return [row[:] for row in grid]

        # Create target grid filled with fill_color
        result = [[fill_color for _ in range(target_width)] for _ in range(target_height)]

        # Compute offset based on alignment
        if align == 'center':
            offset_r = (target_height - current_height) // 2
            offset_c = (target_width - current_width) // 2
        elif align == 'top-left':
            offset_r, offset_c = 0, 0
        elif align == 'top-right':
            offset_r = 0
            offset_c = target_width - current_width
        elif align == 'bottom-left':
            offset_r = target_height - current_height
            offset_c = 0
        elif align == 'bottom-right':
            offset_r = target_height - current_height
            offset_c = target_width - current_width
        else:
            offset_r, offset_c = 0, 0

        # Copy grid content to target
        for r in range(current_height):
            for c in range(current_width):
                target_r = offset_r + r
                target_c = offset_c + c
                if 0 <= target_r < target_height and 0 <= target_c < target_width:
                    result[target_r][target_c] = grid[r][c]

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'manipulation': 1.0,
            'spatial': 1.0,
            'resizing': 1.0,
            'size_change': 1.0,
        }


# =============================================================================
# CONDITIONAL/RULE-BASED PRIMITIVES
# =============================================================================

class ConditionalPrimitive(Primitive):
    """Base class for conditional transformations using spatial predicates."""

    def __init__(self, name: str):
        super().__init__(name, 'conditional')

    @staticmethod
    def get_neighbors(grid: List[List[int]], r: int, c: int, include_diagonals: bool = True) -> List[Tuple[int, int, int]]:
        """Get neighbors of a pixel.

        Returns:
            List of (row, col, color) tuples for valid neighbors
        """
        height = len(grid)
        width = len(grid[0]) if grid else 0
        neighbors = []

        # Orthogonal neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbors.append((nr, nc, grid[nr][nc]))

        # Diagonal neighbors
        if include_diagonals:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbors.append((nr, nc, grid[nr][nc]))

        return neighbors

    @staticmethod
    def has_neighbor_with_color(grid: List[List[int]], r: int, c: int,
                                 target_color: int, include_diagonals: bool = True) -> bool:
        """Check if pixel has a neighbor with specific color."""
        neighbors = ConditionalPrimitive.get_neighbors(grid, r, c, include_diagonals)
        return any(color == target_color for _, _, color in neighbors)

    @staticmethod
    def is_on_edge(grid: List[List[int]], r: int, c: int) -> bool:
        """Check if pixel is on grid edge."""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        return r == 0 or r == height - 1 or c == 0 or c == width - 1

    @staticmethod
    def is_in_corner(grid: List[List[int]], r: int, c: int) -> bool:
        """Check if pixel is in a corner."""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        return ((r == 0 or r == height - 1) and (c == 0 or c == width - 1))

    @staticmethod
    def is_isolated(grid: List[List[int]], r: int, c: int) -> bool:
        """Check if pixel has no neighbors of the same color."""
        pixel_color = grid[r][c]
        if pixel_color == 0:  # Background is never isolated
            return False
        neighbors = ConditionalPrimitive.get_neighbors(grid, r, c, include_diagonals=True)
        return not any(color == pixel_color for _, _, color in neighbors)

    @staticmethod
    def count_neighbors_with_color(grid: List[List[int]], r: int, c: int,
                                    target_color: int, include_diagonals: bool = True) -> int:
        """Count neighbors with specific color."""
        neighbors = ConditionalPrimitive.get_neighbors(grid, r, c, include_diagonals)
        return sum(1 for _, _, color in neighbors if color == target_color)


class RecolorIfHasNeighborPrimitive(ConditionalPrimitive):
    """Recolor pixels that have a neighbor of specific color."""

    def __init__(self):
        super().__init__('recolor_if_has_neighbor')

    def execute(self, grid: List[List[int]], neighbor_color: int, new_color: int,
                target_color: Optional[int] = None) -> List[List[int]]:
        """Recolor pixels that have a neighbor with specified color.

        Args:
            grid: Input grid
            neighbor_color: Color to check for in neighbors
            new_color: Color to change matching pixels to
            target_color: Only check pixels of this color (None = all non-background)

        Returns:
            Transformed grid
        """
        result = [row[:] for row in grid]  # Deep copy

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                pixel_color = grid[r][c]

                # Skip if not target color
                if target_color is not None and pixel_color != target_color:
                    continue

                # Skip background unless explicitly targeted
                if target_color is None and pixel_color == 0:
                    continue

                # Check condition
                if self.has_neighbor_with_color(grid, r, c, neighbor_color):
                    result[r][c] = new_color

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'conditional': 1.0,
            'neighbor_based': 1.0,
            'color_based': 1.0,
            'rule_based': 1.0,
        }


class RecolorIfIsolatedPrimitive(ConditionalPrimitive):
    """Recolor isolated pixels (no same-color neighbors)."""

    def __init__(self):
        super().__init__('recolor_if_isolated')

    def execute(self, grid: List[List[int]], new_color: int,
                target_color: Optional[int] = None) -> List[List[int]]:
        """Recolor isolated pixels.

        Args:
            grid: Input grid
            new_color: Color to change isolated pixels to
            target_color: Only check pixels of this color (None = all non-background)

        Returns:
            Transformed grid
        """
        result = [row[:] for row in grid]

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                pixel_color = grid[r][c]

                # Skip if not target color
                if target_color is not None and pixel_color != target_color:
                    continue

                # Skip background
                if pixel_color == 0:
                    continue

                # Check condition
                if self.is_isolated(grid, r, c):
                    result[r][c] = new_color

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'conditional': 1.0,
            'isolation_based': 1.0,
            'color_based': 1.0,
            'rule_based': 1.0,
        }


class RecolorIfOnEdgePrimitive(ConditionalPrimitive):
    """Recolor pixels on grid edges."""

    def __init__(self):
        super().__init__('recolor_if_on_edge')

    def execute(self, grid: List[List[int]], new_color: int,
                target_color: Optional[int] = None) -> List[List[int]]:
        """Recolor pixels on edges.

        Args:
            grid: Input grid
            new_color: Color to change edge pixels to
            target_color: Only check pixels of this color (None = all non-background)

        Returns:
            Transformed grid
        """
        result = [row[:] for row in grid]

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                pixel_color = grid[r][c]

                # Skip if not target color
                if target_color is not None and pixel_color != target_color:
                    continue

                # Skip background unless explicitly targeted
                if target_color is None and pixel_color == 0:
                    continue

                # Check condition
                if self.is_on_edge(grid, r, c):
                    result[r][c] = new_color

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'conditional': 1.0,
            'edge_based': 1.0,
            'color_based': 1.0,
            'rule_based': 1.0,
        }


class RemoveIfIsolatedPrimitive(ConditionalPrimitive):
    """Remove isolated pixels by setting them to background."""

    def __init__(self):
        super().__init__('remove_if_isolated')

    def execute(self, grid: List[List[int]], target_color: Optional[int] = None) -> List[List[int]]:
        """Remove isolated pixels.

        Args:
            grid: Input grid
            target_color: Only check pixels of this color (None = all non-background)

        Returns:
            Transformed grid
        """
        result = [row[:] for row in grid]

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                pixel_color = grid[r][c]

                # Skip if not target color
                if target_color is not None and pixel_color != target_color:
                    continue

                # Skip background
                if pixel_color == 0:
                    continue

                # Check condition
                if self.is_isolated(grid, r, c):
                    result[r][c] = 0  # Remove (set to background)

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'conditional': 1.0,
            'isolation_based': 1.0,
            'removal': 1.0,
            'rule_based': 1.0,
        }


# =============================================================================
# MORPHOLOGICAL / ITERATIVE PRIMITIVES
# =============================================================================

class DilatePrimitive(Primitive):
    """Expand regions of specified color by N iterations (morphological dilation)."""

    def __init__(self):
        super().__init__('dilate', 'morphology')

    def execute(self, grid: List[List[int]], color: int, iterations: int = 1,
                background: int = 0) -> List[List[int]]:
        """Expand pixels of 'color' into 'background' by N steps.

        Args:
            grid: Input grid
            color: Color to expand
            iterations: Number of dilation steps
            background: Color to expand into (default: 0)

        Returns:
            Grid with dilated regions
        """
        result = [row[:] for row in grid]

        for _ in range(iterations):
            result = self._dilate_once(result, color, background)

        return result

    def _dilate_once(self, grid: List[List[int]], color: int, background: int) -> List[List[int]]:
        """Perform one dilation step."""
        result = [row[:] for row in grid]
        height = len(grid)
        width = len(grid[0]) if grid else 0

        for r in range(height):
            for c in range(width):
                if grid[r][c] == background:
                    # Check if any neighbor is the target color
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr][nc] == color:
                                result[r][c] = color
                                break

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'morphology': 1.0,
            'expansion': 1.0,
            'iterative': 1.0,
            'spatial': 1.0,
        }


class ErodePrimitive(Primitive):
    """Shrink regions of specified color by N iterations (morphological erosion)."""

    def __init__(self):
        super().__init__('erode', 'morphology')

    def execute(self, grid: List[List[int]], color: int, iterations: int = 1,
                background: int = 0) -> List[List[int]]:
        """Shrink pixels of 'color' that border 'background'.

        Args:
            grid: Input grid
            color: Color to shrink
            iterations: Number of erosion steps
            background: Color to erode to (default: 0)

        Returns:
            Grid with eroded regions
        """
        result = [row[:] for row in grid]

        for _ in range(iterations):
            result = self._erode_once(result, color, background)

        return result

    def _erode_once(self, grid: List[List[int]], color: int, background: int) -> List[List[int]]:
        """Perform one erosion step."""
        result = [row[:] for row in grid]
        height = len(grid)
        width = len(grid[0]) if grid else 0

        for r in range(height):
            for c in range(width):
                if grid[r][c] == color:
                    # Check if any neighbor is background
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr][nc] == background:
                                result[r][c] = background
                                break

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'morphology': 1.0,
            'shrinking': 1.0,
            'iterative': 1.0,
            'spatial': 1.0,
        }


class FloodFillPrimitive(Primitive):
    """Fill all connected regions of source color with fill color."""

    def __init__(self):
        super().__init__('flood_fill', 'morphology')

    def execute(self, grid: List[List[int]], source_color: int, fill_color: int) -> List[List[int]]:
        """Fill all connected components of source_color with fill_color.

        Args:
            grid: Input grid
            source_color: Color of regions to fill
            fill_color: Color to fill with

        Returns:
            Grid with filled regions
        """
        result = [row[:] for row in grid]
        height = len(grid)
        width = len(grid[0]) if grid else 0
        visited = set()

        for r in range(height):
            for c in range(width):
                if grid[r][c] == source_color and (r, c) not in visited:
                    self._flood_fill_region(result, r, c, source_color, fill_color, visited)

        return result

    def _flood_fill_region(self, grid: List[List[int]], start_r: int, start_c: int,
                           source_color: int, fill_color: int, visited: set):
        """BFS flood fill from starting position."""
        height = len(grid)
        width = len(grid[0]) if grid else 0
        queue = [(start_r, start_c)]
        visited.add((start_r, start_c))

        while queue:
            r, c = queue.pop(0)
            grid[r][c] = fill_color

            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and
                    (nr, nc) not in visited and grid[nr][nc] == source_color):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    def get_features(self) -> Dict[str, float]:
        return {
            'morphology': 1.0,
            'filling': 1.0,
            'connectivity': 1.0,
            'spatial': 1.0,
        }


class FillEnclosedPrimitive(Primitive):
    """Fill regions enclosed by boundary color."""

    def __init__(self):
        super().__init__('fill_enclosed', 'morphology')

    def execute(self, grid: List[List[int]], boundary_color: int, fill_color: int,
                background: int = 0) -> List[List[int]]:
        """Fill holes: regions of background surrounded by boundary.

        Args:
            grid: Input grid
            boundary_color: Color of enclosing boundary
            fill_color: Color to fill enclosed regions with
            background: Background color (default: 0)

        Returns:
            Grid with filled enclosed regions
        """
        height = len(grid)
        width = len(grid[0]) if grid else 0
        result = [row[:] for row in grid]

        # Flood fill from edges to mark exterior regions
        exterior = set()
        queue = []

        # Add all edge cells that are background
        for r in range(height):
            for c in range(width):
                if (r == 0 or r == height - 1 or c == 0 or c == width - 1):
                    if grid[r][c] == background:
                        queue.append((r, c))
                        exterior.add((r, c))

        # BFS to mark all exterior background cells
        while queue:
            r, c = queue.pop(0)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and
                    (nr, nc) not in exterior and grid[nr][nc] == background):
                    exterior.add((nr, nc))
                    queue.append((nr, nc))

        # Fill interior (non-exterior) background cells
        for r in range(height):
            for c in range(width):
                if grid[r][c] == background and (r, c) not in exterior:
                    result[r][c] = fill_color

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'morphology': 1.0,
            'filling': 1.0,
            'hole_filling': 1.0,
            'spatial': 1.0,
        }


class SpreadToNeighborsPrimitive(Primitive):
    """Spread source color to adjacent target pixels, N times."""

    def __init__(self):
        super().__init__('spread_to_neighbors', 'morphology')

    def execute(self, grid: List[List[int]], source_color: int, target_color: int,
                iterations: int = 1) -> List[List[int]]:
        """Propagate source_color into target_color regions.

        Args:
            grid: Input grid
            source_color: Color that spreads
            target_color: Color that gets replaced
            iterations: Number of propagation steps

        Returns:
            Grid after propagation
        """
        result = [row[:] for row in grid]

        for _ in range(iterations):
            result = self._spread_once(result, source_color, target_color)

        return result

    def _spread_once(self, grid: List[List[int]], source_color: int, target_color: int) -> List[List[int]]:
        """Perform one propagation step."""
        result = [row[:] for row in grid]
        height = len(grid)
        width = len(grid[0]) if grid else 0

        for r in range(height):
            for c in range(width):
                if grid[r][c] == target_color:
                    # Check if any neighbor is source_color
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr][nc] == source_color:
                                result[r][c] = source_color
                                break

        return result

    def get_features(self) -> Dict[str, float]:
        return {
            'morphology': 1.0,
            'propagation': 1.0,
            'iterative': 1.0,
            'spatial': 1.0,
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

        # Object Manipulation (5)
        self.register(RenderObjectsPrimitive())
        self.register(MoveObjectPrimitive())
        self.register(ScaleObjectPrimitive())
        self.register(ReplicateObjectPrimitive())
        self.register(RecolorObjectPrimitive())

        # Size/Shape Change (3)
        self.register(ScaleGridPrimitive())
        self.register(AutoCropPrimitive())
        self.register(ResizeToTargetPrimitive())

        # Conditional/Rule-Based (4)
        self.register(RecolorIfHasNeighborPrimitive())
        self.register(RecolorIfIsolatedPrimitive())
        self.register(RecolorIfOnEdgePrimitive())
        self.register(RemoveIfIsolatedPrimitive())

        # Morphological/Iterative (5)
        self.register(DilatePrimitive())
        self.register(ErodePrimitive())
        self.register(FloodFillPrimitive())
        self.register(FillEnclosedPrimitive())
        self.register(SpreadToNeighborsPrimitive())

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
