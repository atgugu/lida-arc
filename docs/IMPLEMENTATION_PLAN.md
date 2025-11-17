# LIDA-ARC Hybrid Bootstrapping Implementation Plan

## Overview

This document provides a detailed implementation plan for the hybrid bootstrapping approach to LIDA-ARC, which learns transformations from demonstrations without a pre-defined DSL.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Procedural Consolidation                          │
│  - Reusable schemes for successful transformations          │
│  - High utility operations available for quick application  │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ (consolidate successful ops)
                          │
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Abstract Operations (PAM)                         │
│  - Cluster-derived transformation concepts                  │
│  - Spreading activation reveals common patterns            │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ (abstract from specifics)
                          │
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Task-Specific Transformations (Episodic)         │
│  - Direct demonstration-derived transformations             │
│  - Stored in episodic memory with rich features            │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ (extract from demos)
                          │
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Cognitive Primitives                              │
│  - ~25 basic perceptual and manipulation operations         │
│  - Foundation for all learning                              │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. ARC Task Environment

**File**: `src/lida/env/arc_environment.py`

**Purpose**: Load and manage ARC tasks, provide grid observations.

**Key Classes**:
- `ARCTask`: Container for demonstration and test pairs
- `ARCEnvironment`: Environment adapter implementing base Environment interface
- `GridPair`: Input-output grid pair with metadata

**Implementation Details**:
```python
@dataclass
class GridPair:
    input: List[List[int]]  # 2D grid, values 0-9
    output: List[List[int]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (len(self.input), len(self.input[0]))

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (len(self.output), len(self.output[0]))

@dataclass
class ARCTask:
    task_id: str
    train: List[GridPair]  # 2-4 demonstration pairs
    test: List[GridPair]   # 1-2 test pairs

    def get_demonstration(self, idx: int) -> GridPair:
        return self.train[idx]

    def get_test(self, idx: int) -> GridPair:
        return self.test[idx]

class ARCEnvironment:
    def __init__(self, task: ARCTask):
        self.task = task
        self.current_demo_idx = 0
        self.current_grid = None

    def get_observation(self) -> Dict[str, Any]:
        """Returns current grid and metadata"""
        return {
            'grid': self.current_grid,
            'phase': 'demonstration' if self.current_demo_idx < len(self.task.train) else 'test',
            'demo_index': self.current_demo_idx
        }

    def load_task_from_json(path: str) -> ARCTask:
        """Load ARC task from JSON file"""
        pass
```

**Dependencies**: None (foundational)

**Tests**:
- Load sample ARC JSON
- Validate grid shapes
- Test iteration through demonstrations

---

### 2. Object Extraction & Grid Analysis

**File**: `src/lida/perception/arc_perception.py`

**Purpose**: Extract structured objects and features from raw grids.

**Key Classes**:
- `GridObject`: Represents a connected component with properties
- `ObjectExtractor`: Extracts objects from grids
- `GridAnalyzer`: Computes grid-level properties

**Implementation Details**:
```python
@dataclass
class GridObject:
    object_id: str
    pixels: Set[Tuple[int, int]]  # Set of (row, col) coordinates
    color: int  # 0-9
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Center of mass"""
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (sum(rows) / len(rows), sum(cols) / len(cols))

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def width(self) -> int:
        return self.bounding_box[3] - self.bounding_box[1] + 1

    @property
    def height(self) -> int:
        return self.bounding_box[2] - self.bounding_box[0] + 1

    def get_shape_signature(self) -> str:
        """Canonical representation of shape (normalized to bbox)"""
        min_r, min_c, _, _ = self.bounding_box
        normalized = frozenset((r - min_r, c - min_c) for r, c in self.pixels)
        return str(hash(normalized))

    def has_symmetry_vertical(self) -> bool:
        """Check vertical axis symmetry"""
        pass

    def has_symmetry_horizontal(self) -> bool:
        """Check horizontal axis symmetry"""
        pass

class ObjectExtractor:
    def __init__(self, background_color: int = 0):
        self.background_color = background_color

    def extract_objects(self, grid: List[List[int]]) -> List[GridObject]:
        """Connected component analysis to find objects"""
        # Flood fill algorithm
        visited = set()
        objects = []

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if (r, c) not in visited and grid[r][c] != self.background_color:
                    # Start flood fill
                    obj_pixels = self._flood_fill(grid, r, c, visited)
                    if obj_pixels:
                        objects.append(self._create_object(obj_pixels, grid[r][c]))

        return objects

    def _flood_fill(self, grid, start_r, start_c, visited) -> Set[Tuple[int, int]]:
        """BFS flood fill"""
        target_color = grid[start_r][start_c]
        queue = [(start_r, start_c)]
        pixels = set()

        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited:
                continue
            if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                continue
            if grid[r][c] != target_color:
                continue

            visited.add((r, c))
            pixels.add((r, c))

            # Add 4-connected neighbors
            queue.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

        return pixels

    def _create_object(self, pixels: Set[Tuple[int, int]], color: int) -> GridObject:
        """Create GridObject from pixel set"""
        min_r = min(p[0] for p in pixels)
        max_r = max(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        max_c = max(p[1] for p in pixels)

        return GridObject(
            object_id=f"obj_{hash(frozenset(pixels))}",
            pixels=pixels,
            color=color,
            bounding_box=(min_r, min_c, max_r, max_c)
        )

class GridAnalyzer:
    """Analyze grid-level properties"""

    def compute_color_histogram(self, grid: List[List[int]]) -> Dict[int, int]:
        """Count pixels of each color"""
        hist = defaultdict(int)
        for row in grid:
            for val in row:
                hist[val] += 1
        return dict(hist)

    def detect_grid_symmetry(self, grid: List[List[int]]) -> Dict[str, bool]:
        """Check if entire grid has symmetry"""
        return {
            'vertical': self._check_vertical_symmetry(grid),
            'horizontal': self._check_horizontal_symmetry(grid),
            'rotational_180': self._check_rotational_180(grid)
        }

    def compute_grid_features(self, grid: List[List[int]]) -> Dict[str, float]:
        """Extract high-level features"""
        objects = ObjectExtractor().extract_objects(grid)
        color_hist = self.compute_color_histogram(grid)
        symmetry = self.detect_grid_symmetry(grid)

        return {
            'n_objects': len(objects),
            'n_colors': len([c for c, count in color_hist.items() if count > 0]),
            'grid_size': len(grid) * len(grid[0]),
            'max_object_size': max([obj.size for obj in objects], default=0),
            'has_symmetry': any(symmetry.values()),
            'color_diversity': len(color_hist) / 10.0,  # Normalized
        }
```

**Dependencies**: numpy (for array operations)

**Tests**:
- Extract objects from simple grids
- Verify connected components
- Test symmetry detection
- Validate feature extraction

---

### 3. Cognitive Primitives Library

**File**: `src/lida/arc/primitives.py`

**Purpose**: Define the minimal set of ~25 basic cognitive operations.

**Categories**:
1. **Perceptual** (5): detect, compare, find, match
2. **Structural** (5): iterate, select, group, filter
3. **Manipulation** (8): copy, move, recolor, resize, rotate, reflect, crop, extend
4. **Spatial** (4): overlay, tile, align, measure
5. **Logical** (3): if-then-else, for-each, compose

**Implementation Details**:
```python
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Dict

class Primitive(ABC):
    """Base class for cognitive primitives"""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the primitive operation"""
        pass

    @abstractmethod
    def get_features(self) -> Dict[str, float]:
        """Return feature vector for PAM encoding"""
        pass

# PERCEPTUAL PRIMITIVES

class DetectObjectsPrimitive(Primitive):
    def __init__(self):
        super().__init__('detect_objects', 'perceptual')
        self.extractor = ObjectExtractor()

    def execute(self, grid: List[List[int]]) -> List[GridObject]:
        return self.extractor.extract_objects(grid)

    def get_features(self) -> Dict[str, float]:
        return {'perceptual': 1.0, 'object_based': 1.0}

class CompareGridsPrimitive(Primitive):
    def __init__(self):
        super().__init__('compare_grids', 'perceptual')

    def execute(self, grid1: List[List[int]], grid2: List[List[int]]) -> Dict[str, Any]:
        """Find differences between two grids"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return {'size_changed': True, 'delta': None}

        differences = []
        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                if grid1[r][c] != grid2[r][c]:
                    differences.append({
                        'position': (r, c),
                        'from_color': grid1[r][c],
                        'to_color': grid2[r][c]
                    })

        return {
            'size_changed': False,
            'n_differences': len(differences),
            'differences': differences,
            'percent_changed': len(differences) / (len(grid1) * len(grid1[0]))
        }

    def get_features(self) -> Dict[str, float]:
        return {'perceptual': 1.0, 'comparison': 1.0}

class FindPatternPrimitive(Primitive):
    def __init__(self):
        super().__init__('find_pattern', 'perceptual')

    def execute(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Detect repeating patterns in grid"""
        # Check for tiling patterns
        patterns = []

        # Try different tile sizes
        for tile_h in range(1, len(grid) // 2 + 1):
            for tile_w in range(1, len(grid[0]) // 2 + 1):
                if self._is_tiled(grid, tile_h, tile_w):
                    patterns.append({
                        'type': 'tiling',
                        'tile_size': (tile_h, tile_w)
                    })

        return {'patterns': patterns}

    def _is_tiled(self, grid, tile_h, tile_w) -> bool:
        """Check if grid is a perfect tiling of tile_h x tile_w pattern"""
        if len(grid) % tile_h != 0 or len(grid[0]) % tile_w != 0:
            return False

        # Extract base tile
        base_tile = [row[:tile_w] for row in grid[:tile_h]]

        # Check if entire grid is repetitions of base tile
        for r in range(0, len(grid), tile_h):
            for c in range(0, len(grid[0]), tile_w):
                tile = [row[c:c+tile_w] for row in grid[r:r+tile_h]]
                if tile != base_tile:
                    return False

        return True

    def get_features(self) -> Dict[str, float]:
        return {'perceptual': 1.0, 'pattern_detection': 1.0}

class MatchObjectsPrimitive(Primitive):
    def __init__(self):
        super().__init__('match_objects', 'perceptual')

    def execute(self, objects1: List[GridObject], objects2: List[GridObject]) -> Dict[str, Any]:
        """Find correspondence between two sets of objects"""
        # Simple greedy matching based on color and size
        matches = []
        unmatched1 = set(range(len(objects1)))
        unmatched2 = set(range(len(objects2)))

        # Match by color first
        for i, obj1 in enumerate(objects1):
            best_match = None
            best_score = -1

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
            'unmatched2': list(unmatched2)
        }

    def _compute_similarity(self, obj1: GridObject, obj2: GridObject) -> float:
        """Similarity score between two objects"""
        score = 0.0

        # Color match
        if obj1.color == obj2.color:
            score += 0.5

        # Size similarity
        size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
        score += 0.3 * size_ratio

        # Shape similarity
        if obj1.get_shape_signature() == obj2.get_shape_signature():
            score += 0.2

        return score

    def get_features(self) -> Dict[str, float]:
        return {'perceptual': 1.0, 'correspondence': 1.0}

# MANIPULATION PRIMITIVES

class RecolorPrimitive(Primitive):
    def __init__(self):
        super().__init__('recolor', 'manipulation')

    def execute(self, grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
        """Apply color mapping to grid"""
        new_grid = []
        for row in grid:
            new_row = [color_map.get(val, val) for val in row]
            new_grid.append(new_row)
        return new_grid

    def get_features(self) -> Dict[str, float]:
        return {'manipulation': 1.0, 'color_based': 1.0}

class RotateGridPrimitive(Primitive):
    def __init__(self, angle: int = 90):
        super().__init__(f'rotate_{angle}', 'manipulation')
        self.angle = angle

    def execute(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid by angle (90, 180, 270)"""
        if self.angle == 90:
            return self._rotate_90_cw(grid)
        elif self.angle == 180:
            return self._rotate_180(grid)
        elif self.angle == 270:
            return self._rotate_90_ccw(grid)
        return grid

    def _rotate_90_cw(self, grid):
        """Rotate 90 degrees clockwise"""
        return [[grid[len(grid)-1-c][r] for c in range(len(grid))] for r in range(len(grid[0]))]

    def _rotate_180(self, grid):
        """Rotate 180 degrees"""
        return [[grid[len(grid)-1-r][len(grid[0])-1-c] for c in range(len(grid[0]))] for r in range(len(grid))]

    def _rotate_90_ccw(self, grid):
        """Rotate 90 degrees counter-clockwise"""
        return [[grid[c][len(grid[0])-1-r] for c in range(len(grid))] for r in range(len(grid[0]))]

    def get_features(self) -> Dict[str, float]:
        return {'manipulation': 1.0, 'geometric': 1.0, 'rotation': 1.0}

# ... (More primitives for reflect, crop, extend, overlay, etc.)

# PRIMITIVE REGISTRY

class PrimitiveLibrary:
    """Central registry of all cognitive primitives"""

    def __init__(self):
        self.primitives: Dict[str, Primitive] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all default primitives"""
        # Perceptual
        self.register(DetectObjectsPrimitive())
        self.register(CompareGridsPrimitive())
        self.register(FindPatternPrimitive())
        self.register(MatchObjectsPrimitive())

        # Manipulation
        self.register(RecolorPrimitive())
        self.register(RotateGridPrimitive(90))
        self.register(RotateGridPrimitive(180))
        self.register(RotateGridPrimitive(270))
        # ... more primitives

    def register(self, primitive: Primitive):
        """Add a primitive to the library"""
        self.primitives[primitive.name] = primitive

    def get(self, name: str) -> Primitive:
        """Retrieve a primitive by name"""
        return self.primitives.get(name)

    def get_all_by_category(self, category: str) -> List[Primitive]:
        """Get all primitives of a given category"""
        return [p for p in self.primitives.values() if p.category == category]

    def get_features_for_pam(self) -> Dict[str, Dict[str, float]]:
        """Get feature vectors for all primitives for PAM seeding"""
        return {name: prim.get_features() for name, prim in self.primitives.items()}
```

**Dependencies**: ObjectExtractor, GridObject

**Tests**:
- Test each primitive individually
- Validate feature extraction
- Test primitive composition

---

### 4. Demonstration Analyzer

**File**: `src/lida/arc/demonstration_analyzer.py`

**Purpose**: Extract transformation patterns from demonstration pairs.

**Key Classes**:
- `TransformationPattern`: Represents a learned transformation
- `DemonstrationAnalyzer`: Extracts patterns from demo pairs

**Implementation Details**:
```python
@dataclass
class TransformationPattern:
    """Represents a learned transformation from a demonstration pair"""
    pattern_id: str
    input_features: Dict[str, float]
    output_features: Dict[str, float]
    transformation_type: str  # 'pixel_map', 'object_map', 'grid_op'

    # For pixel-level transformations
    pixel_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None
    color_mapping: Optional[Dict[int, int]] = None

    # For object-level transformations
    object_correspondence: Optional[List[Tuple[int, int]]] = None
    object_transformations: Optional[List[Dict[str, Any]]] = None

    # For grid-level operations
    grid_operation: Optional[str] = None
    operation_params: Optional[Dict[str, Any]] = None

    # Metadata
    confidence: float = 0.5
    supporting_demos: List[int] = field(default_factory=list)

    def to_pam_features(self) -> Dict[str, float]:
        """Convert to PAM feature vector"""
        features = {}

        # Add transformation type
        features[f'transform_{self.transformation_type}'] = 1.0

        # Add detected operations
        if self.grid_operation:
            features[f'op_{self.grid_operation}'] = 1.0

        # Add feature changes
        for key in self.input_features:
            if key in self.output_features:
                diff = abs(self.output_features[key] - self.input_features[key])
                if diff > 0.1:
                    features[f'change_{key}'] = diff

        return features

class DemonstrationAnalyzer:
    """Analyze demonstration pairs to extract transformation patterns"""

    def __init__(self, primitive_library: PrimitiveLibrary):
        self.primitives = primitive_library
        self.object_extractor = ObjectExtractor()
        self.grid_analyzer = GridAnalyzer()

    def analyze_pair(self, grid_pair: GridPair) -> TransformationPattern:
        """Extract transformation pattern from a single demo pair"""
        input_grid = grid_pair.input
        output_grid = grid_pair.output

        # Extract features
        input_features = self.grid_analyzer.compute_grid_features(input_grid)
        output_features = self.grid_analyzer.compute_grid_features(output_grid)

        # Try different analysis strategies
        pattern = None

        # Strategy 1: Grid-level operations (rotation, reflection)
        pattern = self._try_grid_operations(input_grid, output_grid)
        if pattern and pattern.confidence > 0.9:
            pattern.input_features = input_features
            pattern.output_features = output_features
            return pattern

        # Strategy 2: Object-level transformations
        pattern = self._try_object_transformations(input_grid, output_grid)
        if pattern and pattern.confidence > 0.7:
            pattern.input_features = input_features
            pattern.output_features = output_features
            return pattern

        # Strategy 3: Pixel-level mapping
        pattern = self._try_pixel_mapping(input_grid, output_grid)
        pattern.input_features = input_features
        pattern.output_features = output_features
        return pattern

    def _try_grid_operations(self, input_grid, output_grid) -> Optional[TransformationPattern]:
        """Try to explain transformation as a grid-level operation"""
        # Try rotation
        for angle in [90, 180, 270]:
            rotated = self.primitives.get(f'rotate_{angle}').execute(input_grid)
            if self._grids_equal(rotated, output_grid):
                return TransformationPattern(
                    pattern_id=f'grid_rotate_{angle}',
                    input_features={},
                    output_features={},
                    transformation_type='grid_op',
                    grid_operation=f'rotate_{angle}',
                    confidence=1.0
                )

        # Try reflection
        # TODO: Implement reflection primitives

        # Try color remapping
        color_map = self._infer_color_mapping(input_grid, output_grid)
        if color_map:
            recolored = self.primitives.get('recolor').execute(input_grid, color_map)
            if self._grids_equal(recolored, output_grid):
                return TransformationPattern(
                    pattern_id='grid_recolor',
                    input_features={},
                    output_features={},
                    transformation_type='grid_op',
                    grid_operation='recolor',
                    operation_params={'color_map': color_map},
                    color_mapping=color_map,
                    confidence=1.0
                )

        return None

    def _try_object_transformations(self, input_grid, output_grid) -> Optional[TransformationPattern]:
        """Try to explain transformation as object-level operations"""
        input_objects = self.object_extractor.extract_objects(input_grid)
        output_objects = self.object_extractor.extract_objects(output_grid)

        # Find object correspondence
        match_result = self.primitives.get('match_objects').execute(input_objects, output_objects)

        if not match_result['matches']:
            return None

        # Analyze transformations for each matched pair
        object_transforms = []
        for in_idx, out_idx, score in match_result['matches']:
            in_obj = input_objects[in_idx]
            out_obj = output_objects[out_idx]

            transform = self._analyze_object_pair(in_obj, out_obj)
            object_transforms.append(transform)

        # Check if all objects underwent the same transformation
        if self._all_transforms_similar(object_transforms):
            return TransformationPattern(
                pattern_id='object_transform_uniform',
                input_features={},
                output_features={},
                transformation_type='object_map',
                object_correspondence=[(m[0], m[1]) for m in match_result['matches']],
                object_transformations=object_transforms,
                confidence=0.8
            )
        else:
            return TransformationPattern(
                pattern_id='object_transform_varied',
                input_features={},
                output_features={},
                transformation_type='object_map',
                object_correspondence=[(m[0], m[1]) for m in match_result['matches']],
                object_transformations=object_transforms,
                confidence=0.6
            )

    def _analyze_object_pair(self, in_obj: GridObject, out_obj: GridObject) -> Dict[str, Any]:
        """Analyze transformation applied to a single object"""
        transform = {
            'color_changed': in_obj.color != out_obj.color,
            'position_changed': in_obj.centroid != out_obj.centroid,
            'size_changed': in_obj.size != out_obj.size,
            'shape_changed': in_obj.get_shape_signature() != out_obj.get_shape_signature()
        }

        if transform['color_changed']:
            transform['color_delta'] = (in_obj.color, out_obj.color)

        if transform['position_changed']:
            dx = out_obj.centroid[0] - in_obj.centroid[0]
            dy = out_obj.centroid[1] - in_obj.centroid[1]
            transform['position_delta'] = (dx, dy)

        return transform

    def _try_pixel_mapping(self, input_grid, output_grid) -> TransformationPattern:
        """Extract pixel-level mapping (fallback)"""
        # Use compare_grids primitive
        comparison = self.primitives.get('compare_grids').execute(input_grid, output_grid)

        # Create pixel mapping
        pixel_map = {}
        color_map_counter = defaultdict(lambda: defaultdict(int))

        for diff in comparison.get('differences', []):
            pos = diff['position']
            from_color = diff['from_color']
            to_color = diff['to_color']
            pixel_map[pos] = (pos, to_color)  # Position stays same, color changes
            color_map_counter[from_color][to_color] += 1

        # Infer dominant color mapping
        color_map = {}
        for from_color, to_colors in color_map_counter.items():
            to_color = max(to_colors.items(), key=lambda x: x[1])[0]
            color_map[from_color] = to_color

        return TransformationPattern(
            pattern_id='pixel_map',
            input_features={},
            output_features={},
            transformation_type='pixel_map',
            pixel_mapping=pixel_map,
            color_mapping=color_map if color_map else None,
            confidence=0.4
        )

    def _grids_equal(self, grid1, grid2) -> bool:
        """Check if two grids are identical"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False
        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                if grid1[r][c] != grid2[r][c]:
                    return False
        return True

    def _infer_color_mapping(self, input_grid, output_grid) -> Optional[Dict[int, int]]:
        """Infer color mapping if grids have same structure"""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None

        color_map = {}
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                in_color = input_grid[r][c]
                out_color = output_grid[r][c]

                if in_color in color_map and color_map[in_color] != out_color:
                    return None  # Inconsistent mapping

                color_map[in_color] = out_color

        return color_map

    def _all_transforms_similar(self, transforms: List[Dict[str, Any]]) -> bool:
        """Check if all object transformations are the same"""
        if not transforms:
            return False

        first = transforms[0]
        for t in transforms[1:]:
            if t != first:
                return False

        return True
```

**Dependencies**: PrimitiveLibrary, ObjectExtractor, GridAnalyzer

**Tests**:
- Analyze rotation demos
- Analyze color swap demos
- Analyze object movement demos
- Test pattern extraction accuracy

---

### 5. PAM Extensions for ARC

**File**: `src/lida/arc/arc_pam.py`

**Purpose**: Extend PAM with transformation-specific nodes and seeding.

**Implementation Details**:
```python
class ARCPAMInitializer:
    """Initialize PAM with ARC-specific concept nodes"""

    def __init__(self, pam: PerceptualAssociativeMemory, primitive_library: PrimitiveLibrary):
        self.pam = pam
        self.primitives = primitive_library

    def seed_primitives(self):
        """Add primitive operations as PAM nodes"""
        for name, primitive in self.primitives.primitives.items():
            # Add node
            self.pam.add_node(
                node_id=f'prim_{name}',
                kind='primitive',
                label=name,
                base_strength=0.3
            )

            # Add feature nodes
            for feature, value in primitive.get_features().items():
                feature_node = f'feat_{feature}'
                if not self.pam.has_node(feature_node):
                    self.pam.add_node(
                        node_id=feature_node,
                        kind='feature',
                        label=feature,
                        base_strength=0.2
                    )

                # Link primitive to its features
                self.pam.add_edge(f'prim_{name}', feature_node, 'has_feature', weight=value)

    def seed_transformation_pattern(self, pattern: TransformationPattern):
        """Add a learned transformation pattern to PAM"""
        pattern_node = f'pattern_{pattern.pattern_id}'

        # Add pattern node
        self.pam.add_node(
            node_id=pattern_node,
            kind='transformation',
            label=pattern.pattern_id,
            base_strength=pattern.confidence
        )

        # Link to features
        for feature, value in pattern.to_pam_features().items():
            feature_node = f'feat_{feature}'
            if not self.pam.has_node(feature_node):
                self.pam.add_node(
                    node_id=feature_node,
                    kind='feature',
                    label=feature,
                    base_strength=0.1
                )

            self.pam.add_edge(pattern_node, feature_node, 'exhibits', weight=value)

        # If grid operation, link to corresponding primitive
        if pattern.grid_operation:
            prim_node = f'prim_{pattern.grid_operation}'
            if self.pam.has_node(prim_node):
                self.pam.add_edge(pattern_node, prim_node, 'uses', weight=0.9)

    def activate_from_demonstration(self, pattern: TransformationPattern, strength: float = 1.0):
        """Activate PAM based on observed demonstration"""
        # Activate the pattern node
        pattern_node = f'pattern_{pattern.pattern_id}'
        if self.pam.has_node(pattern_node):
            self.pam.activate(pattern_node, strength)

        # Activate feature nodes
        for feature, value in pattern.to_pam_features().items():
            feature_node = f'feat_{feature}'
            if self.pam.has_node(feature_node):
                self.pam.activate(feature_node, strength * value)
```

**Dependencies**: PAM, PrimitiveLibrary, TransformationPattern

**Tests**:
- Test PAM seeding with primitives
- Test pattern node creation
- Test activation from demonstrations
- Verify spreading activation

---

### 6. Transformation Hypothesis

**File**: `src/lida/arc/hypothesis.py`

**Purpose**: Represent and execute transformation hypotheses.

**Implementation Details**:
```python
@dataclass
class TransformationHypothesis:
    """A candidate transformation program"""
    hypothesis_id: str
    patterns: List[TransformationPattern]  # Sequence of patterns to apply
    confidence: float
    supporting_demos: List[int]
    validation_score: float = 0.0

    def apply(self, input_grid: List[List[int]],
              primitive_library: PrimitiveLibrary) -> List[List[int]]:
        """Execute hypothesis on input grid"""
        result = input_grid

        for pattern in self.patterns:
            result = self._apply_pattern(result, pattern, primitive_library)

        return result

    def _apply_pattern(self, grid, pattern, primitives):
        """Apply a single transformation pattern"""
        if pattern.transformation_type == 'grid_op':
            # Apply grid operation
            if pattern.grid_operation:
                prim = primitives.get(pattern.grid_operation)
                if prim:
                    if pattern.operation_params:
                        return prim.execute(grid, **pattern.operation_params)
                    else:
                        return prim.execute(grid)

        elif pattern.transformation_type == 'object_map':
            # Object-level transformation
            # TODO: Implement object-level application
            pass

        elif pattern.transformation_type == 'pixel_map':
            # Pixel-level transformation
            if pattern.color_mapping:
                prim = primitives.get('recolor')
                return prim.execute(grid, pattern.color_mapping)

        return grid

    def to_coalition_features(self) -> FrozenSet[Tuple[str, float]]:
        """Convert to features for coalition creation"""
        features = []

        for pattern in self.patterns:
            for feat_name, feat_val in pattern.to_pam_features().items():
                features.append((feat_name, feat_val))

        # Add hypothesis-level features
        features.append(('n_operations', len(self.patterns)))
        features.append(('confidence', self.confidence))

        return frozenset(features)

class HypothesisGenerator:
    """Generate transformation hypotheses from demonstrations"""

    def __init__(self, primitive_library: PrimitiveLibrary,
                 demonstration_analyzer: DemonstrationAnalyzer):
        self.primitives = primitive_library
        self.analyzer = demonstration_analyzer

    def generate_from_demonstrations(self,
                                     demonstrations: List[GridPair],
                                     pam: PerceptualAssociativeMemory) -> List[TransformationHypothesis]:
        """Generate hypotheses from demonstration pairs"""
        hypotheses = []

        # Strategy 1: Single pattern that explains all demos
        for demo_idx, demo in enumerate(demonstrations):
            pattern = self.analyzer.analyze_pair(demo)

            # Check if this pattern works for all demos
            if self._validates_on_all_demos(pattern, demonstrations):
                hyp = TransformationHypothesis(
                    hypothesis_id=f'single_pattern_{pattern.pattern_id}',
                    patterns=[pattern],
                    confidence=pattern.confidence,
                    supporting_demos=list(range(len(demonstrations)))
                )
                hypotheses.append(hyp)

        # Strategy 2: Use PAM activation to suggest composite operations
        pam.spread_activation(iterations=5, decay=0.1)
        active_primitives = pam.get_top_k_active(node_type='primitive', k=5)

        # Try combinations of top active primitives
        for prim_node in active_primitives:
            prim_name = prim_node.label
            # Create hypothesis from this primitive
            # TODO: Implement primitive-based hypothesis generation

        # Strategy 3: Ensemble of different patterns for different demos
        # (For complex tasks where single pattern doesn't fit all)

        return hypotheses

    def _validates_on_all_demos(self, pattern: TransformationPattern,
                                 demos: List[GridPair]) -> bool:
        """Check if a pattern correctly transforms all demonstration inputs"""
        for demo in demos:
            # Create temporary hypothesis
            hyp = TransformationHypothesis(
                hypothesis_id='temp',
                patterns=[pattern],
                confidence=1.0,
                supporting_demos=[]
            )

            result = hyp.apply(demo.input, self.primitives)

            # Check if result matches expected output
            if not self._grids_equal(result, demo.output):
                return False

        return True

    def _grids_equal(self, grid1, grid2):
        """Check grid equality"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False
        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                if grid1[r][c] != grid2[r][c]:
                    return False
        return True
```

**Dependencies**: TransformationPattern, PrimitiveLibrary, PAM

**Tests**:
- Generate hypotheses from simple demos
- Validate hypothesis execution
- Test hypothesis validation
- Verify coalition feature extraction

---

## Implementation Order

### Phase 0: Setup (Day 1)
1. Create directory structure
2. Set up testing infrastructure
3. Add dependencies to pyproject.toml

### Phase 1: Foundation (Days 2-4)
1. ARCEnvironment + GridPair + ARCTask
2. GridObject + ObjectExtractor
3. GridAnalyzer
4. Basic tests

### Phase 2: Primitives (Days 5-7)
1. Primitive base class
2. Perceptual primitives (5)
3. Manipulation primitives (8)
4. PrimitiveLibrary
5. Primitive tests

### Phase 3: Pattern Learning (Days 8-10)
1. TransformationPattern
2. DemonstrationAnalyzer
3. Pattern extraction tests

### Phase 4: Integration (Days 11-13)
1. ARCPAMInitializer
2. TransformationHypothesis
3. HypothesisGenerator
4. Integration tests

### Phase 5: Cognitive Cycle (Days 14-16)
1. ARC-specific codelets
2. Modified cognitive cycle
3. End-to-end test on simple task

### Phase 6: Learning & Refinement (Days 17-20)
1. TD learning for hypothesis utilities
2. Category induction for patterns
3. Iterative refinement
4. Multiple task tests

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock dependencies
- Target: 80% code coverage

### Integration Tests
- Test component interactions
- Use simple synthetic ARC tasks
- Validate data flow through system

### System Tests
- End-to-end on real ARC tasks
- Measure accuracy, convergence, timing
- Compare with baseline approaches

### Test ARC Tasks (for development)
- Rotation (90, 180, 270)
- Reflection (H, V)
- Color swap
- Simple object movement
- Tiling patterns

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.24.0"
networkx = "^3.0"
pytest = "^7.4.0"
```

## Success Metrics

### Phase 1-2
- Can load and parse ARC tasks
- Can extract objects from grids
- All primitives work correctly

### Phase 3-4
- Can analyze demonstration pairs
- Can generate 5-10 hypotheses per task
- At least 1 correct hypothesis for simple tasks

### Phase 5-6
- Can solve rotation tasks (100%)
- Can solve color swap tasks (100%)
- Can solve simple object movement (80%+)
- Overall accuracy on 10 dev tasks: 60%+

## Next Steps

After implementation plan approval:
1. Begin Phase 0 (setup)
2. Implement Phase 1 (foundation)
3. Iterate with tests and validation
