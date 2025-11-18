"""
Demonstration analysis: extract transformation patterns from input-output pairs.

This module implements the core learning mechanism - analyzing demonstration
pairs to discover transformation patterns without a pre-defined DSL.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import copy

from .environment import GridPair
from .perception import GridObject, ObjectExtractor, GridAnalyzer
from .primitives import PrimitiveLibrary, Primitive
from .sequence_detection import SequenceDetector


@dataclass
class TransformationPattern:
    """Represents a learned transformation pattern from demonstrations.

    This is a task-specific transformation extracted from demonstration pairs.
    It can represent different levels of abstraction:
    - Pixel-level: Direct pixel position/color mappings
    - Object-level: Object correspondence and transformations
    - Grid-level: Whole-grid operations (rotation, reflection, etc.)
    """

    pattern_id: str
    transformation_type: str  # 'pixel_map', 'object_map', 'grid_op'

    # Input/output features for PAM activation
    input_features: Dict[str, float] = field(default_factory=dict)
    output_features: Dict[str, float] = field(default_factory=dict)

    # For pixel-level transformations
    pixel_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int, int]]] = None  # (r,c) -> (new_r, new_c, new_color)
    color_mapping: Optional[Dict[int, int]] = None

    # For object-level transformations
    object_correspondence: Optional[List[Tuple[int, int, float]]] = None  # List of (in_idx, out_idx, score)
    object_transformations: Optional[List[Dict[str, Any]]] = None

    # For grid-level operations
    grid_operations: List[str] = field(default_factory=list)  # Sequence of operation names
    operation_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    confidence: float = 0.5
    supporting_demos: List[int] = field(default_factory=list)
    explanation: str = ""

    def to_pam_features(self) -> Dict[str, float]:
        """Convert to PAM feature vector for spreading activation."""
        features = {}

        # Add transformation type
        features[f'transform_{self.transformation_type}'] = 1.0

        # Add detected operations
        for op_name in self.grid_operations:
            features[f'op_{op_name}'] = 1.0

        # Add feature changes between input and output
        for key in self.input_features:
            if key in self.output_features:
                diff = abs(self.output_features[key] - self.input_features[key])
                if diff > 0.1:
                    features[f'change_{key}'] = min(diff, 1.0)

        # Add special markers
        if self.color_mapping:
            features['color_transformation'] = 1.0
        if self.object_correspondence:
            features['object_based'] = 1.0

        return features

    def __repr__(self) -> str:
        if self.grid_operations:
            ops = ' → '.join(self.grid_operations)
            return f"TransformationPattern({ops}, conf={self.confidence:.2f})"
        return f"TransformationPattern({self.transformation_type}, conf={self.confidence:.2f})"


class DemonstrationAnalyzer:
    """Analyze demonstration pairs to extract transformation patterns.

    This is the core learning mechanism that discovers transformations from
    examples without pre-defined rules. It tries multiple analysis strategies
    in order of abstraction level (grid → object → pixel).
    """

    def __init__(self, primitive_library: PrimitiveLibrary):
        """
        Args:
            primitive_library: Library of cognitive primitives to test
        """
        self.primitives = primitive_library
        self.object_extractor = ObjectExtractor()
        self.grid_analyzer = GridAnalyzer()
        # Sequence detector for composite operations (depth-3 with beam search)
        # Larger beam width ensures we find more sequence variants
        self.sequence_detector = SequenceDetector(primitive_library, max_depth=3, beam_width=10)

    def analyze_pair(self, grid_pair: GridPair, demo_index: int = 0) -> List[TransformationPattern]:
        """Extract all possible transformation patterns from a single demo pair.

        Args:
            grid_pair: Input-output demonstration pair
            demo_index: Index of this demo (for tracking)

        Returns:
            List of TransformationPattern instances, sorted by confidence
        """
        input_grid = grid_pair.input
        output_grid = grid_pair.output

        # Extract features for PAM activation
        input_features = self.grid_analyzer.compute_grid_features(input_grid)
        output_features = self.grid_analyzer.compute_grid_features(output_grid)

        patterns = []

        # Strategy 1: Grid-level operations (highest confidence if matches)
        grid_patterns = self._try_grid_operations(input_grid, output_grid, demo_index)
        for pattern in grid_patterns:
            pattern.input_features = input_features
            pattern.output_features = output_features
            patterns.append(pattern)

        # Strategy 2: Object-level transformations
        object_patterns = self._try_object_transformations(input_grid, output_grid, demo_index)
        for pattern in object_patterns:
            pattern.input_features = input_features
            pattern.output_features = output_features
            patterns.append(pattern)

        # Strategy 3: Pixel-level mapping (fallback)
        pixel_pattern = self._try_pixel_mapping(input_grid, output_grid, demo_index)
        pixel_pattern.input_features = input_features
        pixel_pattern.output_features = output_features
        patterns.append(pixel_pattern)

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def analyze_multiple_pairs(self, demo_pairs: List[GridPair]) -> List[TransformationPattern]:
        """Analyze multiple demonstration pairs and find common patterns.

        Args:
            demo_pairs: List of demonstration pairs

        Returns:
            List of patterns that work across multiple (ideally all) demos
        """
        if not demo_pairs:
            return []

        # Analyze each pair individually
        all_patterns_by_demo = []
        for i, demo in enumerate(demo_pairs):
            patterns = self.analyze_pair(demo, demo_index=i)
            all_patterns_by_demo.append(patterns)

        # Find patterns that work across all demos
        universal_patterns = []

        # Check if any pattern from first demo works for all
        for pattern in all_patterns_by_demo[0]:
            if self._pattern_works_for_all(pattern, demo_pairs):
                pattern.supporting_demos = list(range(len(demo_pairs)))
                pattern.confidence = min(1.0, pattern.confidence + 0.2)  # Boost for universality
                universal_patterns.append(pattern)

        # If no universal pattern, return patterns that work for most demos
        if not universal_patterns:
            # Find patterns with highest support
            pattern_support = defaultdict(int)
            pattern_instances = {}

            for demo_idx, patterns in enumerate(all_patterns_by_demo):
                for pattern in patterns[:3]:  # Top 3 per demo
                    key = self._pattern_signature(pattern)
                    pattern_support[key] += 1
                    if key not in pattern_instances:
                        pattern_instances[key] = pattern

            # Return patterns with support from multiple demos
            for key, support in pattern_support.items():
                if support >= max(2, len(demo_pairs) // 2):  # At least half support
                    pattern = pattern_instances[key]
                    pattern.supporting_demos = [i for i in range(support)]
                    pattern.confidence = support / len(demo_pairs)
                    universal_patterns.append(pattern)

        universal_patterns.sort(key=lambda p: p.confidence, reverse=True)
        return universal_patterns

    def _try_grid_operations(self, input_grid: List[List[int]],
                            output_grid: List[List[int]],
                            demo_index: int) -> List[TransformationPattern]:
        """Try to explain transformation as grid-level operations."""
        patterns = []

        # Try single operations
        for prim_name in self.primitives.get_all_names():
            prim = self.primitives.get(prim_name)

            # Try rotation
            if 'rotate' in prim_name:
                try:
                    result = prim.execute(input_grid)
                    if self._grids_equal(result, output_grid):
                        patterns.append(TransformationPattern(
                            pattern_id=f'grid_{prim_name}_demo{demo_index}',
                            transformation_type='grid_op',
                            grid_operations=[prim_name],
                            confidence=1.0,
                            supporting_demos=[demo_index],
                            explanation=f"Apply {prim_name}"
                        ))
                except:
                    pass

            # Try reflection
            if 'reflect' in prim_name:
                try:
                    result = prim.execute(input_grid)
                    if self._grids_equal(result, output_grid):
                        patterns.append(TransformationPattern(
                            pattern_id=f'grid_{prim_name}_demo{demo_index}',
                            transformation_type='grid_op',
                            grid_operations=[prim_name],
                            confidence=1.0,
                            supporting_demos=[demo_index],
                            explanation=f"Apply {prim_name}"
                        ))
                except:
                    pass

        # Try color remapping
        color_map = self._infer_color_mapping(input_grid, output_grid)
        if color_map:
            try:
                recolored = self.primitives.get('recolor').execute(input_grid, color_map)
                if self._grids_equal(recolored, output_grid):
                    patterns.append(TransformationPattern(
                        pattern_id=f'grid_recolor_demo{demo_index}',
                        transformation_type='grid_op',
                        grid_operations=['recolor'],
                        operation_params={'color_map': color_map},
                        color_mapping=color_map,
                        confidence=1.0,
                        supporting_demos=[demo_index],
                        explanation=f"Recolor: {color_map}"
                    ))
            except:
                pass

        # Always try composite operations using sequence detector
        # Generate multiple pattern hypotheses (both simple and complex)
        try:
            # Infer color mapping for sequences that include recolor
            color_map = self._infer_color_mapping(input_grid, output_grid)

            # Find ALL exact-match sequences at all depths
            all_sequences = self.sequence_detector.find_all_sequences(input_grid, output_grid, color_map)

            # Debug: print what was found
            if False:  # Set to True for debugging
                print(f"[DemoAnalyzer] find_all_sequences returned {len(all_sequences)} sequences")
                for seq in all_sequences[:10]:
                    print(f"[DemoAnalyzer]   - {seq}")
                print(f"[DemoAnalyzer] Before loop: {len(patterns)} existing patterns")
                print(f"[DemoAnalyzer] color_map={color_map}")

            # Add sequences with 2+ operations as alternative hypotheses
            # (single operations are already tried above)
            for seq_idx, sequence in enumerate(all_sequences):
                if len(sequence) >= 2 and sequence not in [p.grid_operations for p in patterns]:
                    # Longer sequences get slightly lower confidence (Occam's razor)
                    # But they might generalize better than simpler explanations
                    confidence = 0.95 - (0.05 * (len(sequence) - 2))  # 0.95, 0.90, 0.85...
                    confidence = max(confidence, 0.70)  # Floor at 0.70

                    patterns.append(TransformationPattern(
                        pattern_id=f'grid_sequence_{len(sequence)}_ops_demo{demo_index}_{seq_idx}',
                        transformation_type='grid_op',
                        grid_operations=sequence,
                        operation_params={'color_map': color_map} if color_map and 'recolor' in sequence else {},
                        color_mapping=color_map if 'recolor' in sequence else None,
                        confidence=confidence,
                        supporting_demos=[demo_index],
                        explanation=f"Apply sequence: {' → '.join(sequence)}"
                    ))

                    # Limit to top 3 sequence hypotheses to avoid explosion
                    if seq_idx >= 2:
                        break

            # Debug: show what was added
            if False:  # Set to True for debugging
                print(f"[DemoAnalyzer] After loop: {len(patterns)} total patterns")
                for p in patterns:
                    print(f"[DemoAnalyzer]   - {p.pattern_id}: {p.grid_operations}")

        except Exception as e:
            # Sequence detection failed, log the error
            if False:  # Set to True to debug
                print(f"[DemoAnalyzer] Sequence detection failed: {e}")
                import traceback
                traceback.print_exc()
            pass

        return patterns

    def _try_object_transformations(self, input_grid: List[List[int]],
                                    output_grid: List[List[int]],
                                    demo_index: int) -> List[TransformationPattern]:
        """Try to explain transformation as object-level operations."""
        patterns = []

        # Extract objects
        input_objects = self.object_extractor.extract_objects(input_grid)
        output_objects = self.object_extractor.extract_objects(output_grid)

        if not input_objects or not output_objects:
            return patterns

        # Find object correspondence
        match_prim = self.primitives.get('match_objects')
        match_result = match_prim.execute(input_objects, output_objects)

        if not match_result['matches']:
            return patterns

        # Analyze transformations for each matched pair
        object_transforms = []
        for in_idx, out_idx, score in match_result['matches']:
            in_obj = input_objects[in_idx]
            out_obj = output_objects[out_idx]

            transform = self._analyze_object_pair(in_obj, out_obj)
            object_transforms.append(transform)

        # Check if all objects underwent the same transformation
        if self._all_transforms_similar(object_transforms):
            patterns.append(TransformationPattern(
                pattern_id=f'object_transform_uniform_demo{demo_index}',
                transformation_type='object_map',
                object_correspondence=match_result['matches'],
                object_transformations=object_transforms,
                confidence=0.8,
                supporting_demos=[demo_index],
                explanation=f"All {len(object_transforms)} objects transformed uniformly"
            ))
        else:
            patterns.append(TransformationPattern(
                pattern_id=f'object_transform_varied_demo{demo_index}',
                transformation_type='object_map',
                object_correspondence=match_result['matches'],
                object_transformations=object_transforms,
                confidence=0.6,
                supporting_demos=[demo_index],
                explanation=f"{len(object_transforms)} objects with varied transformations"
            ))

        return patterns

    def _try_pixel_mapping(self, input_grid: List[List[int]],
                          output_grid: List[List[int]],
                          demo_index: int) -> TransformationPattern:
        """Extract pixel-level mapping (lowest-level fallback)."""

        # Use compare_grids primitive
        compare_prim = self.primitives.get('compare_grids')
        comparison = compare_prim.execute(input_grid, output_grid)

        # Handle size changes
        if comparison.get('size_changed'):
            return TransformationPattern(
                pattern_id=f'pixel_map_size_change_demo{demo_index}',
                transformation_type='pixel_map',
                confidence=0.3,
                supporting_demos=[demo_index],
                explanation=f"Size changed from {comparison.get('shape1')} to {comparison.get('shape2')}"
            )

        # Create pixel mapping
        pixel_map = {}
        color_map_counter = defaultdict(lambda: defaultdict(int))

        for diff in comparison.get('differences', []):
            pos = diff['position']
            from_color = diff['from_color']
            to_color = diff['to_color']
            pixel_map[pos] = (pos[0], pos[1], to_color)  # Position stays same, color changes
            color_map_counter[from_color][to_color] += 1

        # Infer dominant color mapping
        color_map = {}
        for from_color, to_colors in color_map_counter.items():
            if to_colors:
                to_color = max(to_colors.items(), key=lambda x: x[1])[0]
                color_map[from_color] = to_color

        return TransformationPattern(
            pattern_id=f'pixel_map_demo{demo_index}',
            transformation_type='pixel_map',
            pixel_mapping=pixel_map,
            color_mapping=color_map if color_map else None,
            confidence=0.4,
            supporting_demos=[demo_index],
            explanation=f"{len(pixel_map)} pixels changed" + (f", color map: {color_map}" if color_map else "")
        )

    def _analyze_object_pair(self, in_obj: GridObject, out_obj: GridObject) -> Dict[str, Any]:
        """Analyze transformation applied to a single object."""
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

        if transform['size_changed']:
            transform['size_delta'] = out_obj.size - in_obj.size

        return transform

    def _all_transforms_similar(self, transforms: List[Dict[str, Any]]) -> bool:
        """Check if all object transformations are the same."""
        if not transforms:
            return False

        first = transforms[0]
        for t in transforms[1:]:
            # Check if key sets match
            if set(first.keys()) != set(t.keys()):
                return False

            # Check if change patterns match
            if (first.get('color_changed') != t.get('color_changed') or
                first.get('position_changed') != t.get('position_changed') or
                first.get('shape_changed') != t.get('shape_changed')):
                return False

        return True

    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are identical."""
        if not grid1 or not grid2:
            return False
        if len(grid1) != len(grid2):
            return False
        if len(grid1[0]) != len(grid2[0]):
            return False

        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                if grid1[r][c] != grid2[r][c]:
                    return False
        return True

    def _infer_color_mapping(self, input_grid: List[List[int]],
                            output_grid: List[List[int]]) -> Optional[Dict[int, int]]:
        """Infer color mapping if grids have same structure."""
        if not input_grid or not output_grid:
            return None
        if len(input_grid) != len(output_grid):
            return None
        if len(input_grid[0]) != len(output_grid[0]):
            return None

        color_map = {}
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                in_color = input_grid[r][c]
                out_color = output_grid[r][c]

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return None  # Inconsistent mapping
                else:
                    color_map[in_color] = out_color

        # Only return if it's actually a transformation (not identity)
        if any(k != v for k, v in color_map.items()):
            return color_map

        return None

    def _pattern_works_for_all(self, pattern: TransformationPattern,
                               demo_pairs: List[GridPair]) -> bool:
        """Check if a pattern correctly transforms all demonstration inputs."""
        for demo in demo_pairs:
            if not self._apply_pattern(pattern, demo.input, demo.output):
                return False
        return True

    def _apply_pattern(self, pattern: TransformationPattern,
                      input_grid: List[List[int]],
                      expected_output: List[List[int]]) -> bool:
        """Test if a pattern produces expected output from input."""
        debug = False  # Set to True for debugging
        try:
            result = input_grid

            # Apply grid operations
            for op_name in pattern.grid_operations:
                prim = self.primitives.get(op_name)
                if not prim:
                    if debug:
                        print(f"[_apply_pattern] Primitive '{op_name}' not found")
                    return False

                # Handle operations with parameters
                if op_name == 'recolor':
                    # CRITICAL: For sequences, dynamically infer color mapping from current state
                    # This handles cases like ['rotate_90', 'recolor'] where the mapping changes after rotation
                    if len(pattern.grid_operations) > 1:
                        # Multi-operation sequence: infer mapping from current result to expected output
                        if debug:
                            print(f"[_apply_pattern]   Before recolor: result={result}")
                        color_mapping = self._infer_color_mapping(result, expected_output)
                        if debug:
                            print(f"[_apply_pattern]   Sequence {pattern.grid_operations}: inferred mapping {color_mapping}")
                        if not color_mapping:
                            if debug:
                                print(f"[_apply_pattern]   Failed to infer color mapping")
                            return False
                        result = prim.execute(result, color_mapping)
                        if debug:
                            print(f"[_apply_pattern]   After recolor: result={result}")
                    elif pattern.color_mapping:
                        # Single recolor operation: use stored mapping
                        result = prim.execute(result, pattern.color_mapping)
                    else:
                        if debug:
                            print(f"[_apply_pattern] No color mapping available")
                        return False
                else:
                    if debug:
                        print(f"[_apply_pattern]   Applying {op_name}: before={result}")
                    result = prim.execute(result)
                    if debug:
                        print(f"[_apply_pattern]   After {op_name}: result={result}")

            matches = self._grids_equal(result, expected_output)
            if debug and not matches:
                print(f"[_apply_pattern] Result {result} != expected {expected_output}")
            return matches
        except Exception as e:
            if debug:
                print(f"[_apply_pattern] Exception: {e}")
            return False

    def _pattern_signature(self, pattern: TransformationPattern) -> str:
        """Create a signature for pattern matching across demos."""
        if pattern.grid_operations:
            return '_'.join(pattern.grid_operations)
        return pattern.transformation_type
