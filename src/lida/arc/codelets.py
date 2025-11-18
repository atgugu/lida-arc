"""
ARC-specific codelets for pattern hypothesis generation, validation, and application.

These codelets integrate with LIDA's cognitive cycle to create a full learning loop:
- Understanding phase: Generate pattern hypotheses from demonstrations via PAM
- Attention phase: Compete hypotheses for workspace access
- Action phase: Apply winning pattern to test input
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from ..core.codelet import Codelet
from ..core.events import Coalition
from ..memory.workspace import SituationalModel
from .demonstration import TransformationPattern, DemonstrationAnalyzer
from .pam_integration import ARCPAMIntegration
from .primitives import PrimitiveLibrary
from .environment import GridPair


@dataclass
class PatternHypothesis:
    """A pattern hypothesis competing for workspace access."""

    hypothesis_id: str
    pattern: TransformationPattern
    salience: float  # Computed from PAM activation + confidence
    support_count: int  # Number of demos this pattern explains
    validation_accuracy: float = 0.0  # Accuracy on demonstrations

    def to_coalition(self) -> Coalition:
        """Convert hypothesis to coalition for workspace competition."""
        features = []

        # Pattern features
        features.append(('confidence', self.pattern.confidence))
        features.append(('support_count', float(self.support_count)))
        features.append(('validation_accuracy', self.validation_accuracy))
        features.append(('salience', self.salience))

        # Operation features
        for op in self.pattern.grid_operations:
            features.append((f'op_{op}', 1.0))

        # Transformation type
        features.append((f'type_{self.pattern.transformation_type}', 1.0))

        summary = f"{self.pattern.pattern_id}: {' -> '.join(self.pattern.grid_operations)}"

        return Coalition(
            id=self.hypothesis_id,
            salience=self.salience,
            features=frozenset(features),
            summary=summary
        )


class ARCCodeletFactory:
    """Factory for creating ARC-specific codelets."""

    def __init__(
        self,
        workspace: SituationalModel,
        pam_integration: ARCPAMIntegration,
        primitive_library: PrimitiveLibrary,
        analyzer: DemonstrationAnalyzer,
        debug: bool = False
    ):
        """
        Args:
            workspace: Global workspace for storing ARC task state
            pam_integration: PAM integration for spreading activation
            primitive_library: Cognitive primitives
            analyzer: Demonstration analyzer
            debug: Enable debug logging
        """
        self.workspace = workspace
        self.pam = pam_integration
        self.primitives = primitive_library
        self.analyzer = analyzer
        self.debug = debug

        # Shared state across codelets
        self.demonstrations: List[GridPair] = []
        self.test_input: Optional[List[List[int]]] = None
        self.hypotheses: List[PatternHypothesis] = []
        self.coalitions: List[Coalition] = []
        self.winning_pattern: Optional[TransformationPattern] = None
        self.final_output: Optional[List[List[int]]] = None

    def _debug(self, msg: str):
        """Log debug message if debug enabled."""
        if self.debug:
            print(f"[Codelet] {msg}")

    @staticmethod
    def _infer_color_mapping(grid1, grid2):
        """Infer color mapping between two grids (same as demonstration.py)."""
        if not grid1 or not grid2:
            return None
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return None

        color_map = {}
        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                in_color = grid1[r][c]
                out_color = grid2[r][c]

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return None  # Inconsistent mapping
                else:
                    color_map[in_color] = out_color

        # Only return if it's actually a transformation
        if any(k != v for k, v in color_map.items()):
            return color_map

        return None

    def make_understanding_codelets(self) -> List[Codelet]:
        """Create codelets for understanding phase: analyze demos and generate hypotheses."""

        def _analyze_demonstrations():
            """Analyze demonstrations to extract patterns."""
            self._debug("  [arc_analyze_demos] Starting demonstration analysis")

            if not self.demonstrations:
                self._debug("    ⚠ No demonstrations to analyze")
                return

            self._debug(f"    Analyzing {len(self.demonstrations)} demonstrations")

            # Clear previous hypotheses
            self.hypotheses.clear()

            # Extract patterns from demonstrations
            try:
                patterns = self.analyzer.analyze_multiple_pairs(self.demonstrations)
                self._debug(f"    Found {len(patterns)} patterns")

                for i, pattern in enumerate(patterns[:5]):
                    self._debug(f"      Pattern {i}: {pattern.pattern_id}")
                    self._debug(f"        Type: {pattern.transformation_type}")
                    self._debug(f"        Operations: {pattern.grid_operations}")
                    self._debug(f"        Confidence: {pattern.confidence:.3f}")
                    self._debug(f"        Support: {len(pattern.supporting_demos)}/{len(self.demonstrations)} demos")
            except Exception as e:
                self._debug(f"    ✗ Pattern extraction failed: {e}")
                patterns = []

            # Learn patterns in PAM
            for pattern in patterns:
                self.pam.learn_from_pattern(pattern, success=True)

            # Store patterns in workspace
            for i, pattern in enumerate(patterns[:5]):  # Top 5 patterns
                self.workspace.upsert_object(
                    oid=f'pattern_{i}',
                    features={
                        'confidence': pattern.confidence,
                        'support': float(len(pattern.supporting_demos)),
                    },
                    properties={
                        'pattern_id': pattern.pattern_id,
                        'operations': ','.join(pattern.grid_operations),
                        'type': pattern.transformation_type
                    }
                )

            self._debug(f"  [arc_analyze_demos] Completed ({len(patterns)} patterns found)")

        def _generate_pam_hypotheses():
            """Generate hypotheses via PAM spreading activation."""
            if not self.demonstrations:
                return

            # Get best pattern from analysis
            patterns = self.analyzer.analyze_multiple_pairs(self.demonstrations)
            if not patterns:
                return

            seed_pattern = patterns[0]

            # Activate PAM and spread
            activations = self.pam.activate_and_spread(
                pattern=seed_pattern,
                iterations=5,
                decay=0.1
            )

            # Get top active operations
            top_ops = self.pam.get_top_active_operations(k=10)

            # Generate hypotheses from activated operations
            for i, pattern in enumerate(patterns[:5]):
                # Compute salience from PAM activation and pattern confidence
                pam_boost = sum(
                    activations.get(f'prim_{op}', 0.0)
                    for op in pattern.grid_operations
                ) / max(len(pattern.grid_operations), 1)

                salience = 0.7 * pattern.confidence + 0.3 * pam_boost

                hypothesis = PatternHypothesis(
                    hypothesis_id=f'hyp_{i}',
                    pattern=pattern,
                    salience=salience,
                    support_count=len(pattern.supporting_demos)
                )

                self.hypotheses.append(hypothesis)

            # Store top operations in workspace
            for op in top_ops[:5]:
                self.workspace.upsert_object(
                    oid=f'op_{op}',
                    features={'activation': activations.get(f'prim_{op}', 0.0)},
                    properties={'operation': op}
                )

        def _induce_categories():
            """Induce new composite operation categories from patterns."""
            new_categories = self.pam.induce_new_categories(min_occurrences=2)

            # Store induced categories in workspace
            for cat_id in new_categories:
                self.workspace.upsert_object(
                    oid=cat_id,
                    features={'category': 1.0},
                    properties={'type': 'composite'}
                )

        return [
            Codelet(
                name='arc_analyze_demos',
                urgency=1.0,
                action=_analyze_demonstrations,
                kind='understanding',
                metadata={'phase': 'analysis'}
            ),
            Codelet(
                name='arc_pam_hypotheses',
                urgency=0.9,
                action=_generate_pam_hypotheses,
                kind='understanding',
                metadata={'phase': 'hypothesis_generation'}
            ),
            Codelet(
                name='arc_category_induction',
                urgency=0.7,
                action=_induce_categories,
                kind='understanding',
                metadata={'phase': 'learning'}
            ),
        ]

    def make_attention_codelets(self) -> List[Codelet]:
        """Create codelets for attention phase: validate and compete hypotheses."""

        def _validate_hypotheses():
            """Validate hypotheses on demonstrations."""
            for hyp in self.hypotheses:
                # Test pattern on demonstrations
                correct = 0
                total = len(self.demonstrations)

                for demo in self.demonstrations:
                    # Apply pattern based on type
                    result = demo.input
                    try:
                        if hyp.pattern.grid_operations:
                            # Grid-level operations
                            for op_name in hyp.pattern.grid_operations:
                                prim = self.primitives.get(op_name)

                                # Handle operations with parameters
                                if op_name == 'recolor':
                                    # CRITICAL: For multi-op sequences, dynamically infer color mapping
                                    if len(hyp.pattern.grid_operations) > 1:
                                        # Infer mapping from current result to expected output
                                        color_mapping = self._infer_color_mapping(result, demo.output)
                                        if not color_mapping:
                                            raise ValueError("Cannot infer color mapping")
                                        result = prim.execute(result, color_mapping)
                                    elif hyp.pattern.color_mapping:
                                        # Single recolor: use stored mapping
                                        result = prim.execute(result, hyp.pattern.color_mapping)
                                    else:
                                        raise ValueError("No color mapping available")

                                elif op_name == 'tile':
                                    # Grid tiling with parameters
                                    params = hyp.pattern.operation_params
                                    repeat_v = params.get('repeat_v', 1)
                                    repeat_w = params.get('repeat_w', 1)
                                    result = prim.execute(result, repeat_v, repeat_w)

                                elif op_name == 'scale_grid':
                                    # Grid scaling with parameters
                                    params = hyp.pattern.operation_params
                                    scale_factor = params.get('scale_factor', 1.0)
                                    result = prim.execute(result, scale_factor)

                                elif op_name == 'recolor_if_has_neighbor':
                                    # Conditional recolor based on neighbors
                                    params = hyp.pattern.operation_params
                                    neighbor_color = params.get('neighbor_color', 1)
                                    new_color = params.get('new_color', 2)
                                    target_color = params.get('target_color', None)
                                    result = prim.execute(result, neighbor_color, new_color, target_color)

                                elif op_name == 'recolor_if_isolated':
                                    # Conditional recolor based on isolation
                                    params = hyp.pattern.operation_params
                                    new_color = params.get('new_color', 1)
                                    target_color = params.get('target_color', None)
                                    result = prim.execute(result, new_color, target_color)

                                elif op_name == 'recolor_if_on_edge':
                                    # Conditional recolor based on edge position
                                    params = hyp.pattern.operation_params
                                    new_color = params.get('new_color', 1)
                                    target_color = params.get('target_color', None)
                                    result = prim.execute(result, new_color, target_color)

                                elif op_name == 'remove_if_isolated':
                                    # Remove isolated pixels
                                    params = hyp.pattern.operation_params
                                    target_color = params.get('target_color', None)
                                    result = prim.execute(result, target_color)

                                elif op_name in ['dilate', 'erode']:
                                    # Morphological operations
                                    params = hyp.pattern.operation_params
                                    color = params.get('color', 1)
                                    iterations = params.get('iterations', 1)
                                    background = params.get('background', 0)
                                    result = prim.execute(result, color, iterations, background)

                                elif op_name == 'flood_fill':
                                    # Flood fill operation
                                    params = hyp.pattern.operation_params
                                    source_color = params.get('source_color', 1)
                                    fill_color = params.get('fill_color', 2)
                                    result = prim.execute(result, source_color, fill_color)

                                elif op_name == 'fill_enclosed':
                                    # Fill enclosed regions
                                    params = hyp.pattern.operation_params
                                    boundary_color = params.get('boundary_color', 1)
                                    fill_color = params.get('fill_color', 2)
                                    background = params.get('background', 0)
                                    result = prim.execute(result, boundary_color, fill_color, background)

                                elif op_name == 'spread_to_neighbors':
                                    # Propagation operation
                                    params = hyp.pattern.operation_params
                                    source_color = params.get('source_color', 1)
                                    target_color = params.get('target_color', 0)
                                    iterations = params.get('iterations', 1)
                                    result = prim.execute(result, source_color, target_color, iterations)

                                else:
                                    # No parameters needed
                                    result = prim.execute(result)

                        elif hyp.pattern.object_transformations:
                            # Object-level transformations
                            detect_prim = self.primitives.get('detect_objects')
                            if not detect_prim:
                                raise ValueError("detect_objects primitive not found")

                            input_objects = detect_prim.execute(demo.input)

                            # Apply transformations
                            output_objects = []
                            for i, obj in enumerate(input_objects):
                                if i < len(hyp.pattern.object_transformations):
                                    transform = hyp.pattern.object_transformations[i]
                                    transformed_obj = obj

                                    if 'color_delta' in transform:
                                        old_color, new_color = transform['color_delta']
                                        if transformed_obj.color == old_color:
                                            recolor_prim = self.primitives.get('recolor_object')
                                            if recolor_prim:
                                                transformed_obj = recolor_prim.execute(transformed_obj, new_color)

                                    if 'position_delta' in transform:
                                        delta_r, delta_c = transform['position_delta']
                                        move_prim = self.primitives.get('move_object')
                                        if move_prim:
                                            transformed_obj = move_prim.execute(transformed_obj, int(delta_r), int(delta_c))

                                    if 'size_delta' in transform and obj.size > 0:
                                        size_delta = transform['size_delta']
                                        new_size = obj.size + size_delta
                                        if new_size > 0:
                                            ratio = new_size / obj.size
                                            if abs(ratio - 1.0) > 0.01:
                                                scale_prim = self.primitives.get('scale_object')
                                                if scale_prim:
                                                    transformed_obj = scale_prim.execute(transformed_obj, ratio)

                                    output_objects.append(transformed_obj)
                                else:
                                    output_objects.append(obj)

                            # Render back to grid
                            render_prim = self.primitives.get('render_objects')
                            if render_prim:
                                result = render_prim.execute(output_objects, len(demo.input[0]), len(demo.input))

                        if result == demo.output:
                            correct += 1
                    except Exception:
                        # Pattern failed to apply
                        pass

                hyp.validation_accuracy = correct / total if total > 0 else 0.0

                # Update salience based on validation
                hyp.salience = 0.5 * hyp.salience + 0.5 * hyp.validation_accuracy

        def _create_coalitions():
            """Create coalitions from validated hypotheses."""
            self.coalitions.clear()

            for hyp in self.hypotheses:
                coalition = hyp.to_coalition()
                self.coalitions.append(coalition)

        return [
            Codelet(
                name='arc_validate_hypotheses',
                urgency=1.0,
                action=_validate_hypotheses,
                kind='attention',
                metadata={'coalitions': self.coalitions}
            ),
            Codelet(
                name='arc_create_coalitions',
                urgency=0.9,
                action=_create_coalitions,
                kind='attention',
                metadata={'coalitions': self.coalitions}
            ),
        ]

    def make_action_codelets(self, winning_coalition_id: Optional[str] = None) -> List[Codelet]:
        """Create codelets for action phase: apply winning pattern."""

        def _select_winning_pattern():
            """Select winning pattern from coalition."""
            self._debug("  [arc_select_winner] Selecting winning pattern")

            if not winning_coalition_id:
                self._debug("    ⚠ No winning coalition ID provided")
                return

            self._debug(f"    Looking for coalition: {winning_coalition_id}")
            self._debug(f"    Available hypotheses: {len(self.hypotheses)}")

            # Find hypothesis matching winning coalition
            found = False
            for hyp in self.hypotheses:
                self._debug(f"      Checking hypothesis: {hyp.hypothesis_id}")
                if hyp.hypothesis_id == winning_coalition_id:
                    self.winning_pattern = hyp.pattern
                    found = True

                    self._debug(f"    ✓ Found matching pattern: {hyp.pattern.pattern_id}")
                    self._debug(f"      Operations: {hyp.pattern.grid_operations}")
                    self._debug(f"      Confidence: {hyp.pattern.confidence:.3f}")
                    self._debug(f"      Validation accuracy: {hyp.validation_accuracy:.3f}")

                    # Store in workspace
                    self.workspace.upsert_object(
                        oid='winning_pattern',
                        features={
                            'confidence': hyp.pattern.confidence,
                            'validation': hyp.validation_accuracy,
                            'salience': hyp.salience
                        },
                        properties={
                            'pattern_id': hyp.pattern.pattern_id,
                            'operations': ','.join(hyp.pattern.grid_operations)
                        }
                    )
                    break

            if not found:
                self._debug(f"    ⚠ No matching hypothesis found for coalition {winning_coalition_id}")

        def _apply_pattern_to_test():
            """Apply winning pattern to test input."""
            self._debug("  [arc_apply_pattern] Applying pattern to test input")

            if not self.winning_pattern:
                self._debug("    ⚠ No winning pattern to apply")
                return

            if not self.test_input:
                self._debug("    ⚠ No test input provided")
                return

            self._debug(f"    Test input shape: {len(self.test_input)}x{len(self.test_input[0]) if self.test_input else 0}")
            self._debug(f"    Pattern type: {self.winning_pattern.transformation_type}")

            result = self.test_input

            try:
                # Branch based on pattern type
                if self.winning_pattern.grid_operations:
                    # GRID-LEVEL OPERATIONS
                    self._debug(f"    Applying {len(self.winning_pattern.grid_operations)} grid operations")

                    for i, op_name in enumerate(self.winning_pattern.grid_operations):
                        self._debug(f"      Step {i+1}/{len(self.winning_pattern.grid_operations)}: {op_name}")

                        prim = self.primitives.get(op_name)
                        if not prim:
                            self._debug(f"        ✗ Primitive '{op_name}' not found")
                            raise ValueError(f"Primitive '{op_name}' not found")

                        # Handle operations with parameters
                        if op_name == 'recolor':
                            # Color mapping
                            if self.winning_pattern.color_mapping:
                                self._debug(f"        Applying recolor with mapping: {self.winning_pattern.color_mapping}")
                                result = prim.execute(result, self.winning_pattern.color_mapping)
                            else:
                                self._debug(f"        ✗ Recolor requires color mapping")
                                raise ValueError("Recolor operation requires color mapping")

                        elif op_name == 'tile':
                            # Grid tiling
                            params = self.winning_pattern.operation_params
                            repeat_v = params.get('repeat_v', 1)
                            repeat_w = params.get('repeat_w', 1)
                            self._debug(f"        Applying tile {repeat_v}x{repeat_w}")
                            result = prim.execute(result, repeat_v, repeat_w)

                        elif op_name == 'scale_grid':
                            # Grid scaling
                            params = self.winning_pattern.operation_params
                            scale_factor = params.get('scale_factor', 1.0)
                            self._debug(f"        Applying scale_grid {scale_factor:.1f}x")
                            result = prim.execute(result, scale_factor)

                        elif op_name == 'recolor_if_has_neighbor':
                            # Conditional recolor based on neighbors
                            params = self.winning_pattern.operation_params
                            neighbor_color = params.get('neighbor_color', 1)
                            new_color = params.get('new_color', 2)
                            target_color = params.get('target_color', None)
                            self._debug(f"        Applying recolor_if_has_neighbor (neighbor={neighbor_color}, new={new_color})")
                            result = prim.execute(result, neighbor_color, new_color, target_color)

                        elif op_name == 'recolor_if_isolated':
                            # Conditional recolor based on isolation
                            params = self.winning_pattern.operation_params
                            new_color = params.get('new_color', 1)
                            target_color = params.get('target_color', None)
                            self._debug(f"        Applying recolor_if_isolated (new_color={new_color})")
                            result = prim.execute(result, new_color, target_color)

                        elif op_name == 'recolor_if_on_edge':
                            # Conditional recolor based on edge position
                            params = self.winning_pattern.operation_params
                            new_color = params.get('new_color', 1)
                            target_color = params.get('target_color', None)
                            self._debug(f"        Applying recolor_if_on_edge (new_color={new_color})")
                            result = prim.execute(result, new_color, target_color)

                        elif op_name == 'remove_if_isolated':
                            # Remove isolated pixels
                            params = self.winning_pattern.operation_params
                            target_color = params.get('target_color', None)
                            self._debug(f"        Applying remove_if_isolated")
                            result = prim.execute(result, target_color)

                        elif op_name in ['dilate', 'erode']:
                            # Morphological operations
                            params = self.winning_pattern.operation_params
                            color = params.get('color', 1)
                            iterations = params.get('iterations', 1)
                            background = params.get('background', 0)
                            self._debug(f"        Applying {op_name} (color={color}, iterations={iterations})")
                            result = prim.execute(result, color, iterations, background)

                        elif op_name == 'flood_fill':
                            # Flood fill operation
                            params = self.winning_pattern.operation_params
                            source_color = params.get('source_color', 1)
                            fill_color = params.get('fill_color', 2)
                            self._debug(f"        Applying flood_fill (source={source_color}, fill={fill_color})")
                            result = prim.execute(result, source_color, fill_color)

                        elif op_name == 'fill_enclosed':
                            # Fill enclosed regions
                            params = self.winning_pattern.operation_params
                            boundary_color = params.get('boundary_color', 1)
                            fill_color = params.get('fill_color', 2)
                            background = params.get('background', 0)
                            self._debug(f"        Applying fill_enclosed (boundary={boundary_color}, fill={fill_color})")
                            result = prim.execute(result, boundary_color, fill_color, background)

                        elif op_name == 'spread_to_neighbors':
                            # Propagation operation
                            params = self.winning_pattern.operation_params
                            source_color = params.get('source_color', 1)
                            target_color = params.get('target_color', 0)
                            iterations = params.get('iterations', 1)
                            self._debug(f"        Applying spread_to_neighbors (source={source_color}, target={target_color}, iters={iterations})")
                            result = prim.execute(result, source_color, target_color, iterations)

                        else:
                            # No parameters needed
                            result = prim.execute(result)

                        self._debug(f"        ✓ Result shape: {len(result)}x{len(result[0]) if result else 0}")

                elif self.winning_pattern.object_transformations:
                    # OBJECT-LEVEL TRANSFORMATIONS
                    self._debug(f"    Applying object-level transformations")

                    # Extract objects from test input
                    detect_prim = self.primitives.get('detect_objects')
                    if not detect_prim:
                        raise ValueError("detect_objects primitive not found")

                    input_objects = detect_prim.execute(self.test_input)
                    self._debug(f"      Detected {len(input_objects)} objects in test input")

                    # Apply transformations to each object
                    output_objects = []
                    for i, obj in enumerate(input_objects):
                        if i < len(self.winning_pattern.object_transformations):
                            transform = self.winning_pattern.object_transformations[i]
                            self._debug(f"      Applying transform to object {i}: {transform}")

                            transformed_obj = obj

                            # Apply color change if needed
                            if 'color_delta' in transform:
                                old_color, new_color = transform['color_delta']
                                if transformed_obj.color == old_color:
                                    recolor_prim = self.primitives.get('recolor_object')
                                    if recolor_prim:
                                        transformed_obj = recolor_prim.execute(transformed_obj, new_color)
                                        self._debug(f"        Recolored: {old_color} → {new_color}")

                            # Apply position change if needed
                            if 'position_delta' in transform:
                                delta_r, delta_c = transform['position_delta']
                                if abs(delta_r) > 0 or abs(delta_c) > 0:
                                    move_prim = self.primitives.get('move_object')
                                    if move_prim:
                                        transformed_obj = move_prim.execute(transformed_obj, delta_r, delta_c)
                                        self._debug(f"        Moved: ({delta_r}, {delta_c})")

                            # Apply size change if needed
                            if 'size_delta' in transform and obj.size > 0:
                                size_delta = transform['size_delta']
                                new_size = obj.size + size_delta
                                if new_size > 0:
                                    ratio = new_size / obj.size
                                    if abs(ratio - 1.0) > 0.01:
                                        scale_prim = self.primitives.get('scale_object')
                                        if scale_prim:
                                            transformed_obj = scale_prim.execute(transformed_obj, ratio)
                                            self._debug(f"        Scaled: {ratio:.2f}x ({obj.size} → {new_size} pixels)")

                            output_objects.append(transformed_obj)
                        else:
                            output_objects.append(obj)

                    # Render objects back to grid
                    render_prim = self.primitives.get('render_objects')
                    if not render_prim:
                        raise ValueError("render_objects primitive not found")

                    output_height = len(self.test_input)
                    output_width = len(self.test_input[0]) if self.test_input else 0
                    result = render_prim.execute(output_objects, output_width, output_height)
                    self._debug(f"      Rendered {len(output_objects)} objects to grid")

                elif self.winning_pattern.pixel_mapping:
                    # PIXEL-LEVEL MAPPING
                    self._debug(f"    Applying pixel-level mapping")
                    # TODO: Implement pixel mapping application
                    raise NotImplementedError("Pixel-level mapping not yet implemented")

                else:
                    self._debug(f"    ⚠ Pattern has no executable operations")
                    raise ValueError("Pattern has no grid_operations, object_transformations, or pixel_mapping")

                self.final_output = result
                self._debug(f"    ✓ Pattern applied successfully")
                self._debug(f"      Final output shape: {len(result)}x{len(result[0]) if result else 0}")

                # Store result in workspace
                self.workspace.upsert_object(
                    oid='test_output',
                    features={'applied': 1.0},
                    properties={'status': 'success'}
                )

            except Exception as e:
                # Pattern application failed
                self._debug(f"    ✗ Pattern application failed: {e}")
                import traceback
                self._debug(f"      Stack trace: {traceback.format_exc()}")

                self.workspace.upsert_object(
                    oid='test_output',
                    features={'applied': 0.0},
                    properties={'status': 'failed', 'error': str(e)}
                )

        def _hebbian_strengthen():
            """Strengthen successful pattern connections in PAM."""
            if self.winning_pattern and self.final_output:
                # Strengthen with reward = 1.0 (success)
                self.pam.hebbian_learner.strengthen_from_pattern(
                    self.winning_pattern,
                    reward=1.0
                )

        return [
            Codelet(
                name='arc_select_winner',
                urgency=1.0,
                action=_select_winning_pattern,
                kind='action',
                metadata={}
            ),
            Codelet(
                name='arc_apply_pattern',
                urgency=0.9,
                action=_apply_pattern_to_test,
                kind='action',
                metadata={}
            ),
            Codelet(
                name='arc_hebbian_learning',
                urgency=0.7,
                action=_hebbian_strengthen,
                kind='action',
                metadata={}
            ),
        ]

    def get_coalitions(self) -> List[Coalition]:
        """Get current coalitions for workspace competition."""
        return self.coalitions

    def set_task(self, demonstrations: List[GridPair], test_input: List[List[int]]):
        """Set the current ARC task to solve."""
        self.demonstrations = demonstrations
        self.test_input = test_input
        self.hypotheses.clear()
        self.coalitions.clear()
        self.winning_pattern = None
        self.final_output = None
        self.workspace.clear()

    def get_output(self) -> Optional[List[List[int]]]:
        """Get the final test output."""
        return self.final_output
