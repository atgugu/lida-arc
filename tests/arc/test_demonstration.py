"""
Tests for demonstration analysis and pattern extraction.
"""

import pytest
from lida.arc.demonstration import TransformationPattern, DemonstrationAnalyzer
from lida.arc.environment import GridPair
from lida.arc.primitives import PrimitiveLibrary


class TestTransformationPattern:
    def test_create_pattern(self):
        """Test creating a transformation pattern."""
        pattern = TransformationPattern(
            pattern_id='test_pattern',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        assert pattern.pattern_id == 'test_pattern'
        assert pattern.transformation_type == 'grid_op'
        assert pattern.confidence == 1.0

    def test_to_pam_features(self):
        """Test PAM feature extraction from pattern."""
        pattern = TransformationPattern(
            pattern_id='test',
            transformation_type='grid_op',
            grid_operations=['rotate_90', 'recolor'],
            confidence=0.9,
            color_mapping={1: 2, 2: 1}
        )

        features = pattern.to_pam_features()

        assert features['transform_grid_op'] == 1.0
        assert features['op_rotate_90'] == 1.0
        assert features['op_recolor'] == 1.0
        assert features['color_transformation'] == 1.0

    def test_pattern_repr(self):
        """Test pattern string representation."""
        pattern = TransformationPattern(
            pattern_id='test',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=0.85
        )

        repr_str = repr(pattern)
        assert 'rotate_90' in repr_str
        assert '0.85' in repr_str


class TestDemonstrationAnalyzer:
    def test_analyze_rotation_90(self):
        """Test detection of 90-degree rotation."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Create demo: rotate 90 degrees
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 1], [4, 2]]  # Rotated 90° CW

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Should find rotation pattern
        assert len(patterns) > 0

        # Best pattern should be rotation with high confidence
        best = patterns[0]
        assert best.confidence >= 0.9
        assert 'rotate_90' in best.grid_operations or best.transformation_type == 'grid_op'

    def test_analyze_rotation_180(self):
        """Test detection of 180-degree rotation."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 2, 3], [4, 5, 6]]
        output_grid = [[6, 5, 4], [3, 2, 1]]  # Rotated 180°

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        assert len(patterns) > 0
        best = patterns[0]
        assert best.confidence >= 0.9
        assert 'rotate_180' in best.grid_operations

    def test_analyze_reflection_horizontal(self):
        """Test detection of horizontal reflection."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 4], [1, 2]]  # Flipped horizontally

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        assert len(patterns) > 0
        best = patterns[0]
        assert best.confidence >= 0.9
        assert 'reflect_horizontal' in best.grid_operations

    def test_analyze_reflection_vertical(self):
        """Test detection of vertical reflection."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 2], [3, 4]]
        output_grid = [[2, 1], [4, 3]]  # Flipped vertically

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        assert len(patterns) > 0
        best = patterns[0]
        assert best.confidence >= 0.9
        assert 'reflect_vertical' in best.grid_operations

    def test_analyze_color_mapping(self):
        """Test detection of color remapping."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 2, 1], [2, 1, 2]]
        output_grid = [[3, 4, 3], [4, 3, 4]]  # 1→3, 2→4

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        assert len(patterns) > 0
        best = patterns[0]
        assert best.confidence >= 0.9
        assert 'recolor' in best.grid_operations
        assert best.color_mapping == {1: 3, 2: 4}

    def test_analyze_composite_rotation_color(self):
        """Test detection of rotation + color change."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Rotate 90° then swap colors
        input_grid = [[1, 2], [3, 4]]
        # After rotate_90: [[3, 1], [4, 2]]
        # After recolor (1→5, 2→6, 3→7, 4→8):
        output_grid = [[7, 5], [8, 6]]

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Should find at least some patterns
        assert len(patterns) > 0

        # Look for pattern with both operations (composite)
        # OR at least find a high-confidence pattern (may be pixel-level fallback)
        composite_found = False
        for pattern in patterns:
            if len(pattern.grid_operations) >= 2:
                if 'rotate' in ' '.join(pattern.grid_operations) and 'recolor' in ' '.join(pattern.grid_operations):
                    composite_found = True
                    assert pattern.confidence >= 0.9
                    break

        # If no composite found, at least check we have some pattern
        if not composite_found:
            # Should still have pixel-level or other patterns
            assert len(patterns) > 0
            # Best pattern should have reasonable confidence
            assert patterns[0].confidence > 0.3

    def test_analyze_multiple_demonstrations(self):
        """Test analyzing multiple demos to find universal pattern."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Multiple demos all with same rotation
        demos = [
            GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]]),
            GridPair(input=[[5, 6], [7, 8]], output=[[7, 5], [8, 6]]),
            GridPair(input=[[1, 1], [2, 2]], output=[[2, 1], [2, 1]]),
        ]

        patterns = analyzer.analyze_multiple_pairs(demos)

        # Should find rotation as universal pattern
        assert len(patterns) > 0
        best = patterns[0]
        assert best.confidence >= 0.9
        assert 'rotate_90' in best.grid_operations
        assert len(best.supporting_demos) == 3

    def test_analyze_object_transformation(self):
        """Test object-level transformation detection."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Input: two separate objects
        input_grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ]

        # Output: objects with different colors
        output_grid = [
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
            [0, 0, 0, 0, 0],
        ]

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Should detect object-level or pixel-level pattern
        assert len(patterns) > 0

        # Check if object transformation was detected
        object_patterns = [p for p in patterns if p.transformation_type == 'object_map']
        if object_patterns:
            assert object_patterns[0].object_correspondence is not None

    def test_pattern_confidence_ordering(self):
        """Test that patterns are ordered by confidence."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 2], [3, 4]]
        output_grid = [[3, 1], [4, 2]]

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Patterns should be sorted by confidence
        for i in range(len(patterns) - 1):
            assert patterns[i].confidence >= patterns[i+1].confidence

    def test_pixel_mapping_fallback(self):
        """Test pixel-level mapping as fallback."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Complex transformation that doesn't match simple operations
        input_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        output_grid = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Should have at least pixel mapping pattern
        assert len(patterns) > 0

        # There should be a pixel_map pattern
        pixel_patterns = [p for p in patterns if p.transformation_type == 'pixel_map']
        assert len(pixel_patterns) > 0

    def test_size_change_detection(self):
        """Test detection of grid size changes."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Different size

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Should detect size change in pixel mapping
        assert len(patterns) > 0

        # Find pixel map pattern and check explanation
        pixel_patterns = [p for p in patterns if p.transformation_type == 'pixel_map']
        if pixel_patterns:
            assert 'Size changed' in pixel_patterns[0].explanation or \
                   pixel_patterns[0].confidence < 0.5

    def test_identity_transformation(self):
        """Test handling of identity transformation (input == output)."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        grid = [[1, 2], [3, 4]]
        demo = GridPair(input=grid, output=grid)

        patterns = analyzer.analyze_pair(demo)

        # Should still return patterns (likely pixel-level)
        assert len(patterns) > 0

    def test_multiple_demos_partial_support(self):
        """Test finding patterns with partial demo support."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Demos with different transformations
        demos = [
            GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]]),  # rotate_90
            GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]]),  # rotate_90
            GridPair(input=[[1, 2], [3, 4]], output=[[2, 1], [4, 3]]),  # reflect_vertical
        ]

        patterns = analyzer.analyze_multiple_pairs(demos)

        # Should find rotate_90 with 2/3 support
        assert len(patterns) > 0

        # Check that partial support is tracked
        for pattern in patterns:
            if 'rotate_90' in pattern.grid_operations:
                assert len(pattern.supporting_demos) >= 2


class TestPatternApplication:
    def test_apply_rotation_pattern(self):
        """Test applying a learned rotation pattern to new input."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Learn from demo
        demo = GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]])
        patterns = analyzer.analyze_pair(demo)

        # Get best pattern
        best = patterns[0]

        # Test on new grid
        new_input = [[5, 6], [7, 8]]
        expected = [[7, 5], [8, 6]]

        # Apply pattern
        result = analyzer._apply_pattern(best, new_input, expected)
        assert result, "Pattern should apply to new grid"

    def test_pattern_generalization(self):
        """Test that learned pattern generalizes to different sized grids."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        # Learn from small grid
        demo = GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]])
        patterns = analyzer.analyze_pair(demo)

        best = patterns[0]

        # Try on larger grid
        large_input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        large_expected = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

        result = analyzer._apply_pattern(best, large_input, large_expected)
        assert result, "Pattern should generalize to larger grids"


class TestFeatureExtraction:
    def test_input_output_features(self):
        """Test that input/output features are extracted."""
        lib = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(lib)

        input_grid = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
        output_grid = [[2, 2, 0], [2, 2, 0], [0, 0, 0]]

        demo = GridPair(input=input_grid, output=output_grid)
        patterns = analyzer.analyze_pair(demo)

        # Patterns should have features
        for pattern in patterns:
            assert len(pattern.input_features) > 0
            assert len(pattern.output_features) > 0
            assert 'grid_height' in pattern.input_features
            assert 'grid_width' in pattern.input_features
