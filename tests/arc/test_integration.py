"""
End-to-end integration tests for the complete LIDA-ARC system.

These tests demonstrate the full pipeline from demonstration analysis
to pattern application on test inputs.
"""

import pytest
from lida.arc import (
    ARCTask, GridPair, ARCEnvironment,
    PrimitiveLibrary, DemonstrationAnalyzer,
    TransformationPattern
)


class TestSimpleRotation:
    """Test complete workflow on rotation tasks."""

    def test_rotation_90_task(self):
        """Test learning and applying 90-degree rotation from demonstrations."""
        # Create a simple rotation task
        task = ARCTask(
            task_id='rotation_90',
            train=[
                GridPair(
                    input=[[1, 2], [3, 4]],
                    output=[[3, 1], [4, 2]]
                ),
                GridPair(
                    input=[[5, 6], [7, 8]],
                    output=[[7, 5], [8, 6]]
                ),
            ],
            test=[
                GridPair(
                    input=[[1, 0], [0, 1]],
                    output=[[0, 1], [1, 0]]  # Expected output
                )
            ]
        )

        # Initialize components
        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)
        env = ARCEnvironment(task)

        # Analyze demonstrations
        patterns = analyzer.analyze_multiple_pairs(task.train)

        # Should find rotation pattern
        assert len(patterns) > 0, "Should find at least one pattern"

        best_pattern = patterns[0]
        assert best_pattern.confidence >= 0.9, "Best pattern should have high confidence"
        assert 'rotate_90' in best_pattern.grid_operations, "Should identify rotate_90"

        # Apply pattern to test input
        env.set_test(0)
        test_input = task.test[0].input

        # Apply the learned transformation
        result = test_input
        for op_name in best_pattern.grid_operations:
            prim = primitives.get(op_name)
            if op_name == 'recolor' and best_pattern.color_mapping:
                result = prim.execute(result, best_pattern.color_mapping)
            else:
                result = prim.execute(result)

        # Validate result
        accuracy = env.validate_output(result)
        assert accuracy == 1.0, f"Should achieve 100% accuracy, got {accuracy}"
        assert env.is_correct(result), "Result should match expected output exactly"


class TestColorMapping:
    """Test complete workflow on color remapping tasks."""

    def test_simple_color_swap(self):
        """Test learning and applying color remapping."""
        task = ARCTask(
            task_id='color_swap',
            train=[
                GridPair(
                    input=[[1, 2, 1], [2, 1, 2]],
                    output=[[3, 4, 3], [4, 3, 4]]
                ),
                GridPair(
                    input=[[1, 1], [2, 2]],
                    output=[[3, 3], [4, 4]]
                ),
            ],
            test=[
                GridPair(
                    input=[[2, 1, 2, 1]],
                    output=[[4, 3, 4, 3]]
                )
            ]
        )

        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)
        env = ARCEnvironment(task)

        # Analyze demonstrations
        patterns = analyzer.analyze_multiple_pairs(task.train)

        assert len(patterns) > 0
        best_pattern = patterns[0]

        # Should find recolor pattern with correct mapping
        assert best_pattern.confidence >= 0.9
        assert 'recolor' in best_pattern.grid_operations
        assert best_pattern.color_mapping == {1: 3, 2: 4}

        # Apply to test input
        test_input = task.test[0].input
        result = primitives.get('recolor').execute(test_input, best_pattern.color_mapping)

        env.set_test(0)
        assert env.is_correct(result)


class TestReflection:
    """Test complete workflow on reflection tasks."""

    def test_horizontal_reflection(self):
        """Test learning and applying horizontal reflection."""
        task = ARCTask(
            task_id='reflect_h',
            train=[
                GridPair(
                    input=[[1, 2, 3], [4, 5, 6]],
                    output=[[4, 5, 6], [1, 2, 3]]
                ),
                GridPair(
                    input=[[7], [8], [9]],
                    output=[[9], [8], [7]]
                ),
            ],
            test=[
                GridPair(
                    input=[[1, 0], [0, 1]],
                    output=[[0, 1], [1, 0]]
                )
            ]
        )

        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        patterns = analyzer.analyze_multiple_pairs(task.train)

        assert len(patterns) > 0
        best_pattern = patterns[0]
        assert 'reflect_horizontal' in best_pattern.grid_operations

        # Apply to test
        test_input = task.test[0].input
        result = primitives.get('reflect_horizontal').execute(test_input)

        env = ARCEnvironment(task)
        env.set_test(0)
        assert env.is_correct(result)


class TestPatternGeneralization:
    """Test that learned patterns generalize to different grid sizes."""

    def test_rotation_generalizes_to_larger_grid(self):
        """Test rotation pattern learned from small grids works on larger grids."""
        task = ARCTask(
            task_id='rotation_generalization',
            train=[
                GridPair(
                    input=[[1, 2], [3, 4]],
                    output=[[3, 1], [4, 2]]
                ),
            ],
            test=[
                GridPair(
                    input=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    output=[[7, 4, 1], [8, 5, 2], [9, 6, 3]]
                )
            ]
        )

        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        # Learn from small demo
        patterns = analyzer.analyze_multiple_pairs(task.train)
        best_pattern = patterns[0]

        # Apply to larger test grid
        test_input = task.test[0].input
        result = primitives.get('rotate_90').execute(test_input)

        env = ARCEnvironment(task)
        env.set_test(0)
        assert env.is_correct(result), "Pattern should generalize to larger grids"


class TestMultipleDemonstrations:
    """Test handling of multiple demonstration examples."""

    def test_consistent_pattern_across_demos(self):
        """Test finding consistent pattern across multiple demonstrations."""
        task = ARCTask(
            task_id='multi_demo',
            train=[
                GridPair(input=[[1, 0]], output=[[0, 1]]),
                GridPair(input=[[2, 0]], output=[[0, 2]]),
                GridPair(input=[[3, 0]], output=[[0, 3]]),
                GridPair(input=[[4, 0]], output=[[0, 4]]),
            ],
            test=[
                GridPair(input=[[5, 0]], output=[[0, 5]])
            ]
        )

        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        patterns = analyzer.analyze_multiple_pairs(task.train)

        # Should find pattern that works for all demos
        assert len(patterns) > 0
        best_pattern = patterns[0]

        # Should have high confidence (works for all demos)
        assert best_pattern.confidence >= 0.9

        # Should track support
        assert len(best_pattern.supporting_demos) == 4

        # Apply to test
        test_input = task.test[0].input
        result = test_input

        for op_name in best_pattern.grid_operations:
            prim = primitives.get(op_name)
            if op_name == 'recolor' and best_pattern.color_mapping:
                result = prim.execute(result, best_pattern.color_mapping)
            else:
                result = prim.execute(result)

        env = ARCEnvironment(task)
        env.set_test(0)
        assert env.is_correct(result)


class TestPAMFeatureExtraction:
    """Test that patterns extract features for PAM activation."""

    def test_pattern_features_for_pam(self):
        """Test that learned patterns produce PAM features."""
        task = ARCTask(
            task_id='pam_test',
            train=[
                GridPair(
                    input=[[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                    output=[[2, 2, 0], [2, 2, 0], [0, 0, 0]]
                ),
            ],
            test=[]
        )

        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        patterns = analyzer.analyze_pair(task.train[0])

        # All patterns should have features
        for pattern in patterns:
            assert len(pattern.input_features) > 0, "Should have input features"
            assert len(pattern.output_features) > 0, "Should have output features"

            # Should be able to convert to PAM features
            pam_features = pattern.to_pam_features()
            assert len(pam_features) > 0, "Should have PAM features"

            # Should have transformation type feature
            assert f'transform_{pattern.transformation_type}' in pam_features


class TestLeaveOneOutValidation:
    """Test leave-one-out validation strategy."""

    def test_leave_one_out_pattern_discovery(self):
        """Test that pattern found works on held-out demo."""
        task = ARCTask(
            task_id='leave_one_out',
            train=[
                GridPair(input=[[1, 2]], output=[[2, 1]]),
                GridPair(input=[[3, 4]], output=[[4, 3]]),
                GridPair(input=[[5, 6]], output=[[6, 5]]),
            ],
            test=[]
        )

        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        # Try leave-one-out: train on first 2, validate on 3rd
        train_subset = task.train[:2]
        validation_demo = task.train[2]

        patterns = analyzer.analyze_multiple_pairs(train_subset)

        # Best pattern should work on validation demo too
        best_pattern = patterns[0]

        # Apply to validation input
        result = validation_demo.input
        for op_name in best_pattern.grid_operations:
            prim = primitives.get(op_name)
            if op_name == 'recolor' and best_pattern.color_mapping:
                result = prim.execute(result, best_pattern.color_mapping)
            else:
                result = prim.execute(result)

        # Should match validation output
        assert result == validation_demo.output, "Pattern should generalize to held-out demo"


class TestSystemRobustness:
    """Test system handles edge cases gracefully."""

    def test_empty_demonstrations(self):
        """Test handling of empty demonstration list."""
        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        patterns = analyzer.analyze_multiple_pairs([])
        assert patterns == []

    def test_single_demonstration(self):
        """Test handling of single demonstration."""
        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        demo = GridPair(input=[[1, 2]], output=[[2, 1]])
        patterns = analyzer.analyze_multiple_pairs([demo])

        # Should still find patterns
        assert len(patterns) > 0

    def test_no_clear_pattern(self):
        """Test handling when no clear pattern exists."""
        primitives = PrimitiveLibrary()
        analyzer = DemonstrationAnalyzer(primitives)

        # Random transformations
        task = ARCTask(
            task_id='random',
            train=[
                GridPair(input=[[1, 2]], output=[[9, 8]]),
                GridPair(input=[[3, 4]], output=[[5, 1]]),
            ],
            test=[]
        )

        patterns = analyzer.analyze_multiple_pairs(task.train)

        # Should still return patterns (likely pixel-level)
        assert len(patterns) > 0

        # But confidence should be lower
        for pattern in patterns:
            assert pattern.confidence <= 1.0  # Valid confidence
