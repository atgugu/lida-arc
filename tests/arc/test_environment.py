"""
Tests for ARC environment and grid structures.
"""

import json
import pytest
from pathlib import Path
from lida.arc.environment import GridPair, ARCTask, ARCEnvironment


class TestGridPair:
    def test_create_grid_pair(self):
        """Test basic GridPair creation."""
        grid_pair = GridPair(
            input=[[1, 2], [3, 4]],
            output=[[4, 3], [2, 1]]
        )

        assert grid_pair.input_shape == (2, 2)
        assert grid_pair.output_shape == (2, 2)

    def test_grid_pair_validation(self):
        """Test grid validation."""
        # Valid grid
        valid = GridPair(
            input=[[1, 2, 3], [4, 5, 6]],
            output=[[6, 5, 4], [3, 2, 1]]
        )
        assert valid.validate()

        # Invalid - values out of range
        invalid = GridPair(
            input=[[1, 2, 10]],  # 10 is out of range [0-9]
            output=[[1, 2, 3]]
        )
        assert not invalid.validate()

        # Invalid - jagged array
        jagged = GridPair(
            input=[[1, 2], [3]],  # Inconsistent row length
            output=[[1, 2], [3, 4]]
        )
        assert not jagged.validate()

    def test_shape_properties(self):
        """Test shape property methods."""
        grid_pair = GridPair(
            input=[[1, 2, 3, 4]],  # 1x4
            output=[[1], [2], [3], [4]]  # 4x1
        )

        assert grid_pair.input_shape == (1, 4)
        assert grid_pair.output_shape == (4, 1)


class TestARCTask:
    def test_create_arc_task(self):
        """Test ARCTask creation."""
        train_pairs = [
            GridPair(input=[[1, 0]], output=[[0, 1]]),
            GridPair(input=[[2, 0]], output=[[0, 2]]),
        ]
        test_pairs = [
            GridPair(input=[[3, 0]], output=[[0, 3]]),
        ]

        task = ARCTask(
            task_id="test_task_001",
            train=train_pairs,
            test=test_pairs
        )

        assert task.n_demonstrations == 2
        assert task.n_tests == 1
        assert task.task_id == "test_task_001"

    def test_get_demonstration(self):
        """Test retrieving demonstrations."""
        train_pairs = [
            GridPair(input=[[1, 0]], output=[[0, 1]]),
            GridPair(input=[[2, 0]], output=[[0, 2]]),
        ]

        task = ARCTask(
            task_id="test",
            train=train_pairs,
            test=[]
        )

        demo = task.get_demonstration(0)
        assert demo.input == [[1, 0]]
        assert demo.output == [[0, 1]]

        # Test index out of range
        with pytest.raises(IndexError):
            task.get_demonstration(5)

    def test_from_json(self):
        """Test creating task from JSON structure."""
        json_data = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 1]]},
                {"input": [[3, 4]], "output": [[4, 3]]},
            ],
            "test": [
                {"input": [[5, 6]], "output": [[6, 5]]},
            ]
        }

        task = ARCTask.from_json(json_data, task_id="json_test")

        assert task.task_id == "json_test"
        assert task.n_demonstrations == 2
        assert task.n_tests == 1
        assert task.train[0].input == [[1, 2]]


class TestARCEnvironment:
    def test_create_environment(self):
        """Test ARCEnvironment creation."""
        task = ARCTask(
            task_id="env_test",
            train=[GridPair(input=[[1, 0]], output=[[0, 1]])],
            test=[GridPair(input=[[2, 0]], output=[[0, 2]])]
        )

        env = ARCEnvironment(task)

        assert env.current_phase == "demonstration"
        assert env.current_grid == [[1, 0]]

    def test_get_observation(self):
        """Test observation retrieval."""
        task = ARCTask(
            task_id="obs_test",
            train=[GridPair(input=[[1, 0]], output=[[0, 1]])],
            test=[GridPair(input=[[2, 0]], output=[[0, 2]])]
        )

        env = ARCEnvironment(task)
        obs = env.get_observation()

        assert obs['grid'] == [[1, 0]]
        assert obs['phase'] == "demonstration"
        assert obs['demo_index'] == 0
        assert obs['task_id'] == "obs_test"

    def test_set_demonstration(self):
        """Test switching between demonstrations."""
        task = ARCTask(
            task_id="demo_switch",
            train=[
                GridPair(input=[[1, 0]], output=[[0, 1]]),
                GridPair(input=[[2, 0]], output=[[0, 2]]),
            ],
            test=[]
        )

        env = ARCEnvironment(task)
        assert env.current_grid == [[1, 0]]

        env.set_demonstration(1)
        assert env.current_grid == [[2, 0]]
        assert env.current_demo_idx == 1

    def test_set_test(self):
        """Test switching to test phase."""
        task = ARCTask(
            task_id="test_switch",
            train=[GridPair(input=[[1, 0]], output=[[0, 1]])],
            test=[GridPair(input=[[3, 0]], output=[[0, 3]])]
        )

        env = ARCEnvironment(task)
        env.set_test(0)

        assert env.current_phase == "test"
        assert env.current_grid == [[3, 0]]
        assert env.current_test_idx == 0

    def test_validate_output(self):
        """Test output validation."""
        task = ARCTask(
            task_id="validate_test",
            train=[GridPair(input=[[1, 0]], output=[[0, 1]])],
            test=[]
        )

        env = ARCEnvironment(task)

        # Perfect match
        assert env.validate_output([[0, 1]]) == 1.0
        assert env.is_correct([[0, 1]])

        # Partial match (1 out of 2 cells)
        assert env.validate_output([[0, 0]]) == 0.5

        # No match
        assert env.validate_output([[1, 0]]) == 0.0

        # Wrong size
        assert env.validate_output([[0, 1, 2]]) == 0.0

    def test_reset(self):
        """Test environment reset."""
        task = ARCTask(
            task_id="reset_test",
            train=[
                GridPair(input=[[1, 0]], output=[[0, 1]]),
                GridPair(input=[[2, 0]], output=[[0, 2]]),
            ],
            test=[GridPair(input=[[3, 0]], output=[[0, 3]])]
        )

        env = ARCEnvironment(task)

        # Move to different state
        env.set_test(0)
        assert env.current_phase == "test"

        # Reset
        env.reset()
        assert env.current_phase == "demonstration"
        assert env.current_demo_idx == 0
        assert env.current_grid == [[1, 0]]
