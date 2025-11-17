"""
ARC task environment and grid data structures.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GridPair:
    """Represents an input-output grid pair from an ARC task."""

    input: List[List[int]]  # 2D grid, values 0-9
    output: List[List[int]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Shape of input grid as (height, width)."""
        if not self.input:
            return (0, 0)
        return (len(self.input), len(self.input[0]) if self.input else 0)

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Shape of output grid as (height, width)."""
        if not self.output:
            return (0, 0)
        return (len(self.output), len(self.output[0]) if self.output else 0)

    def validate(self) -> bool:
        """Validate grid integrity."""
        # Check input is well-formed
        if not self.input:
            return False

        input_width = len(self.input[0])
        for row in self.input:
            if len(row) != input_width:
                return False
            for val in row:
                if not (0 <= val <= 9):
                    return False

        # Check output is well-formed
        if not self.output:
            return False

        output_width = len(self.output[0])
        for row in self.output:
            if len(row) != output_width:
                return False
            for val in row:
                if not (0 <= val <= 9):
                    return False

        return True


@dataclass
class ARCTask:
    """Represents a complete ARC task with demonstrations and test cases."""

    task_id: str
    train: List[GridPair]  # 2-4 demonstration pairs
    test: List[GridPair]   # 1-2 test pairs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_demonstration(self, idx: int) -> GridPair:
        """Get a specific demonstration pair."""
        if idx < 0 or idx >= len(self.train):
            raise IndexError(f"Demo index {idx} out of range [0, {len(self.train)})")
        return self.train[idx]

    def get_test(self, idx: int) -> GridPair:
        """Get a specific test pair."""
        if idx < 0 or idx >= len(self.test):
            raise IndexError(f"Test index {idx} out of range [0, {len(self.test)})")
        return self.test[idx]

    @property
    def n_demonstrations(self) -> int:
        """Number of demonstration pairs."""
        return len(self.train)

    @property
    def n_tests(self) -> int:
        """Number of test pairs."""
        return len(self.test)

    def validate(self) -> bool:
        """Validate task integrity."""
        if not self.train:
            return False
        if not self.test:
            return False

        for demo in self.train:
            if not demo.validate():
                return False

        for test in self.test:
            if not test.validate():
                return False

        return True

    @classmethod
    def from_json(cls, task_data: Dict[str, Any], task_id: str = "unknown") -> "ARCTask":
        """Create ARCTask from JSON data structure.

        Expected format:
        {
            "train": [{"input": [[...]], "output": [[...]]}],
            "test": [{"input": [[...]], "output": [[...]]}]
        }
        """
        train_pairs = [
            GridPair(input=pair["input"], output=pair["output"])
            for pair in task_data.get("train", [])
        ]

        test_pairs = [
            GridPair(input=pair["input"], output=pair["output"])
            for pair in task_data.get("test", [])
        ]

        return cls(
            task_id=task_id,
            train=train_pairs,
            test=test_pairs
        )

    @classmethod
    def load_from_file(cls, path: Path) -> "ARCTask":
        """Load ARC task from JSON file."""
        with open(path, 'r') as f:
            task_data = json.load(f)

        task_id = path.stem  # Use filename without extension as task_id
        return cls.from_json(task_data, task_id=task_id)


class ARCEnvironment:
    """Environment adapter for ARC tasks compatible with LIDA cognitive cycle."""

    def __init__(self, task: ARCTask):
        self.task = task
        self.current_phase: str = "demonstration"  # "demonstration" or "test"
        self.current_demo_idx: int = 0
        self.current_test_idx: int = 0
        self.current_grid: Optional[List[List[int]]] = None

        # Initialize with first demonstration input
        if self.task.train:
            self.current_grid = self.task.train[0].input

    def get_observation(self) -> Dict[str, Any]:
        """Return current state as observation for cognitive cycle."""
        return {
            'grid': self.current_grid,
            'phase': self.current_phase,
            'demo_index': self.current_demo_idx if self.current_phase == "demonstration" else None,
            'test_index': self.current_test_idx if self.current_phase == "test" else None,
            'task_id': self.task.task_id,
        }

    def set_demonstration(self, idx: int):
        """Set current grid to a demonstration input."""
        demo = self.task.get_demonstration(idx)
        self.current_grid = demo.input
        self.current_demo_idx = idx
        self.current_phase = "demonstration"

    def set_test(self, idx: int):
        """Set current grid to a test input."""
        test = self.task.get_test(idx)
        self.current_grid = test.input
        self.current_test_idx = idx
        self.current_phase = "test"

    def get_current_expected_output(self) -> Optional[List[List[int]]]:
        """Get expected output for current grid (for validation)."""
        if self.current_phase == "demonstration":
            return self.task.train[self.current_demo_idx].output
        elif self.current_phase == "test":
            return self.task.test[self.current_test_idx].output
        return None

    def validate_output(self, predicted: List[List[int]]) -> float:
        """Validate predicted output against expected output.

        Returns accuracy as percentage of correct cells (0.0 to 1.0).
        """
        expected = self.get_current_expected_output()
        if expected is None:
            return 0.0

        # Check if shapes match
        if len(predicted) != len(expected):
            return 0.0
        if not predicted or not expected:
            return 0.0
        if len(predicted[0]) != len(expected[0]):
            return 0.0

        # Count matching cells
        total_cells = len(expected) * len(expected[0])
        matching_cells = 0

        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if predicted[r][c] == expected[r][c]:
                    matching_cells += 1

        return matching_cells / total_cells if total_cells > 0 else 0.0

    def is_correct(self, predicted: List[List[int]]) -> bool:
        """Check if predicted output exactly matches expected output."""
        return self.validate_output(predicted) == 1.0

    def reset(self):
        """Reset to initial state (first demonstration)."""
        self.current_phase = "demonstration"
        self.current_demo_idx = 0
        self.current_test_idx = 0
        if self.task.train:
            self.current_grid = self.task.train[0].input


# Utility functions

def load_arc_dataset(dataset_dir: Path, limit: Optional[int] = None) -> List[ARCTask]:
    """Load multiple ARC tasks from a directory of JSON files.

    Args:
        dataset_dir: Directory containing ARC task JSON files
        limit: Optional limit on number of tasks to load

    Returns:
        List of ARCTask objects
    """
    tasks = []
    json_files = sorted(dataset_dir.glob("*.json"))

    if limit:
        json_files = json_files[:limit]

    for json_file in json_files:
        try:
            task = ARCTask.load_from_file(json_file)
            if task.validate():
                tasks.append(task)
            else:
                print(f"Warning: Invalid task in {json_file}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return tasks
