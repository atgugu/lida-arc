"""Core types for ARC puzzles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Grid = List[List[int]]


@dataclass
class GridPair:
    """An input-output grid pair from an ARC task."""

    input: Grid
    output: Grid

    @property
    def input_shape(self) -> Tuple[int, int]:
        if not self.input:
            return (0, 0)
        return (len(self.input), len(self.input[0]))

    @property
    def output_shape(self) -> Tuple[int, int]:
        if not self.output:
            return (0, 0)
        return (len(self.output), len(self.output[0]))

    def validate(self) -> bool:
        for grid in (self.input, self.output):
            if not grid:
                return False
            width = len(grid[0])
            for row in grid:
                if len(row) != width:
                    return False
                if not all(0 <= v <= 9 for v in row):
                    return False
        return True


@dataclass
class ARCTask:
    """A complete ARC task with demonstrations and test cases."""

    task_id: str
    train: List[GridPair]
    test: List[GridPair]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any], task_id: str = "unknown") -> ARCTask:
        train = [GridPair(p["input"], p["output"]) for p in data.get("train", [])]
        test = [GridPair(p["input"], p["output"]) for p in data.get("test", [])]
        return cls(task_id=task_id, train=train, test=test)

    @classmethod
    def load(cls, path: Path) -> ARCTask:
        with open(path) as f:
            return cls.from_json(json.load(f), task_id=path.stem)

    def validate(self) -> bool:
        return bool(self.train and self.test and
                    all(p.validate() for p in self.train + self.test))


def load_dataset(directory: Path, limit: Optional[int] = None) -> List[ARCTask]:
    """Load ARC tasks from a directory of JSON files."""
    files = sorted(Path(directory).glob("*.json"))
    if limit:
        files = files[:limit]
    tasks = []
    for f in files:
        try:
            task = ARCTask.load(f)
            if task.validate():
                tasks.append(task)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
    return tasks
