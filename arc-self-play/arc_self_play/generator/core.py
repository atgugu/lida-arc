"""Core puzzle generator: composes primitives to create ARC puzzles."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..types import Grid, ARCTask, GridPair
from ..primitives import PrimitiveLibrary
from .grids import GridGenerator
from .sampler import TransformationSampler
from .validator import PuzzleValidator


@dataclass
class GeneratedPuzzle:
    """An ARC puzzle produced by primitive composition."""

    puzzle_id: str
    train: List[Tuple[Grid, Grid]]
    test: Tuple[Grid, Grid]
    transformation: List[Tuple[str, Dict]]
    difficulty: str

    def to_arc_task(self) -> ARCTask:
        train_pairs = [GridPair(inp, out) for inp, out in self.train]
        test_pairs = [GridPair(self.test[0], self.test[1])]
        return ARCTask(task_id=self.puzzle_id, train=train_pairs, test=test_pairs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "train": [{"input": i, "output": o} for i, o in self.train],
            "test": {"input": self.test[0], "output": self.test[1]},
            "transformation": [
                {"op": name, "params": params}
                for name, params in self.transformation
            ],
            "difficulty": self.difficulty,
        }


class PuzzleGenerator:
    """Generate ARC puzzles by composing grid primitives.

    The generator works in three steps:
    1. Sample a random transformation sequence (chain of primitives).
    2. Generate random input grids and apply the transformation.
    3. Validate that the puzzle is non-trivial and well-formed.

    Retry logic avoids identity transformations (output == input).

    Example::

        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(difficulty="hard")
        print(puzzle.difficulty, len(puzzle.transformation))
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.library = PrimitiveLibrary()
        self.grids = GridGenerator(seed=seed)
        self.sampler = TransformationSampler(self.library)
        self._count = 0

    def generate(
        self,
        difficulty: str = "medium",
        num_demos: int = 3,
        max_retries: int = 5,
    ) -> Optional[GeneratedPuzzle]:
        """Generate a single puzzle, or ``None`` on failure.

        Args:
            difficulty: "easy", "medium", or "hard".
            num_demos: Number of training demonstration pairs.
            max_retries: Attempts before giving up (avoids identity transforms).
        """
        pid = f"synth_{self._count:06d}"
        self._count += 1

        for _ in range(max_retries):
            transform = self.sampler.sample(difficulty)

            # Build training pairs
            pairs: List[Tuple[Grid, Grid]] = []
            has_nontrivial = False

            for _ in range(num_demos):
                inp = self.grids.generate()
                out = self._apply(inp, transform)
                if out is None:
                    break
                pairs.append((inp, out))
                if inp != out:
                    has_nontrivial = True

            if len(pairs) < num_demos or not has_nontrivial:
                continue

            # Build test pair
            test_in = self.grids.generate()
            test_out = self._apply(test_in, transform)
            if test_out is None or test_in == test_out:
                continue

            # Validate
            ok, _ = PuzzleValidator.validate(pairs, (test_in, test_out))
            if not ok:
                continue

            return GeneratedPuzzle(
                puzzle_id=pid,
                train=pairs,
                test=(test_in, test_out),
                transformation=transform,
                difficulty=difficulty,
            )

        return None

    def generate_batch(
        self,
        n: int,
        difficulty_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> List[GeneratedPuzzle]:
        """Generate *n* valid puzzles with a difficulty distribution.

        Args:
            n: Number of puzzles to generate.
            difficulty_weights: e.g. ``{"easy": 0.1, "medium": 0.3, "hard": 0.6}``
            seed: Optional random seed override.

        Returns:
            List of generated puzzles (may be shorter than *n* if retries are
            exhausted).
        """
        import random as _rng

        if seed is not None:
            _rng.seed(seed)

        weights = difficulty_weights or {"easy": 0.1, "medium": 0.3, "hard": 0.6}
        levels = list(weights.keys())
        probs = [weights[l] for l in levels]

        puzzles: List[GeneratedPuzzle] = []
        max_attempts = n * 3

        for _ in range(max_attempts):
            if len(puzzles) >= n:
                break
            diff = _rng.choices(levels, probs)[0]
            p = self.generate(difficulty=diff)
            if p is not None:
                puzzles.append(p)

        return puzzles

    # ------------------------------------------------------------------

    def _apply(self, grid: Grid, transform: List[Tuple[str, Dict]]) -> Optional[Grid]:
        current = copy.deepcopy(grid)
        for op_name, params in transform:
            prim = self.library.get(op_name)
            if prim is None:
                return None
            try:
                if op_name == "recolor" and "color_map" in params:
                    current = prim.execute(current, params["color_map"])
                elif op_name == "tile" and params:
                    current = prim.execute(current, params["repeat_v"], params["repeat_h"])
                elif params:
                    current = prim.execute(current, **params)
                else:
                    current = prim.execute(current)
            except Exception:
                return None
        return current


def save_puzzles(puzzles: List[GeneratedPuzzle], path: Path) -> None:
    """Serialize puzzles to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([p.to_dict() for p in puzzles], f)


def load_puzzles(path: Path) -> List[Dict[str, Any]]:
    """Load puzzles from a JSON file (returns raw dicts)."""
    with open(path) as f:
        return json.load(f)
