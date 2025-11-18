"""
Self-play puzzle generation for ARC-AGI.

This module implements adversarial puzzle generation where a generator creates
ARC puzzles by composing primitives, and we validate them for training data.
"""

from __future__ import annotations

import random
import uuid
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import copy

from .primitives import PrimitiveLibrary
from .environment import ARCTask, GridPair


# Type aliases
Grid = List[List[int]]


@dataclass
class GeneratedPuzzle:
    """A generated ARC puzzle with metadata."""

    puzzle_id: str
    train: List[Tuple[Grid, Grid]]  # Training demonstration pairs
    test: Tuple[Grid, Grid]  # Test pair
    transformation: List[Tuple[str, Dict]]  # Sequence of (operation_name, params)
    difficulty: str  # 'easy', 'medium', 'hard'
    generation_params: Dict[str, Any]  # Parameters used during generation

    def to_arc_task(self) -> ARCTask:
        """Convert to ARCTask for solver."""
        train_pairs = [GridPair(inp, out) for inp, out in self.train]
        test_pairs = [GridPair(self.test[0], self.test[1])]
        return ARCTask(
            task_id=self.puzzle_id,
            train=train_pairs,
            test=test_pairs
        )


class GridGenerator:
    """Generate random input grids for puzzles."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate(self, height: Optional[int] = None, width: Optional[int] = None,
                 grid_type: str = 'random') -> Grid:
        """Generate a random grid.

        Args:
            height: Grid height (random 3-10 if None)
            width: Grid width (random 3-10 if None)
            grid_type: 'sparse', 'dense', 'structured', or 'random'
        """
        if height is None:
            height = random.randint(3, 10)
        if width is None:
            width = random.randint(3, 10)

        if grid_type == 'random':
            grid_type = random.choice(['sparse', 'dense', 'structured'])

        if grid_type == 'sparse':
            return self._generate_sparse(height, width)
        elif grid_type == 'dense':
            return self._generate_dense(height, width)
        elif grid_type == 'structured':
            return self._generate_structured(height, width)
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

    def _generate_sparse(self, h: int, w: int) -> Grid:
        """Generate sparse grid (15-25% non-zero pixels)."""
        grid = [[0] * w for _ in range(h)]
        # Increase density to reduce identity transformations
        num_pixels = int(h * w * random.uniform(0.15, 0.25))
        num_pixels = max(num_pixels, min(3, h * w))  # At least 3 pixels

        for _ in range(num_pixels):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            color = random.randint(1, 9)  # Colors 1-9
            grid[r][c] = color

        return grid

    def _generate_dense(self, h: int, w: int) -> Grid:
        """Generate dense grid (60-80% non-zero pixels)."""
        grid = [[0] * w for _ in range(h)]
        num_pixels = int(h * w * random.uniform(0.6, 0.8))

        for _ in range(num_pixels):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            color = random.randint(1, 9)
            grid[r][c] = color

        return grid

    def _generate_structured(self, h: int, w: int) -> Grid:
        """Generate grid with geometric patterns."""
        grid = [[0] * w for _ in range(h)]

        # Add 1-3 random rectangles
        num_shapes = random.randint(1, 3)
        for _ in range(num_shapes):
            if h <= 1 or w <= 1:
                break

            r1 = random.randint(0, h - 2)
            c1 = random.randint(0, w - 2)
            r2 = random.randint(r1 + 1, h)
            c2 = random.randint(c1 + 1, w)
            color = random.randint(1, 9)

            # Fill rectangle
            for r in range(r1, r2):
                for c in range(c1, c2):
                    grid[r][c] = color

        return grid


class TransformationSampler:
    """Sample transformation sequences (primitive compositions)."""

    def __init__(self, primitive_library: PrimitiveLibrary):
        self.primitives = primitive_library
        # Primitives that don't require complex parameters
        self.simple_ops = [
            'rotate_90', 'rotate_180', 'rotate_270',
            'reflect_horizontal', 'reflect_vertical', 'reflect_diagonal',
            'auto_crop'
        ]

    def sample_transformation(self, difficulty: str = 'medium') -> List[Tuple[str, Dict]]:
        """Sample a transformation sequence based on difficulty.

        Args:
            difficulty: 'easy' (1 op), 'medium' (2-3 ops), 'hard' (3-5 ops)

        Returns:
            List of (operation_name, parameters) tuples
        """
        if difficulty == 'easy':
            num_ops = 1
        elif difficulty == 'medium':
            num_ops = random.randint(2, 3)
        elif difficulty == 'hard':
            num_ops = random.randint(3, 5)
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        transformation = []
        for _ in range(num_ops):
            op_name = random.choice(self.simple_ops)
            params = self._sample_parameters(op_name)
            transformation.append((op_name, params))

        return transformation

    def _sample_parameters(self, op_name: str) -> Dict:
        """Sample parameters for an operation."""
        # For now, simple operations don't need parameters
        # In future, can add parametric operations with random params
        return {}


class PrimitiveCompositionGenerator:
    """Generate ARC puzzles by composing primitives."""

    def __init__(self, primitive_library: PrimitiveLibrary, seed: Optional[int] = None):
        self.primitives = primitive_library
        self.grid_generator = GridGenerator(seed=seed)
        self.transformation_sampler = TransformationSampler(primitive_library)
        self.generation_count = 0

    def generate_puzzle(self, difficulty: str = 'medium',
                       num_train_demos: int = 3,
                       max_retries: int = 5) -> Optional[GeneratedPuzzle]:
        """Generate a single puzzle.

        Args:
            difficulty: 'easy', 'medium', or 'hard'
            num_train_demos: Number of training demonstrations
            max_retries: Number of retries to avoid identity transformations

        Returns:
            GeneratedPuzzle or None if generation failed
        """
        puzzle_id = f"synthetic_{self.generation_count:06d}"
        self.generation_count += 1

        # Retry logic to avoid identity transformations
        for attempt in range(max_retries):
            # Step 1: Sample transformation
            transformation = self.transformation_sampler.sample_transformation(difficulty)

            # Step 2: Generate training demonstrations
            train_pairs = []
            has_non_trivial = False

            for _ in range(num_train_demos):
                input_grid = self.grid_generator.generate()
                output_grid = self._apply_transformation(input_grid, transformation)

                if output_grid is None:
                    break  # Transformation failed

                train_pairs.append((input_grid, output_grid))

                # Check if at least one demo is non-trivial
                if not self._grids_equal(input_grid, output_grid):
                    has_non_trivial = True

            if len(train_pairs) < num_train_demos:
                continue  # Failed to generate all demos

            if not has_non_trivial:
                continue  # All demos are trivial, retry

            # Step 3: Generate test pair
            test_input = self.grid_generator.generate()
            test_output = self._apply_transformation(test_input, transformation)

            if test_output is None:
                continue  # Transformation failed

            # Check test is non-trivial
            if self._grids_equal(test_input, test_output):
                continue  # Test is trivial, retry

            # Step 4: Create puzzle
            puzzle = GeneratedPuzzle(
                puzzle_id=puzzle_id,
                train=train_pairs,
                test=(test_input, test_output),
                transformation=transformation,
                difficulty=difficulty,
                generation_params={
                    'num_train_demos': num_train_demos,
                    'generation_method': 'primitive_composition',
                    'generation_attempt': attempt + 1
                }
            )

            return puzzle

        # Failed after all retries
        return None

    @staticmethod
    def _grids_equal(g1: Grid, g2: Grid) -> bool:
        """Check if two grids are equal."""
        if len(g1) != len(g2):
            return False
        for r1, r2 in zip(g1, g2):
            if r1 != r2:
                return False
        return True

    def _apply_transformation(self, grid: Grid, transformation: List[Tuple[str, Dict]]) -> Optional[Grid]:
        """Apply a sequence of operations to a grid.

        Returns:
            Transformed grid or None if transformation failed
        """
        current = copy.deepcopy(grid)

        for op_name, params in transformation:
            prim = self.primitives.get(op_name)
            if prim is None:
                return None

            try:
                if params:
                    current = prim.execute(current, **params)
                else:
                    current = prim.execute(current)
            except Exception:
                # Transformation failed (e.g., invalid parameters)
                return None

        return current


class PuzzleValidator:
    """Validate generated puzzles for quality."""

    @staticmethod
    def validate(puzzle: GeneratedPuzzle) -> Tuple[bool, str]:
        """Check if puzzle is valid.

        Returns:
            (is_valid, reason) tuple
        """
        # Check 1: All grids must be non-empty
        if not puzzle.train or not puzzle.test:
            return False, "Empty training or test data"

        for inp, out in puzzle.train:
            if not inp or not out:
                return False, "Empty grid in training data"
            if not inp[0] or not out[0]:
                return False, "Empty grid row"

        if not puzzle.test[0] or not puzzle.test[1]:
            return False, "Empty test grid"

        # Check 2: Non-trivial (output differs from input)
        for inp, out in puzzle.train:
            if PuzzleValidator._grids_equal(inp, out):
                return False, "Trivial transformation (identity)"

        if PuzzleValidator._grids_equal(puzzle.test[0], puzzle.test[1]):
            return False, "Trivial transformation on test"

        # Check 3: Consistent dimensions within each pair
        for inp, out in puzzle.train:
            if not inp or not out:
                return False, "Empty grid"
            if len(inp[0]) != len(inp[-1]) or len(out[0]) != len(out[-1]):
                return False, "Inconsistent row widths"

        # Check 4: At least some diversity in inputs
        if len(puzzle.train) >= 2:
            all_same = all(
                PuzzleValidator._grids_equal(puzzle.train[0][0], pair[0])
                for pair in puzzle.train[1:]
            )
            if all_same:
                return False, "All training inputs are identical"

        return True, "Valid"

    @staticmethod
    def _grids_equal(g1: Grid, g2: Grid) -> bool:
        """Check if two grids are equal."""
        if len(g1) != len(g2):
            return False
        for r1, r2 in zip(g1, g2):
            if len(r1) != len(r2):
                return False
            if r1 != r2:
                return False
        return True


class SelfPlayStatistics:
    """Track statistics during self-play generation."""

    def __init__(self):
        self.total_generated = 0
        self.valid_puzzles = 0
        self.invalid_reasons = defaultdict(int)
        self.by_difficulty = defaultdict(lambda: {'generated': 0, 'valid': 0, 'solved': 0})
        self.solve_times = []

    def record_generation(self, puzzle: Optional[GeneratedPuzzle],
                         is_valid: bool, invalid_reason: str,
                         difficulty: str):
        """Record a generation attempt."""
        self.total_generated += 1
        self.by_difficulty[difficulty]['generated'] += 1

        if is_valid and puzzle is not None:
            self.valid_puzzles += 1
            self.by_difficulty[difficulty]['valid'] += 1
        else:
            self.invalid_reasons[invalid_reason] += 1

    def record_solve_result(self, puzzle: GeneratedPuzzle, solved: bool, time: float):
        """Record solve attempt result."""
        if solved:
            self.by_difficulty[puzzle.difficulty]['solved'] += 1
            self.solve_times.append(time)

    def get_report(self) -> str:
        """Generate statistics report."""
        lines = []
        lines.append("="* 70)
        lines.append("SELF-PLAY GENERATION STATISTICS")
        lines.append("=" * 70)
        lines.append(f"\nTotal generated: {self.total_generated}")
        lines.append(f"Valid puzzles: {self.valid_puzzles} ({100*self.valid_puzzles/max(1, self.total_generated):.1f}%)")

        if self.invalid_reasons:
            lines.append(f"\nInvalid reasons:")
            for reason, count in sorted(self.invalid_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count} ({100*count/max(1, self.total_generated):.1f}%)")

        lines.append(f"\nBy difficulty:")
        for diff in ['easy', 'medium', 'hard']:
            stats = self.by_difficulty[diff]
            if stats['generated'] > 0:
                valid_pct = 100 * stats['valid'] / stats['generated']
                solve_pct = 100 * stats['solved'] / max(1, stats['valid'])
                lines.append(f"  {diff}:")
                lines.append(f"    Generated: {stats['generated']}")
                lines.append(f"    Valid: {stats['valid']} ({valid_pct:.1f}%)")
                lines.append(f"    Solved: {stats['solved']} ({solve_pct:.1f}%)")

        if self.solve_times:
            import statistics
            avg_time = statistics.mean(self.solve_times)
            lines.append(f"\nSolve times:")
            lines.append(f"  Average: {avg_time:.2f}s")
            lines.append(f"  Min: {min(self.solve_times):.2f}s")
            lines.append(f"  Max: {max(self.solve_times):.2f}s")

        lines.append("=" * 70)
        return "\n".join(lines)
