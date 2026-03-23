"""Quality checks for generated puzzles."""

from typing import List, Tuple

Grid = List[List[int]]


class PuzzleValidator:
    """Validate that a generated puzzle meets quality criteria.

    Checks performed:
    1. All grids are non-empty and well-formed.
    2. Outputs differ from inputs (non-trivial transformation).
    3. Training inputs are not all identical (diversity).
    """

    @staticmethod
    def validate(train: List[Tuple[Grid, Grid]], test: Tuple[Grid, Grid]) -> Tuple[bool, str]:
        if not train or not test:
            return False, "empty training or test data"

        for inp, out in train:
            if not inp or not out or not inp[0] or not out[0]:
                return False, "empty grid"
            if _grids_equal(inp, out):
                return False, "trivial transformation (identity)"
            if len(inp[0]) != len(inp[-1]) or len(out[0]) != len(out[-1]):
                return False, "inconsistent row widths"

        if not test[0] or not test[1]:
            return False, "empty test grid"
        if _grids_equal(test[0], test[1]):
            return False, "trivial transformation on test"

        if len(train) >= 2 and all(_grids_equal(train[0][0], p[0]) for p in train[1:]):
            return False, "all training inputs identical"

        return True, "valid"


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))
