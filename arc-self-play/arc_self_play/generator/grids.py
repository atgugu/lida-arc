"""Random grid generators for puzzle inputs."""

import random
from typing import List, Optional

Grid = List[List[int]]


class GridGenerator:
    """Generate random grids with configurable density and structure.

    Supports three generation strategies:
    - sparse: 15-25 % filled, models typical ARC inputs
    - dense:  60-80 % filled, for stress-testing transforms
    - structured: geometric shapes (rectangles)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        style: str = "random",
    ) -> Grid:
        """Return a new random grid.

        Args:
            height: Grid height (random 3-10 if None).
            width:  Grid width  (random 3-10 if None).
            style:  "sparse", "dense", "structured", or "random".
        """
        h = height or random.randint(3, 10)
        w = width or random.randint(3, 10)

        if style == "random":
            style = random.choice(["sparse", "dense", "structured"])

        builders = {
            "sparse": self._sparse,
            "dense": self._dense,
            "structured": self._structured,
        }
        return builders[style](h, w)

    # ------------------------------------------------------------------

    @staticmethod
    def _sparse(h: int, w: int) -> Grid:
        grid = [[0] * w for _ in range(h)]
        n = max(3, int(h * w * random.uniform(0.15, 0.25)))
        for _ in range(n):
            grid[random.randint(0, h - 1)][random.randint(0, w - 1)] = random.randint(1, 9)
        return grid

    @staticmethod
    def _dense(h: int, w: int) -> Grid:
        grid = [[0] * w for _ in range(h)]
        n = int(h * w * random.uniform(0.6, 0.8))
        for _ in range(n):
            grid[random.randint(0, h - 1)][random.randint(0, w - 1)] = random.randint(1, 9)
        return grid

    @staticmethod
    def _structured(h: int, w: int) -> Grid:
        grid = [[0] * w for _ in range(h)]
        for _ in range(random.randint(1, 3)):
            if h <= 1 or w <= 1:
                break
            r1 = random.randint(0, h - 2)
            c1 = random.randint(0, w - 2)
            r2 = random.randint(r1 + 1, h)
            c2 = random.randint(c1 + 1, w)
            color = random.randint(1, 9)
            for r in range(r1, r2):
                for c in range(c1, c2):
                    grid[r][c] = color
        return grid
