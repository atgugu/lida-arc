"""Difficulty-aware transformation sampling."""

import random
from typing import Dict, List, Tuple

from ..primitives import PrimitiveLibrary


class TransformationSampler:
    """Sample transformation sequences at controlled difficulty levels.

    Difficulty controls two axes:
    - **depth**: how many primitives are composed (1 for easy, up to 5 for hard).
    - **parametric ratio**: chance of including operations that need random
      parameters (recolor, tile), which are harder for solvers.
    """

    SIMPLE_OPS = [
        "rotate_90", "rotate_180", "rotate_270",
        "reflect_horizontal", "reflect_vertical", "reflect_diagonal",
        "auto_crop",
    ]

    PARAMETRIC_OPS = [
        "recolor",
        "tile",
    ]

    def __init__(self, library: PrimitiveLibrary) -> None:
        self.library = library

    def sample(self, difficulty: str = "medium") -> List[Tuple[str, Dict]]:
        """Return a random transformation sequence.

        Args:
            difficulty: "easy" | "medium" | "hard"

        Returns:
            List of (operation_name, parameters) tuples.
        """
        num_ops, parametric_prob = {
            "easy":   (1,                     0.0),
            "medium": (random.randint(2, 3),  0.3),
            "hard":   (random.randint(3, 5),  0.6),
        }[difficulty]

        use_parametric = random.random() < parametric_prob
        ops: List[Tuple[str, Dict]] = []

        for _ in range(num_ops):
            if use_parametric and random.random() < 0.4:
                name = random.choice(self.PARAMETRIC_OPS)
            else:
                name = random.choice(self.SIMPLE_OPS)
            ops.append((name, self._make_params(name)))

        return ops

    # ------------------------------------------------------------------

    @staticmethod
    def _make_params(op: str) -> Dict:
        if op == "recolor":
            colors = list(range(1, 10))
            random.shuffle(colors)
            cmap = {i: colors[i - 1] for i in range(1, 10)}
            cmap[0] = 0
            return {"color_map": cmap}

        if op == "tile":
            return {
                "repeat_v": random.randint(2, 3),
                "repeat_h": random.randint(2, 3),
            }

        return {}
