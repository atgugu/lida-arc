from __future__ import annotations

from typing import Dict, List, Tuple

from ..memory.pam import PerceptualAssociativeMemory
from ..memory.workspace import SituationalModel
from .structure_building import ingest_observation


def ingest_grid_neighborhood(
    pam: PerceptualAssociativeMemory,
    ws: SituationalModel,
    grid_token_at: callable,
    x: int,
    y: int,
) -> Dict[str, float]:
    """Ingest tokens from current cell and von Neumann neighborhood with diminishing weights.

    Weights: current=1.0, neighbors=0.5
    """
    obs: List[Tuple[str, float]] = []
    obs.append((grid_token_at(x, y), 1.0))
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        obs.append((grid_token_at(x + dx, y + dy), 0.5))
    return ingest_observation(pam, ws, obs)
