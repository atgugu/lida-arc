from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GridworldConfig:
    width: int = 7
    height: int = 7
    food_positions: Tuple[Tuple[int, int], ...] = ((5, 5), (1, 5))
    shiny_positions: Tuple[Tuple[int, int], ...] = ((3, 3),)
    hazard_positions: Tuple[Tuple[int, int], ...] = ((2, 4), (4, 2))
    step_cost: float = -0.01
    food_reward: float = 1.0
    hazard_penalty: float = -0.2
    safety_bonus: float = 0.005


class Gridworld:
    """Minimal gridworld with food, shiny distractor, and hazards.

    Observations are symbolic tokens: one of {FOOD, SHINY, HAZARD, EMPTY} based on the cell the agent is facing/standing.
    Actions: up, down, left, right, forage, investigate.
    Rewards:
      - moving costs step_cost
      - forage on food gives food_reward
      - stepping on hazard yields hazard_penalty
      - investigate on shiny yields small bonus (0.05)
      - standing on safe tile yields small safety_bonus
    """

    def __init__(self, cfg: GridworldConfig | None = None) -> None:
        self.cfg = cfg or GridworldConfig()
        self.x = 1
        self.y = 1

    def reset(self) -> None:
        self.x, self.y = 1, 1

    def _tile_token(self, x: int, y: int) -> str:
        if (x, y) in self.cfg.food_positions:
            return "FOOD"
        if (x, y) in self.cfg.shiny_positions:
            return "SHINY"
        if (x, y) in self.cfg.hazard_positions:
            return "HAZARD"
        return "EMPTY"

    def observe(self) -> str:
        return self._tile_token(self.x, self.y)

    def step(self, action: str) -> float:
        reward = 0.0
        if action == "up":
            self.y = max(0, self.y - 1)
            reward += self.cfg.step_cost
        elif action == "down":
            self.y = min(self.cfg.height - 1, self.y + 1)
            reward += self.cfg.step_cost
        elif action == "left":
            self.x = max(0, self.x - 1)
            reward += self.cfg.step_cost
        elif action == "right":
            self.x = min(self.cfg.width - 1, self.x + 1)
            reward += self.cfg.step_cost
        elif action == "forage":
            if self._tile_token(self.x, self.y) == "FOOD":
                reward += self.cfg.food_reward
        elif action == "investigate":
            if self._tile_token(self.x, self.y) == "SHINY":
                reward += 0.05
        # Hazard penalty or safety bonus
        if self._tile_token(self.x, self.y) == "HAZARD":
            reward += self.cfg.hazard_penalty
        else:
            reward += self.cfg.safety_bonus
        return reward
