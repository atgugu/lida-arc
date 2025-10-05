from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ForagingConfig:
    food_reward: float = 1.0
    investigate_reward: float = 0.05
    step_cost: float = -0.01


class SymbolicForagingEnv:
    """A minimal environment producing a token stream and reward based on actions.

    Tokens: "FOOD", "SHINY", others ignored.
    Actions: "forage", "investigate" (optional), anything else gets only step cost.
    Reward rules:
      - If action == forage and token == FOOD -> food_reward
      - If action == investigate and token == SHINY -> investigate_reward
      - Otherwise -> step_cost
    """

    def __init__(self, sequence: List[str], cfg: ForagingConfig | None = None) -> None:
        if not sequence:
            raise ValueError("sequence must be non-empty")
        self.sequence = list(sequence)
        self.cfg = cfg or ForagingConfig()
        self._idx = 0
        self._last_token = None

    def reset(self) -> None:
        self._idx = 0
        self._last_token = None

    def observe(self) -> str:
        token = self.sequence[self._idx % len(self.sequence)]
        self._idx += 1
        self._last_token = token
        return token

    def evaluate(self, action_name: str, token: str | None = None) -> float:
        tok = token if token is not None else self._last_token
        if tok is None:
            return 0.0
        if action_name == "forage" and tok == "FOOD":
            return self.cfg.food_reward
        if action_name == "investigate" and tok == "SHINY":
            return self.cfg.investigate_reward
        return self.cfg.step_cost
