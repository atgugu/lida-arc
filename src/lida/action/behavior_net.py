from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict

from .procedural import InstantiatedBehavior


@dataclass
class BNConfig:
    precondition_weight: float = 1.0
    utility_weight: float = 1.0
    cooldown_steps: int = 0  # steps to penalize re-selection of same scheme
    cooldown_penalty: float = 0.0  # value subtracted if on cooldown


class BehaviorNet:
    """Behavior net selecting behavior by weighted sum of preconditions and learned utility with optional cooldown."""

    def __init__(self, cfg: BNConfig | None = None) -> None:
        self.cfg = cfg or BNConfig()
        self._cooldowns: Dict[str, int] = {}

    def tick(self) -> None:
        if not self._cooldowns:
            return
        for k in list(self._cooldowns.keys()):
            self._cooldowns[k] -= 1
            if self._cooldowns[k] <= 0:
                del self._cooldowns[k]

    def on_selected(self, scheme_id: str) -> None:
        if self.cfg.cooldown_steps > 0:
            self._cooldowns[scheme_id] = self.cfg.cooldown_steps

    def select(self, candidates: Iterable[InstantiatedBehavior]) -> Optional[InstantiatedBehavior]:
        best = None
        best_val = float("-inf")
        for ib in candidates:
            val = self.cfg.precondition_weight * ib.precondition_score + self.cfg.utility_weight * ib.scheme.utility
            if self._cooldowns.get(ib.scheme.scheme_id, 0) > 0:
                val -= self.cfg.cooldown_penalty
            if val > best_val:
                best_val = val
                best = ib
        return best
