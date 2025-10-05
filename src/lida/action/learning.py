from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .procedural import Scheme


@dataclass
class TDConfig:
    alpha: float = 0.1  # learning rate
    gamma: float = 0.9  # discount, for 1-step updates we use immediate reward only


class TDLearner:
    def __init__(self, cfg: TDConfig | None = None) -> None:
        self.cfg = cfg or TDConfig()

    def update(self, scheme: Scheme, reward: float, alpha: Optional[float] = None) -> None:
        # One-step TD on utility as value proxy
        u = scheme.utility
        target = reward  # no bootstrapping for now
        a = self.cfg.alpha if alpha is None else float(alpha)
        scheme.utility = u + a * (target - u)

    def decay(self, scheme: Scheme, factor: float = 0.99) -> None:
        scheme.utility *= factor
