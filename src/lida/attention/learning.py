from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


@dataclass
class AttentionalLearningConfig:
    alpha: float = 0.05  # learning rate for boost updates
    decay: float = 0.995  # per-cycle decay
    cap: float = 1.0  # cap on boost per feature


class AttentionalLearner:
    """Learns feature-level attention boosts from reward feedback.

    Maintains a boost map over feature keys (e.g., 'feat:food'). When reward is positive after selecting
    a coalition containing certain features, their boost is increased; otherwise slightly decayed.
    The boost can be used as a relevance add-on during Global Workspace competition.
    """

    def __init__(self, cfg: AttentionalLearningConfig | None = None) -> None:
        self.cfg = cfg or AttentionalLearningConfig()
        self._boost: Dict[str, float] = {}

    def compute_boost(self, features: Iterable[Tuple[str, float]]) -> float:
        b = 0.0
        for k, v in features:
            if not isinstance(k, str):
                continue
            b += self._boost.get(k, 0.0)
        return b

    def update(self, features: Iterable[Tuple[str, float]], reward: float) -> None:
        # Apply decay to all boosts
        for k in list(self._boost.keys()):
            self._boost[k] *= self.cfg.decay
            if abs(self._boost[k]) < 1e-8:
                del self._boost[k]
        # Positive reward strengthens boosts for present features
        if reward > 0.0:
            for k, _ in features:
                if not isinstance(k, str):
                    continue
                newv = min(self.cfg.cap, self._boost.get(k, 0.0) + self.cfg.alpha * float(reward))
                self._boost[k] = newv

    @property
    def boosts(self) -> Dict[str, float]:
        return dict(self._boost)
