from __future__ import annotations

from collections import deque, Counter
from dataclasses import dataclass
from typing import Deque, Dict, Iterable


@dataclass
class NoveltyConfig:
    window: int = 50
    base: float = 1.0  # baseline novelty
    k: float = 1.0     # scaling factor for penalty


class NoveltyTracker:
    """Tracks recent coalition ids and computes novelty scores.

    novelty(id) = base / (1 + k * freq_in_window)
    """

    def __init__(self, cfg: NoveltyConfig | None = None) -> None:
        self.cfg = cfg or NoveltyConfig()
        self._window: Deque[str] = deque(maxlen=self.cfg.window)
        self._counts: Counter[str] = Counter()

    def update(self, winning_id: str | None) -> None:
        if not winning_id:
            return
        # Remove oldest if full
        if len(self._window) == self._window.maxlen:
            oldest = self._window[0]
            self._counts[oldest] -= 1
            if self._counts[oldest] <= 0:
                del self._counts[oldest]
        self._window.append(winning_id)
        self._counts[winning_id] += 1

    def novelty(self, cid: str) -> float:
        freq = self._counts.get(cid, 0)
        return float(self.cfg.base) / (1.0 + float(self.cfg.k) * float(freq))
