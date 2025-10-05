from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from ..core.events import ConsciousContent


def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(float(a.get(k, 0.0)) * float(v) for k, v in b.items())


def _norm(a: Dict[str, float]) -> float:
    return sum(float(v) ** 2 for v in a.values()) ** 0.5


def _cosine(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-9) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na < eps or nb < eps:
        return 0.0
    return _dot(a, b) / (na * nb)


@dataclass
class Episode:
    cycle_index: int
    payload: Dict[str, float]
    tags: Tuple[str, ...]
    strength: float = 1.0


class TransientEpisodicMemory:
    """Simple transient episodic store with decay and similarity-based retrieval."""

    def __init__(self, decay: float = 0.01):
        self.decay = float(decay)
        self._episodes: List[Episode] = []

    def encode(self, cc: ConsciousContent, base_strength: float = 1.0) -> None:
        self._episodes.append(
            Episode(cycle_index=cc.cycle_index, payload=dict(cc.payload), tags=tuple(sorted(cc.tags)), strength=float(base_strength))
        )

    def step_decay(self) -> None:
        for ep in self._episodes:
            ep.strength = max(0.0, ep.strength * (1.0 - self.decay))
        # prune very weak episodes
        self._episodes = [ep for ep in self._episodes if ep.strength > 1e-6]

    def retrieve_similarity(self, cues: Dict[str, float], top_n: int = 3) -> float:
        """Return an aggregated similarity score against top-N matching episodes, weighted by strength."""
        scored: List[Tuple[float, Episode]] = []
        for ep in self._episodes:
            sim = _cosine(cues, ep.payload)
            scored.append((sim * ep.strength, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        total = 0.0
        for s, _ in scored[: top_n if top_n > 0 else None]:
            total += s
        return total


class EpisodicMemory:
    """Placeholder for long-term episodic memory. For now, acts as a stable mirror of strong transient episodes."""

    def __init__(self, consolidate_threshold: float = 0.7):
        self.consolidate_threshold = float(consolidate_threshold)
        self._episodes: List[Episode] = []

    def consolidate_from(self, tem: TransientEpisodicMemory) -> None:
        for ep in tem._episodes:
            if ep.strength >= self.consolidate_threshold:
                # Avoid duplicates by (cycle,tags) heuristic
                key = (ep.cycle_index, ep.tags)
                if not any((e.cycle_index, e.tags) == key for e in self._episodes):
                    self._episodes.append(Episode(cycle_index=ep.cycle_index, payload=dict(ep.payload), tags=ep.tags, strength=ep.strength))

    def retrieve_similarity(self, cues: Dict[str, float], top_n: int = 3) -> float:
        scored: List[Tuple[float, Episode]] = []
        for ep in self._episodes:
            sim = _cosine(cues, ep.payload)
            scored.append((sim * ep.strength, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        total = 0.0
        for s, _ in scored[: top_n if top_n > 0 else None]:
            total += s
        return total
