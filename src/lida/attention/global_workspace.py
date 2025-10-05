from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from ..core.events import Coalition, ConsciousContent


@dataclass
class CompetitionWeights:
    salience_w: float = 0.6
    relevance_w: float = 0.3
    novelty_w: float = 0.1


class GlobalWorkspace:
    def __init__(self, weights: CompetitionWeights | None = None) -> None:
        self.weights = weights or CompetitionWeights()

    def compete(
        self,
        coalitions: Iterable[Coalition],
        relevance: Dict[str, float] | None = None,
        novelty_fn: callable | None = None,
    ) -> Tuple[Coalition | None, List[Tuple[str, float]]]:
        scored: List[Tuple[str, float]] = []
        best: Coalition | None = None
        best_score = float("-inf")
        relmap = relevance or {}
        for c in coalitions:
            sal = float(c.salience)
            rel = float(relmap.get(c.id, 0.0))
            nov = float(novelty_fn(c.id)) if novelty_fn else 0.0
            score = self.weights.salience_w * sal + self.weights.relevance_w * rel + self.weights.novelty_w * nov
            scored.append((c.id, score))
            if score > best_score:
                best_score = score
                best = c
        return best, scored

    def broadcast(self, cycle_index: int, winning: Coalition | None) -> ConsciousContent | None:
        if winning is None:
            return None
        payload = {k: v for k, v in winning.features}
        tags = {"conscious", winning.summary}
        return ConsciousContent(cycle_index=cycle_index, tags=frozenset(tags), payload=payload)
