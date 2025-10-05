from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Iterable


@dataclass
class Scheme:
    scheme_id: str
    context_cues: Dict[str, float]  # features from conscious content or workspace
    action_name: str
    expected_effects: Dict[str, float]
    utility: float = 0.0


@dataclass
class InstantiatedBehavior:
    scheme: Scheme
    arguments: Dict[str, float] = field(default_factory=dict)
    precondition_score: float = 0.0


class ProceduralMemory:
    def __init__(self) -> None:
        self._schemes: Dict[str, Scheme] = {}

    def upsert_scheme(self, scheme: Scheme) -> None:
        self._schemes[scheme.scheme_id] = scheme

    def get_scheme(self, scheme_id: str) -> Optional[Scheme]:
        return self._schemes.get(scheme_id)

    def iter_schemes(self) -> Iterable[Scheme]:
        return self._schemes.values()

    def instantiate(self, conscious_payload: Dict[str, float]) -> List[InstantiatedBehavior]:
        out: List[InstantiatedBehavior] = []
        for sc in self._schemes.values():
            score = 0.0
            for k, w in sc.context_cues.items():
                score += float(conscious_payload.get(k, 0.0)) * float(w)
            if score > 0.0:
                out.append(InstantiatedBehavior(scheme=sc, arguments={}, precondition_score=score))
        return out
