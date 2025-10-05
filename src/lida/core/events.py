from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class ConsciousContent:
    """Immutable payload broadcast to all modules during the attention phase."""

    cycle_index: int
    tags: FrozenSet[str]
    payload: Dict[str, Any]


@dataclass(frozen=True)
class Coalition:
    """A coalition competing for access to the global workspace."""

    id: str
    salience: float
    features: FrozenSet[Tuple[str, Any]]
    summary: str
