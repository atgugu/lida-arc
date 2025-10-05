from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TokenSpec:
    token: str
    feature_node: str  # PAM node id, e.g., "feat:food"
    categories: List[str]  # PAM category node ids, e.g., ["cat:resource"]


# A small built-in catalog for the symbolic demo
CATALOG: Dict[str, TokenSpec] = {
    "FOOD": TokenSpec(token="FOOD", feature_node="feat:food", categories=["cat:resource"]),
    "SHINY": TokenSpec(token="SHINY", feature_node="feat:shiny", categories=["cat:distractor"]),
    "HAZARD": TokenSpec(token="HAZARD", feature_node="feat:hazard", categories=["cat:danger"]),
}
