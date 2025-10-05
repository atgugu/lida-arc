from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..memory.pam import PerceptualAssociativeMemory, PAMNode
from ..memory.workspace import SituationalModel
from .catalog import CATALOG


def ingest_symbol(pam: PerceptualAssociativeMemory, ws: SituationalModel, token: str, weight: float = 1.0) -> Dict[str, float]:
    """Map a token to PAM activations and an object in workspace.

    Returns a dict of seed activations for spreading (scaled by weight).
    """
    spec = CATALOG.get(token)
    seeds: Dict[str, float] = {}
    if not spec:
        return seeds
    # Ensure nodes present
    pam.ensure_node(PAMNode(node_id=spec.feature_node, kind="feature", label=spec.token))
    for cat in spec.categories:
        pam.ensure_node(PAMNode(node_id=cat, kind="category", label=cat))
        pam.add_link(spec.feature_node, cat, rel="is-a", weight=0.5)
    # Seed feature with weight
    seeds[spec.feature_node] = 1.0 * float(weight)
    # Update workspace with an object for this percept
    ws.upsert_object(spec.feature_node, features={spec.feature_node: float(weight)})
    return seeds


def ingest_observation(
    pam: PerceptualAssociativeMemory,
    ws: SituationalModel,
    observed: Iterable[Tuple[str, float]],
) -> Dict[str, float]:
    """Ingest a collection of (token, weight) observations and aggregate seed activations."""
    agg: Dict[str, float] = {}
    for tok, w in observed:
        seeds = ingest_symbol(pam, ws, tok, weight=w)
        for k, v in seeds.items():
            agg[k] = agg.get(k, 0.0) + v
    return agg
