from __future__ import annotations

from typing import Iterable

from ..core.codelet import Codelet
from ..memory.workspace import SituationalModel


def make_relations_codelet(ws: SituationalModel) -> Codelet:
    """Create a codelet that builds simple co-occurrence relations among objects in the workspace.

    For any pair of objects present in the workspace for this cycle, add a relation subj --co-occurs--> obj
    with confidence proportional to the product of their strongest feature activations.
    """

    def _action() -> None:
        # Collect strongest activation per object
        objs = list(ws.objects.items())
        strengths = []
        for oid, obj in objs:
            s = max(obj.features.values()) if obj.features else 0.0
            strengths.append((oid, s))
        # Pairwise co-occurrence
        n = len(strengths)
        for i in range(n):
            for j in range(i + 1, n):
                oi, si = strengths[i]
                oj, sj = strengths[j]
                conf = (si * sj) ** 0.5  # geometric mean to temper extremes
                if conf > 0.0:
                    ws.add_relation(oi, "co-occurs", oj, confidence=conf)

    return Codelet(name="relations_builder", urgency=0.8, action=_action, kind="structure")
