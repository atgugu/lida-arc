from __future__ import annotations

from typing import List

from ..core.codelet import Codelet
from ..core.events import Coalition
from ..memory.workspace import SituationalModel


def make_attention_seed_codelets(ws: SituationalModel) -> List[Codelet]:
    """Create attention codelets that propose coalitions from workspace objects and relations.

    Each codelet will append coalition proposals into a shared list located in metadata under 'coalitions'.
    A later stage (global workspace) will read them and run competition.
    """

    proposals: list[Coalition] = []

    def _from_objects() -> None:
        # Clear proposals each cycle to avoid unbounded growth
        proposals.clear()
        for oid, obj in ws.objects.items():
            sal = max(obj.features.values()) if obj.features else 0.0
            feats = frozenset((k, v) for k, v in obj.features.items())
            proposals.append(Coalition(id=oid, salience=sal, features=feats, summary=f"obj:{oid}"))

    def _from_relations() -> None:
        for i, r in enumerate(ws.relations):
            sal = r.confidence * 1.1
            feats = frozenset({(f"rel:{r.rel}", sal), (f"subj:{r.subj}", 1.0), (f"obj:{r.obj}", 1.0)})
            proposals.append(Coalition(id=f"rel:{i}", salience=sal, features=feats, summary=f"rel:{r.subj}-{r.rel}-{r.obj}"))

    return [
        Codelet(name="attn_from_objects", urgency=1.0, action=_from_objects, kind="attention", metadata={"coalitions": proposals}),
        Codelet(name="attn_from_relations", urgency=0.9, action=_from_relations, kind="attention", metadata={"coalitions": proposals}),
    ]