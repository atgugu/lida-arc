from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Set, Tuple

from ..core.codelet import Codelet
from ..memory.pam import PerceptualAssociativeMemory, PAMNode
from ..memory.workspace import SituationalModel


class CategoryInductionTracker:
    """Tracks co-activation of feature nodes across cycles and induces categories when thresholds are met."""

    def __init__(self) -> None:
        self.co_counts: Counter[Tuple[str, str]] = Counter()
        self.induced: Set[str] = set()

    def update_and_induce(
        self,
        pam: PerceptualAssociativeMemory,
        ws: SituationalModel,
        count_threshold: int = 5,
        min_act: float = 0.3,
        link_weight: float = 0.3,
    ) -> None:
        # Collect feature-like nodes present in this cycle's workspace above threshold
        present: Set[str] = set()
        for oid, obj in ws.objects.items():
            if max(obj.features.values()) if obj.features else 0.0 >= min_act:
                # Consider only features and categories from PAM for now
                if oid in pam.g and pam.g.nodes[oid].get("kind") in {"feature"}:
                    present.add(oid)
        # Update co-occurrence counts for unordered pairs
        for a, b in combinations(sorted(present), 2):
            key = (a, b)
            self.co_counts[key] += 1
            # Check threshold and induce category if not already
            if self.co_counts[key] >= count_threshold:
                cat_id = f"cat:{a}+{b}"
                if cat_id not in self.induced:
                    self.induced.add(cat_id)
                    # Create category node and is-a links
                    pam.ensure_node(PAMNode(cat_id, kind="category", label=cat_id))
                    pam.add_link(a, cat_id, rel="is-a", weight=link_weight)
                    pam.add_link(b, cat_id, rel="is-a", weight=link_weight)
                    # Optionally represent has-a in workspace this cycle
                    ws.add_relation(cat_id, "has-a", a, confidence=0.8)
                    ws.add_relation(cat_id, "has-a", b, confidence=0.8)


def make_category_induction_codelet(
    tracker: CategoryInductionTracker,
    pam: PerceptualAssociativeMemory,
    ws: SituationalModel,
    count_threshold: int = 5,
    min_act: float = 0.3,
    link_weight: float = 0.3,
) -> Codelet:
    def _action() -> None:
        tracker.update_and_induce(pam, ws, count_threshold=count_threshold, min_act=min_act, link_weight=link_weight)

    return Codelet(name="category_induction", urgency=0.85, action=_action, kind="perceptual")
