from __future__ import annotations

from ..core.codelet import Codelet
from ..memory.pam import PerceptualAssociativeMemory
from ..memory.workspace import SituationalModel


def make_category_promotion_codelet(
    pam: PerceptualAssociativeMemory, ws: SituationalModel, alpha: float = 0.05, gamma: float = 0.5, wmax: float = 2.0
) -> Codelet:
    """Promote category links from active features (Hebbian-like strengthening) and activate category nodes.

    - For each object in workspace interpreted as a feature node id, strengthen outgoing 'is-a' links.
    - Increase the activation of category nodes based on feature activation.
    """

    def _action() -> None:
        for oid, obj in ws.objects.items():
            # strongest feature on the object, default 0
            if not obj.features:
                continue
            feat_act = max(obj.features.values())
            # iterate over outgoing edges of this node with rel 'is-a'
            if oid not in pam.g:
                continue
            for _src, dst, edata in pam.g.out_edges(oid, data=True):
                if edata.get("rel") != "is-a":
                    continue
                w = float(edata.get("weight", 0.0))
                new_w = min(wmax, w + alpha * feat_act)
                edata["weight"] = new_w
                # Also activate the category node
                cat_act = pam.get_activation(dst)
                pam.set_activation(dst, max(cat_act, gamma * feat_act))

    return Codelet(name="category_promotion", urgency=0.9, action=_action, kind="perceptual")
