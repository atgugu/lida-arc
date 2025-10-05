from __future__ import annotations

from lida.memory.pam import PerceptualAssociativeMemory
from lida.memory.workspace import SituationalModel
from lida.perception.structure_building import ingest_symbol
from lida.perception.category_promotion import make_category_promotion_codelet


def test_category_promotion_strengthens_is_a_and_activates_category():
    pam = PerceptualAssociativeMemory()
    ws = SituationalModel()
    # Ingest a FOOD token to create feature and category link (feat:food -> cat:resource)
    seeds = ingest_symbol(pam, ws, "FOOD")
    # Simulate that the feature is active in the workspace for this cycle
    ws.upsert_object("feat:food", features={"feat:food": 1.0})

    # Find initial weight of is-a link
    initial_w = None
    for _s, _d, ed in pam.g.out_edges("feat:food", data=True):
        if ed.get("rel") == "is-a":
            initial_w = ed.get("weight")
            break
    assert initial_w is not None and initial_w > 0

    promote = make_category_promotion_codelet(pam, ws, alpha=0.1, gamma=0.5)
    promote.run()

    # Weight should have increased; category node should have non-zero activation
    new_w = None
    cat_id = None
    for _s, d, ed in pam.g.out_edges("feat:food", data=True):
        if ed.get("rel") == "is-a":
            new_w = ed.get("weight")
            cat_id = d
            break
    assert new_w is not None and new_w > initial_w
    assert cat_id is not None and pam.get_activation(cat_id) > 0.0
