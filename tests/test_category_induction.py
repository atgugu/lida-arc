from __future__ import annotations

from lida.memory.pam import PerceptualAssociativeMemory, PAMNode
from lida.memory.workspace import SituationalModel
from lida.perception.category_induction import CategoryInductionTracker, make_category_induction_codelet


def test_category_induction_creates_cat_after_threshold():
    pam = PerceptualAssociativeMemory()
    for node in [
        PAMNode("feat:A", kind="feature", label="A"),
        PAMNode("feat:B", kind="feature", label="B"),
    ]:
        pam.ensure_node(node)
    ws = SituationalModel()

    tracker = CategoryInductionTracker()
    induce = make_category_induction_codelet(tracker, pam, ws, count_threshold=3, min_act=0.5, link_weight=0.4)

    # Simulate 3 cycles where A and B co-occur strongly
    for _ in range(3):
        ws.clear()
        ws.upsert_object("feat:A", features={"feat:A": 0.8})
        ws.upsert_object("feat:B", features={"feat:B": 0.7})
        induce.run()

    cat_id = "cat:feat:A+feat:B"
    assert cat_id in pam.g
    # Both is-a links should exist with weight >= 0.4
    assert pam.g.has_edge("feat:A", cat_id)
    assert pam.g.has_edge("feat:B", cat_id)
    assert pam.g.get_edge_data("feat:A", cat_id)["weight"] >= 0.4
    assert pam.g.get_edge_data("feat:B", cat_id)["weight"] >= 0.4
