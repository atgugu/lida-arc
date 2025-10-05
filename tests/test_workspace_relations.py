from __future__ import annotations

from lida.memory.workspace import SituationalModel


def test_relations_added_and_cleared():
    ws = SituationalModel()
    ws.upsert_object("A", features={"A": 0.9})
    ws.upsert_object("B", features={"B": 0.8})
    ws.add_relation("A", "co-occurs", "B", confidence=0.7)

    assert len(ws.relations) == 1
    assert ws.relations[0].subj == "A"
    assert ws.relations[0].obj == "B"

    ws.clear()
    assert not ws.objects and not ws.relations
