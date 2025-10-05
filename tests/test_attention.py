from __future__ import annotations

from lida.core.events import Coalition
from lida.attention.global_workspace import GlobalWorkspace


def test_global_workspace_picks_highest_salience():
    gw = GlobalWorkspace()
    coalitions = [
        Coalition(id="c1", salience=0.2, features=frozenset(), summary="c1"),
        Coalition(id="c2", salience=0.8, features=frozenset(), summary="c2"),
        Coalition(id="c3", salience=0.5, features=frozenset(), summary="c3"),
    ]
    winner, scores = gw.compete(coalitions)
    assert winner and winner.id == "c2"
