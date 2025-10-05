from __future__ import annotations

from lida.core.events import Coalition
from lida.attention.global_workspace import GlobalWorkspace, CompetitionWeights


def test_competition_uses_salience_relevance_novelty():
    gw = GlobalWorkspace(CompetitionWeights(salience_w=0.5, relevance_w=0.3, novelty_w=0.2))
    coals = [
        Coalition(id="A", salience=0.5, features=frozenset(), summary="A"),
        Coalition(id="B", salience=0.4, features=frozenset(), summary="B"),
    ]
    relevance = {"B": 1.0}  # B is more relevant
    novelty = {"A": 0.1, "B": 0.9}
    winner, scores = gw.compete(coals, relevance=relevance, novelty_fn=lambda cid: novelty.get(cid, 0.0))
    assert winner and winner.id == "B"
