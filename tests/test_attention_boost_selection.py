from __future__ import annotations

from lida.attention.global_workspace import GlobalWorkspace, CompetitionWeights
from lida.core.events import Coalition
from lida.attention.learning import AttentionalLearner, AttentionalLearningConfig


def test_attention_boost_sways_selection():
    gw = GlobalWorkspace(CompetitionWeights(salience_w=1.0, relevance_w=1.0, novelty_w=0.0))
    al = AttentionalLearner(AttentionalLearningConfig(alpha=0.5, decay=1.0, cap=10.0))
    # Two coalitions with equal salience
    food = Coalition(id="feat:food", salience=0.5, features=frozenset({("feat:food", 1.0)}), summary="food")
    shiny = Coalition(id="feat:shiny", salience=0.5, features=frozenset({("feat:shiny", 1.0)}), summary="shiny")
    # Apply reward to food features
    al.update(food.features, reward=1.0)
    relevance = {
        "feat:food": al.compute_boost(food.features),
        "feat:shiny": al.compute_boost(shiny.features),
    }
    winner, scores = gw.compete([food, shiny], relevance=relevance)
    assert winner and winner.id == "feat:food"
