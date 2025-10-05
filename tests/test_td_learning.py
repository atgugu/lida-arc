from __future__ import annotations

from lida.action.learning import TDLearner, TDConfig
from lida.action.procedural import Scheme


def test_td_updates_utility_toward_reward():
    td = TDLearner(TDConfig(alpha=0.5))
    sc = Scheme(scheme_id="s", context_cues={}, action_name="a", expected_effects={}, utility=0.0)
    td.update(sc, reward=1.0)
    assert 0.4 < sc.utility < 0.6
    td.update(sc, reward=1.0)
    assert sc.utility > 0.6


def test_td_decay_reduces_utility():
    td = TDLearner()
    sc = Scheme(scheme_id="s", context_cues={}, action_name="a", expected_effects={}, utility=1.0)
    td.decay(sc, factor=0.9)
    assert 0.89 < sc.utility < 0.91
