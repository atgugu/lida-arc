from __future__ import annotations

from lida.action.behavior_net import BehaviorNet, BNConfig
from lida.action.procedural import Scheme, InstantiatedBehavior


def test_cooldown_penalizes_reselection():
    bn = BehaviorNet(BNConfig(precondition_weight=1.0, utility_weight=1.0, cooldown_steps=2, cooldown_penalty=1.0))
    s1 = Scheme("s1", {}, "a", {}, utility=1.0)
    s2 = Scheme("s2", {}, "b", {}, utility=0.9)
    cands = [InstantiatedBehavior(s1, precondition_score=0.0), InstantiatedBehavior(s2, precondition_score=0.0)]
    w1 = bn.select(cands)
    assert w1 and w1.scheme.scheme_id == "s1"
    bn.on_selected("s1")
    # Next selection should favor s2 due to cooldown penalty
    w2 = bn.select(cands)
    assert w2 and w2.scheme.scheme_id == "s2"
    bn.tick()
    bn.tick()
    # Cooldown expired, s1 should win again
    w3 = bn.select(cands)
    assert w3 and w3.scheme.scheme_id == "s1"
