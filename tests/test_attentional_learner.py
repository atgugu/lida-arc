from __future__ import annotations

from lida.attention.learning import AttentionalLearner, AttentionalLearningConfig


def test_attentional_boost_increases_after_reward():
    al = AttentionalLearner(AttentionalLearningConfig(alpha=0.1, decay=1.0, cap=1.0))
    feats = [("feat:food", 1.0)]
    before = al.compute_boost(feats)
    assert before == 0.0
    al.update(feats, reward=1.0)
    after = al.compute_boost(feats)
    assert after > before
