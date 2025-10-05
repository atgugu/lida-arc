from __future__ import annotations

from lida.memory.episodic import TransientEpisodicMemory, EpisodicMemory
from lida.core.events import ConsciousContent


def test_tem_encode_retrieve_and_decay():
    tem = TransientEpisodicMemory(decay=0.1)
    cc1 = ConsciousContent(cycle_index=1, tags=frozenset({"obj:A"}), payload={"feat:A": 1.0})
    cc2 = ConsciousContent(cycle_index=2, tags=frozenset({"obj:B"}), payload={"feat:B": 1.0})
    tem.encode(cc1)
    tem.encode(cc2)

    score_a = tem.retrieve_similarity({"feat:A": 1.0})
    score_b = tem.retrieve_similarity({"feat:B": 1.0})
    assert score_a > 0.0 and score_b > 0.0

    tem.step_decay()
    score_a2 = tem.retrieve_similarity({"feat:A": 1.0})
    assert score_a2 < score_a


def test_em_consolidation_and_retrieval():
    tem = TransientEpisodicMemory(decay=0.0)
    em = EpisodicMemory(consolidate_threshold=0.5)

    cc = ConsciousContent(cycle_index=5, tags=frozenset({"obj:A"}), payload={"feat:A": 1.0})
    tem.encode(cc, base_strength=0.6)

    em.consolidate_from(tem)
    score = em.retrieve_similarity({"feat:A": 1.0})
    assert score > 0.0
