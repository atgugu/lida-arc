from __future__ import annotations

from lida.memory.pam import PerceptualAssociativeMemory, PAMNode


def test_spreading_activation_and_decay():
    pam = PerceptualAssociativeMemory()
    pam.ensure_node(PAMNode("A", kind="feature", label="A"))
    pam.ensure_node(PAMNode("B", kind="feature", label="B"))
    pam.add_link("A", "B", rel="assoc", weight=0.5)
    pam.set_activation("A", 1.0)
    pam.decay_all(0.1)
    assert 0.8 < pam.get_activation("A") <= 0.9
    pam.spread_activation({"A": 1.0}, iterations=1, decay=0.1)
    # Some activation should reach B
    assert pam.get_activation("B") > 0.0
    top = pam.top_k_active(2)
    assert top[0][0] in {"A", "B"}
