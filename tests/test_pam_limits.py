from __future__ import annotations

from lida.memory.pam import PerceptualAssociativeMemory, PAMNode


def test_spread_caps_and_normalization():
    pam = PerceptualAssociativeMemory()
    for n in ["A", "B", "C"]:
        pam.ensure_node(PAMNode(n, kind="feature", label=n))
    pam.add_link("A", "B", rel="assoc", weight=10.0)
    pam.add_link("B", "C", rel="assoc", weight=10.0)

    pam.set_activation("A", 1.0)
    pam.spread_activation({"A": 1.0}, iterations=2, decay=0.0, max_transfer_per_edge=0.2, max_input_per_node=0.3, normalize_after=True, max_activation=0.5)

    # Caps should keep activations bounded and normalization should clip to <= 0.5
    assert 0.0 < pam.get_activation("B") <= 0.5
    assert 0.0 < pam.get_activation("C") <= 0.5
