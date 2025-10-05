from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


@dataclass
class PAMNode:
    node_id: str
    kind: str  # feature | category | relation | feeling | concept
    label: str
    base_strength: float = 1.0
    activation: float = 0.0


class PerceptualAssociativeMemory:
    """Slipnet-like semantic network with spreading activation.

    Nodes and edges carry weights. Activation spreads for a bounded number of iterations per cycle.
    """

    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def ensure_node(self, node: PAMNode) -> None:
        if node.node_id not in self.g:
            self.g.add_node(
                node.node_id,
                kind=node.kind,
                label=node.label,
                base_strength=float(node.base_strength),
                activation=float(node.activation),
            )
        else:
            # Update metadata but preserve activation
            self.g.nodes[node.node_id]["kind"] = node.kind
            self.g.nodes[node.node_id]["label"] = node.label
            self.g.nodes[node.node_id]["base_strength"] = float(node.base_strength)

    def add_link(self, src: str, dst: str, rel: str, weight: float = 1.0) -> None:
        self.g.add_edge(src, dst, rel=rel, weight=float(weight))

    def set_activation(self, node_id: str, value: float) -> None:
        if node_id in self.g:
            self.g.nodes[node_id]["activation"] = float(value)

    def get_activation(self, node_id: str) -> float:
        return float(self.g.nodes[node_id].get("activation", 0.0))

    def decay_all(self, decay: float = 0.05) -> None:
        for n in self.g.nodes:
            a = float(self.g.nodes[n].get("activation", 0.0))
            a = max(0.0, a * (1.0 - decay))
            self.g.nodes[n]["activation"] = a

    def normalize_activations(self, max_activation: float = 1.0) -> None:
        """Clip activations to [0, max_activation]."""
        for n in self.g.nodes:
            a = float(self.g.nodes[n].get("activation", 0.0))
            if a < 0.0:
                a = 0.0
            if a > max_activation:
                a = max_activation
            self.g.nodes[n]["activation"] = a

    def spread_activation(
        self,
        seeds: Dict[str, float],
        iterations: int = 2,
        decay: float = 0.05,
        cap: Optional[int] = 10_000,
        max_transfer_per_edge: Optional[float] = None,
        max_input_per_node: Optional[float] = None,
        normalize_after: bool = True,
        max_activation: float = 1.0,
    ) -> None:
        """Spread activation from seeds across weighted edges for bounded iterations.

        Simple synchronous update rule per iteration:
          new_a[j] += sum_i( a[i] * w(i,j) ) ; then decay and clip.
        Caps can be applied to transfers and per-node inputs; normalization can clip activations.
        """
        # Initialize seeds
        for nid, val in seeds.items():
            if nid in self.g:
                self.g.nodes[nid]["activation"] = max(self.get_activation(nid), float(val))
        # Iterative spread
        for _ in range(max(0, iterations)):
            if cap is not None and self.g.number_of_nodes() > cap:
                break
            updates: Dict[str, float] = {}
            for i, j, edata in self.g.edges(data=True):
                ai = self.get_activation(i)
                if ai <= 0.0:
                    continue
                w = float(edata.get("weight", 0.0))
                if w <= 0.0:
                    continue
                delta = ai * w
                if max_transfer_per_edge is not None:
                    delta = min(delta, max_transfer_per_edge)
                updates[j] = updates.get(j, 0.0) + delta
            # Apply per-node cap
            if max_input_per_node is not None:
                for n in list(updates.keys()):
                    updates[n] = min(updates[n], max_input_per_node)
            # Apply updates with decay
            for n, delta in updates.items():
                a = self.get_activation(n)
                a = (a + delta) * (1.0 - decay)
                # clip tiny
                if a < 1e-6:
                    a = 0.0
                self.set_activation(n, a)
        if normalize_after:
            self.normalize_activations(max_activation=max_activation)

    def top_k_active(self, k: int = 10, kind_filter: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        nodes = list(self.g.nodes())
        if kind_filter is not None:
            nodes = [n for n in nodes if self.g.nodes[n].get("kind") in set(kind_filter)]
        scored = [(n, float(self.g.nodes[n].get("activation", 0.0))) for n in nodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def ensure_feeling(self, label: str) -> str:
        node_id = f"feeling:{label}"
        self.ensure_node(PAMNode(node_id=node_id, kind="feeling", label=label, base_strength=1.0, activation=0.0))
        return node_id

    def feelings_from_map(self, feelings: Dict[str, float]) -> None:
        for label, act in feelings.items():
            nid = self.ensure_feeling(label)
            self.set_activation(nid, act)
