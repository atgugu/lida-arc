from __future__ import annotations

from typing import Dict

from ..core.codelet import Codelet
from ..sensors.symbolic import SymbolStream
from ..memory.pam import PerceptualAssociativeMemory
from ..memory.workspace import SituationalModel
from .structure_building import ingest_symbol


def make_ingest_codelet(stream: SymbolStream, pam: PerceptualAssociativeMemory, ws: SituationalModel) -> Codelet:
    shared: Dict[str, Dict[str, float]] = {}

    def _action() -> None:
        sym = stream.next_symbol()
        seeds = ingest_symbol(pam, ws, sym.token)
        shared["seeds"] = seeds

    c = Codelet(name="ingest_symbol", urgency=2.0, action=_action, kind="perceptual", metadata={"shared": shared})
    return c


def make_spread_codelet(pam: PerceptualAssociativeMemory, shared: Dict[str, Dict[str, float]]) -> Codelet:
    def _action() -> None:
        seeds = shared.get("seeds", {})
        if seeds:
            pam.decay_all(0.05)
            pam.spread_activation(seeds, iterations=1, decay=0.05)

    c = Codelet(name="spread_activation", urgency=1.0, action=_action, kind="perceptual")
    return c
