from __future__ import annotations

from lida.sensors.symbolic import SymbolStream
from lida.memory.pam import PerceptualAssociativeMemory
from lida.memory.workspace import SituationalModel
from lida.perception.codelets import make_ingest_codelet, make_spread_codelet


def test_ingest_and_spread_update_workspace_and_pam():
    pam = PerceptualAssociativeMemory()
    ws = SituationalModel()
    stream = SymbolStream(["FOOD"])  # constant
    ingest = make_ingest_codelet(stream, pam, ws)
    shared = ingest.metadata["shared"]
    spread = make_spread_codelet(pam, shared)

    ingest.run()
    spread.run()

    # After one ingest and spread, there should be an object and activated feature
    assert ws.objects
    top = pam.top_k_active(1)
    assert top and top[0][1] > 0.0
