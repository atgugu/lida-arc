from __future__ import annotations

import pytest

from lida.core.cycle import CognitiveCycleEngine, CycleConfig
from lida.core.events import Coalition
from lida.attention.global_workspace import GlobalWorkspace


@pytest.mark.asyncio
async def test_single_winner_per_cycle():
    gw = GlobalWorkspace()
    engine = CognitiveCycleEngine(CycleConfig(hz=50.0))

    winners = []

    async def understanding():
        return None

    async def attention():
        coals = [
            Coalition(id="A", salience=0.6, features=frozenset(), summary="A"),
            Coalition(id="B", salience=0.4, features=frozenset(), summary="B"),
        ]
        w, _ = gw.compete(coals)
        winners.append(w.id if w else None)

    async def action():
        return None

    engine.on_understanding(understanding)
    engine.on_attention(attention)
    engine.on_action(action)

    await engine.run(max_cycles=5)

    # Exactly one winner id recorded per cycle
    assert len(winners) == 5
    assert all(isinstance(w, str) for w in winners)
