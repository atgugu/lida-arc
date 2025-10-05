from __future__ import annotations

import asyncio
import pytest

from lida.core.cycle import CognitiveCycleEngine, CycleConfig


@pytest.mark.asyncio
async def test_cycle_runs_3_loops(tmp_path):
    logs = []

    def log_fn(m: str) -> None:
        logs.append(m)

    engine = CognitiveCycleEngine(CycleConfig(hz=8.0, log_fn=log_fn, trace_dir=tmp_path))

    async def u():
        return None

    async def a():
        return None

    async def x():
        return None

    engine.on_understanding(u)
    engine.on_attention(a)
    engine.on_action(x)

    await engine.run(max_cycles=3)

    assert any("cycle idx=1" in m for m in logs)
    assert any("cycle idx=2" in m for m in logs)
    assert any("cycle idx=3" in m for m in logs)
