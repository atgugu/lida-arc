from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable, Optional, Dict, Any

from ..tools.inspection import TraceWriter, CycleMetrics


PhaseHook = Callable[[], Awaitable[None]]


@dataclass
class CycleConfig:
    hz: float = 8.0
    understanding_budget_ms: int = 80
    attention_budget_ms: int = 60
    action_budget_ms: int = 40
    log_fn: Optional[Callable[[str], None]] = None
    trace_dir: Optional[Path] = None


class CognitiveCycleEngine:
    def __init__(self, config: CycleConfig):
        self.config = config
        self._understanding: PhaseHook | None = None
        self._attention: PhaseHook | None = None
        self._action: PhaseHook | None = None
        self._running = False
        self._cycle_index = 0
        self._trace: TraceWriter | None = None
        if self.config.trace_dir is not None:
            self._trace = TraceWriter(self.config.trace_dir)

    def on_understanding(self, hook: PhaseHook) -> None:
        self._understanding = hook

    def on_attention(self, hook: PhaseHook) -> None:
        self._attention = hook

    def on_action(self, hook: PhaseHook) -> None:
        self._action = hook

    async def _run_phase(self, name: str, hook: Optional[PhaseHook], budget_ms: int) -> float:
        start = time.perf_counter()
        if hook is not None:
            try:
                await asyncio.wait_for(hook(), timeout=budget_ms / 1000.0)
            except asyncio.TimeoutError:
                self._log(f"phase-timeout name={name} budget_ms={budget_ms}")
            except Exception as e:  # noqa: BLE001
                self._log(f"phase-error name={name} error={e}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > budget_ms:
            self._log(f"phase-overrun name={name} elapsed_ms={elapsed_ms:.1f} budget_ms={budget_ms}")
        return elapsed_ms

    def _log(self, msg: str) -> None:
        if self.config.log_fn:
            self.config.log_fn(msg)

    def trace(self, record: Dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.write(record)

    async def run(self, max_cycles: Optional[int] = None) -> None:
        self._running = True
        target_period = 1.0 / self.config.hz
        while self._running and (max_cycles is None or self._cycle_index < max_cycles):
            cycle_start = time.perf_counter()
            self._cycle_index += 1
            idx = self._cycle_index

            u = await self._run_phase("understanding", self._understanding, self.config.understanding_budget_ms)
            a = await self._run_phase("attention", self._attention, self.config.attention_budget_ms)
            x = await self._run_phase("action", self._action, self.config.action_budget_ms)

            elapsed = time.perf_counter() - cycle_start
            sleep_for = max(0.0, target_period - elapsed)
            self._log(
                f"cycle idx={idx} u_ms={u:.1f} a_ms={a:.1f} x_ms={x:.1f} total_ms={elapsed*1000.0:.1f} sleep_s={sleep_for:.3f}"
            )
            if self._trace is not None:
                self._trace.write_cycle(CycleMetrics(idx=idx, u_ms=u, a_ms=a, x_ms=x, total_ms=elapsed * 1000.0, sleep_s=sleep_for))
            if sleep_for:
                await asyncio.sleep(sleep_for)

    def stop(self) -> None:
        self._running = False
        if self._trace is not None:
            self._trace.close()
