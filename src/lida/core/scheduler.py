from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple

from .codelet import Codelet


@dataclass(order=True)
class _QItem:
    priority: float
    idx: int
    codelet: Codelet = field(compare=False)


class CodeletScheduler:
    """Simple urgency-priority queue for codelets within a phase time slice."""

    def __init__(self) -> None:
        self._q: List[_QItem] = []
        self._counter = 0

    def push(self, c: Codelet) -> None:
        item = _QItem(priority=-float(c.urgency), idx=self._counter, codelet=c)
        self._counter += 1
        heapq.heappush(self._q, item)

    def pop(self, max_runs: int = 1) -> List[Codelet]:
        runs: List[Codelet] = []
        while self._q and len(runs) < max_runs:
            runs.append(heapq.heappop(self._q).codelet)
        return runs

    def __len__(self) -> int:
        return len(self._q)
