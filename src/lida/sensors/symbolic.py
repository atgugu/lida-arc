from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Symbol:
    obj_id: str
    token: str


class SymbolStream:
    def __init__(self, sequence: List[str]):
        if not sequence:
            raise ValueError("sequence must be non-empty")
        self.sequence = list(sequence)
        self._idx = 0
        self._counter = 0

    def next_symbol(self) -> Symbol:
        token = self.sequence[self._idx % len(self.sequence)]
        self._idx += 1
        self._counter += 1
        return Symbol(obj_id=f"obj:{self._counter}", token=token)

    @property
    def step(self) -> int:
        return self._counter
