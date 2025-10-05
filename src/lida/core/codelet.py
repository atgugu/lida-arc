from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Codelet:
    name: str
    urgency: float = 1.0
    action: Optional[Callable[[], None]] = None
    kind: str = "generic"
    metadata: dict = field(default_factory=dict)

    def run(self) -> None:
        if self.action:
            self.action()
