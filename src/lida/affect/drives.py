from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Drive:
    name: str
    level: float  # 0..1, where higher means more satisfied
    decay_per_cycle: float = 0.0
    setpoint: float = 0.8

    def deficit(self) -> float:
        return max(0.0, self.setpoint - self.level)


class Drives:
    def __init__(self) -> None:
        self.energy = Drive(name="energy", level=0.6, decay_per_cycle=0.01, setpoint=0.9)
        self.safety = Drive(name="safety", level=0.8, decay_per_cycle=0.0, setpoint=0.95)

    def step_decay(self) -> None:
        self.energy.level = max(0.0, self.energy.level - self.energy.decay_per_cycle)
        # safety has no passive decay by default

    def apply_effect(self, name: str, delta: float) -> None:
        if name == "energy":
            self.energy.level = min(1.0, max(0.0, self.energy.level + float(delta)))
        elif name == "safety":
            self.safety.level = min(1.0, max(0.0, self.safety.level + float(delta)))

    def hunger_boost(self, k: float = 2.0) -> float:
        d = self.energy.deficit()
        return 1.0 + k * d

    def safety_boost(self, k: float = 2.0) -> float:
        d = self.safety.deficit()
        return 1.0 + k * d
