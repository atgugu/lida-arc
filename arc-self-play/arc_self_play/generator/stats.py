"""Statistics tracking for puzzle generation runs."""

from __future__ import annotations

import statistics as pystats
from collections import defaultdict
from typing import Optional


class GenerationStats:
    """Accumulates metrics during a generation run."""

    def __init__(self) -> None:
        self.total_attempts = 0
        self.valid = 0
        self.invalid_reasons: dict[str, int] = defaultdict(int)
        self.by_difficulty: dict[str, dict[str, int]] = defaultdict(
            lambda: {"generated": 0, "valid": 0, "solved": 0}
        )
        self.solve_times: list[float] = []

    def record_attempt(self, difficulty: str, is_valid: bool, reason: str = "") -> None:
        self.total_attempts += 1
        self.by_difficulty[difficulty]["generated"] += 1
        if is_valid:
            self.valid += 1
            self.by_difficulty[difficulty]["valid"] += 1
        elif reason:
            self.invalid_reasons[reason] += 1

    def record_solve(self, difficulty: str, solved: bool, elapsed: float = 0.0) -> None:
        if solved:
            self.by_difficulty[difficulty]["solved"] += 1
            self.solve_times.append(elapsed)

    @property
    def validity_rate(self) -> float:
        return self.valid / max(1, self.total_attempts)

    @property
    def solve_rate(self) -> Optional[float]:
        total_tested = sum(d["valid"] for d in self.by_difficulty.values())
        total_solved = sum(d["solved"] for d in self.by_difficulty.values())
        return total_solved / max(1, total_tested) if total_tested else None

    def report(self) -> str:
        lines = [
            "=" * 60,
            "GENERATION STATISTICS",
            "=" * 60,
            f"Attempts:  {self.total_attempts}",
            f"Valid:     {self.valid} ({self.validity_rate:.1%})",
        ]

        if self.invalid_reasons:
            lines.append("\nInvalid reasons:")
            for reason, n in sorted(self.invalid_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {n}")

        lines.append("\nBy difficulty:")
        for diff in ("easy", "medium", "hard"):
            d = self.by_difficulty.get(diff)
            if d and d["generated"]:
                vr = d["valid"] / d["generated"]
                sr = d["solved"] / max(1, d["valid"])
                lines.append(
                    f"  {diff:8s}  gen={d['generated']:4d}  "
                    f"valid={d['valid']:4d} ({vr:.0%})  "
                    f"solved={d['solved']:4d} ({sr:.0%})"
                )

        if self.solve_times:
            lines.append(
                f"\nSolve time  avg={pystats.mean(self.solve_times):.2f}s  "
                f"min={min(self.solve_times):.2f}s  "
                f"max={max(self.solve_times):.2f}s"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
