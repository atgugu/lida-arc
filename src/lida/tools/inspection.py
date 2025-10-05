from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class CycleMetrics:
    idx: int
    u_ms: float
    a_ms: float
    x_ms: float
    total_ms: float
    sleep_s: float


class TraceWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.file = (self.out_dir / f"trace_{ts}.jsonl").open("a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def write_cycle(self, m: CycleMetrics) -> None:
        self.write({"type": "cycle", **asdict(m)})

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass
