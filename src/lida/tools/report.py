from __future__ import annotations

import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .metrics import cycles_to_first_action, per_episode_action_counts, cycles_to_action_by_episode, hazard_exposures_by_episode, selection_latency_by_episode, avg_utils_by_episode


@dataclass
class DemoMetrics:
    total_cycles: int
    avg_cycle_ms: float
    pct_within_100_250: float
    actions: Dict[str, int]
    avg_utilities: Dict[str, float]
    selection_latency_avg: float
    winner_distribution: Dict[str, int]
    cycles_to_first_forage: int | None
    per_episode_actions: List[Dict[str, int]]
    per_episode_cycles_to_forage: List[int | None]
    per_episode_hazard_exposures: List[int]
    per_episode_selection_latency: List[float]
    per_episode_avg_utils: List[Dict[str, float]]


def load_traces(trace_dir: Path) -> List[dict]:
    traces: List[dict] = []
    for p in trace_dir.glob("trace_*.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except Exception:
                    pass
    return traces


def compute_metrics(traces: List[dict]) -> DemoMetrics:
    cycles = [t for t in traces if t.get("type") == "cycle"]
    acts = [t for t in traces if t.get("type") == "action"]
    attn = [t for t in traces if t.get("type") == "attention"]

    total_cycles = len(cycles)
    avg_cycle_ms = sum(c.get("total_ms", 0.0) for c in cycles) / total_cycles if total_cycles else 0.0
    within = [100.0 <= float(c.get("total_ms", 0.0)) <= 250.0 for c in cycles]
    pct_within = 100.0 * (sum(1 for x in within if x) / total_cycles) if total_cycles else 0.0

    actions = defaultdict(int)
    for a in acts:
        actions[a.get("action", "?")] += 1

    avg_utils: Dict[str, float] = {}
    for a in reversed(attn):
        if "avg_utils" in a and a["avg_utils"]:
            avg_utils = a["avg_utils"]
            break

    attn_cycles = set(t.get("cycle") for t in attn)
    act_cycles = set(t.get("cycle") for t in acts)
    latencies: List[int] = []
    for ac in sorted(attn_cycles):
        next_actions = [c for c in act_cycles if c >= ac]
        if next_actions:
            latencies.append(min(next_actions) - ac)
    selection_latency_avg = sum(latencies) / len(latencies) if latencies else 0.0

    winners = [t.get("winner") for t in attn if t.get("winner")]
    winner_distribution = dict(Counter(winners))

    cycles_to_first_forage = cycles_to_first_action(traces, "forage")
    per_episode_actions = per_episode_action_counts(traces)
    per_episode_cycles_to_forage = cycles_to_action_by_episode(traces, "forage")
    per_episode_hazard = hazard_exposures_by_episode(traces)
    per_episode_sel_latency = selection_latency_by_episode(traces)
    per_episode_utils = avg_utils_by_episode(traces)

    return DemoMetrics(
        total_cycles=total_cycles,
        avg_cycle_ms=avg_cycle_ms,
        pct_within_100_250=pct_within,
        actions=dict(actions),
        avg_utilities=avg_utils,
        selection_latency_avg=selection_latency_avg,
        winner_distribution=winner_distribution,
        cycles_to_first_forage=cycles_to_first_forage,
        per_episode_actions=per_episode_actions,
        per_episode_cycles_to_forage=per_episode_cycles_to_forage,
        per_episode_hazard_exposures=per_episode_hazard,
        per_episode_selection_latency=per_episode_sel_latency,
        per_episode_avg_utils=per_episode_utils,
    )


def write_markdown(report_dir: Path, metrics: DemoMetrics, pam_dot_path: Path | None = None) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    md = report_dir / "report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# LIDA Demo Report\n\n")
        f.write(f"Total cycles: {metrics.total_cycles}\n\n")
        f.write(f"Average cycle time (ms): {metrics.avg_cycle_ms:.2f}\n\n")
        f.write(f"% cycles within 100–250 ms: {metrics.pct_within_100_250:.1f}%\n\n")
        f.write(f"Average selection latency (cycles): {metrics.selection_latency_avg:.2f}\n\n")
        f.write(f"Cycles to first forage: {metrics.cycles_to_first_forage}\n\n")
        f.write("## Action counts (aggregate)\n\n")
        for k, v in metrics.actions.items():
            f.write(f"- {k}: {v}\n")
        if metrics.per_episode_actions:
            f.write("\n## Per-episode action counts\n\n")
            for idx, d in enumerate(metrics.per_episode_actions):
                f.write(f"Episode {idx}: ")
                f.write(", ".join(f"{k}={v}" for k, v in d.items()))
                f.write("\n")
        if metrics.per_episode_cycles_to_forage:
            f.write("\n## Per-episode cycles to first forage\n\n")
            f.write(", ".join(str(x) for x in metrics.per_episode_cycles_to_forage))
            f.write("\n")
        if metrics.per_episode_hazard_exposures:
            f.write("\n## Per-episode hazard exposures\n\n")
            f.write(", ".join(str(x) for x in metrics.per_episode_hazard_exposures))
            f.write("\n")
        if metrics.per_episode_selection_latency:
            f.write("\n## Per-episode selection latency (avg cycles)\n\n")
            f.write(", ".join(f"{x:.2f}" for x in metrics.per_episode_selection_latency))
            f.write("\n")
        if metrics.per_episode_avg_utils:
            f.write("\n## Per-episode average utilities\n\n")
            for idx, d in enumerate(metrics.per_episode_avg_utils):
                f.write(f"Episode {idx}: ")
                f.write(", ".join(f"{k}={v:.3f}" for k, v in d.items()))
                f.write("\n")
        f.write("\n## Winner distribution (attention)\n\n")
        for k, v in metrics.winner_distribution.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Average utilities (last snapshot)\n\n")
        for k, v in metrics.avg_utilities.items():
            f.write(f"- {k}: {v:.3f}\n")
        if pam_dot_path:
            f.write("\n## PAM Graph\n\n")
            f.write(f"DOT export saved at: {pam_dot_path}\n")
    return md
