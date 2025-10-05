from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def compute_episode_bounds(traces: List[dict]) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    start_cycle: Optional[int] = None
    for t in traces:
        if t.get("type") == "episode" and t.get("event") == "start":
            start_cycle = int(t.get("cycle", 0))
        elif t.get("type") == "episode" and t.get("event") == "end" and start_cycle is not None:
            end_cycle = int(t.get("cycle", 0))
            bounds.append((start_cycle, end_cycle))
            start_cycle = None
    return bounds


def cycles_to_first_action(traces: List[dict], action_name: str) -> int | None:
    for t in traces:
        if t.get("type") == "action" and t.get("action") == action_name:
            return int(t.get("cycle", 0))
    return None


def cycles_to_action_by_episode(traces: List[dict], action_name: str) -> List[int | None]:
    bounds = compute_episode_bounds(traces)
    per_ep: List[int | None] = []
    actions = [t for t in traces if t.get("type") == "action" and t.get("action") == action_name]
    for (s, e) in bounds:
        found = None
        for a in actions:
            c = int(a.get("cycle", 0))
            if s <= c <= e:
                found = c - s
                break
        per_ep.append(found)
    return per_ep


def hazard_exposures_by_episode(traces: List[dict]) -> List[int]:
    bounds = compute_episode_bounds(traces)
    acts = [t for t in traces if t.get("type") == "action"]
    out: List[int] = []
    for (s, e) in bounds:
        count = 0
        for a in acts:
            c = int(a.get("cycle", 0))
            if s <= c <= e and a.get("tile") == "HAZARD":
                count += 1
        out.append(count)
    return out


def per_episode_action_counts(traces: List[dict]) -> List[Dict[str, int]]:
    bounds = compute_episode_bounds(traces)
    acts = [t for t in traces if t.get("type") == "action"]
    per: List[Dict[str, int]] = []
    for (s, e) in bounds:
        d: Dict[str, int] = defaultdict(int)
        for a in acts:
            c = int(a.get("cycle", 0))
            if s <= c <= e:
                d[a.get("action", "?")] += 1
        per.append(dict(d))
    return per


def selection_latency_by_episode(traces: List[dict]) -> List[float]:
    bounds = compute_episode_bounds(traces)
    attn = [t for t in traces if t.get("type") == "attention"]
    acts = [t for t in traces if t.get("type") == "action"]
    out: List[float] = []
    for (s, e) in bounds:
        latencies: List[int] = []
        attn_cycles = [int(t.get("cycle", 0)) for t in attn if s <= int(t.get("cycle", 0)) <= e]
        act_cycles = [int(t.get("cycle", 0)) for t in acts if s <= int(t.get("cycle", 0)) <= e]
        for ac in sorted(attn_cycles):
            next_actions = [c for c in act_cycles if c >= ac]
            if next_actions:
                latencies.append(min(next_actions) - ac)
        out.append(sum(latencies) / len(latencies) if latencies else 0.0)
    return out


def avg_utils_by_episode(traces: List[dict]) -> List[Dict[str, float]]:
    bounds = compute_episode_bounds(traces)
    attn = [t for t in traces if t.get("type") == "attention" and t.get("avg_utils")]
    out: List[Dict[str, float]] = []
    for (s, e) in bounds:
        # take last snapshot in episode
        snap = None
        for t in attn:
            c = int(t.get("cycle", 0))
            if s <= c <= e:
                snap = t.get("avg_utils")
        out.append(snap or {})
    return out
