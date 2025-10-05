from __future__ import annotations

from lida.tools.metrics import selection_latency_by_episode


def test_selection_latency_metric_handles_empty():
    traces = []
    trends = selection_latency_by_episode(traces)
    assert isinstance(trends, list)
    assert trends == []
