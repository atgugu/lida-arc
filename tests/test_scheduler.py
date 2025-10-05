from __future__ import annotations

from lida.core.scheduler import CodeletScheduler
from lida.core.codelet import Codelet


def test_scheduler_orders_by_urgency_and_fifo_ties():
    sch = CodeletScheduler()
    # Two equal urgency first, then a higher urgency, then a lower
    sch.push(Codelet(name="c1", urgency=1.0))  # tie 1
    sch.push(Codelet(name="c2", urgency=1.0))  # tie 2
    sch.push(Codelet(name="c3", urgency=2.0))  # highest
    sch.push(Codelet(name="c4", urgency=0.5))  # lowest

    popped = []
    for _ in range(4):
        popped.extend(sch.pop(max_runs=1))
    names = [c.name for c in popped]
    assert names[0] == "c3"  # highest first
    assert names[1] == "c1"  # tie keeps insertion order
    assert names[2] == "c2"
    assert names[3] == "c4"  # lowest last
