# User Guide

## Install
```
pip install -e ".[dev]"
```

## Commands

- Run a session:
```
lida run --hz 8 --max-cycles 50 --trace-dir ./lida/runs
```

- Demo (multi-episode learning) with report and snapshots:
```
lida demo --env grid --episodes 5 --hz 8 --trace-dir ./lida/runs --report --snapshot-prefix ./lida/runs/demo
```

- Snapshots:
```
lida run --save-snapshot ./lida/runs/agent.db
lida run --load-snapshot ./lida/runs/agent.db --max-cycles 100 --trace-dir ./lida/runs
```

## Outputs
- Console logs show action selections and per-cycle timing.
- Traces are written to `./lida/runs/trace_*.jsonl`.
- Reports are written to `./lida/runs/report/report.md` and include graphs in DOT format.

## Tips
- Start with the symbolic environment to see quick reward learning.
- Use the grid environment to observe hazard avoidance and episodic retrieval effects across episodes.
- Open the DOT files with Graphviz to view the PAM and Workspace graphs.
