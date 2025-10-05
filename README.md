# LIDA (Learning Intelligent Decision Agent) — Python Implementation

A systems-level implementation of the LIDA cognitive architecture in Python. It includes full cognitive cycles with perception, attention (global workspace), action selection, learning, and multiple memory systems. This repository aims to deliver a complete, runnable agent with demos, tests, persistence, and reporting.

## Install

```
pip install -e ".[dev]"
```

## Quick start

- Run a quick session:
```
lida run --hz 8 --max-cycles 50 --trace-dir ./lida/runs
```
- Multi-episode demo with report and snapshots:
```
lida demo --env grid --episodes 5 --hz 8 --trace-dir ./lida/runs --report --snapshot-prefix ./lida/runs/demo
```
- Save/load snapshots during run:
```
lida run --save-snapshot ./lida/runs/agent.db
lida run --load-snapshot ./lida/runs/agent.db --max-cycles 100 --trace-dir ./lida/runs
```

## Project structure

- Core cycle engine, codelets, scheduler, events: file 'lida/src/lida/core/*'
- Perception (symbolic sensors, structure-building, neighborhood): file 'lida/src/lida/perception/*'
- Memories (PAM, Workspace, Episodic), attention (GW), affect (drives), and action selection (procedural, BN): file 'lida/src/lida/memory/*', file 'lida/src/lida/attention/*', file 'lida/src/lida/affect/*', file 'lida/src/lida/action/*'
- Environments: file 'lida/src/lida/env/*'
- Persistence: file 'lida/src/lida/persistence/*'
- Tools (reporting, graph export, metrics): file 'lida/src/lida/tools/*'
- CLI entrypoint: file 'lida/src/lida/cli.py'

## Tests

Run all tests:
```
pytest -q
```

## Demo report

Generates:
- Timing distributions and selection latency (aggregate + per episode)
- Action counts, winner distributions
- Cycles-to-first-forage and per-episode cycles-to-forage
- Hazard exposure counts per episode
- Average utilities (aggregate + per episode)
- PAM and Workspace DOT graph exports

Artifacts are written to file 'lida/runs/report/*'.

## References

See file 'lida/docs/roadmap.md' for the design plan and citations to the LIDA literature.
