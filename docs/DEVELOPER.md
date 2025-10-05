# Developer Guide

## Code layout
- Core engine and codelets: src/lida/core
- Memory systems: src/lida/memory
- Attention system: src/lida/attention
- Action systems: src/lida/action
- Perception pipeline and codelets: src/lida/perception
- Affect/Drives: src/lida/affect
- Environments: src/lida/env
- Persistence and tools: src/lida/persistence, src/lida/tools
- CLI: src/lida/cli.py

## Running
- Quick run: `lida run --hz 8 --max-cycles 50 --trace-dir ./lida/runs`
- Demo with report: `lida demo --env grid --episodes 5 --hz 8 --trace-dir ./lida/runs --report`

## Tracing
- The cycle engine writes JSONL traces with entries of type: cycle, attention, action, episode.
- Use src/lida/tools/report.py to compute metrics and write a markdown report.

## Persistence
- Use SQLiteStore to save/load PAM, schemes, TEM, EM.
- `lida run --load-snapshot <db> --save-snapshot <db>` for live sessions; `lida demo --snapshot-prefix` for pre/post snapshots.

## Testing
- Run `pytest -q`.
- Key tests: cycle engine, scheduler, PAM spread/limits, attentional competition, category promotion/induction, perception codelets, BehaviorNet cooldown, attentional learning boost.

## Extending
- Add new environments under src/lida/env following Gridworld patterns.
- Implement new codelets in src/lida/perception for different modalities.
- Add drives in src/lida/affect and map them to PAM feelings in CLI ingestion.
