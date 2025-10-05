# Changelog

## 1.0.0 (Initial Release)
- Full cognitive cycle engine with phase budgets and telemetry
- Perception pipeline with PAM (spreading activation, caps/normalization), workspace, and codelets
- Attention/Global Workspace with salience, relevance (drives/episodic), novelty, and attentional learning boosts
- Procedural memory (schemes), BehaviorNet with utility weighting and cooldown; TD learning with affect-modulated alpha; utility decay
- Episodic memory (TEM/EM) with encoding, decay, consolidation, cue-based retrieval; episodic tags
- Affect: energy and safety drives; feelings mapped to PAM and episodic tags
- Environments: symbolic foraging and gridworld (multi-food, hazards), neighborhood perception
- Persistence: SQLite snapshots (PAM, schemes, TEM/EM); run/demos support load/save
- Reporting: timing, selection latency, winner distribution, action counts; per-episode trends (cycles-to-forage, hazard exposures, latency, avg utilities); DOT graph exports of PAM and workspace
- CLI: run, snapshot, and demo commands
- Tests: 19 passing covering engine, scheduler, PAM, attention, perception, learning, BN cooldown, episodic, metrics
