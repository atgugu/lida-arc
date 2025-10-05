# LIDA Build Plan and Roadmap (Cited)

This document defines the scope, milestones, acceptance criteria, and the final demo for a full Python implementation of the LIDA cognitive architecture.

References: [1]–[10] at end.

## Scope and modules

- core: cycle engine, codelets, scheduler, events (conscious content, coalitions)
- memory: PAM (slipnet-like), workspace, episodic (TEM/EM), declarative (optional), persistence
- attention: coalitions, global workspace competition, broadcast
- action: procedural memory (schemes), behavior net, motor/actuators
- affect: drives, emotions/feelings, modulation hooks
- learning: perceptual, episodic, procedural, attentional
- sensors/env: inputs and environment adapters
- tools: inspection, config, CLI

## Milestones and deliverables

1) Project scaffolding and foundations (Week 1)
- Cycle engine with phase budgets, telemetry; codelet scheduler.
- Deliverable: Runs empty cycles with 95% within budget.

2) Perception, PAM, and Workspace (Weeks 2–3)
- Graph PAM with spreading activation; structure-building codelets; workspace blackboard.
- Deliverable: Streaming symbols → recognized categories/relations.

3) Attention system and Global Workspace (Weeks 4–5)
- Coalition builder; competition metric; WTA conscious broadcast.
- Deliverable: Single conscious content per cycle with reproducible selection.

4) Episodic memory (Weeks 6–7)
- TEM/EM stores; encoding on broadcast; cue-based retrieval; consolidation.
- Deliverable: Episodes recalled by cues; forgetting curves verified.

5) Procedural memory and Behavior Net (Weeks 8–9)
- Schemes; instantiation; utilities; Behavior Net selection.
- Deliverable: Context-appropriate behavior selection; conflict resolution.

6) Motor planning and environment adapters (Week 10)
- Actuator API; adapters for gridworld/symbolic tasks; reward plumbing.
- Deliverable: End-to-end perceive → attend → act in environment.

7) Learning across modules (Weeks 11–13)
- Perceptual: link weight updates, category induction; Procedural: TD/Q updates; Attentional: urgency tuning.
- Deliverable: Measurable performance improvements and persistence of learned structures.

8) Motivation and affect (Weeks 14–15)
- Drives, setpoints, feelings in PAM; modulation of salience/learning.
- Deliverable: Behavior shifts with drive deficits; emotional tags bias attention.

9) Persistence, snapshots, inspection (Week 16)
- sqlite-backed stores; snapshot/replay; graph exports.
- Deliverable: Deterministic replay of sessions.

10) Evaluation suite and documentation (Weeks 17–18)
- Tasks, metrics dashboards, docs.
- Deliverable: Reports on timing conformance and learning curves.

11) Hardening and release (Week 19)
- Profiling, optimizations, config presets, 1.0.0.

## Acceptance criteria (selected)

- Single conscious content per cycle; 95% cycles within configured timing bands [5].
- Attention competition favors higher salience/relevance coalitions [1,6].
- EM retrieval improves task performance; learned schemes reduce latency [7].
- Drives modulate attention and selection [1].

## Final demo (provable full functionality)

We will deliver a scripted, repeatable experiment that simultaneously exercises perception, attention, episodic memory, procedural memory with action selection, learning, and affect, with cycle-level timing conformance. It consists of three stages in a gridworld-foraging environment with distractors and homeostatic drives:

- Setup:
  - Drives: energy (hunger), safety. Energy decays over time; food restores energy; unsafe tiles increase risk. Feelings (stress/relief) represented as PAM nodes, modulating salience and learning.
  - Environment: 15x15 grid; foods (varying nutritional value), obstacles, and distractors that are visually salient but not useful. Periodic hazards.
  - Sensors: symbolic features for nearby tiles; stream includes novelty events.

- Stage A: Attentional Load and Novelty
  - Stream a mixture of salient distractors and task-relevant cues.
  - Criterion: Global Workspace selects task-relevant coalitions >70% under load after learning; single conscious content per cycle; timing bands satisfied [5].

- Stage B: Episodic Retrieval and Planning
  - Place hidden high-value food in consistent contexts that must be recalled via cues.
  - Criterion: After initial exposure, subsequent runs show faster routes and earlier selection of the correct behavior due to episodic retrieval; significant reduction in cycles-to-goal; recall traces show cue→episode activation.

- Stage C: Procedural Learning and Drive-Modulated Selection
  - Multiple candidate behaviors compete (forage, explore, avoid hazard, investigate distractor). Utilities updated via TD; Behavior Net resolves conflicts.
  - Criterion: Over 10 episodes, success rate and energy efficiency improve; utilities converge; selection latency decreases; drive deficit systematically shifts selection toward foraging when energy low.

- Global Criteria and Proof Artifacts:
  - Timing: >95% cycles in 100–250 ms window; distributions reported.
  - Consciousness: exactly one conscious content per cycle; logs prove no dual frames.
  - Memory operations: counts of encodes/retrievals/consolidations with examples.
  - Learning curves: reward, energy efficiency, selection latency; pre/post comparisons with confidence intervals.
  - Reproducibility: Deterministic replay from snapshot yields identical traces.
  - All metrics and plots saved under `runs/<timestamp>/report/`.

Run command:

```
lida demo --preset foraging --episodes 10 --seed 123 --hz 8 --report
```

Outputs:
- JSONL traces per cycle; sqlite snapshot; Graphviz exports; markdown report linking figures.

## References

[1]: https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1030&context=ccrg_papers
[2]: https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture)
[3]: https://ccrg.cs.memphis.edu/tutorial/mindAccordingToLIDA/Brief-Account.pdf
[4]: https://ccrg.cs.memphis.edu/tutorial/mind-according-to-lida.html
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3081809/
[6]: https://cdn.aaai.org/Symposia/Fall/2007/FS-07-01/FS07-01-011.pdf
[7]: https://ccrg.cs.memphis.edu/assets/papers/ICCM06-UR.pdf
[8]: https://research.manchester.ac.uk/files/33082375/FULL_TEXT.PDF
[9]: https://cdn.aaai.org/ocs/1308/1308-7791-1-PB.pdf
[10]: https://www.sciencedirect.com/science/article/pii/S2212683X16300196
