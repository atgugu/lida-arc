"""
Cognitive solver for ARC tasks using LIDA's cognitive cycle architecture.

This module integrates:
- Demonstration analysis (understanding)
- PAM spreading activation (understanding)
- Pattern hypothesis competition (attention)
- Pattern application (action)
- Hebbian learning (action)

The solver operates in a cognitive cycle:
1. Understanding: Analyze demos, generate hypotheses via PAM
2. Attention: Validate hypotheses, compete for workspace
3. Action: Apply winning pattern, strengthen connections
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..core.cycle import CognitiveCycleEngine, CycleConfig
from ..attention.global_workspace import GlobalWorkspace, CompetitionWeights
from ..memory.workspace import SituationalModel
from ..memory.pam import PerceptualAssociativeMemory
from .codelets import ARCCodeletFactory, PatternHypothesis
from .demonstration import DemonstrationAnalyzer
from .pam_integration import ARCPAMIntegration
from .primitives import PrimitiveLibrary
from .environment import ARCTask, ARCEnvironment, GridPair


@dataclass
class ARCSolverConfig:
    """Configuration for ARC cognitive solver."""

    # Cognitive cycle parameters
    cycle_hz: float = 10.0
    understanding_budget_ms: int = 100
    attention_budget_ms: int = 50
    action_budget_ms: int = 50

    # Global workspace competition weights
    salience_weight: float = 0.5
    relevance_weight: float = 0.3
    novelty_weight: float = 0.2

    # Multi-prediction strategy (ARC-AGI allows 2 predictions per test)
    max_predictions: int = 2  # Generate up to k diverse predictions
    min_prediction_salience: float = 0.5  # Minimum salience to make a prediction

    # PAM spreading activation parameters
    pam_iterations: int = 5
    pam_decay: float = 0.1

    # Hypothesis validation
    max_hypotheses: int = 10

    # Logging
    verbose: bool = False
    debug: bool = False  # Extra detailed logging


class ARCCognitiveSolver:
    """Cognitive solver for ARC tasks using LIDA's full cognitive architecture."""

    def __init__(self, config: ARCSolverConfig = None):
        """
        Args:
            config: Solver configuration
        """
        self.config = config or ARCSolverConfig()

        # Initialize LIDA components
        self.workspace = SituationalModel()
        self.pam = PerceptualAssociativeMemory()
        self.global_workspace = GlobalWorkspace(
            weights=CompetitionWeights(
                salience_w=self.config.salience_weight,
                relevance_w=self.config.relevance_weight,
                novelty_w=self.config.novelty_weight
            )
        )

        # Initialize ARC components
        self.primitives = PrimitiveLibrary()
        self.analyzer = DemonstrationAnalyzer(self.primitives)
        self.pam_integration = ARCPAMIntegration(self.pam, self.primitives)

        # Codelet factory
        self.codelet_factory = ARCCodeletFactory(
            workspace=self.workspace,
            pam_integration=self.pam_integration,
            primitive_library=self.primitives,
            analyzer=self.analyzer,
            debug=self.config.debug
        )

        # Cognitive cycle engine
        self.cycle_engine = CognitiveCycleEngine(
            config=CycleConfig(
                hz=self.config.cycle_hz,
                understanding_budget_ms=self.config.understanding_budget_ms,
                attention_budget_ms=self.config.attention_budget_ms,
                action_budget_ms=self.config.action_budget_ms,
                log_fn=self._log if self.config.verbose else None
            )
        )

        # Task state
        self.current_task: Optional[ARCTask] = None
        self.current_test_index: int = 0
        self.winning_coalition_id: Optional[str] = None  # Primary winner (backward compat)
        self.winning_coalitions: List = []  # All winning coalitions for multi-prediction
        self.cycle_count = 0
        self.all_predictions: List = []  # Store multiple predictions for diversity filtering

        # Setup cognitive cycle hooks
        self._setup_cycle_hooks()

    def _log(self, msg: str):
        """Log message if verbose mode enabled."""
        if self.config.verbose:
            print(f"[ARCSolver] {msg}")

    def _debug(self, msg: str):
        """Log debug message if debug mode enabled."""
        if self.config.debug:
            print(f"[DEBUG] {msg}")

    def _setup_cycle_hooks(self):
        """Setup cognitive cycle phase hooks."""

        async def understanding_phase():
            """Understanding phase: analyze demonstrations and generate hypotheses."""
            self._debug("=== UNDERSTANDING PHASE START ===")

            # Run understanding codelets
            codelets = self.codelet_factory.make_understanding_codelets()
            self._debug(f"Created {len(codelets)} understanding codelets")

            for codelet in sorted(codelets, key=lambda c: c.urgency, reverse=True):
                self._debug(f"Running codelet: {codelet.name} (urgency={codelet.urgency})")
                try:
                    codelet.run()
                    self._debug(f"  ✓ {codelet.name} completed")
                except Exception as e:
                    self._debug(f"  ✗ {codelet.name} failed: {e}")

            hypotheses = self.codelet_factory.hypotheses
            self._log(f"Understanding: Generated {len(hypotheses)} hypotheses")

            if self.config.debug:
                for i, hyp in enumerate(hypotheses):
                    self._debug(f"  Hypothesis {i}: {hyp.pattern.pattern_id}")
                    self._debug(f"    Operations: {hyp.pattern.grid_operations}")
                    self._debug(f"    Salience: {hyp.salience:.3f}")
                    self._debug(f"    Confidence: {hyp.pattern.confidence:.3f}")
                    self._debug(f"    Support: {hyp.support_count} demos")

            self._debug("=== UNDERSTANDING PHASE END ===")

        async def attention_phase():
            """Attention phase: validate hypotheses and compete for workspace."""
            self._debug("=== ATTENTION PHASE START ===")

            # Run attention codelets
            codelets = self.codelet_factory.make_attention_codelets()
            self._debug(f"Created {len(codelets)} attention codelets")

            for codelet in sorted(codelets, key=lambda c: c.urgency, reverse=True):
                self._debug(f"Running codelet: {codelet.name} (urgency={codelet.urgency})")
                try:
                    codelet.run()
                    self._debug(f"  ✓ {codelet.name} completed")
                except Exception as e:
                    self._debug(f"  ✗ {codelet.name} failed: {e}")

            # Get coalitions
            coalitions = self.codelet_factory.get_coalitions()
            self._debug(f"Created {len(coalitions)} coalitions")

            if coalitions:
                if self.config.debug:
                    for i, coalition in enumerate(coalitions):
                        self._debug(f"  Coalition {i}: {coalition.id}")
                        self._debug(f"    Summary: {coalition.summary}")
                        self._debug(f"    Salience: {coalition.salience:.3f}")

                # SALIENCE-BASED SELECTION: Trust the salience mechanism
                # (Salience already incorporates validation via: salience = 0.5 * initial + 0.5 * validation_accuracy)
                # Sort by salience and select top-k patterns
                sorted_coalitions = sorted(coalitions, key=lambda c: c.salience, reverse=True)

                # MULTI-PREDICTION STRATEGY: Take top-k candidates
                # ARC-AGI allows 2 predictions per test
                top_k = sorted_coalitions[:self.config.max_predictions]

                # QUALITY THRESHOLD: Only make predictions if salience meets minimum threshold
                # This prevents predictions with very low confidence
                self.winning_coalitions = [
                    coal for coal in top_k
                    if coal.salience >= self.config.min_prediction_salience
                ]

                if self.config.debug:
                    self._debug(f"  Selected {len(self.winning_coalitions)}/{len(top_k)} patterns above salience threshold {self.config.min_prediction_salience}")
                    for i, coal in enumerate(self.winning_coalitions):
                        hyp = next((h for h in self.codelet_factory.hypotheses if h.hypothesis_id == coal.id), None)
                        val_acc = hyp.validation_accuracy if hyp else 0.0
                        self._debug(f"    {i+1}. {coal.id}: {coal.summary} (salience={coal.salience:.3f}, validation={val_acc:.2f})")

                    # Show filtered patterns
                    filtered = [coal for coal in top_k if coal.salience < self.config.min_prediction_salience]
                    if filtered:
                        self._debug(f"  Filtered {len(filtered)} patterns below threshold:")
                        for coal in filtered:
                            self._debug(f"    ✗ {coal.id}: salience={coal.salience:.3f} < {self.config.min_prediction_salience}")

                # Primary winner for backward compatibility
                winner = self.winning_coalitions[0] if self.winning_coalitions else None
                self.winning_coalition_id = winner.id if winner else None

                # Compute scores for workspace broadcast
                if self.winning_coalitions:
                    scores = [(c.id, c.salience) for c in sorted_coalitions]
                else:
                    scores = []

                if winner:
                    self._log(f"Attention: Winner = {winner.summary} (salience={winner.salience:.3f})")

                    if self.config.debug:
                        self._debug("  Competition scores:")
                        for cid, score in sorted(scores, key=lambda x: x[1], reverse=True):
                            self._debug(f"    {cid}: {score:.3f}")

                    # Broadcast primary winner to workspace
                    conscious_content = self.global_workspace.broadcast(
                        cycle_index=self.cycle_count,
                        winning=winner
                    )

                    if conscious_content:
                        self._debug("  Broadcast conscious content to workspace")
                        # Store conscious content in workspace
                        self.workspace.upsert_object(
                            oid='conscious',
                            features=conscious_content.payload,
                            properties={'cycle': str(conscious_content.cycle_index)}
                        )
                else:
                    self._log("Attention: No winner")
                    self._debug("  ⚠ Global workspace competition produced no winner")
            else:
                self._log("Attention: No coalitions")
                self._debug("  ⚠ No coalitions created from hypotheses")

            self._debug("=== ATTENTION PHASE END ===")

        async def action_phase():
            """Action phase: apply winning pattern and learn."""
            self._debug("=== ACTION PHASE START ===")

            # Run action codelets
            codelets = self.codelet_factory.make_action_codelets(self.winning_coalition_id)
            self._debug(f"Created {len(codelets)} action codelets")
            self._debug(f"Winning coalition ID: {self.winning_coalition_id}")

            for codelet in sorted(codelets, key=lambda c: c.urgency, reverse=True):
                self._debug(f"Running codelet: {codelet.name} (urgency={codelet.urgency})")
                try:
                    codelet.run()
                    self._debug(f"  ✓ {codelet.name} completed")
                except Exception as e:
                    self._debug(f"  ✗ {codelet.name} failed: {e}")
                    import traceback
                    if self.config.debug:
                        self._debug(f"  Stack trace: {traceback.format_exc()}")

            output = self.codelet_factory.get_output()
            if output:
                self._log(f"Action: Applied pattern, output shape={len(output)}x{len(output[0]) if output else 0}")
                self._debug(f"  Output produced successfully")
            else:
                self._debug("  ⚠ No output produced")

            # Increment cycle count
            self.cycle_count += 1
            self._debug("=== ACTION PHASE END ===")

        self.cycle_engine.on_understanding(understanding_phase)
        self.cycle_engine.on_attention(attention_phase)
        self.cycle_engine.on_action(action_phase)

    async def solve_async(self, task: ARCTask, test_index: int = 0, max_cycles: int = 3) -> Optional[List[List[int]]]:
        """
        Solve an ARC task asynchronously using cognitive cycles.

        Args:
            task: ARC task to solve
            test_index: Which test pair to solve
            max_cycles: Maximum number of cognitive cycles to run

        Returns:
            Predicted output grid, or None if failed
        """
        self.current_task = task
        self.current_test_index = test_index
        self.winning_coalition_id = None
        self.cycle_count = 0

        # Setup task in codelet factory
        test_input = task.test[test_index].input
        self.codelet_factory.set_task(task.train, test_input)

        self._log(f"Solving task {task.task_id}, test {test_index}")
        self._log(f"Training examples: {len(task.train)}")

        # Run cognitive cycles
        await self.cycle_engine.run(max_cycles=max_cycles)

        # Get final output
        output = self.codelet_factory.get_output()

        return output

    def solve(self, task: ARCTask, test_index: int = 0, max_cycles: int = 3) -> Optional[List[List[int]]]:
        """
        Solve an ARC task synchronously using cognitive cycles.

        Args:
            task: ARC task to solve
            test_index: Which test pair to solve
            max_cycles: Maximum number of cognitive cycles to run

        Returns:
            Predicted output grid, or None if failed
        """
        return asyncio.run(self.solve_async(task, test_index, max_cycles))

    def evaluate(self, task: ARCTask, test_index: int = 0) -> Dict[str, Any]:
        """
        Solve and evaluate an ARC task with multi-prediction support.

        Args:
            task: ARC task to solve
            test_index: Which test pair to evaluate

        Returns:
            Evaluation metrics
        """
        # Solve task (runs cognitive cycles, sets winning_coalitions)
        predicted = self.solve(task, test_index)

        # MULTI-PREDICTION: After solving, apply all winning patterns
        all_predictions = [predicted] if predicted else []

        if self.config.max_predictions > 1 and len(self.winning_coalitions) > 1:
            # Apply all other winning patterns to test input
            test_input = task.test[test_index].input

            for i, coalition in enumerate(self.winning_coalitions[1:], start=2):  # Skip first (already applied)
                # Find the corresponding hypothesis
                hyp = next((h for h in self.codelet_factory.hypotheses if h.hypothesis_id == coalition.id), None)

                if hyp and hyp.pattern:
                    try:
                        # Apply pattern to test input
                        result = test_input
                        for op_name in hyp.pattern.grid_operations:
                            prim = self.primitives.get(op_name)
                            if not prim:
                                break

                            if op_name == 'recolor' and hyp.pattern.color_mapping:
                                result = prim.execute(result, hyp.pattern.color_mapping)
                            else:
                                result = prim.execute(result)

                        # DIVERSITY FILTER: Only add if different from existing predictions
                        if result not in all_predictions:
                            all_predictions.append(result)
                            if self.config.debug:
                                self._debug(f"  Added prediction #{len(all_predictions)}: {coalition.id}")
                        elif self.config.debug:
                            self._debug(f"  Skipped duplicate prediction from: {coalition.id}")

                    except Exception as e:
                        if self.config.debug:
                            self._debug(f"  Failed to apply pattern {coalition.id}: {e}")

            if self.config.debug and len(all_predictions) > 1:
                self._debug(f"Generated {len(all_predictions)} diverse predictions")

        # Evaluate: Try all predictions, success if ANY matches
        env = ARCEnvironment(task)
        env.set_test(test_index)

        if not all_predictions or all_predictions[0] is None:
            return {
                'task_id': task.task_id,
                'test_index': test_index,
                'solved': False,
                'accuracy': 0.0,
                'exact_match': False,
                'predictions_tried': 0,
                'error': 'No output produced'
            }

        # Try each prediction
        best_accuracy = 0.0
        exact_match = False
        for i, predicted in enumerate(all_predictions):
            accuracy = env.validate_output(predicted)
            is_exact = env.is_correct(predicted)

            if is_exact:
                exact_match = True
                best_accuracy = 1.0
                if self.config.debug:
                    self._debug(f"  ✓ Prediction #{i+1} SOLVED the task!")
                break  # Found exact match!

            best_accuracy = max(best_accuracy, accuracy)

        accuracy = best_accuracy
        exact_match = exact_match

        # Use first prediction for shape reporting
        first_pred = all_predictions[0] if all_predictions else None

        return {
            'task_id': task.task_id,
            'test_index': test_index,
            'solved': exact_match,
            'accuracy': accuracy,
            'exact_match': exact_match,
            'predictions_tried': len(all_predictions),
            'predicted_shape': (len(first_pred), len(first_pred[0]) if first_pred else 0) if first_pred else (0, 0),
            'expected_shape': (len(task.test[test_index].output), len(task.test[test_index].output[0]) if task.test[test_index].output else 0)
        }

    def batch_evaluate(self, tasks: List[ARCTask]) -> Dict[str, Any]:
        """
        Evaluate solver on multiple ARC tasks.

        Args:
            tasks: List of ARC tasks

        Returns:
            Aggregate evaluation metrics
        """
        results = []
        solved_count = 0
        total_accuracy = 0.0

        for task in tasks:
            for test_idx in range(len(task.test)):
                result = self.evaluate(task, test_idx)
                results.append(result)

                if result['exact_match']:
                    solved_count += 1

                total_accuracy += result['accuracy']

        total_tests = len(results)

        return {
            'total_tasks': len(tasks),
            'total_tests': total_tests,
            'solved': solved_count,
            'solve_rate': solved_count / total_tests if total_tests > 0 else 0.0,
            'average_accuracy': total_accuracy / total_tests if total_tests > 0 else 0.0,
            'results': results
        }

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state for inspection."""
        return {
            'objects': {oid: {'features': obj.features, 'properties': obj.properties}
                       for oid, obj in self.workspace.objects.items()},
            'relations': [(r.subj, r.rel, r.obj, r.confidence)
                         for r in self.workspace.relations],
            'tags': self.workspace.tags
        }

    def get_pam_state(self) -> Dict[str, Any]:
        """Get PAM state for inspection."""
        return {
            'nodes': len(self.pam.g.nodes()),
            'edges': len(self.pam.g.edges()),
            'activations': {
                node_id: self.pam.get_activation(node_id)
                for node_id in list(self.pam.g.nodes())[:20]  # Top 20 nodes
            }
        }

    def reset(self):
        """Reset solver state for new task."""
        self.workspace.clear()
        self.pam_integration.reset_activations()
        self.codelet_factory.set_task([], [])
        self.cycle_count = 0
        self.winning_coalition_id = None
        self.winning_coalitions = []
        self.all_predictions = []

        # CRITICAL FIX: Reset the cycle engine's internal state
        # Without this, _cycle_index accumulates across tasks causing
        # the cycle to never run after the first task
        self.cycle_engine._cycle_index = 0
        self.cycle_engine._running = False

        self._debug("Solver state reset for new task")
