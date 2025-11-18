"""
Tests for the ARC cognitive solver with full cognitive cycle integration.
"""

import pytest
from lida.arc import (
    ARCTask, GridPair,
    ARCCognitiveSolver, ARCSolverConfig
)


class TestCognitiveSolverBasic:
    """Test basic cognitive solver functionality."""

    def test_solver_initialization(self):
        """Test solver initializes correctly."""
        solver = ARCCognitiveSolver()

        assert solver.workspace is not None
        assert solver.pam is not None
        assert solver.global_workspace is not None
        assert solver.primitives is not None
        assert solver.analyzer is not None
        assert solver.codelet_factory is not None

    def test_solver_with_config(self):
        """Test solver with custom configuration."""
        config = ARCSolverConfig(
            cycle_hz=5.0,
            verbose=True,
            max_hypotheses=5
        )
        solver = ARCCognitiveSolver(config)

        assert solver.config.cycle_hz == 5.0
        assert solver.config.verbose == True
        assert solver.config.max_hypotheses == 5


class TestRotationTasks:
    """Test cognitive solver on rotation tasks."""

    def test_solve_rotation_90(self):
        """Test solving 90-degree rotation task."""
        task = ARCTask(
            task_id='rotation_90',
            train=[
                GridPair(
                    input=[[1, 2], [3, 4]],
                    output=[[3, 1], [4, 2]]
                ),
                GridPair(
                    input=[[5, 6], [7, 8]],
                    output=[[7, 5], [8, 6]]
                ),
            ],
            test=[
                GridPair(
                    input=[[1, 0], [0, 1]],
                    output=[[0, 1], [1, 0]]
                )
            ]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task, test_index=0)

        assert result['solved'], f"Should solve rotation task, got accuracy {result['accuracy']}"
        assert result['accuracy'] == 1.0

    def test_solve_rotation_180(self):
        """Test solving 180-degree rotation task."""
        task = ARCTask(
            task_id='rotation_180',
            train=[
                GridPair(
                    input=[[1, 2], [3, 4]],
                    output=[[4, 3], [2, 1]]
                ),
            ],
            test=[
                GridPair(
                    input=[[1, 0], [0, 1]],
                    output=[[1, 0], [0, 1]]
                )
            ]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task)

        assert result['accuracy'] >= 0.5, "Should achieve decent accuracy on rotation"


class TestReflectionTasks:
    """Test cognitive solver on reflection tasks."""

    def test_solve_horizontal_reflection(self):
        """Test solving horizontal reflection task."""
        task = ARCTask(
            task_id='reflect_h',
            train=[
                GridPair(
                    input=[[1, 2, 3], [4, 5, 6]],
                    output=[[4, 5, 6], [1, 2, 3]]
                ),
                GridPair(
                    input=[[7], [8], [9]],
                    output=[[9], [8], [7]]
                ),
            ],
            test=[
                GridPair(
                    input=[[1, 0], [0, 1]],
                    output=[[0, 1], [1, 0]]
                )
            ]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task)

        assert result['solved'], "Should solve horizontal reflection"

    def test_solve_vertical_reflection(self):
        """Test solving vertical reflection task."""
        task = ARCTask(
            task_id='reflect_v',
            train=[
                GridPair(
                    input=[[1, 2], [3, 4]],
                    output=[[2, 1], [4, 3]]
                ),
            ],
            test=[
                GridPair(
                    input=[[5, 6], [7, 8]],
                    output=[[6, 5], [8, 7]]
                )
            ]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task)

        assert result['solved'], "Should solve vertical reflection"


class TestColorMappingTasks:
    """Test cognitive solver on color mapping tasks."""

    def test_solve_color_swap(self):
        """Test solving color swap task."""
        task = ARCTask(
            task_id='color_swap',
            train=[
                GridPair(
                    input=[[1, 2, 1], [2, 1, 2]],
                    output=[[3, 4, 3], [4, 3, 4]]
                ),
                GridPair(
                    input=[[1, 1], [2, 2]],
                    output=[[3, 3], [4, 4]]
                ),
            ],
            test=[
                GridPair(
                    input=[[2, 1, 2, 1]],
                    output=[[4, 3, 4, 3]]
                )
            ]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task)

        assert result['solved'], "Should solve color swap task"
        assert result['accuracy'] == 1.0


class TestCognitiveCycleIntegration:
    """Test full cognitive cycle integration."""

    def test_understanding_phase(self):
        """Test understanding phase generates hypotheses."""
        task = ARCTask(
            task_id='test',
            train=[
                GridPair(input=[[1, 2]], output=[[2, 1]]),
            ],
            test=[
                GridPair(input=[[3, 4]], output=[[4, 3]])
            ]
        )

        solver = ARCCognitiveSolver()
        solver.codelet_factory.set_task(task.train, task.test[0].input)

        # Run understanding codelets
        codelets = solver.codelet_factory.make_understanding_codelets()
        for codelet in codelets:
            codelet.run()

        # Should generate hypotheses
        assert len(solver.codelet_factory.hypotheses) > 0

    def test_attention_phase(self):
        """Test attention phase validates and competes hypotheses."""
        task = ARCTask(
            task_id='test',
            train=[
                GridPair(input=[[1, 2]], output=[[2, 1]]),
            ],
            test=[
                GridPair(input=[[3, 4]], output=[[4, 3]])
            ]
        )

        solver = ARCCognitiveSolver()
        solver.codelet_factory.set_task(task.train, task.test[0].input)

        # Run understanding phase
        understanding_codelets = solver.codelet_factory.make_understanding_codelets()
        for codelet in understanding_codelets:
            codelet.run()

        # Run attention phase
        attention_codelets = solver.codelet_factory.make_attention_codelets()
        for codelet in attention_codelets:
            codelet.run()

        # Should validate hypotheses
        for hyp in solver.codelet_factory.hypotheses:
            assert hyp.validation_accuracy >= 0.0

        # Should create coalitions
        coalitions = solver.codelet_factory.get_coalitions()
        assert len(coalitions) > 0

    def test_action_phase(self):
        """Test action phase applies pattern."""
        task = ARCTask(
            task_id='test',
            train=[
                GridPair(input=[[1, 2]], output=[[2, 1]]),
            ],
            test=[
                GridPair(input=[[3, 4]], output=[[4, 3]])
            ]
        )

        solver = ARCCognitiveSolver()
        solver.codelet_factory.set_task(task.train, task.test[0].input)

        # Run understanding
        for codelet in solver.codelet_factory.make_understanding_codelets():
            codelet.run()

        # Run attention
        for codelet in solver.codelet_factory.make_attention_codelets():
            codelet.run()

        # Get winning coalition
        coalitions = solver.codelet_factory.get_coalitions()
        if coalitions:
            winner = max(coalitions, key=lambda c: c.salience)
            winning_id = winner.id

            # Run action
            for codelet in solver.codelet_factory.make_action_codelets(winning_id):
                codelet.run()

            # Should produce output
            output = solver.codelet_factory.get_output()
            assert output is not None


class TestGlobalWorkspaceCompetition:
    """Test global workspace competition for pattern hypotheses."""

    def test_coalition_competition(self):
        """Test that coalitions compete for workspace access."""
        task = ARCTask(
            task_id='test',
            train=[
                GridPair(input=[[1, 2]], output=[[2, 1]]),
                GridPair(input=[[3, 4]], output=[[4, 3]]),
            ],
            test=[
                GridPair(input=[[5, 6]], output=[[6, 5]])
            ]
        )

        solver = ARCCognitiveSolver()
        solver.codelet_factory.set_task(task.train, task.test[0].input)

        # Generate hypotheses
        for codelet in solver.codelet_factory.make_understanding_codelets():
            codelet.run()

        # Validate and create coalitions
        for codelet in solver.codelet_factory.make_attention_codelets():
            codelet.run()

        coalitions = solver.codelet_factory.get_coalitions()
        assert len(coalitions) > 0

        # Compete in global workspace
        winner, scores = solver.global_workspace.compete(coalitions)

        assert winner is not None
        assert winner.salience > 0

        # Winner should have highest score
        winner_score = next(score for cid, score in scores if cid == winner.id)
        all_scores = [score for _, score in scores]
        assert winner_score == max(all_scores)


class TestPAMIntegration:
    """Test PAM spreading activation in cognitive cycle."""

    def test_pam_activation_boosts_salience(self):
        """Test that PAM activation boosts pattern salience."""
        task = ARCTask(
            task_id='test',
            train=[
                GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]]),
            ],
            test=[
                GridPair(input=[[5, 6], [7, 8]], output=[[7, 5], [8, 6]])
            ]
        )

        solver = ARCCognitiveSolver()
        solver.codelet_factory.set_task(task.train, task.test[0].input)

        # Generate hypotheses (includes PAM activation)
        for codelet in solver.codelet_factory.make_understanding_codelets():
            codelet.run()

        # Check that hypotheses have salience influenced by PAM
        assert len(solver.codelet_factory.hypotheses) > 0

        for hyp in solver.codelet_factory.hypotheses:
            assert hyp.salience > 0


class TestBatchEvaluation:
    """Test batch evaluation on multiple tasks."""

    def test_batch_evaluate(self):
        """Test evaluating multiple tasks."""
        tasks = [
            ARCTask(
                task_id='task1',
                train=[GridPair(input=[[1, 2]], output=[[2, 1]])],
                test=[GridPair(input=[[3, 4]], output=[[4, 3]])]
            ),
            ARCTask(
                task_id='task2',
                train=[GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]])],
                test=[GridPair(input=[[5, 6], [7, 8]], output=[[7, 5], [8, 6]])]
            ),
        ]

        solver = ARCCognitiveSolver()
        results = solver.batch_evaluate(tasks)

        assert results['total_tasks'] == 2
        assert results['total_tests'] == 2
        assert results['solve_rate'] >= 0.0
        assert results['solve_rate'] <= 1.0
        assert len(results['results']) == 2


class TestWorkspaceInspection:
    """Test workspace state inspection."""

    def test_get_workspace_state(self):
        """Test getting workspace state."""
        solver = ARCCognitiveSolver()

        task = ARCTask(
            task_id='test',
            train=[GridPair(input=[[1, 2]], output=[[2, 1]])],
            test=[GridPair(input=[[3, 4]], output=[[4, 3]])]
        )

        solver.solve(task)

        state = solver.get_workspace_state()
        assert 'objects' in state
        assert 'relations' in state
        assert 'tags' in state

    def test_get_pam_state(self):
        """Test getting PAM state."""
        solver = ARCCognitiveSolver()

        state = solver.get_pam_state()
        assert 'nodes' in state
        assert 'edges' in state
        assert 'activations' in state


class TestSolverReset:
    """Test solver reset between tasks."""

    def test_reset_clears_state(self):
        """Test that reset clears solver state."""
        solver = ARCCognitiveSolver()

        task = ARCTask(
            task_id='test',
            train=[GridPair(input=[[1, 2]], output=[[2, 1]])],
            test=[GridPair(input=[[3, 4]], output=[[4, 3]])]
        )

        solver.solve(task)

        # State should be populated
        assert len(solver.workspace.objects) > 0

        # Reset
        solver.reset()

        # State should be cleared
        assert len(solver.workspace.objects) == 0
        assert solver.cycle_count == 0


class TestErrorHandling:
    """Test error handling in cognitive solver."""

    def test_handle_empty_task(self):
        """Test handling task with no training examples."""
        task = ARCTask(
            task_id='empty',
            train=[],
            test=[GridPair(input=[[1, 2]], output=[[2, 1]])]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task)

        # Should not crash
        assert 'solved' in result

    def test_handle_pattern_application_failure(self):
        """Test handling when pattern fails to apply."""
        # Create task with incompatible train/test
        task = ARCTask(
            task_id='incompatible',
            train=[GridPair(input=[[1, 2]], output=[[3, 4, 5]])],  # Different sizes
            test=[GridPair(input=[[1, 2]], output=[[2, 1]])]
        )

        solver = ARCCognitiveSolver()
        result = solver.evaluate(task)

        # Should handle gracefully
        assert 'solved' in result
