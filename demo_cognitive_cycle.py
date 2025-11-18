#!/usr/bin/env python
"""
Demo of the LIDA-ARC cognitive cycle solving ARC tasks.

This demonstrates the full cognitive learning loop:
1. Understanding: Analyze demonstrations, activate PAM
2. Attention: Compete pattern hypotheses in global workspace
3. Action: Apply winning pattern, Hebbian strengthening

Run with: python demo_cognitive_cycle.py
"""

from lida.arc import (
    ARCTask, GridPair,
    ARCCognitiveSolver, ARCSolverConfig
)


def print_grid(grid, label="Grid"):
    """Pretty print a grid."""
    print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(x) for x in row))


def demo_rotation_task():
    """Demo: 90-degree rotation task."""
    print("="*60)
    print("DEMO 1: 90-Degree Rotation Task")
    print("="*60)

    task = ARCTask(
        task_id='rotation_90_demo',
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

    print("\nTraining demonstrations:")
    for i, demo in enumerate(task.train):
        print(f"\n--- Demo {i+1} ---")
        print_grid(demo.input, "Input")
        print_grid(demo.output, "Output")

    print("\n" + "-"*60)
    print("COGNITIVE CYCLE STARTING...")
    print("-"*60)

    # Create solver with verbose mode
    config = ARCSolverConfig(verbose=True)
    solver = ARCCognitiveSolver(config)

    # Solve task
    result = solver.evaluate(task, test_index=0)

    print("\n" + "-"*60)
    print("COGNITIVE CYCLE COMPLETE")
    print("-"*60)

    print("\nTest input:")
    print_grid(task.test[0].input, "Input")

    print("\nExpected output:")
    print_grid(task.test[0].output, "Expected")

    if result['solved']:
        # Get predicted output
        predicted = solver.codelet_factory.get_output()
        print("\nPredicted output:")
        print_grid(predicted, "Predicted")

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Task ID: {result['task_id']}")
    print(f"Solved: {result['solved']}")
    print(f"Accuracy: {result['accuracy']*100:.1f}%")
    print(f"Exact match: {result['exact_match']}")

    # Inspect workspace
    workspace_state = solver.get_workspace_state()
    print(f"\nWorkspace objects: {len(workspace_state['objects'])}")

    # Inspect PAM
    pam_state = solver.get_pam_state()
    print(f"PAM nodes: {pam_state['nodes']}")
    print(f"PAM edges: {pam_state['edges']}")

    return result


def demo_color_mapping_task():
    """Demo: Color remapping task."""
    print("\n\n")
    print("="*60)
    print("DEMO 2: Color Remapping Task")
    print("="*60)

    task = ARCTask(
        task_id='color_swap_demo',
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

    print("\nTraining demonstrations:")
    for i, demo in enumerate(task.train):
        print(f"\n--- Demo {i+1} ---")
        print_grid(demo.input, "Input")
        print_grid(demo.output, "Output")

    print("\n" + "-"*60)
    print("COGNITIVE CYCLE STARTING...")
    print("-"*60)

    # Create solver
    config = ARCSolverConfig(verbose=True)
    solver = ARCCognitiveSolver(config)

    # Solve
    result = solver.evaluate(task)

    print("\n" + "-"*60)
    print("COGNITIVE CYCLE COMPLETE")
    print("-"*60)

    print("\nTest input:")
    print_grid(task.test[0].input, "Input")

    print("\nExpected output:")
    print_grid(task.test[0].output, "Expected")

    if result['solved']:
        predicted = solver.codelet_factory.get_output()
        print("\nPredicted output:")
        print_grid(predicted, "Predicted")

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Task ID: {result['task_id']}")
    print(f"Solved: {result['solved']}")
    print(f"Accuracy: {result['accuracy']*100:.1f}%")

    return result


def demo_batch_evaluation():
    """Demo: Batch evaluation on multiple tasks."""
    print("\n\n")
    print("="*60)
    print("DEMO 3: Batch Evaluation")
    print("="*60)

    tasks = [
        ARCTask(
            task_id='rotate_90',
            train=[GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]])],
            test=[GridPair(input=[[5, 6], [7, 8]], output=[[7, 5], [8, 6]])]
        ),
        ARCTask(
            task_id='reflect_h',
            train=[GridPair(input=[[1, 2], [3, 4]], output=[[3, 4], [1, 2]])],
            test=[GridPair(input=[[5, 6], [7, 8]], output=[[7, 8], [5, 6]])]
        ),
        ARCTask(
            task_id='reflect_v',
            train=[GridPair(input=[[1, 2], [3, 4]], output=[[2, 1], [4, 3]])],
            test=[GridPair(input=[[5, 6], [7, 8]], output=[[6, 5], [8, 7]])]
        ),
    ]

    print(f"\nEvaluating {len(tasks)} tasks...")

    config = ARCSolverConfig(verbose=False)  # Quiet mode for batch
    solver = ARCCognitiveSolver(config)

    results = solver.batch_evaluate(tasks)

    print("\n" + "="*60)
    print("BATCH RESULTS:")
    print("="*60)
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Total tests: {results['total_tests']}")
    print(f"Solved: {results['solved']}/{results['total_tests']}")
    print(f"Solve rate: {results['solve_rate']*100:.1f}%")
    print(f"Average accuracy: {results['average_accuracy']*100:.1f}%")

    print("\nPer-task results:")
    for r in results['results']:
        status = "✓" if r['solved'] else "✗"
        print(f"  {status} {r['task_id']}: {r['accuracy']*100:.1f}% accuracy")

    return results


def demo_pam_spreading():
    """Demo: PAM spreading activation visualization."""
    print("\n\n")
    print("="*60)
    print("DEMO 4: PAM Spreading Activation")
    print("="*60)

    task = ARCTask(
        task_id='rotation_pam',
        train=[
            GridPair(input=[[1, 2], [3, 4]], output=[[3, 1], [4, 2]]),
        ],
        test=[
            GridPair(input=[[5, 6], [7, 8]], output=[[7, 5], [8, 6]])
        ]
    )

    print("\nAnalyzing demonstrations with PAM...")

    config = ARCSolverConfig(verbose=False)
    solver = ARCCognitiveSolver(config)

    # Solve to populate PAM
    solver.solve(task)

    # Get PAM state
    pam_state = solver.get_pam_state()

    print(f"\nPAM Network:")
    print(f"  Nodes: {pam_state['nodes']}")
    print(f"  Edges: {pam_state['edges']}")

    print(f"\nTop activated nodes:")
    activations = sorted(
        pam_state['activations'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for node_id, activation in activations[:10]:
        if activation > 0.01:
            print(f"  {node_id}: {activation:.3f}")

    # Get top active operations
    top_ops = solver.pam_integration.get_top_active_operations(k=5)
    print(f"\nTop active operations (for hypothesis generation):")
    for i, op in enumerate(top_ops[:5], 1):
        print(f"  {i}. {op}")

    return pam_state


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("LIDA-ARC COGNITIVE CYCLE DEMONSTRATION")
    print("="*60)
    print("\nThis demonstrates the full cognitive learning loop:")
    print("  1. Understanding: Analyze demos, activate PAM")
    print("  2. Attention: Compete hypotheses in global workspace")
    print("  3. Action: Apply winning pattern, Hebbian learning")
    print("\n")

    # Run demos
    demo_rotation_task()
    demo_color_mapping_task()
    demo_batch_evaluation()
    demo_pam_spreading()

    print("\n\n" + "="*60)
    print("ALL DEMOS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
