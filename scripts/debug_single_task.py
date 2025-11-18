#!/usr/bin/env python
"""
Debug script to run a single task with verbose logging.

This script runs one task with debug=True to diagnose why tasks fail silently.

Usage: python scripts/debug_single_task.py <task_id>
"""

import sys
import json
from pathlib import Path

from lida.arc import (
    ARCTask, GridPair,
    ARCCognitiveSolver, ARCSolverConfig
)


def load_task(task_id: str, data_dir: Path) -> ARCTask:
    """Load a single task by ID."""
    # Try training first
    task_file = data_dir / "training" / f"{task_id}.json"
    if not task_file.exists():
        # Try evaluation
        task_file = data_dir / "evaluation" / f"{task_id}.json"

    if not task_file.exists():
        raise FileNotFoundError(f"Task {task_id} not found in training or evaluation")

    with open(task_file) as f:
        data = json.load(f)

    train_pairs = [
        GridPair(input=ex['input'], output=ex['output'])
        for ex in data.get('train', [])
    ]

    test_pairs = [
        GridPair(input=ex['input'], output=ex['output'])
        for ex in data.get('test', [])
    ]

    return ARCTask(
        task_id=task_id,
        train=train_pairs,
        test=test_pairs
    )


def print_grid(grid, label="Grid"):
    """Pretty print a grid."""
    print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(x) for x in row))


def debug_task(task_id: str):
    """Debug a single task with verbose logging."""
    print("="*80)
    print(f"DEBUGGING TASK: {task_id}")
    print("="*80)

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_root = project_root / "data" / "arc_tasks"

    # Load task
    print("\nLoading task...")
    try:
        task = load_task(task_id, data_root)
        print(f"✓ Task loaded: {task.task_id}")
        print(f"  Training examples: {len(task.train)}")
        print(f"  Test examples: {len(task.test)}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return

    # Show demonstrations
    print("\n" + "-"*80)
    print("TRAINING DEMONSTRATIONS")
    print("-"*80)
    for i, demo in enumerate(task.train):
        print(f"\nDemo {i+1}:")
        print_grid(demo.input, "  Input")
        print_grid(demo.output, "  Output")

    # Show test input
    print("\n" + "-"*80)
    print("TEST INPUT")
    print("-"*80)
    print_grid(task.test[0].input, "Input")
    print_grid(task.test[0].output, "Expected Output")

    # Create solver with debug=True
    print("\n" + "="*80)
    print("COGNITIVE CYCLE EXECUTION (DEBUG MODE)")
    print("="*80)

    config = ARCSolverConfig(
        verbose=True,
        debug=True,  # Enable detailed debug logging
        cycle_hz=10.0,
        understanding_budget_ms=100,
        attention_budget_ms=50,
        action_budget_ms=50
    )

    solver = ARCCognitiveSolver(config)

    # Solve task
    result = solver.evaluate(task, test_index=0)

    # Show results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nTask ID: {result['task_id']}")
    print(f"Solved: {result['solved']}")
    print(f"Accuracy: {result['accuracy']*100:.1f}%")
    print(f"Exact match: {result['exact_match']}")

    # Show predicted output if available
    predicted = solver.codelet_factory.get_output()
    if predicted:
        print("\nPredicted Output:")
        print_grid(predicted, "Predicted")
    else:
        print("\n⚠ No output produced")

    # Show workspace state
    print("\n" + "-"*80)
    print("WORKSPACE STATE")
    print("-"*80)
    workspace_state = solver.get_workspace_state()
    print(f"Objects in workspace: {len(workspace_state['objects'])}")
    for oid, obj in list(workspace_state['objects'].items())[:10]:  # Show first 10
        print(f"  {oid}:")
        print(f"    Features: {obj['features']}")
        print(f"    Properties: {obj['properties']}")

    # Show PAM state
    print("\n" + "-"*80)
    print("PAM STATE")
    print("-"*80)
    pam_state = solver.get_pam_state()
    print(f"PAM nodes: {pam_state['nodes']}")
    print(f"PAM edges: {pam_state['edges']}")

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

    return result


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_single_task.py <task_id>")
        print("\nAvailable tasks:")
        print("  Training:")
        print("    00d62c1b - Rotation 90°")
        print("    1e0a9b12 - Horizontal reflection")
        print("    2f876c35 - Vertical reflection")
        print("    3c9b0459 - Color mapping")
        print("    4be741c5 - Rotation 180°")
        print("    5bd6f4ac - Rotation 270°")
        print("  Evaluation:")
        print("    6e82a1ae - Diagonal reflection")
        print("    7df24a62 - Fill background")
        print("    8f2ea7aa - Rotation (large)")
        print("    9ecd008a - Composite (rotation + color)")
        sys.exit(1)

    task_id = sys.argv[1]
    debug_task(task_id)


if __name__ == '__main__':
    main()
