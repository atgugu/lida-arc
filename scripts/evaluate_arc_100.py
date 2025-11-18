#!/usr/bin/env python
"""
Evaluate ARC-AGI solver on 100 tasks with multi-prediction.

This script evaluates the solver with:
- Single prediction (baseline)
- Multi-prediction (2 attempts per task)

Key metric: Exact match rate (100% pixel accuracy)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from lida.arc.environment import ARCTask, GridPair
from lida.arc.cognitive_solver import ARCCognitiveSolver, ARCSolverConfig


def load_arc_tasks(data_dirs: List[Path]) -> List[ARCTask]:
    """Load all ARC tasks from multiple directories."""
    tasks = []

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue

        for json_file in sorted(data_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Convert to GridPair objects
                train = [GridPair(pair['input'], pair['output']) for pair in data['train']]
                test = [GridPair(pair['input'], pair['output']) for pair in data['test']]

                task = ARCTask(
                    task_id=json_file.stem,
                    train=train,
                    test=test
                )
                tasks.append(task)

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    return tasks


def evaluate_with_multi_prediction(tasks: List[ARCTask], max_predictions: int = 2, verbose: bool = False):
    """Evaluate tasks with multi-prediction."""

    config = ARCSolverConfig(
        verbose=verbose,
        debug=False,
        max_predictions=max_predictions
    )

    solver = ARCCognitiveSolver(config)

    results = []
    exact_matches = 0
    total_tasks = 0
    total_accuracy = 0.0
    predictions_stats = defaultdict(int)  # Track how many predictions were tried

    print(f"\nEvaluating {len(tasks)} tasks with max_predictions={max_predictions}...")
    print("="*70)

    start_time = time.time()

    for i, task in enumerate(tasks):
        # Test first test case only (competition standard)
        result = solver.evaluate(task, test_index=0)

        total_tasks += 1
        total_accuracy += result['accuracy']

        if result['solved']:
            exact_matches += 1

        predictions_tried = result.get('predictions_tried', 1)
        predictions_stats[predictions_tried] += 1

        results.append(result)

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(tasks)} tasks ({exact_matches} solved so far)")

        # Reset solver for next task
        solver.reset()

    elapsed = time.time() - start_time

    # Print summary
    print("="*70)
    print(f"\nRESULTS (max_predictions={max_predictions}):")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Exact matches (100% pixel accuracy): {exact_matches}/{total_tasks} ({100*exact_matches/total_tasks:.1f}%)")
    print(f"  Average accuracy: {100*total_accuracy/total_tasks:.1f}%")
    print(f"  Average time per task: {elapsed/total_tasks:.1f}s")
    print()
    print(f"Prediction statistics:")
    for num_preds, count in sorted(predictions_stats.items()):
        print(f"  {num_preds} prediction(s): {count} tasks")

    return {
        'max_predictions': max_predictions,
        'total_tasks': total_tasks,
        'exact_matches': exact_matches,
        'solve_rate': exact_matches / total_tasks if total_tasks > 0 else 0.0,
        'average_accuracy': total_accuracy / total_tasks if total_tasks > 0 else 0.0,
        'total_time': elapsed,
        'predictions_stats': dict(predictions_stats),
        'results': results
    }


def main():
    """Main evaluation."""

    # Locate task directories
    project_root = Path(__file__).parent.parent
    data_dirs = [
        project_root / "data" / "arc_tasks" / "training",
        project_root / "data" / "arc_tasks" / "evaluation",
        project_root / "data" / "arc_100" / "training",
        project_root / "data" / "arc_100" / "evaluation",
    ]

    # Load tasks
    print("Loading ARC tasks...")
    tasks = load_arc_tasks(data_dirs)
    print(f"✓ Loaded {len(tasks)} tasks")

    if len(tasks) == 0:
        print("ERROR: No tasks found!")
        return

    print("\n" + "="*70)
    print("ARC-AGI EVALUATION: SINGLE VS MULTI-PREDICTION")
    print("="*70)

    # Evaluate with single prediction (baseline)
    print("\n### BASELINE: Single Prediction (max_predictions=1) ###")
    baseline_results = evaluate_with_multi_prediction(tasks, max_predictions=1, verbose=False)

    # Evaluate with multi-prediction
    print("\n### MULTI-PREDICTION: 2 Attempts per Task (max_predictions=2) ###")
    multi_results = evaluate_with_multi_prediction(tasks, max_predictions=2, verbose=False)

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: Single vs Multi-Prediction")
    print("="*70)

    baseline_rate = baseline_results['solve_rate']
    multi_rate = multi_results['solve_rate']
    improvement = multi_rate - baseline_rate

    print(f"\nExact Match Rate (KEY METRIC):")
    print(f"  Single prediction:  {100*baseline_rate:.1f}% ({baseline_results['exact_matches']}/{baseline_results['total_tasks']})")
    print(f"  Multi-prediction:   {100*multi_rate:.1f}% ({multi_results['exact_matches']}/{multi_results['total_tasks']})")
    print(f"  Improvement:        {100*improvement:+.1f}% ({int(improvement*baseline_results['total_tasks'])} more tasks)")

    print(f"\nAverage Accuracy:")
    print(f"  Single prediction:  {100*baseline_results['average_accuracy']:.1f}%")
    print(f"  Multi-prediction:   {100*multi_results['average_accuracy']:.1f}%")
    print(f"  Improvement:        {100*(multi_results['average_accuracy']-baseline_results['average_accuracy']):+.1f}%")

    if improvement > 0:
        print(f"\n✓ Multi-prediction IMPROVES performance!")
        print(f"  Benefit: {100*improvement:.1f}% absolute improvement in solve rate")
        print(f"  This demonstrates the value of generating diverse predictions")
    elif improvement == 0:
        print(f"\n= Multi-prediction shows NO CHANGE (all tasks solved or failed with single prediction)")
    else:
        print(f"\n! Multi-prediction shows regression (unexpected)")

    # Save results
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "arc_100_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump({
            'baseline': {k: v for k, v in baseline_results.items() if k != 'results'},
            'multi_prediction': {k: v for k, v in multi_results.items() if k != 'results'},
            'improvement': {
                'solve_rate': improvement,
                'additional_solves': int(improvement * baseline_results['total_tasks'])
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
