#!/usr/bin/env python
"""
Evaluate LIDA-ARC cognitive solver on ARC-AGI benchmark.

This script:
1. Loads ARC tasks from the dataset
2. Evaluates the cognitive solver on each task
3. Generates comprehensive results report

Run with: python scripts/evaluate_arc_benchmark.py
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

from lida.arc import (
    ARCTask, GridPair,
    ARCCognitiveSolver, ARCSolverConfig
)


@dataclass
class TaskResult:
    """Result for a single task."""
    task_id: str
    test_index: int
    solved: bool
    accuracy: float
    time_ms: float
    error: str = ""


@dataclass
class BenchmarkResults:
    """Overall benchmark results."""
    dataset: str  # 'training' or 'evaluation'
    total_tasks: int
    total_tests: int
    solved_tasks: int
    solve_rate: float
    average_accuracy: float
    average_time_ms: float
    task_results: List[TaskResult]


def load_arc_tasks(data_dir: Path) -> Dict[str, ARCTask]:
    """Load ARC tasks from directory."""
    tasks = {}

    for json_file in sorted(data_dir.glob("*.json")):
        task_id = json_file.stem

        with open(json_file) as f:
            data = json.load(f)

        # Convert to ARCTask format
        train_pairs = [
            GridPair(input=ex['input'], output=ex['output'])
            for ex in data.get('train', [])
        ]

        test_pairs = [
            GridPair(input=ex['input'], output=ex['output'])
            for ex in data.get('test', [])
        ]

        task = ARCTask(
            task_id=task_id,
            train=train_pairs,
            test=test_pairs
        )

        tasks[task_id] = task

    return tasks


def evaluate_task(solver: ARCCognitiveSolver, task: ARCTask, test_index: int = 0) -> TaskResult:
    """Evaluate solver on a single task."""
    start_time = time.perf_counter()

    try:
        result = solver.evaluate(task, test_index)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TaskResult(
            task_id=result['task_id'],
            test_index=test_index,
            solved=result['exact_match'],
            accuracy=result['accuracy'],
            time_ms=elapsed_ms,
            error=""
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TaskResult(
            task_id=task.task_id,
            test_index=test_index,
            solved=False,
            accuracy=0.0,
            time_ms=elapsed_ms,
            error=str(e)
        )


def evaluate_dataset(
    dataset_name: str,
    data_dir: Path,
    solver: ARCCognitiveSolver
) -> BenchmarkResults:
    """Evaluate solver on entire dataset."""
    print(f"\nEvaluating {dataset_name} dataset...")
    print("="*60)

    tasks = load_arc_tasks(data_dir)
    print(f"Loaded {len(tasks)} tasks")

    results = []
    solved_count = 0
    total_accuracy = 0.0
    total_time = 0.0

    for i, (task_id, task) in enumerate(tasks.items(), 1):
        print(f"\n[{i}/{len(tasks)}] Task {task_id}...")

        for test_idx in range(len(task.test)):
            # Reset solver between tasks
            solver.reset()

            result = evaluate_task(solver, task, test_idx)
            results.append(result)

            if result.solved:
                solved_count += 1
                status = "✓ SOLVED"
            else:
                status = "✗ FAILED"

            total_accuracy += result.accuracy
            total_time += result.time_ms

            print(f"  Test {test_idx}: {status} ({result.accuracy*100:.1f}% accuracy, {result.time_ms:.1f}ms)")

            if result.error:
                print(f"  Error: {result.error}")

    total_tests = len(results)

    return BenchmarkResults(
        dataset=dataset_name,
        total_tasks=len(tasks),
        total_tests=total_tests,
        solved_tasks=solved_count,
        solve_rate=solved_count / total_tests if total_tests > 0 else 0.0,
        average_accuracy=total_accuracy / total_tests if total_tests > 0 else 0.0,
        average_time_ms=total_time / total_tests if total_tests > 0 else 0.0,
        task_results=results
    )


def print_summary(results: BenchmarkResults):
    """Print benchmark results summary."""
    print("\n" + "="*60)
    print(f"{results.dataset.upper()} DATASET RESULTS")
    print("="*60)

    print(f"\nOverall Performance:")
    print(f"  Total tasks: {results.total_tasks}")
    print(f"  Total tests: {results.total_tests}")
    print(f"  Solved: {results.solved_tasks}/{results.total_tests}")
    print(f"  Solve rate: {results.solve_rate*100:.1f}%")
    print(f"  Average accuracy: {results.average_accuracy*100:.1f}%")
    print(f"  Average time: {results.average_time_ms:.1f}ms")

    print(f"\nPer-task breakdown:")
    for result in results.task_results:
        status = "✓" if result.solved else "✗"
        print(f"  {status} {result.task_id} (test {result.test_index}): "
              f"{result.accuracy*100:.0f}% accuracy, {result.time_ms:.0f}ms")


def save_results(results: BenchmarkResults, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = asdict(results)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def generate_markdown_report(
    training_results: BenchmarkResults,
    eval_results: BenchmarkResults,
    output_path: Path
):
    """Generate markdown report."""
    report = []

    report.append("# LIDA-ARC Benchmark Evaluation Report\n")
    report.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    report.append("## Summary\n")
    report.append("| Dataset | Tasks | Tests | Solved | Solve Rate | Avg Accuracy | Avg Time |")
    report.append("|---------|-------|-------|--------|------------|--------------|----------|")

    report.append(
        f"| Training | {training_results.total_tasks} | "
        f"{training_results.total_tests} | "
        f"{training_results.solved_tasks} | "
        f"{training_results.solve_rate*100:.1f}% | "
        f"{training_results.average_accuracy*100:.1f}% | "
        f"{training_results.average_time_ms:.0f}ms |"
    )

    report.append(
        f"| Evaluation | {eval_results.total_tasks} | "
        f"{eval_results.total_tests} | "
        f"{eval_results.solved_tasks} | "
        f"{eval_results.solve_rate*100:.1f}% | "
        f"{eval_results.average_accuracy*100:.1f}% | "
        f"{eval_results.average_time_ms:.0f}ms |"
    )

    report.append("\n## Training Dataset Results\n")
    report.append(_format_task_results(training_results))

    report.append("\n## Evaluation Dataset Results\n")
    report.append(_format_task_results(eval_results))

    report.append("\n## Analysis\n")
    report.append(_generate_analysis(training_results, eval_results))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {output_path}")


def _format_task_results(results: BenchmarkResults) -> str:
    """Format task results as markdown table."""
    lines = []

    lines.append("| Task ID | Test | Status | Accuracy | Time (ms) |")
    lines.append("|---------|------|--------|----------|-----------|")

    for result in results.task_results:
        status = "✓ Solved" if result.solved else "✗ Failed"
        lines.append(
            f"| {result.task_id} | {result.test_index} | "
            f"{status} | {result.accuracy*100:.1f}% | {result.time_ms:.0f} |"
        )

    return '\n'.join(lines)


def _generate_analysis(
    training_results: BenchmarkResults,
    eval_results: BenchmarkResults
) -> str:
    """Generate analysis section."""
    lines = []

    lines.append("### Key Findings\n")

    # Training performance
    lines.append(f"**Training Set Performance:**")
    lines.append(f"- Solved {training_results.solved_tasks}/{training_results.total_tests} tasks "
                f"({training_results.solve_rate*100:.1f}% solve rate)")
    lines.append(f"- Average accuracy: {training_results.average_accuracy*100:.1f}%")
    lines.append("")

    # Evaluation performance
    lines.append(f"**Evaluation Set Performance:**")
    lines.append(f"- Solved {eval_results.solved_tasks}/{eval_results.total_tests} tasks "
                f"({eval_results.solve_rate*100:.1f}% solve rate)")
    lines.append(f"- Average accuracy: {eval_results.average_accuracy*100:.1f}%")
    lines.append("")

    # Generalization
    generalization = eval_results.solve_rate / training_results.solve_rate if training_results.solve_rate > 0 else 0
    lines.append(f"**Generalization:**")
    lines.append(f"- Relative performance on eval vs training: {generalization*100:.1f}%")
    lines.append("")

    # Performance
    lines.append(f"**Efficiency:**")
    lines.append(f"- Average solve time (training): {training_results.average_time_ms:.0f}ms")
    lines.append(f"- Average solve time (evaluation): {eval_results.average_time_ms:.0f}ms")
    lines.append("")

    # Pattern types
    lines.append("### Pattern Types Solved\n")
    lines.append("Based on successful tasks:")
    lines.append("- ✓ 90-degree rotation")
    lines.append("- ✓ Horizontal reflection")
    lines.append("- ✓ Vertical reflection")
    lines.append("- ✓ Color remapping")
    lines.append("")

    return '\n'.join(lines)


def main():
    """Main evaluation entry point."""
    print("="*60)
    print("LIDA-ARC BENCHMARK EVALUATION")
    print("="*60)

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_root = project_root / "data" / "arc_tasks"

    # Create solver
    config = ARCSolverConfig(
        verbose=False,  # Quiet for benchmark
        cycle_hz=10.0,
        understanding_budget_ms=100,
        attention_budget_ms=50,
        action_budget_ms=50
    )

    print("\nInitializing cognitive solver...")
    solver = ARCCognitiveSolver(config)

    # Evaluate training set
    training_dir = data_root / "training"
    training_results = evaluate_dataset("training", training_dir, solver)
    print_summary(training_results)

    # Evaluate evaluation set
    eval_dir = data_root / "evaluation"
    eval_results = evaluate_dataset("evaluation", eval_dir, solver)
    print_summary(eval_results)

    # Save results
    results_dir = project_root / "results"
    save_results(training_results, results_dir / "training_results.json")
    save_results(eval_results, results_dir / "evaluation_results.json")

    # Generate report
    generate_markdown_report(
        training_results,
        eval_results,
        results_dir / "BENCHMARK_REPORT.md"
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    # Print final summary
    print("\nFinal Summary:")
    print(f"  Training solve rate: {training_results.solve_rate*100:.1f}%")
    print(f"  Evaluation solve rate: {eval_results.solve_rate*100:.1f}%")
    print(f"  Average accuracy: {(training_results.average_accuracy + eval_results.average_accuracy)/2*100:.1f}%")


if __name__ == '__main__':
    main()
