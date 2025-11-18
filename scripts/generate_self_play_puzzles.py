#!/usr/bin/env python
"""
Generate synthetic ARC puzzles using self-play.

Month 1 Proof of Concept:
- Generate 1K puzzles
- Validate quality (target: 80% valid)
- Test solvability (target: 40-60% solvable)
"""

import json
import time
from pathlib import Path
from typing import List
import argparse

from lida.arc.self_play import (
    PrimitiveCompositionGenerator,
    PuzzleValidator,
    SelfPlayStatistics,
    GeneratedPuzzle
)
from lida.arc.primitives import PrimitiveLibrary
from lida.arc.cognitive_solver import ARCCognitiveSolver, ARCSolverConfig


def generate_puzzles(num_puzzles: int, difficulty_distribution: dict,
                     output_dir: Path, test_solvability: bool = True):
    """Generate synthetic puzzles and optionally test them.

    Args:
        num_puzzles: Number of puzzles to generate
        difficulty_distribution: Dict with 'easy', 'medium', 'hard' probabilities
        output_dir: Directory to save results
        test_solvability: Whether to test with solver (slow)
    """
    print("="* 70)
    print("SELF-PLAY PUZZLE GENERATION - PROOF OF CONCEPT")
    print("=" * 70)
    print(f"\nTarget: Generate {num_puzzles} puzzles")
    print(f"Difficulty distribution: {difficulty_distribution}")
    print(f"Test solvability: {test_solvability}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize
    primitive_library = PrimitiveLibrary()
    generator = PrimitiveCompositionGenerator(primitive_library, seed=42)
    validator = PuzzleValidator()
    statistics = SelfPlayStatistics()

    if test_solvability:
        solver_config = ARCSolverConfig(verbose=False, debug=False, max_predictions=1)
        solver = ARCCognitiveSolver(solver_config)

    # Storage for valid puzzles
    valid_puzzles: List[GeneratedPuzzle] = []
    solved_puzzles: List[dict] = []

    # Difficulty sampling
    difficulties = list(difficulty_distribution.keys())
    probabilities = [difficulty_distribution[d] for d in difficulties]

    # Generation loop
    print("Generating puzzles...")
    start_time = time.time()

    attempts = 0
    max_attempts = num_puzzles * 3  # Allow up to 3x attempts to reach target

    while len(valid_puzzles) < num_puzzles and attempts < max_attempts:
        attempts += 1

        # Sample difficulty
        import random
        difficulty = random.choices(difficulties, weights=probabilities)[0]

        # Generate puzzle
        puzzle = generator.generate_puzzle(difficulty=difficulty)

        # Validate
        if puzzle is None:
            statistics.record_generation(None, False, "Generation failed", difficulty)
            continue

        is_valid, reason = validator.validate(puzzle)
        statistics.record_generation(puzzle, is_valid, reason, difficulty)

        if not is_valid:
            continue

        # Valid puzzle!
        valid_puzzles.append(puzzle)

        # Progress indicator
        if len(valid_puzzles) % 100 == 0:
            elapsed = time.time() - start_time
            rate = len(valid_puzzles) / elapsed
            print(f"  Generated {len(valid_puzzles)}/{num_puzzles} valid puzzles "
                  f"({attempts} attempts, {rate:.1f} valid/s)")

    generation_time = time.time() - start_time

    print(f"\nGeneration complete!")
    print(f"  Time: {generation_time:.1f}s")
    print(f"  Valid puzzles: {len(valid_puzzles)}")
    print(f"  Attempts: {attempts}")
    print(f"  Validity rate: {100*len(valid_puzzles)/attempts:.1f}%")

    # Test solvability
    if test_solvability and valid_puzzles:
        print(f"\nTesting solvability with current solver...")
        solve_start = time.time()

        for i, puzzle in enumerate(valid_puzzles):
            # Convert to ARCTask
            task = puzzle.to_arc_task()

            # Solve
            result = solver.evaluate(task, test_index=0)

            # Record
            solve_time = result.get('time', 0)
            statistics.record_solve_result(puzzle, result['solved'], solve_time)

            if result['solved']:
                solved_puzzles.append({
                    'puzzle_id': puzzle.puzzle_id,
                    'difficulty': puzzle.difficulty,
                    'transformation': puzzle.transformation,
                    'accuracy': result['accuracy'],
                    'time': solve_time
                })

            # Reset solver
            solver.reset()

            # Progress
            if (i + 1) % 100 == 0:
                elapsed = time.time() - solve_start
                rate = (i + 1) / elapsed
                solved_count = len(solved_puzzles)
                print(f"  Tested {i+1}/{len(valid_puzzles)} puzzles "
                      f"({solved_count} solved, {rate:.1f} tests/s)")

        solve_time = time.time() - solve_start
        print(f"\nSolve testing complete!")
        print(f"  Time: {solve_time:.1f}s")
        print(f"  Solved: {len(solved_puzzles)}/{len(valid_puzzles)} ({100*len(solved_puzzles)/len(valid_puzzles):.1f}%)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics
    stats_file = output_dir / "generation_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(statistics.get_report())
    print(f"\nSaved statistics to: {stats_file}")

    # Save puzzle data
    puzzles_data = []
    for puzzle in valid_puzzles:
        puzzles_data.append({
            'puzzle_id': puzzle.puzzle_id,
            'difficulty': puzzle.difficulty,
            'transformation': puzzle.transformation,
            'train': puzzle.train,
            'test': puzzle.test,
            'generation_params': puzzle.generation_params
        })

    puzzles_file = output_dir / "generated_puzzles.json"
    with open(puzzles_file, 'w') as f:
        json.dump(puzzles_data, f, indent=2)
    print(f"Saved {len(puzzles_data)} puzzles to: {puzzles_file}")

    # Save solved puzzles
    if solved_puzzles:
        solved_file = output_dir / "solved_puzzles.json"
        with open(solved_file, 'w') as f:
            json.dump(solved_puzzles, f, indent=2)
        print(f"Saved {len(solved_puzzles)} solved puzzles to: {solved_file}")

    # Print final statistics
    print("\n" + statistics.get_report())

    # Check targets
    print("\n" + "="*70)
    print("TARGET VALIDATION")
    print("="*70)

    validity_rate = 100 * len(valid_puzzles) / attempts if attempts > 0 else 0
    target_validity = 80.0
    validity_status = "✓" if validity_rate >= target_validity else "✗"
    print(f"{validity_status} Validity rate: {validity_rate:.1f}% (target: {target_validity}%)")

    if test_solvability and valid_puzzles:
        solve_rate = 100 * len(solved_puzzles) / len(valid_puzzles)
        target_solve_min = 40.0
        target_solve_max = 60.0
        solve_status = "✓" if target_solve_min <= solve_rate <= target_solve_max else "✗"
        print(f"{solve_status} Solve rate: {solve_rate:.1f}% (target: {target_solve_min}-{target_solve_max}%)")

        if solve_rate < target_solve_min:
            print(f"  ⚠ Puzzles too hard - need easier transformations")
        elif solve_rate > target_solve_max:
            print(f"  ⚠ Puzzles too easy - need harder transformations")
        else:
            print(f"  ✓ Difficulty is in target range!")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate self-play ARC puzzles")
    parser.add_argument('--num-puzzles', type=int, default=1000,
                       help='Number of puzzles to generate (default: 1000)')
    parser.add_argument('--no-solve-test', action='store_true',
                       help='Skip solvability testing (faster)')
    parser.add_argument('--output-dir', type=str, default='results/self_play',
                       help='Output directory (default: results/self_play)')
    parser.add_argument('--easy', type=float, default=0.3,
                       help='Probability of easy puzzles (default: 0.3)')
    parser.add_argument('--medium', type=float, default=0.5,
                       help='Probability of medium puzzles (default: 0.5)')
    parser.add_argument('--hard', type=float, default=0.2,
                       help='Probability of hard puzzles (default: 0.2)')

    args = parser.parse_args()

    # Normalize probabilities
    total = args.easy + args.medium + args.hard
    difficulty_distribution = {
        'easy': args.easy / total,
        'medium': args.medium / total,
        'hard': args.hard / total
    }

    output_dir = Path(args.output_dir)

    generate_puzzles(
        num_puzzles=args.num_puzzles,
        difficulty_distribution=difficulty_distribution,
        output_dir=output_dir,
        test_solvability=not args.no_solve_test
    )


if __name__ == '__main__':
    main()
