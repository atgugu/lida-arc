#!/usr/bin/env python3
"""Generate synthetic ARC puzzles via primitive composition.

Usage::

    python scripts/generate.py --num 1000 --output data/puzzles.json
    python scripts/generate.py --num 100 --easy 0.1 --medium 0.3 --hard 0.6
"""

import argparse
import json
import time
from pathlib import Path

from arc_self_play.generator import PuzzleGenerator, GenerationStats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ARC puzzles")
    parser.add_argument("-n", "--num", type=int, default=100, help="Number of puzzles")
    parser.add_argument("-o", "--output", type=str, default="data/puzzles.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--easy", type=float, default=0.1)
    parser.add_argument("--medium", type=float, default=0.3)
    parser.add_argument("--hard", type=float, default=0.6)
    args = parser.parse_args()

    weights = {"easy": args.easy, "medium": args.medium, "hard": args.hard}

    print(f"Generating {args.num} puzzles  (distribution: {weights})")
    gen = PuzzleGenerator(seed=args.seed)
    stats = GenerationStats()

    t0 = time.time()
    puzzles = gen.generate_batch(args.num, difficulty_weights=weights, seed=args.seed)
    elapsed = time.time() - t0

    # Record stats
    for p in puzzles:
        stats.record_attempt(p.difficulty, True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([p.to_dict() for p in puzzles], f)

    print(f"\nGenerated {len(puzzles)} puzzles in {elapsed:.2f}s")
    print(f"Saved to {out}")
    print(stats.report())


if __name__ == "__main__":
    main()
