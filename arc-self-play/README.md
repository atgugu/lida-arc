# ARC Self-Play

Adversarial puzzle generation and neural guidance for [ARC-AGI](https://arcprize.org/).

Generate unlimited synthetic ARC training puzzles by composing grid primitives, then train lightweight neural networks to guide symbolic solvers.

## Why

ARC-AGI has only ~400 public training tasks -- far too few for data-hungry approaches. This project side-steps the bottleneck by **generating puzzles programmatically**: random grids go in, a randomly sampled chain of primitives transforms them, and out comes a new puzzle with known ground-truth. The generator can produce thousands of valid puzzles per second.

Because we know *exactly* which primitives were used, every generated puzzle provides perfect supervised labels for training neural guidance networks that can steer a symbolic solver's search.

## Quick start

```bash
pip install -e .            # base install (numpy only)
pip install -e ".[neural]"  # + PyTorch for neural guidance
pip install -e ".[dev]"     # + pytest / ruff
```

Generate 1000 puzzles:

```bash
python scripts/generate.py -n 1000 -o data/puzzles.json
```

Run tests:

```bash
pytest
```

## Architecture

```
arc_self_play/
├── primitives/          # Grid transformation operations
│   ├── operations.py    # Pure functions: rotate, reflect, tile, recolor, ...
│   └── library.py       # Named registry of all 9 primitives
├── generator/           # Puzzle generation engine
│   ├── core.py          # PuzzleGenerator — compose primitives into puzzles
│   ├── grids.py         # Random grid generators (sparse, dense, structured)
│   ├── sampler.py       # Difficulty-aware transformation sampling
│   ├── validator.py     # Quality checks (non-trivial, well-formed, diverse)
│   └── stats.py         # Generation run statistics
└── neural/              # Neural guidance for hybrid solving
    ├── encoding.py      # One-hot grid encoding to (10, H, W) tensors
    ├── labels.py        # Extract task / primitive / sequence labels
    └── dataset.py       # PyTorch Dataset + DataLoader utilities
```

## Primitives

Nine composable grid operations, each a pure function:

| Category | Primitives |
|----------|-----------|
| **Geometric** | `rotate_90`, `rotate_180`, `rotate_270`, `reflect_horizontal`, `reflect_vertical`, `reflect_diagonal` |
| **Spatial** | `auto_crop`, `tile` |
| **Color** | `recolor` |

```python
from arc_self_play.primitives import operations as ops

grid = [[1, 2], [3, 4]]
ops.rotate_90(grid)           # [[3, 1], [4, 2]]
ops.reflect_vertical(grid)    # [[2, 1], [4, 3]]
ops.tile(grid, 2, 2)          # 4x4 grid, pattern repeated 2x2
```

## Puzzle generation

The generator composes primitives into transformation chains of controlled difficulty:

| Difficulty | Chain length | Parametric ops | Typical solve rate |
|------------|-------------|----------------|-------------------|
| **easy** | 1 | none | ~100% |
| **medium** | 2--3 | 30% chance | ~70% |
| **hard** | 3--5 | 60% chance | ~40% |

```python
from arc_self_play.generator import PuzzleGenerator

gen = PuzzleGenerator(seed=42)

# Single puzzle
puzzle = gen.generate(difficulty="hard")
print(puzzle.transformation)  # [('rotate_90', {}), ('recolor', {...}), ...]

# Batch with difficulty distribution
puzzles = gen.generate_batch(
    n=1000,
    difficulty_weights={"easy": 0.1, "medium": 0.3, "hard": 0.6},
)
```

Each puzzle contains:
- **train**: 3 input-output demonstration pairs
- **test**: 1 held-out pair for evaluation
- **transformation**: ground-truth primitive chain
- **difficulty**: easy / medium / hard

### Validation

Every generated puzzle is checked for:
- Non-empty, well-formed grids
- Non-trivial transformations (output differs from input)
- Diverse training inputs (not all identical)

Retry logic automatically re-samples when a transformation produces an identity (e.g., reflecting a symmetric grid).

### Performance

Generating 1000 valid puzzles takes ~0.02 seconds. The bottleneck is solvability testing, not generation.

## Neural guidance

The neural module provides PyTorch infrastructure for training guidance networks on generated puzzles. Since we know the ground-truth transformation for every puzzle, we get perfect supervision for free.

### Three training signals

1. **Task classification** (5 categories: geometric, color_mapping, tiling, cropping, composition)
2. **Primitive relevance** (binary label per primitive -- which ones appear in the transformation?)
3. **Composition sequences** (given partial application, predict the next primitive)

### Usage

```python
from arc_self_play.primitives import PrimitiveLibrary
from arc_self_play.neural import create_dataloaders

lib = PrimitiveLibrary()
train_loader, val_loader, test_loader, dataset = create_dataloaders(
    puzzles_path="data/puzzles.json",
    all_primitives=lib.get_all_names(),
    batch_size=32,
)

for batch in train_loader:
    grid_pairs = batch["grid_pair"]        # (B, 2, 10, 30, 30)
    task_labels = batch["task_index"]       # (B,)  — 5 classes
    prim_labels = batch["primitive_labels"] # (B, 9) — binary
    break
```

### Grid encoding

Grids are one-hot encoded across 10 colour channels and zero-padded to 30x30:

```python
from arc_self_play.neural import encode_grid

grid = [[1, 0], [0, 2]]
tensor = encode_grid(grid)  # shape: (10, 30, 30)
```

### Composition dataset

For training sequence models (e.g., predict the next primitive):

```python
from arc_self_play.neural import CompositionDataset

comp_ds = CompositionDataset(dataset, max_seq=5)
# Each example: input_grid, primitive_sequence, next_primitive
```

## Hybrid solver concept

The intended use is a **hybrid symbolic-neural solver**:

```
ARC puzzle
    │
    ▼
Neural guidance (fast, learned)
    ├── Task classifier → focus search on relevant category
    ├── Primitive predictor → prune 9 → top-k primitives
    └── Composition predictor → suggest next primitive
    │
    ▼
Symbolic search (exact, compositional)
    └── Beam search over primitive compositions
        guided by neural scores
```

**Expected impact**: reduce search space from 9^5 ≈ 59K to ~5^5 ≈ 3K candidate sequences (20x speedup), while maintaining the correctness guarantees of symbolic execution.

## Project structure

```
arc-self-play/
├── arc_self_play/           # Main package
│   ├── types.py             # Grid, GridPair, ARCTask
│   ├── primitives/          # 9 composable operations
│   ├── generator/           # Puzzle generation engine
│   └── neural/              # PyTorch data pipeline
├── scripts/
│   └── generate.py          # CLI for batch generation
├── tests/                   # 25 tests
├── data/                    # Generated puzzles (gitignored)
└── pyproject.toml
```

## License

MIT
