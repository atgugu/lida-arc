# Month 1 Proof of Concept: Self-Play Puzzle Generation

*Progress Report: 2025-11-18*
*Status: In Progress - Solvability Testing*

## Overview

Implementing Month 1 of the self-play architecture to overcome the data bottleneck (only 400 public ARC tasks). Target: Generate 1K synthetic puzzles with 80% validity and 40-60% solvability.

## Implementation Status

### ✅ Completed Components

**1. Core Module: `src/lida/arc/self_play.py`**
- **GridGenerator:** Creates random input grids
  - Sparse grids (15-25% non-zero pixels)
  - Dense grids (60-80% non-zero pixels)
  - Structured grids (geometric patterns)
  - At least 3 non-zero pixels to ensure non-trivial transformations

- **TransformationSampler:** Samples primitive sequences by difficulty
  - Easy: 1 operation
  - Medium: 2-3 operations
  - Hard: 3-5 operations
  - Currently uses simple geometric ops (rotate, reflect)

- **PrimitiveCompositionGenerator:** Generates complete puzzles
  - Creates train + test pairs
  - Applies transformation consistently across all demonstrations
  - Retry logic (up to 5 attempts) to avoid identity transformations
  - Returns None if generation fails

- **PuzzleValidator:** Validates generated puzzles
  - Non-empty grids
  - Non-trivial transformations (output ≠ input)
  - Consistent dimensions
  - Input diversity across demonstrations

- **SelfPlayStatistics:** Tracks generation metrics
  - Total generated / valid / solved
  - Invalid reasons breakdown
  - Statistics by difficulty level
  - Solve time distribution

**2. Generation Script: `scripts/generate_self_play_puzzles.py`**
- Command-line interface for puzzle generation
- Configurable difficulty distribution
- Optional solvability testing
- JSON output for training data
- Detailed statistics reporting

### 🔄 In Progress

**Solvability Testing (50 puzzles)**
- Testing synthetic puzzles with current solver
- Running in background (estimated ~2 minutes for 50 puzzles)
- Will determine if we hit 40-60% solvability target

## Initial Results

### Test 1: Small Sample (10 puzzles, no solving)
```
Total generated: 13
Valid puzzles: 10 (76.9%)
Invalid: 3 (23.1% - all "Trivial transformation")

By difficulty:
  easy: 3/6 valid (50.0%)
  medium: 7/7 valid (100.0%)
```

**Issue identified:** Some transformations on sparse grids produced identity (output = input)

### Test 2: After Improvements (50 puzzles, no solving)
```
Total generated: 51
Valid puzzles: 50 (98.0%) ✓
Invalid: 1 (2.0% - trivial transformation)

By difficulty:
  easy: 15/15 valid (100.0%)
  medium: 29/30 valid (96.7%)
  hard: 6/6 valid (100.0%)
```

**Improvements made:**
1. Increased sparse grid density (10-20% → 15-25%)
2. Minimum 3 non-zero pixels per grid
3. Retry logic with up to 5 attempts
4. Check non-triviality during generation

**Result:** **98% validity rate** - Exceeds 80% target! ✓

### Test 3: Solvability Testing (50 puzzles, WITH solving)
**Status:** Running...
**Expected completion:** ~2 minutes
**Will check:** Do we hit 40-60% solve rate target?

## Technical Highlights

### Retry Logic for Quality

```python
def generate_puzzle(self, difficulty, max_retries=5):
    for attempt in range(max_retries):
        # Generate transformation
        transformation = sample_transformation(difficulty)

        # Generate demos
        train_pairs = []
        has_non_trivial = False

        for _ in range(num_demos):
            input_grid = generate_grid()
            output_grid = apply(input_grid, transformation)

            # Check non-triviality
            if input_grid != output_grid:
                has_non_trivial = True

            train_pairs.append((input_grid, output_grid))

        # Only return if at least one demo is non-trivial
        if has_non_trivial:
            return puzzle

    return None  # Failed after retries
```

### Validation Checks

1. **Non-empty:** All grids must have content
2. **Non-trivial:** Output must differ from input
3. **Consistent:** Same transformation works on all demonstrations
4. **Diverse:** Training inputs vary

## Next Steps

### Immediate (Today)
1. ⏳ Complete solvability test on 50 puzzles
2. ⏳ Analyze solve rate by difficulty
3. ⏳ Validate 40-60% target

### If Solvability is Too Low (<40%)
**Problem:** Puzzles too hard

**Solutions:**
- Increase proportion of easy puzzles
- Add more single-operation transformations
- Use more structured input grids (less randomness)

### If Solvability is Too High (>60%)
**Problem:** Puzzles too easy

**Solutions:**
- Increase proportion of hard puzzles
- Add parametric operations (tile, scale)
- Use more diverse transformation sequences

### If Solvability is in Target (40-60%)
**Next:** Generate full 1K puzzles!

```bash
python scripts/generate_self_play_puzzles.py \
    --num-puzzles 1000 \
    --output-dir results/self_play_1k \
    --easy 0.3 \
    --medium 0.5 \
    --hard 0.2
```

**Estimated time:** ~40-50 minutes
- Generation: ~1 minute (fast)
- Solvability testing: ~40 minutes (50s for 1K at 2.4s/puzzle)

## Metrics Tracking

### Generation Efficiency

| Test | Attempts | Valid | Validity Rate | Time |
|------|----------|-------|---------------|------|
| Test 1 (10) | 13 | 10 | 76.9% | <1s |
| Test 2 (50) | 51 | 50 | 98.0% ✓ | <1s |
| Test 3 (50) | 51 | 50 | 98.0% ✓ | <1s |

**Observation:** Generation is very fast (<1s for 50 puzzles). Bottleneck is solvability testing.

### Solvability (Pending)

| Difficulty | Generated | Solved | Solve Rate |
|------------|-----------|--------|------------|
| Easy | 15 | ? | ? |
| Medium | 29 | ? | ? |
| Hard | 6 | ? | ? |
| **Total** | **50** | **?** | **?** (target: 40-60%) |

## Challenges Encountered

### Challenge 1: Identity Transformations
**Problem:** Some primitive compositions (especially on sparse grids) resulted in output = input

**Root cause:**
- Geometric operations on mostly-empty grids
- Example: reflect_horizontal on symmetric empty grid
- Example: rotate_180 on single center pixel

**Solution:**
- Increased sparse grid density
- Added minimum pixel count
- Retry logic with non-triviality check
- Result: 76.9% → 98.0% validity

### Challenge 2: Validation Complexity
**Problem:** Need multiple validation checks to ensure quality

**Solution:** Layered validation
1. Generation-time: Check non-triviality during creation
2. Validation-time: Comprehensive checks (consistency, diversity)
3. Statistics-time: Track reasons for failures

## Future Enhancements (Post-Month 1)

### Month 2: Quality Control
- **Diversity scoring:** Measure similarity to existing puzzles
- **Difficulty scoring:** Estimate solve time
- **Educational value:** Target under-represented patterns
- **Evaluator module:** Assign quality scores

### Month 2-3: Parametric Operations
- Add operations with parameters (tile, scale, recolor)
- Parameter sampling logic
- Increases puzzle diversity
- May affect solvability (likely reduces it)

### Month 3: Curriculum Learning
- Adaptive difficulty based on solve rates
- Target: 80% easy, 50% medium, 20% hard
- Adjust sampling probabilities dynamically

### Month 3-4: Scale to 10K
- Parallel generation
- Batch processing
- Database storage
- Training data pipeline

## Success Criteria (Month 1 PoC)

| Criterion | Target | Current | Status |
|-----------|--------|---------|--------|
| **Validity Rate** | ≥80% | **98.0%** | ✅ **EXCEEDED** |
| **Solvability Rate** | 40-60% | **Testing...** | ⏳ Pending |
| **Puzzle Count** | 1K | 50 (test) | ⏳ Ready to scale |
| **Generation Speed** | <5 min | <1s (50 puzzles) | ✅ Fast |
| **Diversity** | 10+ patterns | Moderate | ⏳ To measure |

## Conclusion (Preliminary)

**Achieved so far:**
- ✅ 98% validity rate (target: 80%)
- ✅ Fast generation (<1s for 50 puzzles)
- ✅ Clean implementation with retry logic
- ✅ Comprehensive validation

**Pending:**
- ⏳ Solvability rate testing (in progress)
- ⏳ Full 1K generation
- ⏳ Final validation report

**Key insight:** Generation works well! Validity is high, speed is excellent. The critical unknown is **solvability** - will our generated puzzles challenge the solver appropriately?

If solvability is in the 40-60% range, we've successfully proven the concept and can proceed to generate 1K+ puzzles for training data.

---

*Status: Waiting for solvability test results...*
*Next update: After 50-puzzle test completes*
