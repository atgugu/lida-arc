# Sequence Detection Analysis: Current Limitations & Improvement Plan

*Analysis Date: 2025-11-18*
*Current System: 36 Primitives, 22.4% Solve Rate (11/49 tasks)*

## Executive Summary

After 2 consecutive failed primitive additions (morphological 0%, pattern extraction 0%), we're pivoting to **improve sequence detection** instead of adding more primitives. The hypothesis: existing primitives may already be able to solve more tasks if we can find the right multi-step compositions.

**Key Findings:**
- Current system solves mostly **single-operation tasks** (rotations, reflections, recolors)
- **Compositional transformations** (extract→tile, conditional→geometric) are not being found
- Beam search limitations prevent discovering promising intermediate states
- max_depth=3 and beam_width=10 may be too conservative

## Current Sequence Detection Configuration

### From `src/lida/arc/sequence_detection.py`:

```python
class SequenceDetector:
    def __init__(self, primitives: PrimitiveLibrary, max_depth: int = 3, beam_width: int = 5):
        self.primitives = primitives
        self.max_depth = max_depth  # Maximum sequence length
        self.beam_width = beam_width  # Candidates kept at each depth
        self.pruner = SequencePruner()
```

### Actual Usage (in `src/lida/arc/demonstration.py` line 105):

```python
self.sequence_detector = SequenceDetector(primitive_library, max_depth=3, beam_width=10)
```

**Current Settings:**
- **max_depth = 3** operations
- **beam_width = 10** candidates per depth
- **Aggressive pruning** of inverse pairs, redundant sequences, color→geometric orders

## Pruning Rules Analysis

###  1. Inverse Pair Pruning

```python
INVERSE_PAIRS = {
    ('rotate_90', 'rotate_270'),  # Cancel out
    ('rotate_270', 'rotate_90'),
    ('rotate_180', 'rotate_180'),  # Self-inverse
    ('reflect_horizontal', 'reflect_horizontal'),
    ('reflect_vertical', 'reflect_vertical'),
    ('reflect_diagonal', 'reflect_diagonal'),
}
```

**Impact:** ✓ Good - prevents wasteful cancellations

### 2. Redundant Pair Pruning

```python
REDUNDANT_PAIRS = {
    ('rotate_90', 'rotate_90'),  # Can be rotate_180
    ('rotate_90', 'rotate_180'),  # Can be rotate_270
    ('rotate_180', 'rotate_270'),  # Can be rotate_90
}
```

**Impact:** ✓ Good - simplifies rotation sequences

### 3. Rotation Count Limit

```python
rotation_count = sum(1 for op in sequence if op in rotation_ops)
if rotation_count > 1:
    return False  # Only 1 rotation allowed
```

**Impact:** ✓ Good - prevents multiple rotations (can be combined)

### 4. Reflection Count Limit

```python
reflection_count = sum(1 for op in sequence if op in reflection_ops)
if reflection_count > 2:
    return False  # At most 2 reflections
```

**Impact:** ✓ Reasonable - allows rotation+reflection but not 3+ reflections

### 5. Color-Then-Geometric Pruning

```python
# Color operations generally come last
# Rotating after recoloring is uncommon in ARC tasks

if last_color_idx != -1:
    for i in range(last_color_idx + 1, len(sequence)):
        if sequence[i] in geometric_ops:
            return False  # No geometric after color
```

**Impact:** ⚠️ **PROBLEMATIC** - This assumes a specific task structure!

Some tasks MAY need: recolor → rotate (e.g., color-code regions then rotate)

###  6. Beam Pruning by Distance

```python
def prune_beam(self, candidates, beam_width):
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.distance_to_target, len(c.sequence), -c.confidence)
    )
    return sorted_candidates[:beam_width]  # Keep top beam_width
```

**Impact:** 🔴 **MAJOR PROBLEM for compositions!**

**Why it's problematic:**
- Candidates are sorted by **distance to target**
- Compositions like **extract→tile** create intermediate states FAR from target
- Example: extract_top_left(3x3) on 7x7 grid:
  - Intermediate: 3x3 grid (distance to 7x7 target = 1.0, maximum!)
  - Gets pruned before tile step can be tried
  - But tile(2,2) would create the 6x6 target (close!)

**This is likely the #1 reason compositions aren't being found.**

## Solved Tasks Analysis (11/49)

### Solved Task IDs:
1. 00d62c1b - Recolor/conditional
2. 1e0a9b12 - Rotation/reflection
3. 2f876c35 - reflect_horizontal (2x2 grid)
4. 3c9b0459 - recolor (color mapping 1→3, 2→4)
5. 4be741c5 - rotate_180 (2x2 grid)
6. 5bd6f4ac - (unknown)
7. 6e82a1ae - (unknown)
8. 7df24a62 - (unknown)
9. 8f2ea7aa - (unknown)
10. 08ed6ac7 - (unknown)
11. 1cf80156 - (unknown)

### Pattern from First 5:
- **4/5 are same-size transformations** (6x6→6x6, 4x4→4x4, 2x2→2x2, 2x3→2x3)
- **Simple operations:** Single geometric (rotation, reflection) or single color mapping
- **No multi-step compositions observed**

**Hypothesis:** Current system solves **single-operation tasks** well, struggles with **multi-step compositions**.

## Why Compositions Fail

### Example: extract_top_left(3x3) → tile(2x2)

**Step 1: Initial state**
- Input: 7x7 grid with sparse pattern in top-left
- Target: 7x7 grid with tiled pattern
- Distance to target: 0.6 (high)

**Step 2: After extract_top_left(3x3)**
- Current: 3x3 grid (extracted pattern)
- Target: 7x7 grid
- **Distance to target: 1.0 (MAXIMUM!)**
- **This candidate gets pruned** (worst distance in beam)

**Step 3: (NEVER REACHED)**
- Would apply tile(2x2)
- Result: 6x6 grid (close to 7x7 target)
- Distance: ~0.1 (excellent!)
- But we never get here because step 2 was pruned

**Root cause:** Distance-based pruning can't recognize "promising" intermediate states.

## Improvement Options

### Option 1: Increase Search Budget (Low Risk, Quick)

**Changes:**
- max_depth: 3 → 5 (explore longer sequences)
- beam_width: 10 → 20 (keep more candidates)

**Pros:**
- Simple to implement (2-line change)
- May find compositions current system misses
- No new code, just parameter tuning

**Cons:**
- Computational cost increases (~4x time)
- May still miss compositions if pruning is wrong
- Doesn't fix fundamental beam pruning issue

**Estimated impact:** +2-5% solve rate (1-2 tasks)

### Option 2: Composition-Aware Pruning (Medium Risk, Medium Effort)

**Changes:**
- Don't prune based on distance alone
- Recognize "promising" intermediate states:
  - Small grids (might be tiled)
  - Clean patterns (might be replicated)
  - Intermediate results from extraction/transformation ops

**Implementation:**
```python
def is_promising_intermediate(self, candidate, target):
    # Don't prune if candidate just did an extraction
    if candidate.sequence and candidate.sequence[-1] in ['extract_pattern', 'extract_top_left']:
        return True  # Keep for potential tiling

    # Don't prune if small grid might be tiled to target size
    if candidate.current_grid:
        h, w = len(candidate.current_grid), len(candidate.current_grid[0])
        target_h, target_w = len(target), len(target[0])
        if h < target_h and target_h % h <= 1:  # Might tile vertically
            if w < target_w and target_w % w <= 1:  # Might tile horizontally
                return True

    return False

def prune_beam(self, candidates, beam_width):
    # Separate promising from non-promising
    promising = [c for c in candidates if self.is_promising_intermediate(c, target)]
    others = [c for c in candidates if not self.is_promising_intermediate(c, target)]

    # Keep all promising + top non-promising to fill beam
    sorted_others = sorted(others, key=lambda c: (c.distance_to_target, len(c.sequence)))
    return promising + sorted_others[:max(0, beam_width - len(promising))]
```

**Pros:**
- Addresses root cause (distance-based pruning problem)
- Enables composition discovery
- Still maintains beam size limit

**Cons:**
- More complex implementation
- Risk of keeping too many bad candidates
- Heuristics may be task-specific

**Estimated impact:** +5-10% solve rate (3-5 tasks)

### Option 3: Hierarchical Search (High Risk, High Effort)

**Changes:**
- Two-phase search:
  1. Find intermediate transformations (ignore distance to final target)
  2. Find second transformation (from intermediate to target)

**Implementation:**
```python
def find_compositional_sequence(self, input_grid, output_grid):
    # Phase 1: Try single-op transformations, collect all results
    intermediates = []
    for op in self.primitives.get_all_names():
        result = apply_operation(input_grid, op)
        intermediates.append((op, result))

    # Phase 2: Try second op from each intermediate
    for op1, intermediate in intermediates:
        for op2 in self.primitives.get_all_names():
            result = apply_operation(intermediate, op2)
            if result == output_grid:
                return [op1, op2]

    return None
```

**Pros:**
- Explicitly searches for compositions
- No distance-based pruning issues
- Guarantees finding 2-step sequences if they exist

**Cons:**
- High computational cost (N² for 2-step, N³ for 3-step)
- Doesn't scale to longer sequences
- Major code refactoring

**Estimated impact:** +10-15% solve rate (5-7 tasks) but very expensive

### Option 4: Remove Color-After-Geometric Constraint (Low Risk, Quick)

**Changes:**
- Remove the rule that prevents geometric ops after color ops

**Code change:**
```python
# DELETE THIS:
if last_color_idx != -1:
    for i in range(last_color_idx + 1, len(sequence)):
        if sequence[i] in geometric_ops:
            return False  # Geometric after color is unusual
```

**Pros:**
- Simple 1-line change (comment out the check)
- Allows sequences like: recolor → rotate
- No computational cost

**Cons:**
- May increase search space slightly
- Could find "false positive" sequences

**Estimated impact:** +0-2% solve rate (0-1 tasks)

## Recommended Approach: Staged Implementation

### Stage 1: Quick Wins (2-3 hours)

1. **Increase search budget**
   - max_depth: 3 → 5
   - beam_width: 10 → 20
   - Estimated cost: +300% time per task (2.4s → 7s)

2. **Remove color-after-geometric constraint**
   - Comment out the pruning rule
   - No additional cost

**Run benchmark after Stage 1**
- If +3-5% improvement: composition-aware pruning may not be needed
- If 0% improvement: move to Stage 2

###Stage 2: Composition-Aware Pruning (3-4 hours)

3. **Implement is_promising_intermediate heuristic**
   - Recognize extraction ops
   - Recognize small grids that might tile
   - Recognize clean patterns

4. **Modify prune_beam to keep promising candidates**

**Run benchmark after Stage 2**
- If +5-10% improvement: SUCCESS
- If <3% improvement: consider Stage 3

### Stage 3: Advanced (if needed, 5-6 hours)

5. **Hierarchical search for 2-step compositions**
   - Only if Stages 1-2 don't work
   - High computational cost

## Success Metrics

**Minimal Success (Stage 1 only):**
- Solve rate: 22.4% → 25%+ (+2 tasks)
- Time per task: 2.4s → 8s (acceptable)
- Unlock some 2-step compositions

**Good Success (Stage 1 + 2):**
- Solve rate: 22.4% → 30%+ (+4-5 tasks)
- Time per task: 2.4s → 10s (acceptable)
- Reliably find extract→tile, conditional→geometric compositions

**Excellent Success (All stages):**
- Solve rate: 22.4% → 35%+ (+6-7 tasks)
- Time per task: 2.4s → 15s (borderline)
- Find complex 3-4 step compositions

## Risk Analysis

**Low Risk:**
- Increasing max_depth/beam_width (just slower, no correctness issues)
- Removing geometric-after-color constraint (small search space increase)

**Medium Risk:**
- Composition-aware pruning (heuristics might keep bad candidates)
- Possible to keep too many candidates, explode search space

**High Risk:**
- Hierarchical search (major refactoring, high computational cost)
- May not be feasible for sequences >2 operations

## Next Steps

1. ✅ Analyze current limitations (DONE)
2. ✅ Analyze solved tasks (DONE)
3. ⏭️ **Implement Stage 1 improvements**
   - Increase max_depth to 5
   - Increase beam_width to 20
   - Remove color→geometric constraint
4. ⏭️ **Benchmark Stage 1**
   - Compare to 22.4% baseline
   - Measure time cost
   - Identify which new tasks are solved
5. ⏭️ **If needed: Implement Stage 2**
   - Composition-aware pruning
   - Benchmark again

## Conclusion

The current sequence detection system is **biased toward single-operation transformations** due to:
1. Low max_depth (3) and beam_width (10)
2. Distance-based pruning that eliminates promising intermediates
3. Restrictive ordering constraints (no geometric after color)

**Staged improvements** starting with simple parameter increases may unlock 2-5 additional tasks (9-22% relative improvement) with acceptable computational cost.

**Key insight:** Not all improvements require new primitives. Better search may unlock existing capabilities.

---

*Status: Ready to implement Stage 1 improvements*
