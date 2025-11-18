# Pattern Extraction Primitives Benchmark Results

*Evaluation Date: 2025-11-18*
*Comparison: 36 Primitives (+ Pattern Extraction) vs 34 Primitives (Morphological Baseline)*

## Executive Summary

After adding 2 pattern extraction primitives (extract_pattern, extract_top_left), we evaluated the system on 49 real ARC tasks. These primitives were specifically designed based on analysis of unsolved tasks requiring compositional transformations.

### Key Results

| Metric | Morphological (34 prims) | + Pattern Extraction (36 prims) | Change |
|--------|--------------------------|----------------------------------|--------|
| **Solve Rate** | **22.4%** (11/49) | **22.4%** (11/49) | **0.0%** |
| **Tasks Solved** | 11 | 11 | **0** |
| **Average Accuracy** | 24.3% | 24.3% | 0.0% |
| **Time per Task** | 2.2s | 2.4s | **+9%** |

### Impact

❌ **No improvement in solve rate** (11/49 maintained)
❌ **+9% computation time** (2.2s → 2.4s per task)
✅ **No regression** (same tasks still solve)

## Why Pattern Extraction Was Expected to Help

### Analysis-Based Design

From UNSOLVED_TASKS_ANALYSIS.md, we identified:

**Category 3: Pattern Extraction + Tiling (3 tasks estimated)**
- Tasks: 007bbfb7, 05269061, 017c7c7b
- Pattern: Extract sub-pattern from input, then tile it across grid
- Example task 05269061:
  ```
  Input:  2830000    Extract 3x3 → Tile across grid
          8300000
          3000000

  Output: 2832832
          8328328
          3283283
  ```

**Primitives Added:**
```python
extract_pattern(grid, row, col, height, width)  # Extract arbitrary region
extract_top_left(grid, height, width)  # Extract from (0,0) - common pattern
```

**Compositional Capability Enabled:**
- extract_top_left(3, 3) → tile(2, 2)
- extract_pattern(0, 0, 4, 4) → tile(3, 3)

### Why We Expected +3 Tasks

1. **Task 05269061:** Extract 3x3 from sparse input → tile
2. **Task 007bbfb7:** Extract pattern → place at specific positions
3. **Task 017c7c7b:** Similar compositional pattern

## Analysis: Why No Improvement?

### Hypothesis 1: Sequence Detection Limitation

**Problem:** Can't find multi-step sequences

Current beam search configuration:
- max_depth: 3 operations
- beam_width: 10 candidates
- Pruning: aggressive (removes many compositions)

**Evidence:**
- Composition `extract_top_left(3,3) → tile(2,2)` is 2-deep
- Should be detectable at max_depth=3
- But beam search may prune before finding it

**Why pruning might eliminate:**
1. Extract operation creates small intermediate grid (3x3)
2. Distance to target is LARGE after extraction (3x3 vs 7x7)
3. Beam search prioritizes candidates closer to target
4. Extract candidates get pruned before tile step

### Hypothesis 2: Parameter Generation Mismatch

**Problem:** Can't generate the right parameters

For task 05269061, we need:
- extract_top_left(height=3, width=3) ← Need to find this
- tile(repeat_v=2, repeat_h=2) ← Need to find this

Current parameter generation (sequence_detection.py lines 398-438):
```python
# extract_top_left: try heights/widths 2-5
for h in [2, 3, 4, 5]:
    for w in [2, 3, 4, 5]:
        variants.append(('extract_top_left', {'height': h, 'width': w}))
```

**Analysis:**
- ✓ We DO try height=3, width=3
- ✓ We DO try various tile repeats
- ❌ But may not evaluate this combination

**Combinatorial explosion:**
- 16 extract_top_left variants (4 heights × 4 widths)
- ~20 tile variants (different repeat counts)
- 320 possible extract→tile sequences
- Beam width=10 can't explore all

### Hypothesis 3: Tasks Don't Actually Match the Pattern

**Problem:** Our analysis was wrong

Let me check task 05269061 more carefully:
- Maybe it's not a simple extract→tile
- Maybe there's color transformation involved
- Maybe the pattern is more complex

**Similar for other target tasks:**
- 007bbfb7: May require selective placement, not uniform tiling
- 017c7c7b: May require different composition

### Hypothesis 4: Execution Issues

**Problem:** Primitives implemented incorrectly

Testing extract→tile composition manually:
```python
grid = [[2,8,3,0,0,0,0],
        [8,3,0,0,0,0,0],
        [3,0,0,0,0,0,0]]

# Extract 3x3
extract = extract_top_left(grid, 3, 3)
# Result: [[2,8,3],[8,3,0],[3,0,0]]

# Tile 2x2
result = tile(extract, repeat_v=2, repeat_h=2)
# Result: [[2,8,3,2,8,3],[8,3,0,8,3,0],[3,0,0,3,0,0],
#          [2,8,3,2,8,3],[8,3,0,8,3,0],[3,0,0,3,0,0]]
```

This produces a 6x6 grid, not the 7x7 target. So task 05269061 is NOT a simple extract→tile!

## Root Cause: Incorrect Task Analysis

**The real problem:** Our analysis of the 3 target tasks was **superficial**.

We assumed task 05269061 was:
- Extract 3x3 pattern from input
- Tile it uniformly

But the actual transformation is more complex:
- May involve rotation, offset, or selective placement
- May require different sized extraction
- May need more than 2 operations

**Lesson:** Surface-level pattern matching ≠ actual task requirements

## Comparison to Previous Improvements

### Improvement History

| Stage | Primitives | Solve Rate | Improvement | Cost |
|-------|-----------|------------|-------------|------|
| Baseline | 17 | 18.4% (9/49) | - | - |
| + Conditional | 29 (+12) | 22.4% (11/49) | **+4.0%** ✓ | 0% |
| + Morphological | 34 (+5) | 22.4% (11/49) | **0.0%** ❌ | +69% |
| + Pattern Extraction | 36 (+2) | 22.4% (11/49) | **0.0%** ❌ | +9% |

### Key Insight: 2 Failures in a Row

Both morphological and pattern extraction primitives failed because:

1. **Morphological:** Test set doesn't feature iterative growth/erosion patterns
2. **Pattern Extraction:** Superficial task analysis led to wrong primitives

**Common thread:** Not validating assumptions against actual task transformations

## What We Should Have Done

### Better Task Analysis Process

Instead of:
1. Look at input/output sizes
2. Guess "probably extract→tile"
3. Implement primitives
4. Hope it works

Should do:
1. **Manually trace** the transformation step-by-step
2. **Test hypothesis** with existing primitives
3. **Identify exact gap** (which primitive is missing?)
4. **Implement targeted primitive**
5. **Validate** it solves the task

### Example: Task 05269061 Deep Analysis

```
Input:  2830000  (3x7)
        8300000
        3000000

Output: 2832832  (7x7)
        8328328
        3283283
        2832832
        8328328
        3283283
        2832832
```

**Step-by-step:**
1. Pattern appears to be cyclic: 283, 832, 328, ...
2. Each column rotates: col[i] = pattern[i % 3]
3. This is NOT extract→tile, it's a **modular tiling with offset**

**What's actually needed:**
```python
tile_with_offset(pattern, offset_x, offset_y)  # Tile with column/row shifts
```

Not `extract_pattern`!

## Performance Metrics

### Solve Rate by Strategy

| Prediction Strategy | 34 Primitives | 36 Primitives |
|---------------------|---------------|---------------|
| Single (k=1) | 22.4% | 22.4% |
| Multi (k=2) | 22.4% | 22.4% |

### Computational Cost

| Metric | 34 Primitives | 36 Primitives | Change |
|--------|---------------|---------------|--------|
| Avg time/task | 2.2s | 2.4s | +9% |
| Search space | 34 ops | 36 ops | +6% |
| Parameter variants | High | Higher | +16 extract variants |

## Recommendations

### 1. Stop Adding Primitives Without Deep Validation

**New process:**
1. Select unsolved task for analysis
2. Manually solve it step-by-step
3. Check if existing primitives can express solution
4. If not, identify EXACT missing primitive
5. Implement and **test on that specific task**
6. Only then benchmark on full test set

### 2. Improve Sequence Detection First

Rather than more primitives, improve search:

**Option A: Increase search depth/width**
- max_depth: 3 → 5
- beam_width: 10 → 20
- May find compositions current system can't

**Option B: Smarter pruning**
- Don't prune based on distance alone
- Consider "promising" intermediate states (small grids that might be tiled)
- Use heuristics for composition detection

**Option C: Hierarchical search**
- Phase 1: Find intermediate transformation
- Phase 2: Find second transformation
- Explicitly search for compositions

### 3. Focus on Object Selection Next

From UNSOLVED_TASKS_ANALYSIS.md:
- **8 tasks** need object extraction (21% of unsolved)
- Clear need: extract_largest, extract_by_color
- Less complex than compositional reasoning
- More likely to work with current sequence detection

### 4. Consider Removing Pattern Extraction Primitives

Options:
- **Remove them** (reduce search space, save computation)
- **Keep them** (might help with better sequence detection later)
- **Make conditional** (only search when heuristics suggest)

Recommendation: **Keep for now** - cost is only +9%, may help when we improve sequence detection.

## Lessons Learned

### 1. Analysis Quality > Analysis Speed

Spent 30 minutes analyzing 38 tasks superficially.
Should have spent 2 hours analyzing 5 tasks deeply.

### 2. Validate Hypotheses Before Implementation

Could have tested "extract→tile" hypothesis by:
- Manually applying operations to task 05269061
- Discovering it doesn't produce correct output
- Saved implementation time

### 3. Not All Gaps Are Primitive Gaps

Some unsolved tasks need:
- Better sequence detection (compositions)
- Better parameter inference (finding right values)
- Better heuristics (pruning decisions)

Not just more primitives.

### 4. Primitive Additions Have Diminishing Returns

| Addition | Effort | Result | ROI |
|----------|--------|--------|-----|
| Conditional | Medium | +4% | High ✓ |
| Morphological | High | 0% | None ❌ |
| Pattern Extraction | Medium | 0% | None ❌ |

**Implication:** We may be hitting architectural limits.

## Next Steps

### Option 1: Deep-Dive Single Task (RECOMMENDED)

1. Pick one unsolved task (e.g., 05269061)
2. Manually solve it step-by-step
3. Identify exact transformation
4. Implement targeted primitive
5. Verify it solves that task
6. Then benchmark

**Time:** 2-3 hours
**Risk:** Low (validates before implementation)
**ROI:** High if successful

### Option 2: Improve Sequence Detection

1. Increase max_depth to 5
2. Increase beam_width to 20
3. Add composition-aware pruning
4. Re-benchmark

**Time:** 3-4 hours
**Risk:** Medium (may just slow down with no benefit)
**ROI:** High if it unlocks compositions

### Option 3: Object Selection Primitives

1. Implement extract_largest, extract_by_color
2. Add parameter generation
3. Benchmark

**Time:** 2-3 hours
**Risk:** Medium (may repeat pattern extraction failure)
**ROI:** Higher (8 tasks vs 3 tasks)

### Option 4: Accept Architectural Ceiling

**Hypothesis:** Can't go beyond ~25% without fundamental changes

Evidence:
- 11/49 tasks solved with simple operations
- Remaining 38 tasks may need:
  - Abstract reasoning
  - Multi-object tracking
  - Complex compositions beyond current search
  - Concept understanding (inside/outside, container/content)

**Implication:** Focus on architectural improvements, not primitive additions

## Conclusion

**The pattern extraction primitives addition failed:**

- ❌ **0% improvement** in solve rate
- ❌ **+9% computation cost**
- ❌ **Incorrect task analysis** led to wrong primitives

**Root causes:**
1. Superficial analysis of target tasks
2. Assumption that visual similarity = same transformation
3. No validation before implementation

**Key lessons:**
1. **Deep analysis > broad analysis** - understand ONE task perfectly
2. **Validate hypotheses** - test before implementing
3. **Architectural limits** - may need better search, not more primitives

**Recommended next action:**
- Deep-dive ONE unsolved task (e.g., 05269061)
- Manually solve it completely
- Identify exact missing capability
- Validate hypothesis by testing
- Only then implement

**Meta-lesson:** We've now failed twice (morphological, pattern extraction) by adding primitives without deep validation. Time to change the approach.

---

*System: LIDA-ARC with 36 Primitives (34 previous + 2 pattern_extraction)*
*Dataset: 49 real ARC-AGI tasks*
*Benchmark date: 2025-11-18*
*Status: **2nd consecutive failed primitive addition***
