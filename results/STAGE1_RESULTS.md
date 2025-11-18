# Stage 1 Sequence Detection Improvements: Results & Analysis

*Evaluation Date: 2025-11-18*
*Changes: max_depth 3→5, beam_width 10→20, removed color→geometric constraint*

## Executive Summary

**Stage 1 improvements FAILED to increase solve rate despite 3.7x computational cost increase.**

| Metric | Baseline (36 prims) | Stage 1 (Improved Search) | Change |
|--------|---------------------|---------------------------|--------|
| **Solve Rate** | **22.4%** (11/49) | **22.4%** (11/49) | **0.0%** ❌ |
| **Tasks Solved** | 11 | 11 | **0** |
| **Avg Accuracy** | 24.3% | 23.6% | **-0.7%** |
| **Time/Task** | 2.4s | 8.8s | **+267%** 🔴 |

**Conclusion:** Simply increasing search budget does NOT unlock compositional transformations. The problem is **architectural**, not parametric.

## Stage 1 Changes

### 1. Increased Search Depth
```python
# Before:
max_depth = 3  # Explore sequences up to 3 operations

# After:
max_depth = 5  # Explore sequences up to 5 operations
```

**Rationale:** Multi-step compositions (extract→tile) need 2+ operations. Depth=3 may miss longer sequences.

**Result:** No new tasks solved.

### 2. Increased Beam Width
```python
# Before:
beam_width = 10  # Keep top 10 candidates at each depth

# After:
beam_width = 20  # Keep top 20 candidates at each depth
```

**Rationale:** Wider beam keeps more candidates, including "promising" intermediates that might lead to solutions.

**Result:** No new tasks solved, just 2x more candidates evaluated.

### 3. Removed Color→Geometric Constraint
```python
# Before:
if last_color_idx != -1:
    for i in range(last_color_idx + 1, len(sequence)):
        if sequence[i] in geometric_ops:
            return False  # Prune sequences like: recolor → rotate

# After:
# Constraint removed - allow any ordering
```

**Rationale:** Some tasks might need unconventional orderings like recolor→rotate.

**Result:** No new tasks solved. This constraint wasn't the limiting factor.

## Why Stage 1 Failed

### Hypothesis: Distance-Based Pruning is Still the Problem

Even with beam_width=20, the fundamental issue remains:

**Candidates are sorted by distance to target:**
```python
sorted_candidates = sorted(
    candidates,
    key=lambda c: (c.distance_to_target, len(c.sequence), -c.confidence)
)
return sorted_candidates[:beam_width]
```

**Example: extract_top_left(3x3) → tile(2x2)**

**Step 1: After extract_top_left**
- Input: 7x7 grid
- After extraction: 3x3 grid
- Distance to 7x7 target: **1.0** (maximum - completely different size!)
- Rank in beam: **LAST** (worst distance)
- With beam_width=20: Still gets pruned if there are 20+ better candidates

**Step 2: (NEVER REACHED)**
- Would apply tile(2,2)
- Result: 6x6 grid (close to target!)
- But we never get here

**Key insight:** Even beam_width=100 won't help if ALL 20+ single-operation candidates have better distance than the extract candidate.

### Analysis: Search Space Expansion

With max_depth=5 and beam_width=20:

**At each depth:**
- 36 primitives × parameter variants × 20 previous candidates
- Depth 1: ~36 × 5 = 180 candidates → keep top 20
- Depth 2: 20 × 36 × 5 = 3,600 candidates → keep top 20
- Depth 3: 20 × 36 × 5 = 3,600 candidates → keep top 20
- Depth 4: 20 × 36 × 5 = 3,600 candidates → keep top 20
- Depth 5: 20 × 36 × 5 = 3,600 candidates → keep top 20

**Total candidates evaluated:** ~14,580 per task (vs ~3,660 at depth=3)

**Increase:** 4x more evaluations
**Result:** Same 11 tasks solved

**Conclusion:** We're evaluating 4x more candidates but finding the SAME solutions (single-operation transformations).

### Why Compositions Still Aren't Found

Looking at the 11 solved tasks (from previous analysis):
- 2f876c35: reflect_horizontal (single-op)
- 3c9b0459: recolor (single-op)
- 4be741c5: rotate_180 (single-op)
- Others: Similar single-op transformations

**None use multi-step compositions!**

Even with increased search:
1. Single-op solutions have distance ~0.0-0.2 to target
2. Extract operations have distance ~1.0 (worst possible)
3. Even with beam_width=20, all 20 slots fill with single-ops
4. Extract ops never make it into the beam
5. Therefore tile step never gets tried

**The problem:** Distance-based ranking fundamentally prevents discovering compositions that require temporarily moving FAR from the target.

## Computational Cost Analysis

### Time Breakdown

| Configuration | Time/Task | Total (49 tasks) |
|---------------|-----------|------------------|
| Baseline (depth=3, width=10) | 2.4s | 118s (~2 min) |
| Stage 1 (depth=5, width=20) | 8.8s | 431s (~7 min) |

**Increase:** 3.7x slower

### Cost-Benefit Analysis

**Investment:** 3.7x more computation
**Return:** 0 additional tasks solved

**ROI:** **NEGATIVE** - Pure cost with no benefit

## Comparison to Previous Improvements

| Improvement | Primitives | Solve Rate | Time/Task | ROI |
|-------------|-----------|------------|-----------|-----|
| Baseline | 17 | 18.4% (9/49) | ~1.3s | - |
| + Conditional | 29 | 22.4% (11/49) | 1.3s | ✓ High (+4%, same cost) |
| + Morphological | 34 | 22.4% (11/49) | 2.2s | ❌ None (0%, +69% cost) |
| + Pattern Extract | 36 | 22.4% (11/49) | 2.4s | ❌ None (0%, +9% cost) |
| + **Stage 1 Search** | 36 | **22.4% (11/49)** | **8.8s** | ❌ **NEGATIVE** (0%, +267% cost) |

**Pattern:** Three consecutive failed improvements (morphological, pattern extraction, Stage 1 search)

**Common thread:** All assume the problem is missing primitives or insufficient search. But the real problem is **architectural**.

## Why Stage 2 (Composition-Aware Pruning) Likely Won't Help Either

Stage 2 plan was to add heuristics like:
```python
def is_promising_intermediate(candidate, target):
    if candidate.sequence[-1] in ['extract_pattern', 'extract_top_left']:
        return True  # Keep for potential tiling
    if small_grid_might_tile(candidate.current_grid, target):
        return True
    return False
```

**Problem:** This is task-specific and brittle.

**Why it won't scale:**
- Needs heuristics for EVERY type of composition
- extract→tile needs "small grid" heuristic
- Other compositions need other heuristics
- Infinite regress: need to know composition type to detect it!

**Better approach:** Don't use distance-based pruning at all for detecting compositions.

## Root Cause: Wrong Search Strategy

The current beam search optimizes for:
> "Find sequences that monotonically decrease distance to target"

But compositional transformations require:
> "Find sequences that temporarily INCREASE distance, then decrease it"

**Example trajectories:**

**Single-op (works with current search):**
```
Input (distance=0.6)
  → rotate_90
  → Output (distance=0.0) ✓
```

**Composition (fails with current search):**
```
Input (distance=0.6)
  → extract_top_left (distance=1.0) ← PRUNED HERE!
  → tile (distance=0.0) ← Never reached
```

**Fundamental mismatch:** Greedy distance-based search can't find non-monotonic paths.

## Lessons Learned

### 1. Increasing Search Budget ≠ Finding Better Solutions

Evaluating 4x more candidates doesn't help if we're searching the wrong space.

**Analogy:** Searching for keys under a lamppost because the light is good, even though you lost them in the dark.

We're searching in "low distance" space because it's easy to evaluate, even though solutions are in "high initial distance, then low" space.

### 2. The Ceiling is Architectural, Not Parametric

We've hit the limit of what the current architecture can do:

**Current architecture strengths:**
- Single-operation transformations ✓
- Transformations that monotonically approach target ✓
- Simple geometric and color operations ✓

**Current architecture weaknesses:**
- Multi-step compositions ❌
- Non-monotonic distance trajectories ❌
- Extract + combine patterns ❌
- Conditional logic chains ❌

**Solve rate ceiling:** ~22-25% with current approach

### 3. Heuristics are Band-Aids, Not Solutions

Adding composition-specific heuristics (Stage 2) would just be whack-a-mole:
- Add heuristic for extract→tile
- Add heuristic for conditional→geometric
- Add heuristic for pattern→replicate
- Add heuristic for...

**Problem:** Infinite regression. We can't anticipate all composition types.

**Real solution:** Change the search strategy entirely.

## Alternative Approaches

### Option 1: Two-Phase Search (Promising)

Don't use distance-based pruning at all.

**Phase 1: Generate all 1-step transformations**
```python
intermediates = []
for op in primitives:
    result = apply(input, op)
    intermediates.append((op, result))
# Don't prune by distance - keep ALL
```

**Phase 2: From each intermediate, try second operation**
```python
for op1, intermediate in intermediates:
    for op2 in primitives:
        result = apply(intermediate, op2)
        if matches(result, target):
            return [op1, op2]
```

**Pros:**
- No distance-based pruning
- Guaranteed to find 2-step sequences if they exist
- Explicit composition search

**Cons:**
- High computational cost: O(N²) for 2-step, O(N³) for 3-step
- With 36 primitives × 5 param variants = 180 ops
- 2-step: 180² = 32,400 evaluations
- 3-step: 180³ = 5,832,000 evaluations (infeasible!)

**Estimated impact:** +5-10% solve rate, but may be too slow for 3+ steps

### Option 2: Goal-Conditioned Search (Best)

Search backwards from target:

**Forward search (current):**
```
Input → ? → ? → Target
(Don't know what intermediates should be)
```

**Backward search:**
```
Target → reverse_op → intermediate → reverse_op → Input
(Each step gets closer to input)
```

**Pros:**
- Distance to input guides search (monotonically decreasing)
- Avoids the "high distance intermediate" problem
- Still uses beam search efficiently

**Cons:**
- Requires reversible operations (not all ops are reversible)
- extract_pattern is NOT reversible (lossy)
- May still struggle with some compositions

**Estimated impact:** +5-10% solve rate

### Option 3: Accept the Ceiling (Pragmatic)

**22.4% may be the limit of the primitive-based approach.**

Evidence:
- 3 consecutive failures (morphological, pattern extract, Stage 1)
- All failures point to same issue: architectural limitations
- Fundamental mismatch between search strategy and composition requirements

**Next steps if accepting ceiling:**
1. Focus on reducing cost (make 22.4% faster, not slower)
2. Analyze what 22.4% buys us (which tasks? which types?)
3. Pivot to different architecture (learned models, neural program synthesis)

## Recommendations

### Immediate: Revert Stage 1 Changes

**Reason:** Stage 1 makes system 3.7x slower with no benefit.

**Action:**
```
git revert HEAD  # Revert to baseline
```

**Result:** Back to 22.4% solve rate at 2.4s/task (acceptable performance)

### Short-Term: Try Option 1 (Two-Phase Search)

**Why:** Most promising for finding 2-step compositions without heuristics.

**Implementation:** 3-4 hours
- Phase 1: Generate all 1-step results
- Phase 2: Try all 2nd operations
- Skip distance-based pruning

**Test on:** 5-10 unsolved tasks that appear to need 2-step compositions

**Decision criteria:**
- If +3-5 tasks solved: Good ROI, keep it
- If 0-1 tasks solved: Abandon, accept ceiling

### Long-Term: Pivot to Learned Approaches

**If two-phase search fails:**
- Current primitive-based approach may be fundamentally limited
- 22.4% ceiling is real
- Need architectural change, not parameter tuning

**Alternatives:**
- Neural program synthesis (learn to generate sequences)
- Learned pattern recognizers (learn what intermediates are "promising")
- Hybrid: primitives + learned search heuristics

## Conclusion

**Stage 1 (increasing search budget) failed:**
- ❌ 0% improvement in solve rate
- ❌ 3.7x slower (267% cost increase)
- ❌ Didn't unlock any multi-step compositions

**Root cause:**
- Distance-based pruning prevents discovering non-monotonic composition paths
- extract→tile requires temporarily moving FAR from target
- Greedy search can't handle this

**Next steps:**
1. **Revert Stage 1** (restore baseline performance)
2. **Try two-phase search** (explicit composition search without distance pruning)
3. **If that fails: Accept 22.4% ceiling** and pivot to architectural changes

**Meta-lesson:** Three consecutive improvements failed (morphological, pattern extraction, Stage 1). Time to stop iterating on the same approach and try something fundamentally different.

---

*Status: Stage 1 failed. Recommending reversion and pivot to two-phase search or acceptance of architectural ceiling.*
