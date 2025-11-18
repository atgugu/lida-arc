# Morphological Primitives Benchmark Results

*Evaluation Date: 2025-11-18*
*Comparison: 34 Primitives (with Morphological) vs 29 Primitives (Conditional Baseline)*

## Executive Summary

After adding 5 morphological/iterative primitives (dilate, erode, flood_fill, fill_enclosed, spread_to_neighbors), we evaluated the system on 49 real ARC tasks.

### Key Results

| Metric | Conditional (29 prims) | + Morphological (34 prims) | Change |
|--------|------------------------|----------------------------|--------|
| **Solve Rate** | **22.4%** (11/49) | **22.4%** (11/49) | **0.0%** |
| **Tasks Solved** | 11 | 11 | **0** |
| **Average Accuracy** | 24.3% | 24.3% | 0.0% |
| **Time per Task** | 1.3s | 2.2s | **+69%** ⚠️ |

### Impact

❌ **No improvement in solve rate** (11/49 maintained)
❌ **+69% computation time** (1.3s → 2.2s per task)
✅ **No regression** (same tasks still solve)

## Analysis: Why No Improvement?

### Solved Tasks Breakdown (11/49)

All 11 solved tasks use **non-morphological** operations:

1. **00d62c1b** - Uses existing operations
2. **08ed6ac7** - Object-level transformations (recolor objects)
3. **1cf80156** - Auto-crop primitive (size/shape)
4. **1e0a9b12** - Rotation/reflection
5. **2f876c35** - Rotation/reflection
6. **3c9b0459** - Rotation/reflection
7. **4be741c5** - Rotation/reflection
8. **5bd6f4ac** - Rotation/reflection
9. **6e82a1ae** - Rotation/reflection
10. **7df24a62** - Color mapping
11. **8f2ea7aa** - Rotation + color

**Morphological operations used:** **0/11 tasks** (0%)

### Why Morphological Primitives Didn't Help

**1. Task Characteristics Mismatch**

The 49 test tasks don't heavily feature iterative/morphological patterns:
- ❌ Few tasks requiring region growing/shrinking
- ❌ Few tasks requiring flood fill operations
- ❌ Few tasks requiring hole filling
- ❌ Few tasks requiring iterative propagation

**2. Solved Tasks Use Simpler Patterns**

The 11 already-solved tasks use:
- ✓ Geometric transformations (rotation, reflection)
- ✓ Color mapping (simple recolor)
- ✓ Object-level operations (recolor objects)
- ✓ Size operations (auto-crop)

**3. Unsolved Tasks Need Different Capabilities**

The 38 unsolved tasks likely require:
- Complex multi-object reasoning
- Abstract pattern recognition
- Compositional transformations beyond current sequences
- Different types of spatial reasoning

**4. Increased Search Space Cost**

Adding 5 more primitives:
- Expanded search space from 29 to 34 operations (+17%)
- Beam search must evaluate more operation variants
- Parameter generation creates many morphological variants
- Result: **+69% computation time** with **no benefit**

## Comparison to Previous Improvements

### Improvement History

| Stage | Primitives | Solve Rate | Improvement |
|-------|-----------|------------|-------------|
| Baseline | 17 | 18.4% (9/49) | - |
| + Conditional | 29 (+12) | 22.4% (11/49) | **+4.0%** ✓ |
| + Morphological | 34 (+5) | 22.4% (11/49) | **0.0%** ❌ |

### Key Insight

**Not all primitive additions are equal:**
- Conditional primitives: +4% solve rate improvement
- Morphological primitives: 0% improvement, +69% cost

This suggests primitive additions must be **targeted** based on analysis of unsolved tasks.

## Performance Metrics

### Solve Rate by Strategy

| Prediction Strategy | 29 Primitives | 34 Primitives |
|---------------------|---------------|---------------|
| Single (k=1) | 22.4% | 22.4% |
| Multi (k=2) | 22.4% | 22.4% |

### Computational Cost

| Metric | 29 Primitives | 34 Primitives | Change |
|--------|---------------|---------------|--------|
| Avg time/task | 1.3s | 2.2s | +69% |
| Search space | 29 ops | 34 ops | +17% |
| Parameter variants | Moderate | High | +morphological combos |

## Deep Dive: Why These Particular Primitives Didn't Help

### Morphological Operations in ARC

Looking at the broader ARC dataset, morphological operations **are** important for tasks like:
- Growing objects until they connect
- Eroding noise from boundaries
- Filling enclosed regions (common pattern)
- Propagating colors through mazes

### But Not in Our Test Set

Our 49-task sample may be **unrepresentative**:
- Small sample size (49 out of 800 ARC tasks)
- May be biased toward simpler transformation types
- Missing tasks that specifically require morphological reasoning

### Evidence

The solve rate breakdown:
- 9/49 original baseline (geometric + color)
- +2/49 with object ops + conditionals
- +0/49 with morphological ops

This suggests our test set is dominated by:
1. Geometric transformations
2. Color mapping
3. Object-level reasoning
4. Simple size changes

And lacks:
1. Iterative growth/shrinkage patterns
2. Flood fill requirements
3. Hole filling patterns
4. Propagation through connected regions

## Recommendations

### 1. Analyze Unsolved Tasks

Before adding more primitives, deeply analyze the 38 unsolved tasks:
```python
# What patterns do they exhibit?
# What operations would solve them?
# Are they solvable with ANY primitive additions?
```

### 2. Consider Removing Morphological Primitives (For Now)

Options:
- **Option A:** Keep them (they don't hurt, just slow down)
- **Option B:** Remove them to improve performance
- **Option C:** Make them opt-in (only search when heuristics suggest)

### 3. Focus on High-ROI Primitives

Next primitive additions should target:
- Actual patterns in unsolved tasks
- Operations used by successful ARC solvers
- Validated on representative task sample

### 4. Improve Pattern Detection

Rather than more primitives, improve:
- Compositional sequence detection
- Parameter inference
- Pattern generalization
- Multi-example learning

## Technical Observations

### Parameter Generation Overhead

Morphological operations generate many parameter variants:
```python
# dilate: 3 colors × 3 iterations = 9 variants per grid state
# erode: 3 colors × 3 iterations = 9 variants
# flood_fill: 3 source × 3 fill = 9 variants
# fill_enclosed: 3 boundary × 2 fill = 6 variants
# spread: 3 source × 2 iterations = 6 variants
# Total: ~39 new variants per search node
```

This explains the +69% time increase.

### Beam Search Implications

With beam_width=10:
- Must evaluate 34 operations × parameter variants at each depth
- Morphological ops add ~39 variants
- Total search nodes increased significantly
- Pruning may eliminate morphological solutions prematurely

## Conclusion

**The morphological primitives addition was unsuccessful for this task set:**

- ❌ **0% improvement** in solve rate
- ❌ **+69% computation cost**
- ✅ **No regression** (maintained 11/49 solves)

**Root cause:**
- Task set doesn't feature morphological patterns
- 11 solved tasks use simpler operations
- 38 unsolved tasks need different capabilities

**Lessons learned:**
1. **Primitive additions must be targeted** based on unsolved task analysis
2. **Not all theoretically useful primitives help** in practice
3. **Search space expansion has real cost** (computation time)
4. **Sample size matters** - 49 tasks may not be representative

**Next steps:**
1. Analyze the 38 unsolved tasks to identify missing capabilities
2. Consider if morphological ops should be kept, removed, or made conditional
3. Focus on primitives that target actual gaps in unsolved tasks

---

*System: LIDA-ARC with 34 Primitives (29 previous + 5 morphological)*
*Dataset: 49 real ARC-AGI tasks (38 training + 10 sample + 1 evaluation)*
*Evaluation: Single and multi-prediction strategies*
*Benchmark date: 2025-11-18*
