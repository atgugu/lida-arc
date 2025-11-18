# Analysis of 38 Unsolved ARC Tasks

*Analysis Date: 2025-11-18*
*Current System: 34 Primitives (22.4% solve rate, 11/49 tasks)*

## Executive Summary

Analyzed 38 unsolved tasks to identify missing primitive capabilities. Found that the unsolved tasks require fundamentally different operations than the solved tasks, clustered into 4 main categories.

### Top Missing Capabilities

| Category | Tasks | Current Support | Gap |
|----------|-------|----------------|-----|
| **Object Extraction** | 8 tasks | Partial (auto_crop) | Need selective object extraction |
| **Symmetry Operations** | 4 tasks | None | Need symmetry completion/reflection |
| **Pattern Extraction + Tiling** | 3 tasks | None | Can't extract sub-patterns |
| **Complex Spatial** | 21+ tasks | Minimal | Various specialized operations |

## Detailed Analysis

### Category 1: Object Extraction (8 tasks)

**Tasks:** 0520fde7, 0b148d64, 1190e5a7, 137eaa0f, 1b2d62fb, 1f85a75f, 1fad071e, 2013d3e2

**Pattern:**
- Input: Large grid with multiple objects/regions
- Output: Smaller grid containing extracted/selected object
- Transformation: Size decrease (extract specific region)

**Example - Task 0520fde7:**
```
Input: 3x7 grid with colors [0, 1, 5]
Output: 3x3 grid with colors [0, 2]
```

**Current Support:**
- ✓ `auto_crop` - crops to bounding box of non-background
- ❌ Can't select WHICH object to extract
- ❌ Can't extract based on properties (size, color, position)

**Missing Primitives:**
```python
extract_object(grid, object_index)  # Extract Nth object
extract_by_color(grid, color)  # Extract all regions of color
extract_largest(grid)  # Extract largest object
extract_by_property(grid, property, value)  # e.g., size > 5
```

**Priority:** **HIGH** - 8 tasks (21% of unsolved)

---

### Category 2: Symmetry Operations (4 tasks)

**Tasks:** 0a938d79, 0d3d703e, 1bfc4729, 22168020

**Pattern:**
- Input: Asymmetric or partially symmetric grid
- Output: Completed symmetric grid
- Transformation: Same size, adds symmetry

**Example - Task 0d3d703e:**
- Output shows symmetry that input doesn't have
- Suggests: reflect partial pattern to create full symmetric pattern

**Current Support:**
- ✓ `reflect_horizontal`, `reflect_vertical`, `reflect_diagonal` - reflect entire grid
- ❌ Can't detect partial symmetry
- ❌ Can't complete symmetric patterns
- ❌ Can't reflect only parts of grid

**Missing Primitives:**
```python
complete_symmetry_horizontal(grid)  # Fill to create h-symmetry
complete_symmetry_vertical(grid)  # Fill to create v-symmetry
complete_symmetry_rotational(grid)  # Fill to create rotational symmetry
detect_symmetry_axis(grid)  # Find axis of partial symmetry
```

**Priority:** **MEDIUM** - 4 tasks (11% of unsolved)

---

### Category 3: Pattern Extraction + Tiling (3 tasks)

**Tasks:** 007bbfb7, 05269061, 017c7c7b

**Pattern Type A - Extract sub-pattern then tile (05269061):**
```
Input:  2830000    →  Extract 3x3:  283
        8300000                      832
        3000000                      328
        0000000

Output: 2832832    (Tile the 3x3 pattern across full grid)
        8328328
        3283283
        2832832
        8328328
        3283283
        2832832
```

**Current Support:**
- ✓ `tile` - tiles entire input grid
- ❌ Can't extract sub-pattern from input
- ❌ Can't tile a sub-region

**Pattern Type B - Complex spatial tiling (007bbfb7):**
```
Input: 3x3 grid
Output: 9x9 grid where input appears at specific positions (not uniform tiling)

Example:
Input:  077    Output: 000|077|077  (Input placed at specific offsets)
        777            000|777|777
        077            000|077|077
                       ---|---|---
                       077|077|077
                       777|777|777
                       077|077|077
                       ...
```

**Missing Primitives:**
```python
extract_pattern(grid, top_left, height, width)  # Extract sub-region
tile_pattern(pattern, repeat_h, repeat_w)  # Tile extracted pattern
place_at_grid(pattern, positions)  # Place pattern at multiple positions
sparse_tile(grid, offsets)  # Tile with specific offsets/spacing
```

**Priority:** **HIGH** - Fundamental composition capability

---

### Category 4: Complex Spatial Transformations (21+ tasks)

**Tasks:** 025d127b, 05f2a901, 06df4c85, 09629e4f, 0962bcdd, 0a938d79, 0b148d64, 0ca9ddb6, 0d3d703e, 0dfd9992, 0e206a2e, 150deff5, 1b60fb0c, 1bfc4729, 1f85a75f, 22168020, 22233c11, 23b5c85d, 2dee498d, 54d9e175, a699fb00

**Patterns (varied):**
- Object rearrangement
- Selective transformations (transform only certain objects)
- Conditional operations based on context
- Multi-step compositional transformations
- Abstract pattern recognition

**Example - Task 025d127b:**
- 14x9 grid with multiple colored objects
- Same size output
- Objects appear rotated/reflected/rearranged
- Requires: detect objects, transform individually, reassemble

**Current Support:**
- Partial - have object detection, individual transforms
- ❌ Can't compose object-level operations in sequences
- ❌ Can't apply different transforms to different objects
- ❌ Can't detect and use spatial relationships

**This category is TOO DIVERSE for simple primitive additions.**

---

## Key Insights

### 1. Detection vs Primitives Problem

Some failures may be **detection issues**, not missing primitives:

**Evidence:**
- Task 05269061 could theoretically solve with: `extract_pattern → tile`
- We have `tile` but not `extract_pattern`
- Even if we add `extract_pattern`, will sequence detection find it?

**Implication:** Need better compositional sequence detection, not just more primitives.

### 2. The "Long Tail" Problem

Task complexity distribution:
- **Easy (11 tasks):** Solved with current primitives
- **Medium (12 tasks):** Need 1-2 new primitive types
- **Hard (26 tasks):** Need complex compositions, context-awareness, abstract reasoning

Adding primitives helps the medium tasks, but the hard tasks need architectural improvements.

### 3. Compositional Sequences

Many unsolved tasks need **multi-step transformations** we can't detect:
- Extract pattern → tile it
- Detect objects → transform individually → reassemble
- Find symmetry axis → reflect around it

Current sequence detection:
- Max depth: 3 operations
- Beam width: 10
- Parameters: limited variants

May need:
- Deeper search (depth 5+)
- Wider beam (width 20+)
- Better parameter inference
- Hierarchical composition (operations on operations)

### 4. Architectural Ceiling

Some tasks may be fundamentally beyond current architecture:
- Require understanding "concepts" (inside/outside, container/contained)
- Need abstract reasoning (analogy, pattern completion)
- Require variable binding (track multiple objects, reuse patterns)

## Recommendations

### Tier 1: High ROI Primitive Additions

**Add Pattern Extraction Primitives (3-5 tasks estimated):**
```python
extract_pattern(grid, row, col, height, width)  # Extract sub-region
tile_extracted(grid, extract_region, tile_count)  # Extract + tile in one operation
```

**Add Object Selection Primitives (8 tasks estimated):**
```python
extract_largest_object(grid)  # Get largest connected component
extract_by_color(grid, color)  # Get all objects of color
extract_nth_object(grid, n)  # Get specific object by index
```

**Estimated Impact:** +11 tasks (9% → 31% solve rate)

### Tier 2: Medium ROI Additions

**Symmetry Completion Primitives (4 tasks estimated):**
```python
complete_symmetry_h(grid)  # Complete horizontal symmetry
complete_symmetry_v(grid)  # Complete vertical symmetry
```

**Estimated Impact:** +4 tasks (31% → 39% solve rate)

### Tier 3: Architectural Improvements

**Rather than more primitives, improve:**

1. **Sequence Detection:**
   - Increase max_depth from 3 to 5
   - Increase beam_width from 10 to 20
   - Better parameter inference

2. **Compositional Reasoning:**
   - Hierarchical operations (apply X to result of Y)
   - Conditional sequences (if pattern A then operation B)
   - Variables/bindings (store intermediate results)

3. **Pattern Learning:**
   - Learn from multiple tasks
   - Transfer patterns across tasks
   - Meta-learning for operation selection

## Comparison: Actual Additions vs Unsolved Tasks

### What We Added vs What We Need

| Addition | Primitives Added | Tasks Helped | Tasks Still Unsolved |
|----------|-----------------|--------------|---------------------|
| Conditional | +4 | +2 (9→11) | Need object selection |
| Morphological | +5 | 0 (11→11) | Wrong patterns for test set |
| **Recommended: Pattern Extraction** | +2 | ~+3 estimated | Object selection still needed |
| **Recommended: Object Selection** | +3 | ~+8 estimated | Complex tasks still hard |
| **Recommended: Symmetry** | +2 | ~+4 estimated | Architectural limits remain |

### Cumulative Projection

| Stage | Solve Rate | Tasks Solved |
|-------|-----------|--------------|
| Current | 22.4% | 11/49 |
| + Pattern Extraction | ~28% | ~14/49 |
| + Object Selection | ~38% | ~19/49 |
| + Symmetry | ~47% | ~23/49 |
| **Theoretical Max** | ~47% | ~23/49 |

**Hard ceiling:** ~26 tasks may be architecturally unsolvable without major improvements.

## Conclusion

**Evidence-based primitive prioritization:**

1. **Pattern Extraction** (High priority, 3 tasks, foundational)
2. **Object Selection** (High priority, 8 tasks, clear need)
3. **Symmetry Operations** (Medium priority, 4 tasks, specialized)

**Beyond primitives:**
- Improve sequence detection (depth, beam width, parameters)
- Add compositional reasoning capabilities
- Consider architectural enhancements for abstract reasoning

**Next Steps:**
1. Implement pattern extraction primitives
2. Test on tasks 007bbfb7, 05269061, 017c7c7b
3. If successful (+3 tasks), implement object selection
4. If successful (+8 tasks), implement symmetry operations
5. At ~47% solve rate, hit architectural ceiling

**Meta-lesson:** Empirical analysis > theoretical elegance. The morphological primitives were elegant but didn't match actual task needs.

---

*Analysis based on 38 unsolved tasks from 49-task test set*
*Current system: 34 primitives, 22.4% solve rate*
*Recommendations target specific, analyzed gaps*
