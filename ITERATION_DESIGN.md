# Iteration/Looping Primitives: Design Analysis

## Problem: Iterative Patterns in ARC

Many ARC tasks require iterative operations:
- **Flood fill**: Propagate color through connected regions
- **Morphology**: Grow/shrink objects repeatedly
- **Propagation**: Color spreads outward step-by-step
- **Cellular rules**: Apply neighbor-based rules until stable

## Design Approaches

### Approach 1: Higher-Order Primitives (Most Composable)

```python
apply_until_stable(operation_name, max_iters=10)
repeat_n_times(operation_name, n)
for_each_object(operation_name)
```

**Pros:**
- ✅ Extremely composable - works with ANY operation
- ✅ Minimal primitives needed (2-3 vs 10+)
- ✅ Elegant abstraction - operations as first-class values
- ✅ Can iterate ANY future primitive we add

**Cons:**
- ❌ Harder to detect from demonstrations (no intermediate states shown)
- ❌ Implementation complexity (need to dispatch operations dynamically)
- ❌ Nested parameters: `{'operation': 'recolor_if_has_neighbor', 'params': {...}}`

**Example:**
```python
# Propagate color until stable
apply_until_stable('recolor_if_has_neighbor', params={'neighbor_color': 1, 'new_color': 2})

# Grow objects 3 times
repeat_n_times('dilate_objects', n=3)
```

### Approach 2: Specific Convergent Primitives (Most Practical)

```python
dilate(color, iterations=1)
erode(color, iterations=1)
flood_fill(source, target)
propagate(source, medium, steps=1)
```

**Pros:**
- ✅ Easy to implement - just regular primitives
- ✅ Easy to detect - compare input/output directly
- ✅ Clear semantics - each does one thing well
- ✅ Simple parameters - just integers and colors

**Cons:**
- ❌ Need separate primitive for each iterative pattern
- ❌ Less composable - can't iterate arbitrary operations
- ❌ More code to maintain

**Example:**
```python
# Grow red regions by 3 pixels
dilate(color=RED, iterations=3)

# Fill all connected blue regions with green
flood_fill(source=BLUE, target=GREEN)
```

### Approach 3: Hybrid (Pragmatic Balance)

```python
# Convergent primitives for common patterns
dilate(color, iterations=1)
erode(color, iterations=1)
flood_fill(source, target)

# Meta-primitive for general iteration
apply_until_stable(operation_name)  # For edge cases
```

**Pros:**
- ✅ Covers 90% of cases with specific primitives
- ✅ Meta-primitive handles remaining 10%
- ✅ Easy to detect and implement common cases
- ✅ Extensible for future needs

**Cons:**
- ❌ Larger primitive library
- ❌ Two conceptual models (specific + meta)

## Analysis: What's Most Elegant?

### Composability Metric

The functional programming perspective says: **higher-order functions are most composable**.

```
map(f, list) > [f1, f2, f3, ...]  # map is more general than specific functions
```

Applied to our DSL:
```
apply_until_stable(op) > [dilate, erode, flood_fill, ...]  # More general
```

### Integration Metric

Current system architecture:
- Primitives: objects with `execute(grid, **params)` methods
- Parameters: simple types (int, float, str, dict)
- Sequence detection: finds chains of operations
- Parameter generation: tries variants of simple parameters

**Challenge**: Operations-as-parameters requires:
1. Nested parameter structures
2. Dynamic operation dispatch
3. Detection of "which operation should iterate?"

This is a **significant architectural change**.

### Detection Metric

ARC demonstrations show: `input → output` (initial and final states only)

**For specific primitives:**
- dilate: output has bigger objects → DETECTABLE
- flood_fill: output has filled regions → DETECTABLE

**For meta-primitives:**
- apply_until_stable: need to infer WHICH operation was iterated → HARD
- Would need to: (1) guess operation, (2) simulate iteration, (3) check if matches output

## Recommendation: Evolutionary Approach

**Phase 1: Specific Convergent Primitives (Now)**

Start with 4-5 morphological/iterative primitives that cover most ARC patterns:

1. **`dilate_color(color, iterations=1, background=0)`** - Grow regions
2. **`erode_color(color, iterations=1, background=0)`** - Shrink regions
3. **`flood_fill_color(source, fill_color)`** - Fill connected components
4. **`fill_enclosed(boundary_color, fill_color, background=0)`** - Fill holes
5. **`spread_to_neighbors(source, target, iterations=1)`** - Propagation

**Why this first:**
- ✅ Easy to implement (standard image processing algorithms)
- ✅ Easy to detect (direct input/output comparison)
- ✅ Covers 80-90% of iterative ARC patterns
- ✅ Fits current architecture perfectly
- ✅ Parameters are simple (color, iterations)
- ✅ Sequence detection can try different iteration counts

**Phase 2: Meta-Primitives (Later, if needed)**

After gathering data on which patterns are still missing, consider adding:

```python
apply_until_stable(operation_name, max_iters=10)
```

This would require:
- Architectural enhancement for nested parameters
- New detection strategy (try all operations, simulate convergence)
- More complex parameter generation

## Proposed Implementation: Phase 1

### Core Iterative Primitives (5 primitives)

```python
class DilatePrimitive(Primitive):
    """Expand regions of specified color by N iterations (morphological dilation)."""
    def __init__(self):
        super().__init__('dilate', 'morphology')

    def execute(self, grid, color, iterations=1, background=0):
        # Expand pixels of 'color' into 'background' by N steps
        result = grid
        for _ in range(iterations):
            result = self._dilate_once(result, color, background)
        return result

class ErodePrimitive(Primitive):
    """Shrink regions of specified color by N iterations (morphological erosion)."""
    def __init__(self):
        super().__init__('erode', 'morphology')

    def execute(self, grid, color, iterations=1, background=0):
        # Shrink pixels of 'color' that border 'background'
        result = grid
        for _ in range(iterations):
            result = self._erode_once(result, color, background)
        return result

class FloodFillPrimitive(Primitive):
    """Fill all connected regions of source color with fill color."""
    def __init__(self):
        super().__init__('flood_fill', 'morphology')

    def execute(self, grid, source_color, fill_color):
        # BFS/DFS to fill all connected components
        result = [row[:] for row in grid]
        visited = set()
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == source_color and (r,c) not in visited:
                    self._flood_fill_region(result, r, c, source_color, fill_color, visited)
        return result

class FillEnclosedPrimitive(Primitive):
    """Fill regions enclosed by boundary color."""
    def __init__(self):
        super().__init__('fill_enclosed', 'morphology')

    def execute(self, grid, boundary_color, fill_color, background=0):
        # Fill holes: regions of background surrounded by boundary
        # Use flood fill from edges, then invert
        ...

class SpreadToNeighborsPrimitive(Primitive):
    """Spread source color to adjacent target pixels, N times."""
    def __init__(self):
        super().__init__('spread_to_neighbors', 'morphology')

    def execute(self, grid, source_color, target_color, iterations=1):
        # Propagate source_color into target_color regions
        result = grid
        for _ in range(iterations):
            result = self._spread_once(result, source_color, target_color)
        return result
```

### Detection Strategy

In `DemonstrationAnalyzer.analyze_demonstration()`:

```python
# Detect morphological operations
if self._objects_grew(input_grid, output_grid):
    patterns.append(self._detect_dilation(input_grid, output_grid))

if self._objects_shrank(input_grid, output_grid):
    patterns.append(self._detect_erosion(input_grid, output_grid))

if self._regions_filled(input_grid, output_grid):
    patterns.append(self._detect_flood_fill(input_grid, output_grid))

if self._holes_filled(input_grid, output_grid):
    patterns.append(self._detect_fill_enclosed(input_grid, output_grid))
```

### Parameter Generation

In `SequenceDetector._generate_operation_variants()`:

```python
elif op_name == 'dilate':
    # Try different iteration counts
    for iters in [1, 2, 3]:
        for color in colors_in_grid:
            if color != 0:  # Don't dilate background
                variants.append((op_name, {
                    'color': color,
                    'iterations': iters,
                    'background': 0
                }))
```

## Advantages of This Approach

1. **Elegant**: Clean, well-defined mathematical operations
2. **Composable**: Can combine in sequences: `['erode', 'dilate']` (morphological opening)
3. **Efficient**: Simple to implement using standard algorithms
4. **Detectable**: Direct pattern matching on input/output
5. **Parametric**: Iteration count can be discovered via parameter search
6. **Extensible**: Can add meta-primitives later if needed

## Comparison to State-of-the-Art

Many successful ARC solvers use similar approaches:
- DreamCoder: Domain-specific primitives + program synthesis
- LARC: Hand-crafted primitives for common patterns
- Kaggle winners: Morphological operations featured prominently

The key insight: **Start specific, generalize later** is more practical than **start general, hope it works**.

## Conclusion

**Recommended approach: Phase 1 - Specific Convergent Primitives**

Implement 5 morphological primitives that cover ~90% of iterative ARC patterns:
- dilate, erode, flood_fill, fill_enclosed, spread_to_neighbors

**Future work: Phase 2 - Meta-primitives if needed**

After evaluating Phase 1 results, consider adding `apply_until_stable` for remaining edge cases.

This is the most elegant approach because:
- ✅ Maximizes composability within current architecture
- ✅ Minimizes implementation complexity
- ✅ Maximizes detectability from demonstrations
- ✅ Provides clear path to generalization

---

**Next step**: Implement the 5 morphological primitives and integrate into pattern detection.
