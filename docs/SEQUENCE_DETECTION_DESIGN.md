# Sequence Detection Design: Robust Composite Operation Learning

## Problem Analysis

**Current limitation:** The system finds single operations (rotate_90, recolor) but not sequences (rotate_90 THEN recolor).

**Impact:** 1/10 tasks fails because it requires detecting a 2-operation sequence.

**Challenge:** Naive search over sequences is combinatorially expensive:
- 17 primitives
- Depth 2: 17² = 289 pairs
- Depth 3: 17³ = 4,913 triples
- Depth 4: 17⁴ = 83,521 quadruples

**Goal:** Detect sequences efficiently without exhaustive search.

---

## Design Principles

1. **Iterative Deepening:** Try simple explanations before complex ones
2. **Heuristic Pruning:** Use domain knowledge to avoid impossible sequences
3. **PAM-Guided Search:** Use spreading activation to rank likely operations
4. **Distance-Based Early Termination:** Stop when output matches
5. **Caching:** Store intermediate results to avoid recomputation
6. **Incremental Learning:** Store successful sequences as composites

---

## Proposed Architecture

### 1. **Iterative Deepening with Beam Search**

```python
def find_transformation_sequence(input_grid, output_grid, max_depth=3, beam_width=5):
    """
    Find sequence of operations that transforms input to output.

    Args:
        input_grid: Starting grid
        output_grid: Target grid
        max_depth: Maximum sequence length to try
        beam_width: Number of candidates to keep at each depth

    Returns:
        Best sequence found (list of operation names)
    """

    # Depth 1: Try single operations (current approach)
    candidates = try_single_operations(input_grid, output_grid)
    if any(c.distance == 0 for c in candidates):
        return best_match(candidates).sequence

    # Depth 2+: Iteratively deepen
    for depth in range(2, max_depth + 1):
        # Keep top beam_width candidates from previous depth
        candidates = prune_beam(candidates, beam_width)

        # Extend each candidate with one more operation
        new_candidates = []
        for candidate in candidates:
            extensions = extend_candidate(candidate, output_grid)
            new_candidates.extend(extensions)

        candidates = new_candidates

        # Check if any candidate matches perfectly
        if any(c.distance == 0 for c in candidates):
            return best_match(candidates).sequence

    # Return best effort
    return best_match(candidates).sequence if candidates else []
```

### 2. **Intelligent Pruning Rules**

```python
class SequencePruner:
    """Prunes invalid or unlikely operation sequences."""

    def is_valid_sequence(self, sequence: List[str]) -> bool:
        """Check if sequence is valid and sensible."""

        # Rule 1: No immediate inversions
        for i in range(len(sequence) - 1):
            if self._are_inverses(sequence[i], sequence[i+1]):
                return False

        # Rule 2: No repeated rotations (use single rotation instead)
        rotation_count = sum(1 for op in sequence if 'rotate' in op)
        if rotation_count > 1:
            return False  # rotate_90 + rotate_90 = rotate_180

        # Rule 3: At most one reflection (multiple reflections cancel)
        reflection_count = sum(1 for op in sequence if 'reflect' in op)
        if reflection_count > 1:
            return False

        # Rule 4: Color operations should come last (usually)
        # Rotating after recoloring is uncommon
        color_ops = ['recolor', 'fill_background']
        last_color_idx = -1
        for i, op in enumerate(sequence):
            if op in color_ops:
                last_color_idx = i

        if last_color_idx != -1 and last_color_idx < len(sequence) - 1:
            # Check if anything after color op
            if any(op not in color_ops for op in sequence[last_color_idx+1:]):
                return False  # Geometric op after color op is unusual

        return True

    def _are_inverses(self, op1: str, op2: str) -> bool:
        """Check if two operations are inverses."""
        inverses = {
            ('rotate_90', 'rotate_270'),
            ('rotate_270', 'rotate_90'),
            ('reflect_horizontal', 'reflect_horizontal'),  # Self-inverse
            ('reflect_vertical', 'reflect_vertical'),      # Self-inverse
            ('reflect_diagonal', 'reflect_diagonal'),       # Self-inverse
        }
        return (op1, op2) in inverses or (op2, op1) in inverses
```

### 3. **Distance-Based Guidance**

```python
class GridDistance:
    """Compute distance between grids to guide search."""

    @staticmethod
    def compute(grid1, grid2) -> float:
        """
        Compute normalized distance between two grids.

        Returns value in [0, 1] where 0 = identical, 1 = maximally different
        """
        # Handle size mismatch
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return 1.0  # Different sizes = maximum distance

        # Count differing pixels
        total_pixels = len(grid1) * len(grid1[0])
        different_pixels = 0

        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] != grid2[i][j]:
                    different_pixels += 1

        return different_pixels / total_pixels

    @staticmethod
    def is_closer(grid1, grid2, target) -> bool:
        """Check if grid1 is closer to target than grid2."""
        return GridDistance.compute(grid1, target) < GridDistance.compute(grid2, target)
```

### 4. **PAM-Guided Operation Ranking**

```python
class PAMGuidedSearch:
    """Use PAM spreading activation to rank likely operations."""

    def __init__(self, pam_integration: ARCPAMIntegration):
        self.pam = pam_integration

    def rank_operations(self, current_state, target, primitives) -> List[str]:
        """
        Rank operations by likelihood of being useful.

        Uses:
        1. PAM activation (from spreading)
        2. Grid features (what's needed: rotation? color change?)
        3. Historical success (what worked on similar tasks)
        """
        # Get activated operations from PAM
        activations = self.pam.pam.get_all_activations()
        pam_scores = {
            op: activations.get(f'prim_{op}', 0.0)
            for op in primitives.get_all_names()
        }

        # Analyze what transformation is needed
        feature_scores = self._analyze_needed_transformation(current_state, target)

        # Combine scores
        combined_scores = {}
        for op in primitives.get_all_names():
            pam_score = pam_scores.get(op, 0.0)
            feature_score = feature_scores.get(op, 0.0)

            # Weighted combination
            combined_scores[op] = 0.6 * pam_score + 0.4 * feature_score

        # Return operations sorted by score (descending)
        return sorted(combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True)

    def _analyze_needed_transformation(self, current, target) -> Dict[str, float]:
        """Analyze grids to determine what transformation is likely needed."""
        scores = {}

        # Check if shapes match (if not, might need rotation/reflection)
        if self._shapes_transposed(current, target):
            scores['rotate_90'] = 1.0
            scores['rotate_270'] = 1.0
            scores['reflect_diagonal'] = 0.8

        # Check if colors differ (might need recolor)
        if self._colors_differ(current, target):
            scores['recolor'] = 1.0
            scores['fill_background'] = 0.5

        # Check if flipped horizontally
        if self._is_horizontal_flip(current, target):
            scores['reflect_horizontal'] = 1.0

        # Check if flipped vertically
        if self._is_vertical_flip(current, target):
            scores['reflect_vertical'] = 1.0

        return scores
```

### 5. **Sequence Candidate Structure**

```python
@dataclass
class SequenceCandidate:
    """A candidate sequence being explored."""

    sequence: List[str]              # Operation names
    intermediate_grids: List[Grid]   # Grid after each operation
    distance_to_target: float        # How far from goal
    confidence: float                # Overall confidence

    def extend(self, operation: str, result_grid: Grid, target: Grid) -> 'SequenceCandidate':
        """Extend this candidate with one more operation."""
        new_sequence = self.sequence + [operation]
        new_intermediates = self.intermediate_grids + [result_grid]
        new_distance = GridDistance.compute(result_grid, target)

        # Confidence decreases with sequence length (prefer shorter)
        length_penalty = 0.9 ** len(new_sequence)
        # Confidence increases with getting closer
        progress_bonus = 1.0 - new_distance

        new_confidence = length_penalty * progress_bonus

        return SequenceCandidate(
            sequence=new_sequence,
            intermediate_grids=new_intermediates,
            distance_to_target=new_distance,
            confidence=new_confidence
        )
```

### 6. **Caching for Efficiency**

```python
class SequenceCache:
    """Cache intermediate results to avoid recomputation."""

    def __init__(self):
        self.cache: Dict[Tuple[GridHash, Tuple[str, ...]], Grid] = {}

    def get(self, input_grid: Grid, sequence: List[str]) -> Optional[Grid]:
        """Get cached result if available."""
        key = (self._hash_grid(input_grid), tuple(sequence))
        return self.cache.get(key)

    def put(self, input_grid: Grid, sequence: List[str], result: Grid):
        """Store result in cache."""
        key = (self._hash_grid(input_grid), tuple(sequence))
        self.cache[key] = result

    def _hash_grid(self, grid: Grid) -> GridHash:
        """Create hashable representation of grid."""
        return tuple(tuple(row) for row in grid)
```

---

## Implementation Strategy

### Phase 1: Iterative Deepening (Immediate)
1. Modify `DemonstrationAnalyzer` to try sequences
2. Add depth parameter (default 2 for efficiency)
3. Use beam search with width=5
4. Implement basic pruning rules

### Phase 2: Intelligent Guidance (Short-term)
1. Add `GridDistance` metric
2. Implement PAM-guided operation ranking
3. Add feature-based heuristics
4. Cache intermediate results

### Phase 3: Learning and Optimization (Medium-term)
1. Store successful sequences as composites
2. Learn which sequences commonly occur
3. Add to PAM as higher-level operations
4. Use meta-learning across tasks

### Phase 4: Advanced Search (Long-term)
1. Monte Carlo Tree Search (MCTS)
2. Neural-guided search (use LLM for heuristics)
3. Parallel beam search
4. Adaptive beam width based on problem complexity

---

## Expected Performance Impact

### Computational Cost

**Current (single operations only):**
- Try 17 operations
- Time: ~1-2ms per demo

**With sequences (depth 2, beam width 5):**
- Depth 1: Try 17 operations
- Depth 2: Extend top 5 candidates × 17 operations = 85 tries
- With pruning: ~30-40 tries (prune ~50%)
- Time: ~5-10ms per demo
- **Total overhead: ~5-8ms (negligible compared to 305ms total)**

### Solve Rate Improvement

**Expected results:**
- Current: 9/10 (90%)
- With depth-2 sequences: 10/10 (100%)
- With depth-3 sequences: Handle more complex tasks

**Confidence: Very High**
- The failing task (9ecd008a) requires exactly depth 2
- Most ARC tasks are depth 1-2
- Pruning keeps search tractable

---

## Comparison to Alternatives

### Alternative 1: Exhaustive Search
- **Pros:** Guaranteed to find optimal sequence
- **Cons:** Exponential time complexity (289 pairs, 4913 triples)
- **Verdict:** Too slow

### Alternative 2: Neural-Guided Search
- **Pros:** Can learn complex heuristics
- **Cons:** Requires training data, adds neural dependency
- **Verdict:** Overkill for current problem

### Alternative 3: Manual Composite Primitives
- **Pros:** Simple, deterministic
- **Cons:** Hard-codes specific sequences, doesn't generalize
- **Verdict:** Not scalable

### Alternative 4: Genetic Programming
- **Pros:** Can discover novel sequences
- **Cons:** Slow, non-deterministic, complex
- **Verdict:** Too heavyweight

### **Our Approach: Iterative Deepening + Heuristic Pruning**
- **Pros:** Fast, deterministic, generalizable, scalable
- **Cons:** Might miss very deep sequences (depth 4+)
- **Verdict:** Optimal for ARC tasks (most are depth 1-2)

---

## Code Structure

```
src/lida/arc/
├── sequence_detection.py       # New: Sequence detection logic
│   ├── SequenceDetector
│   ├── SequenceCandidate
│   ├── SequencePruner
│   ├── GridDistance
│   └── PAMGuidedSearch
│
├── demonstration.py            # Modified: Use sequence detector
│   └── DemonstrationAnalyzer.analyze_pair()
│       ├── Try single operations (current)
│       └── Try sequences if no match (new)
│
└── sequence_cache.py           # New: Caching layer
    └── SequenceCache
```

---

## Testing Strategy

### Unit Tests
1. **GridDistance:** Test distance metric correctness
2. **SequencePruner:** Test pruning rules
3. **SequenceDetector:** Test depth-1, depth-2, depth-3 searches
4. **Caching:** Test cache hits/misses

### Integration Tests
1. **Composite task (9ecd008a):** Should now solve with depth-2 search
2. **All other tasks:** Should still solve (no regression)
3. **Performance:** Should stay under 400ms per task

### Benchmark
1. Re-run full benchmark
2. Expected: 10/10 (100%) solve rate
3. Expected time: ~310-320ms per task (small overhead)

---

## Rollout Plan

### Week 1: Core Implementation
- [ ] Implement `GridDistance`
- [ ] Implement `SequenceCandidate`
- [ ] Implement `SequencePruner`
- [ ] Implement `SequenceDetector` (depth-2 only)

### Week 2: Integration
- [ ] Integrate with `DemonstrationAnalyzer`
- [ ] Add unit tests
- [ ] Test on composite task

### Week 3: Optimization
- [ ] Add `PAMGuidedSearch`
- [ ] Add `SequenceCache`
- [ ] Tune beam width and depth parameters

### Week 4: Validation
- [ ] Full benchmark evaluation
- [ ] Performance profiling
- [ ] Documentation and examples

---

## Success Metrics

### Primary
- **Solve rate:** 100% (10/10 tasks)
- **Composite task:** Solves 9ecd008a correctly
- **No regression:** All 9 previously solved tasks still solve

### Secondary
- **Performance:** < 400ms per task (< 30% overhead)
- **Generalization:** Works on new composite tasks
- **Maintainability:** Clean, testable, documented code

---

## Future Enhancements

### 1. Adaptive Depth
- Start with depth 1, increase only if needed
- Most tasks solve at depth 1, don't waste time

### 2. Meta-Learning
- Track successful sequences across tasks
- Build library of common composites
- Prefer proven sequences

### 3. Parallel Search
- Search multiple sequences in parallel
- Useful for depth 3+ where candidates explode

### 4. Neural Heuristics
- Train small neural network to predict likely operations
- Use as heuristic to guide search
- Optional enhancement (don't depend on it)

---

## Conclusion

**Proposed solution:** Iterative deepening with beam search, heuristic pruning, and PAM guidance.

**Why this works:**
1. ✅ Handles sequences efficiently (beam search)
2. ✅ Avoids combinatorial explosion (pruning)
3. ✅ Guided by domain knowledge (PAM + heuristics)
4. ✅ Fast enough for real-time use (~5-10ms overhead)
5. ✅ Generalizes to new tasks (not hard-coded)
6. ✅ Learns over time (store composites in PAM)

**Expected outcome:**
- 100% solve rate (10/10 tasks)
- < 400ms per task
- Robust to new composite tasks

**Implementation complexity:** Medium (2-3 weeks)

**Risk:** Low (can fall back to single operations if sequence search fails)

This design provides a robust, efficient solution for sequence detection that will enable the system to handle composite transformations while maintaining excellent performance.
