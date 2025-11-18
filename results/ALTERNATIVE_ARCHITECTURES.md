# Alternative Architectures for ARC-AGI: Beyond Primitives

*Analysis Date: 2025-11-18*
*Context: Current primitive-based approach has hit 22% ceiling after 4 failed improvements*

## Executive Summary

After hitting an architectural ceiling with the primitive-based approach, we need fundamentally different strategies. This document explores three alternative architectures that address the root limitations of distance-based beam search over hand-coded primitives.

**Current System Limitations:**
- Can't find multi-step compositions (extract→tile)
- Distance-based pruning prevents non-monotonic search paths
- Hand-coded primitives can't capture all transformation types
- Ceiling: ~22-25% solve rate

**Alternative Approaches:**
1. **Neural Program Synthesis** - Learn to generate programs
2. **Learned Pattern Recognition** - Learn what makes intermediates "promising"
3. **Hybrid** - Combine primitives with learned guidance

---

## Approach 1: Neural Program Synthesis

### Core Idea

Instead of searching through combinations of hand-coded primitives, **learn to directly generate transformation programs** from input/output examples.

**Paradigm shift:**
```
Current:  Search over primitives → Find sequence → Execute
New:      Encode examples → Generate program → Execute
```

### Architecture

#### 1. Encoder: Task Representation

**Input:** Multiple input/output demonstration pairs
**Output:** Task embedding vector

```python
class TaskEncoder(nn.Module):
    """Encode multiple I/O examples into task representation."""

    def forward(self, demonstrations: List[Tuple[Grid, Grid]]):
        # Process each demonstration
        demo_embeddings = []
        for input_grid, output_grid in demonstrations:
            # Grid encoder (could be CNN, Transformer, or custom)
            input_emb = self.grid_encoder(input_grid)    # Shape: [H, W, D]
            output_emb = self.grid_encoder(output_grid)  # Shape: [H, W, D]

            # Combine input/output
            demo_emb = self.combine(input_emb, output_emb)  # Shape: [D]
            demo_embeddings.append(demo_emb)

        # Aggregate across demonstrations
        task_embedding = self.aggregate(demo_embeddings)  # Shape: [D]
        return task_embedding
```

**Key design choices:**
- **Grid encoder:** CNN for spatial structure? Transformer for long-range dependencies?
- **Aggregation:** Mean-pooling? Attention over demonstrations?
- **Embedding size:** 128-512 dimensions typical

#### 2. Program Generator: Sequence Model

**Input:** Task embedding + current grid state
**Output:** Next operation in program

```python
class ProgramGenerator(nn.Module):
    """Generate transformation program autoregressively."""

    def __init__(self, vocab_size, hidden_dim):
        # Vocabulary: all primitive operations + parameters
        self.vocab_size = vocab_size  # e.g., 36 primitives × parameter values
        self.decoder = TransformerDecoder(hidden_dim)

    def forward(self, task_emb, program_so_far, current_grid):
        # Encode current state
        grid_emb = self.grid_encoder(current_grid)

        # Decode next operation
        # Input: [task_emb, program_so_far, grid_emb]
        # Output: distribution over next operations
        logits = self.decoder(task_emb, program_so_far, grid_emb)

        # Sample or greedy decode
        next_op = torch.argmax(logits, dim=-1)
        return next_op
```

**Generation strategies:**
- **Greedy:** Always pick highest probability operation
- **Beam search:** Keep top-k program candidates
- **Sampling:** Stochastic generation for diversity

#### 3. Executor: Program Execution

**Input:** Generated program
**Output:** Transformed grid

```python
def execute_program(input_grid, program):
    current = input_grid
    for operation in program:
        op_name, params = parse_operation(operation)
        primitive = get_primitive(op_name)
        current = primitive.execute(current, **params)
    return current
```

### Training

#### Dataset Requirements

**Need:** Large corpus of ARC tasks with solutions

**Options:**
1. **Human-annotated solutions:** 400 public ARC training tasks (limited)
2. **Synthetic tasks:** Generate tasks with known solutions
3. **Self-play:** Generate random transformations, reverse-engineer solutions

#### Training Objective

**Goal:** Maximize likelihood of correct program given demonstrations

```python
def training_step(task):
    # Encode task from demonstrations
    task_emb = encoder(task.train_examples)

    # Generate program
    program = []
    current_grid = task.test_input

    for step in range(max_steps):
        # Predict next operation
        op_logits = generator(task_emb, program, current_grid)

        # Supervised learning: compare to ground-truth program
        loss = cross_entropy(op_logits, ground_truth_program[step])

        # Update model
        loss.backward()
        optimizer.step()

        # Execute operation for next step
        next_op = ground_truth_program[step]
        current_grid = execute(current_grid, next_op)
        program.append(next_op)

    return loss
```

**Challenge:** Need ground-truth programs for training. Options:
- Hand-annotate solutions for 400 tasks (expensive)
- Use current system's discovered programs as weak supervision
- Program search + policy distillation

### Advantages

1. **No hand-coded primitives required**
   - Model learns what operations are useful
   - Can discover novel operation combinations
   - Not limited by human intuition

2. **End-to-end differentiable**
   - Can optimize directly for task-solving
   - Backpropagation through entire pipeline
   - No discrete search over primitives

3. **Can learn composition patterns**
   - Model sees which operation sequences work
   - Learns "extract then tile" if that pattern appears in training
   - Generalizes to new tasks with similar patterns

4. **Scales with data**
   - More training tasks → better performance
   - Can use synthetic data generation
   - Transfer learning from related domains

### Disadvantages

1. **Requires large training dataset**
   - 400 public ARC tasks may not be enough
   - Need program annotations or synthesis
   - Data augmentation critical

2. **Interpretability loss**
   - Neural network is black box
   - Can't explain why it chose operations
   - Hard to debug failures

3. **Execution correctness**
   - Generated programs may be syntactically invalid
   - Need constrained decoding or post-processing
   - May generate infinite loops

4. **Computational cost**
   - Training neural models is expensive (GPU hours)
   - Inference faster than search, but still costly
   - Need architecture tuning

### Prior Work

**Neural Program Synthesis:**
- **DeepCoder** (Microsoft, 2017): Learns to predict which operations are useful
- **RobustFill** (Google, 2017): Synthesizes string manipulation programs
- **DreamCoder** (MIT, 2020): Learns library of reusable abstractions

**ARC-specific:**
- **Kaggle ARC winners** (2020): Used neural networks for pattern recognition + heuristic search
- **LARC** (2022): Large language models for ARC program generation
- **DSL-based approaches:** Learn domain-specific languages from data

### Estimated Impact

**Optimistic:** 35-50% solve rate
- If trained on sufficient data with good architectures
- Likely needs 10K+ synthetic tasks
- GPU resources: ~100-500 hours training

**Realistic:** 25-35% solve rate
- With 400 public tasks + data augmentation
- Modest architecture (Transformer)
- GPU resources: ~50-100 hours training

**Risk:** May not exceed 22% baseline if:
- Insufficient training data
- Model can't learn compositional reasoning
- Execution correctness issues

---

## Approach 2: Learned Pattern Recognition

### Core Idea

Keep the primitive-based search, but **learn which intermediate states are "promising"** instead of using distance-based pruning.

**Paradigm shift:**
```
Current:  Prune by distance to target
New:      Prune by learned "promise score"
```

### Architecture

#### 1. State Value Network

**Input:** Current grid + target grid + partial program
**Output:** Probability this state leads to solution

```python
class StateValueNetwork(nn.Module):
    """Predict if a partial solution will lead to success."""

    def forward(self, current_grid, target_grid, partial_program):
        # Encode grids
        current_emb = self.grid_encoder(current_grid)
        target_emb = self.grid_encoder(target_grid)

        # Encode partial program
        program_emb = self.program_encoder(partial_program)

        # Combine all features
        combined = torch.cat([current_emb, target_emb, program_emb], dim=-1)

        # Predict value
        value = self.mlp(combined)  # Scalar in [0, 1]
        return value
```

**Training signal:**
- Positive examples: Intermediate states from successful solutions
- Negative examples: Intermediate states from failed search paths

#### 2. Operation Selector Network

**Input:** Current grid + target grid + available operations
**Output:** Distribution over which operation to try next

```python
class OperationSelector(nn.Module):
    """Predict which operation is most likely to help."""

    def forward(self, current_grid, target_grid, operations):
        # Encode state
        state_emb = self.encode_state(current_grid, target_grid)

        # Score each operation
        op_scores = []
        for op in operations:
            op_emb = self.operation_encoder(op)
            score = torch.dot(state_emb, op_emb)  # Compatibility score
            op_scores.append(score)

        # Return distribution
        return torch.softmax(torch.stack(op_scores), dim=0)
```

**Usage:** Prioritize operations with high predicted scores in beam search

### Modified Search Algorithm

**Combine learned guidance with heuristic search:**

```python
def learned_beam_search(input_grid, target_grid, primitives):
    # Initialize beam
    candidates = [Candidate([], input_grid, confidence=1.0)]

    for depth in range(max_depth):
        new_candidates = []

        for candidate in candidates:
            # Use learned value network to filter
            if depth > 0:
                value = value_network(candidate.current_grid, target_grid, candidate.sequence)
                if value < threshold:  # Prune low-value states
                    continue

            # Use learned operation selector to prioritize
            op_scores = operation_selector(candidate.current_grid, target_grid, primitives)
            sorted_ops = sorted(zip(primitives, op_scores), key=lambda x: -x[1])

            # Try top-k operations
            for op, score in sorted_ops[:top_k]:
                result = op.execute(candidate.current_grid)

                # Create new candidate with learned score
                new_cand = Candidate(
                    sequence=candidate.sequence + [op],
                    current_grid=result,
                    confidence=score * candidate.confidence
                )
                new_candidates.append(new_cand)

        # Prune by learned value, not distance!
        candidates = prune_by_learned_value(new_candidates, beam_width)

        # Check for solutions
        for cand in candidates:
            if matches(cand.current_grid, target_grid):
                return cand.sequence

    return None
```

**Key differences from current system:**
1. **Value network** replaces distance-based pruning
2. **Operation selector** prioritizes promising ops
3. **No geometric heuristics** needed - learned from data

### Training

#### Data Collection

**Positive examples:** Run current system on tasks, collect all intermediate states from successful solution paths

```python
# From successful solve of task X:
successful_path = [
    (input_grid, [], target),           # Start
    (after_op1, ['rotate_90'], target), # After first op
    (after_op2, ['rotate_90', 'recolor'], target),  # After second op
    (target, ['rotate_90', 'recolor'], target)      # Success!
]

# Each state gets value = 1.0 (led to success)
for state, program, target in successful_path:
    train_data.append((state, program, target, value=1.0))
```

**Negative examples:** Collect intermediate states from failed searches

```python
# From failed search:
failed_paths = [
    (after_tile, ['tile'], target),  # Dead end, value = 0.0
    (after_crop, ['crop'], target),  # Wrong direction, value = 0.0
]
```

**Challenge:** Need lots of data. Solutions:
- Run current system on all 400 training tasks
- Generate synthetic tasks
- Use search failures as negative examples

#### Training Objective

```python
def train_value_network(batch):
    for state, program, target, true_value in batch:
        pred_value = value_network(state, target, program)
        loss = (pred_value - true_value) ** 2  # MSE loss
        loss.backward()
        optimizer.step()
```

```python
def train_operation_selector(batch):
    for state, target, correct_op in batch:
        op_distribution = operation_selector(state, target, all_ops)
        loss = cross_entropy(op_distribution, correct_op)
        loss.backward()
        optimizer.step()
```

### Advantages

1. **Minimal architecture change**
   - Keep existing primitives and search
   - Only replace pruning heuristic
   - Incremental improvement path

2. **Addresses root problem**
   - Learned value can recognize "extract→tile" intermediates
   - No distance-based bias
   - Can learn task-specific patterns

3. **Interpretable**
   - Still using hand-coded primitives
   - Can inspect which states have high value
   - Can visualize operation preferences

4. **Data efficiency**
   - Can train from existing system's solutions
   - Doesn't need program annotations
   - Active learning: collect more data where model is uncertain

### Disadvantages

1. **Still limited by primitive library**
   - If no primitive can do X, model can't help
   - Can't discover novel operations
   - Ceiling determined by primitives

2. **Bootstrap problem**
   - Need successful solutions to train
   - Current system only solves 22% of tasks
   - May not have enough positive examples

3. **Generalization uncertainty**
   - Will model generalize to unseen task types?
   - Training distribution bias (only 22% of tasks)
   - May overfit to solved tasks

4. **Two-model complexity**
   - Need to train both value network and operation selector
   - Hyperparameter tuning for both
   - Coordination between models

### Prior Work

**Learned search heuristics:**
- **AlphaGo/AlphaZero**: Value network + policy network for game search
- **Graph Neural Networks for search**: Learn heuristics for planning problems
- **Neural Guided Search**: Use neural networks to guide symbolic search

**Program synthesis:**
- **Neural-Guided Deduction** (MIT, 2019): Learn to guide theorem proving
- **DeepCoder**: Learn to predict useful operations

### Estimated Impact

**Optimistic:** 30-40% solve rate
- If value network correctly identifies promising intermediates
- Can unlock extract→tile and similar compositions
- +8-18% over baseline

**Realistic:** 25-30% solve rate
- Modest improvement from better pruning
- May not solve fundamentally new task types
- +3-8% over baseline

**Risk:** May not exceed 22% if:
- Insufficient training data (only 11 successful tasks)
- Model doesn't generalize well
- Primitives are still the bottleneck

---

## Approach 3: Hybrid (Primitives + Learned Guidance)

### Core Idea

**Best of both worlds:** Combine hand-coded primitives (interpretable, guaranteed correct) with learned pattern recognition (flexible, data-driven).

**Paradigm:** Use learning where it helps, keep symbolic reasoning where it's needed.

### Architecture

#### Three-Level System

**Level 1: Primitive Library** (Symbolic)
- Hand-coded transformation operations
- Guaranteed correctness (no hallucination)
- Interpretable and debuggable

**Level 2: Learned Pattern Recognizer** (Neural)
- Identify task type / pattern category
- Suggest relevant primitives
- Guide search strategy

**Level 3: Hybrid Search** (Combined)
- Symbolic search over primitives
- Neural guidance for prioritization
- Fallback to heuristics when uncertain

### Components

#### 1. Task Classifier

**Input:** Training demonstrations
**Output:** Task category + confidence

```python
class TaskClassifier(nn.Module):
    """Classify task into known pattern categories."""

    def __init__(self):
        self.categories = [
            'rotation', 'reflection', 'color_mapping', 'tiling',
            'object_extraction', 'pattern_completion', 'symmetry',
            'composition', 'other'
        ]

    def forward(self, demonstrations):
        # Encode demos
        task_emb = self.encoder(demonstrations)

        # Classify
        logits = self.classifier(task_emb)
        probs = torch.softmax(logits, dim=-1)

        category = self.categories[torch.argmax(probs)]
        confidence = torch.max(probs)

        return category, confidence
```

**Training:** Annotate training tasks with categories

**Usage:** Narrow primitive search space based on category

#### 2. Primitive Relevance Predictor

**Input:** Task demonstrations + primitive name
**Output:** Relevance score

```python
class PrimitiveRelevancePredictor(nn.Module):
    """Predict if a primitive is relevant for a task."""

    def forward(self, task_emb, primitive_name):
        primitive_emb = self.primitive_encoder(primitive_name)
        relevance = torch.sigmoid(torch.dot(task_emb, primitive_emb))
        return relevance
```

**Training:** From successful solutions, mark which primitives were used

**Usage:** Filter primitive library before search
```python
# Before search:
relevant_prims = []
task_emb = encoder(demonstrations)
for prim in primitive_library:
    score = relevance_predictor(task_emb, prim.name)
    if score > threshold:
        relevant_prims.append(prim)

# Search with reduced space (36 → 10-15 relevant primitives)
sequence = beam_search(input, output, relevant_prims)
```

#### 3. Composition Detector

**Input:** Task demonstrations
**Output:** Is this a composition task? Which types?

```python
class CompositionDetector(nn.Module):
    """Detect if task requires multi-step composition."""

    def forward(self, demonstrations):
        task_emb = self.encoder(demonstrations)

        # Binary: is composition needed?
        is_composition = torch.sigmoid(self.composition_head(task_emb))

        # If yes, what type?
        composition_types = self.type_head(task_emb)  # [extract→tile, conditional→geometric, etc.]

        return is_composition, composition_types
```

**Training:** Annotate tasks as single-step vs multi-step

**Usage:** Switch search strategy
```python
is_comp, comp_types = composition_detector(demonstrations)

if is_comp > 0.5:
    # Use two-phase search for compositions
    sequence = two_phase_search(input, output, primitives)
else:
    # Use standard beam search
    sequence = beam_search(input, output, primitives)
```

### Hybrid Search Algorithm

```python
def hybrid_search(task, primitives):
    # Step 1: Classify task
    category, conf = task_classifier(task.train)
    print(f"Detected category: {category} (confidence: {conf:.2f})")

    # Step 2: Filter primitives
    task_emb = encoder(task.train)
    relevant_prims = []
    for prim in primitives:
        relevance = relevance_predictor(task_emb, prim.name)
        if relevance > 0.3:
            relevant_prims.append((prim, relevance))

    # Sort by relevance
    relevant_prims.sort(key=lambda x: -x[1])
    print(f"Filtered to {len(relevant_prims)} relevant primitives")

    # Step 3: Detect if composition needed
    is_comp, comp_types = composition_detector(task.train)

    # Step 4: Choose search strategy
    if is_comp > 0.7:
        print("Using two-phase composition search")
        sequence = two_phase_search(
            task.test_input,
            task.test_output,
            relevant_prims,
            composition_types
        )
    elif conf > 0.8 and category in ['rotation', 'reflection', 'color_mapping']:
        print(f"Using category-specific search for {category}")
        sequence = category_specific_search(
            task.test_input,
            task.test_output,
            category,
            relevant_prims
        )
    else:
        print("Using standard beam search")
        sequence = learned_beam_search(
            task.test_input,
            task.test_output,
            relevant_prims
        )

    return sequence
```

### Advantages

1. **Best of both worlds**
   - Symbolic: Correctness, interpretability
   - Neural: Flexibility, pattern recognition
   - Fallback: Use heuristics when model uncertain

2. **Graceful degradation**
   - If neural components fail, still have symbolic search
   - No worse than baseline (only better)
   - Can deploy incrementally

3. **Targeted learning**
   - Only need to learn high-level patterns
   - Don't need to learn primitive execution (already have it)
   - Smaller model, less data needed

4. **Interpretable decisions**
   - Can see which category was predicted
   - Can see which primitives were relevant
   - Can explain why composition search was used

5. **Modular improvements**
   - Can improve each component independently
   - Add new primitives without retraining
   - Update neural components without changing primitives

### Disadvantages

1. **Complexity**
   - Multiple models to train and coordinate
   - More hyperparameters to tune
   - Integration challenges

2. **Still limited by primitives**
   - If no primitive exists for transformation, can't help
   - Neural guidance can't create new operations
   - Ceiling still determined by primitive expressiveness

3. **Training data requirements**
   - Need task category annotations
   - Need primitive relevance labels
   - Need composition type labels

4. **Error propagation**
   - Wrong category → wrong search strategy → failure
   - Wrong primitive filtering → miss correct operation
   - Composition detector false positive → unnecessary complexity

### Prior Work

**Hybrid systems:**
- **AlphaGo**: Neural value/policy + Monte Carlo Tree Search
- **Neuro-Symbolic Program Synthesis**: Combine neural and symbolic reasoning
- **Hybrid Planning**: Neural heuristics + classical planning

**Multi-model systems:**
- **Mixture of Experts**: Route inputs to specialist models
- **Cascaded classifiers**: Fast simple model → slow complex model

### Estimated Impact

**Optimistic:** 35-45% solve rate
- Task classifier improves primitive selection
- Composition detector enables two-phase search
- Relevance predictor speeds up search
- +13-23% over baseline

**Realistic:** 28-35% solve rate
- Modest improvements from each component
- Some tasks benefit greatly, others not at all
- +6-13% over baseline

**Risk:** 22-25% if:
- Task classification is inaccurate
- Primitive filtering removes needed operations
- Composition detection has false negatives

---

## Comparison Matrix

| Aspect | Neural Synthesis | Learned Recognition | Hybrid |
|--------|-----------------|---------------------|--------|
| **Interpretability** | Low (black box) | Medium (symbolic search) | High (best of both) |
| **Training Data Needed** | High (10K+ tasks) | Medium (100s of tasks) | Medium (100s + annotations) |
| **Development Effort** | High (new architecture) | Medium (augment existing) | High (integrate multiple) |
| **Maintenance** | Medium (retrain model) | Medium (retrain networks) | High (multiple components) |
| **Failure Mode** | Silent errors (bad programs) | Falls back to distance | Graceful degradation |
| **Primitive Dependency** | None (learns operations) | High (limited by library) | High (limited by library) |
| **Composition Handling** | Can learn | Needs value network | Explicit detector |
| **Estimated Solve Rate** | 25-50% | 25-40% | 28-45% |
| **Estimated Dev Time** | 6-12 months | 2-4 months | 4-8 months |
| **GPU Requirements** | High (training) | Medium | Medium |
| **Incremental Deployment** | No (replace system) | Yes (augment pruning) | Yes (add components) |

---

## Recommendations

### Short-Term (Next 2-4 months): Hybrid Approach

**Why:**
- Leverages existing primitive library
- Incremental improvements
- Interpretable and debuggable
- Can deploy one component at a time

**Implementation plan:**

**Phase 1: Task Classifier** (1 month)
- Annotate 400 tasks with categories (rotation, tiling, etc.)
- Train simple classifier (ResNet on grid images)
- Use to filter primitive library before search
- Target: +2-5% solve rate from better primitive selection

**Phase 2: Composition Detector** (1 month)
- Annotate tasks as single-step vs multi-step
- Train binary classifier
- Implement two-phase search for detected compositions
- Target: +3-6% solve rate from unlocking compositions

**Phase 3: Primitive Relevance Predictor** (1 month)
- Train from successful solution traces
- Use to prioritize operations in beam search
- Target: +1-3% solve rate from better search ordering

**Total estimated impact:** +6-14% solve rate (28-36% total)

### Medium-Term (6-12 months): Neural Synthesis

**Why:**
- Not limited by primitive library
- Can discover novel operation patterns
- Scales with data

**Prerequisites:**
- Generate synthetic training data (10K+ tasks)
- Build data pipeline
- GPU infrastructure

**Implementation plan:**

**Phase 1: Data Generation** (2 months)
- Create synthetic task generator
- Generate 10K tasks with known solutions
- Verify diversity and difficulty

**Phase 2: Model Development** (3 months)
- Implement encoder-decoder architecture
- Train on synthetic + real data
- Evaluation and iteration

**Phase 3: Integration** (1 month)
- Replace search with neural generation
- Fallback to primitives when confidence low
- Production deployment

**Total estimated impact:** +3-28% solve rate (25-50% total)

### Long-Term (12+ months): Hybrid + Synthesis

**Ultimate system:**
- Neural synthesis for novel tasks
- Learned guidance for primitive search
- Symbolic primitives for guaranteed correctness
- Fallback hierarchy

**Target:** 40-60% solve rate

---

## Conclusion

**All three approaches address the fundamental limitation:** Current distance-based beam search over hand-coded primitives hits a ceiling at ~22-25%.

**Best immediate path: Hybrid approach**
- Incremental improvements
- Leverages existing work
- Interpretable and deployable
- 6-14% estimated gain

**Best long-term path: Neural synthesis**
- Not limited by primitives
- Can learn novel patterns
- Requires significant investment
- 3-28% estimated gain

**Pragmatic recommendation:**
1. Start with Hybrid (2-4 months) to get quick wins
2. Collect more data while deploying Hybrid
3. Invest in Neural Synthesis (6-12 months) for long-term gains
4. Combine both in ultimate system

The key insight: **We've learned what doesn't work** (more primitives, more search budget). Time to try fundamentally different approaches.
