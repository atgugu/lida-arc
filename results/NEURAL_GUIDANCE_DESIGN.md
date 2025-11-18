# Neural Guidance Network: Self-Play Training Design

*Design Date: 2025-11-18*
*Status: 📋 Design Phase*

---

## Overview

**Goal:** Train neural networks using self-play puzzles to guide the symbolic solver's search, creating a hybrid system that combines symbolic reasoning with learned heuristics.

**Key Insight:** Since we generated 1K puzzles with known ground-truth transformations, we have perfect supervision for training guidance networks.

---

## Architecture: Hybrid Solver

```
┌─────────────────────────────────────────────────────────────┐
│                    ARC Puzzle Input                          │
│                   (train pairs + test input)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              NEURAL GUIDANCE NETWORKS                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Task Classifier                                  │   │
│  │     Input: Grid features                             │   │
│  │     Output: Task type probabilities                  │   │
│  │     (geometric, color, tiling, composition)          │   │
│  └──────────────────────────────────────────────────────┘   │
│                       │                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Primitive Relevance Predictor                    │   │
│  │     Input: Grid features + task type                 │   │
│  │     Output: Score for each primitive (0-1)           │   │
│  │     Purpose: Prune search space                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                       │                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Composition Predictor                            │   │
│  │     Input: Grid features + primitives used so far    │   │
│  │     Output: Next primitive probabilities             │   │
│  │     Purpose: Guide composition sequences             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ Guidance scores
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SYMBOLIC SOLVER (Existing)                      │
│  - Beam search with neural-guided priorities                │
│  - Primitive library (36 operations)                        │
│  - Stage 1: Single primitive search                         │
│  - Stage 2: Composition search                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Task Classifier

### Purpose
Quickly identify the type of transformation needed, enabling targeted primitive selection.

### Architecture
```python
class TaskClassifier(nn.Module):
    """Classify puzzle into task categories."""

    def __init__(self):
        super().__init__()
        self.categories = [
            'geometric',      # rotate, reflect, transpose
            'color_mapping',  # recolor operations
            'tiling',         # tile, repeat patterns
            'cropping',       # auto_crop, extract
            'composition',    # multiple operations
        ]

        # Grid encoder (CNN)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),  # 10 color channels (0-9)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(self.categories))
        )

    def forward(self, grid_pair):
        """
        Args:
            grid_pair: (batch, 2, 10, H, W) - input and output grids
        Returns:
            logits: (batch, num_categories)
        """
        # Encode both input and output
        input_feat = self.grid_encoder(grid_pair[:, 0])
        output_feat = self.grid_encoder(grid_pair[:, 1])

        # Combine features
        combined = input_feat.squeeze() + output_feat.squeeze()

        # Classify
        return self.classifier(combined)
```

### Training Data Extraction
```python
def extract_task_labels(puzzle):
    """Extract task type from ground truth transformation."""
    transformation = puzzle['transformation']

    # Check transformation operations
    ops = [op_name for op_name, _ in transformation]

    if len(ops) > 2:
        return 'composition'
    elif 'tile' in ops:
        return 'tiling'
    elif 'recolor' in ops:
        return 'color_mapping'
    elif 'auto_crop' in ops:
        return 'cropping'
    elif any(op in ops for op in ['rotate_90', 'rotate_180', 'rotate_270',
                                    'reflect_horizontal', 'reflect_vertical']):
        return 'geometric'
    else:
        return 'composition'
```

---

## Component 2: Primitive Relevance Predictor

### Purpose
Score each of the 36 primitives by relevance to the current puzzle, reducing search space from 36^depth to top-k^depth.

### Architecture
```python
class PrimitiveRelevancePredictor(nn.Module):
    """Predict which primitives are relevant for a puzzle."""

    def __init__(self, num_primitives=36):
        super().__init__()
        self.num_primitives = num_primitives

        # Grid encoder (shared)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Task embedding (from classifier)
        self.task_embedding = nn.Embedding(5, 32)  # 5 task types

        # Relevance predictor
        self.relevance_head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_primitives),
            nn.Sigmoid()  # Output scores 0-1
        )

    def forward(self, grid_pair, task_type):
        """
        Args:
            grid_pair: (batch, 2, 10, H, W)
            task_type: (batch,) - task type indices
        Returns:
            relevance_scores: (batch, num_primitives)
        """
        # Encode grids
        input_feat = self.grid_encoder(grid_pair[:, 0])
        output_feat = self.grid_encoder(grid_pair[:, 1])
        combined = (input_feat.squeeze() + output_feat.squeeze()) / 2

        # Embed task type
        task_emb = self.task_embedding(task_type)

        # Predict relevance
        features = torch.cat([combined, task_emb], dim=1)
        return self.relevance_head(features)
```

### Training Data Extraction
```python
def extract_primitive_labels(puzzle, all_primitives):
    """Extract binary labels for primitive relevance.

    Args:
        puzzle: Generated puzzle with ground truth transformation
        all_primitives: List of all 36 primitive names

    Returns:
        labels: Binary array (36,) - 1 if primitive is used, 0 otherwise
    """
    transformation = puzzle['transformation']
    used_ops = set(op_name for op_name, _ in transformation)

    labels = np.zeros(len(all_primitives))
    for i, prim_name in enumerate(all_primitives):
        if prim_name in used_ops:
            labels[i] = 1.0

    return labels
```

### Usage in Solver
```python
# During search, filter primitives by relevance threshold
relevance_scores = primitive_predictor(grid_pair, task_type)
top_k_prims = torch.topk(relevance_scores, k=10).indices

# Only search these top-k primitives instead of all 36
for prim_idx in top_k_prims:
    primitive = primitives[prim_idx]
    # ... continue search
```

---

## Component 3: Composition Predictor

### Purpose
Given a partial composition sequence, predict the next primitive to try. Enables guided multi-step search.

### Architecture
```python
class CompositionPredictor(nn.Module):
    """Predict next primitive in a composition sequence."""

    def __init__(self, num_primitives=36, max_seq_len=5):
        super().__init__()
        self.num_primitives = num_primitives

        # Grid encoder
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Sequence encoder (LSTM over primitives used so far)
        self.primitive_embedding = nn.Embedding(num_primitives + 1, 64)  # +1 for padding
        self.sequence_encoder = nn.LSTM(64, 128, batch_first=True)

        # Next primitive predictor
        self.next_primitive_head = nn.Sequential(
            nn.Linear(128 + 128, 128),  # grid + sequence features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_primitives),
        )

    def forward(self, grid_pair, primitive_sequence):
        """
        Args:
            grid_pair: (batch, 2, 10, H, W) - current state
            primitive_sequence: (batch, seq_len) - primitives used so far
        Returns:
            next_primitive_logits: (batch, num_primitives)
        """
        # Encode current grid state
        input_feat = self.grid_encoder(grid_pair[:, 0])

        # Encode primitive sequence
        seq_emb = self.primitive_embedding(primitive_sequence)
        _, (hidden, _) = self.sequence_encoder(seq_emb)
        seq_feat = hidden[-1]  # Last hidden state

        # Predict next primitive
        combined = torch.cat([input_feat.squeeze(), seq_feat], dim=1)
        return self.next_primitive_head(combined)
```

### Training Data Extraction
```python
def extract_composition_sequences(puzzle):
    """Extract training examples for composition prediction.

    For transformation [A, B, C], generate:
    - (input, []) -> A
    - (intermediate_1, [A]) -> B
    - (intermediate_2, [A, B]) -> C
    """
    transformation = puzzle['transformation']
    train_pairs = puzzle['train']

    examples = []
    for input_grid, output_grid in train_pairs:
        # Apply transformations incrementally
        current = input_grid
        sequence = []

        for i, (op_name, params) in enumerate(transformation):
            # Create training example: (current_state, sequence_so_far) -> next_op
            examples.append({
                'input_grid': current,
                'output_grid': output_grid,
                'primitive_sequence': sequence.copy(),
                'next_primitive': op_name
            })

            # Apply operation to get next state
            current = apply_primitive(current, op_name, params)
            sequence.append(op_name)

    return examples
```

---

## Training Pipeline

### Dataset Preparation
```python
class SelfPlayDataset(torch.utils.data.Dataset):
    """Dataset from self-play generated puzzles."""

    def __init__(self, puzzles_path, primitive_library):
        """
        Args:
            puzzles_path: Path to generated_puzzles.json
            primitive_library: PrimitiveLibrary instance
        """
        with open(puzzles_path) as f:
            self.puzzles = json.load(f)

        self.primitives = primitive_library
        self.primitive_to_idx = {
            name: i for i, name in enumerate(primitive_library.list_primitives())
        }

        # Extract all training examples
        self.examples = self._prepare_examples()

    def _prepare_examples(self):
        """Extract training examples from puzzles."""
        examples = []

        for puzzle in self.puzzles:
            # Extract task classification label
            task_label = extract_task_labels(puzzle)

            # Extract primitive relevance labels
            prim_labels = extract_primitive_labels(
                puzzle, self.primitive_to_idx.keys()
            )

            # Extract composition sequences
            comp_examples = extract_composition_sequences(puzzle)

            # Use first training pair for grid encoding
            input_grid, output_grid = puzzle['train'][0]

            examples.append({
                'input_grid': input_grid,
                'output_grid': output_grid,
                'task_label': task_label,
                'primitive_labels': prim_labels,
                'composition_examples': comp_examples,
                'puzzle_id': puzzle['puzzle_id'],
                'difficulty': puzzle['difficulty']
            })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
```

### Training Loop
```python
def train_guidance_networks(
    train_puzzles_path='results/self_play_1k/generated_puzzles.json',
    epochs=50,
    batch_size=32,
    device='cuda'
):
    """Train all neural guidance networks."""

    # Load primitive library
    primitive_library = PrimitiveLibrary()

    # Create dataset
    dataset = SelfPlayDataset(train_puzzles_path, primitive_library)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    task_classifier = TaskClassifier().to(device)
    primitive_predictor = PrimitiveRelevancePredictor(
        num_primitives=len(primitive_library.list_primitives())
    ).to(device)
    composition_predictor = CompositionPredictor(
        num_primitives=len(primitive_library.list_primitives())
    ).to(device)

    # Optimizers
    opt_task = torch.optim.Adam(task_classifier.parameters(), lr=1e-3)
    opt_prim = torch.optim.Adam(primitive_predictor.parameters(), lr=1e-3)
    opt_comp = torch.optim.Adam(composition_predictor.parameters(), lr=1e-3)

    # Loss functions
    task_loss_fn = nn.CrossEntropyLoss()
    prim_loss_fn = nn.BCELoss()  # Multi-label classification
    comp_loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        task_losses = []
        prim_losses = []
        comp_losses = []

        for batch in train_loader:
            # Prepare batch
            input_grids = batch['input_grid'].to(device)
            output_grids = batch['output_grid'].to(device)
            task_labels = batch['task_label'].to(device)
            prim_labels = batch['primitive_labels'].to(device)

            # Grid pair: (batch, 2, 10, H, W)
            grid_pair = encode_grid_pair(input_grids, output_grids)

            # 1. Train task classifier
            opt_task.zero_grad()
            task_logits = task_classifier(grid_pair)
            task_loss = task_loss_fn(task_logits, task_labels)
            task_loss.backward()
            opt_task.step()
            task_losses.append(task_loss.item())

            # 2. Train primitive predictor
            opt_prim.zero_grad()
            prim_scores = primitive_predictor(grid_pair, task_labels)
            prim_loss = prim_loss_fn(prim_scores, prim_labels)
            prim_loss.backward()
            opt_prim.step()
            prim_losses.append(prim_loss.item())

            # 3. Train composition predictor
            # (More complex - need to iterate over composition examples)
            for comp_example in batch['composition_examples']:
                opt_comp.zero_grad()
                comp_logits = composition_predictor(
                    comp_example['grid_pair'],
                    comp_example['primitive_sequence']
                )
                comp_loss = comp_loss_fn(
                    comp_logits,
                    comp_example['next_primitive']
                )
                comp_loss.backward()
                opt_comp.step()
                comp_losses.append(comp_loss.item())

        # Log progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Task Loss: {np.mean(task_losses):.4f}")
        print(f"  Primitive Loss: {np.mean(prim_losses):.4f}")
        print(f"  Composition Loss: {np.mean(comp_losses):.4f}")

    # Save models
    torch.save(task_classifier.state_dict(), 'models/task_classifier.pt')
    torch.save(primitive_predictor.state_dict(), 'models/primitive_predictor.pt')
    torch.save(composition_predictor.state_dict(), 'models/composition_predictor.pt')

    return task_classifier, primitive_predictor, composition_predictor
```

---

## Integration with Solver

### Guided Beam Search

```python
class NeuralGuidedSolver(ARCCognitiveSolver):
    """Solver with neural guidance networks."""

    def __init__(self, config, guidance_models):
        super().__init__(config)

        self.task_classifier = guidance_models['task_classifier']
        self.primitive_predictor = guidance_models['primitive_predictor']
        self.composition_predictor = guidance_models['composition_predictor']

    def solve_with_guidance(self, task):
        """Solve using neural guidance."""

        # Step 1: Classify task type
        grid_pair = encode_task_examples(task)
        task_type = self.task_classifier(grid_pair).argmax()

        # Step 2: Get relevant primitives
        prim_scores = self.primitive_predictor(grid_pair, task_type)
        top_k_prims = torch.topk(prim_scores, k=10).indices

        # Step 3: Guided beam search
        beam = self._initialize_beam()

        for depth in range(self.max_depth):
            new_beam = []

            for state in beam:
                # Get next primitive suggestions
                comp_logits = self.composition_predictor(
                    state.grid_pair,
                    state.primitive_sequence
                )

                # Sort primitives by neural score
                sorted_prims = torch.argsort(comp_logits, descending=True)

                # Try top primitives first (neural-guided)
                for prim_idx in sorted_prims[:self.beam_width]:
                    if prim_idx not in top_k_prims:
                        continue  # Skip if not in relevant set

                    # Apply primitive
                    new_state = self._apply_primitive(state, prim_idx)

                    # Score with combined metric
                    neural_score = comp_logits[prim_idx].item()
                    symbolic_score = self._evaluate_state(new_state)
                    combined_score = 0.7 * neural_score + 0.3 * symbolic_score

                    new_beam.append((combined_score, new_state))

            # Keep top-k states
            beam = [state for _, state in sorted(new_beam, reverse=True)[:self.beam_width]]

            # Check for solution
            for state in beam:
                if self._is_solution(state, task):
                    return state.primitive_sequence

        return None  # No solution found
```

---

## Training Schedule

### Month 2: Foundation (Weeks 5-8)

**Week 5: Data Preparation**
- Implement `SelfPlayDataset` class
- Extract all training labels from 1K puzzles
- Create train/val/test splits (70/15/15)
- Implement grid encoding functions
- **Deliverable:** PyTorch dataset with 1K examples

**Week 6: Task Classifier**
- Implement `TaskClassifier` model
- Train on 1K puzzles
- Evaluate classification accuracy
- **Target:** >85% accuracy on task type prediction

**Week 7: Primitive Predictor**
- Implement `PrimitiveRelevancePredictor` model
- Train on 1K puzzles
- Evaluate precision/recall for relevant primitives
- **Target:** Recall >90% (don't miss relevant primitives)

**Week 8: Composition Predictor**
- Implement `CompositionPredictor` model
- Extract composition sequences (3-5K training examples)
- Train sequence prediction
- **Target:** Top-3 accuracy >70%

### Month 3: Integration (Weeks 9-12)

**Week 9: Guided Solver**
- Implement `NeuralGuidedSolver` class
- Integrate neural guidance into beam search
- Test on self-play validation set

**Week 10: Hyperparameter Tuning**
- Tune neural/symbolic score weighting
- Tune beam width with guidance
- Tune top-k primitive filtering

**Week 11: Real ARC Evaluation**
- Test on 49 real ARC evaluation tasks
- Compare to baseline (22.4% solve rate)
- **Target:** 28-32% solve rate (+5-10%)

**Week 12: Analysis & Iteration**
- Analyze failure cases
- Identify which guidance component helps most
- Generate 10K more puzzles if needed

---

## Expected Improvements

### Search Space Reduction
**Baseline:** 36 primitives → 36^5 = 60M possible sequences (depth 5)
**With guidance:** Top-10 primitives → 10^5 = 100K sequences (600x reduction)

### Solve Rate Prediction
| Component | Expected Gain | Mechanism |
|-----------|---------------|-----------|
| **Task Classifier** | +2-3% | Better primitive selection per task type |
| **Primitive Predictor** | +3-5% | Prune irrelevant primitives early |
| **Composition Predictor** | +2-4% | Guide multi-step search |
| **Combined** | +7-12% | Synergistic effects |

**Projected solve rate:** 22.4% baseline → **29-34%** with guidance

### Training Efficiency
- 1K puzzles = 1K task examples + 3-5K composition examples
- Training time: ~2-4 hours on single GPU
- Inference time: <10ms per puzzle (negligible overhead)

---

## Validation Metrics

### Offline Metrics (on held-out self-play puzzles)
1. **Task Accuracy:** % of task types correctly classified
2. **Primitive Recall:** % of ground-truth primitives ranked in top-10
3. **Composition Top-K:** % of next primitives in top-3 predictions
4. **Search Efficiency:** Average depth to find solution

### Online Metrics (on real ARC tasks)
1. **Solve Rate:** % of tasks solved (target: 28-32%)
2. **Search Speedup:** Time to solution vs baseline
3. **Beam Efficiency:** % reduction in beam candidates explored
4. **Failure Analysis:** Where does guidance fail?

---

## Risks & Mitigation

### Risk 1: Overfitting to Self-Play Distribution
**Problem:** Neural models learn self-play patterns but don't generalize to real ARC

**Mitigation:**
- Use data augmentation (rotate, reflect puzzles)
- Train on diverse difficulty levels
- Validate on real ARC tasks during training
- Use regularization (dropout, weight decay)

### Risk 2: Self-Play Puzzles Too Simple
**Problem:** 1K puzzles cover only ~40% of real ARC transformation types

**Mitigation:**
- Generate 10K+ puzzles with more primitive types
- Add scaling, object manipulation, pattern extraction primitives
- Mix self-play data with real ARC demonstrations

### Risk 3: Neural Guidance Misleads Solver
**Problem:** Incorrect neural predictions lead solver astray

**Mitigation:**
- Use neural scores as soft guidance, not hard constraints
- Combine neural + symbolic scores (weighted average)
- Fallback to symbolic search if guided search fails
- Monitor "guidance accuracy" - how often top-1 prediction is correct

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Task classifier: >80% accuracy
- ✅ Primitive predictor: >85% recall
- ✅ Composition predictor: >60% top-3 accuracy
- ✅ Integrated solver: >25% solve rate on real ARC

### Target Performance
- ✅ Task classifier: >90% accuracy
- ✅ Primitive predictor: >95% recall, <5% false positive rate
- ✅ Composition predictor: >75% top-3 accuracy
- ✅ Integrated solver: >30% solve rate on real ARC (+8% over baseline)

### Stretch Goals
- ✅ >35% solve rate with 10K+ training puzzles
- ✅ <1s average solve time (10x speedup)
- ✅ Generalizes to novel transformation types
- ✅ Curriculum learning: progressively harder puzzles → better performance

---

## Implementation Plan

### Phase 1: Foundation (2 weeks)
1. ✅ Design neural architectures (this document)
2. ⏭ Implement `SelfPlayDataset` class
3. ⏭ Implement grid encoding utilities
4. ⏭ Create train/val/test splits

### Phase 2: Model Training (3 weeks)
5. ⏭ Implement & train `TaskClassifier`
6. ⏭ Implement & train `PrimitiveRelevancePredictor`
7. ⏭ Implement & train `CompositionPredictor`
8. ⏭ Evaluate offline metrics

### Phase 3: Integration (2 weeks)
9. ⏭ Implement `NeuralGuidedSolver`
10. ⏭ Integrate guidance into beam search
11. ⏭ Tune hyperparameters (score weighting, beam width)

### Phase 4: Evaluation (1 week)
12. ⏭ Test on real ARC evaluation set (49 tasks)
13. ⏭ Compare to baseline (22.4%)
14. ⏭ Analyze results and iterate

**Total timeline:** 8 weeks to neural-guided solver

---

## Conclusion

**Key Innovation:** Leverage self-play puzzles with known ground-truth transformations to train neural guidance networks that augment symbolic reasoning.

**Expected Impact:**
- 7-12% improvement in solve rate (22% → 29-34%)
- 600x search space reduction
- <10ms inference overhead
- Scalable to 10K+ training puzzles

**Next Steps:**
1. Implement data preparation pipeline
2. Train initial models on 1K puzzles
3. Validate offline metrics
4. Integrate into solver and evaluate on real ARC

**Meta-insight:** Self-play doesn't just generate training data—it creates a supervised learning problem from an unsupervised one. By knowing the ground truth transformations, we can train guidance networks to learn "how to search" rather than "how to transform."

---

*Status: 📋 Design Complete - Ready for Implementation*
*Estimated completion: 8 weeks*
*Expected solve rate: 29-34% (up from 22.4%)*
