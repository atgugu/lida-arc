# Self-Play Architecture for ARC-AGI: Adversarial Puzzle Generation

*Design Date: 2025-11-18*
*Context: Generate synthetic training data to overcome 22% ceiling*

## Executive Summary

**Core Idea:** Create a system where a **Generator** creates ARC puzzles and a **Solver** tries to solve them, with adversarial dynamics that push both to improve.

**Key Innovation:** Unlike supervised learning (limited by 400 public tasks), self-play can generate unlimited training data while ensuring puzzles are:
1. **Valid** - Consistent transformation rules
2. **Solvable** - Within system's capabilities (with difficulty curve)
3. **Diverse** - Cover wide range of patterns
4. **Educational** - Target solver's weaknesses

**Estimated Impact:**
- Generate 10K+ synthetic tasks for training neural components
- Identify system weaknesses systematically
- Curriculum learning from easy → hard puzzles
- Bootstrap neural synthesis without manual annotations

---

## Architecture Overview

### Three-Component System

```
┌─────────────┐
│  Generator  │ ──(puzzle)──> │
│             │               │
│ Creates     │               ▼
│ Puzzles     │         ┌──────────┐      solve success?
└─────────────┘         │  Solver  │ ───────────────>
      ▲                 │          │
      │                 │ Attempts │
      │                 │ Solution │
   reward               └──────────┘
      │                       │
      │                  difficulty,
      │                  diversity
      │                       │
      │                       ▼
      │              ┌─────────────┐
      └──────────────│  Evaluator  │
                     │             │
                     │ Judges both │
                     └─────────────┘
```

**Adversarial Dynamics:**
- Generator wants: Create challenging puzzles Solver can barely solve
- Solver wants: Solve any puzzle Generator creates
- Evaluator: Ensures quality, diversity, difficulty curve

---

## Component 1: Puzzle Generator

### Approach A: Primitive Composition (Simple, Start Here)

**Idea:** Compose existing primitives to create transformations

```python
class PrimitiveCompositionGenerator:
    """Generate puzzles by composing primitives."""

    def __init__(self, primitive_library, max_composition_length=3):
        self.primitives = primitive_library
        self.max_length = max_composition_length

    def generate_puzzle(self, difficulty='medium'):
        """Generate a puzzle with specified difficulty."""

        # Step 1: Choose transformation (sequence of primitives)
        transformation = self._sample_transformation(difficulty)

        # Step 2: Generate input grid
        input_grid = self._generate_input_grid()

        # Step 3: Apply transformation to create output
        output_grid = self._apply_transformation(input_grid, transformation)

        # Step 4: Create multiple demonstration pairs
        demonstrations = []
        for _ in range(num_demos):
            demo_input = self._generate_input_grid()
            demo_output = self._apply_transformation(demo_input, transformation)
            demonstrations.append((demo_input, demo_output))

        # Step 5: Create test pair
        test_input = self._generate_input_grid()
        test_output = self._apply_transformation(test_input, transformation)

        return {
            'train': demonstrations,
            'test': [(test_input, test_output)],
            'transformation': transformation,  # Ground truth (for training)
            'difficulty': difficulty
        }

    def _sample_transformation(self, difficulty):
        """Sample transformation based on difficulty."""

        if difficulty == 'easy':
            # Single operation
            num_ops = 1
        elif difficulty == 'medium':
            # 2-3 operations
            num_ops = np.random.randint(2, 4)
        else:  # hard
            # 3-5 operations
            num_ops = np.random.randint(3, 6)

        # Sample operations
        transformation = []
        for _ in range(num_ops):
            op = np.random.choice(self.primitives.get_all_names())
            params = self._sample_parameters(op)
            transformation.append((op, params))

        return transformation

    def _generate_input_grid(self):
        """Generate random input grid."""

        # Sample grid size
        height = np.random.randint(3, 10)
        width = np.random.randint(3, 10)

        # Sample grid type
        grid_type = np.random.choice([
            'sparse',     # Few non-zero pixels
            'dense',      # Many non-zero pixels
            'structured', # Geometric patterns
            'random'      # Random colors
        ])

        if grid_type == 'sparse':
            grid = self._generate_sparse_grid(height, width)
        elif grid_type == 'dense':
            grid = self._generate_dense_grid(height, width)
        elif grid_type == 'structured':
            grid = self._generate_structured_grid(height, width)
        else:
            grid = self._generate_random_grid(height, width)

        return grid

    def _generate_sparse_grid(self, h, w):
        """Grid with ~10-20% non-zero pixels."""
        grid = [[0] * w for _ in range(h)]
        num_pixels = int(h * w * np.random.uniform(0.1, 0.2))

        for _ in range(num_pixels):
            r = np.random.randint(0, h)
            c = np.random.randint(0, w)
            color = np.random.randint(1, 10)  # Colors 1-9
            grid[r][c] = color

        return grid

    def _generate_structured_grid(self, h, w):
        """Grid with geometric patterns (rectangles, lines, etc.)."""
        grid = [[0] * w for _ in range(h)]

        # Add random rectangles
        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            r1 = np.random.randint(0, h-1)
            c1 = np.random.randint(0, w-1)
            r2 = np.random.randint(r1+1, h)
            c2 = np.random.randint(c1+1, w)
            color = np.random.randint(1, 10)

            # Fill rectangle
            for r in range(r1, r2):
                for c in range(c1, c2):
                    grid[r][c] = color

        return grid

    def _apply_transformation(self, grid, transformation):
        """Apply sequence of operations to grid."""
        current = grid
        for op_name, params in transformation:
            prim = self.primitives.get(op_name)
            try:
                if params:
                    current = prim.execute(current, **params)
                else:
                    current = prim.execute(current)
            except Exception as e:
                # If transformation fails, return original grid
                # (will be detected as invalid puzzle)
                return None

        return current

    def _sample_parameters(self, op_name):
        """Sample parameters for parametric operations."""

        if op_name == 'recolor':
            # Sample random color mapping
            colors = list(range(1, 10))
            np.random.shuffle(colors)
            color_map = {i: colors[i-1] for i in range(1, 10)}
            return {'color_map': color_map}

        elif op_name == 'tile':
            repeat_v = np.random.randint(2, 4)
            repeat_w = np.random.randint(2, 4)
            return {'repeat_v': repeat_v, 'repeat_w': repeat_w}

        elif op_name == 'scale_grid':
            scale = np.random.choice([0.5, 2.0, 3.0])
            return {'scale_factor': scale}

        # ... more parameter sampling for other operations

        return {}
```

**Validation:** Ensure puzzle is valid before using
```python
def validate_puzzle(puzzle):
    """Check if puzzle is well-formed."""

    # 1. Check consistency: Same transformation applies to all demos
    for demo_input, demo_output in puzzle['train']:
        predicted = apply_transformation(demo_input, puzzle['transformation'])
        if predicted != demo_output:
            return False, "Inconsistent transformation"

    # 2. Check non-trivial: Output differs from input
    for demo_input, demo_output in puzzle['train']:
        if demo_input == demo_output:
            return False, "Trivial transformation (identity)"

    # 3. Check solvability: Transformation produces valid grids
    if puzzle['test'][0][1] is None:
        return False, "Transformation produces invalid grid"

    # 4. Check diversity: Demos have different inputs
    inputs = [demo[0] for demo in puzzle['train']]
    if len(set(map(tuple, map(tuple, inputs)))) < len(inputs) * 0.8:
        return False, "Insufficient input diversity"

    return True, "Valid"
```

### Approach B: Neural Generator (Advanced)

**Idea:** Learn to generate puzzles using neural network

```python
class NeuralPuzzleGenerator(nn.Module):
    """Neural network that generates ARC puzzles."""

    def __init__(self):
        self.latent_dim = 256
        self.encoder = nn.TransformerEncoder(...)
        self.decoder = nn.TransformerDecoder(...)

    def forward(self, difficulty, diversity_target):
        """Generate puzzle conditioned on difficulty and desired diversity."""

        # Sample from latent space
        z = torch.randn(self.latent_dim)

        # Condition on difficulty
        z = torch.cat([z, self.difficulty_embedding(difficulty)])

        # Generate transformation program
        program = self.decoder(z)

        # Generate input grids
        input_grids = self.input_generator(z, num_demos=3)

        # Apply program to create outputs
        output_grids = [self.execute(inp, program) for inp in input_grids]

        return {
            'train': list(zip(input_grids[:-1], output_grids[:-1])),
            'test': [(input_grids[-1], output_grids[-1])],
            'program': program
        }
```

**Training:** GAN-like adversarial training
- Generator tries to fool Solver
- Discriminator ensures valid puzzles
- Reward for puzzles that challenge Solver

### Approach C: Mutation-Based (Hybrid)

**Idea:** Start from real ARC tasks, apply mutations

```python
class MutationGenerator:
    """Generate variations of existing tasks."""

    def mutate_task(self, original_task):
        """Create variant of existing task."""

        mutation_type = np.random.choice([
            'change_colors',      # Remap colors
            'change_size',        # Scale grids
            'add_noise',          # Add random pixels
            'compose_operation',  # Add extra transformation step
            'simplify',           # Remove transformation step
        ])

        if mutation_type == 'change_colors':
            return self._mutate_colors(original_task)
        elif mutation_type == 'change_size':
            return self._mutate_size(original_task)
        elif mutation_type == 'compose_operation':
            return self._compose_operation(original_task)
        # ... etc

    def _mutate_colors(self, task):
        """Remap colors in task."""
        # Sample random color permutation
        colors = list(range(1, 10))
        np.random.shuffle(colors)
        color_map = {i: colors[i-1] for i in range(1, 10)}

        # Apply to all grids
        new_train = []
        for inp, out in task['train']:
            new_inp = self._remap_colors(inp, color_map)
            new_out = self._remap_colors(out, color_map)
            new_train.append((new_inp, new_out))

        # Apply to test
        test_inp, test_out = task['test'][0]
        new_test_inp = self._remap_colors(test_inp, color_map)
        new_test_out = self._remap_colors(test_out, color_map)

        return {
            'train': new_train,
            'test': [(new_test_inp, new_test_out)],
            'parent_task': task['task_id'],
            'mutation': 'color_remap'
        }
```

---

## Component 2: Solver (Existing System)

**Use current primitive-based solver:**

```python
class Solver:
    """Wrapper around existing ARC solver."""

    def __init__(self):
        self.cognitive_solver = ARCCognitiveSolver()

    def solve(self, puzzle):
        """Attempt to solve puzzle."""

        task = ARCTask(
            task_id=f"synthetic_{uuid.uuid4()}",
            train=[GridPair(inp, out) for inp, out in puzzle['train']],
            test=[GridPair(inp, out) for inp, out in puzzle['test']]
        )

        result = self.cognitive_solver.evaluate(task, test_index=0)

        return {
            'solved': result['solved'],
            'accuracy': result['accuracy'],
            'time': result.get('time', None),
            'pattern': result.get('pattern', None)
        }
```

---

## Component 3: Evaluator (Quality Control)

**Ensures generated puzzles are useful:**

```python
class PuzzleEvaluator:
    """Evaluate puzzle quality for self-play."""

    def evaluate(self, puzzle, solver_result):
        """Score puzzle on multiple criteria."""

        scores = {}

        # 1. Validity (hard constraint)
        is_valid, reason = validate_puzzle(puzzle)
        if not is_valid:
            return {'valid': False, 'reason': reason, 'total_score': 0.0}

        # 2. Difficulty (sweet spot)
        scores['difficulty'] = self._score_difficulty(puzzle, solver_result)

        # 3. Diversity (compared to existing puzzles)
        scores['diversity'] = self._score_diversity(puzzle)

        # 4. Educational value (targets weaknesses)
        scores['educational'] = self._score_educational_value(puzzle, solver_result)

        # 5. Efficiency (not too slow to solve)
        scores['efficiency'] = self._score_efficiency(solver_result)

        # Weighted total
        total = (
            0.3 * scores['difficulty'] +
            0.25 * scores['diversity'] +
            0.25 * scores['educational'] +
            0.2 * scores['efficiency']
        )

        return {
            'valid': True,
            'scores': scores,
            'total_score': total
        }

    def _score_difficulty(self, puzzle, solver_result):
        """Score based on difficulty sweet spot."""

        if solver_result['solved']:
            # Solved: Reward if took effort
            time = solver_result.get('time', 0)

            if time < 1.0:
                # Too easy
                return 0.3
            elif time < 5.0:
                # Good difficulty
                return 1.0
            else:
                # Too hard but solvable
                return 0.7

        else:
            # Not solved
            accuracy = solver_result['accuracy']

            if accuracy < 0.1:
                # Way too hard (no progress)
                return 0.1
            elif accuracy < 0.5:
                # Very hard (some progress)
                return 0.5
            else:
                # Almost solved (near miss)
                return 0.8

    def _score_diversity(self, puzzle):
        """Score based on novelty compared to existing puzzles."""

        # Extract features
        features = self._extract_features(puzzle)

        # Compare to database of existing puzzles
        similarities = []
        for existing in self.puzzle_database:
            existing_features = self._extract_features(existing)
            sim = cosine_similarity(features, existing_features)
            similarities.append(sim)

        # Reward if different from existing
        max_similarity = max(similarities) if similarities else 0.0
        diversity_score = 1.0 - max_similarity

        return diversity_score

    def _score_educational_value(self, puzzle, solver_result):
        """Score based on whether puzzle targets known weaknesses."""

        # Identify puzzle type
        puzzle_type = self._classify_puzzle(puzzle)

        # Check if this type is under-represented in training
        type_counts = self.get_type_distribution()
        current_count = type_counts.get(puzzle_type, 0)

        # Reward under-represented types
        if current_count < 100:
            return 1.0
        elif current_count < 500:
            return 0.7
        else:
            return 0.3

    def _extract_features(self, puzzle):
        """Extract feature vector for puzzle."""

        features = []

        # Transformation complexity
        if 'transformation' in puzzle:
            features.append(len(puzzle['transformation']))  # Num operations

        # Grid properties
        for inp, out in puzzle['train']:
            features.append(len(inp))  # Height
            features.append(len(inp[0]))  # Width
            features.append(self._count_colors(inp))
            features.append(self._count_colors(out))
            features.append(self._sparsity(inp))

        # Size change
        inp, out = puzzle['train'][0]
        features.append(len(out) / len(inp))  # Height ratio
        features.append(len(out[0]) / len(inp[0]))  # Width ratio

        return np.array(features)
```

---

## Self-Play Training Loop

### Curriculum Learning Strategy

```python
class SelfPlayTrainer:
    """Orchestrate self-play training."""

    def __init__(self):
        self.generator = PrimitiveCompositionGenerator(PrimitiveLibrary())
        self.solver = Solver()
        self.evaluator = PuzzleEvaluator()

        self.puzzle_database = []
        self.statistics = {
            'generated': 0,
            'valid': 0,
            'solved': 0,
            'by_difficulty': {'easy': 0, 'medium': 0, 'hard': 0}
        }

    def train(self, num_iterations=10000):
        """Main self-play training loop."""

        for iteration in range(num_iterations):
            # 1. Sample difficulty (curriculum)
            difficulty = self._curriculum_difficulty(iteration)

            # 2. Generate puzzle
            puzzle = self.generator.generate_puzzle(difficulty=difficulty)
            self.statistics['generated'] += 1

            # 3. Validate
            is_valid, reason = validate_puzzle(puzzle)
            if not is_valid:
                continue
            self.statistics['valid'] += 1

            # 4. Solve
            solver_result = self.solver.solve(puzzle)
            if solver_result['solved']:
                self.statistics['solved'] += 1
                self.statistics['by_difficulty'][difficulty] += 1

            # 5. Evaluate quality
            evaluation = self.evaluator.evaluate(puzzle, solver_result)

            # 6. Add to database if high quality
            if evaluation['total_score'] > 0.6:
                puzzle['evaluation'] = evaluation
                puzzle['solver_result'] = solver_result
                self.puzzle_database.append(puzzle)

            # 7. Update generator based on feedback
            self._update_generator(puzzle, evaluation, solver_result)

            # 8. Periodic reporting
            if iteration % 100 == 0:
                self._report_progress(iteration)

    def _curriculum_difficulty(self, iteration):
        """Gradually increase difficulty."""

        progress = iteration / 10000  # 0 to 1

        if progress < 0.3:
            # Early: mostly easy
            return np.random.choice(['easy', 'medium', 'hard'], p=[0.7, 0.2, 0.1])
        elif progress < 0.7:
            # Middle: balanced
            return np.random.choice(['easy', 'medium', 'hard'], p=[0.2, 0.5, 0.3])
        else:
            # Late: mostly hard
            return np.random.choice(['easy', 'medium', 'hard'], p=[0.1, 0.3, 0.6])

    def _update_generator(self, puzzle, evaluation, solver_result):
        """Adjust generator based on feedback."""

        # If puzzle was too easy (solved quickly)
        if solver_result['solved'] and solver_result.get('time', 0) < 1.0:
            # Increase complexity
            self.generator.complexity_bias += 0.01

        # If puzzle was unsolvable (very low accuracy)
        if not solver_result['solved'] and solver_result['accuracy'] < 0.1:
            # Decrease complexity
            self.generator.complexity_bias -= 0.01

        # If low diversity
        if evaluation['scores']['diversity'] < 0.5:
            # Increase randomness in generation
            self.generator.randomness += 0.01

    def _report_progress(self, iteration):
        """Print training statistics."""

        print(f"\n=== Iteration {iteration} ===")
        print(f"Generated: {self.statistics['generated']}")
        print(f"Valid: {self.statistics['valid']} ({100*self.statistics['valid']/max(1, self.statistics['generated']):.1f}%)")
        print(f"Solved: {self.statistics['solved']} ({100*self.statistics['solved']/max(1, self.statistics['valid']):.1f}%)")
        print(f"\nBy difficulty:")
        for diff in ['easy', 'medium', 'hard']:
            count = self.statistics['by_difficulty'][diff]
            print(f"  {diff}: {count}")
        print(f"\nDatabase size: {len(self.puzzle_database)}")
```

### Adaptive Difficulty

```python
def adaptive_difficulty_sampling(self):
    """Dynamically adjust difficulty based on solver performance."""

    # Get recent solve rates by difficulty
    recent_window = self.puzzle_database[-100:]

    solve_rates = {}
    for difficulty in ['easy', 'medium', 'hard']:
        puzzles = [p for p in recent_window if p['difficulty'] == difficulty]
        if puzzles:
            solved = sum(1 for p in puzzles if p['solver_result']['solved'])
            solve_rates[difficulty] = solved / len(puzzles)
        else:
            solve_rates[difficulty] = 0.5  # Default

    # Target: 80% easy, 50% medium, 20% hard
    targets = {'easy': 0.8, 'medium': 0.5, 'hard': 0.2}

    # Adjust sampling probabilities
    probs = {}
    for diff in ['easy', 'medium', 'hard']:
        # If solving too many: increase that difficulty
        # If solving too few: decrease that difficulty
        delta = solve_rates[diff] - targets[diff]
        probs[diff] = 0.33 + delta * 0.5  # Adjust from baseline 0.33

    # Normalize
    total = sum(probs.values())
    probs = {k: v/total for k, v in probs.items()}

    # Sample
    return np.random.choice(['easy', 'medium', 'hard'], p=list(probs.values()))
```

---

## Use Cases for Generated Puzzles

### 1. Training Data for Neural Components

```python
# Generate 10K puzzles
trainer = SelfPlayTrainer()
trainer.train(num_iterations=10000)

# Filter high-quality puzzles
training_data = [
    p for p in trainer.puzzle_database
    if p['evaluation']['total_score'] > 0.7
]

print(f"Created {len(training_data)} high-quality training puzzles")

# Train neural program synthesis model
neural_model = NeuralProgramSynthesis()
neural_model.train(training_data)
```

### 2. Weakness Identification

```python
# Analyze unsolved puzzles
unsolved = [
    p for p in trainer.puzzle_database
    if not p['solver_result']['solved']
]

# Cluster by transformation type
from collections import defaultdict
by_transformation = defaultdict(list)

for puzzle in unsolved:
    trans_type = puzzle['transformation'][0][0]  # First operation
    by_transformation[trans_type].append(puzzle)

# Report weaknesses
print("Solver Weaknesses:")
for trans_type, puzzles in sorted(by_transformation.items(), key=lambda x: -len(x[1])):
    print(f"  {trans_type}: {len(puzzles)} unsolved ({len(puzzles)/len(unsolved)*100:.1f}%)")
```

### 3. Curriculum for Incremental Learning

```python
# Sort puzzles by difficulty
sorted_puzzles = sorted(
    trainer.puzzle_database,
    key=lambda p: p['evaluation']['scores']['difficulty']
)

# Train in curriculum: easy → hard
for phase, puzzles in enumerate(np.array_split(sorted_puzzles, 5)):
    print(f"\n=== Phase {phase+1}: Training on {len(puzzles)} puzzles ===")

    # Fine-tune model on this difficulty level
    model.train(puzzles)

    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_set)
    print(f"Test accuracy: {test_accuracy:.1%}")
```

### 4. Data Augmentation

```python
# For each real ARC task, generate variants
augmented_dataset = []

for real_task in arc_training_tasks:
    augmented_dataset.append(real_task)

    # Generate 10 mutations
    mutator = MutationGenerator()
    for _ in range(10):
        variant = mutator.mutate_task(real_task)
        augmented_dataset.append(variant)

print(f"Augmented dataset: {len(arc_training_tasks)} → {len(augmented_dataset)} tasks")
```

---

## Advanced: Adversarial Co-Evolution

### Generator-Solver Arms Race

```python
class AdversarialTrainer:
    """Train generator and solver adversarially."""

    def __init__(self):
        self.generator = NeuralPuzzleGenerator()
        self.solver = NeuralSolver()  # Assume we have neural solver

    def train_epoch(self):
        """One epoch of adversarial training."""

        # Phase 1: Train Generator to challenge Solver
        for _ in range(100):
            # Generate puzzle
            puzzle = self.generator.generate()

            # Try to solve
            solver_result = self.solver.solve(puzzle, return_confidence=True)

            # Generator reward: Challenge but not impossible
            if solver_result['solved']:
                reward = -solver_result['confidence']  # Prefer low confidence
            else:
                reward = solver_result['accuracy'] - 0.5  # Prefer near-miss

            # Update generator
            self.generator.update(reward)

        # Phase 2: Train Solver to solve Generator's puzzles
        for _ in range(100):
            # Generate challenging puzzle
            puzzle = self.generator.generate(difficulty='hard')

            # Train solver to solve it
            self.solver.train_on_puzzle(puzzle)

        # Phase 3: Evaluate progress
        test_puzzles = self.generator.generate_batch(100)
        solve_rate = sum(self.solver.solve(p)['solved'] for p in test_puzzles) / 100

        print(f"Solve rate: {solve_rate:.1%}")

        return solve_rate
```

**Dynamics:**
- Generator creates increasingly challenging puzzles
- Solver improves to handle them
- System evolves together
- Both improve via competition

---

## Implementation Roadmap

### Phase 1: Primitive Composition (Month 1)

**Goal:** Generate 1K valid puzzles using existing primitives

**Tasks:**
1. Implement `PrimitiveCompositionGenerator`
2. Implement validation logic
3. Implement basic evaluator
4. Run self-play loop
5. Collect statistics on generated puzzles

**Success criteria:**
- 80%+ puzzles are valid
- 40-60% of valid puzzles are solvable
- Diversity: 10+ different transformation patterns

### Phase 2: Quality Control (Month 2)

**Goal:** Improve puzzle quality and diversity

**Tasks:**
1. Implement comprehensive evaluator
2. Add diversity metrics
3. Implement curriculum learning
4. Filter high-quality puzzles

**Success criteria:**
- 50%+ puzzles scored >0.7 by evaluator
- Cover 20+ primitive types
- Solve rate matches target curve

### Phase 3: Data Generation (Month 3)

**Goal:** Generate 10K high-quality puzzles

**Tasks:**
1. Scale up generation (parallelization)
2. Balance difficulty distribution
3. Ensure transformation diversity
4. Export for training

**Success criteria:**
- 10K+ puzzles in database
- Balanced difficulty (30% easy, 40% medium, 30% hard)
- Cover all 36 primitive types

### Phase 4: Neural Training (Month 4)

**Goal:** Use generated data to train neural components

**Tasks:**
1. Train task classifier on generated puzzles
2. Train primitive relevance predictor
3. Train composition detector
4. Evaluate on real ARC tasks

**Success criteria:**
- Task classifier: 80%+ accuracy
- Primitive predictor: 70%+ recall on relevant ops
- System solve rate: 28%+ (up from 22%)

---

## Evaluation Metrics

### Generator Quality

```python
def evaluate_generator(puzzle_database):
    """Comprehensive generator evaluation."""

    metrics = {}

    # 1. Validity rate
    valid = sum(1 for p in puzzle_database if p['evaluation']['valid'])
    metrics['validity_rate'] = valid / len(puzzle_database)

    # 2. Difficulty distribution
    by_difficulty = defaultdict(int)
    for p in puzzle_database:
        by_difficulty[p['difficulty']] += 1
    metrics['difficulty_distribution'] = dict(by_difficulty)

    # 3. Diversity (unique transformations)
    transformations = set()
    for p in puzzle_database:
        trans_str = str(p['transformation'])
        transformations.add(trans_str)
    metrics['unique_transformations'] = len(transformations)

    # 4. Coverage (which primitives used)
    primitive_usage = defaultdict(int)
    for p in puzzle_database:
        for op, _ in p['transformation']:
            primitive_usage[op] += 1
    metrics['primitive_coverage'] = len(primitive_usage)
    metrics['primitive_distribution'] = dict(primitive_usage)

    # 5. Solve rate
    solved = sum(1 for p in puzzle_database if p['solver_result']['solved'])
    metrics['solve_rate'] = solved / len(puzzle_database)

    return metrics
```

### System Improvement

```python
def evaluate_self_play_benefit(baseline_solver, improved_solver, test_set):
    """Compare solver before/after self-play training."""

    baseline_results = [baseline_solver.solve(task) for task in test_set]
    improved_results = [improved_solver.solve(task) for task in test_set]

    baseline_solve_rate = sum(r['solved'] for r in baseline_results) / len(test_set)
    improved_solve_rate = sum(r['solved'] for r in improved_results) / len(test_set)

    improvement = improved_solve_rate - baseline_solve_rate

    print(f"Baseline: {baseline_solve_rate:.1%}")
    print(f"Improved: {improved_solve_rate:.1%}")
    print(f"Gain: +{improvement:.1%}")

    return {
        'baseline': baseline_solve_rate,
        'improved': improved_solve_rate,
        'absolute_gain': improvement,
        'relative_gain': improvement / baseline_solve_rate if baseline_solve_rate > 0 else 0
    }
```

---

## Challenges and Solutions

### Challenge 1: Validity

**Problem:** Generated puzzles may be invalid (inconsistent transformations)

**Solution:**
- Strict validation before adding to database
- Retry generation if invalid
- Track validity rate, adjust generator if too low

### Challenge 2: Solvability

**Problem:** Puzzles may be too hard or too easy

**Solution:**
- Adaptive difficulty sampling based on solve rates
- Curriculum learning (easy → hard)
- Evaluator scores based on "sweet spot" difficulty

### Challenge 3: Diversity

**Problem:** Generator may produce similar puzzles

**Solution:**
- Diversity bonus in evaluator
- Track feature distributions
- Reject puzzles too similar to existing ones
- Mutation-based generation for variations

### Challenge 4: Efficiency

**Problem:** Generating and solving puzzles is slow

**Solution:**
- Parallelize generation across multiple processes
- Cache validation results
- Use faster solver for initial screening
- Batch puzzle generation

### Challenge 5: Quality vs Quantity

**Problem:** Need both high-quality and large quantity

**Solution:**
- Generate many (10K+), filter aggressively (keep 50%)
- Multiple quality tiers (use all for different purposes)
- Continuous generation as background process

---

## Expected Outcomes

### Short-Term (Month 1-2)

- Generate 1K-5K synthetic puzzles
- Identify system weaknesses systematically
- Validate self-play approach works

**Estimated Impact:** +0-2% (proof of concept)

### Medium-Term (Month 3-4)

- Generate 10K+ high-quality puzzles
- Train neural components (classifier, predictor)
- Deploy hybrid system

**Estimated Impact:** +6-10% (28-32% solve rate)

### Long-Term (Month 6-12)

- Adversarial co-evolution with neural solver
- Continuous puzzle generation pipeline
- Transfer learning from synthetic to real tasks

**Estimated Impact:** +15-25% (37-47% solve rate)

---

## Conclusion

**Self-play addresses the fundamental data bottleneck:**
- Only 400 public ARC tasks (insufficient for neural training)
- Can generate unlimited synthetic tasks
- Targets system weaknesses automatically
- Enables curriculum learning

**Key advantages:**
1. **Unlimited data** - Generate 10K+ tasks
2. **Targeted** - Focus on weak areas
3. **Curriculum** - Easy → hard progression
4. **Adversarial** - Push system to improve
5. **No manual annotation** - Fully automated

**Recommended implementation:**
1. Start with Primitive Composition (simple, fast)
2. Validate with 1K puzzles (prove concept)
3. Scale to 10K+ (provide training data)
4. Use for training hybrid system (immediate benefit)
5. Explore adversarial co-evolution (long-term)

**This could be the breakthrough** that overcomes the 22% ceiling by providing the training data needed for neural components!
