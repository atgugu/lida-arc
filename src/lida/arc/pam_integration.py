"""
PAM integration for ARC transformation patterns.

This module connects the DemonstrationAnalyzer to LIDA's Perceptual Associative
Memory, enabling spreading activation and concept emergence for transformations.
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict

from ..memory.pam import PerceptualAssociativeMemory, PAMNode
from .primitives import PrimitiveLibrary, Primitive
from .demonstration import TransformationPattern


class ARCPAMSeeder:
    """Seeds PAM with ARC-specific concepts: primitives, features, and patterns.

    This creates a semantic network where:
    - Primitives are connected to their features
    - Patterns are connected to primitives they use
    - Features cluster related transformations
    - Spreading activation reveals related operations
    """

    def __init__(self, pam: PerceptualAssociativeMemory, primitive_library: PrimitiveLibrary):
        """
        Args:
            pam: The PAM network to seed
            primitive_library: Library of cognitive primitives
        """
        self.pam = pam
        self.primitives = primitive_library

    def seed_primitives(self):
        """Add all primitive operations to PAM as concept nodes."""
        for name, primitive in self.primitives.primitives.items():
            # Add primitive node
            prim_node = PAMNode(
                node_id=f'prim_{name}',
                kind='concept',
                label=name,
                base_strength=0.5
            )
            self.pam.ensure_node(prim_node)

            # Add feature nodes and link to primitive
            features = primitive.get_features()
            for feature_name, feature_value in features.items():
                # Ensure feature node exists
                feat_node = PAMNode(
                    node_id=f'feat_{feature_name}',
                    kind='feature',
                    label=feature_name,
                    base_strength=0.3
                )
                self.pam.ensure_node(feat_node)

                # Link primitive to feature (bidirectional)
                self.pam.add_link(f'prim_{name}', f'feat_{feature_name}',
                                 'has_feature', weight=feature_value)
                self.pam.add_link(f'feat_{feature_name}', f'prim_{name}',
                                 'is_feature_of', weight=feature_value * 0.8)

    def seed_pattern(self, pattern: TransformationPattern):
        """Add a learned transformation pattern to PAM.

        Args:
            pattern: TransformationPattern to add
        """
        pattern_node = PAMNode(
            node_id=f'pattern_{pattern.pattern_id}',
            kind='concept',
            label=pattern.pattern_id,
            base_strength=pattern.confidence
        )
        self.pam.ensure_node(pattern_node)

        # Link pattern to its features
        pam_features = pattern.to_pam_features()
        for feature_name, feature_value in pam_features.items():
            # Ensure feature exists
            feat_node = PAMNode(
                node_id=f'feat_{feature_name}',
                kind='feature',
                label=feature_name,
                base_strength=0.3
            )
            self.pam.ensure_node(feat_node)

            # Bidirectional link
            self.pam.add_link(f'pattern_{pattern.pattern_id}', f'feat_{feature_name}',
                             'exhibits', weight=feature_value)
            self.pam.add_link(f'feat_{feature_name}', f'pattern_{pattern.pattern_id}',
                             'exhibited_by', weight=feature_value * 0.7)

        # Link pattern to primitives it uses
        for op_name in pattern.grid_operations:
            prim_node_id = f'prim_{op_name}'
            if prim_node_id in self.pam.g:
                # Pattern uses primitive
                self.pam.add_link(f'pattern_{pattern.pattern_id}', prim_node_id,
                                 'uses', weight=0.9)
                # Primitive is used by pattern
                self.pam.add_link(prim_node_id, f'pattern_{pattern.pattern_id}',
                                 'used_in', weight=0.7)

    def activate_from_pattern(self, pattern: TransformationPattern, strength: float = 1.0):
        """Activate PAM nodes based on a pattern.

        This seeds spreading activation from:
        - The pattern itself
        - Features it exhibits
        - Primitives it uses

        Args:
            pattern: Pattern to activate from
            strength: Base activation strength
        """
        seeds = {}

        # Activate pattern node
        pattern_id = f'pattern_{pattern.pattern_id}'
        if pattern_id in self.pam.g:
            seeds[pattern_id] = strength * pattern.confidence

        # Activate feature nodes
        for feature_name, feature_value in pattern.to_pam_features().items():
            feat_id = f'feat_{feature_name}'
            if feat_id in self.pam.g:
                seeds[feat_id] = strength * feature_value * 0.8

        # Activate primitive nodes
        for op_name in pattern.grid_operations:
            prim_id = f'prim_{op_name}'
            if prim_id in self.pam.g:
                seeds[prim_id] = strength * 0.9

        return seeds


class ARCCategoryInduction:
    """Induces new composite operation categories from co-occurring patterns.

    When multiple patterns consistently use the same sequence of primitives,
    create a new abstract category (composite operation).
    """

    def __init__(self, pam: PerceptualAssociativeMemory):
        """
        Args:
            pam: PAM network to add categories to
        """
        self.pam = pam
        self.operation_sequences: Dict[str, int] = defaultdict(int)
        self.sequence_instances: Dict[str, List[str]] = defaultdict(list)

    def observe_pattern(self, pattern: TransformationPattern):
        """Track pattern for category induction.

        Args:
            pattern: Pattern that was successfully applied
        """
        if len(pattern.grid_operations) >= 2:
            # Create signature from operation sequence
            seq_signature = '_'.join(sorted(pattern.grid_operations))
            self.operation_sequences[seq_signature] += 1
            self.sequence_instances[seq_signature].append(pattern.pattern_id)

    def induce_categories(self, min_occurrences: int = 3) -> List[str]:
        """Create category nodes for frequently co-occurring operation sequences.

        Args:
            min_occurrences: Minimum times a sequence must occur to become a category

        Returns:
            List of newly created category node IDs
        """
        new_categories = []

        for seq_signature, count in self.operation_sequences.items():
            if count >= min_occurrences:
                # Create category node
                category_id = f'cat_{seq_signature}'

                if category_id not in self.pam.g:
                    cat_node = PAMNode(
                        node_id=category_id,
                        kind='category',
                        label=f"composite_{seq_signature}",
                        base_strength=min(1.0, count / (min_occurrences * 2))
                    )
                    self.pam.ensure_node(cat_node)

                    # Link category to component operations
                    operations = seq_signature.split('_')
                    for op_name in operations:
                        prim_id = f'prim_{op_name}'
                        if prim_id in self.pam.g:
                            # Category includes primitive
                            self.pam.add_link(category_id, prim_id, 'includes', weight=0.9)
                            # Primitive is part of category
                            self.pam.add_link(prim_id, category_id, 'part_of', weight=0.7)

                    # Link category to pattern instances
                    for pattern_id in self.sequence_instances[seq_signature]:
                        full_pattern_id = f'pattern_{pattern_id}'
                        if full_pattern_id in self.pam.g:
                            self.pam.add_link(category_id, full_pattern_id, 'generalizes', weight=0.8)
                            self.pam.add_link(full_pattern_id, category_id, 'instance_of', weight=0.8)

                    new_categories.append(category_id)

        return new_categories


class ARCHebbianLearning:
    """Hebbian-style strengthening of links between co-activated concepts.

    When operations co-occur in successful patterns, strengthen their connections.
    """

    def __init__(self, pam: PerceptualAssociativeMemory):
        """
        Args:
            pam: PAM network to update
        """
        self.pam = pam

    def strengthen_from_pattern(self, pattern: TransformationPattern, reward: float = 1.0):
        """Strengthen links between operations in a successful pattern.

        Implements Hebbian learning: "neurons that fire together, wire together"

        Args:
            pattern: Successful transformation pattern
            reward: Reward signal (1.0 = success, 0.0 = failure)
        """
        if reward <= 0.0 or len(pattern.grid_operations) < 2:
            return

        operations = pattern.grid_operations

        # Strengthen links between consecutive operations
        for i in range(len(operations) - 1):
            op1_id = f'prim_{operations[i]}'
            op2_id = f'prim_{operations[i+1]}'

            if op1_id in self.pam.g and op2_id in self.pam.g:
                # Get current weight
                if self.pam.g.has_edge(op1_id, op2_id):
                    current_weight = self.pam.g.edges[op1_id, op2_id].get('weight', 0.5)
                    # Strengthen (with cap)
                    new_weight = min(1.0, current_weight + 0.1 * reward)
                    self.pam.g.edges[op1_id, op2_id]['weight'] = new_weight
                else:
                    # Create new link
                    self.pam.add_link(op1_id, op2_id, 'precedes', weight=0.5 * reward)

                # Bidirectional strengthening (weaker)
                if self.pam.g.has_edge(op2_id, op1_id):
                    current_weight = self.pam.g.edges[op2_id, op1_id].get('weight', 0.3)
                    new_weight = min(1.0, current_weight + 0.05 * reward)
                    self.pam.g.edges[op2_id, op1_id]['weight'] = new_weight
                else:
                    self.pam.add_link(op2_id, op1_id, 'follows', weight=0.3 * reward)

        # Strengthen links between all pairs (not just consecutive)
        for i, op1 in enumerate(operations):
            for op2 in operations[i+1:]:
                op1_id = f'prim_{op1}'
                op2_id = f'prim_{op2}'

                if op1_id in self.pam.g and op2_id in self.pam.g:
                    # Weaker association for non-consecutive
                    if self.pam.g.has_edge(op1_id, op2_id):
                        current_weight = self.pam.g.edges[op1_id, op2_id].get('weight', 0.3)
                        new_weight = min(1.0, current_weight + 0.05 * reward)
                        self.pam.g.edges[op1_id, op2_id]['weight'] = new_weight
                    else:
                        self.pam.add_link(op1_id, op2_id, 'co_occurs_with', weight=0.3 * reward)


class ARCPAMIntegration:
    """High-level integration of PAM with ARC pattern learning.

    Combines seeding, activation, category induction, and Hebbian learning.
    """

    def __init__(self, pam: PerceptualAssociativeMemory, primitive_library: PrimitiveLibrary):
        """
        Args:
            pam: PAM network
            primitive_library: Cognitive primitives
        """
        self.pam = pam
        self.primitives = primitive_library

        self.seeder = ARCPAMSeeder(pam, primitive_library)
        self.category_inducer = ARCCategoryInduction(pam)
        self.hebbian_learner = ARCHebbianLearning(pam)

        # Initialize with primitives
        self.seeder.seed_primitives()

    def learn_from_pattern(self, pattern: TransformationPattern, success: bool = True):
        """Learn from a transformation pattern.

        Args:
            pattern: Pattern to learn from
            success: Whether pattern was successful
        """
        # Seed pattern in PAM
        self.seeder.seed_pattern(pattern)

        # Track for category induction
        if success:
            self.category_inducer.observe_pattern(pattern)

        # Hebbian strengthening
        reward = 1.0 if success else 0.0
        self.hebbian_learner.strengthen_from_pattern(pattern, reward)

    def activate_and_spread(self, pattern: TransformationPattern,
                           iterations: int = 5,
                           decay: float = 0.1) -> Dict[str, float]:
        """Activate PAM from pattern and spread activation.

        Args:
            pattern: Pattern to activate from
            iterations: Spreading activation iterations
            decay: Activation decay per iteration

        Returns:
            Final activation values for all nodes
        """
        # Get seed activations
        seeds = self.seeder.activate_from_pattern(pattern)

        # Spread activation
        self.pam.spread_activation(
            seeds=seeds,
            iterations=iterations,
            decay=decay,
            normalize_after=True,
            max_activation=1.0
        )

        # Return all activations
        activations = {}
        for node_id in self.pam.g.nodes():
            activations[node_id] = self.pam.get_activation(node_id)

        return activations

    def get_top_active_operations(self, k: int = 5) -> List[str]:
        """Get top-k active primitive operations.

        Args:
            k: Number of operations to return

        Returns:
            List of primitive names, sorted by activation
        """
        operation_activations = []

        for node_id in self.pam.g.nodes():
            if node_id.startswith('prim_'):
                activation = self.pam.get_activation(node_id)
                op_name = node_id[5:]  # Remove 'prim_' prefix
                operation_activations.append((op_name, activation))

        # Sort by activation
        operation_activations.sort(key=lambda x: x[1], reverse=True)

        return [op for op, _ in operation_activations[:k]]

    def induce_new_categories(self, min_occurrences: int = 3) -> List[str]:
        """Induce new category nodes from observed patterns.

        Args:
            min_occurrences: Minimum pattern occurrences for category creation

        Returns:
            List of newly created category node IDs
        """
        return self.category_inducer.induce_categories(min_occurrences)

    def reset_activations(self):
        """Reset all activations to zero."""
        for node_id in self.pam.g.nodes():
            self.pam.set_activation(node_id, 0.0)
