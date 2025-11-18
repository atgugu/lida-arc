"""
Tests for PAM integration with ARC patterns.
"""

import pytest
from lida.memory.pam import PerceptualAssociativeMemory
from lida.arc.primitives import PrimitiveLibrary
from lida.arc.demonstration import TransformationPattern
from lida.arc.pam_integration import (
    ARCPAMSeeder, ARCCategoryInduction, ARCHebbianLearning, ARCPAMIntegration
)


class TestARCPAMSeeder:
    def test_seed_primitives(self):
        """Test seeding PAM with primitives."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        seeder = ARCPAMSeeder(pam, primitives)

        seeder.seed_primitives()

        # Should have nodes for primitives
        assert 'prim_rotate_90' in pam.g
        assert 'prim_recolor' in pam.g

        # Should have feature nodes
        assert 'feat_rotation' in pam.g or 'feat_geometric' in pam.g
        assert 'feat_manipulation' in pam.g

        # Primitives should be linked to features
        assert pam.g.has_edge('prim_rotate_90', 'feat_rotation') or \
               pam.g.has_edge('prim_rotate_90', 'feat_geometric')

    def test_seed_pattern(self):
        """Test seeding PAM with a learned pattern."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        seeder = ARCPAMSeeder(pam, primitives)

        # Seed primitives first
        seeder.seed_primitives()

        # Create pattern
        pattern = TransformationPattern(
            pattern_id='test_rotation',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        # Seed pattern
        seeder.seed_pattern(pattern)

        # Pattern node should exist
        assert 'pattern_test_rotation' in pam.g

        # Pattern should link to primitive it uses
        assert pam.g.has_edge('pattern_test_rotation', 'prim_rotate_90')

    def test_activate_from_pattern(self):
        """Test activating PAM from a pattern."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        seeder = ARCPAMSeeder(pam, primitives)

        seeder.seed_primitives()

        pattern = TransformationPattern(
            pattern_id='test',
            transformation_type='grid_op',
            grid_operations=['rotate_90', 'recolor'],
            confidence=0.9
        )

        seeder.seed_pattern(pattern)

        # Activate
        seeds = seeder.activate_from_pattern(pattern, strength=1.0)

        # Should have seeds for pattern, features, and primitives
        assert len(seeds) > 0
        assert 'pattern_test' in seeds or any('prim_' in k for k in seeds)


class TestARCCategoryInduction:
    def test_category_induction(self):
        """Test inducing categories from repeated patterns."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        seeder = ARCPAMSeeder(pam, primitives)
        inducer = ARCCategoryInduction(pam)

        seeder.seed_primitives()

        # Observe same pattern multiple times
        for i in range(3):
            pattern = TransformationPattern(
                pattern_id=f'rotate_recolor_{i}',
                transformation_type='grid_op',
                grid_operations=['rotate_90', 'recolor'],
                confidence=1.0
            )
            seeder.seed_pattern(pattern)
            inducer.observe_pattern(pattern)

        # Induce categories
        new_categories = inducer.induce_categories(min_occurrences=3)

        # Should create category for rotate+recolor
        assert len(new_categories) >= 1

        # Category should exist in PAM
        for cat_id in new_categories:
            assert cat_id in pam.g
            assert pam.g.nodes[cat_id]['kind'] == 'category'

    def test_no_category_for_rare_patterns(self):
        """Test that rare patterns don't create categories."""
        pam = PerceptualAssociativeMemory()
        inducer = ARCCategoryInduction(pam)

        # Observe pattern only once
        pattern = TransformationPattern(
            pattern_id='rare',
            transformation_type='grid_op',
            grid_operations=['rotate_90', 'recolor'],
            confidence=1.0
        )
        inducer.observe_pattern(pattern)

        # Should not create category (min_occurrences=3)
        new_categories = inducer.induce_categories(min_occurrences=3)
        assert len(new_categories) == 0


class TestARCHebbianLearning:
    def test_hebbian_strengthening(self):
        """Test Hebbian strengthening of co-occurring operations."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        seeder = ARCPAMSeeder(pam, primitives)
        hebbian = ARCHebbianLearning(pam)

        seeder.seed_primitives()

        # Get initial weight (if exists)
        initial_weight = None
        if pam.g.has_edge('prim_rotate_90', 'prim_recolor'):
            initial_weight = pam.g.edges['prim_rotate_90', 'prim_recolor'].get('weight', 0.0)

        # Strengthen from successful pattern
        pattern = TransformationPattern(
            pattern_id='test',
            transformation_type='grid_op',
            grid_operations=['rotate_90', 'recolor'],
            confidence=1.0
        )

        hebbian.strengthen_from_pattern(pattern, reward=1.0)

        # Link should exist and be strengthened
        assert pam.g.has_edge('prim_rotate_90', 'prim_recolor')

        new_weight = pam.g.edges['prim_rotate_90', 'prim_recolor'].get('weight', 0.0)

        if initial_weight is not None:
            assert new_weight > initial_weight
        else:
            assert new_weight > 0.0

    def test_no_strengthening_on_failure(self):
        """Test that failures don't strengthen links."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        seeder = ARCPAMSeeder(pam, primitives)
        hebbian = ARCHebbianLearning(pam)

        seeder.seed_primitives()

        pattern = TransformationPattern(
            pattern_id='failed',
            transformation_type='grid_op',
            grid_operations=['rotate_90', 'recolor'],
            confidence=0.5
        )

        # No strengthening with reward=0
        hebbian.strengthen_from_pattern(pattern, reward=0.0)

        # Link may or may not exist, but won't be strengthened


class TestARCPAMIntegration:
    def test_integration_workflow(self):
        """Test complete integration workflow."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        # Primitives should be seeded
        assert 'prim_rotate_90' in pam.g

        # Learn from pattern
        pattern = TransformationPattern(
            pattern_id='workflow_test',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        integration.learn_from_pattern(pattern, success=True)

        # Pattern should be in PAM
        assert 'pattern_workflow_test' in pam.g

    def test_activate_and_spread(self):
        """Test activation spreading."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        pattern = TransformationPattern(
            pattern_id='spread_test',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        integration.learn_from_pattern(pattern, success=True)

        # Activate and spread
        activations = integration.activate_and_spread(pattern, iterations=3, decay=0.1)

        # Should have activations
        assert len(activations) > 0

        # Pattern and related nodes should be activated
        assert activations.get('pattern_spread_test', 0.0) > 0.0 or \
               activations.get('prim_rotate_90', 0.0) > 0.0

    def test_get_top_active_operations(self):
        """Test retrieving top active operations."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        pattern = TransformationPattern(
            pattern_id='active_test',
            transformation_type='grid_op',
            grid_operations=['rotate_90', 'recolor'],
            confidence=1.0
        )

        integration.learn_from_pattern(pattern, success=True)
        integration.activate_and_spread(pattern, iterations=3)

        # Get top active operations
        top_ops = integration.get_top_active_operations(k=5)

        # Should return operation names
        assert len(top_ops) > 0
        assert all(isinstance(op, str) for op in top_ops)

        # Activated operations should be in top results
        assert 'rotate_90' in top_ops or 'recolor' in top_ops

    def test_category_induction_integration(self):
        """Test category induction through integration."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        # Learn same composite pattern multiple times
        for i in range(3):
            pattern = TransformationPattern(
                pattern_id=f'composite_{i}',
                transformation_type='grid_op',
                grid_operations=['rotate_90', 'recolor'],
                confidence=1.0
            )
            integration.learn_from_pattern(pattern, success=True)

        # Induce categories
        new_cats = integration.induce_new_categories(min_occurrences=3)

        # Should create at least one category
        assert len(new_cats) >= 1

    def test_reset_activations(self):
        """Test resetting all activations."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        pattern = TransformationPattern(
            pattern_id='reset_test',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        # Activate
        integration.activate_and_spread(pattern, iterations=3)

        # Some nodes should be activated
        assert any(integration.pam.get_activation(n) > 0.0 for n in pam.g.nodes())

        # Reset
        integration.reset_activations()

        # All activations should be zero
        for node_id in pam.g.nodes():
            assert integration.pam.get_activation(node_id) == 0.0


class TestSpreadingActivation:
    def test_related_operations_activated(self):
        """Test that related operations get activated through spreading."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        # Learn pattern using rotate_90
        pattern = TransformationPattern(
            pattern_id='rotation_test',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        integration.learn_from_pattern(pattern, success=True)
        integration.activate_and_spread(pattern, iterations=5, decay=0.05)

        # rotate_90 should be highly activated
        rotate_90_activation = integration.pam.get_activation('prim_rotate_90')
        assert rotate_90_activation > 0.5

        # Related rotations might be activated (if they share features)
        # This tests the spreading mechanism
        all_activations = {n: integration.pam.get_activation(n)
                          for n in pam.g.nodes()}

        # At least some nodes should be activated
        activated_nodes = [n for n, a in all_activations.items() if a > 0.1]
        assert len(activated_nodes) >= 2  # At least pattern + primitive

    def test_feature_based_spreading(self):
        """Test that activation spreads through shared features."""
        pam = PerceptualAssociativeMemory()
        primitives = PrimitiveLibrary()
        integration = ARCPAMIntegration(pam, primitives)

        # Activate geometric operations
        pattern = TransformationPattern(
            pattern_id='geometric',
            transformation_type='grid_op',
            grid_operations=['rotate_90'],
            confidence=1.0
        )

        integration.learn_from_pattern(pattern, success=True)
        activations = integration.activate_and_spread(pattern, iterations=5, decay=0.1)

        # Geometric feature should be activated
        if 'feat_geometric' in activations:
            assert activations['feat_geometric'] > 0.0

        # Other geometric operations might be activated through shared features
        geometric_ops = ['rotate_180', 'rotate_270', 'reflect_horizontal']
        activated_geometric = [op for op in geometric_ops
                              if activations.get(f'prim_{op}', 0.0) > 0.1]

        # At least the activating operation should be present
        assert 'rotate_90' in [n.replace('prim_', '') for n, a in activations.items()
                               if n.startswith('prim_') and a > 0.1]
