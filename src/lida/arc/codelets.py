"""
ARC-specific codelets for pattern hypothesis generation, validation, and application.

These codelets integrate with LIDA's cognitive cycle to create a full learning loop:
- Understanding phase: Generate pattern hypotheses from demonstrations via PAM
- Attention phase: Compete hypotheses for workspace access
- Action phase: Apply winning pattern to test input
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from ..core.codelet import Codelet
from ..core.events import Coalition
from ..memory.workspace import SituationalModel
from .demonstration import TransformationPattern, DemonstrationAnalyzer
from .pam_integration import ARCPAMIntegration
from .primitives import PrimitiveLibrary
from .environment import GridPair


@dataclass
class PatternHypothesis:
    """A pattern hypothesis competing for workspace access."""

    hypothesis_id: str
    pattern: TransformationPattern
    salience: float  # Computed from PAM activation + confidence
    support_count: int  # Number of demos this pattern explains
    validation_accuracy: float = 0.0  # Accuracy on demonstrations

    def to_coalition(self) -> Coalition:
        """Convert hypothesis to coalition for workspace competition."""
        features = []

        # Pattern features
        features.append(('confidence', self.pattern.confidence))
        features.append(('support_count', float(self.support_count)))
        features.append(('validation_accuracy', self.validation_accuracy))
        features.append(('salience', self.salience))

        # Operation features
        for op in self.pattern.grid_operations:
            features.append((f'op_{op}', 1.0))

        # Transformation type
        features.append((f'type_{self.pattern.transformation_type}', 1.0))

        summary = f"{self.pattern.pattern_id}: {' -> '.join(self.pattern.grid_operations)}"

        return Coalition(
            id=self.hypothesis_id,
            salience=self.salience,
            features=frozenset(features),
            summary=summary
        )


class ARCCodeletFactory:
    """Factory for creating ARC-specific codelets."""

    def __init__(
        self,
        workspace: SituationalModel,
        pam_integration: ARCPAMIntegration,
        primitive_library: PrimitiveLibrary,
        analyzer: DemonstrationAnalyzer
    ):
        """
        Args:
            workspace: Global workspace for storing ARC task state
            pam_integration: PAM integration for spreading activation
            primitive_library: Cognitive primitives
            analyzer: Demonstration analyzer
        """
        self.workspace = workspace
        self.pam = pam_integration
        self.primitives = primitive_library
        self.analyzer = analyzer

        # Shared state across codelets
        self.demonstrations: List[GridPair] = []
        self.test_input: Optional[List[List[int]]] = None
        self.hypotheses: List[PatternHypothesis] = []
        self.coalitions: List[Coalition] = []
        self.winning_pattern: Optional[TransformationPattern] = None
        self.final_output: Optional[List[List[int]]] = None

    def make_understanding_codelets(self) -> List[Codelet]:
        """Create codelets for understanding phase: analyze demos and generate hypotheses."""

        def _analyze_demonstrations():
            """Analyze demonstrations to extract patterns."""
            if not self.demonstrations:
                return

            # Clear previous hypotheses
            self.hypotheses.clear()

            # Extract patterns from demonstrations
            patterns = self.analyzer.analyze_multiple_pairs(self.demonstrations)

            # Learn patterns in PAM
            for pattern in patterns:
                self.pam.learn_from_pattern(pattern, success=True)

            # Store patterns in workspace
            for i, pattern in enumerate(patterns[:5]):  # Top 5 patterns
                self.workspace.upsert_object(
                    oid=f'pattern_{i}',
                    features={
                        'confidence': pattern.confidence,
                        'support': float(len(pattern.supporting_demos)),
                    },
                    properties={
                        'pattern_id': pattern.pattern_id,
                        'operations': ','.join(pattern.grid_operations),
                        'type': pattern.transformation_type
                    }
                )

        def _generate_pam_hypotheses():
            """Generate hypotheses via PAM spreading activation."""
            if not self.demonstrations:
                return

            # Get best pattern from analysis
            patterns = self.analyzer.analyze_multiple_pairs(self.demonstrations)
            if not patterns:
                return

            seed_pattern = patterns[0]

            # Activate PAM and spread
            activations = self.pam.activate_and_spread(
                pattern=seed_pattern,
                iterations=5,
                decay=0.1
            )

            # Get top active operations
            top_ops = self.pam.get_top_active_operations(k=10)

            # Generate hypotheses from activated operations
            for i, pattern in enumerate(patterns[:5]):
                # Compute salience from PAM activation and pattern confidence
                pam_boost = sum(
                    activations.get(f'prim_{op}', 0.0)
                    for op in pattern.grid_operations
                ) / max(len(pattern.grid_operations), 1)

                salience = 0.7 * pattern.confidence + 0.3 * pam_boost

                hypothesis = PatternHypothesis(
                    hypothesis_id=f'hyp_{i}',
                    pattern=pattern,
                    salience=salience,
                    support_count=len(pattern.supporting_demos)
                )

                self.hypotheses.append(hypothesis)

            # Store top operations in workspace
            for op in top_ops[:5]:
                self.workspace.upsert_object(
                    oid=f'op_{op}',
                    features={'activation': activations.get(f'prim_{op}', 0.0)},
                    properties={'operation': op}
                )

        def _induce_categories():
            """Induce new composite operation categories from patterns."""
            new_categories = self.pam.induce_new_categories(min_occurrences=2)

            # Store induced categories in workspace
            for cat_id in new_categories:
                self.workspace.upsert_object(
                    oid=cat_id,
                    features={'category': 1.0},
                    properties={'type': 'composite'}
                )

        return [
            Codelet(
                name='arc_analyze_demos',
                urgency=1.0,
                action=_analyze_demonstrations,
                kind='understanding',
                metadata={'phase': 'analysis'}
            ),
            Codelet(
                name='arc_pam_hypotheses',
                urgency=0.9,
                action=_generate_pam_hypotheses,
                kind='understanding',
                metadata={'phase': 'hypothesis_generation'}
            ),
            Codelet(
                name='arc_category_induction',
                urgency=0.7,
                action=_induce_categories,
                kind='understanding',
                metadata={'phase': 'learning'}
            ),
        ]

    def make_attention_codelets(self) -> List[Codelet]:
        """Create codelets for attention phase: validate and compete hypotheses."""

        def _validate_hypotheses():
            """Validate hypotheses on demonstrations."""
            for hyp in self.hypotheses:
                # Test pattern on demonstrations
                correct = 0
                total = len(self.demonstrations)

                for demo in self.demonstrations:
                    # Apply pattern
                    result = demo.input
                    try:
                        for op_name in hyp.pattern.grid_operations:
                            prim = self.primitives.get(op_name)
                            if op_name == 'recolor' and hyp.pattern.color_mapping:
                                result = prim.execute(result, hyp.pattern.color_mapping)
                            else:
                                result = prim.execute(result)

                        if result == demo.output:
                            correct += 1
                    except Exception:
                        # Pattern failed to apply
                        pass

                hyp.validation_accuracy = correct / total if total > 0 else 0.0

                # Update salience based on validation
                hyp.salience = 0.5 * hyp.salience + 0.5 * hyp.validation_accuracy

        def _create_coalitions():
            """Create coalitions from validated hypotheses."""
            self.coalitions.clear()

            for hyp in self.hypotheses:
                coalition = hyp.to_coalition()
                self.coalitions.append(coalition)

        return [
            Codelet(
                name='arc_validate_hypotheses',
                urgency=1.0,
                action=_validate_hypotheses,
                kind='attention',
                metadata={'coalitions': self.coalitions}
            ),
            Codelet(
                name='arc_create_coalitions',
                urgency=0.9,
                action=_create_coalitions,
                kind='attention',
                metadata={'coalitions': self.coalitions}
            ),
        ]

    def make_action_codelets(self, winning_coalition_id: Optional[str] = None) -> List[Codelet]:
        """Create codelets for action phase: apply winning pattern."""

        def _select_winning_pattern():
            """Select winning pattern from coalition."""
            if not winning_coalition_id:
                return

            # Find hypothesis matching winning coalition
            for hyp in self.hypotheses:
                if hyp.hypothesis_id == winning_coalition_id:
                    self.winning_pattern = hyp.pattern

                    # Store in workspace
                    self.workspace.upsert_object(
                        oid='winning_pattern',
                        features={
                            'confidence': hyp.pattern.confidence,
                            'validation': hyp.validation_accuracy,
                            'salience': hyp.salience
                        },
                        properties={
                            'pattern_id': hyp.pattern.pattern_id,
                            'operations': ','.join(hyp.pattern.grid_operations)
                        }
                    )
                    break

        def _apply_pattern_to_test():
            """Apply winning pattern to test input."""
            if not self.winning_pattern or not self.test_input:
                return

            result = self.test_input

            try:
                # Apply pattern operations
                for op_name in self.winning_pattern.grid_operations:
                    prim = self.primitives.get(op_name)
                    if op_name == 'recolor' and self.winning_pattern.color_mapping:
                        result = prim.execute(result, self.winning_pattern.color_mapping)
                    else:
                        result = prim.execute(result)

                self.final_output = result

                # Store result in workspace
                self.workspace.upsert_object(
                    oid='test_output',
                    features={'applied': 1.0},
                    properties={'status': 'success'}
                )

            except Exception as e:
                # Pattern application failed
                self.workspace.upsert_object(
                    oid='test_output',
                    features={'applied': 0.0},
                    properties={'status': 'failed', 'error': str(e)}
                )

        def _hebbian_strengthen():
            """Strengthen successful pattern connections in PAM."""
            if self.winning_pattern and self.final_output:
                # Strengthen with reward = 1.0 (success)
                self.pam.hebbian_learner.strengthen_from_pattern(
                    self.winning_pattern,
                    reward=1.0
                )

        return [
            Codelet(
                name='arc_select_winner',
                urgency=1.0,
                action=_select_winning_pattern,
                kind='action',
                metadata={}
            ),
            Codelet(
                name='arc_apply_pattern',
                urgency=0.9,
                action=_apply_pattern_to_test,
                kind='action',
                metadata={}
            ),
            Codelet(
                name='arc_hebbian_learning',
                urgency=0.7,
                action=_hebbian_strengthen,
                kind='action',
                metadata={}
            ),
        ]

    def get_coalitions(self) -> List[Coalition]:
        """Get current coalitions for workspace competition."""
        return self.coalitions

    def set_task(self, demonstrations: List[GridPair], test_input: List[List[int]]):
        """Set the current ARC task to solve."""
        self.demonstrations = demonstrations
        self.test_input = test_input
        self.hypotheses.clear()
        self.coalitions.clear()
        self.winning_pattern = None
        self.final_output = None
        self.workspace.clear()

    def get_output(self) -> Optional[List[List[int]]]:
        """Get the final test output."""
        return self.final_output
