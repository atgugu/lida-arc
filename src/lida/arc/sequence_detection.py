"""
Sequence detection for composite operations.

This module implements efficient detection of operation sequences (e.g., rotate_90 THEN recolor)
using iterative deepening with beam search, heuristic pruning, and PAM-guided ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
import copy

from .primitives import PrimitiveLibrary


# Type aliases
Grid = List[List[int]]
GridHash = Tuple[Tuple[int, ...], ...]


class GridDistance:
    """Compute distance between grids to guide search."""

    @staticmethod
    def compute(grid1: Grid, grid2: Grid) -> float:
        """
        Compute normalized distance between two grids.

        Returns value in [0, 1] where 0 = identical, 1 = maximally different.
        """
        # Handle size mismatch
        if len(grid1) != len(grid2):
            return 1.0
        if grid1 and grid2 and len(grid1[0]) != len(grid2[0]):
            return 1.0

        if not grid1 or not grid2:
            return 1.0

        # Count differing pixels
        total_pixels = len(grid1) * len(grid1[0])
        if total_pixels == 0:
            return 0.0

        different_pixels = 0
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] != grid2[i][j]:
                    different_pixels += 1

        return different_pixels / total_pixels

    @staticmethod
    def is_closer(grid1: Grid, grid2: Grid, target: Grid) -> bool:
        """Check if grid1 is closer to target than grid2."""
        return GridDistance.compute(grid1, target) < GridDistance.compute(grid2, target)


class SequencePruner:
    """Prunes invalid or unlikely operation sequences."""

    # Operation inversions (op1 followed by op2 cancels out or is redundant)
    INVERSE_PAIRS: Set[Tuple[str, str]] = {
        ('rotate_90', 'rotate_270'),
        ('rotate_270', 'rotate_90'),
        ('rotate_180', 'rotate_180'),  # Self-inverse
        ('reflect_horizontal', 'reflect_horizontal'),  # Self-inverse
        ('reflect_vertical', 'reflect_vertical'),  # Self-inverse
        ('reflect_diagonal', 'reflect_diagonal'),  # Self-inverse
    }

    # Redundant pairs (can be replaced with single operation)
    REDUNDANT_PAIRS: Set[Tuple[str, str]] = {
        ('rotate_90', 'rotate_90'),  # = rotate_180
        ('rotate_90', 'rotate_180'),  # = rotate_270
        ('rotate_180', 'rotate_270'),  # = rotate_90
    }

    def is_valid_sequence(self, sequence: List[str]) -> bool:
        """Check if sequence is valid and sensible."""
        if len(sequence) <= 1:
            return True

        # Check for inversions and redundancies
        for i in range(len(sequence) - 1):
            op1, op2 = sequence[i], sequence[i + 1]

            # No inverse pairs
            if (op1, op2) in self.INVERSE_PAIRS or (op2, op1) in self.INVERSE_PAIRS:
                return False

            # No redundant pairs
            if (op1, op2) in self.REDUNDANT_PAIRS:
                return False

        # No more than one rotation in sequence (multiple rotations = single rotation)
        rotation_ops = ['rotate_90', 'rotate_180', 'rotate_270']
        rotation_count = sum(1 for op in sequence if op in rotation_ops)
        if rotation_count > 1:
            return False

        # Allow at most 2 reflections (rotation + reflection is valid, but 3+ reflections is excessive)
        # Note: This allows valid combinations like [rotate_90, reflect_vertical]
        reflection_ops = ['reflect_horizontal', 'reflect_vertical', 'reflect_diagonal']
        reflection_count = sum(1 for op in sequence if op in reflection_ops)
        if reflection_count > 2:
            return False

        # Color operations generally come last
        # (Rotating after recoloring is uncommon in ARC tasks)
        color_ops = {'recolor', 'fill_background'}
        geometric_ops = set(rotation_ops + reflection_ops)

        last_color_idx = -1
        for i, op in enumerate(sequence):
            if op in color_ops:
                last_color_idx = i

        if last_color_idx != -1:
            # Check if any geometric operation comes after color operation
            for i in range(last_color_idx + 1, len(sequence)):
                if sequence[i] in geometric_ops:
                    return False  # Geometric after color is unusual

        return True

    def prune_beam(self, candidates: List['SequenceCandidate'], beam_width: int) -> List['SequenceCandidate']:
        """
        Prune candidates to beam_width best options.

        Ranking criteria:
        1. Exact match (distance = 0) always kept
        2. Lower distance to target preferred
        3. Shorter sequences preferred (tie-breaker)
        4. Higher confidence preferred (tie-breaker)

        IMPORTANT: When finding all sequences, we keep both exact matches AND
        promising non-matches to continue exploring longer sequences.
        """
        if len(candidates) <= beam_width:
            return candidates

        # Sort all candidates by quality
        # Exact matches (distance=0) will naturally sort to the top
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.distance_to_target, len(c.sequence), -c.confidence)
        )

        # Keep top beam_width candidates (includes both matches and near-matches)
        return sorted_candidates[:beam_width]


@dataclass
class SequenceCandidate:
    """A candidate sequence being explored."""

    sequence: List[str] = field(default_factory=list)
    current_grid: Optional[Grid] = None
    distance_to_target: float = 1.0
    confidence: float = 1.0
    operation_params: Dict[str, Dict] = field(default_factory=dict)  # Store parameters for parametric ops

    def extend(
        self,
        operation: str,
        result_grid: Grid,
        target: Grid,
        params: Optional[Dict] = None,
        color_mapping: Optional[Dict[int, int]] = None
    ) -> 'SequenceCandidate':
        """Extend this candidate with one more operation."""
        new_sequence = self.sequence + [operation]
        new_distance = GridDistance.compute(result_grid, target)

        # Copy existing parameters and add new ones if provided
        new_params = self.operation_params.copy()
        if params:
            new_params[operation] = params

        # Confidence calculation:
        # - Length penalty: prefer shorter sequences (0.9^length)
        # - Progress bonus: reward getting closer (1.0 - distance)
        length_penalty = 0.9 ** len(new_sequence)
        progress = 1.0 - new_distance

        new_confidence = length_penalty * progress

        return SequenceCandidate(
            sequence=new_sequence,
            current_grid=copy.deepcopy(result_grid),
            distance_to_target=new_distance,
            confidence=new_confidence,
            operation_params=new_params
        )


class SequenceDetector:
    """Detect operation sequences using iterative deepening with beam search."""

    def __init__(self, primitives: PrimitiveLibrary, max_depth: int = 3, beam_width: int = 5):
        """
        Args:
            primitives: Library of cognitive primitives
            max_depth: Maximum sequence length to try
            beam_width: Number of candidates to keep at each depth
        """
        self.primitives = primitives
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.pruner = SequencePruner()

    def _generate_operation_variants(self, op_name: str, current_grid: Grid, target_grid: Grid) -> List[Tuple[str, Dict]]:
        """
        Generate variants of an operation with different parameters.

        Returns:
            List of (operation_name, parameters_dict) tuples to try
        """
        variants = []

        if op_name == 'tile':
            # Try different tiling factors based on size ratio
            try:
                if current_grid and target_grid:
                    in_h, in_w = len(current_grid), len(current_grid[0])
                    out_h, out_w = len(target_grid), len(target_grid[0])

                    # Try factors that make sense
                    for repeat_v in [2, 3, 4]:
                        for repeat_w in [2, 3, 4]:
                            if repeat_v * in_h <= out_h + 1 and repeat_w * in_w <= out_w + 1:
                                variants.append((op_name, {'repeat_v': repeat_v, 'repeat_w': repeat_w}))
            except (KeyError, IndexError, TypeError):
                pass

        elif op_name == 'scale_grid':
            # Try common scale factors
            try:
                if current_grid and target_grid:
                    in_h, in_w = len(current_grid), len(current_grid[0])
                    out_h, out_w = len(target_grid), len(target_grid[0])

                    # Compute approximate scale factor
                    if in_h > 0 and in_w > 0:
                        scale_h = out_h / in_h
                        scale_w = out_w / in_w

                        # Try exact scale if uniform
                        if abs(scale_h - scale_w) < 0.1:
                            variants.append((op_name, {'scale_factor': scale_h}))

                        # Try common scales: 0.5, 2, 3
                        for scale in [0.5, 2.0, 3.0]:
                            if 0.8 < scale / scale_h < 1.2 or 0.8 < scale / scale_w < 1.2:
                                variants.append((op_name, {'scale_factor': scale}))
            except (KeyError, IndexError, TypeError, ZeroDivisionError):
                pass

        elif op_name in ['crop', 'extend', 'overlay']:
            # Skip parametric operations that need specific coordinates for now
            pass

        elif op_name == 'auto_crop':
            # No parameters needed
            variants.append((op_name, {}))

        else:
            # No parameters needed
            variants.append((op_name, {}))

        return variants if variants else [(op_name, {})]

    @staticmethod
    def _infer_color_mapping(grid1: Grid, grid2: Grid) -> Optional[Dict[int, int]]:
        """Infer color mapping between two grids of same size."""
        if not grid1 or not grid2:
            return None
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return None

        color_map = {}
        for r in range(len(grid1)):
            for c in range(len(grid1[0])):
                in_color = grid1[r][c]
                out_color = grid2[r][c]

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return None  # Inconsistent mapping
                else:
                    color_map[in_color] = out_color

        # Only return if it's actually a transformation
        if any(k != v for k, v in color_map.items()):
            return color_map

        return None

    def find_sequence(
        self,
        input_grid: Grid,
        output_grid: Grid,
        color_mapping: Optional[Dict[int, int]] = None
    ) -> Optional[List[str]]:
        """
        Find sequence of operations that transforms input to output.

        Args:
            input_grid: Starting grid
            output_grid: Target grid
            color_mapping: Optional color mapping (for recolor operation)

        Returns:
            List of operation names, or None if no sequence found
        """
        # Start with single candidate (empty sequence, starting grid)
        candidates = [SequenceCandidate(
            sequence=[],
            current_grid=copy.deepcopy(input_grid),
            distance_to_target=GridDistance.compute(input_grid, output_grid),
            confidence=1.0
        )]

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            # Extend each candidate with one more operation
            new_candidates = []

            for candidate in candidates:
                # Try each primitive operation
                for prim_name in self.primitives.get_all_names():
                    # Create extended sequence
                    extended_sequence = candidate.sequence + [prim_name]

                    # Check if sequence is valid
                    if not self.pruner.is_valid_sequence(extended_sequence):
                        continue

                    # Apply operation
                    try:
                        result_grid = copy.deepcopy(candidate.current_grid)
                        prim = self.primitives.get(prim_name)

                        # Special handling for recolor
                        if prim_name == 'recolor' and color_mapping:
                            result_grid = prim.execute(result_grid, color_mapping)
                        else:
                            result_grid = prim.execute(result_grid)

                        # Create new candidate
                        new_candidate = candidate.extend(
                            prim_name,
                            result_grid,
                            output_grid,
                            color_mapping
                        )

                        new_candidates.append(new_candidate)

                        # Note: No early termination - explore all depths to find best solution
                        # With single training examples, simple solutions might not generalize

                    except Exception:
                        # Operation failed, skip this candidate
                        continue

            if not new_candidates:
                break

            # Prune to beam width
            candidates = self.pruner.prune_beam(new_candidates, self.beam_width)

        # Return best candidate found
        # Prefer: exact matches > closer matches, then shorter sequences
        if candidates:
            # First try to find exact matches
            exact_matches = [c for c in candidates if c.distance_to_target == 0]

            if exact_matches:
                # Among exact matches, prefer shorter sequences (Occam's razor)
                # But also consider confidence (longer well-validated sequences might be better)
                best = min(exact_matches, key=lambda c: (len(c.sequence), -c.confidence))
                return best.sequence

            # No exact match, return closest
            best = min(candidates, key=lambda c: (c.distance_to_target, len(c.sequence)))
            if best.distance_to_target < 0.5:  # Accept if at least 50% correct
                return best.sequence

        return None

    def find_all_sequences(
        self,
        input_grid: Grid,
        output_grid: Grid,
        color_mapping: Optional[Dict[int, int]] = None
    ) -> List[Tuple[List[str], Dict[str, Dict]]]:
        """
        Find ALL exact-match sequences across all depths.

        Returns:
            List of (sequence, operation_params) tuples where:
            - sequence: List of operation names
            - operation_params: Dict mapping operation names to their parameters
            Sorted by length (shortest first)
        """
        all_exact_matches = []

        # Start with single candidate
        candidates = [SequenceCandidate(
            sequence=[],
            current_grid=copy.deepcopy(input_grid),
            distance_to_target=GridDistance.compute(input_grid, output_grid),
            confidence=1.0
        )]

        # Iterative deepening - collect all exact matches
        debug = False  # Set to True for debugging
        for depth in range(1, self.max_depth + 1):
            if debug:
                print(f"[find_all_sequences] Depth {depth}: Starting with {len(candidates)} candidates")

            new_candidates = []

            # IMPORTANT: We need to explore from the INITIAL state at each depth,
            # not just extend existing candidates. This ensures we find all sequences.
            # For example, we want to find both ['recolor'] and ['rotate_90', 'recolor']

            # Start fresh from input grid for each depth search
            if depth == 1:
                # Depth 1: try all single operations
                base_candidates = [SequenceCandidate([], input_grid, GridDistance.compute(input_grid, output_grid), 1.0)]
            else:
                # Depth 2+: extend candidates from previous depth that haven't matched yet
                # But also keep exact matches to prevent losing them
                base_candidates = [c for c in candidates if c.distance_to_target > 0]
                if debug:
                    print(f"[find_all_sequences]   Filtered to {len(base_candidates)} non-matching candidates")
                # If all candidates matched, we're done exploring
                if not base_candidates:
                    if debug:
                        print(f"[find_all_sequences]   All candidates matched - stopping search")
                    break

            for candidate in base_candidates:
                for prim_name in self.primitives.get_all_names():
                    extended_sequence = candidate.sequence + [prim_name]

                    if not self.pruner.is_valid_sequence(extended_sequence):
                        continue

                    # Generate parameter variants for this operation
                    variants = self._generate_operation_variants(prim_name, candidate.current_grid, output_grid)

                    for op_name, params in variants:
                        try:
                            result_grid = copy.deepcopy(candidate.current_grid)
                            prim = self.primitives.get(op_name)

                            # Handle different operation types with parameters
                            if op_name == 'recolor':
                                # Infer color mapping from current grid to target
                                inferred_mapping = self._infer_color_mapping(candidate.current_grid, output_grid)

                                if inferred_mapping:
                                    result_grid = prim.execute(result_grid, inferred_mapping)
                                elif color_mapping:
                                    # Fall back to provided mapping
                                    result_grid = prim.execute(result_grid, color_mapping)
                                else:
                                    # No mapping available, skip
                                    continue

                            elif op_name == 'tile' and params:
                                # Apply tiling with parameters
                                result_grid = prim.execute(result_grid, params['repeat_v'], params['repeat_w'])

                            elif op_name == 'scale_grid' and params:
                                # Apply scaling with parameters
                                result_grid = prim.execute(result_grid, params['scale_factor'])

                            else:
                                # No parameters needed
                                result_grid = prim.execute(result_grid)

                            new_candidate = candidate.extend(
                                op_name,
                                result_grid,
                                output_grid,
                                params=params,
                                color_mapping=color_mapping
                            )

                            new_candidates.append(new_candidate)

                            # Collect exact matches (sequence + params)
                            if new_candidate.distance_to_target == 0:
                                all_exact_matches.append((new_candidate.sequence, new_candidate.operation_params))

                        except Exception:
                            continue

            if not new_candidates:
                if debug:
                    print(f"[find_all_sequences]   No new candidates generated - stopping")
                break

            if debug:
                print(f"[find_all_sequences]   Generated {len(new_candidates)} new candidates")
                print(f"[find_all_sequences]   Exact matches so far: {len(all_exact_matches)}")

            candidates = self.pruner.prune_beam(new_candidates, self.beam_width)

            if debug:
                print(f"[find_all_sequences]   After pruning: {len(candidates)} candidates kept")

        # Sort by length (prefer simpler explanations first)
        all_exact_matches.sort(key=lambda x: len(x[0]))

        # Debug output
        if False:  # Set to True for debugging
            print(f"[find_all_sequences] Returning {len(all_exact_matches)} sequences")
            for seq, params in all_exact_matches[:10]:
                print(f"[find_all_sequences]   - {seq} with params {params}")

        return all_exact_matches

    def find_sequence_with_ranking(
        self,
        input_grid: Grid,
        output_grid: Grid,
        operation_ranking: List[str],
        color_mapping: Optional[Dict[int, int]] = None
    ) -> Optional[List[str]]:
        """
        Find sequence with PAM-guided operation ranking.

        Similar to find_sequence but tries operations in ranked order.
        """
        # Start with single candidate
        candidates = [SequenceCandidate(
            sequence=[],
            current_grid=copy.deepcopy(input_grid),
            distance_to_target=GridDistance.compute(input_grid, output_grid),
            confidence=1.0
        )]

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            new_candidates = []

            for candidate in candidates:
                # Try operations in ranked order (PAM-guided)
                for prim_name in operation_ranking:
                    extended_sequence = candidate.sequence + [prim_name]

                    if not self.pruner.is_valid_sequence(extended_sequence):
                        continue

                    try:
                        result_grid = copy.deepcopy(candidate.current_grid)
                        prim = self.primitives.get(prim_name)

                        if prim_name == 'recolor' and color_mapping:
                            result_grid = prim.execute(result_grid, color_mapping)
                        else:
                            result_grid = prim.execute(result_grid)

                        new_candidate = candidate.extend(
                            prim_name,
                            result_grid,
                            output_grid,
                            color_mapping
                        )

                        new_candidates.append(new_candidate)

                        if new_candidate.distance_to_target == 0:
                            return new_candidate.sequence

                    except Exception:
                        continue

            if not new_candidates:
                break

            candidates = self.pruner.prune_beam(new_candidates, self.beam_width)

        if candidates:
            best = min(candidates, key=lambda c: c.distance_to_target)
            if best.distance_to_target < 0.5:
                return best.sequence

        return None
