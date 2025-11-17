"""
Tests for grid perception: object extraction and analysis.
"""

import pytest
from lida.arc.perception import GridObject, ObjectExtractor, GridAnalyzer


class TestGridObject:
    def test_create_grid_object(self):
        """Test GridObject creation."""
        pixels = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])
        obj = GridObject(
            object_id="test_obj",
            pixels=pixels,
            color=1,
            bounding_box=(0, 0, 1, 1)
        )

        assert obj.size == 4
        assert obj.width == 2
        assert obj.height == 2
        assert obj.color == 1

    def test_centroid(self):
        """Test centroid calculation."""
        # Square object at origin
        pixels = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])
        obj = GridObject(
            object_id="square",
            pixels=pixels,
            color=1,
            bounding_box=(0, 0, 1, 1)
        )

        centroid = obj.centroid
        assert centroid == (0.5, 0.5)

        # L-shaped object
        pixels = frozenset([(0, 0), (1, 0), (2, 0), (2, 1)])
        obj = GridObject(
            object_id="L",
            pixels=pixels,
            color=2,
            bounding_box=(0, 0, 2, 1)
        )

        centroid = obj.centroid
        # Centroid should be average of (0,0), (1,0), (2,0), (2,1)
        assert centroid == (1.25, 0.25)

    def test_density(self):
        """Test density calculation."""
        # Filled square (density = 1.0)
        pixels = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])
        obj = GridObject(
            object_id="filled",
            pixels=pixels,
            color=1,
            bounding_box=(0, 0, 1, 1)
        )
        assert obj.density == 1.0

        # Hollow square (density < 1.0)
        pixels = frozenset([(0, 0), (0, 2), (2, 0), (2, 2)])  # Corners only
        obj = GridObject(
            object_id="hollow",
            pixels=pixels,
            color=1,
            bounding_box=(0, 0, 2, 2)
        )
        assert obj.density == 4 / 9  # 4 pixels in 3x3 bbox

    def test_symmetry_vertical(self):
        """Test vertical symmetry detection."""
        # Symmetric square
        pixels = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])
        obj = GridObject(
            object_id="sym_square",
            pixels=pixels,
            color=1,
            bounding_box=(0, 0, 1, 1)
        )
        assert obj.has_symmetry_vertical()

        # Asymmetric L-shape
        pixels = frozenset([(0, 0), (1, 0), (2, 0), (2, 1)])
        obj = GridObject(
            object_id="asym_L",
            pixels=pixels,
            color=1,
            bounding_box=(0, 0, 2, 1)
        )
        assert not obj.has_symmetry_vertical()

    def test_shape_signature(self):
        """Test shape signature (normalized representation)."""
        # Same shape at different positions should have same signature
        pixels1 = frozenset([(0, 0), (0, 1), (1, 0)])  # L at origin
        obj1 = GridObject("obj1", pixels1, 1, (0, 0, 1, 1))

        pixels2 = frozenset([(5, 5), (5, 6), (6, 5)])  # L at (5,5)
        obj2 = GridObject("obj2", pixels2, 1, (5, 5, 6, 6))

        assert obj1.get_shape_signature() == obj2.get_shape_signature()

        # Different shapes should have different signatures
        pixels3 = frozenset([(0, 0), (0, 1)])  # Line
        obj3 = GridObject("obj3", pixels3, 1, (0, 0, 0, 1))

        assert obj1.get_shape_signature() != obj3.get_shape_signature()

    def test_to_features(self):
        """Test feature extraction."""
        pixels = frozenset([(0, 0), (0, 1), (1, 0), (1, 1)])
        obj = GridObject(
            object_id="feature_test",
            pixels=pixels,
            color=3,
            bounding_box=(0, 0, 1, 1)
        )

        features = obj.to_features()

        assert features['size'] == 4.0
        assert features['width'] == 2.0
        assert features['height'] == 2.0
        assert features['color'] == 3.0
        assert features['density'] == 1.0
        assert features['aspect_ratio'] == 1.0  # Square


class TestObjectExtractor:
    def test_extract_single_object(self):
        """Test extracting a single object."""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]

        extractor = ObjectExtractor(background_color=0)
        objects = extractor.extract_objects(grid)

        assert len(objects) == 1
        assert objects[0].size == 4
        assert objects[0].color == 1

    def test_extract_multiple_objects(self):
        """Test extracting multiple disconnected objects."""
        grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 4],
        ]

        extractor = ObjectExtractor(background_color=0)
        objects = extractor.extract_objects(grid)

        assert len(objects) == 4

        # Check colors
        colors = sorted([obj.color for obj in objects])
        assert colors == [1, 2, 3, 4]

        # Check sizes
        sizes = {obj.color: obj.size for obj in objects}
        assert sizes[1] == 4  # 2x2 square
        assert sizes[2] == 4  # 2x2 square
        assert sizes[3] == 1  # Single pixel
        assert sizes[4] == 1  # Single pixel

    def test_8_connectivity(self):
        """Test 8-connected component extraction."""
        grid = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

        # With 4-connectivity, should be 3 separate objects
        extractor4 = ObjectExtractor(background_color=0, connectivity=4)
        objects4 = extractor4.extract_objects(grid)
        assert len(objects4) == 3

        # With 8-connectivity, should be 1 connected object (diagonal)
        extractor8 = ObjectExtractor(background_color=0, connectivity=8)
        objects8 = extractor8.extract_objects(grid)
        assert len(objects8) == 1
        assert objects8[0].size == 3

    def test_empty_grid(self):
        """Test handling empty grid."""
        extractor = ObjectExtractor()

        # Empty grid
        assert extractor.extract_objects([]) == []

        # Grid with only background
        grid = [[0, 0], [0, 0]]
        assert extractor.extract_objects(grid) == []


class TestGridAnalyzer:
    def test_color_histogram(self):
        """Test color histogram computation."""
        grid = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 0],
        ]

        analyzer = GridAnalyzer()
        hist = analyzer.compute_color_histogram(grid)

        assert hist[1] == 4
        assert hist[2] == 4
        assert hist[3] == 3
        assert hist[0] == 1

    def test_vertical_symmetry(self):
        """Test vertical symmetry detection."""
        analyzer = GridAnalyzer()

        # Symmetric grid
        symmetric = [
            [1, 2, 1],
            [3, 4, 3],
        ]
        symmetry = analyzer.detect_grid_symmetry(symmetric)
        assert symmetry['vertical']

        # Asymmetric grid
        asymmetric = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        symmetry = analyzer.detect_grid_symmetry(asymmetric)
        assert not symmetry['vertical']

    def test_horizontal_symmetry(self):
        """Test horizontal symmetry detection."""
        analyzer = GridAnalyzer()

        # Symmetric grid
        symmetric = [
            [1, 2, 3],
            [1, 2, 3],
        ]
        symmetry = analyzer.detect_grid_symmetry(symmetric)
        assert symmetry['horizontal']

        # Asymmetric grid
        asymmetric = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        symmetry = analyzer.detect_grid_symmetry(asymmetric)
        assert not symmetry['horizontal']

    def test_rotational_symmetry(self):
        """Test rotational symmetry detection."""
        analyzer = GridAnalyzer()

        # 180-degree rotational symmetry
        rot180 = [
            [1, 2, 3],
            [3, 2, 1],
        ]
        symmetry = analyzer.detect_grid_symmetry(rot180)
        assert symmetry['rotational_180']

        # 90-degree rotational symmetry (must be square)
        rot90 = [
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1],
        ]
        symmetry = analyzer.detect_grid_symmetry(rot90)
        assert symmetry['rotational_90']

    def test_grid_features(self):
        """Test comprehensive feature extraction."""
        grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ]

        analyzer = GridAnalyzer()
        features = analyzer.compute_grid_features(grid)

        assert features['grid_height'] == 3.0
        assert features['grid_width'] == 5.0
        assert features['grid_size'] == 15.0
        assert features['n_objects'] == 2.0  # Two 2x2 squares
        assert features['n_colors'] == 3.0  # Colors 0, 1, 2
        assert features['max_object_size'] == 4.0

    def test_find_repeating_pattern(self):
        """Test tiling pattern detection."""
        analyzer = GridAnalyzer()

        # 2x2 tile repeated 2x2
        tiled = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]

        pattern = analyzer.find_repeating_pattern(tiled)

        assert pattern is not None
        assert pattern['type'] == 'tiling'
        assert pattern['tile_height'] == 2
        assert pattern['tile_width'] == 2
        assert pattern['repetitions_v'] == 2
        assert pattern['repetitions_h'] == 2

        # Non-tiled grid
        non_tiled = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        pattern = analyzer.find_repeating_pattern(non_tiled)
        assert pattern is None

    def test_compare_grids(self):
        """Test grid comparison."""
        analyzer = GridAnalyzer()

        grid1 = [[1, 2, 3], [4, 5, 6]]
        grid2 = [[1, 2, 0], [4, 5, 6]]  # One difference

        result = analyzer.compare_grids(grid1, grid2)

        assert not result['size_changed']
        assert result['n_differences'] == 1
        assert result['percent_changed'] == 1/6
        assert not result['is_identical']

        # Identical grids
        result = analyzer.compare_grids(grid1, grid1)
        assert result['is_identical']
        assert result['n_differences'] == 0

        # Different sizes
        grid3 = [[1, 2], [3, 4]]
        result = analyzer.compare_grids(grid1, grid3)
        assert result['size_changed']
