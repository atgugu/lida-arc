"""
Tests for cognitive primitives library.
"""

import pytest
from lida.arc.primitives import (
    DetectObjectsPrimitive, CompareGridsPrimitive, FindPatternPrimitive,
    MatchObjectsPrimitive, AnalyzeFeaturesPrimitive,
    RecolorPrimitive, RotateGridPrimitive, ReflectGridPrimitive,
    CropGridPrimitive, ExtendGridPrimitive, TileGridPrimitive,
    OverlayGridsPrimitive, FillBackgroundPrimitive,
    PrimitiveLibrary
)


class TestPerceptualPrimitives:
    def test_detect_objects(self):
        """Test object detection primitive."""
        prim = DetectObjectsPrimitive()

        grid = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ]

        objects = prim.execute(grid)
        assert len(objects) == 2

        colors = sorted([obj.color for obj in objects])
        assert colors == [1, 2]

    def test_compare_grids(self):
        """Test grid comparison primitive."""
        prim = CompareGridsPrimitive()

        grid1 = [[1, 2, 3], [4, 5, 6]]
        grid2 = [[1, 2, 0], [4, 5, 6]]

        result = prim.execute(grid1, grid2)

        assert not result['size_changed']
        assert result['n_differences'] == 1
        assert result['differences'][0]['position'] == (0, 2)
        assert result['differences'][0]['from_color'] == 3
        assert result['differences'][0]['to_color'] == 0

    def test_find_pattern(self):
        """Test pattern detection primitive."""
        prim = FindPatternPrimitive()

        # Tiled pattern
        tiled = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]

        result = prim.execute(tiled)
        assert result is not None
        assert result['type'] == 'tiling'
        assert result['tile_height'] == 2
        assert result['tile_width'] == 2

        # Non-tiled
        non_tiled = [[1, 2, 3], [4, 5, 6]]
        result = prim.execute(non_tiled)
        assert result is None

    def test_match_objects(self):
        """Test object matching primitive."""
        prim = MatchObjectsPrimitive()

        grid1 = [[1, 1], [2, 2]]
        grid2 = [[2, 2], [1, 1]]

        detect = DetectObjectsPrimitive()
        objects1 = detect.execute(grid1)
        objects2 = detect.execute(grid2)

        result = prim.execute(objects1, objects2)

        # Should find 2 matches (color 1 and color 2)
        assert len(result['matches']) == 2

    def test_analyze_features(self):
        """Test feature analysis primitive."""
        prim = AnalyzeFeaturesPrimitive()

        grid = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]

        features = prim.execute(grid)

        assert features['grid_height'] == 3.0
        assert features['grid_width'] == 3.0
        assert features['n_objects'] == 1.0
        assert features['n_colors'] == 2.0  # 0 and 1


class TestManipulationPrimitives:
    def test_recolor(self):
        """Test recolor primitive."""
        prim = RecolorPrimitive()

        grid = [[1, 2], [3, 4]]
        color_map = {1: 5, 2: 6, 3: 7, 4: 8}

        result = prim.execute(grid, color_map)

        assert result == [[5, 6], [7, 8]]

    def test_rotate_90(self):
        """Test 90-degree rotation."""
        prim = RotateGridPrimitive(90)

        grid = [
            [1, 2],
            [3, 4],
        ]

        result = prim.execute(grid)

        # 90° CW: [1,2]  ->  [3,1]
        #         [3,4]      [4,2]
        assert result == [[3, 1], [4, 2]]

    def test_rotate_180(self):
        """Test 180-degree rotation."""
        prim = RotateGridPrimitive(180)

        grid = [
            [1, 2],
            [3, 4],
        ]

        result = prim.execute(grid)

        # 180°: [1,2]  ->  [4,3]
        #       [3,4]      [2,1]
        assert result == [[4, 3], [2, 1]]

    def test_rotate_270(self):
        """Test 270-degree rotation."""
        prim = RotateGridPrimitive(270)

        grid = [
            [1, 2],
            [3, 4],
        ]

        result = prim.execute(grid)

        # 270° CW (= 90° CCW): [1,2]  ->  [2,4]
        #                      [3,4]      [1,3]
        assert result == [[2, 4], [1, 3]]

    def test_reflect_horizontal(self):
        """Test horizontal reflection."""
        prim = ReflectGridPrimitive('horizontal')

        grid = [
            [1, 2],
            [3, 4],
        ]

        result = prim.execute(grid)

        # Flip vertically: [1,2]  ->  [3,4]
        #                  [3,4]      [1,2]
        assert result == [[3, 4], [1, 2]]

    def test_reflect_vertical(self):
        """Test vertical reflection."""
        prim = ReflectGridPrimitive('vertical')

        grid = [
            [1, 2],
            [3, 4],
        ]

        result = prim.execute(grid)

        # Flip horizontally: [1,2]  ->  [2,1]
        #                    [3,4]      [4,3]
        assert result == [[2, 1], [4, 3]]

    def test_reflect_diagonal(self):
        """Test diagonal reflection (transpose)."""
        prim = ReflectGridPrimitive('diagonal')

        grid = [
            [1, 2, 3],
            [4, 5, 6],
        ]

        result = prim.execute(grid)

        # Transpose: [1,2,3]  ->  [1,4]
        #            [4,5,6]      [2,5]
        #                         [3,6]
        assert result == [[1, 4], [2, 5], [3, 6]]

    def test_crop(self):
        """Test crop primitive."""
        prim = CropGridPrimitive()

        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
        ]

        # Crop to center 2x2
        result = prim.execute(grid, min_r=0, min_c=1, max_r=1, max_c=2)

        assert result == [[2, 3], [6, 7]]

    def test_extend(self):
        """Test extend primitive."""
        prim = ExtendGridPrimitive()

        grid = [[1, 2], [3, 4]]

        # Add 1 padding on all sides with 0
        result = prim.execute(grid, top=1, bottom=1, left=1, right=1, fill_value=0)

        assert result == [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]

    def test_tile(self):
        """Test tile primitive."""
        prim = TileGridPrimitive()

        pattern = [[1, 2]]

        # Tile 2x3
        result = prim.execute(pattern, repeat_v=2, repeat_h=3)

        assert result == [
            [1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2],
        ]

    def test_overlay_replace(self):
        """Test overlay with replace mode."""
        prim = OverlayGridsPrimitive()

        grid1 = [[1, 1, 1], [1, 1, 1]]
        grid2 = [[0, 2, 0], [2, 0, 2]]

        result = prim.execute(grid1, grid2, mode='replace')

        # grid2 overwrites grid1 where non-zero
        assert result == [[1, 2, 1], [2, 1, 2]]

    def test_overlay_max(self):
        """Test overlay with max mode."""
        prim = OverlayGridsPrimitive()

        grid1 = [[1, 2], [3, 4]]
        grid2 = [[2, 1], [2, 5]]

        result = prim.execute(grid1, grid2, mode='max')

        assert result == [[2, 2], [3, 5]]

    def test_fill_background(self):
        """Test fill background primitive."""
        prim = FillBackgroundPrimitive()

        grid = [[1, 0, 2], [0, 3, 0]]

        result = prim.execute(grid, fill_color=9)

        assert result == [[1, 9, 2], [9, 3, 9]]


class TestPrimitiveLibrary:
    def test_library_creation(self):
        """Test primitive library initialization."""
        lib = PrimitiveLibrary()

        # Should have 36 primitives (5 perceptual + 12 manipulation + 8 object/size + 4 conditional + 5 morphological + 2 pattern_extraction)
        assert len(lib) == 36

    def test_get_primitive(self):
        """Test retrieving primitives by name."""
        lib = PrimitiveLibrary()

        # Get by name
        prim = lib.get('rotate_90')
        assert prim is not None
        assert prim.name == 'rotate_90'

        # Non-existent primitive
        assert lib.get('nonexistent') is None

    def test_get_by_category(self):
        """Test retrieving primitives by category."""
        lib = PrimitiveLibrary()

        perceptual = lib.get_all_by_category('perceptual')
        assert len(perceptual) == 5

        manipulation = lib.get_all_by_category('manipulation')
        assert len(manipulation) == 15  # 12 original + scale_grid + auto_crop + resize_to_target

        object_manipulation = lib.get_all_by_category('object_manipulation')
        assert len(object_manipulation) == 5  # render_objects, move, scale, replicate, recolor

        conditional = lib.get_all_by_category('conditional')
        assert len(conditional) == 4  # recolor_if_has_neighbor, isolated, on_edge, remove_if_isolated

        morphology = lib.get_all_by_category('morphology')
        assert len(morphology) == 5  # dilate, erode, flood_fill, fill_enclosed, spread_to_neighbors

        pattern_extraction = lib.get_all_by_category('pattern_extraction')
        assert len(pattern_extraction) == 2  # extract_pattern, extract_top_left

    def test_register_custom_primitive(self):
        """Test registering a custom primitive."""
        lib = PrimitiveLibrary()

        # Create a dummy primitive
        class CustomPrimitive:
            def __init__(self):
                self.name = 'custom'
                self.category = 'test'

            def execute(self):
                return "custom"

            def get_features(self):
                return {'custom': 1.0}

        lib.register(CustomPrimitive())

        assert 'custom' in lib
        assert lib.get('custom').name == 'custom'

    def test_get_features_for_pam(self):
        """Test feature extraction for PAM seeding."""
        lib = PrimitiveLibrary()

        features = lib.get_features_for_pam()

        # Should have features for all 36 primitives
        assert len(features) == 36

        # Check specific primitive
        assert 'rotate_90' in features
        assert 'rotation' in features['rotate_90']
        assert features['rotate_90']['rotation'] == 1.0

    def test_primitive_features(self):
        """Test that all primitives have valid features."""
        lib = PrimitiveLibrary()

        for name, prim in lib.primitives.items():
            features = prim.get_features()

            # Should have at least category feature
            assert len(features) > 0
            assert prim.category in features or any(prim.category in k for k in features.keys())


class TestPrimitiveIntegration:
    def test_rotation_composition(self):
        """Test composing rotation primitives."""
        lib = PrimitiveLibrary()

        grid = [[1, 2], [3, 4]]

        # Rotate 90° four times should return to original
        result = grid
        for _ in range(4):
            result = lib.get('rotate_90').execute(result)

        assert result == grid

    def test_transformation_pipeline(self):
        """Test pipeline of transformations."""
        lib = PrimitiveLibrary()

        grid = [[1, 2], [3, 4]]

        # Pipeline: rotate 90° then reflect horizontal
        result = lib.get('rotate_90').execute(grid)
        result = lib.get('reflect_horizontal').execute(result)

        # Verify result makes sense
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_color_then_pattern(self):
        """Test color transformation followed by pattern detection."""
        lib = PrimitiveLibrary()

        # Create a grid with a clear tiling pattern
        grid = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]

        # Recolor
        color_map = {1: 5, 2: 6, 3: 7, 4: 8}
        result = lib.get('recolor').execute(grid, color_map)

        assert result[0][0] == 5  # Verify color mapping worked

        # Detect pattern
        pattern = lib.get('find_pattern').execute(result)

        # Should detect 2x2 tiling
        assert pattern is not None
        assert pattern['tile_height'] == 2
        assert pattern['tile_width'] == 2
