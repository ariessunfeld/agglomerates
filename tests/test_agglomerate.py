"""
Test suite for the fluffy-clumps agglomerate generator.
"""

import numpy as np
import pytest
import tempfile
import os
import struct

from agglomerate import (
    Nanorod,
    random_unit_vector,
    random_point_on_sphere,
    segment_segment_distance,
    min_distance_to_agglomerate,
    check_collision,
    sample_escape_angle,
    calculate_agglomerate_com,
    generate_agglomerate,
    create_cylinder_mesh,
    create_prism_mesh,
    create_rod_mesh,
    write_stl_binary,
    write_stl_ascii,
    calculate_fractal_dimension,
)


class TestNanorod:
    """Tests for the Nanorod dataclass."""

    def test_nanorod_creation(self):
        """Test basic nanorod creation."""
        rod = Nanorod(
            center=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
            length=100.0,
            diameter=10.0
        )
        assert rod.length == 100.0
        assert rod.diameter == 10.0
        np.testing.assert_array_equal(rod.center, [0, 0, 0])

    def test_nanorod_endpoints(self):
        """Test endpoint calculation."""
        rod = Nanorod(
            center=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
            length=100.0,
            diameter=10.0
        )
        np.testing.assert_array_almost_equal(rod.endpoint1, [0, 0, -50])
        np.testing.assert_array_almost_equal(rod.endpoint2, [0, 0, 50])

    def test_nanorod_endpoints_offset_center(self):
        """Test endpoints with offset center."""
        rod = Nanorod(
            center=np.array([10, 20, 30]),
            direction=np.array([1, 0, 0]),
            length=20.0,
            diameter=5.0
        )
        np.testing.assert_array_almost_equal(rod.endpoint1, [0, 20, 30])
        np.testing.assert_array_almost_equal(rod.endpoint2, [20, 20, 30])

    def test_nanorod_copy(self):
        """Test nanorod copy method."""
        rod = Nanorod(
            center=np.array([1, 2, 3]),
            direction=np.array([0, 1, 0]),
            length=50.0,
            diameter=8.0
        )
        rod_copy = rod.copy()

        # Verify copy has same values
        np.testing.assert_array_equal(rod_copy.center, rod.center)
        np.testing.assert_array_equal(rod_copy.direction, rod.direction)
        assert rod_copy.length == rod.length
        assert rod_copy.diameter == rod.diameter

        # Verify copy is independent
        rod_copy.center[0] = 999
        assert rod.center[0] == 1


class TestGeometryFunctions:
    """Tests for geometry utility functions."""

    def test_random_unit_vector_is_unit(self):
        """Test that random unit vectors have magnitude 1."""
        for _ in range(100):
            vec = random_unit_vector()
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-10

    def test_random_unit_vector_distribution(self):
        """Test that random vectors cover all directions."""
        np.random.seed(42)
        vectors = [random_unit_vector() for _ in range(1000)]

        # Check that we have vectors in all octants
        has_positive_x = any(v[0] > 0.5 for v in vectors)
        has_negative_x = any(v[0] < -0.5 for v in vectors)
        has_positive_y = any(v[1] > 0.5 for v in vectors)
        has_negative_y = any(v[1] < -0.5 for v in vectors)
        has_positive_z = any(v[2] > 0.5 for v in vectors)
        has_negative_z = any(v[2] < -0.5 for v in vectors)

        assert all([has_positive_x, has_negative_x, has_positive_y,
                    has_negative_y, has_positive_z, has_negative_z])

    def test_random_point_on_sphere_distance(self):
        """Test that random points are at correct radius."""
        center = np.array([10, 20, 30])
        radius = 50.0

        for _ in range(100):
            point = random_point_on_sphere(center, radius)
            dist = np.linalg.norm(point - center)
            assert abs(dist - radius) < 1e-10

    def test_segment_distance_parallel_segments(self):
        """Test distance between parallel segments."""
        # Two parallel segments along z-axis, separated in x
        dist = segment_segment_distance(
            p1=np.array([0, 0, 0]), d1=np.array([0, 0, 1]), len1=10,
            p2=np.array([5, 0, 0]), d2=np.array([0, 0, 1]), len2=10
        )
        assert abs(dist - 5.0) < 1e-10

    def test_segment_distance_perpendicular_segments(self):
        """Test distance between perpendicular non-intersecting segments."""
        # One along z, one along x, both at origin but offset
        dist = segment_segment_distance(
            p1=np.array([0, 0, 0]), d1=np.array([0, 0, 1]), len1=10,
            p2=np.array([0, 3, 0]), d2=np.array([1, 0, 0]), len2=10
        )
        assert abs(dist - 3.0) < 1e-10

    def test_segment_distance_intersecting(self):
        """Test distance between intersecting segments is zero."""
        dist = segment_segment_distance(
            p1=np.array([0, 0, 0]), d1=np.array([0, 0, 1]), len1=10,
            p2=np.array([0, 0, 0]), d2=np.array([1, 0, 0]), len2=10
        )
        assert dist < 1e-10


class TestCollisionDetection:
    """Tests for collision detection functions."""

    def test_min_distance_to_agglomerate(self):
        """Test minimum distance calculation to agglomerate."""
        rod1 = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        rod2 = Nanorod(np.array([20, 0, 0]), np.array([0, 0, 1]), 100, 10)

        agglomerate = [rod1]

        # Distance should be 20 (center-to-center) minus 5 (radius1) minus 5 (radius2) = 10
        dist = min_distance_to_agglomerate(rod2, agglomerate)
        assert abs(dist - 10.0) < 1e-10

    def test_check_collision_no_collision(self):
        """Test collision check when rods don't touch."""
        rod1 = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        rod2 = Nanorod(np.array([50, 0, 0]), np.array([0, 0, 1]), 100, 10)

        assert not check_collision(rod2, [rod1])

    def test_check_collision_with_collision(self):
        """Test collision check when rods touch."""
        rod1 = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        rod2 = Nanorod(np.array([10, 0, 0]), np.array([0, 0, 1]), 100, 10)

        assert check_collision(rod2, [rod1])


class TestEscapeAngle:
    """Tests for escape angle sampling."""

    def test_sample_escape_angle_range(self):
        """Test that sampled angles are in valid range."""
        np.random.seed(42)
        for p_esc in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(100):
                theta = sample_escape_angle(p_esc)
                assert 0 <= theta <= np.pi

    def test_sample_escape_angle_full_escape(self):
        """Test that p_esc >= 1 returns pi."""
        theta = sample_escape_angle(1.0)
        assert abs(theta - np.pi) < 1e-10


class TestAgglomerateGeneration:
    """Tests for agglomerate generation."""

    def test_generate_agglomerate_count(self):
        """Test that correct number of particles is generated."""
        np.random.seed(42)
        agglomerate = generate_agglomerate(
            num_particles=5,
            length=100,
            diameter=10,
            seed=42,
            verbose=False
        )
        assert len(agglomerate) == 5

    def test_generate_agglomerate_reproducibility(self):
        """Test that same seed produces same result."""
        agg1 = generate_agglomerate(5, 100, 10, seed=123, verbose=False)
        agg2 = generate_agglomerate(5, 100, 10, seed=123, verbose=False)

        for rod1, rod2 in zip(agg1, agg2):
            np.testing.assert_array_almost_equal(rod1.center, rod2.center)
            np.testing.assert_array_almost_equal(rod1.direction, rod2.direction)

    def test_generate_agglomerate_dimensions(self):
        """Test that rods have correct dimensions."""
        agglomerate = generate_agglomerate(3, length=50, diameter=8, seed=42, verbose=False)

        for rod in agglomerate:
            assert rod.length == 50
            assert rod.diameter == 8

    def test_calculate_agglomerate_com(self):
        """Test center of mass calculation."""
        rod1 = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        rod2 = Nanorod(np.array([10, 0, 0]), np.array([0, 0, 1]), 100, 10)
        rod3 = Nanorod(np.array([0, 10, 0]), np.array([0, 0, 1]), 100, 10)

        com = calculate_agglomerate_com([rod1, rod2, rod3])
        expected = np.array([10/3, 10/3, 0])
        np.testing.assert_array_almost_equal(com, expected)


class TestMeshGeneration:
    """Tests for mesh generation functions."""

    def test_cylinder_mesh_vertices_count(self):
        """Test cylinder mesh has correct vertex count."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        vertices, faces = create_cylinder_mesh(rod, num_segments=16)

        # 16 bottom + 16 top + 2 centers = 34 vertices
        assert len(vertices) == 34

    def test_cylinder_mesh_faces_count(self):
        """Test cylinder mesh has correct face count."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        vertices, faces = create_cylinder_mesh(rod, num_segments=16)

        # 16 bottom + 16 top + 32 sides = 64 faces
        assert len(faces) == 64

    def test_prism_mesh_vertices_count(self):
        """Test prism mesh has correct vertex count."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        vertices, faces = create_prism_mesh(rod)

        # 8 vertices for a rectangular prism
        assert len(vertices) == 8

    def test_prism_mesh_faces_count(self):
        """Test prism mesh has correct face count."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        vertices, faces = create_prism_mesh(rod)

        # 6 faces * 2 triangles = 12 faces
        assert len(faces) == 12

    def test_create_rod_mesh_cylinder(self):
        """Test create_rod_mesh with cylinder shape."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        vertices, faces = create_rod_mesh(rod, shape='cylinder', num_segments=8)

        # 8 + 8 + 2 = 18 vertices for 8-segment cylinder
        assert len(vertices) == 18

    def test_create_rod_mesh_prism(self):
        """Test create_rod_mesh with prism shape."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)
        vertices, faces = create_rod_mesh(rod, shape='prism')

        assert len(vertices) == 8
        assert len(faces) == 12


class TestSTLOutput:
    """Tests for STL file writing."""

    def test_write_stl_binary_creates_file(self):
        """Test that binary STL file is created."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            write_stl_binary(filename, [rod])
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            os.unlink(filename)

    def test_write_stl_binary_format(self):
        """Test binary STL file format."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            write_stl_binary(filename, [rod], shape='prism')

            with open(filename, 'rb') as f:
                header = f.read(80)
                num_triangles = struct.unpack('<I', f.read(4))[0]

            # Prism has 12 triangles
            assert num_triangles == 12
        finally:
            os.unlink(filename)

    def test_write_stl_ascii_creates_file(self):
        """Test that ASCII STL file is created."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            write_stl_ascii(filename, [rod])
            assert os.path.exists(filename)

            with open(filename, 'r') as f:
                content = f.read()

            assert content.startswith('solid agglomerate')
            assert 'endsolid agglomerate' in content
            assert 'facet normal' in content
        finally:
            os.unlink(filename)

    def test_write_stl_with_prism_shape(self):
        """Test STL output with prism shape."""
        rod = Nanorod(np.array([0, 0, 0]), np.array([0, 0, 1]), 100, 10)

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            write_stl_binary(filename, [rod], shape='prism')

            with open(filename, 'rb') as f:
                f.read(80)  # header
                num_triangles = struct.unpack('<I', f.read(4))[0]

            assert num_triangles == 12  # 6 faces * 2 triangles
        finally:
            os.unlink(filename)


class TestFractalDimension:
    """Tests for fractal dimension calculation."""

    def test_fractal_dimension_range(self):
        """Test that fractal dimension is in reasonable range."""
        agglomerate = generate_agglomerate(10, 100, 10, seed=42, verbose=False)
        fd = calculate_fractal_dimension(agglomerate)

        # Fractal dimension should be between 1 (line) and 3 (solid)
        assert 1.0 <= fd <= 3.0

    def test_fractal_dimension_reproducibility(self):
        """Test fractal dimension calculation is reproducible."""
        agglomerate = generate_agglomerate(10, 100, 10, seed=42, verbose=False)

        np.random.seed(123)
        fd1 = calculate_fractal_dimension(agglomerate, num_samples=500)

        np.random.seed(123)
        fd2 = calculate_fractal_dimension(agglomerate, num_samples=500)

        assert abs(fd1 - fd2) < 1e-10


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_cylinder(self):
        """Test complete workflow with cylinder shape."""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            agglomerate = generate_agglomerate(5, 100, 10, seed=42, verbose=False)
            write_stl_binary(filename, agglomerate, shape='cylinder', num_segments=8)

            assert os.path.exists(filename)
            file_size = os.path.getsize(filename)

            # 5 rods * (8*4 triangles) = 160 triangles
            # Binary STL: 80 header + 4 count + 160 * 50 bytes = 8084 bytes
            expected_size = 80 + 4 + 160 * 50
            assert file_size == expected_size
        finally:
            os.unlink(filename)

    def test_full_workflow_prism(self):
        """Test complete workflow with prism shape."""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            agglomerate = generate_agglomerate(5, 100, 10, seed=42, verbose=False)
            write_stl_binary(filename, agglomerate, shape='prism')

            assert os.path.exists(filename)
            file_size = os.path.getsize(filename)

            # 5 rods * 12 triangles = 60 triangles
            # Binary STL: 80 header + 4 count + 60 * 50 bytes = 3084 bytes
            expected_size = 80 + 4 + 60 * 50
            assert file_size == expected_size
        finally:
            os.unlink(filename)
