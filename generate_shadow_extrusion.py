"""
Generate extruded shadow projections from STL files.

This script creates "solid shadows" by:
1. Loading an STL mesh
2. Projecting it onto a 2D plane from a specified viewing angle
3. Extruding that 2D silhouette into a 3D solid of specified thickness
4. Saving the result as a new STL file

Supports finding minimal, maximal, and random projection orientations.
"""

import os
import struct
import argparse
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


# =============================================================================
# STL File I/O (adapted from analyze_batch_paper_method.py)
# =============================================================================

def parse_stl_binary(filepath: str) -> np.ndarray:
    """Parse a binary STL file and return triangle vertices."""
    triangles = []
    with open(filepath, 'rb') as f:
        f.read(80)  # Skip header
        n_triangles = struct.unpack('<I', f.read(4))[0]
        for _ in range(n_triangles):
            f.read(12)  # Skip normal
            vertices = []
            for _ in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                vertices.append([x, y, z])
            triangles.append(vertices)
            f.read(2)  # Skip attribute
    return np.array(triangles)


def parse_stl_ascii(filepath: str) -> np.ndarray:
    """Parse an ASCII STL file and return triangle vertices."""
    triangles = []
    current_vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith('vertex'):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_vertices.append(vertex)
                if len(current_vertices) == 3:
                    triangles.append(current_vertices)
                    current_vertices = []
    return np.array(triangles)


def parse_stl(filepath: str) -> np.ndarray:
    """Parse an STL file (auto-detect binary vs ASCII)."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
    try:
        header_text = header.decode('ascii').strip().lower()
        if header_text.startswith('solid'):
            with open(filepath, 'r') as f:
                content = f.read(1000)
                if 'vertex' in content.lower():
                    return parse_stl_ascii(filepath)
    except:
        pass
    return parse_stl_binary(filepath)


def write_stl_binary(filename: str, triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    """
    Write triangles to a binary STL file.

    Args:
        filename: Output filename
        triangles: List of (normal, v0, v1, v2) tuples
    """
    with open(filename, 'wb') as f:
        # 80-byte header
        header = f"Shadow extrusion generated {datetime.now().isoformat()}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        # Write each triangle
        for normal, v0, v1, v2 in triangles:
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))  # Attribute byte count


def calculate_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Calculate the normal vector for a triangle."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        return normal / norm
    return np.array([0.0, 0.0, 1.0])


# =============================================================================
# Rotation Matrix Generation
# =============================================================================

def random_rotation_matrix() -> np.ndarray:
    """Generate a random 3D rotation matrix (uniform on SO(3))."""
    random_matrix = np.random.randn(3, 3)
    q, r = np.linalg.qr(random_matrix)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rotation_matrix_from_angles(theta: float, phi: float) -> np.ndarray:
    """
    Generate a rotation matrix from spherical angles.

    Args:
        theta: Polar angle (0 to pi) - angle from Z axis
        phi: Azimuthal angle (0 to 2*pi) - angle in XY plane from X axis

    Returns:
        3x3 rotation matrix that rotates the viewing direction to align with Z axis
    """
    # The viewing direction in spherical coordinates
    view_dir = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    # We want to rotate so that view_dir becomes the Z axis
    # This means the projection plane (XY) will be perpendicular to view_dir
    z_axis = np.array([0.0, 0.0, 1.0])

    if np.allclose(view_dir, z_axis):
        return np.eye(3)
    if np.allclose(view_dir, -z_axis):
        return np.diag([1.0, -1.0, -1.0])

    # Rotation axis is perpendicular to both
    axis = np.cross(view_dir, z_axis)
    axis = axis / np.linalg.norm(axis)

    # Rotation angle
    angle = np.arccos(np.clip(np.dot(view_dir, z_axis), -1, 1))

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R


# =============================================================================
# Projection and Area Calculation
# =============================================================================

def project_mesh_to_2d(triangles: np.ndarray, rotation_matrix: np.ndarray,
                       resolution: int = 500) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Project a 3D mesh to a 2D binary image.

    Returns:
        Tuple of (binary_image, pixel_size, grid_origin)
    """
    # Rotate all vertices
    all_vertices = triangles.reshape(-1, 3)
    rotated = all_vertices @ rotation_matrix.T

    # Project to XY (drop Z)
    projected_2d = rotated[:, :2]

    # Find bounding box
    min_coords = projected_2d.min(axis=0)
    max_coords = projected_2d.max(axis=0)

    # Make it a square with some padding
    extent = max_coords - min_coords
    max_extent = max(extent) * 1.05  # 5% padding

    center = (min_coords + max_coords) / 2
    grid_origin = center - max_extent / 2

    pixel_size = max_extent / resolution

    # Rasterize triangles
    image = np.zeros((resolution, resolution), dtype=bool)

    rotated_triangles = triangles @ rotation_matrix.T

    for tri in rotated_triangles:
        tri_2d = tri[:, :2]
        _rasterize_triangle(tri_2d, grid_origin, pixel_size, resolution, image)

    return image, pixel_size, grid_origin


def _rasterize_triangle(vertices_2d: np.ndarray, grid_origin: np.ndarray,
                        pixel_size: float, grid_size: int, image: np.ndarray):
    """Rasterize a 2D triangle onto a binary grid (in-place)."""
    # Convert to pixel coordinates
    pixels = ((vertices_2d - grid_origin) / pixel_size).astype(int)

    min_x = max(0, min(pixels[:, 0]))
    max_x = min(grid_size - 1, max(pixels[:, 0]))
    min_y = max(0, min(pixels[:, 1]))
    max_y = min(grid_size - 1, max(pixels[:, 1]))

    v0, v1, v2 = vertices_2d

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = grid_origin + np.array([x + 0.5, y + 0.5]) * pixel_size
            d1 = sign(p, v0, v1)
            d2 = sign(p, v1, v2)
            d3 = sign(p, v2, v0)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            if not (has_neg and has_pos):
                image[y, x] = True


def calculate_projection_area(triangles: np.ndarray, rotation_matrix: np.ndarray,
                              resolution: int = 300) -> float:
    """Calculate the projected area of a mesh from a given orientation."""
    image, pixel_size, _ = project_mesh_to_2d(triangles, rotation_matrix, resolution)
    return np.sum(image) * (pixel_size ** 2)


# =============================================================================
# Minimum Bounding Rectangle (Rotating Calipers)
# =============================================================================

def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of 2D points using Graham scan.

    Args:
        points: Array of shape (n, 2)

    Returns:
        Array of hull vertices in counter-clockwise order
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(map(tuple, points))
    if len(points) <= 1:
        return np.array(points)

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])


def minimum_bounding_rectangle(points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Find the minimum area bounding rectangle for a set of 2D points.
    Uses rotating calipers algorithm on the convex hull.

    Args:
        points: Array of shape (n, 2)

    Returns:
        Tuple of (rectangle_corners, width, height, rotation_angle)
        - rectangle_corners: 4 corners of the bounding rectangle
        - width: smaller dimension
        - height: larger dimension
        - rotation_angle: angle to rotate points so bbox is axis-aligned (radians)
    """
    hull = convex_hull_2d(points)

    if len(hull) < 3:
        # Degenerate case
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        corners = np.array([
            [min_pt[0], min_pt[1]],
            [max_pt[0], min_pt[1]],
            [max_pt[0], max_pt[1]],
            [min_pt[0], max_pt[1]]
        ])
        return corners, min(width, height), max(width, height), 0.0

    # Get edges of convex hull
    edges = np.diff(np.vstack([hull, hull[0]]), axis=0)

    # Get angles of edges
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.unique(np.mod(angles, np.pi / 2))  # Only need 0-90 degrees

    min_area = float('inf')
    best_angle = 0
    best_bbox = None

    for angle in angles:
        # Rotation matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        # Rotate hull
        rotated = hull @ R.T

        # Axis-aligned bounding box
        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)

        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]
        area = width * height

        if area < min_area:
            min_area = area
            best_angle = angle
            best_bbox = (min_xy, max_xy, width, height)

    # Compute corners of best bounding box
    min_xy, max_xy, width, height = best_bbox
    corners_rotated = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]]
    ])

    # Rotate back to original frame
    cos_a, sin_a = np.cos(-best_angle), np.sin(-best_angle)
    R_inv = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    corners = corners_rotated @ R_inv.T

    # Return with width as smaller dimension
    if width > height:
        width, height = height, width

    return corners, width, height, best_angle


def calculate_bounding_box_dimensions(triangles: np.ndarray, rotation_matrix: np.ndarray,
                                       resolution: int = 100) -> Tuple[float, float, float, float]:
    """
    Calculate the minimum bounding rectangle dimensions for a projected mesh.

    Returns:
        Tuple of (width, height, bbox_area, rotation_angle_2d)
    """
    image, pixel_size, grid_origin = project_mesh_to_2d(triangles, rotation_matrix, resolution)

    # Get filled pixel coordinates
    filled_pixels = np.argwhere(image)
    if len(filled_pixels) == 0:
        return 0, 0, 0, 0

    # Convert to world coordinates
    points_2d = grid_origin + (filled_pixels[:, ::-1] + 0.5) * pixel_size

    # Find minimum bounding rectangle
    _, width, height, angle = minimum_bounding_rectangle(points_2d)

    return width, height, width * height, angle


# =============================================================================
# PCA-Based Heuristics for Optimal Projection Search
# =============================================================================

def compute_principal_axes(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal axes of the mesh using PCA on vertex positions.

    Args:
        triangles: Array of shape (n_triangles, 3, 3) with vertex coordinates

    Returns:
        Tuple of (principal_axes, eigenvalues)
        - principal_axes: 3x3 array where each row is a principal axis (sorted by eigenvalue, descending)
        - eigenvalues: Array of 3 eigenvalues (sorted descending)
    """
    # Get all unique vertices
    vertices = triangles.reshape(-1, 3)

    # Center the vertices
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Each column is an eigenvector, transpose so each row is a principal axis
    principal_axes = eigenvectors.T

    return principal_axes, eigenvalues


def rotation_matrix_align_axis_to_z(axis: np.ndarray) -> np.ndarray:
    """
    Create a rotation matrix that aligns the given axis with the Z axis.

    This means viewing along the given axis direction.

    Args:
        axis: Unit vector to align with Z

    Returns:
        3x3 rotation matrix
    """
    axis = axis / np.linalg.norm(axis)
    z_axis = np.array([0.0, 0.0, 1.0])

    if np.allclose(axis, z_axis):
        return np.eye(3)
    if np.allclose(axis, -z_axis):
        return np.diag([1.0, -1.0, -1.0])

    # Rotation axis is perpendicular to both
    rot_axis = np.cross(axis, z_axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # Rotation angle
    angle = np.arccos(np.clip(np.dot(axis, z_axis), -1, 1))

    # Rodrigues' rotation formula
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R


def generate_pca_heuristic_rotations(triangles: np.ndarray) -> List[np.ndarray]:
    """
    Generate rotation matrices based on principal axes of the mesh.

    For elongated structures, viewing along the primary axis gives minimum projection,
    viewing perpendicular to it gives maximum projection.

    Args:
        triangles: Mesh triangles

    Returns:
        List of rotation matrices to test
    """
    principal_axes, eigenvalues = compute_principal_axes(triangles)

    rotations = []

    # For each principal axis, create rotations that view along that axis
    # (both positive and negative directions)
    for i in range(3):
        axis = principal_axes[i]
        rotations.append(rotation_matrix_align_axis_to_z(axis))
        rotations.append(rotation_matrix_align_axis_to_z(-axis))

    # Also add rotations viewing along cross-products of principal axes
    # These can catch intermediate optimal views
    for i in range(3):
        for j in range(i + 1, 3):
            cross = np.cross(principal_axes[i], principal_axes[j])
            if np.linalg.norm(cross) > 1e-6:
                cross = cross / np.linalg.norm(cross)
                rotations.append(rotation_matrix_align_axis_to_z(cross))
                rotations.append(rotation_matrix_align_axis_to_z(-cross))

    return rotations


# =============================================================================
# Find Optimal Projections
# =============================================================================

def find_min_max_projections(triangles: np.ndarray, n_samples: int = 500,
                             resolution: int = 200) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Find the rotation matrices that give minimal and maximal bounding box areas.

    Uses PCA-based heuristics to seed the search, then adds random sampling.
    For simple elongated structures (like single rods), the PCA heuristics
    will find the exact optimal orientations.

    Args:
        triangles: Mesh triangles
        n_samples: Number of random orientations to sample (in addition to PCA heuristics)
        resolution: Resolution for bbox calculation

    Returns:
        Tuple of (min_rotation, max_rotation, min_info, max_info)
        where info dicts contain 'width', 'height', 'bbox_area', 'angle_2d'
    """
    min_bbox_area = float('inf')
    max_bbox_area = 0
    min_rotation = None
    max_rotation = None
    min_info = {}
    max_info = {}

    def evaluate_rotation(rotation):
        nonlocal min_bbox_area, max_bbox_area, min_rotation, max_rotation, min_info, max_info

        width, height, bbox_area, angle_2d = calculate_bounding_box_dimensions(
            triangles, rotation, resolution
        )

        if bbox_area > 0 and bbox_area < min_bbox_area:
            min_bbox_area = bbox_area
            min_rotation = rotation.copy()
            min_info = {'width': width, 'height': height, 'bbox_area': bbox_area, 'angle_2d': angle_2d}

        if bbox_area > max_bbox_area:
            max_bbox_area = bbox_area
            max_rotation = rotation.copy()
            max_info = {'width': width, 'height': height, 'bbox_area': bbox_area, 'angle_2d': angle_2d}

    # First, evaluate PCA-based heuristic rotations
    pca_rotations = generate_pca_heuristic_rotations(triangles)
    for rotation in pca_rotations:
        evaluate_rotation(rotation)

    # Then add random sampling
    for _ in range(n_samples):
        rotation = random_rotation_matrix()
        evaluate_rotation(rotation)

    return min_rotation, max_rotation, min_info, max_info


# =============================================================================
# Contour Extraction from Binary Image
# =============================================================================

def extract_contour_pixels(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract boundary pixels from a binary image using simple edge detection.
    Returns pixels that are on the boundary (have at least one empty neighbor).
    """
    rows, cols = image.shape
    boundary = []

    for y in range(rows):
        for x in range(cols):
            if image[y, x]:
                # Check if this is a boundary pixel
                is_boundary = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= rows or nx < 0 or nx >= cols or not image[ny, nx]:
                        is_boundary = True
                        break
                if is_boundary:
                    boundary.append((x, y))

    return boundary


def trace_contour(image: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Trace contours in a binary image using a simple boundary following algorithm.
    Returns a list of contours, where each contour is a list of (x, y) pixel coordinates.
    """
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)
    contours = []

    # Direction vectors for 8-connectivity (clockwise starting from right)
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def is_boundary_pixel(y, x):
        if not image[y, x]:
            return False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= rows or nx < 0 or nx >= cols or not image[ny, nx]:
                return True
        return False

    for start_y in range(rows):
        for start_x in range(cols):
            if image[start_y, start_x] and not visited[start_y, start_x] and is_boundary_pixel(start_y, start_x):
                # Start a new contour
                contour = []
                x, y = start_x, start_y

                # Find initial direction (pointing to background)
                start_dir = 0
                for i, (dx, dy) in enumerate(directions):
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= cols or ny < 0 or ny >= rows or not image[ny, nx]:
                        start_dir = i
                        break

                current_dir = start_dir

                while True:
                    contour.append((x, y))
                    visited[y, x] = True

                    # Look for next boundary pixel (turn clockwise)
                    found = False
                    search_dir = (current_dir + 5) % 8  # Start looking from behind-left

                    for _ in range(8):
                        dx, dy = directions[search_dir]
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < cols and 0 <= ny < rows and image[ny, nx]:
                            x, y = nx, ny
                            current_dir = search_dir
                            found = True
                            break

                        search_dir = (search_dir + 1) % 8

                    if not found or (x == start_x and y == start_y):
                        break

                    if len(contour) > rows * cols:  # Safety limit
                        break

                if len(contour) >= 3:
                    contours.append(contour)

    return contours


def simplify_contour(contour: List[Tuple[int, int]], tolerance: float = 1.5) -> List[Tuple[int, int]]:
    """
    Simplify a contour using the Ramer-Douglas-Peucker algorithm.
    """
    if len(contour) < 3:
        return contour

    def point_line_distance(point, line_start, line_end):
        if np.allclose(line_start, line_end):
            return np.linalg.norm(np.array(point) - np.array(line_start))

        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len = np.linalg.norm(line_vec)
        line_unit = line_vec / line_len
        proj_length = np.dot(point_vec, line_unit)
        proj_length = max(0, min(line_len, proj_length))
        proj_point = np.array(line_start) + proj_length * line_unit
        return np.linalg.norm(np.array(point) - proj_point)

    # Find the point with maximum distance from the line
    start, end = contour[0], contour[-1]
    max_dist = 0
    max_idx = 0

    for i in range(1, len(contour) - 1):
        dist = point_line_distance(contour[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > tolerance:
        # Recursive simplification
        left = simplify_contour(contour[:max_idx + 1], tolerance)
        right = simplify_contour(contour[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [start, end]


def pixels_to_world(contour: List[Tuple[int, int]], grid_origin: np.ndarray,
                    pixel_size: float) -> np.ndarray:
    """Convert pixel coordinates to world coordinates."""
    points = []
    for x, y in contour:
        world_x = grid_origin[0] + (x + 0.5) * pixel_size
        world_y = grid_origin[1] + (y + 0.5) * pixel_size
        points.append([world_x, world_y])
    return np.array(points)


# =============================================================================
# Triangulation (Ear Clipping)
# =============================================================================

def is_convex(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    """Check if angle at p2 is convex (counter-clockwise)."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0


def point_in_triangle(p: np.ndarray, t1: np.ndarray, t2: np.ndarray, t3: np.ndarray) -> bool:
    """Check if point p is inside triangle (t1, t2, t3)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, t1, t2)
    d2 = sign(p, t2, t3)
    d3 = sign(p, t3, t1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def triangulate_polygon(vertices: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Triangulate a simple polygon using ear clipping algorithm.

    Args:
        vertices: Array of shape (n, 2) with polygon vertices in order

    Returns:
        List of (i, j, k) index tuples representing triangles
    """
    n = len(vertices)
    if n < 3:
        return []

    # Ensure counter-clockwise orientation
    signed_area = 0
    for i in range(n):
        j = (i + 1) % n
        signed_area += vertices[i][0] * vertices[j][1]
        signed_area -= vertices[j][0] * vertices[i][1]

    if signed_area < 0:
        vertices = vertices[::-1]

    indices = list(range(n))
    triangles = []

    while len(indices) > 3:
        ear_found = False

        for i in range(len(indices)):
            i_prev = (i - 1) % len(indices)
            i_next = (i + 1) % len(indices)

            p_prev = vertices[indices[i_prev]]
            p_curr = vertices[indices[i]]
            p_next = vertices[indices[i_next]]

            # Check if this is a convex vertex (potential ear)
            if not is_convex(p_prev, p_curr, p_next):
                continue

            # Check if any other vertex is inside the triangle
            is_ear = True
            for j in range(len(indices)):
                if j in [i_prev, i, i_next]:
                    continue
                if point_in_triangle(vertices[indices[j]], p_prev, p_curr, p_next):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((indices[i_prev], indices[i], indices[i_next]))
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            # Fallback: just take any triangle (may produce degenerate results)
            if len(indices) >= 3:
                triangles.append((indices[0], indices[1], indices[2]))
                indices.pop(1)

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles


# =============================================================================
# Mesh Generation from Contour
# =============================================================================

def create_extruded_mesh(contour_2d: np.ndarray, thickness: float,
                         rotation_matrix: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create an extruded 3D mesh from a 2D contour.

    Args:
        contour_2d: Array of shape (n, 2) with 2D contour vertices
        thickness: Extrusion thickness in the Z direction (in rotated frame)
        rotation_matrix: The rotation matrix used for projection (to transform back)

    Returns:
        List of (normal, v0, v1, v2) tuples for STL output
    """
    triangles = []
    n = len(contour_2d)

    if n < 3:
        return triangles

    # Create 3D vertices in the rotated frame
    # Top face at z=0, bottom face at z=-thickness
    top_vertices = np.column_stack([contour_2d, np.zeros(n)])
    bottom_vertices = np.column_stack([contour_2d, np.full(n, -thickness)])

    # Transform back to original coordinate system
    inv_rotation = rotation_matrix.T
    top_vertices_world = top_vertices @ inv_rotation.T
    bottom_vertices_world = bottom_vertices @ inv_rotation.T

    # Triangulate the 2D contour for top and bottom faces
    tri_indices = triangulate_polygon(contour_2d)

    # Top face triangles (normal pointing up in rotated frame = along view direction)
    for i, j, k in tri_indices:
        v0 = top_vertices_world[i]
        v1 = top_vertices_world[j]
        v2 = top_vertices_world[k]
        normal = calculate_normal(v0, v1, v2)
        triangles.append((normal, v0, v1, v2))

    # Bottom face triangles (reversed winding for outward normal)
    for i, j, k in tri_indices:
        v0 = bottom_vertices_world[i]
        v1 = bottom_vertices_world[k]  # Reversed
        v2 = bottom_vertices_world[j]  # Reversed
        normal = calculate_normal(v0, v1, v2)
        triangles.append((normal, v0, v1, v2))

    # Side wall triangles
    for i in range(n):
        i_next = (i + 1) % n

        # Four corners of this wall segment
        t0 = top_vertices_world[i]
        t1 = top_vertices_world[i_next]
        b0 = bottom_vertices_world[i]
        b1 = bottom_vertices_world[i_next]

        # Two triangles per wall segment
        normal1 = calculate_normal(t0, b0, t1)
        triangles.append((normal1, t0, b0, t1))

        normal2 = calculate_normal(t1, b0, b1)
        triangles.append((normal2, t1, b0, b1))

    return triangles


def create_extruded_mesh_from_image(image: np.ndarray, pixel_size: float,
                                     grid_origin: np.ndarray, thickness: float,
                                     rotation_matrix: np.ndarray,
                                     simplify_tolerance: float = 2.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create an extruded mesh from a binary image.

    This extracts contours from the image and creates a 3D extruded mesh.
    """
    all_triangles = []

    # Trace contours in the image
    contours = trace_contour(image)

    if not contours:
        print("  Warning: No contours found in projection")
        return all_triangles

    # Process each contour
    for i, contour in enumerate(contours):
        # Simplify the contour
        simplified = simplify_contour(contour, simplify_tolerance)

        if len(simplified) < 3:
            continue

        # Convert to world coordinates
        contour_2d = pixels_to_world(simplified, grid_origin, pixel_size)

        # Create extruded mesh
        mesh_triangles = create_extruded_mesh(contour_2d, thickness, rotation_matrix)
        all_triangles.extend(mesh_triangles)

    return all_triangles


# =============================================================================
# Alternative: Voxel-based Extrusion (more robust)
# =============================================================================

def create_voxel_extruded_mesh_flat(image: np.ndarray, pixel_size: float,
                                     grid_origin: np.ndarray, thickness: float,
                                     angle_2d: float = 0.0,
                                     center_output: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create an extruded mesh using a voxel-based approach, output flat in XY plane.

    Each filled pixel becomes a small rectangular prism. The output is oriented
    flat in the XY plane (Z from 0 to thickness) with optional 2D rotation to
    align the minimum bounding box with the axes.

    Args:
        image: Binary image of the projection
        pixel_size: Size of each pixel in nm
        grid_origin: Origin of the grid in 2D
        thickness: Extrusion thickness in nm (height in Z)
        angle_2d: Rotation angle to apply in 2D for axis-aligned bounding box
        center_output: If True, center the output at origin in XY

    Returns:
        List of (normal, v0, v1, v2) tuples for STL output
    """
    triangles = []
    rows, cols = image.shape

    # 2D rotation matrix for axis alignment
    cos_a, sin_a = np.cos(angle_2d), np.sin(angle_2d)

    # Collect all filled pixel coordinates first (for centering)
    filled_coords = []
    for y in range(rows):
        for x in range(cols):
            if image[y, x]:
                # Original 2D coordinates
                x_center = grid_origin[0] + (x + 0.5) * pixel_size
                y_center = grid_origin[1] + (y + 0.5) * pixel_size
                # Apply 2D rotation
                x_rot = cos_a * x_center + sin_a * y_center
                y_rot = -sin_a * x_center + cos_a * y_center
                filled_coords.append((x, y, x_rot, y_rot))

    if not filled_coords:
        return triangles

    # Calculate center offset if centering is requested
    if center_output and filled_coords:
        x_rots = [c[2] for c in filled_coords]
        y_rots = [c[3] for c in filled_coords]
        center_x = (min(x_rots) + max(x_rots)) / 2
        center_y = (min(y_rots) + max(y_rots)) / 2
    else:
        center_x, center_y = 0, 0

    # Generate mesh for each filled pixel
    for x, y, x_rot, y_rot in filled_coords:
        # Check neighbors for face visibility
        left_empty = x == 0 or not image[y, x - 1]
        right_empty = x == cols - 1 or not image[y, x + 1]
        bottom_empty = y == 0 or not image[y - 1, x]
        top_empty = y == rows - 1 or not image[y + 1, x]

        # Pixel corner coordinates in original 2D frame
        x_min_orig = grid_origin[0] + x * pixel_size
        x_max_orig = grid_origin[0] + (x + 1) * pixel_size
        y_min_orig = grid_origin[1] + y * pixel_size
        y_max_orig = grid_origin[1] + (y + 1) * pixel_size

        # Rotate and center each corner
        def transform_2d(px, py):
            rx = cos_a * px + sin_a * py - center_x
            ry = -sin_a * px + cos_a * py - center_y
            return rx, ry

        # 4 corners in rotated 2D
        c00 = transform_2d(x_min_orig, y_min_orig)
        c10 = transform_2d(x_max_orig, y_min_orig)
        c11 = transform_2d(x_max_orig, y_max_orig)
        c01 = transform_2d(x_min_orig, y_max_orig)

        # 8 corners of the voxel in 3D (flat in XY plane)
        # Bottom at Z=0, top at Z=thickness
        corners = np.array([
            [c00[0], c00[1], 0],           # 0: bottom, min-min
            [c10[0], c10[1], 0],           # 1: bottom, max-min
            [c11[0], c11[1], 0],           # 2: bottom, max-max
            [c01[0], c01[1], 0],           # 3: bottom, min-max
            [c00[0], c00[1], thickness],   # 4: top, min-min
            [c10[0], c10[1], thickness],   # 5: top, max-min
            [c11[0], c11[1], thickness],   # 6: top, max-max
            [c01[0], c01[1], thickness],   # 7: top, min-max
        ])

        # Top face (Z = thickness, always visible)
        n = calculate_normal(corners[4], corners[5], corners[6])
        triangles.append((n, corners[4], corners[5], corners[6]))
        n = calculate_normal(corners[4], corners[6], corners[7])
        triangles.append((n, corners[4], corners[6], corners[7]))

        # Bottom face (Z = 0, always visible)
        n = calculate_normal(corners[0], corners[3], corners[2])
        triangles.append((n, corners[0], corners[3], corners[2]))
        n = calculate_normal(corners[0], corners[2], corners[1])
        triangles.append((n, corners[0], corners[2], corners[1]))

        # Side faces (only if exposed)
        if left_empty:  # -X face (original frame)
            n = calculate_normal(corners[0], corners[4], corners[7])
            triangles.append((n, corners[0], corners[4], corners[7]))
            n = calculate_normal(corners[0], corners[7], corners[3])
            triangles.append((n, corners[0], corners[7], corners[3]))

        if right_empty:  # +X face (original frame)
            n = calculate_normal(corners[1], corners[2], corners[6])
            triangles.append((n, corners[1], corners[2], corners[6]))
            n = calculate_normal(corners[1], corners[6], corners[5])
            triangles.append((n, corners[1], corners[6], corners[5]))

        if bottom_empty:  # -Y face (original frame)
            n = calculate_normal(corners[0], corners[1], corners[5])
            triangles.append((n, corners[0], corners[1], corners[5]))
            n = calculate_normal(corners[0], corners[5], corners[4])
            triangles.append((n, corners[0], corners[5], corners[4]))

        if top_empty:  # +Y face (original frame)
            n = calculate_normal(corners[3], corners[7], corners[6])
            triangles.append((n, corners[3], corners[7], corners[6]))
            n = calculate_normal(corners[3], corners[6], corners[2])
            triangles.append((n, corners[3], corners[6], corners[2]))

    return triangles


# Legacy function for backwards compatibility
def create_voxel_extruded_mesh(image: np.ndarray, pixel_size: float,
                                grid_origin: np.ndarray, thickness: float,
                                rotation_matrix: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create an extruded mesh using a voxel-based approach (legacy, tilted output).
    """
    return create_voxel_extruded_mesh_flat(image, pixel_size, grid_origin, thickness,
                                            angle_2d=0.0, center_output=False)


# =============================================================================
# Main Processing Functions
# =============================================================================

def generate_shadow_extrusion(input_stl: str, output_stl: str,
                               rotation_matrix: np.ndarray,
                               thickness: float = 160.0,
                               resolution: int = 500,
                               angle_2d: float = 0.0,
                               method: str = 'voxel') -> dict:
    """
    Generate an extruded shadow projection from an STL file.

    Output is flat in the XY plane with Z from 0 to thickness.

    Args:
        input_stl: Path to input STL file
        output_stl: Path to output STL file
        rotation_matrix: 3x3 rotation matrix for viewing angle
        thickness: Extrusion thickness in nm
        resolution: Resolution for projection rasterization
        angle_2d: 2D rotation angle for axis-aligned bounding box
        method: 'voxel' or 'contour'

    Returns:
        Dictionary with statistics about the generated mesh
    """
    # Load the STL
    triangles = parse_stl(input_stl)

    # Project to 2D
    image, pixel_size, grid_origin = project_mesh_to_2d(triangles, rotation_matrix, resolution)

    # Calculate projection area and bounding box
    projection_area = np.sum(image) * (pixel_size ** 2)

    # Get bounding box dimensions
    filled_pixels = np.argwhere(image)
    if len(filled_pixels) > 0:
        points_2d = grid_origin + (filled_pixels[:, ::-1] + 0.5) * pixel_size
        _, width, height, _ = minimum_bounding_rectangle(points_2d)
    else:
        width, height = 0, 0

    # Create extruded mesh (flat in XY plane)
    if method == 'voxel':
        mesh_triangles = create_voxel_extruded_mesh_flat(
            image, pixel_size, grid_origin, thickness, angle_2d, center_output=True
        )
    else:
        # Contour method not updated for flat output yet, use voxel
        mesh_triangles = create_voxel_extruded_mesh_flat(
            image, pixel_size, grid_origin, thickness, angle_2d, center_output=True
        )

    # Write output STL
    write_stl_binary(output_stl, mesh_triangles)

    return {
        'input_file': input_stl,
        'output_file': output_stl,
        'projection_area': projection_area,
        'bbox_width': width,
        'bbox_height': height,
        'bbox_area': width * height,
        'num_triangles': len(mesh_triangles),
        'thickness': thickness,
        'resolution': resolution,
        'pixel_size': pixel_size
    }


def process_stl_file(input_stl: str, output_dir: str, thickness: float = 160.0,
                     resolution: int = 300, n_search_samples: int = 200,
                     method: str = 'voxel', quiet: bool = False) -> List[dict]:
    """
    Process a single STL file, generating min, max, and random projections.

    Optimization is based on minimum bounding box area (not silhouette area).
    Output shadows are flat in the XY plane, axis-aligned, and centered at origin.

    Args:
        input_stl: Path to input STL file
        output_dir: Directory for output files
        thickness: Extrusion thickness in nm (Z dimension)
        resolution: Resolution for final projection (search uses lower)
        n_search_samples: Number of samples for min/max search
        method: 'voxel' or 'contour'
        quiet: Suppress progress output

    Returns:
        List of result dictionaries for each projection type
    """
    if not quiet:
        print(f"\nProcessing: {os.path.basename(input_stl)}")

    # Load the STL
    triangles = parse_stl(input_stl)
    if not quiet:
        print(f"  Loaded {len(triangles)} triangles")

    # Find min and max projections based on bounding box area
    search_resolution = min(100, resolution // 3)
    if not quiet:
        print(f"  Searching for optimal projections ({n_search_samples} samples at res {search_resolution})...")

    min_rot, max_rot, min_info, max_info = find_min_max_projections(
        triangles, n_search_samples, resolution=search_resolution
    )

    if not quiet:
        print(f"  Min bounding box: {min_info['width']:.1f} x {min_info['height']:.1f} nm "
              f"(area: {min_info['bbox_area']:.0f} nm^2)")
        print(f"  Max bounding box: {max_info['width']:.1f} x {max_info['height']:.1f} nm "
              f"(area: {max_info['bbox_area']:.0f} nm^2)")
        print(f"  Area ratio (max/min): {max_info['bbox_area']/min_info['bbox_area']:.2f}")

    # Generate a random projection for comparison
    random_rot = random_rotation_matrix()
    random_width, random_height, random_bbox_area, random_angle = calculate_bounding_box_dimensions(
        triangles, random_rot, search_resolution
    )
    random_info = {'width': random_width, 'height': random_height,
                   'bbox_area': random_bbox_area, 'angle_2d': random_angle}

    results = []
    base_name = os.path.splitext(os.path.basename(input_stl))[0]

    # Process each projection type
    projections = [
        ('min', min_rot, min_info, 'minimal'),
        ('max', max_rot, max_info, 'maximal'),
        ('random', random_rot, random_info, 'random')
    ]

    for suffix, rotation, info, label in projections:
        output_file = os.path.join(output_dir, f"{base_name}_shadow_{suffix}.stl")

        if not quiet:
            print(f"  Generating {label} projection shadow...")

        result = generate_shadow_extrusion(
            input_stl, output_file, rotation, thickness, resolution,
            angle_2d=info['angle_2d'], method=method
        )
        result['projection_type'] = label
        results.append(result)

        if not quiet:
            print(f"    Output: {os.path.basename(output_file)}")
            print(f"    Bounding box: {result['bbox_width']:.1f} x {result['bbox_height']:.1f} nm")
            print(f"    Triangles: {result['num_triangles']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate extruded shadow projections from STL files"
    )
    parser.add_argument('input', nargs='?',
                        help='Input STL file or directory containing STL files')
    parser.add_argument('-o', '--output',
                        help='Output directory (default: input_shadows/)')
    parser.add_argument('-t', '--thickness', type=float, default=160.0,
                        help='Extrusion thickness in nm (default: 160)')
    parser.add_argument('-r', '--resolution', type=int, default=300,
                        help='Projection resolution for final output (default: 300)')
    parser.add_argument('-s', '--samples', type=int, default=200,
                        help='Number of samples for min/max search (default: 200)')
    parser.add_argument('--method', choices=['voxel', 'contour'], default='voxel',
                        help='Mesh generation method (default: voxel)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Default to the output_20260121_2000nm directory if no input specified
    if args.input is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input = os.path.join(script_dir, 'output_20260121_2000nm')
        if not os.path.exists(args.input):
            parser.error("No input specified and default directory not found")

    # Determine input files
    if os.path.isfile(args.input):
        input_files = [args.input]
        default_output = os.path.join(os.path.dirname(args.input),
                                      os.path.splitext(os.path.basename(args.input))[0] + '_shadows')
    elif os.path.isdir(args.input):
        input_files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith('.stl')
        ])
        default_output = args.input + '_shadows'
    else:
        parser.error(f"Input not found: {args.input}")

    if not input_files:
        parser.error(f"No STL files found in: {args.input}")

    # Set output directory
    output_dir = args.output if args.output else default_output
    os.makedirs(output_dir, exist_ok=True)

    if not args.quiet:
        print("=" * 60)
        print("Shadow Extrusion Generator")
        print("=" * 60)
        print(f"Input: {args.input}")
        print(f"Output directory: {output_dir}")
        print(f"Extrusion thickness: {args.thickness} nm")
        print(f"Resolution: {args.resolution}")
        print(f"Min/Max search samples: {args.samples}")
        print(f"Method: {args.method}")
        print(f"Files to process: {len(input_files)}")

    # Process each file
    all_results = []
    for input_file in input_files:
        results = process_stl_file(
            input_file, output_dir,
            thickness=args.thickness,
            resolution=args.resolution,
            n_search_samples=args.samples,
            method=args.method,
            quiet=args.quiet
        )
        all_results.extend(results)

    if not args.quiet:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Processed {len(input_files)} STL files")
        print(f"Generated {len(all_results)} shadow extrusions")
        print(f"Output directory: {output_dir}")

    return all_results


if __name__ == '__main__':
    main()
