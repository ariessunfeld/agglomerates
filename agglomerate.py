"""
Brownian Collision Agglomerate Generator

Simulates nanoparticle agglomeration via Brownian motion collisions.
Based on Abomailek et al. Small 2025 DOI: 10.1002/smll.202409673

This module generates 3D agglomerates of nanorods/nanowires and exports them as STL files.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import struct


@dataclass
class Nanorod:
    """Represents a 1D nanoparticle (nanorod/nanowire) in 3D space."""
    center: np.ndarray  # 3D position of center
    direction: np.ndarray  # Unit vector for orientation
    length: float
    diameter: float

    @property
    def endpoint1(self) -> np.ndarray:
        """Get first endpoint of the rod."""
        return self.center - (self.length / 2) * self.direction

    @property
    def endpoint2(self) -> np.ndarray:
        """Get second endpoint of the rod."""
        return self.center + (self.length / 2) * self.direction

    def copy(self) -> 'Nanorod':
        """Create a copy of this nanorod."""
        return Nanorod(
            center=self.center.copy(),
            direction=self.direction.copy(),
            length=self.length,
            diameter=self.diameter
        )


def random_unit_vector() -> np.ndarray:
    """Generate a random unit vector uniformly distributed on the unit sphere."""
    # Use Gaussian distribution for uniform sampling on sphere
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)


def random_point_on_sphere(center: np.ndarray, radius: float) -> np.ndarray:
    """Generate a random point uniformly distributed on a sphere surface."""
    direction = random_unit_vector()
    return center + radius * direction


def segment_segment_distance(p1: np.ndarray, d1: np.ndarray, len1: float,
                             p2: np.ndarray, d2: np.ndarray, len2: float) -> float:
    """
    Calculate the minimum distance between two line segments in 3D.

    Args:
        p1: Center of first segment
        d1: Direction unit vector of first segment
        len1: Length of first segment
        p2: Center of second segment
        d2: Direction unit vector of second segment
        len2: Length of second segment

    Returns:
        Minimum distance between the two segments
    """
    # Endpoints
    a1 = p1 - (len1 / 2) * d1
    b1 = p1 + (len1 / 2) * d1
    a2 = p2 - (len2 / 2) * d2
    b2 = p2 + (len2 / 2) * d2

    # Direction vectors of segments
    u = b1 - a1
    v = b2 - a2
    w = a1 - a2

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b

    # Compute parameters for closest points
    if D < 1e-10:
        # Segments are nearly parallel
        s = 0.0
        t = d / b if b > 1e-10 else 0.0
    else:
        s = (b * e - c * d) / D
        t = (a * e - b * d) / D

    # Clamp s to [0, 1]
    if s < 0:
        s = 0
        t = e / c if c > 1e-10 else 0
    elif s > 1:
        s = 1
        t = (e + b) / c if c > 1e-10 else 0

    # Clamp t to [0, 1]
    if t < 0:
        t = 0
        s = -d / a if a > 1e-10 else 0
        s = max(0, min(1, s))
    elif t > 1:
        t = 1
        s = (b - d) / a if a > 1e-10 else 0
        s = max(0, min(1, s))

    # Closest points
    closest1 = a1 + s * u
    closest2 = a2 + t * v

    return np.linalg.norm(closest1 - closest2)


def min_distance_to_agglomerate(rod: Nanorod, agglomerate: List[Nanorod]) -> float:
    """Calculate minimum surface-to-surface distance from rod to agglomerate."""
    min_dist = float('inf')
    for agg_rod in agglomerate:
        # Distance between centerlines
        centerline_dist = segment_segment_distance(
            rod.center, rod.direction, rod.length,
            agg_rod.center, agg_rod.direction, agg_rod.length
        )
        # Surface-to-surface distance (subtract both radii)
        surface_dist = centerline_dist - (rod.diameter / 2) - (agg_rod.diameter / 2)
        min_dist = min(min_dist, surface_dist)
    return max(0, min_dist)


def vectorized_segment_distances(p1: np.ndarray, d1: np.ndarray, len1: float,
                                  p2_arr: np.ndarray, d2_arr: np.ndarray, len2: float) -> np.ndarray:
    """
    Calculate minimum distances from one line segment to N line segments.

    Vectorized version of segment_segment_distance that processes all
    agglomerate rods simultaneously using numpy array operations.

    Args:
        p1: (3,) Center of query segment
        d1: (3,) Direction unit vector of query segment
        len1: Length of query segment
        p2_arr: (N, 3) Centers of target segments
        d2_arr: (N, 3) Direction unit vectors of target segments
        len2: Length of target segments (same for all)

    Returns:
        (N,) array of minimum distances between query and each target segment
    """
    N = len(p2_arr)

    # Endpoints of query segment (fixed for all comparisons)
    a1 = p1 - (len1 / 2) * d1  # (3,)
    b1 = p1 + (len1 / 2) * d1  # (3,)

    # Endpoints of all target segments
    half2 = len2 / 2
    a2 = p2_arr - half2 * d2_arr  # (N, 3)
    b2 = p2_arr + half2 * d2_arr  # (N, 3)

    # Direction vectors
    u = b1 - a1  # (3,)
    v = b2 - a2  # (N, 3)
    w = a1 - a2  # (N, 3) via broadcast

    # Dot products — use explicit element-wise ops to match scalar np.dot exactly
    a = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]                     # scalar
    b_arr = v[:, 0]*u[0] + v[:, 1]*u[1] + v[:, 2]*u[2]        # (N,)
    c_arr = v[:, 0]*v[:, 0] + v[:, 1]*v[:, 1] + v[:, 2]*v[:, 2]  # (N,)
    d_arr = w[:, 0]*u[0] + w[:, 1]*u[1] + w[:, 2]*u[2]        # (N,)
    e_arr = v[:, 0]*w[:, 0] + v[:, 1]*w[:, 1] + v[:, 2]*w[:, 2]  # (N,)

    D = a * c_arr - b_arr * b_arr  # (N,)

    # Step 1: Initial s, t (unconstrained solution)
    parallel = D < 1e-10
    nonpar = ~parallel

    s = np.zeros(N)
    t = np.zeros(N)

    if np.any(nonpar):
        D_safe = np.where(nonpar, D, 1.0)  # avoid division by zero
        s = np.where(nonpar, (b_arr * e_arr - c_arr * d_arr) / D_safe, 0.0)
        t = np.where(nonpar, (a * e_arr - b_arr * d_arr) / D_safe, 0.0)

    # Parallel case: s=0, t = d/b if |b| > 1e-10 else 0
    par_bvalid = parallel & (np.abs(b_arr) > 1e-10)
    b_safe = np.where(par_bvalid, b_arr, 1.0)
    t = np.where(par_bvalid, d_arr / b_safe, t)

    # Step 2: Clamp s to [0, 1] and recalculate t
    c_valid = c_arr > 1e-10
    c_safe = np.where(c_valid, c_arr, 1.0)

    s_neg = s < 0
    s = np.where(s_neg, 0.0, s)
    t = np.where(s_neg & c_valid, e_arr / c_safe, t)
    t = np.where(s_neg & ~c_valid, 0.0, t)

    s_big = s > 1
    s = np.where(s_big, 1.0, s)
    t = np.where(s_big & c_valid, (e_arr + b_arr) / c_safe, t)
    t = np.where(s_big & ~c_valid, 0.0, t)

    # Step 3: Clamp t to [0, 1] and recalculate s (with final clamp)
    t_neg = t < 0
    t = np.where(t_neg, 0.0, t)
    if a > 1e-10:
        s = np.where(t_neg, np.clip(-d_arr / a, 0.0, 1.0), s)
    else:
        s = np.where(t_neg, 0.0, s)

    t_big = t > 1
    t = np.where(t_big, 1.0, t)
    if a > 1e-10:
        s = np.where(t_big, np.clip((b_arr - d_arr) / a, 0.0, 1.0), s)
    else:
        s = np.where(t_big, 0.0, s)

    # Closest points and distances
    closest1 = a1 + np.outer(s, u)         # (N, 3)
    closest2 = a2 + t[:, np.newaxis] * v   # (N, 3)

    diff = closest1 - closest2
    distances = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2)  # (N,)

    return distances


def min_distance_to_agglomerate_fast(rod_center: np.ndarray, rod_direction: np.ndarray,
                                      rod_length: float, rod_diameter: float,
                                      agg_centers: np.ndarray, agg_directions: np.ndarray,
                                      agg_length: float, agg_diameter: float) -> float:
    """
    Vectorized minimum surface-to-surface distance from a rod to the agglomerate.

    Args:
        rod_center: (3,) center of query rod
        rod_direction: (3,) direction of query rod
        rod_length: length of query rod
        rod_diameter: diameter of query rod
        agg_centers: (N, 3) centers of agglomerate rods
        agg_directions: (N, 3) directions of agglomerate rods
        agg_length: length of agglomerate rods (uniform)
        agg_diameter: diameter of agglomerate rods (uniform)

    Returns:
        Minimum surface-to-surface distance (clamped to >= 0)
    """
    centerline_dists = vectorized_segment_distances(
        rod_center, rod_direction, rod_length,
        agg_centers, agg_directions, agg_length
    )
    surface_dists = centerline_dists - (rod_diameter / 2) - (agg_diameter / 2)
    return max(0.0, float(np.min(surface_dists)))


def check_collision(rod: Nanorod, agglomerate: List[Nanorod], tolerance: float = 1e-6) -> bool:
    """Check if rod collides with any rod in the agglomerate."""
    return min_distance_to_agglomerate(rod, agglomerate) <= tolerance


def sample_escape_angle(p_esc: float) -> float:
    """
    Sample angle θ from the escape/return distribution.

    w(θ) = (2*P_esc - P_esc^2) / (4π * (2*(1-P_esc)*(1-cos(θ)) + P_esc^2)^(3/2))

    Uses rejection sampling.
    """
    if p_esc >= 1.0:
        return np.pi  # Full escape

    # The distribution peaks near θ=0, so we use rejection sampling
    # Maximum value is at θ=0: w(0) = (2*P_esc - P_esc^2) / (4π * P_esc^3)
    #                              = (2 - P_esc) / (4π * P_esc^2)

    numerator = 2 * p_esc - p_esc ** 2

    # For small P_esc, the distribution is very peaked near 0
    # We'll use importance sampling with a proposal distribution

    max_iterations = 10000
    for _ in range(max_iterations):
        # Sample θ uniformly from [0, π]
        theta = np.random.uniform(0, np.pi)

        # Calculate w(θ)
        cos_theta = np.cos(theta)
        denom_inner = 2 * (1 - p_esc) * (1 - cos_theta) + p_esc ** 2
        if denom_inner < 1e-12:
            continue
        w_theta = numerator / (4 * np.pi * (denom_inner ** 1.5))

        # Envelope: use maximum at θ=0 (or near it)
        denom_max = p_esc ** 2
        if denom_max < 1e-12:
            # Very small P_esc, return small angle
            return np.random.uniform(0, 0.1)
        w_max = numerator / (4 * np.pi * (denom_max ** 1.5))

        # Accept/reject
        if np.random.uniform(0, w_max) < w_theta:
            return theta

    # Fallback: return random angle
    return np.random.uniform(0, np.pi)


def redirect_to_sphere(escaped_position: np.ndarray, agglomerate_com: np.ndarray,
                       r_out: float, p_esc: float) -> np.ndarray:
    """
    Redirect an escaped particle back to the R_out sphere.

    Args:
        escaped_position: Position where particle escaped
        agglomerate_com: Center of mass of agglomerate
        r_out: Outer radius
        p_esc: Escape probability

    Returns:
        New position on the sphere surface
    """
    # Sample angle θ from distribution
    theta = sample_escape_angle(p_esc)

    # Sample φ uniformly from [0, 2π]
    phi = np.random.uniform(0, 2 * np.pi)

    # Direction from COM to escaped position (defines the axis)
    escape_dir = escaped_position - agglomerate_com
    escape_dir = escape_dir / np.linalg.norm(escape_dir)

    # Create orthonormal basis with escape_dir as z-axis
    if abs(escape_dir[0]) < 0.9:
        perp1 = np.cross(escape_dir, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(escape_dir, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(escape_dir, perp1)

    # New direction using spherical coordinates relative to escape direction
    new_dir = (np.cos(theta) * escape_dir +
               np.sin(theta) * np.cos(phi) * perp1 +
               np.sin(theta) * np.sin(phi) * perp2)

    return agglomerate_com + r_out * new_dir


def calculate_agglomerate_com(agglomerate: List[Nanorod]) -> np.ndarray:
    """Calculate the center of mass of the agglomerate."""
    if not agglomerate:
        return np.zeros(3)
    centers = np.array([rod.center for rod in agglomerate])
    return np.mean(centers, axis=0)


def generate_agglomerate(num_particles: int, length: float, diameter: float,
                         r_out_factor: float = 3.0, seed: Optional[int] = None,
                         verbose: bool = True) -> List[Nanorod]:
    """
    Generate a nanoparticle agglomerate via Brownian collision simulation.

    Args:
        num_particles: Target number of particles in the agglomerate
        length: Length of each nanorod
        diameter: Diameter of each nanorod
        r_out_factor: Factor to multiply length for R_out (default 3.0)
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        List of Nanorod objects forming the agglomerate
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize with first particle at origin
    initial_rod = Nanorod(
        center=np.zeros(3),
        direction=np.array([0, 0, 1], dtype=float),  # Initial rod along z-axis
        length=length,
        diameter=diameter
    )
    agglomerate = [initial_rod]

    # Pre-allocate arrays for vectorized distance computation
    agg_centers = np.empty((num_particles, 3))
    agg_directions = np.empty((num_particles, 3))
    agg_centers[0] = initial_rod.center
    agg_directions[0] = initial_rod.direction
    # Running sum for incremental center-of-mass
    com_sum = initial_rod.center.copy()

    if verbose:
        print(f"Starting agglomerate generation with {num_particles} particles")
        print(f"Rod length: {length}, diameter: {diameter}")

    particles_added = 1
    total_attempts = 0
    max_attempts_per_particle = 100000

    while particles_added < num_particles:
        # Calculate current center of mass and R_out incrementally
        com = com_sum / particles_added
        r_out = r_out_factor * length * (1 + 0.1 * particles_added)  # Grow with agglomerate

        # Create new particle at random position on R_out sphere
        new_center = random_point_on_sphere(com, r_out)
        new_direction = random_unit_vector()

        attempts = 0
        while attempts < max_attempts_per_particle:
            attempts += 1
            total_attempts += 1

            # Calculate minimum distance to agglomerate (vectorized)
            d_min = min_distance_to_agglomerate_fast(
                new_center, new_direction, length, diameter,
                agg_centers[:particles_added], agg_directions[:particles_added],
                length, diameter
            )

            # Check for collision
            if d_min <= 0:
                # Collision! Add to agglomerate
                new_rod = Nanorod(
                    center=new_center.copy(),
                    direction=new_direction.copy(),
                    length=length,
                    diameter=diameter
                )
                agglomerate.append(new_rod)
                agg_centers[particles_added] = new_center
                agg_directions[particles_added] = new_direction
                com_sum += new_center
                particles_added += 1
                if verbose:
                    print(f"  Particle {particles_added}/{num_particles} added after {attempts} steps")
                break

            # Move particle by d_min in random direction
            move_direction = random_unit_vector()
            new_center = new_center + d_min * move_direction
            # Also randomly rotate the rod
            new_direction = random_unit_vector()

            # Check if particle left the R_out sphere
            dist_from_com = np.linalg.norm(new_center - com)

            if dist_from_com > r_out:
                # Calculate escape probability
                p_esc = 1 - r_out / dist_from_com

                # Random test for escape
                if np.random.random() < p_esc:
                    # Particle escapes - start with new particle
                    new_center = random_point_on_sphere(com, r_out)
                    new_direction = random_unit_vector()
                else:
                    # Redirect particle back to sphere
                    new_center = redirect_to_sphere(new_center, com, r_out, p_esc)
                    new_direction = random_unit_vector()

        if attempts >= max_attempts_per_particle:
            if verbose:
                print(f"  Warning: Max attempts reached for particle {particles_added + 1}")

    if verbose:
        print(f"Agglomerate generated with {len(agglomerate)} particles")
        print(f"Total Brownian steps: {total_attempts}")

    return agglomerate


def create_cylinder_mesh(rod: Nanorod, num_segments: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh for a cylinder representing a nanorod.

    Args:
        rod: Nanorod to convert to mesh
        num_segments: Number of segments around the circumference

    Returns:
        Tuple of (vertices, faces) arrays
    """
    radius = rod.diameter / 2

    # Create orthonormal basis
    axis = rod.direction
    if abs(axis[0]) < 0.9:
        perp1 = np.cross(axis, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(axis, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    # Generate vertices for the cylinder
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    # Bottom cap center
    bottom_center = rod.endpoint1
    # Top cap center
    top_center = rod.endpoint2

    vertices = []

    # Bottom cap vertices (index 0 to num_segments-1)
    for angle in angles:
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(bottom_center + offset)

    # Top cap vertices (index num_segments to 2*num_segments-1)
    for angle in angles:
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(top_center + offset)

    # Cap centers
    vertices.append(bottom_center)  # index 2*num_segments
    vertices.append(top_center)     # index 2*num_segments + 1

    vertices = np.array(vertices)

    # Generate faces
    faces = []
    bottom_center_idx = 2 * num_segments
    top_center_idx = 2 * num_segments + 1

    for i in range(num_segments):
        next_i = (i + 1) % num_segments

        # Bottom cap triangle
        faces.append([bottom_center_idx, next_i, i])

        # Top cap triangle
        faces.append([top_center_idx, num_segments + i, num_segments + next_i])

        # Side quad (as two triangles)
        faces.append([i, next_i, num_segments + next_i])
        faces.append([i, num_segments + next_i, num_segments + i])

    faces = np.array(faces)

    return vertices, faces


def create_prism_mesh(rod: Nanorod) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh for a rectangular prism representing a nanorod.

    The prism has square cross-section with side length equal to the rod diameter.

    Args:
        rod: Nanorod to convert to mesh

    Returns:
        Tuple of (vertices, faces) arrays
    """
    half_width = rod.diameter / 2

    # Create orthonormal basis
    axis = rod.direction
    if abs(axis[0]) < 0.9:
        perp1 = np.cross(axis, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(axis, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    # Endpoints along the rod axis
    bottom_center = rod.endpoint1
    top_center = rod.endpoint2

    # 8 vertices of the rectangular prism
    # Bottom face (4 vertices)
    v0 = bottom_center + half_width * (-perp1 - perp2)  # bottom-left-back
    v1 = bottom_center + half_width * (+perp1 - perp2)  # bottom-right-back
    v2 = bottom_center + half_width * (+perp1 + perp2)  # bottom-right-front
    v3 = bottom_center + half_width * (-perp1 + perp2)  # bottom-left-front

    # Top face (4 vertices)
    v4 = top_center + half_width * (-perp1 - perp2)  # top-left-back
    v5 = top_center + half_width * (+perp1 - perp2)  # top-right-back
    v6 = top_center + half_width * (+perp1 + perp2)  # top-right-front
    v7 = top_center + half_width * (-perp1 + perp2)  # top-left-front

    vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7])

    # 12 triangles (2 per face, 6 faces)
    faces = np.array([
        # Bottom face (normal pointing down along -axis)
        [0, 2, 1],
        [0, 3, 2],
        # Top face (normal pointing up along +axis)
        [4, 5, 6],
        [4, 6, 7],
        # Front face (normal along +perp2)
        [3, 7, 6],
        [3, 6, 2],
        # Back face (normal along -perp2)
        [0, 1, 5],
        [0, 5, 4],
        # Right face (normal along +perp1)
        [1, 2, 6],
        [1, 6, 5],
        # Left face (normal along -perp1)
        [0, 4, 7],
        [0, 7, 3],
    ])

    return vertices, faces


def create_rod_mesh(rod: Nanorod, shape: str = 'cylinder', num_segments: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh for a nanorod with the specified shape.

    Args:
        rod: Nanorod to convert to mesh
        shape: Either 'cylinder' or 'prism'
        num_segments: Number of segments for cylinder (ignored for prism)

    Returns:
        Tuple of (vertices, faces) arrays
    """
    if shape == 'prism':
        return create_prism_mesh(rod)
    else:
        return create_cylinder_mesh(rod, num_segments)


def write_stl_binary(filename: str, agglomerate: List[Nanorod],
                     shape: str = 'cylinder', num_segments: int = 16):
    """
    Write agglomerate to a binary STL file.

    Args:
        filename: Output filename
        agglomerate: List of nanorods to export
        shape: Shape of rods - 'cylinder' or 'prism'
        num_segments: Number of segments for cylinder approximation
    """
    all_triangles = []

    for rod in agglomerate:
        vertices, faces = create_rod_mesh(rod, shape, num_segments)

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Calculate normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])

            all_triangles.append((normal, v0, v1, v2))

    # Write binary STL
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        header = b'Binary STL generated by fluffy-clumps agglomerate generator'
        header = header.ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', len(all_triangles)))

        # Write each triangle
        for normal, v0, v1, v2 in all_triangles:
            # Normal vector
            f.write(struct.pack('<fff', *normal))
            # Vertices
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            # Attribute byte count (unused)
            f.write(struct.pack('<H', 0))

    print(f"STL file written: {filename}")
    print(f"  Total triangles: {len(all_triangles)}")


def write_stl_ascii(filename: str, agglomerate: List[Nanorod],
                    shape: str = 'cylinder', num_segments: int = 16):
    """
    Write agglomerate to an ASCII STL file.

    Args:
        filename: Output filename
        agglomerate: List of nanorods to export
        shape: Shape of rods - 'cylinder' or 'prism'
        num_segments: Number of segments for cylinder approximation
    """
    with open(filename, 'w') as f:
        f.write("solid agglomerate\n")

        for rod in agglomerate:
            vertices, faces = create_rod_mesh(rod, shape, num_segments)

            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])

                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

        f.write("endsolid agglomerate\n")

    print(f"ASCII STL file written: {filename}")


def calculate_fractal_dimension(agglomerate: List[Nanorod], num_samples: int = 1000) -> float:
    """
    Estimate the fractal dimension using box-counting method.

    Args:
        agglomerate: List of nanorods
        num_samples: Number of points to sample on the agglomerate surface

    Returns:
        Estimated fractal dimension
    """
    # Sample points on the surface of all rods
    points = []
    for rod in agglomerate:
        # Sample along the length
        for _ in range(num_samples // len(agglomerate)):
            t = np.random.uniform(-0.5, 0.5)
            angle = np.random.uniform(0, 2 * np.pi)

            # Point on cylinder surface
            axis = rod.direction
            if abs(axis[0]) < 0.9:
                perp1 = np.cross(axis, np.array([1, 0, 0]))
            else:
                perp1 = np.cross(axis, np.array([0, 1, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(axis, perp1)

            center_point = rod.center + t * rod.length * rod.direction
            surface_point = center_point + (rod.diameter / 2) * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            points.append(surface_point)

    points = np.array(points)

    # Box counting
    # Find bounding box
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    extent = max_coords - min_coords
    max_extent = max(extent)

    # Use various box sizes
    box_sizes = []
    box_counts = []

    for power in range(1, 8):
        num_boxes = 2 ** power
        box_size = max_extent / num_boxes

        # Count occupied boxes
        box_indices = ((points - min_coords) / box_size).astype(int)
        box_indices = np.clip(box_indices, 0, num_boxes - 1)
        unique_boxes = set(map(tuple, box_indices))

        box_sizes.append(box_size)
        box_counts.append(len(unique_boxes))

    # Fit line to log-log plot
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)

    # Linear regression
    slope, _ = np.polyfit(log_sizes, log_counts, 1)

    return -slope


def main():
    """Main function to generate agglomerates from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate nanoparticle agglomerates via Brownian collision simulation'
    )
    parser.add_argument('-n', '--num-particles', type=int, default=10,
                        help='Number of particles in agglomerate (default: 10)')
    parser.add_argument('-l', '--length', type=float, default=100.0,
                        help='Length of each nanorod in nm (default: 100)')
    parser.add_argument('-d', '--diameter', type=float, default=10.0,
                        help='Diameter of each nanorod in nm (default: 10)')
    parser.add_argument('-o', '--output', type=str, default='agglomerate.stl',
                        help='Output STL filename (default: agglomerate.stl)')
    parser.add_argument('--ascii', action='store_true',
                        help='Write ASCII STL instead of binary')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--shape', type=str, choices=['cylinder', 'prism'], default='cylinder',
                        help='Shape of particles: cylinder or prism (default: cylinder)')
    parser.add_argument('--segments', type=int, default=16,
                        help='Number of segments for cylinder mesh (default: 16)')
    parser.add_argument('--fractal', action='store_true',
                        help='Calculate and print fractal dimension')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Generate agglomerate
    agglomerate = generate_agglomerate(
        num_particles=args.num_particles,
        length=args.length,
        diameter=args.diameter,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Write STL file
    if args.ascii:
        write_stl_ascii(args.output, agglomerate, args.shape, args.segments)
    else:
        write_stl_binary(args.output, agglomerate, args.shape, args.segments)

    # Calculate fractal dimension if requested
    if args.fractal:
        fd = calculate_fractal_dimension(agglomerate)
        print(f"Estimated fractal dimension: {fd:.2f}")


if __name__ == '__main__':
    main()
