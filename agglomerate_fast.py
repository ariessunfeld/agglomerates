"""
Numba-JIT accelerated agglomerate generator.

Drop-in replacement for generate_agglomerate() from agglomerate.py.
The entire Brownian walk loop is compiled to machine code, eliminating
Python/numpy overhead that dominates runtime at small-to-medium n.

Note: Numba uses its own RNG, so output differs from the numpy version
for the same seed. Results are physically equivalent but not bitwise
identical. Internally reproducible (same seed → same output).

Usage:
    from agglomerate_fast import generate_agglomerate_fast
    agglomerate = generate_agglomerate_fast(num_particles=100, length=4000, diameter=160, seed=1)
"""

import numpy as np
from numba import njit
from typing import List, Optional

from agglomerate import Nanorod


# =============================================================================
# JIT-compiled core functions
# =============================================================================

@njit(cache=True)
def _dot3(a, b):
    """Dot product of two 3-vectors."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@njit(cache=True)
def _norm3(a):
    """Norm of a 3-vector."""
    return np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])


@njit(cache=True)
def _random_unit_vector():
    """Random unit vector uniform on S2."""
    vec = np.empty(3)
    vec[0] = np.random.randn()
    vec[1] = np.random.randn()
    vec[2] = np.random.randn()
    n = _norm3(vec)
    vec[0] /= n
    vec[1] /= n
    vec[2] /= n
    return vec


@njit(cache=True)
def _random_point_on_sphere(center, radius):
    """Random point uniform on sphere surface."""
    d = _random_unit_vector()
    result = np.empty(3)
    result[0] = center[0] + radius * d[0]
    result[1] = center[1] + radius * d[1]
    result[2] = center[2] + radius * d[2]
    return result


@njit(cache=True)
def _segment_segment_distance(p1, d1, len1, p2, d2, len2):
    """Minimum distance between two line segments in 3D."""
    half1 = len1 / 2.0
    half2 = len2 / 2.0

    # Endpoints
    a1 = p1 - half1 * d1
    b1 = p1 + half1 * d1
    a2 = p2 - half2 * d2
    b2 = p2 + half2 * d2

    u = b1 - a1
    v = b2 - a2
    w = a1 - a2

    a = _dot3(u, u)
    b = _dot3(u, v)
    c = _dot3(v, v)
    d = _dot3(u, w)
    e = _dot3(v, w)

    D = a * c - b * b

    if D < 1e-10:
        s = 0.0
        t = d / b if b > 1e-10 else 0.0
    else:
        s = (b * e - c * d) / D
        t = (a * e - b * d) / D

    if s < 0.0:
        s = 0.0
        t = e / c if c > 1e-10 else 0.0
    elif s > 1.0:
        s = 1.0
        t = (e + b) / c if c > 1e-10 else 0.0

    if t < 0.0:
        t = 0.0
        s = -d / a if a > 1e-10 else 0.0
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
    elif t > 1.0:
        t = 1.0
        s = (b - d) / a if a > 1e-10 else 0.0
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0

    closest1 = a1 + s * u
    closest2 = a2 + t * v
    diff = closest1 - closest2
    return _norm3(diff)


@njit(cache=True)
def _min_surface_distance(rod_center, rod_direction, length, diameter,
                           agg_centers, agg_directions, n_rods):
    """Minimum surface-to-surface distance from rod to agglomerate."""
    radius = diameter / 2.0
    min_dist = 1e30  # large number
    for i in range(n_rods):
        centerline_dist = _segment_segment_distance(
            rod_center, rod_direction, length,
            agg_centers[i], agg_directions[i], length
        )
        surface_dist = centerline_dist - radius - radius
        if surface_dist < min_dist:
            min_dist = surface_dist
    if min_dist < 0.0:
        return 0.0
    return min_dist


@njit(cache=True)
def _sample_escape_angle(p_esc):
    """Sample angle from escape/return distribution via rejection sampling."""
    if p_esc >= 1.0:
        return np.pi

    numerator = 2.0 * p_esc - p_esc * p_esc
    denom_max = p_esc * p_esc
    if denom_max < 1e-12:
        return np.random.uniform(0.0, 0.1)
    w_max = numerator / (4.0 * np.pi * (denom_max ** 1.5))

    for _ in range(10000):
        theta = np.random.uniform(0.0, np.pi)
        cos_theta = np.cos(theta)
        denom_inner = 2.0 * (1.0 - p_esc) * (1.0 - cos_theta) + p_esc * p_esc
        if denom_inner < 1e-12:
            continue
        w_theta = numerator / (4.0 * np.pi * (denom_inner ** 1.5))
        if np.random.uniform(0.0, w_max) < w_theta:
            return theta

    return np.random.uniform(0.0, np.pi)


@njit(cache=True)
def _redirect_to_sphere(escaped_position, com, r_out, p_esc):
    """Redirect escaped particle back to R_out sphere."""
    theta = _sample_escape_angle(p_esc)
    phi = np.random.uniform(0.0, 2.0 * np.pi)

    escape_dir = escaped_position - com
    n = _norm3(escape_dir)
    escape_dir = escape_dir / n

    # Orthonormal basis
    if abs(escape_dir[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    perp1 = np.cross(escape_dir, ref)
    perp1 = perp1 / _norm3(perp1)
    perp2 = np.cross(escape_dir, perp1)

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    new_dir = ct * escape_dir + st * cp * perp1 + st * sp * perp2
    return com + r_out * new_dir


@njit(cache=True)
def _generate_agglomerate_core(num_particles, length, diameter, r_out_factor, seed):
    """
    Core Brownian collision simulation compiled to machine code.

    Returns (centers, directions) as (N,3) arrays.
    """
    np.random.seed(seed)

    centers = np.empty((num_particles, 3))
    directions = np.empty((num_particles, 3))

    # First particle at origin along z
    centers[0, 0] = 0.0
    centers[0, 1] = 0.0
    centers[0, 2] = 0.0
    directions[0, 0] = 0.0
    directions[0, 1] = 0.0
    directions[0, 2] = 1.0

    com_sum = np.zeros(3)  # running sum for COM
    particles_added = 1
    max_attempts = 100000

    while particles_added < num_particles:
        # Center of mass and R_out
        com = com_sum / particles_added
        r_out = r_out_factor * length * (1.0 + 0.1 * particles_added)

        # New particle on R_out sphere
        new_center = _random_point_on_sphere(com, r_out)
        new_direction = _random_unit_vector()

        for _ in range(max_attempts):
            d_min = _min_surface_distance(
                new_center, new_direction, length, diameter,
                centers, directions, particles_added
            )

            if d_min <= 0.0:
                # Collision — add to agglomerate
                centers[particles_added, 0] = new_center[0]
                centers[particles_added, 1] = new_center[1]
                centers[particles_added, 2] = new_center[2]
                directions[particles_added, 0] = new_direction[0]
                directions[particles_added, 1] = new_direction[1]
                directions[particles_added, 2] = new_direction[2]
                com_sum[0] += new_center[0]
                com_sum[1] += new_center[1]
                com_sum[2] += new_center[2]
                particles_added += 1
                break

            # Brownian step
            move_dir = _random_unit_vector()
            new_center = new_center + d_min * move_dir
            new_direction = _random_unit_vector()

            # Check escape
            diff = new_center - com
            dist_from_com = _norm3(diff)

            if dist_from_com > r_out:
                p_esc = 1.0 - r_out / dist_from_com
                if np.random.random() < p_esc:
                    # Escape — new particle
                    new_center = _random_point_on_sphere(com, r_out)
                    new_direction = _random_unit_vector()
                else:
                    # Redirect
                    new_center = _redirect_to_sphere(new_center, com, r_out, p_esc)
                    new_direction = _random_unit_vector()

    return centers, directions


# =============================================================================
# Public API
# =============================================================================

def generate_agglomerate_fast(num_particles: int, length: float, diameter: float,
                               r_out_factor: float = 3.0, seed: Optional[int] = None,
                               verbose: bool = True) -> List[Nanorod]:
    """
    Generate agglomerate using Numba-JIT compiled Brownian simulation.

    Same interface as agglomerate.generate_agglomerate(). Output is
    physically equivalent but uses Numba's RNG (not bitwise identical
    to the numpy version for the same seed).
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    if verbose:
        print(f"Starting agglomerate generation with {num_particles} particles (Numba)")
        print(f"Rod length: {length}, diameter: {diameter}")

    centers, directions = _generate_agglomerate_core(
        num_particles, length, diameter, r_out_factor, seed
    )

    # Wrap in Nanorod objects for compatibility
    agglomerate = []
    for i in range(num_particles):
        agglomerate.append(Nanorod(
            center=centers[i].copy(),
            direction=directions[i].copy(),
            length=length,
            diameter=diameter,
        ))

    if verbose:
        print(f"Agglomerate generated with {num_particles} particles")

    return agglomerate


def warmup():
    """Pre-compile JIT functions with a tiny agglomerate."""
    _generate_agglomerate_core(3, 100.0, 10.0, 3.0, 0)
