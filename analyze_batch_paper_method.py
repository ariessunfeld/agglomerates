"""
Analyze agglomerate batches using the paper's 2D projection box-counting method.

This implements the fractal dimension calculation as described in:
Abomailek et al., Small 2025, DOI: 10.1002/smll.202409673

Key differences from our original 3D method:
1. Uses 2D PROJECTIONS of the 3D agglomerate (not 3D point sampling)
2. Generates 3 random projections per agglomerate and averages D_f
3. Uses specific grid divisions: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105
4. Box-counting on the minimum enclosing square of the projection

This script can load agglomerates from:
- metadata.json (if available, contains rod parameters)
- STL files directly (parses mesh triangles)
"""

import os
import re
import glob
import json
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from agglomerate import Nanorod


# =============================================================================
# STL File Parsing Functions
# =============================================================================

def parse_stl_ascii(filepath: str) -> np.ndarray:
    """
    Parse an ASCII STL file and return triangle vertices.

    Args:
        filepath: Path to the ASCII STL file

    Returns:
        Array of shape (n_triangles, 3, 3) containing vertex coordinates
    """
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


def parse_stl_binary(filepath: str) -> np.ndarray:
    """
    Parse a binary STL file and return triangle vertices.

    Args:
        filepath: Path to the binary STL file

    Returns:
        Array of shape (n_triangles, 3, 3) containing vertex coordinates
    """
    triangles = []

    with open(filepath, 'rb') as f:
        # Skip 80-byte header
        f.read(80)

        # Read number of triangles (4-byte unsigned int)
        n_triangles = struct.unpack('<I', f.read(4))[0]

        for _ in range(n_triangles):
            # Read normal (3 floats, ignored)
            f.read(12)

            # Read 3 vertices (9 floats total)
            vertices = []
            for _ in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                vertices.append([x, y, z])

            triangles.append(vertices)

            # Read attribute byte count (2 bytes, ignored)
            f.read(2)

    return np.array(triangles)


def parse_stl(filepath: str) -> np.ndarray:
    """
    Parse an STL file (auto-detect binary vs ASCII).

    Args:
        filepath: Path to the STL file

    Returns:
        Array of shape (n_triangles, 3, 3) containing vertex coordinates
    """
    with open(filepath, 'rb') as f:
        header = f.read(80)

    # ASCII STL files start with "solid"
    try:
        header_text = header.decode('ascii').strip().lower()
        if header_text.startswith('solid'):
            # Could be ASCII, but let's verify by checking for binary structure
            with open(filepath, 'r') as f:
                content = f.read(1000)
                if 'vertex' in content.lower():
                    return parse_stl_ascii(filepath)
    except:
        pass

    # Default to binary
    return parse_stl_binary(filepath)


def extract_stl_info_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract n_particles and seed from STL filename.

    Expected format: agglomerate_n0050_seed46.stl

    Args:
        filename: STL filename

    Returns:
        Tuple of (n_particles, seed) or (None, None) if not parseable
    """
    basename = os.path.basename(filename)

    # Try pattern: agglomerate_n0050_seed46.stl
    match = re.match(r'agglomerate_n(\d+)_seed(\d+)\.stl', basename)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try simpler pattern: n0050_seed46.stl
    match = re.match(r'n(\d+)_seed(\d+)\.stl', basename)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


# =============================================================================
# Mesh Projection Functions (for STL-based analysis)
# =============================================================================

def project_triangle_to_2d(triangle: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Project a 3D triangle onto a 2D plane after applying rotation.

    Args:
        triangle: Array of shape (3, 3) with vertex coordinates
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Array of shape (3, 2) with 2D projected vertices
    """
    rotated = np.dot(triangle, rotation_matrix.T)
    return rotated[:, :2]  # Drop Z coordinate


def rasterize_triangle_2d(vertices_2d: np.ndarray,
                          grid_origin: np.ndarray,
                          pixel_size: float,
                          grid_size: int) -> np.ndarray:
    """
    Rasterize a 2D triangle onto a binary grid using scanline algorithm.

    Args:
        vertices_2d: Array of shape (3, 2) with 2D vertex coordinates
        grid_origin: Origin (min corner) of the grid
        pixel_size: Size of each pixel
        grid_size: Number of pixels per side

    Returns:
        Binary mask where the triangle is present
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)

    # Convert vertices to pixel coordinates
    pixels = ((vertices_2d - grid_origin) / pixel_size).astype(int)

    # Bounding box of triangle
    min_x = max(0, min(pixels[:, 0]))
    max_x = min(grid_size - 1, max(pixels[:, 0]))
    min_y = max(0, min(pixels[:, 1]))
    max_y = min(grid_size - 1, max(pixels[:, 1]))

    # Simple point-in-triangle test for each pixel in bounding box
    v0, v1, v2 = vertices_2d

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Convert pixel center to world coordinates
            p = grid_origin + np.array([x + 0.5, y + 0.5]) * pixel_size

            d1 = sign(p, v0, v1)
            d2 = sign(p, v1, v2)
            d3 = sign(p, v2, v0)

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            if not (has_neg and has_pos):
                mask[y, x] = True

    return mask


def project_mesh_to_2d(triangles: np.ndarray,
                       rotation_matrix: np.ndarray,
                       resolution: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Project a 3D mesh (triangles) to a 2D binary image.

    Args:
        triangles: Array of shape (n_triangles, 3, 3) with vertex coordinates
        rotation_matrix: 3x3 rotation matrix for projection orientation
        resolution: Resolution of the output image

    Returns:
        Tuple of (binary_image, pixel_size)
    """
    # Project all triangles to 2D
    projected_triangles = []
    all_points = []

    for triangle in triangles:
        tri_2d = project_triangle_to_2d(triangle, rotation_matrix)
        projected_triangles.append(tri_2d)
        all_points.extend(tri_2d)

    all_points = np.array(all_points)

    # Find minimum enclosing square
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    extent = max_coords - min_coords
    max_extent = max(extent)

    # Center the square
    center = (min_coords + max_coords) / 2
    grid_origin = center - max_extent / 2

    pixel_size = max_extent / resolution

    # Rasterize all triangles
    image = np.zeros((resolution, resolution), dtype=bool)

    for tri_2d in projected_triangles:
        tri_mask = rasterize_triangle_2d(tri_2d, grid_origin, pixel_size, resolution)
        image |= tri_mask

    return image, pixel_size


def calculate_fractal_dimension_from_mesh(triangles: np.ndarray,
                                          n_projections: int = 3,
                                          resolution: int = 1000) -> Tuple[float, float, List[float]]:
    """
    Calculate fractal dimension from mesh triangles using 2D projection method.

    Args:
        triangles: Array of shape (n_triangles, 3, 3) with vertex coordinates
        n_projections: Number of random projections to average
        resolution: Resolution for rasterization

    Returns:
        Tuple of (mean_Df, std_Df, individual_Dfs)
    """
    fractal_dims = []

    for _ in range(n_projections):
        rotation = random_rotation_matrix()
        binary_image, pixel_size = project_mesh_to_2d(triangles, rotation, resolution)
        box_sizes, box_counts = box_counting_2d(binary_image)
        df, r2 = calculate_fractal_dimension_2d(box_sizes, box_counts)
        fractal_dims.append(df)

    mean_df = np.mean(fractal_dims)
    std_df = np.std(fractal_dims)

    return mean_df, std_df, fractal_dims


# =============================================================================
# Random Rotation and Original Rod-based Functions
# =============================================================================

def random_rotation_matrix() -> np.ndarray:
    """Generate a random 3D rotation matrix for random projection orientation."""
    # Use QR decomposition of random matrix for uniform random rotation
    random_matrix = np.random.randn(3, 3)
    q, r = np.linalg.qr(random_matrix)
    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def project_rod_to_2d(rod: Nanorod, rotation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Project a 3D rod onto a 2D plane after applying rotation.

    Args:
        rod: Nanorod object
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Tuple of (endpoint1_2d, endpoint2_2d, radius) in the projection plane
    """
    # Rotate the rod endpoints
    ep1_rotated = rotation_matrix @ rod.endpoint1
    ep2_rotated = rotation_matrix @ rod.endpoint2

    # Project to XY plane (drop Z coordinate)
    ep1_2d = ep1_rotated[:2]
    ep2_2d = ep2_rotated[:2]

    # Radius in projection (approximate - actual projection of cylinder is more complex)
    radius = rod.diameter / 2

    return ep1_2d, ep2_2d, radius


def rasterize_rod_2d(ep1: np.ndarray, ep2: np.ndarray, radius: float,
                     grid_origin: np.ndarray, pixel_size: float,
                     grid_size: int) -> np.ndarray:
    """
    Rasterize a 2D rod (line segment with thickness) onto a binary grid.

    Args:
        ep1, ep2: 2D endpoints of the rod centerline
        radius: Rod radius
        grid_origin: Origin (min corner) of the grid
        pixel_size: Size of each pixel
        grid_size: Number of pixels per side

    Returns:
        Binary mask where the rod is present
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)

    # Direction and length of rod
    direction = ep2 - ep1
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return mask
    direction = direction / length

    # Normal to the rod direction
    normal = np.array([-direction[1], direction[0]])

    # Sample points along the rod and perpendicular to it
    # Number of samples along length
    n_length = max(2, int(length / pixel_size * 2))
    # Number of samples across width
    n_width = max(2, int(radius * 2 / pixel_size * 2))

    for t in np.linspace(0, 1, n_length):
        center = ep1 + t * (ep2 - ep1)
        for w in np.linspace(-radius, radius, n_width):
            point = center + w * normal

            # Convert to grid coordinates
            grid_x = int((point[0] - grid_origin[0]) / pixel_size)
            grid_y = int((point[1] - grid_origin[1]) / pixel_size)

            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                mask[grid_y, grid_x] = True

    return mask


def project_agglomerate_to_2d(agglomerate: List[Nanorod],
                               rotation_matrix: np.ndarray,
                               resolution: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Project a 3D agglomerate to a 2D binary image.

    Args:
        agglomerate: List of Nanorod objects
        rotation_matrix: 3x3 rotation matrix for projection orientation
        resolution: Resolution of the output image

    Returns:
        Tuple of (binary_image, pixel_size)
    """
    # First pass: find bounding box of projected rods
    all_points = []
    projected_rods = []

    for rod in agglomerate:
        ep1_2d, ep2_2d, radius = project_rod_to_2d(rod, rotation_matrix)
        projected_rods.append((ep1_2d, ep2_2d, radius))

        # Add endpoints with radius buffer for bounding box
        for ep in [ep1_2d, ep2_2d]:
            all_points.append(ep + np.array([radius, radius]))
            all_points.append(ep - np.array([radius, radius]))

    all_points = np.array(all_points)

    # Find minimum enclosing square (as per paper)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    # Make it square (use largest dimension)
    extent = max_coords - min_coords
    max_extent = max(extent)

    # Center the square
    center = (min_coords + max_coords) / 2
    grid_origin = center - max_extent / 2

    pixel_size = max_extent / resolution

    # Rasterize all rods
    image = np.zeros((resolution, resolution), dtype=bool)

    for ep1_2d, ep2_2d, radius in projected_rods:
        rod_mask = rasterize_rod_2d(ep1_2d, ep2_2d, radius, grid_origin, pixel_size, resolution)
        image |= rod_mask

    return image, pixel_size


def box_counting_2d(binary_image: np.ndarray, grid_divisions: List[int] = None) -> Tuple[List[float], List[int]]:
    """
    Perform 2D box counting on a binary image using the paper's method.

    Args:
        binary_image: 2D boolean array
        grid_divisions: List of number of divisions per side
                       (paper uses: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105)

    Returns:
        Tuple of (box_sizes, box_counts)
    """
    if grid_divisions is None:
        # Paper's specific grid divisions
        grid_divisions = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105]

    image_size = binary_image.shape[0]  # Assume square

    box_sizes = []
    box_counts = []

    for n_boxes in grid_divisions:
        box_size = image_size / n_boxes
        box_sizes.append(box_size)

        # Count occupied boxes
        count = 0
        for i in range(n_boxes):
            for j in range(n_boxes):
                # Get the region for this box
                i_start = int(i * box_size)
                i_end = int((i + 1) * box_size)
                j_start = int(j * box_size)
                j_end = int((j + 1) * box_size)

                # Clamp to image bounds
                i_end = min(i_end, image_size)
                j_end = min(j_end, image_size)

                # Check if any pixel in this box is occupied
                if binary_image[i_start:i_end, j_start:j_end].any():
                    count += 1

        box_counts.append(count)

    return box_sizes, box_counts


def calculate_fractal_dimension_2d(box_sizes: List[float], box_counts: List[int]) -> Tuple[float, float]:
    """
    Calculate fractal dimension from box counting data.

    Args:
        box_sizes: List of box sizes
        box_counts: List of corresponding box counts

    Returns:
        Tuple of (fractal_dimension, r_squared)
    """
    # Filter out any zero counts
    valid = [(s, c) for s, c in zip(box_sizes, box_counts) if c > 0]
    if len(valid) < 2:
        return 0.0, 0.0

    sizes, counts = zip(*valid)

    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    # Linear regression: log(N) = -D * log(s) + const
    # So slope = -D, and D = -slope
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    slope = coeffs[0]

    # Calculate R-squared
    y_pred = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_counts - y_pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    fractal_dimension = -slope

    return fractal_dimension, r_squared


def calculate_fractal_dimension_paper_method(agglomerate: List[Nanorod],
                                              n_projections: int = 3,
                                              resolution: int = 1000) -> Tuple[float, float, List[float]]:
    """
    Calculate fractal dimension using the paper's 2D projection method.

    Args:
        agglomerate: List of Nanorod objects
        n_projections: Number of random projections to average (paper uses 3)
        resolution: Resolution for rasterization

    Returns:
        Tuple of (mean_Df, std_Df, individual_Dfs)
    """
    fractal_dims = []

    for _ in range(n_projections):
        # Generate random projection orientation
        rotation = random_rotation_matrix()

        # Project agglomerate to 2D
        binary_image, pixel_size = project_agglomerate_to_2d(agglomerate, rotation, resolution)

        # Perform box counting
        box_sizes, box_counts = box_counting_2d(binary_image)

        # Calculate fractal dimension
        df, r2 = calculate_fractal_dimension_2d(box_sizes, box_counts)
        fractal_dims.append(df)

    mean_df = np.mean(fractal_dims)
    std_df = np.std(fractal_dims)

    return mean_df, std_df, fractal_dims


def load_batch(batch_dir: str) -> dict:
    """
    Load a batch of agglomerates from a directory.
    """
    metadata_file = os.path.join(batch_dir, 'metadata.json')

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"No metadata.json found in {batch_dir}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    params = metadata['parameters']
    length = params['length_nm']
    diameter = params['diameter_nm']

    agglomerates = []
    for agg_meta in metadata['agglomerates']:
        rods = []
        positions = agg_meta['rod_positions']
        directions = agg_meta['rod_directions']

        for pos, dir in zip(positions, directions):
            rod = Nanorod(
                center=np.array(pos),
                direction=np.array(dir),
                length=length,
                diameter=diameter
            )
            rods.append(rod)

        agglomerates.append({
            'n_particles': agg_meta['n_particles'],
            'seed': agg_meta['seed'],
            'filename': agg_meta['filename'],
            'rods': rods
        })

    return {
        'metadata': metadata,
        'agglomerates': agglomerates,
        'source': 'metadata'
    }


def load_batch_from_stl(batch_dir: str, verbose: bool = True) -> dict:
    """
    Load agglomerates directly from STL files in a directory.

    This function discovers all STL files matching the pattern 'agglomerate_*.stl'
    and loads them directly without requiring metadata.json.

    Args:
        batch_dir: Directory containing STL files
        verbose: Print progress information

    Returns:
        Dictionary with 'agglomerates' list containing mesh data
    """
    # Find all STL files
    stl_pattern = os.path.join(batch_dir, 'agglomerate_*.stl')
    stl_files = glob.glob(stl_pattern)

    if not stl_files:
        # Try without 'agglomerate_' prefix
        stl_pattern = os.path.join(batch_dir, '*.stl')
        stl_files = glob.glob(stl_pattern)

    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {batch_dir}")

    # Sort by n_particles if we can extract it from filename
    def sort_key(f):
        n, _ = extract_stl_info_from_filename(f)
        return n if n is not None else 0

    stl_files.sort(key=sort_key)

    if verbose:
        print(f"Found {len(stl_files)} STL files in {batch_dir}")

    agglomerates = []

    for stl_file in stl_files:
        n_particles, seed = extract_stl_info_from_filename(stl_file)

        if verbose:
            basename = os.path.basename(stl_file)
            print(f"  Loading {basename}...", end=" ", flush=True)

        triangles = parse_stl(stl_file)

        if verbose:
            print(f"{len(triangles)} triangles")

        agglomerates.append({
            'n_particles': n_particles,
            'seed': seed,
            'filename': os.path.basename(stl_file),
            'filepath': stl_file,
            'triangles': triangles  # Mesh data instead of rods
        })

    return {
        'agglomerates': agglomerates,
        'source': 'stl'
    }


def load_batch_auto(batch_dir: str, prefer_stl: bool = False, verbose: bool = True) -> dict:
    """
    Load agglomerates from a directory, auto-detecting the best source.

    Tries metadata.json first (unless prefer_stl=True), falls back to STL files.

    Args:
        batch_dir: Directory containing the batch
        prefer_stl: If True, prefer STL files even if metadata.json exists
        verbose: Print progress information

    Returns:
        Dictionary with agglomerate data
    """
    metadata_file = os.path.join(batch_dir, 'metadata.json')
    has_metadata = os.path.exists(metadata_file)

    if has_metadata and not prefer_stl:
        try:
            return load_batch(batch_dir)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load metadata.json: {e}")
                print("Falling back to STL files...")
            return load_batch_from_stl(batch_dir, verbose)
    else:
        return load_batch_from_stl(batch_dir, verbose)


def calculate_all_fractal_dimensions(batch_data: dict,
                                      n_projections: int = 3,
                                      resolution: int = 1000,
                                      verbose: bool = True) -> list:
    """
    Calculate fractal dimensions for all agglomerates using the paper's method.

    Handles both rod-based data (from metadata.json) and mesh-based data (from STL files).
    """
    results = []
    source = batch_data.get('source', 'metadata')

    if verbose:
        print("Calculating fractal dimensions (paper's 2D projection method)...")
        print(f"  Data source: {source}")
        print(f"  Using {n_projections} random projections per agglomerate")
        print(f"  Grid divisions: 5, 15, 25, ..., 105")
        print("-" * 60)

    for i, agg in enumerate(batch_data['agglomerates']):
        n = agg['n_particles']

        if verbose:
            n_str = str(n) if n is not None else "?"
            print(f"[{i+1}/{len(batch_data['agglomerates'])}] n={n_str}...", end=" ", flush=True)

        # Check data type and call appropriate function
        if 'triangles' in agg:
            # Mesh-based data from STL
            triangles = agg['triangles']
            mean_df, std_df, individual_dfs = calculate_fractal_dimension_from_mesh(
                triangles, n_projections=n_projections, resolution=resolution
            )
        elif 'rods' in agg:
            # Rod-based data from metadata
            rods = agg['rods']
            mean_df, std_df, individual_dfs = calculate_fractal_dimension_paper_method(
                rods, n_projections=n_projections, resolution=resolution
            )
        else:
            if verbose:
                print("SKIPPED (no data)")
            continue

        results.append({
            'n_particles': n,
            'fractal_dimension': mean_df,
            'fractal_dimension_std': std_df,
            'individual_dfs': individual_dfs,
            'seed': agg.get('seed'),
            'filename': agg['filename']
        })

        if verbose:
            print(f"D = {mean_df:.3f} ± {std_df:.3f}")

    if verbose:
        print("-" * 60)

    return results


def plot_fractal_dimension(results: list, output_file: str = None,
                           title: str = None, show: bool = True,
                           show_error_bars: bool = True):
    """
    Plot fractal dimension vs particle count with error bars.
    """
    n_values = [r['n_particles'] for r in results]
    fd_values = [r['fractal_dimension'] for r in results]
    fd_stds = [r['fractal_dimension_std'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    if show_error_bars:
        ax.errorbar(n_values, fd_values, yerr=fd_stds, fmt='bo-',
                    markersize=6, linewidth=1.5, capsize=3, label='Measured (mean ± std)')
    else:
        ax.plot(n_values, fd_values, 'bo-', markersize=8, linewidth=2, label='Measured')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='D=1 (line)')
    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='D=1.5')
    ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='D=1.8')
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='D=2 (filled plane)')

    # Literature range for 2D projections
    ax.axhspan(1.5, 1.8, alpha=0.2, color='green', label='Literature range (1.5-1.8)')

    ax.set_xlabel('Number of Particles (n)', fontsize=12)
    ax.set_ylabel('2D Fractal Dimension (D$_{f,BC}$)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Fractal Dimension vs Number of Particles\n(Paper Method: 2D Projection Box-Counting)', fontsize=14)

    ax.set_xlim(0, max(n_values) * 1.05)
    ax.set_ylim(0.8, 2.2)

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()

    return fig, ax


def export_results(results: list, output_file: str):
    """Export results to CSV."""
    with open(output_file, 'w') as f:
        f.write('n_particles,fractal_dimension,fractal_dimension_std,seed,filename\n')
        for r in results:
            n = r['n_particles'] if r['n_particles'] is not None else ''
            seed = r['seed'] if r['seed'] is not None else ''
            f.write(f"{n},{r['fractal_dimension']:.4f},{r['fractal_dimension_std']:.4f},{seed},{r['filename']}\n")

    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze agglomerates using the paper's 2D projection box-counting method"
    )
    parser.add_argument('batch_dir', type=str,
                        help='Directory containing the batch (with metadata.json or STL files)')
    parser.add_argument('--projections', type=int, default=3,
                        help='Number of random projections per agglomerate (default: 3)')
    parser.add_argument('--resolution', type=int, default=1000,
                        help='Resolution for 2D rasterization (default: 1000)')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Output plot filename')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV filename')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (just save)')
    parser.add_argument('--no-error-bars', action='store_true',
                        help='Do not show error bars on plot')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--use-stl', action='store_true',
                        help='Load STL files directly instead of using metadata.json')

    args = parser.parse_args()

    # Load batch (auto-detect source or use --use-stl flag)
    print(f"Loading batch from: {args.batch_dir}")
    batch_data = load_batch_auto(args.batch_dir, prefer_stl=args.use_stl, verbose=not args.quiet)

    # Print batch info based on data source
    if batch_data.get('source') == 'metadata':
        params = batch_data['metadata']['parameters']
        print(f"Batch timestamp: {batch_data['metadata']['timestamp']}")
        print(f"Parameters: length={params['length_nm']}nm, diameter={params['diameter_nm']}nm, shape={params['shape']}")
    else:
        print("Data source: STL files (no metadata available)")

    print(f"Number of agglomerates: {len(batch_data['agglomerates'])}")

    # Calculate fractal dimensions using paper method
    results = calculate_all_fractal_dimensions(
        batch_data,
        n_projections=args.projections,
        resolution=args.resolution,
        verbose=not args.quiet
    )

    # Generate output filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_plot is None:
        args.output_plot = os.path.join(args.batch_dir, f'fractal_analysis_paper_method_{timestamp}.png')
    if args.output_csv is None:
        args.output_csv = os.path.join(args.batch_dir, f'fractal_analysis_paper_method_{timestamp}.csv')

    # Export results
    export_results(results, args.output_csv)

    # Plot
    plot_fractal_dimension(
        results,
        output_file=args.output_plot,
        title=args.title,
        show=not args.no_show,
        show_error_bars=not args.no_error_bars
    )

    # Print summary
    fd_values = [r['fractal_dimension'] for r in results]
    print("\nSummary (Paper Method - 2D Projection Box-Counting):")
    print(f"  Min fractal dimension: {min(fd_values):.3f}")
    print(f"  Max fractal dimension: {max(fd_values):.3f}")
    print(f"  Mean fractal dimension: {np.mean(fd_values):.3f}")

    # Check for opacity transition (D approaching 2.0 for large N)
    large_n_results = [r for r in results if r['n_particles'] is not None and r['n_particles'] >= 100]
    if large_n_results:
        large_n_mean = np.mean([r['fractal_dimension'] for r in large_n_results])
        print(f"  Mean D for n≥100: {large_n_mean:.3f}")
        if large_n_mean > 1.8:
            print("  Note: D > 1.8 for large n may indicate opacity transition")


if __name__ == '__main__':
    main()
