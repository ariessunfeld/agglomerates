"""
Generate smooth extruded shadow projections from STL files.

Unlike generate_shadow_extrusion.py which rasterizes to a pixel grid (producing
blocky/voxelized output), this script works with vector geometry directly:
1. Projects 3D mesh triangles to 2D
2. Computes the exact 2D silhouette using polygon boolean union (shapely)
3. Triangulates the silhouette
4. Extrudes into a smooth 3D solid

The output is resolution-independent with smooth edges.
"""

import os
import argparse
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from scipy.spatial import Delaunay

from generate_shadow_extrusion import (
    parse_stl,
    write_stl_binary,
    calculate_normal,
    find_min_max_projections,
    random_rotation_matrix,
    calculate_bounding_box_dimensions,
    minimum_bounding_rectangle,
    convex_hull_2d,
)


def project_triangles_to_2d(triangles: np.ndarray,
                             rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D triangles to 2D by applying rotation and dropping Z.

    Args:
        triangles: Array of shape (n, 3, 3)
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Array of shape (n, 3, 2) — n triangles, each with 3 vertices of (x, y)
    """
    rotated = triangles @ rotation_matrix.T
    return rotated[:, :, :2]


def compute_silhouette(triangles_2d: np.ndarray,
                       batch_size: int = 500) -> Polygon | MultiPolygon:
    """
    Compute the 2D silhouette (union of all projected triangles).

    Uses shapely's unary_union with batched cascaded union for performance.

    Args:
        triangles_2d: Array of shape (n, 3, 2)
        batch_size: Number of triangles to union per batch

    Returns:
        Shapely Polygon or MultiPolygon representing the silhouette
    """
    # Create shapely polygons for each triangle, skipping degenerate ones
    polys = []
    for tri in triangles_2d:
        # Check for degenerate triangle (zero area)
        v0, v1, v2 = tri
        area = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                    (v2[0] - v0[0]) * (v1[1] - v0[1])) / 2
        if area < 1e-12:
            continue
        try:
            p = Polygon(tri)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty and p.area > 1e-12:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return Polygon()

    # Batched union for performance, with error recovery
    while len(polys) > 1:
        next_batch = []
        for i in range(0, len(polys), batch_size):
            batch = polys[i:i + batch_size]
            try:
                merged = unary_union(batch)
            except Exception:
                # Fallback: buffer each polygon to fix topology, then retry
                fixed = []
                for p in batch:
                    try:
                        fp = p.buffer(0)
                        if not fp.is_empty:
                            fixed.append(fp)
                    except Exception:
                        continue
                if not fixed:
                    continue
                merged = unary_union(fixed)
            if not merged.is_valid:
                merged = make_valid(merged)
            if not merged.is_empty:
                next_batch.append(merged)
        if not next_batch:
            return Polygon()
        polys = next_batch

    result = polys[0]
    if not result.is_valid:
        result = make_valid(result)

    return result


def _extract_polygons(geom) -> List[Polygon]:
    """Extract all Polygon objects from any shapely geometry (Polygon, Multi, Collection)."""
    from shapely.geometry import GeometryCollection
    if isinstance(geom, Polygon):
        return [geom]
    elif isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        result = []
        for g in geom.geoms:
            result.extend(_extract_polygons(g))
        return result
    return []


def triangulate_silhouette(silhouette) -> List[np.ndarray]:
    """
    Triangulate a shapely polygon silhouette using constrained Delaunay.

    Returns list of 2D triangles, each as array of shape (3, 2).
    """
    if silhouette.is_empty:
        return []

    geoms = _extract_polygons(silhouette)

    all_triangles = []

    for geom in geoms:
        if geom.is_empty or geom.area < 1e-12:
            continue

        # Collect boundary points (exterior + holes)
        exterior_coords = np.array(geom.exterior.coords[:-1])  # remove closing duplicate
        all_ring_coords = [exterior_coords]
        for interior in geom.interiors:
            all_ring_coords.append(np.array(interior.coords[:-1]))

        all_points = np.vstack(all_ring_coords)

        if len(all_points) < 3:
            continue

        # Delaunay triangulation of all boundary points
        try:
            tri = Delaunay(all_points)
        except Exception:
            continue

        # Filter: keep only triangles whose centroid is inside the polygon
        for simplex in tri.simplices:
            v0, v1, v2 = all_points[simplex]
            cx = (v0[0] + v1[0] + v2[0]) / 3
            cy = (v0[1] + v1[1] + v2[1]) / 3

            from shapely.geometry import Point
            if geom.contains(Point(cx, cy)):
                all_triangles.append(np.array([v0, v1, v2]))

    return all_triangles


def extrude_silhouette(silhouette, face_triangles: List[np.ndarray],
                       thickness: float, angle_2d: float = 0.0,
                       center_output: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extrude a 2D silhouette into a 3D mesh.

    Args:
        silhouette: Shapely polygon for extracting boundary rings
        face_triangles: Triangulated 2D faces from triangulate_silhouette
        thickness: Extrusion height (Z dimension)
        angle_2d: 2D rotation for axis alignment
        center_output: Center the mesh at origin in XY

    Returns:
        List of (normal, v0, v1, v2) for STL output
    """
    if not face_triangles:
        return []

    cos_a, sin_a = np.cos(angle_2d), np.sin(angle_2d)

    def rotate_2d(x, y):
        return cos_a * x + sin_a * y, -sin_a * x + cos_a * y

    # Collect all polygons (handles Polygon, MultiPolygon, GeometryCollection)
    geoms = _extract_polygons(silhouette)

    # Collect all boundary rings (for side walls)
    all_rings = []
    for geom in geoms:
        if geom.is_empty:
            continue
        all_rings.append(np.array(geom.exterior.coords[:-1]))
        for interior in geom.interiors:
            all_rings.append(np.array(interior.coords[:-1]))

    # Apply 2D rotation to everything and compute bounds for centering
    def rotate_points(pts):
        result = np.empty_like(pts)
        for i, (x, y) in enumerate(pts):
            result[i] = rotate_2d(x, y)
        return result

    rotated_rings = [rotate_points(ring) for ring in all_rings]
    rotated_faces = []
    for tri in face_triangles:
        rotated_faces.append(rotate_points(tri))

    # Compute centering offset
    if center_output and rotated_rings:
        all_pts = np.vstack(rotated_rings)
        cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
    else:
        cx, cy = 0, 0

    # Build STL triangles
    stl_tris = []
    up = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.0, -1.0])

    # Top and bottom faces
    for tri_2d in rotated_faces:
        v0 = np.array([tri_2d[0][0] - cx, tri_2d[0][1] - cy, thickness])
        v1 = np.array([tri_2d[1][0] - cx, tri_2d[1][1] - cy, thickness])
        v2 = np.array([tri_2d[2][0] - cx, tri_2d[2][1] - cy, thickness])

        # Ensure top face normal points up
        normal = calculate_normal(v0, v1, v2)
        if normal[2] < 0:
            v1, v2 = v2, v1
        stl_tris.append((up, v0, v1, v2))

        # Bottom face (reversed winding)
        b0 = np.array([tri_2d[0][0] - cx, tri_2d[0][1] - cy, 0])
        b1 = np.array([tri_2d[1][0] - cx, tri_2d[1][1] - cy, 0])
        b2 = np.array([tri_2d[2][0] - cx, tri_2d[2][1] - cy, 0])
        normal = calculate_normal(b0, b1, b2)
        if normal[2] < 0:
            stl_tris.append((down, b0, b1, b2))
        else:
            stl_tris.append((down, b0, b2, b1))

    # Side walls from boundary rings
    for ring in rotated_rings:
        n_pts = len(ring)
        for i in range(n_pts):
            j = (i + 1) % n_pts

            x0, y0 = ring[i][0] - cx, ring[i][1] - cy
            x1, y1 = ring[j][0] - cx, ring[j][1] - cy

            t0 = np.array([x0, y0, thickness])
            t1 = np.array([x1, y1, thickness])
            b0 = np.array([x0, y0, 0])
            b1 = np.array([x1, y1, 0])

            n1 = calculate_normal(t0, b0, t1)
            stl_tris.append((n1, t0, b0, t1))
            n2 = calculate_normal(t1, b0, b1)
            stl_tris.append((n2, t1, b0, b1))

    return stl_tris


def generate_shadow_extrusion_smooth(input_stl: str, output_stl: str,
                                      rotation_matrix: np.ndarray,
                                      thickness: float = 160.0,
                                      angle_2d: float = 0.0) -> dict:
    """
    Generate a smooth extruded shadow projection from an STL file.

    Args:
        input_stl: Path to input STL file
        output_stl: Path to output STL file
        rotation_matrix: 3x3 rotation matrix for viewing angle
        thickness: Extrusion thickness in nm
        angle_2d: 2D rotation angle for axis-aligned bounding box

    Returns:
        Dictionary with statistics
    """
    triangles = parse_stl(input_stl)

    # Project to 2D
    tris_2d = project_triangles_to_2d(triangles, rotation_matrix)

    # Compute vector silhouette
    silhouette = compute_silhouette(tris_2d)

    if silhouette.is_empty or silhouette.area < 1e-12:
        # Fallback: write empty STL
        write_stl_binary(output_stl, [])
        return {
            'input_file': input_stl,
            'output_file': output_stl,
            'num_triangles': 0,
            'projection_area': 0,
            'bbox_width': 0,
            'bbox_height': 0,
            'bbox_area': 0,
            'thickness': thickness,
        }

    # Get bounding box info from the silhouette
    bounds = silhouette.bounds  # (minx, miny, maxx, maxy)
    projection_area = silhouette.area

    # Compute minimum bounding rectangle from boundary points
    exterior_pts = np.array(silhouette.convex_hull.exterior.coords[:-1])
    _, width, height, _ = minimum_bounding_rectangle(exterior_pts)

    # Triangulate
    face_tris = triangulate_silhouette(silhouette)

    # Extrude
    stl_tris = extrude_silhouette(silhouette, face_tris, thickness,
                                   angle_2d, center_output=True)

    # Write STL
    write_stl_binary(output_stl, stl_tris)

    return {
        'input_file': input_stl,
        'output_file': output_stl,
        'num_triangles': len(stl_tris),
        'projection_area': projection_area,
        'bbox_width': width,
        'bbox_height': height,
        'bbox_area': width * height,
        'thickness': thickness,
    }


def process_stl_file(input_stl: str, output_dir: str, thickness: float = 160.0,
                     resolution: int = 300, n_search_samples: int = 200,
                     quiet: bool = False) -> List[dict]:
    """
    Process a single STL file, generating min, max, and random smooth projections.
    """
    if not quiet:
        print(f"\nProcessing: {os.path.basename(input_stl)}")

    triangles = parse_stl(input_stl)
    if not quiet:
        print(f"  Loaded {len(triangles)} triangles")

    # Find min and max projections (uses rasterization for search only)
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

    # Random projection
    random_rot = random_rotation_matrix()
    _, _, _, random_angle = calculate_bounding_box_dimensions(
        triangles, random_rot, search_resolution
    )

    results = []
    base_name = os.path.splitext(os.path.basename(input_stl))[0]

    projections = [
        ('min', min_rot, min_info, 'minimal'),
        ('max', max_rot, max_info, 'maximal'),
        ('random', random_rot, {'angle_2d': random_angle}, 'random'),
    ]

    for suffix, rotation, info, label in projections:
        output_file = os.path.join(output_dir, f"{base_name}_shadow_{suffix}.stl")

        if not quiet:
            print(f"  Generating {label} projection shadow (smooth)...")

        result = generate_shadow_extrusion_smooth(
            input_stl, output_file, rotation, thickness,
            angle_2d=info.get('angle_2d', 0.0)
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
        description="Generate smooth extruded shadow projections from STL files (vector-based)"
    )
    parser.add_argument('input', nargs='?',
                        help='Input STL file or directory containing STL files')
    parser.add_argument('-o', '--output',
                        help='Output directory (default: input_shadows/)')
    parser.add_argument('-t', '--thickness', type=float, default=160.0,
                        help='Extrusion thickness in nm (default: 160)')
    parser.add_argument('-r', '--resolution', type=int, default=300,
                        help='Resolution for projection search (default: 300)')
    parser.add_argument('-s', '--samples', type=int, default=200,
                        help='Number of samples for min/max search (default: 200)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    if args.input is None:
        parser.error("No input specified. Provide an STL file or directory.")

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

    output_dir = args.output if args.output else default_output
    os.makedirs(output_dir, exist_ok=True)

    if not args.quiet:
        print("=" * 60)
        print("Smooth Shadow Extrusion Generator (vector-based)")
        print("=" * 60)
        print(f"Input: {args.input}")
        print(f"Output directory: {output_dir}")
        print(f"Extrusion thickness: {args.thickness} nm")
        print(f"Search resolution: {args.resolution}")
        print(f"Min/Max search samples: {args.samples}")
        print(f"Files to process: {len(input_files)}")

    all_results = []
    for input_file in input_files:
        results = process_stl_file(
            input_file, output_dir,
            thickness=args.thickness,
            resolution=args.resolution,
            n_search_samples=args.samples,
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
