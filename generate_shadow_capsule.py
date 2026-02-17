"""
Generate smooth extruded shadow projections using analytical rod projection.

Instead of projecting triangle meshes (which produces thin walls and slivers at
triangle seams) or rasterizing to a pixel grid (which produces pixelated output),
this script projects each *rod* analytically as a 2D rectangle:

    LineString([endpoint1_2d, endpoint2_2d]).buffer(radius, cap_style='flat')

The union of all rectangles gives a perfectly smooth, resolution-independent
silhouette. Holes between rods are preserved automatically by shapely's
polygon union.

Pipeline:
1. Load metadata.json to get rod centers, directions, length, diameter
2. Find optimal viewing angles (reuses rasterization-based search from
   generate_shadow_extrusion.py — only used for angle search, not output)
3. For each viewing direction (min, max, random):
   a. Rotate rod centers/directions, project to 2D
   b. Create rectangle (buffered LineString with flat caps) for each rod
   c. Union all rectangles → smooth silhouette
   d. Triangulate with constrained Delaunay (Triangle library)
   e. Extrude to 3D and write binary STL
"""

import os
import json
import argparse
import numpy as np
from typing import List, Tuple, Optional

from shapely.geometry import LineString, Polygon, MultiPolygon, Point, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import triangle as tr

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


# =============================================================================
# Load agglomerate data from metadata.json
# =============================================================================

def load_agglomerate_data(metadata_path: str) -> Tuple[dict, List[dict]]:
    """
    Load agglomerate rod data from a metadata.json file.

    Args:
        metadata_path: Path to metadata.json

    Returns:
        Tuple of (parameters dict, list of agglomerate dicts)
        Each agglomerate dict contains:
          - filename, n_particles, seed
          - rod_positions: list of [x, y, z] centers
          - rod_directions: list of [dx, dy, dz] unit vectors
          - bounding_box: dict with length, width, height, min_coords, max_coords
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    parameters = metadata['parameters']
    agglomerates = metadata['agglomerates']

    return parameters, agglomerates


# =============================================================================
# Project rods to 2D rectangles
# =============================================================================

def project_rods_to_stadiums(rod_positions: np.ndarray,
                              rod_directions: np.ndarray,
                              rotation_matrix: np.ndarray,
                              length: float,
                              radius: float) -> List:
    """
    Project 3D rods to 2D rectangles (buffered line segments with flat caps).

    Each rod is defined by its center, direction, and length. The rod's
    3D endpoints are:
        center ± (length/2) * direction

    After rotation, we drop Z and create a 2D LineString buffered by the
    rod radius with flat end caps, producing a rectangle.

    Args:
        rod_positions: (n, 3) array of rod centers
        rod_directions: (n, 3) array of rod direction unit vectors
        rotation_matrix: 3x3 rotation matrix for viewing angle
        length: Rod length in nm
        radius: Rod radius in nm

    Returns:
        List of shapely Polygon objects (rectangles)
    """
    half_length = length / 2.0
    n_rods = len(rod_positions)

    stadiums = []
    for i in range(n_rods):
        center = np.array(rod_positions[i])
        direction = np.array(rod_directions[i])

        # 3D endpoints
        ep1 = center - half_length * direction
        ep2 = center + half_length * direction

        # Rotate to viewing frame
        ep1_rot = rotation_matrix @ ep1
        ep2_rot = rotation_matrix @ ep2

        # Project to 2D (drop Z)
        ep1_2d = ep1_rot[:2]
        ep2_2d = ep2_rot[:2]

        # Create stadium: LineString buffered by radius
        line = LineString([ep1_2d, ep2_2d])
        stadium = line.buffer(radius, cap_style='flat')

        if not stadium.is_empty and stadium.is_valid:
            stadiums.append(stadium)
        elif not stadium.is_empty:
            fixed = make_valid(stadium)
            if not fixed.is_empty:
                stadiums.append(fixed)

    return stadiums


# =============================================================================
# Compute shadow silhouette from stadiums
# =============================================================================

def compute_shadow_silhouette(stadiums: List) -> Polygon | MultiPolygon:
    """
    Compute the 2D shadow silhouette by unioning all stadium shapes.

    Args:
        stadiums: List of shapely Polygon objects (stadiums)

    Returns:
        Shapely Polygon or MultiPolygon representing the shadow silhouette
    """
    if not stadiums:
        return Polygon()

    try:
        result = unary_union(stadiums)
    except Exception:
        # Fallback: fix each polygon then retry
        fixed = []
        for s in stadiums:
            try:
                fs = make_valid(s)
                if not fs.is_empty:
                    fixed.append(fs)
            except Exception:
                continue
        if not fixed:
            return Polygon()
        result = unary_union(fixed)

    if not result.is_valid:
        result = make_valid(result)

    return result


# =============================================================================
# Extract polygons from any shapely geometry
# =============================================================================

def _extract_polygons(geom) -> List[Polygon]:
    """Extract all Polygon objects from any shapely geometry."""
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


# =============================================================================
# Triangulate and extrude
# =============================================================================

def triangulate_and_extrude(silhouette,
                             thickness: float,
                             angle_2d: float = 0.0,
                             center_output: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Triangulate a 2D silhouette and extrude it into a 3D STL mesh.

    Uses the Triangle library for constrained Delaunay triangulation, which
    respects the exact polygon boundary. This produces smooth edges that
    follow the analytical capsule shapes exactly — no rasterization, no
    pixel grid, no staircase artifacts.

    Args:
        silhouette: Shapely Polygon or MultiPolygon
        thickness: Extrusion height in nm
        angle_2d: 2D rotation angle for axis-aligned bounding box
        center_output: If True, center the mesh at origin in XY

    Returns:
        List of (normal, v0, v1, v2) tuples for STL output
    """
    if silhouette.is_empty:
        return []

    geoms = _extract_polygons(silhouette)
    if not geoms:
        return []

    cos_a, sin_a = np.cos(angle_2d), np.sin(angle_2d)

    def rotate_2d(x, y):
        return cos_a * x + sin_a * y, -sin_a * x + cos_a * y

    all_face_tris_2d = []  # list of (3, 2) arrays
    all_rings_2d = []      # list of (n, 2) arrays for side walls

    for geom in geoms:
        if geom.is_empty or geom.area < 1e-12:
            continue

        # Build input for Triangle library
        # Collect all vertices from exterior and interior rings
        exterior_coords = np.array(geom.exterior.coords[:-1])  # Remove closing duplicate
        n_exterior = len(exterior_coords)

        all_vertices = [exterior_coords]
        ring_sizes = [n_exterior]

        for interior in geom.interiors:
            interior_coords = np.array(interior.coords[:-1])
            all_vertices.append(interior_coords)
            ring_sizes.append(len(interior_coords))

        vertices = np.vstack(all_vertices)
        n_vertices = len(vertices)

        # Build segments (edges) for each ring
        segments = []
        offset = 0
        for ring_size in ring_sizes:
            for i in range(ring_size):
                j = (i + 1) % ring_size
                segments.append([offset + i, offset + j])
            offset += ring_size
        segments = np.array(segments)

        # Mark holes: need a point inside each hole
        holes = []
        for interior in geom.interiors:
            # Use centroid of hole ring as the hole marker
            hole_poly = Polygon(interior.coords)
            hole_pt = hole_poly.centroid
            if not hole_poly.contains(hole_pt):
                # Centroid outside (concave hole), use representative_point
                hole_pt = hole_poly.representative_point()
            holes.append([hole_pt.x, hole_pt.y])

        # Build Triangle input
        tri_input = {
            'vertices': vertices,
            'segments': segments,
        }
        if holes:
            tri_input['holes'] = np.array(holes)

        # Triangulate with constrained Delaunay ('p' = polygon mode)
        try:
            tri_output = tr.triangulate(tri_input, 'p')
        except Exception as e:
            # Fallback: skip this geometry
            continue

        if 'triangles' not in tri_output or len(tri_output['triangles']) == 0:
            continue

        # Extract triangles
        tri_vertices = tri_output['vertices']
        for tri_indices in tri_output['triangles']:
            v0 = tri_vertices[tri_indices[0]]
            v1 = tri_vertices[tri_indices[1]]
            v2 = tri_vertices[tri_indices[2]]
            all_face_tris_2d.append(np.array([v0, v1, v2]))

        # Collect boundary rings for side walls
        all_rings_2d.append(exterior_coords)
        for interior in geom.interiors:
            all_rings_2d.append(np.array(interior.coords[:-1]))

    if not all_face_tris_2d:
        return []

    # Apply 2D rotation to all geometry
    def rotate_points(pts):
        result = np.empty_like(pts)
        for i, (x, y) in enumerate(pts):
            result[i] = rotate_2d(x, y)
        return result

    rotated_faces = [rotate_points(tri) for tri in all_face_tris_2d]
    rotated_rings = [rotate_points(ring) for ring in all_rings_2d]

    # Compute centering offset
    if center_output and rotated_rings:
        all_pts = np.vstack(rotated_rings)
        cx_off = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
        cy_off = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
    else:
        cx_off, cy_off = 0.0, 0.0

    # Build STL triangles
    stl_tris = []
    up = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.0, -1.0])

    # Top and bottom faces
    for tri_2d in rotated_faces:
        v0 = np.array([tri_2d[0][0] - cx_off, tri_2d[0][1] - cy_off, thickness])
        v1 = np.array([tri_2d[1][0] - cx_off, tri_2d[1][1] - cy_off, thickness])
        v2 = np.array([tri_2d[2][0] - cx_off, tri_2d[2][1] - cy_off, thickness])

        # Ensure top face normal points up
        normal = calculate_normal(v0, v1, v2)
        if normal[2] < 0:
            v1, v2 = v2, v1
        stl_tris.append((up, v0, v1, v2))

        # Bottom face (reversed winding)
        b0 = np.array([tri_2d[0][0] - cx_off, tri_2d[0][1] - cy_off, 0.0])
        b1 = np.array([tri_2d[1][0] - cx_off, tri_2d[1][1] - cy_off, 0.0])
        b2 = np.array([tri_2d[2][0] - cx_off, tri_2d[2][1] - cy_off, 0.0])
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

            x0, y0 = ring[i][0] - cx_off, ring[i][1] - cy_off
            x1, y1 = ring[j][0] - cx_off, ring[j][1] - cy_off

            t0 = np.array([x0, y0, thickness])
            t1 = np.array([x1, y1, thickness])
            b0 = np.array([x0, y0, 0.0])
            b1 = np.array([x1, y1, 0.0])

            n1 = calculate_normal(t0, b0, t1)
            stl_tris.append((n1, t0, b0, t1))
            n2 = calculate_normal(t1, b0, b1)
            stl_tris.append((n2, t1, b0, b1))

    return stl_tris


# =============================================================================
# Per-agglomerate processing
# =============================================================================

def process_agglomerate(agglomerate: dict,
                         parameters: dict,
                         input_dir: str,
                         output_dir: str,
                         thickness: float = 160.0,
                         n_search_samples: int = 200,
                         search_resolution: int = 100,
                         quiet: bool = False) -> List[dict]:
    """
    Process a single agglomerate: find optimal projections and generate
    capsule-based shadow STLs.

    Args:
        agglomerate: Dict from metadata.json with rod_positions, rod_directions, etc.
        parameters: Dict with length_nm, diameter_nm from metadata.json
        input_dir: Directory containing the source STL files
        output_dir: Directory for output shadow STLs
        thickness: Extrusion thickness in nm
        n_search_samples: Number of random samples for projection search
        search_resolution: Resolution for rasterization-based angle search
        quiet: Suppress progress output

    Returns:
        List of result dicts (one per projection type: min, max, random)
    """
    filename = agglomerate['filename']
    stl_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]

    rod_positions = np.array(agglomerate['rod_positions'])
    rod_directions = np.array(agglomerate['rod_directions'])
    length_nm = parameters['length_nm']
    diameter_nm = parameters['diameter_nm']
    radius = diameter_nm / 2.0

    if not quiet:
        print(f"\nProcessing: {filename}")
        print(f"  Rods: {len(rod_positions)}, length: {length_nm} nm, diameter: {diameter_nm} nm")

    # Load STL for projection search (rasterization-based angle finding)
    if not os.path.exists(stl_path):
        print(f"  WARNING: STL file not found: {stl_path}, skipping")
        return []

    triangles = parse_stl(stl_path)
    if not quiet:
        print(f"  Loaded {len(triangles)} mesh triangles for angle search")

    # Find min and max projection angles using rasterization
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

    projections = [
        ('min', min_rot, min_info, 'minimal'),
        ('max', max_rot, max_info, 'maximal'),
        ('random', random_rot, {'angle_2d': random_angle}, 'random'),
    ]

    for suffix, rotation, info, label in projections:
        output_file = os.path.join(output_dir, f"{base_name}_shadow_{suffix}.stl")
        angle_2d = info.get('angle_2d', 0.0)

        if not quiet:
            print(f"  Generating {label} capsule projection shadow...")

        # Project rods to stadiums
        stadiums = project_rods_to_stadiums(
            rod_positions, rod_directions, rotation, length_nm, radius
        )

        # Union stadiums into silhouette
        silhouette = compute_shadow_silhouette(stadiums)

        if silhouette.is_empty:
            if not quiet:
                print(f"    WARNING: Empty silhouette, skipping")
            continue

        projection_area = silhouette.area

        # Get bounding box info from silhouette
        exterior_pts = np.array(silhouette.convex_hull.exterior.coords[:-1])
        _, bbox_width, bbox_height, _ = minimum_bounding_rectangle(exterior_pts)

        # Triangulate and extrude
        stl_tris = triangulate_and_extrude(
            silhouette, thickness, angle_2d=angle_2d, center_output=True
        )

        # Write STL
        write_stl_binary(output_file, stl_tris)

        result = {
            'input_file': stl_path,
            'output_file': output_file,
            'projection_type': label,
            'num_triangles': len(stl_tris),
            'num_rods': len(rod_positions),
            'projection_area': projection_area,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_area': bbox_width * bbox_height,
            'thickness': thickness,
        }
        results.append(result)

        if not quiet:
            print(f"    Output: {os.path.basename(output_file)}")
            print(f"    Bounding box: {bbox_width:.1f} x {bbox_height:.1f} nm")
            print(f"    Projection area: {projection_area:.0f} nm^2")
            print(f"    Triangles: {len(stl_tris)}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate smooth shadow projections using analytical capsule (stadium) projection of rods"
    )
    parser.add_argument('input', nargs='?',
                        help='Input directory containing metadata.json and STL files')
    parser.add_argument('-o', '--output',
                        help='Output directory (default: <input>_shadows/)')
    parser.add_argument('-t', '--thickness', type=float, default=160.0,
                        help='Extrusion thickness in nm (default: 160)')
    parser.add_argument('-s', '--samples', type=int, default=200,
                        help='Number of samples for min/max angle search (default: 200)')
    parser.add_argument('-r', '--resolution', type=int, default=100,
                        help='Resolution for angle search rasterization (default: 100)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    if args.input is None:
        parser.error("No input specified. Provide a directory containing metadata.json and STL files.")

    if not os.path.isdir(args.input):
        parser.error(f"Input must be a directory containing metadata.json: {args.input}")

    metadata_path = os.path.join(args.input, 'metadata.json')
    if not os.path.exists(metadata_path):
        parser.error(f"metadata.json not found in: {args.input}")

    # Set output directory
    output_dir = args.output if args.output else args.input + '_shadows'
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    parameters, agglomerates = load_agglomerate_data(metadata_path)

    if not args.quiet:
        print("=" * 60)
        print("Capsule-Projection Shadow Generator")
        print("=" * 60)
        print(f"Input: {args.input}")
        print(f"Output directory: {output_dir}")
        print(f"Rod length: {parameters['length_nm']} nm")
        print(f"Rod diameter: {parameters['diameter_nm']} nm")
        print(f"Extrusion thickness: {args.thickness} nm")
        print(f"Angle search samples: {args.samples}")
        print(f"Angle search resolution: {args.resolution}")
        print(f"Agglomerates to process: {len(agglomerates)}")

    # Process each agglomerate
    all_results = []
    for agglomerate in agglomerates:
        results = process_agglomerate(
            agglomerate, parameters, args.input, output_dir,
            thickness=args.thickness,
            n_search_samples=args.samples,
            search_resolution=args.resolution,
            quiet=args.quiet,
        )
        all_results.extend(results)

    if not args.quiet:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Processed {len(agglomerates)} agglomerates")
        print(f"Generated {len(all_results)} shadow extrusions")
        print(f"Output directory: {output_dir}")

    return all_results


if __name__ == '__main__':
    main()
