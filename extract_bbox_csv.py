"""Extract bounding box dimensions from metadata.json or STL files to CSV."""

import csv
import json
import argparse
import os
import struct


def parse_stl_binary(filepath: str):
    """Parse a binary STL file and return triangle vertices as nested list."""
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
    return triangles


def parse_stl_ascii(filepath: str):
    """Parse an ASCII STL file and return triangle vertices as nested list."""
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
    return triangles


def parse_stl(filepath: str):
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


def compute_bbox_from_stl(filepath: str) -> dict:
    """Compute bounding box dimensions from an STL file."""
    triangles = parse_stl(filepath)

    if not triangles:
        return {'length': 0, 'width': 0, 'height': 0}

    # Flatten to get all vertices
    all_x = []
    all_y = []
    all_z = []

    for tri in triangles:
        for vertex in tri:
            all_x.append(vertex[0])
            all_y.append(vertex[1])
            all_z.append(vertex[2])

    x_extent = max(all_x) - min(all_x)
    y_extent = max(all_y) - min(all_y)
    z_extent = max(all_z) - min(all_z)

    # Sort dimensions: length >= width >= height
    dims = sorted([x_extent, y_extent, z_extent], reverse=True)

    return {
        'length': dims[0],
        'width': dims[1],
        'height': dims[2]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract bounding box dimensions to CSV (from metadata.json or STL files)'
    )
    parser.add_argument('output_folder', help='Path to output folder containing STL files or metadata.json')
    args = parser.parse_args()

    metadata_path = os.path.join(args.output_folder, 'metadata.json')
    csv_path = os.path.join(args.output_folder, 'bounding_boxes.csv')

    # Try to use metadata.json if it exists
    if os.path.exists(metadata_path):
        print(f"Using metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'bbox_L', 'bbox_W', 'bbox_H'])

            for agg in metadata['agglomerates']:
                bbox = agg['bounding_box']
                writer.writerow([
                    agg['filename'],
                    bbox['length'],
                    bbox['width'],
                    bbox['height']
                ])
    else:
        # Compute bounding boxes directly from STL files
        print(f"No metadata.json found, computing bounding boxes from STL files...")
        stl_files = sorted([
            f for f in os.listdir(args.output_folder)
            if f.endswith('.stl')
        ])

        if not stl_files:
            print(f"No STL files found in {args.output_folder}")
            return

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'bbox_L', 'bbox_W', 'bbox_H'])

            for stl_file in stl_files:
                stl_path = os.path.join(args.output_folder, stl_file)
                bbox = compute_bbox_from_stl(stl_path)
                writer.writerow([
                    stl_file,
                    f"{bbox['length']:.2f}",
                    f"{bbox['width']:.2f}",
                    f"{bbox['height']:.2f}"
                ])
                print(f"  {stl_file}: {bbox['length']:.1f} x {bbox['width']:.1f} x {bbox['height']:.1f}")

    print(f"Wrote {csv_path}")


if __name__ == '__main__':
    main()
