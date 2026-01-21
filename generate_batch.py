"""
Batch generation of nanoparticle agglomerates with timestamped output.

Generates agglomerates with varying numbers of particles and saves both
STL files and metadata for later analysis.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

import random

import numpy as np

from agglomerate import generate_agglomerate, write_stl_binary, write_stl_ascii, Nanorod


def calculate_bounding_box(agglomerate: List[Nanorod]) -> Dict[str, float]:
    """
    Calculate axis-aligned bounding box dimensions for an agglomerate.

    Returns dict with 'length', 'width', 'height' (sorted largest to smallest)
    and 'min_coords', 'max_coords' for the bounding box corners.
    """
    # Collect all rod endpoints (accounting for rod radius)
    all_points = []
    for rod in agglomerate:
        radius = rod.diameter / 2
        for endpoint in [rod.endpoint1, rod.endpoint2]:
            # Add corner offsets to account for rod thickness
            for dx in [-radius, radius]:
                for dy in [-radius, radius]:
                    for dz in [-radius, radius]:
                        all_points.append(endpoint + np.array([dx, dy, dz]))

    points = np.array(all_points)
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Compute dimensions
    dims = max_coords - min_coords
    length, width, height = sorted(dims, reverse=True)  # L >= W >= H

    return {
        'length': float(length),
        'width': float(width),
        'height': float(height),
        'min_coords': min_coords.tolist(),
        'max_coords': max_coords.tolist()
    }


def generate_batch(
    n_values: list,
    length: float,
    diameter: float,
    output_dir: str,
    shape: str = 'cylinder',
    num_segments: int = 16,
    seed: int = None,
    ascii_format: bool = False
):
    """
    Generate a batch of agglomerates with different particle counts.

    Args:
        n_values: List of particle counts to generate
        length: Length of each nanorod (nm)
        diameter: Diameter of each nanorod (nm)
        output_dir: Directory to save outputs
        shape: 'cylinder' or 'prism'
        num_segments: Mesh segments for cylinders
        seed: Random seed (if None, uses different seed for each)
        ascii_format: Write ASCII STL instead of binary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Metadata for this batch
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'length_nm': length,
            'diameter_nm': diameter,
            'shape': shape,
            'num_segments': num_segments,
            'base_seed': seed,
        },
        'agglomerates': []
    }

    print(f"Generating batch of {len(n_values)} agglomerates")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: length={length}nm, diameter={diameter}nm, shape={shape}")
    print("-" * 60)

    for i, n in enumerate(n_values):
        # Use provided seed if given, otherwise pick a random seed
        current_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        print(f"[{i+1}/{len(n_values)}] Generating n={n} particles (seed={current_seed})...", end=" ", flush=True)

        # Generate agglomerate
        agglomerate = generate_agglomerate(
            num_particles=n,
            length=length,
            diameter=diameter,
            seed=current_seed,
            verbose=False
        )

        # Create filename
        ext = '.stl'
        filename = f"agglomerate_n{n:04d}_seed{current_seed}{ext}"
        filepath = os.path.join(output_dir, filename)

        # Write STL
        if ascii_format:
            write_stl_ascii(filepath, agglomerate, shape, num_segments)
        else:
            write_stl_binary(filepath, agglomerate, shape, num_segments)

        # Calculate bounding box
        bbox = calculate_bounding_box(agglomerate)

        # Record metadata
        agglomerate_meta = {
            'filename': filename,
            'n_particles': n,
            'seed': current_seed,
            'rod_positions': [rod.center.tolist() for rod in agglomerate],
            'rod_directions': [rod.direction.tolist() for rod in agglomerate],
            'bounding_box': {
                'length': bbox['length'],
                'width': bbox['width'],
                'height': bbox['height'],
                'min_coords': bbox['min_coords'],
                'max_coords': bbox['max_coords']
            }
        }
        metadata['agglomerates'].append(agglomerate_meta)

        print("done")

    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("-" * 60)
    print(f"Batch complete. {len(n_values)} agglomerates generated.")
    print(f"Metadata saved to: {metadata_file}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate batch of nanoparticle agglomerates with timestamps'
    )
    parser.add_argument('-n', '--n-values', type=str, default='5,10,20,50,100',
                        help='Comma-separated list of particle counts (default: 5,10,20,50,100)')
    parser.add_argument('--n-range', type=str, default=None,
                        help='Range of n values as start:stop:step (e.g., 5:100:5)')
    parser.add_argument('-l', '--length', type=float, default=100.0,
                        help='Length of each nanorod in nm (default: 100)')
    parser.add_argument('-d', '--diameter', type=float, default=10.0,
                        help='Diameter of each nanorod in nm (default: 10)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: output_YYYYMMDD_HHMMSS)')
    parser.add_argument('--shape', type=str, choices=['cylinder', 'prism'], default='cylinder',
                        help='Shape of particles (default: cylinder)')
    parser.add_argument('--segments', type=int, default=16,
                        help='Number of segments for cylinder mesh (default: 16)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Base random seed (default: varies per agglomerate)')
    parser.add_argument('--ascii', action='store_true',
                        help='Write ASCII STL instead of binary')

    args = parser.parse_args()

    # Parse n values
    if args.n_range:
        parts = args.n_range.split(':')
        start, stop, step = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 1
        n_values = list(range(start, stop + 1, step))
    else:
        n_values = [int(x.strip()) for x in args.n_values.split(',')]

    # Generate output directory name if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'output_{timestamp}'

    generate_batch(
        n_values=n_values,
        length=args.length,
        diameter=args.diameter,
        output_dir=args.output_dir,
        shape=args.shape,
        num_segments=args.segments,
        seed=args.seed,
        ascii_format=args.ascii
    )


if __name__ == '__main__':
    main()
