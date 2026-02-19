#!/usr/bin/env python3
"""
Run full rod clump and shadow experiment.

Generates rectangular rod clumps (prisms) and their random-orientation shadows.

Configuration:
- Rod lengths: 4um (4000nm) and 9um (9000nm)
- Cross-section: 160nm x 160nm (square prism)
- Clump sizes: 10, 20, 30, 40, 50, 75, 100, 150, 200, 250 rods
- 5 seeds per clump size
- 2 random-orientation shadow projections per clump
- Shadow extrusion thickness: 160nm

Output structure:
  output_4um_160nm/
    seed_1/         (STLs + metadata.json)
    seed_2/
    ...
    seed_5/
    shadows/        (shadow STLs)
    shadow_summary.csv
  output_9um_160nm/
    (same structure)
"""

import os
import sys
import csv
import time
import numpy as np

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_batch import generate_batch
from generate_shadow_capsule import (
    load_agglomerate_data,
    project_rods_to_stadiums,
    compute_shadow_silhouette,
    triangulate_and_extrude,
)
from generate_shadow_extrusion import (
    random_rotation_matrix,
    write_stl_binary,
)

# ============================================================
# Configuration
# ============================================================
LENGTHS_NM = [4000, 9000]
DIAMETER_NM = 160
N_VALUES = [10, 20, 30, 40, 50, 75, 100, 150, 200, 250]
SEEDS = [1, 2, 3, 4, 5]
N_RANDOM_ORIENTATIONS = 2
SHADOW_THICKNESS_NM = 160
SHAPE = 'prism'


def format_length(length_nm):
    """Format length in nm as human-readable string (e.g., 4000 -> '4um')."""
    um = length_nm / 1000
    if um == int(um):
        return f"{int(um)}um"
    return f"{um}um"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    total_start = time.time()
    date_str = time.strftime('%Y%m%d')

    for length_nm in LENGTHS_NM:
        length_label = format_length(length_nm)
        experiment_dir = os.path.join(script_dir, f"output_{date_str}_{length_label}_{DIAMETER_NM}nm")
        shadow_dir = os.path.join(experiment_dir, "shadows")
        os.makedirs(shadow_dir, exist_ok=True)

        shadow_results = []

        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENT: {length_label} rods, {DIAMETER_NM}nm cross-section")
        print(f"{'#' * 60}")

        for seed in SEEDS:
            batch_dir = os.path.join(experiment_dir, f"seed_{seed}")

            # ── Step 1: Generate clumps ──
            print(f"\n{'=' * 60}")
            print(f"Generating clumps: length={length_nm}nm, seed={seed}")
            print(f"{'=' * 60}")
            t0 = time.time()
            generate_batch(
                n_values=N_VALUES,
                length=float(length_nm),
                diameter=float(DIAMETER_NM),
                output_dir=batch_dir,
                shape=SHAPE,
                seed=seed,
            )
            print(f"Batch generation took {time.time() - t0:.1f}s")

            # ── Step 2: Generate shadows ──
            print(f"\nGenerating random-orientation shadows for seed={seed}...")
            metadata_path = os.path.join(batch_dir, 'metadata.json')
            parameters, agglomerates = load_agglomerate_data(metadata_path)
            radius = DIAMETER_NM / 2.0

            for agg in agglomerates:
                rod_positions = np.array(agg['rod_positions'])
                rod_directions = np.array(agg['rod_directions'])
                n_particles = agg['n_particles']
                base_name = os.path.splitext(agg['filename'])[0]

                for orient_idx in range(1, N_RANDOM_ORIENTATIONS + 1):
                    # Deterministic seed for each orientation so results are reproducible
                    orient_seed = seed * 10000 + n_particles * 100 + orient_idx
                    np.random.seed(orient_seed)
                    rot = random_rotation_matrix()

                    # Project rods to 2D rectangles (flat-capped buffered line segments)
                    stadiums = project_rods_to_stadiums(
                        rod_positions, rod_directions, rot,
                        float(length_nm), radius
                    )

                    # Compute shadow silhouette (union of all projected rectangles)
                    silhouette = compute_shadow_silhouette(stadiums)

                    if silhouette.is_empty:
                        print(f"  WARNING: Empty silhouette for {base_name} orient {orient_idx}")
                        continue

                    shadow_area = silhouette.area

                    # Triangulate and extrude to 3D
                    stl_tris = triangulate_and_extrude(
                        silhouette, SHADOW_THICKNESS_NM, angle_2d=0.0
                    )

                    if not stl_tris:
                        print(f"  WARNING: Triangulation failed for {base_name} orient {orient_idx}")
                        continue

                    # Save shadow STL
                    shadow_filename = f"{base_name}_shadow_rand{orient_idx}.stl"
                    shadow_path = os.path.join(shadow_dir, shadow_filename)
                    write_stl_binary(shadow_path, stl_tris)

                    shadow_results.append({
                        'length_nm': length_nm,
                        'n_particles': n_particles,
                        'seed': seed,
                        'orientation': orient_idx,
                        'shadow_area_nm2': shadow_area,
                        'shadow_file': shadow_filename,
                    })

                    print(f"  {shadow_filename}: area = {shadow_area:.0f} nm^2")

        # Save summary CSV with all shadow areas
        csv_path = os.path.join(experiment_dir, 'shadow_summary.csv')
        fieldnames = [
            'length_nm', 'n_particles', 'seed', 'orientation',
            'shadow_area_nm2', 'shadow_file',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shadow_results)

        print(f"\n{'=' * 60}")
        print(f"Summary for {length_label}:")
        print(f"  Clumps generated: {len(N_VALUES) * len(SEEDS)}")
        print(f"  Shadows generated: {len(shadow_results)}")
        print(f"  CSV saved to: {csv_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 60}")
    print(f"# COMPLETE - Total time: {total_elapsed / 60:.1f} minutes")
    print(f"{'#' * 60}")


if __name__ == '__main__':
    main()
