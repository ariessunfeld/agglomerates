#!/usr/bin/env python3
"""
Generate 9um rod clumps (160nm cross-section) for sizes 15, 20, 25.
10 clumps per size, each with a unique random 4-digit seed in the filename.
"""

import os
import sys
import json
import time
import random
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agglomerate import generate_agglomerate, write_stl_binary
from generate_batch import calculate_bounding_box

# ── Configuration ──
LENGTH_NM = 9000.0
DIAMETER_NM = 160.0
N_VALUES = [15, 20, 25]
CLUMPS_PER_SIZE = 10
SHAPE = 'prism'

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = time.strftime('%Y%m%d')
    output_dir = os.path.join(script_dir, f"output_{date_str}_9um_160nm_small")
    os.makedirs(output_dir, exist_ok=True)

    # Generate all unique 4-digit seeds (sample without replacement)
    total_clumps = len(N_VALUES) * CLUMPS_PER_SIZE
    random.seed(42)  # reproducible seed selection
    all_seeds = random.sample(range(1000, 10000), total_clumps)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'length_nm': LENGTH_NM,
            'diameter_nm': DIAMETER_NM,
            'shape': SHAPE,
            'n_values': N_VALUES,
            'clumps_per_size': CLUMPS_PER_SIZE,
        },
        'agglomerates': []
    }

    print(f"Output directory: {output_dir}")
    print(f"Rod: {LENGTH_NM:.0f}nm long, {DIAMETER_NM:.0f}nm cross-section ({SHAPE})")
    print(f"Sizes: {N_VALUES}, {CLUMPS_PER_SIZE} clumps each")
    print(f"Total clumps: {total_clumps}")
    print("=" * 60)

    seed_idx = 0
    t_start = time.time()

    for n in N_VALUES:
        print(f"\n--- n={n} rods ({CLUMPS_PER_SIZE} clumps) ---")
        for i in range(CLUMPS_PER_SIZE):
            seed = all_seeds[seed_idx]
            seed_idx += 1

            t0 = time.time()
            agg = generate_agglomerate(
                num_particles=n,
                length=LENGTH_NM,
                diameter=DIAMETER_NM,
                seed=seed,
                verbose=False,
            )
            elapsed = time.time() - t0

            filename = f"agglomerate_n{n:04d}_seed{seed}.stl"
            filepath = os.path.join(output_dir, filename)
            write_stl_binary(filepath, agg, SHAPE)

            bbox = calculate_bounding_box(agg)

            metadata['agglomerates'].append({
                'filename': filename,
                'n_particles': n,
                'seed': seed,
                'rod_positions': [rod.center.tolist() for rod in agg],
                'rod_directions': [rod.direction.tolist() for rod in agg],
                'bounding_box': bbox,
            })

            print(f"  [{i+1}/{CLUMPS_PER_SIZE}] seed={seed}  {elapsed:.1f}s  -> {filename}")

    # Save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Complete: {total_clumps} clumps in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output:   {output_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == '__main__':
    main()
