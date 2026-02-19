#!/usr/bin/env python3
"""
Test that optimized agglomerate generation produces identical results
to the original implementation.

Reference: n=100, length=4000nm, diameter=160nm, seed=1
from output_20260219_4um_160nm/seed_1/metadata.json
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agglomerate import generate_agglomerate


def load_reference():
    """Load reference rod positions/directions from existing metadata."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(
        script_dir, 'output_20260219_4um_160nm', 'seed_1', 'metadata.json'
    )
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Find the n=100 agglomerate
    for agg in metadata['agglomerates']:
        if agg['n_particles'] == 100:
            return (
                np.array(agg['rod_positions']),
                np.array(agg['rod_directions']),
            )

    raise ValueError("n=100 agglomerate not found in metadata")


def run_test():
    """Generate agglomerate and compare to reference."""
    print("Loading reference data...")
    ref_positions, ref_directions = load_reference()
    print(f"  Reference: {len(ref_positions)} rods")

    print("\nGenerating agglomerate (n=100, length=4000, diameter=160, seed=1)...")
    t0 = time.time()
    agglomerate = generate_agglomerate(
        num_particles=100,
        length=4000.0,
        diameter=160.0,
        seed=1,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Generation took {elapsed:.2f}s")

    # Extract positions and directions
    test_positions = np.array([rod.center for rod in agglomerate])
    test_directions = np.array([rod.direction for rod in agglomerate])

    # Compare
    pos_match = np.allclose(test_positions, ref_positions, atol=1e-10)
    dir_match = np.allclose(test_directions, ref_directions, atol=1e-10)

    print(f"\n  Positions match: {pos_match}")
    print(f"  Directions match: {dir_match}")

    if pos_match and dir_match:
        print("\n  PASS: Output is identical to reference.")
    else:
        print("\n  FAIL: Output differs from reference!")
        if not pos_match:
            diffs = np.abs(test_positions - ref_positions)
            max_diff = np.max(diffs)
            first_mismatch = np.argmax(np.any(diffs > 1e-10, axis=1))
            print(f"    Max position difference: {max_diff}")
            print(f"    First mismatch at rod index: {first_mismatch}")
            print(f"    Reference: {ref_positions[first_mismatch]}")
            print(f"    Got:       {test_positions[first_mismatch]}")
        if not dir_match:
            diffs = np.abs(test_directions - ref_directions)
            max_diff = np.max(diffs)
            first_mismatch = np.argmax(np.any(diffs > 1e-10, axis=1))
            print(f"    Max direction difference: {max_diff}")
            print(f"    First mismatch at rod index: {first_mismatch}")

    return pos_match and dir_match, elapsed


if __name__ == '__main__':
    passed, elapsed = run_test()
    sys.exit(0 if passed else 1)
