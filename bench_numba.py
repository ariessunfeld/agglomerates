#!/usr/bin/env python3
"""
Side-by-side benchmark: numpy vectorized vs Numba JIT.

Compares timing and validates that both produce valid agglomerates.
"""
import time
import sys
import numpy as np
sys.path.insert(0, '.')

from agglomerate import generate_agglomerate
from agglomerate_fast import generate_agglomerate_fast, warmup

# Warmup JIT (first call compiles — don't count this)
print("Warming up Numba JIT compilation...")
t0 = time.time()
warmup()
print(f"JIT warmup: {time.time() - t0:.1f}s\n")

LENGTH = 4000.0
DIAMETER = 160.0

print(f"Benchmark: length={LENGTH}nm, diameter={DIAMETER}nm")
print(f"{'n':>6}  {'numpy (s)':>10}  {'numba (s)':>10}  {'speedup':>8}  {'valid':>6}")
print("-" * 55)

for n in [50, 100, 250, 500]:
    # Numpy vectorized version
    t0 = time.time()
    agg_np = generate_agglomerate(n, LENGTH, DIAMETER, seed=1, verbose=False)
    t_np = time.time() - t0

    # Numba JIT version
    t0 = time.time()
    agg_nb = generate_agglomerate_fast(n, LENGTH, DIAMETER, seed=1, verbose=False)
    t_nb = time.time() - t0

    # Validate Numba output
    assert len(agg_nb) == n, f"Wrong particle count: {len(agg_nb)} vs {n}"

    # Check bounding box is reasonable (not degenerate)
    pos = np.array([r.center for r in agg_nb])
    bbox = pos.max(axis=0) - pos.min(axis=0)
    valid = all(d > 0 for d in bbox) and len(agg_nb) == n

    speedup = t_np / t_nb
    print(f"{n:>6}  {t_np:>10.2f}  {t_nb:>10.2f}  {speedup:>7.1f}x  {'ok' if valid else 'FAIL':>6}")
