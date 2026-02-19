# AGENTS.md

## Project Overview

Nanorod agglomerate generator using Brownian collision simulation (Abomailek et al., Small 2025). Generates 3D clumps of nanorods as STL files and produces 2D shadow projections for geometric cross-section analysis.

## Environment Setup

This project runs on **macOS (Apple Silicon)** using **miniforge/conda** with the **base** environment.

```bash
# Python is at:
/opt/homebrew/Caskroom/miniforge/base/bin/python3

# No project-specific conda env — uses base. Activate if needed:
conda activate base

# Install dependencies:
pip install -r requirements.txt

# Optional but strongly recommended for performance (~20x speedup):
pip install numba
```

There is no dedicated conda environment for this project. All dependencies are installed in the base miniforge environment. Do NOT create a new conda environment unless explicitly asked.

## Key Architecture

### Core module: `agglomerate.py`

This is the single most important file. It contains:
- `Nanorod` dataclass — represents a rod with center, direction, length, diameter
- `generate_agglomerate()` — the main Brownian collision simulation
- `vectorized_segment_distances()` — numpy-vectorized distance computation
- `_jit_segment_segment_distance()` / `_jit_min_surface_distance()` — Numba JIT-compiled distance functions (optional, auto-detected)
- STL mesh generation and export (`write_stl_binary`, `create_prism_mesh`, `create_cylinder_mesh`)

**CRITICAL: Seed reproducibility must be preserved.** The Brownian walk uses `numpy.random` (NOT Numba's RNG). For a given seed, `generate_agglomerate()` must produce bitwise-identical rod positions and directions. Any optimization to the distance computation must not alter floating-point results — use explicit element-wise dot products (`v[:, 0]*u[0] + v[:, 1]*u[1] + v[:, 2]*u[2]`) rather than `v @ u` or `np.einsum`, which use BLAS routines with different FP accumulation order.

**Numba is optional.** The import uses `try/except` with a `_HAS_NUMBA` flag. When numba is unavailable, distance computation falls back to the numpy-vectorized path. Both paths produce identical results.

### Batch generation: `generate_batch.py`

Generates multiple agglomerates at different sizes, writes STLs and `metadata.json` with rod positions/directions/bounding boxes.

### Shadow projection: `generate_shadow_capsule.py`

The preferred shadow method for rod agglomerates. Projects each rod analytically as a 2D stadium (buffered LineString via shapely), unions them, and extrudes via constrained Delaunay triangulation. Requires `metadata.json` from `generate_batch.py`.

Other shadow methods (`generate_shadow_extrusion.py`, `generate_shadow_extrusion_smooth.py`) work directly from STL files but are lower quality.

### Bounding box extraction: `extract_bbox_csv.py`

```bash
python extract_bbox_csv.py <output_folder>
```

Reads `metadata.json` if available (fast), otherwise parses STL files directly. Outputs `bounding_boxes.csv`.

### Analysis scripts

- `analyze_batch.py` — fractal dimensions and plots
- `analyze_batch_paper_method.py` — 2D projection box-counting (paper method)
- `analyze_fractal_dimension.py` — fractal dimension vs particle count study

### Experiment runner scripts

- `run_experiment.py` — main experiment: multiple rod lengths, clump sizes, seeds, and shadows
- `run_experiment_large.py` — extends experiments with large clump sizes (500, 1000, 2000)
- `run_9um_small_clumps.py` — 9um rods at small clump sizes (15, 20, 25) with random seeds

These are one-off scripts. They import from `agglomerate.py`, `generate_batch.py`, and `generate_shadow_capsule.py`.

### Temporary/development files (can be ignored)

- `agglomerate_fast.py` — standalone Numba prototype (superseded by JIT in `agglomerate.py`)
- `bench_numba.py` — benchmark script for comparing numpy vs numba
- `test_optimization.py` — identity test against reference data

## Running Tests

```bash
pytest tests/ -v
```

Tests are in `tests/test_agglomerate.py`. They cover the core `agglomerate.py` module (geometry, collisions, generation, mesh, STL output).

## Performance Notes

- The bottleneck is `_jit_min_surface_distance` (or `min_distance_to_agglomerate_fast` without numba) — called on every Brownian step, O(n) per call where n is current agglomerate size.
- With numba: n=100 takes ~0.5s, n=500 ~6s, n=1000 ~25s, n=2000 ~60s.
- Without numba (vectorized numpy only): roughly 2-3x slower.
- Original unoptimized code: roughly 20x slower than numba.
- Cost scales as O(n^2) overall (n particles, each requiring O(n) distance checks per Brownian step).
- KD-tree spatial indexing was tested and found unhelpful — rod length (4000-9000nm) makes the pruning threshold too generous.

## Output Directory Conventions

Output directories follow the pattern: `output_{YYYYMMDD}_{length}um_{diameter}nm/`

Inside each:
- `seed_{N}/` — batch output per seed (STLs + `metadata.json`)
- `seed_{N}_large/` — large clump extensions
- `shadows/` — shadow STL files
- `shadow_summary.csv` — shadow areas for all clumps
- `bounding_boxes.csv` — bounding box dimensions (generated by `extract_bbox_csv.py`)

## Dependencies

All listed in `requirements.txt` and `pyproject.toml`:
- **numpy** — core numerics
- **matplotlib** — analysis/plotting
- **shapely** — polygon boolean ops for shadow projection
- **scipy** — Delaunay triangulation for smooth shadows
- **triangle** — constrained Delaunay for capsule shadows
- **numba** (optional) — JIT compilation for ~20x total speedup
