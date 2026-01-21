# Fluffy Clumps

A Python tool for generating 3D nanoparticle agglomerates via Brownian collision simulation. Produces STL files suitable for 3D printing, visualization, or further simulation.

## Background

This implementation is based on the algorithm described in [Abomailek et al., Small 2025](https://doi.org/10.1002/smll.202409673), which simulates how nanorods/nanowires aggregate through random Brownian motion and collisions. The resulting agglomerates exhibit fractal dimensions typically in the range of 1.5-1.8, characteristic of extended nanorod clumps.

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using Poetry

```bash
poetry install
```

## Usage

### Basic Usage

Generate a 10-particle agglomerate with default parameters:

```bash
python agglomerate.py
```

### Command Line Options

```
usage: agglomerate.py [-h] [-n NUM_PARTICLES] [-l LENGTH] [-d DIAMETER]
                      [-o OUTPUT] [--ascii] [-s SEED] [--shape {cylinder,prism}]
                      [--segments SEGMENTS] [--fractal] [-q]

Generate nanoparticle agglomerates via Brownian collision simulation

options:
  -h, --help            show this help message and exit
  -n, --num-particles   Number of particles in agglomerate (default: 10)
  -l, --length          Length of each nanorod in nm (default: 100)
  -d, --diameter        Diameter of each nanorod in nm (default: 10)
  -o, --output          Output STL filename (default: agglomerate.stl)
  --ascii               Write ASCII STL instead of binary
  -s, --seed            Random seed for reproducibility
  --shape               Shape of particles: cylinder or prism (default: cylinder)
  --segments            Number of segments for cylinder mesh (default: 16)
  --fractal             Calculate and print fractal dimension
  -q, --quiet           Suppress progress output
```

### Examples

Generate a 20-particle agglomerate with cylindrical rods:

```bash
python agglomerate.py -n 20 -l 100 -d 10 -o my_agglomerate.stl
```

Generate with rectangular prism-shaped rods:

```bash
python agglomerate.py -n 15 --shape prism -o prism_agglomerate.stl
```

Generate with a specific random seed for reproducibility:

```bash
python agglomerate.py -n 10 -s 42 -o reproducible.stl
```

Calculate fractal dimension of the generated agglomerate:

```bash
python agglomerate.py -n 25 --fractal -o large_agglomerate.stl
```

Generate high-resolution cylinders (more mesh segments):

```bash
python agglomerate.py -n 10 --segments 32 -o smooth_cylinders.stl
```

### Python API

You can also use the module programmatically:

```python
from agglomerate import generate_agglomerate, write_stl_binary, calculate_fractal_dimension

# Generate agglomerate
agglomerate = generate_agglomerate(
    num_particles=20,
    length=100.0,
    diameter=10.0,
    seed=42
)

# Export as STL (cylinder shape)
write_stl_binary('output.stl', agglomerate, shape='cylinder')

# Or as rectangular prisms
write_stl_binary('output_prisms.stl', agglomerate, shape='prism')

# Calculate fractal dimension
fd = calculate_fractal_dimension(agglomerate)
print(f"Fractal dimension: {fd:.2f}")
```

## Batch Generation

Generate multiple agglomerates with varying particle counts:

```bash
python generate_batch.py --n-values 5,10,20,50,100 --length 100 --diameter 10
```

This creates a timestamped output directory containing:
- STL files for each agglomerate
- `metadata.json` with rod positions, directions, and bounding box dimensions

### Batch Generation Options

```
--n-values      Comma-separated particle counts (default: 5,10,20,50,100)
--n-range       Range as start:stop:step (e.g., 5:100:5)
-l, --length    Rod length in nm (default: 100)
-d, --diameter  Rod diameter in nm (default: 10)
-o, --output-dir  Output directory (default: output_YYYYMMDD_HHMMSS)
--shape         cylinder or prism (default: cylinder)
-s, --seed      Fixed seed for all agglomerates (default: random per agglomerate)
--ascii         Write ASCII STL instead of binary
```

### Extract Bounding Box Data

Extract bounding box dimensions to CSV for analysis:

```bash
python extract_bbox_csv.py <output_folder>
```

Creates `bounding_boxes.csv` with columns: `file`, `bbox_L`, `bbox_W`, `bbox_H`.

## Analysis Tools

### Analyze Batch

Calculate fractal dimensions and generate plots for a previously generated batch:

```bash
python analyze_batch.py <batch_directory>
```

### Analyze with Paper Method

Use the 2D projection box-counting method from Abomailek et al.:

```bash
python analyze_batch_paper_method.py <batch_directory>
```

### Fractal Dimension Study

Generate agglomerates and plot fractal dimension vs particle count:

```bash
python analyze_fractal_dimension.py
```

## Algorithm

The simulation follows these steps:

1. Place an initial nanorod at the origin
2. Define an outer sphere radius `R_out = 3 × length`
3. For each new particle:
   - Place it at a random point on the `R_out` sphere
   - Simulate Brownian motion by moving the particle by `d_min` (minimum distance to agglomerate) in a random direction
   - If collision occurs: add particle to agglomerate
   - If particle escapes beyond `R_out`: calculate escape probability `P_esc = 1 - R_out/d`
     - With probability `P_esc`: particle escapes, start with new particle
     - Otherwise: redirect particle back to sphere using angular distribution from Eq. 29
4. Repeat until desired number of particles is reached

## Output Format

The tool generates STL (stereolithography) files, which can be:
- Opened in 3D viewers (MeshLab, Blender, etc.)
- Used for 3D printing
- Imported into simulation software
- Converted to other mesh formats

Both binary (default, smaller file size) and ASCII formats are supported.

## Particle Shapes

Two shapes are available for representing nanorods:

- **Cylinder** (default): Smooth cylindrical rods with configurable mesh resolution
- **Prism**: Rectangular prisms with square cross-section (12 triangles per rod)

## Testing

The project includes a comprehensive test suite using pytest.

### Install Test Dependencies

```bash
# Using pip
pip install -r requirements-dev.txt

# Using Poetry (automatically includes dev dependencies)
poetry install
```

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test class
pytest tests/test_agglomerate.py::TestNanorod

# Run a specific test
pytest tests/test_agglomerate.py::TestNanorod::test_nanorod_creation

# Run with coverage (requires pytest-cov)
pip install pytest-cov
pytest --cov=agglomerate --cov-report=term-missing
```

### Test Categories

The test suite covers:

| Category | Description |
|----------|-------------|
| `TestNanorod` | Nanorod class creation, endpoints, and copying |
| `TestGeometryFunctions` | Random vectors, sphere sampling, segment distances |
| `TestCollisionDetection` | Distance calculations and collision checks |
| `TestEscapeAngle` | Escape angle probability sampling |
| `TestAgglomerateGeneration` | Full agglomerate generation and reproducibility |
| `TestMeshGeneration` | Cylinder and prism mesh creation |
| `TestSTLOutput` | Binary and ASCII STL file writing |
| `TestFractalDimension` | Fractal dimension calculation |
| `TestIntegration` | End-to-end workflow tests |

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib (for analysis and plotting scripts)

### Development Dependencies

- pytest
- pytest-cov

## References

- Abomailek et al., "Fractal Dimension of Nanorod Agglomerates", Small 2025, DOI: [10.1002/smll.202409673](https://doi.org/10.1002/smll.202409673)

## License

MIT License
