"""
Analyze previously generated agglomerate batches.

Loads agglomerates from a batch directory and calculates fractal dimensions,
generates plots, and exports analysis results.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from agglomerate import Nanorod, calculate_fractal_dimension


def load_batch(batch_dir: str) -> dict:
    """
    Load a batch of agglomerates from a directory.

    Args:
        batch_dir: Directory containing metadata.json and STL files

    Returns:
        Dictionary with metadata and reconstructed agglomerates
    """
    metadata_file = os.path.join(batch_dir, 'metadata.json')

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"No metadata.json found in {batch_dir}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Reconstruct agglomerates from metadata
    params = metadata['parameters']
    length = params['length_nm']
    diameter = params['diameter_nm']

    agglomerates = []
    for agg_meta in metadata['agglomerates']:
        rods = []
        positions = agg_meta['rod_positions']
        directions = agg_meta['rod_directions']

        for pos, dir in zip(positions, directions):
            rod = Nanorod(
                center=np.array(pos),
                direction=np.array(dir),
                length=length,
                diameter=diameter
            )
            rods.append(rod)

        agglomerates.append({
            'n_particles': agg_meta['n_particles'],
            'seed': agg_meta['seed'],
            'filename': agg_meta['filename'],
            'rods': rods
        })

    return {
        'metadata': metadata,
        'agglomerates': agglomerates
    }


def calculate_all_fractal_dimensions(batch_data: dict, num_samples: int = 2000, verbose: bool = True) -> list:
    """
    Calculate fractal dimensions for all agglomerates in a batch.

    Args:
        batch_data: Data from load_batch()
        num_samples: Number of surface samples for box counting
        verbose: Print progress

    Returns:
        List of (n_particles, fractal_dimension) tuples
    """
    results = []

    if verbose:
        print("Calculating fractal dimensions...")
        print("-" * 60)

    for i, agg in enumerate(batch_data['agglomerates']):
        n = agg['n_particles']
        rods = agg['rods']

        if verbose:
            print(f"[{i+1}/{len(batch_data['agglomerates'])}] n={n}...", end=" ", flush=True)

        # Use more samples for larger agglomerates
        samples = min(5000, max(num_samples, n * 50))
        fd = calculate_fractal_dimension(rods, num_samples=samples)

        results.append({
            'n_particles': n,
            'fractal_dimension': fd,
            'seed': agg['seed'],
            'filename': agg['filename']
        })

        if verbose:
            print(f"D = {fd:.3f}")

    if verbose:
        print("-" * 60)

    return results


def plot_fractal_dimension(results: list, output_file: str = None, title: str = None, show: bool = True):
    """
    Plot fractal dimension vs particle count.

    Args:
        results: List of result dicts from calculate_all_fractal_dimensions()
        output_file: Path to save plot (optional)
        title: Custom title (optional)
        show: Whether to display the plot
    """
    n_values = [r['n_particles'] for r in results]
    fd_values = [r['fractal_dimension'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_values, fd_values, 'bo-', markersize=8, linewidth=2, label='Measured')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='D=1 (line)')
    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='D=1.5')
    ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='D=1.8')
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='D=2 (surface)')

    # Literature range
    ax.axhspan(1.5, 1.8, alpha=0.2, color='green', label='Literature range (1.5-1.8)')

    ax.set_xlabel('Number of Particles (n)', fontsize=12)
    ax.set_ylabel('Fractal Dimension', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Fractal Dimension vs Number of Particles', fontsize=14)

    ax.set_xlim(0, max(n_values) * 1.05)
    ax.set_ylim(0.8, 2.2)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()

    return fig, ax


def export_results(results: list, output_file: str):
    """
    Export results to CSV.

    Args:
        results: List of result dicts
        output_file: Output CSV path
    """
    with open(output_file, 'w') as f:
        f.write('n_particles,fractal_dimension,seed,filename\n')
        for r in results:
            f.write(f"{r['n_particles']},{r['fractal_dimension']:.4f},{r['seed']},{r['filename']}\n")

    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze previously generated agglomerate batches'
    )
    parser.add_argument('batch_dir', type=str,
                        help='Directory containing the batch (with metadata.json)')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of surface samples for fractal calculation (default: 2000)')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Output plot filename (default: fractal_analysis_TIMESTAMP.png)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV filename (default: fractal_analysis_TIMESTAMP.csv)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (just save)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Load batch
    print(f"Loading batch from: {args.batch_dir}")
    batch_data = load_batch(args.batch_dir)

    params = batch_data['metadata']['parameters']
    print(f"Batch timestamp: {batch_data['metadata']['timestamp']}")
    print(f"Parameters: length={params['length_nm']}nm, diameter={params['diameter_nm']}nm, shape={params['shape']}")
    print(f"Number of agglomerates: {len(batch_data['agglomerates'])}")

    # Calculate fractal dimensions
    results = calculate_all_fractal_dimensions(
        batch_data,
        num_samples=args.samples,
        verbose=not args.quiet
    )

    # Generate output filenames if not specified
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_plot is None:
        args.output_plot = os.path.join(args.batch_dir, f'fractal_analysis_{timestamp}.png')
    if args.output_csv is None:
        args.output_csv = os.path.join(args.batch_dir, f'fractal_analysis_{timestamp}.csv')

    # Export results
    export_results(results, args.output_csv)

    # Plot
    plot_fractal_dimension(
        results,
        output_file=args.output_plot,
        title=args.title,
        show=not args.no_show
    )

    # Print summary
    fd_values = [r['fractal_dimension'] for r in results]
    print("\nSummary:")
    print(f"  Min fractal dimension: {min(fd_values):.3f}")
    print(f"  Max fractal dimension: {max(fd_values):.3f}")
    print(f"  Mean fractal dimension: {np.mean(fd_values):.3f}")


if __name__ == '__main__':
    main()
