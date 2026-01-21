"""
Analyze fractal dimension as a function of particle count.

Generates agglomerates with varying numbers of particles and plots
the fractal dimension vs particle count.
"""

import numpy as np
import matplotlib.pyplot as plt
from agglomerate import generate_agglomerate, calculate_fractal_dimension

def main():
    # Particle counts to test (logarithmically spaced for better coverage)
    n_values = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]

    fractal_dimensions = []

    print("Generating agglomerates and calculating fractal dimensions...")
    print("-" * 60)

    for n in n_values:
        print(f"Generating agglomerate with n={n} particles...", end=" ", flush=True)

        # Generate agglomerate with fixed seed for reproducibility
        agglomerate = generate_agglomerate(
            num_particles=n,
            length=100.0,
            diameter=10.0,
            seed=42,
            verbose=False
        )

        # Calculate fractal dimension with more samples for larger agglomerates
        num_samples = min(5000, max(1000, n * 50))
        fd = calculate_fractal_dimension(agglomerate, num_samples=num_samples)
        fractal_dimensions.append(fd)

        print(f"Fractal dimension = {fd:.3f}")

    print("-" * 60)
    print("Done generating all agglomerates.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_values, fractal_dimensions, 'bo-', markersize=8, linewidth=2, label='Measured')

    # Add horizontal reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='D=1 (line)')
    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='D=1.5')
    ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='D=1.8')
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='D=2 (surface)')

    # Add shaded region for expected range from literature (1.5-1.8)
    ax.axhspan(1.5, 1.8, alpha=0.2, color='green', label='Literature range (1.5-1.8)')

    ax.set_xlabel('Number of Particles (n)', fontsize=12)
    ax.set_ylabel('Fractal Dimension', fontsize=12)
    ax.set_title('Fractal Dimension vs Number of Particles in Agglomerate', fontsize=14)

    ax.set_xlim(0, max(n_values) + 20)
    ax.set_ylim(0.8, 2.2)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig('fractal_dimension_vs_n.png', dpi=150)
    print(f"\nPlot saved to: fractal_dimension_vs_n.png")

    # Also save data to CSV
    with open('fractal_dimension_data.csv', 'w') as f:
        f.write('n,fractal_dimension\n')
        for n, fd in zip(n_values, fractal_dimensions):
            f.write(f'{n},{fd:.4f}\n')
    print(f"Data saved to: fractal_dimension_data.csv")

    plt.show()


if __name__ == '__main__':
    main()
