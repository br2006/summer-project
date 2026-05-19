"""
Plotting examples: fitness history, FFT, rollout (no training required).

Run from project root:
  python examples/plotting_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from signal_processing.fft import compute_fft
from visualisation.plots import plot_fft, plot_fitness_history, plot_species_history


def main() -> None:
    # Synthetic demo data
    best = [0.5 + 0.1 * i + 0.05 * np.random.randn() for i in range(20)]
    mean = [b - 0.15 for b in best]
    species = [5, 6, 7, 6, 5, 4, 5, 6, 5, 4] * 2

    plot_fitness_history(best, mean, show=True)
    plot_species_history(species, show=True)

    t = np.linspace(0, 5, 500)
    signal = 0.3 * np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
    plot_fft(signal, sample_rate_hz=100.0, title="Example FFT", show=True)


if __name__ == "__main__":
    main()
