"""
Visualization utilities for training diagnostics and spectral analysis.

Use these after training/evaluation to inspect fitness trends, FFT spectra,
and pendulum response time series.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from signal_processing.fft import compute_fft
from simulation.pendulum_env import SimulationResult


def plot_fitness_history(
    best: List[float],
    mean: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot best (and optional mean) fitness per generation."""
    fig, ax = plt.subplots(figsize=(8, 4))
    gens = range(len(best))
    ax.plot(gens, best, label="Best fitness", color="steelblue", linewidth=2)
    if mean is not None:
        ax.plot(gens, mean, label="Mean fitness", color="coral", alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("NEAT Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_species_history(
    species_counts: List[int],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot number of species over generations (diversity indicator)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(species_counts, color="seagreen", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of species")
    ax.set_title("Species Diversity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_weight_schedule_history(
    generations: List[float],
    stability_weights: List[float],
    amplification_weights: List[float],
    sigmoid_values: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot generation-dependent fitness schedule diagnostics.

    This visualization helps verify that the weighting schedule transitions
    smoothly from stability-dominant to amplification-dominant optimization.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(generations, stability_weights, label="Stability weight", color="royalblue", linewidth=2)
    ax.plot(generations, amplification_weights, label="Amplification weight", color="darkorange", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weight")
    ax.set_title("Generation-Dependent Fitness Weights")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if sigmoid_values is not None and len(sigmoid_values) == len(generations):
        ax2 = ax.twinx()
        ax2.plot(generations, sigmoid_values, label="Sigmoid(g)", color="gray", linestyle="--", alpha=0.8)
        ax2.set_ylabel("Sigmoid output")
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fft(
    signal: np.ndarray,
    sample_rate_hz: float,
    title: str = "FFT",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot amplitude spectrum of a 1D signal."""
    freqs, amp = compute_fft(signal, sample_rate_hz)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, amp, color="purple")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, min(10.0, freqs[-1] if len(freqs) else 10.0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rollout(
    result: SimulationResult,
    target_band_hz: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Multi-panel plot: angle, torques, and FFT with target band highlighted.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    axes[0].plot(result.time, result.angle, label="Angle (rad)")
    axes[0].plot(result.time, result.seismic_input * 0.1, label="Seismic (scaled)", alpha=0.6)
    axes[0].set_ylabel("Angle")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.time, result.pid_output, label="PID (norm)")
    axes[1].plot(result.time, result.nn_output, label="NN (norm)")
    axes[1].plot(result.time, result.total_torque, label="Total torque", alpha=0.7)
    axes[1].set_ylabel("Control")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    sample_rate = 1.0 / (result.time[1] - result.time[0]) if len(result.time) > 1 else 100.0
    freqs, amp = compute_fft(result.angle, sample_rate)
    axes[2].plot(freqs, amp, color="purple")
    if target_band_hz and len(target_band_hz) >= 2:
        axes[2].axvspan(target_band_hz[0], target_band_hz[1], alpha=0.2, color="green", label="Target band")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Pendulum angle spectrum")
    axes[2].set_xlim(0, 8)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Rollout: hybrid PID + NEAT control")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_resonance_comparison(
    result: SimulationResult,
    target_band_hz: List[float],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Compare target-band vs noise-band power (amplification diagnostic)."""
    sample_rate = 1.0 / (result.time[1] - result.time[0]) if len(result.time) > 1 else 100.0
    from signal_processing.spectral_metrics import compute_spectral_metrics

    m = compute_spectral_metrics(
        result.angle,
        result.seismic_input,
        sample_rate,
        target_band_hz,
        [3.0, 8.0],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Target band", "Noise band"]
    values = [m.target_band_power, m.noise_band_power]
    ax.bar(labels, values, color=["green", "gray"])
    ax.set_ylabel("Band power")
    ax.set_title(f"Amplification ratio ≈ {m.amplification_ratio:.2f}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
