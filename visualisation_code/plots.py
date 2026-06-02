"""
Visualization utilities for training diagnostics, spectral analysis, and rollout diagnostics.

These helpers are used across training/evaluation scripts to keep plotting consistent
and presentation-ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from signal_processing.fft import compute_fft
from simulation.pendulum_env import SimulationResult

SPECTRUM_LINE_COLOR = "#4E79A7"
TARGET_BAND_COLOR = "#59A14F"
NOISE_BAND_COLOR = "#E15759"


def _dominant_peaks(
    signal: np.ndarray,
    sample_rate_hz: float,
    max_freq_hz: float = 25.0,
    n_peaks: int = 5,
) -> List[Tuple[float, float]]:
    """Return the strongest FFT peaks as (frequency_hz, amplitude)."""
    freqs, amp = compute_fft(signal, sample_rate_hz)
    mask = (freqs > 0.0) & (freqs <= max_freq_hz)
    if not np.any(mask):
        return []
    f_sel = freqs[mask]
    a_sel = amp[mask]
    peak_idx = np.argsort(a_sel)[-n_peaks:][::-1]
    return [(float(f_sel[i]), float(a_sel[i])) for i in peak_idx]


def _infer_sample_rate(result: SimulationResult) -> float:
    if len(result.time) <= 1:
        return 100.0
    dt = float(result.time[1] - result.time[0])
    return 1.0 / dt if dt > 0 else 100.0


def _annotate_peaks(
    ax: plt.Axes,
    peaks: Sequence[Tuple[float, float]],
    max_labels: int = 3,
) -> None:
    for freq, amp in peaks[:max_labels]:
        ax.annotate(
            f"{freq:.2f} Hz",
            xy=(freq, amp),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#333333",
        )


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
    """Plot generation-dependent fitness schedule diagnostics."""
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


def plot_training_summary(
    best: List[float],
    mean: Optional[List[float]] = None,
    species_counts: Optional[List[int]] = None,
    schedule_generations: Optional[List[float]] = None,
    stability_weights: Optional[List[float]] = None,
    amplification_weights: Optional[List[float]] = None,
    sigmoid_values: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Create a combined training-progress dashboard for presentation/analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    gens = np.arange(len(best))
    axes[0].plot(gens, best, label="Best fitness", color="steelblue", linewidth=2.2)
    if mean is not None and len(mean) == len(best):
        axes[0].plot(gens, mean, label="Mean fitness", color="coral", alpha=0.9, linewidth=1.8)
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("Training progress", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    if len(best) > 0:
        axes[0].scatter([gens[-1]], [best[-1]], color="steelblue", zorder=5)
        axes[0].annotate(
            f"Final best: {best[-1]:.3f}",
            xy=(gens[-1], best[-1]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
        )

    if species_counts and len(species_counts) > 0:
        axes[1].plot(np.arange(len(species_counts)), species_counts, color="seagreen", linewidth=2)
        axes[1].set_ylabel("# species")
        axes[1].set_title("Species diversity", fontsize=12)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Species history unavailable for this backend",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            color="0.4",
        )
        axes[1].set_ylabel("# species")
        axes[1].set_title("Species diversity", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    has_schedule = (
        schedule_generations is not None
        and stability_weights is not None
        and amplification_weights is not None
        and len(schedule_generations) > 0
    )
    if has_schedule:
        axes[2].plot(
            schedule_generations,
            stability_weights,
            label="Stability weight",
            color="royalblue",
            linewidth=2,
        )
        axes[2].plot(
            schedule_generations,
            amplification_weights,
            label="Amplification weight",
            color="darkorange",
            linewidth=2,
        )
        axes[2].set_ylabel("Weight")
        axes[2].set_title("Fitness-objective schedule", fontsize=12)
        axes[2].legend(loc="upper left")

        if sigmoid_values is not None and len(sigmoid_values) == len(schedule_generations):
            ax2 = axes[2].twinx()
            ax2.plot(
                schedule_generations,
                sigmoid_values,
                label="Sigmoid(g)",
                color="gray",
                linestyle="--",
                alpha=0.8,
            )
            ax2.set_ylabel("Sigmoid")
            ax2.set_ylim(0.0, 1.0)
    else:
        axes[2].text(
            0.5,
            0.5,
            "Weight schedule unavailable",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            color="0.4",
        )
        axes[2].set_ylabel("Weight")
        axes[2].set_title("Fitness-objective schedule", fontsize=12)

    axes[2].set_xlabel("Generation")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("NEAT training summary", fontsize=14)
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
    max_freq_hz: Optional[float] = None,
    annotate_peaks: bool = True,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot amplitude spectrum of a 1D signal."""
    freqs, amp = compute_fft(signal, sample_rate_hz)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(freqs, amp, color=SPECTRUM_LINE_COLOR, linewidth=1.6)

    if len(freqs):
        upper = freqs[-1]
        if max_freq_hz is not None:
            upper = min(upper, max_freq_hz)
        ax.set_xlim(0.0, upper)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if annotate_peaks and len(freqs) > 0:
        peaks = _dominant_peaks(signal, sample_rate_hz, max_freq_hz or freqs[-1])
        _annotate_peaks(ax, peaks)

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
    """Summarise a rollout with angle/acceleration, control signals, and spectrum."""

    time = result.time
    sample_rate = _infer_sample_rate(result)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=False)

    ax_angle = axes[0]
    ax_angle.plot(time, result.angle, color="steelblue", linewidth=2.0, label="Angle (rad)")
    ax_angle.set_ylabel("Angle (rad)", color="steelblue")
    ax_angle.tick_params(axis="y", labelcolor="steelblue")
    ax_angle.grid(True, alpha=0.3)

    ax_accel = ax_angle.twinx()
    ax_accel.plot(
        time,
        result.base_acceleration,
        color="darkorange",
        linewidth=1.6,
        alpha=0.85,
        label="Base accel. (m/s²)",
    )
    ax_accel.set_ylabel("Base acceleration (m/s²)", color="darkorange")
    ax_accel.tick_params(axis="y", labelcolor="darkorange")

    lines, labels = ax_angle.get_legend_handles_labels()
    lines2, labels2 = ax_accel.get_legend_handles_labels()
    ax_angle.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)

    ax_ctrl = axes[1]
    ax_ctrl.plot(time, result.pid_output, color="seagreen", linewidth=1.8, label="PID output")
    ax_ctrl.plot(time, result.nn_output, color="purple", linewidth=1.6, alpha=0.9, label="NN output")
    ax_ctrl.set_ylabel("Controller output (norm)")
    ax_ctrl.grid(True, alpha=0.3)

    ax_torque = ax_ctrl.twinx()
    ax_torque.plot(
        time,
        result.commanded_torque,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label="Commanded torque",
    )
    ax_torque.plot(time, result.actual_torque, color="#d62728", linewidth=1.6, label="Actual torque")
    ax_torque.set_ylabel("Torque (Nm)")

    ctrl_lines, ctrl_labels = ax_ctrl.get_legend_handles_labels()
    torque_lines, torque_labels = ax_torque.get_legend_handles_labels()
    ax_ctrl.legend(ctrl_lines + torque_lines, ctrl_labels + torque_labels, loc="upper right", frameon=False)

    ax_fft = axes[2]
    freqs, amp = compute_fft(result.angle, sample_rate)
    ax_fft.plot(freqs, amp, color=SPECTRUM_LINE_COLOR, linewidth=1.8)
    if target_band_hz and len(target_band_hz) >= 2:
        ax_fft.axvspan(
            target_band_hz[0],
            target_band_hz[1],
            alpha=0.18,
            color=TARGET_BAND_COLOR,
            label="Target band",
        )
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_title("Pendulum angle spectrum")
    if len(freqs):
        ax_fft.set_xlim(0, min(8.0, freqs[-1]))
        _annotate_peaks(ax_fft, _dominant_peaks(result.angle, sample_rate, max_freq_hz=8.0))
    ax_fft.grid(True, alpha=0.3)
    handles, labels = ax_fft.get_legend_handles_labels()
    if handles:
        ax_fft.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle("Rollout diagnostics: hybrid PID + NEAT controller", fontsize=14)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_frequency_diagnostics(
    result: SimulationResult,
    target_band_hz: Optional[List[float]] = None,
    noise_band_hz: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot FFTs of key signals from a rollout for frequency-domain analysis."""
    sample_rate = _infer_sample_rate(result)
    series = [
        ("Base accelerometer (m/s²)", result.base_acceleration),
        ("Pendulum angle (rad)", result.angle),
        ("NN output (norm)", result.nn_output),
        ("Actual torque (Nm)", result.actual_torque),
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(10, 10), sharex=True)
    for ax, (label, signal) in zip(axes, series):
        freqs, amp = compute_fft(signal, sample_rate)
        ax.plot(freqs, amp, linewidth=1.5, color=SPECTRUM_LINE_COLOR)
        if target_band_hz and len(target_band_hz) >= 2:
            ax.axvspan(target_band_hz[0], target_band_hz[1], alpha=0.18, color=TARGET_BAND_COLOR)
        if noise_band_hz and len(noise_band_hz) >= 2:
            ax.axvspan(noise_band_hz[0], noise_band_hz[1], alpha=0.12, color=NOISE_BAND_COLOR)
        peaks = _dominant_peaks(signal, sample_rate)
        peak_text = ", ".join(f"{f:.2f} Hz" for f, _ in peaks[:3])
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{label} FFT | strongest: {peak_text}")
        _annotate_peaks(ax, peaks)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(0, 25)
    fig.suptitle("Frequency diagnostics: tabletop footstep/base-accelerometer demo")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_frequency_diagnostics(
    result: SimulationResult,
    max_freq_hz: float = 25.0,
    n_peaks: int = 5,
) -> None:
    """Print dominant spectral peaks for key rollout signals."""
    sample_rate = _infer_sample_rate(result)
    series = [
        ("base_acceleration", result.base_acceleration),
        ("angle", result.angle),
        ("nn_output", result.nn_output),
        ("actual_torque", result.actual_torque),
    ]
    for name, signal in series:
        peaks = _dominant_peaks(signal, sample_rate, max_freq_hz, n_peaks)
        formatted = ", ".join(f"{freq:.2f} Hz ({amp:.3g})" for freq, amp in peaks)
        print(f"{name}: {formatted}")


def plot_resonance_comparison(
    result: SimulationResult,
    target_band_hz: List[float],
    noise_band_hz: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Compare target-band vs noise-band power (amplification diagnostic)."""
    sample_rate = _infer_sample_rate(result)
    from signal_processing.spectral_metrics import compute_spectral_metrics

    metrics = compute_spectral_metrics(
        result.angle,
        result.seismic_input,
        sample_rate,
        target_band_hz,
        noise_band_hz or [5.0, 20.0],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Target band", "Noise band"]
    values = [metrics.target_band_power, metrics.noise_band_power]
    ax.bar(labels, values, color=[TARGET_BAND_COLOR, "#BDBDBD"])
    ax.set_ylabel("Band power")
    ax.set_title(f"Amplification ratio ≈ {metrics.amplification_ratio:.2f}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
