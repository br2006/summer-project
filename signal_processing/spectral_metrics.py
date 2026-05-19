"""
Spectral fitness metrics derived from FFT analysis.

These metrics implement the A (amplification) and N (noise) terms in:
  F = w_s*S + w_a*A - w_n*N - w_e*E - w_u*U
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from signal_processing.fft import compute_fft


@dataclass
class SpectralMetrics:
    """Container for spectral diagnostics from one rollout."""

    target_band_power: float
    noise_band_power: float
    total_power: float
    amplification_ratio: float
    selectivity: float


def band_power(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    band_hz: List[float],
) -> float:
    """Integrate spectral power in [band_hz[0], band_hz[1]]."""
    if len(band_hz) < 2:
        return 0.0
    low, high = band_hz[0], band_hz[1]
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(amplitude[mask] ** 2))


def target_band_amplification(
    response_signal: np.ndarray,
    reference_signal: np.ndarray,
    sample_rate_hz: float,
    target_band_hz: List[float],
) -> float:
    """
    Ratio of target-band power in pendulum response vs seismic reference.

    Values > 1 indicate amplification of desired frequencies.
    """
    f_r, a_r = compute_fft(response_signal, sample_rate_hz)
    f_ref, a_ref = compute_fft(reference_signal, sample_rate_hz)
    p_resp = band_power(f_r, a_r, target_band_hz)
    p_ref = band_power(f_ref, a_ref, target_band_hz)
    if p_ref < 1e-12:
        return 0.0
    return p_resp / p_ref


def compute_spectral_metrics(
    angle_signal: np.ndarray,
    seismic_signal: np.ndarray,
    sample_rate_hz: float,
    target_band_hz: List[float],
    noise_band_hz: List[float],
) -> SpectralMetrics:
    """
    Compute amplification, noise power, and spectral selectivity for fitness.

    selectivity = target_power / (noise_power + epsilon) — higher is better.
    """
    freqs, amp = compute_fft(angle_signal, sample_rate_hz)
    target_p = band_power(freqs, amp, target_band_hz)
    noise_p = band_power(freqs, amp, noise_band_hz)
    total_p = float(np.sum(amp ** 2)) + 1e-12
    amp_ratio = target_band_amplification(
        angle_signal, seismic_signal, sample_rate_hz, target_band_hz
    )
    selectivity = target_p / (noise_p + 1e-9)
    return SpectralMetrics(
        target_band_power=target_p,
        noise_band_power=noise_p,
        total_power=total_p,
        amplification_ratio=amp_ratio,
        selectivity=selectivity,
    )
