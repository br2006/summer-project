"""
FFT utilities for fitness evaluation and visualization.

FFT is used to measure resonance amplification, target-band gain, and noise
rejection. Raw FFT vectors are NOT fed into the neural network in this prototype.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_fft(
    signal: np.ndarray,
    sample_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided amplitude spectrum.

    Returns
    -------
    freqs : frequency bins (Hz)
    amplitude : magnitude spectrum (same length as freqs)
    """
    n = len(signal)
    if n < 4:
        return np.array([0.0]), np.array([0.0])
    windowed = signal * np.hanning(n)
    spectrum = rfft(windowed)
    freqs = rfftfreq(n, d=1.0 / sample_rate_hz)
    amplitude = np.abs(spectrum) * 2.0 / n
    return freqs, amplitude
