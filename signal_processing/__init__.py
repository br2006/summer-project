"""FFT and spectral metrics for fitness evaluation and diagnostics."""

from signal_processing.fft import compute_fft
from signal_processing.spectral_metrics import (
    band_power,
    compute_spectral_metrics,
    target_band_amplification,
)

__all__ = [
    "compute_fft",
    "band_power",
    "compute_spectral_metrics",
    "target_band_amplification",
]
