"""
Realistic tabletop disturbance generation.

Approximates:
- building vibration
- environmental motion
- support resonance
- occasional bumps/impulses

NOT true earthquake simulation.
"""

from __future__ import annotations

import numpy as np


class DisturbanceGenerator:
    def __init__(
        self,
        dt: float,
        broadband_gain: float = 0.05,
        vibration_freqs: list[float] | None = None,
        vibration_amps: list[float] | None = None,
        impulse_probability: float = 0.001,
        impulse_magnitude: float = 0.3,
    ) -> None:

        self.dt = dt

        self.broadband_gain = broadband_gain

        self.vibration_freqs = (
            vibration_freqs or [3.0, 7.0]
        )

        self.vibration_amps = (
            vibration_amps or [0.03, 0.02]
        )

        self.impulse_probability = impulse_probability
        self.impulse_magnitude = impulse_magnitude

        self.lowpass_state = 0.0

    def sample(
        self,
        t: float,
    ) -> float:

        # Low-frequency colored noise
        white = np.random.normal(
            0.0,
            self.broadband_gain,
        )

        self.lowpass_state += 0.02 * (
            white - self.lowpass_state
        )

        broadband = self.lowpass_state

        # Narrowband vibration
        vibration = 0.0

        for f, a in zip(
            self.vibration_freqs,
            self.vibration_amps,
        ):
            vibration += a * np.sin(
                2.0 * np.pi * f * t
            )

        # Random impulse disturbances
        impulse = 0.0

        if np.random.rand() < self.impulse_probability:
            impulse = np.random.normal(
                0.0,
                self.impulse_magnitude,
            )

        return (
            broadband
            + vibration
            + impulse
        )