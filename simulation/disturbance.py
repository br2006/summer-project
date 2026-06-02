"""
Realistic tabletop base-acceleration disturbance generation.

The generated signal represents what a base-mounted accelerometer could measure
on a demonstration tabletop:
- slow building/table drift
- human footstep cadence
- damped table ringing after each step
- occasional taps/bumps
- accelerometer noise

It is intentionally not a true earthquake simulator.
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
        model: str = "sinusoidal",
        footstep_rate_hz: float = 1.4,
        footstep_jitter: float = 0.12,
        footstep_accel_mps2: float = 0.25,
        footstep_pulse_width_s: float = 0.06,
        table_ring_freqs_hz: list[float] | None = None,
        table_ring_amps_mps2: list[float] | None = None,
        table_ring_decay_s: float = 0.30,
        accelerometer_noise_std_mps2: float = 0.015,
    ) -> None:

        self.dt = dt
        self.model = model

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

        # Footstep/tabletop model parameters. Values are in acceleration units
        # (m/s^2), matching a base-mounted accelerometer.
        self.footstep_rate_hz = footstep_rate_hz
        self.footstep_jitter = footstep_jitter
        self.footstep_accel_mps2 = footstep_accel_mps2
        self.footstep_pulse_width_s = footstep_pulse_width_s
        self.table_ring_freqs_hz = table_ring_freqs_hz or [8.0, 14.0]
        self.table_ring_amps_mps2 = table_ring_amps_mps2 or [0.08, 0.04]
        self.table_ring_decay_s = table_ring_decay_s
        self.accelerometer_noise_std_mps2 = accelerometer_noise_std_mps2

        self._next_footstep_time = self._draw_next_footstep_time(0.0)
        self._footstep_times: list[float] = []

    def _draw_next_footstep_time(self, current_time: float) -> float:
        """Draw the next footstep time with small cadence jitter."""
        nominal_period = 1.0 / max(self.footstep_rate_hz, 1e-6)
        jitter = np.random.normal(0.0, self.footstep_jitter * nominal_period)
        return current_time + max(0.25 * nominal_period, nominal_period + jitter)

    def _sample_sinusoidal(self, t: float) -> float:
        """Backward-compatible simple colored-noise + sine disturbance."""

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

    def _sample_footstep_tabletop(self, t: float) -> float:
        """Sample a realistic tabletop footstep/base-accelerometer signal."""

        while t >= self._next_footstep_time:
            self._footstep_times.append(self._next_footstep_time)
            self._next_footstep_time = self._draw_next_footstep_time(
                self._next_footstep_time
            )

        # Keep only recent footsteps that can still contribute ringing.
        history_window = max(1.5, 5.0 * self.table_ring_decay_s)
        self._footstep_times = [
            ts for ts in self._footstep_times if t - ts <= history_window
        ]

        # Slow floor/building drift measured by the accelerometer.
        white = np.random.normal(0.0, self.broadband_gain)
        self.lowpass_state += 0.02 * (white - self.lowpass_state)
        base = self.lowpass_state

        footstep = 0.0
        for ts in self._footstep_times:
            age = t - ts
            if age < 0.0:
                continue

            # Main low-frequency step impulse: a short Gaussian pulse.
            pulse = self.footstep_accel_mps2 * np.exp(
                -0.5 * (age / max(self.footstep_pulse_width_s, 1e-6)) ** 2
            )

            # Table response: damped ringing from structural modes.
            ring = 0.0
            decay = np.exp(-age / max(self.table_ring_decay_s, 1e-6))
            for freq, amp in zip(
                self.table_ring_freqs_hz,
                self.table_ring_amps_mps2,
            ):
                ring += amp * decay * np.sin(2.0 * np.pi * freq * age)

            footstep += pulse + ring

        tap = 0.0
        if np.random.rand() < self.impulse_probability:
            tap = np.random.normal(0.0, self.impulse_magnitude)

        noise = np.random.normal(0.0, self.accelerometer_noise_std_mps2)

        return base + footstep + tap + noise

    def sample(
        self,
        t: float,
    ) -> float:
        if self.model == "footstep_tabletop":
            return self._sample_footstep_tabletop(t)
        return self._sample_sinusoidal(t)