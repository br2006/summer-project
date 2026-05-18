"""
External PID controller interface (placeholder — not a real PID implementation).

In deployment, u_PID would come from existing stabilizing firmware/software.
The NEAT layer must never replace primary stabilization; it only supplements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from simulation.sensors import SensorReading


class PIDInterface(ABC):
    """Abstract PID: implement on hardware or replace with your production controller."""

    @abstractmethod
    def compute(self, sensors: SensorReading) -> float:
        """
        Return normalized corrective torque in approximately [-1, 1].

        Real systems may return physical units; normalize before hybrid blending.
        """
        raise NotImplementedError


class PlaceholderPIDController(PIDInterface):
    """
    Simple stabilizing placeholder mimicking an external PID.

    Uses angle and angular velocity only (not seismic input) to keep the pendulum
    upright while the NEAT network shapes resonance / spectral response.
    """

    def __init__(self, kp: float = 1.2, kd: float = 0.4) -> None:
        self.kp = kp
        self.kd = kd

    def compute(self, sensors: SensorReading) -> float:
        u = -self.kp * sensors.angle - self.kd * sensors.angular_velocity
        return max(-1.0, min(1.0, u))
