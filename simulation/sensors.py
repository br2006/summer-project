"""
Sensor interface abstraction for future hardware integration.

Real hardware would implement read() by sampling IMU, encoders, and accelerometers.
The simulation provides the same normalized interface for NEAT training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorReading:
    """
    Normalized sensor bundle fed to the NEAT controller.

    All values are scaled to approximately [-1, 1] before network evaluation.
    """

    angle: float
    angular_velocity: float
    base_acceleration: float
    wheel_velocity: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.angle,
                self.angular_velocity,
                self.base_acceleration,
                self.wheel_velocity,
            ],
            dtype=np.float64,
        )


class SensorInterface(ABC):
    """Abstract sensor layer; subclass for real devices or simulation."""

    @abstractmethod
    def read(self) -> SensorReading:
        """Return the latest normalized sensor reading."""
        raise NotImplementedError


class SimulatedSensor(SensorInterface):
    """Wraps raw simulation state into normalized SensorReading values."""

    def __init__(
        self,
        angle_scale: float = np.pi,
        velocity_scale: float = 10.0,
        accel_scale: float = 5.0,
        wheel_scale: float = 20.0,
    ) -> None:
        self.angle_scale = angle_scale
        self.velocity_scale = velocity_scale
        self.accel_scale = accel_scale
        self.wheel_scale = wheel_scale
        self._state = dict(angle=0.0, omega=0.0, accel=0.0, wheel_omega=0.0)

    def update_raw(
        self,
        angle: float,
        omega: float,
        accel: float,
        wheel_omega: float,
    ) -> None:
        self._state = dict(
            angle=angle, omega=omega, accel=accel, wheel_omega=wheel_omega
        )

    def read(self) -> SensorReading:
        s = self._state
        return SensorReading(
            angle=np.clip(s["angle"] / self.angle_scale, -1.0, 1.0),
            angular_velocity=np.clip(s["omega"] / self.velocity_scale, -1.0, 1.0),
            base_acceleration=np.clip(s["accel"] / self.accel_scale, -1.0, 1.0),
            wheel_velocity=np.clip(s["wheel_omega"] / self.wheel_scale, -1.0, 1.0),
        )
