"""
Sensor abstraction with realistic IMU imperfections.

The NN never receives perfect state information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorReading:
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
    @abstractmethod
    def read(self) -> SensorReading:
        raise NotImplementedError


class SimulatedSensor(SensorInterface):
    """
    Simulates:
    - Gaussian sensor noise
    - gyro bias drift
    - imperfect measurements
    """

    def __init__(
        self,
        angle_scale: float = np.pi,
        velocity_scale: float = 10.0,
        accel_scale: float = 5.0,
        wheel_scale: float = 40.0,
        angle_noise_std: float = 0.002,
        gyro_noise_std: float = 0.01,
        gyro_bias_drift_rate: float = 0.0001,
    ) -> None:

        self.angle_scale = angle_scale
        self.velocity_scale = velocity_scale
        self.accel_scale = accel_scale
        self.wheel_scale = wheel_scale

        self.angle_noise_std = angle_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_drift_rate = gyro_bias_drift_rate

        self.gyro_bias = 0.0

        self._state = dict(
            angle=0.0,
            omega=0.0,
            accel=0.0,
            wheel_omega=0.0,
        )

    def reset(self) -> None:
        """Reset sensor internal state between episodes."""
        self.gyro_bias = 0.0
        self._state = dict(
            angle=0.0,
            omega=0.0,
            accel=0.0,
            wheel_omega=0.0,
        )

    def update_raw(
        self,
        angle: float,
        omega: float,
        accel: float,
        wheel_omega: float,
    ) -> None:
        self._state = dict(
            angle=angle,
            omega=omega,
            accel=accel,
            wheel_omega=wheel_omega,
        )

    def read(self) -> SensorReading:
        s = self._state

        # Slowly drifting gyro bias
        self.gyro_bias += np.random.normal(
            0.0,
            self.gyro_bias_drift_rate,
        )

        angle_measured = (
            s["angle"]
            + np.random.normal(0.0, self.angle_noise_std)
        )

        omega_measured = (
            s["omega"]
            + self.gyro_bias
            + np.random.normal(0.0, self.gyro_noise_std)
        )

        return SensorReading(
            angle=np.clip(angle_measured / self.angle_scale, -1.0, 1.0),
            angular_velocity=np.clip(
                omega_measured / self.velocity_scale,
                -1.0,
                1.0,
            ),
            base_acceleration=np.clip(
                s["accel"] / self.accel_scale,
                -1.0,
                1.0,
            ),
            wheel_velocity=np.clip(
                s["wheel_omega"] / self.wheel_scale,
                -1.0,
                1.0,
            ),
        )