"""
Reaction-wheel PID controller interface.

This module provides:
- abstract PID interface
- reaction-wheel pendulum stabilizing controller

The controller computes a normalized corrective torque signal using:
    tau = -(kp * theta + kd * omega + ki * integral(theta))

The environment remains responsible for:
- wheel dynamics
- actuator lag
- torque saturation
- pendulum integration
- reaction-wheel physics

This keeps the architecture modular and compatible with:
- NEAT supplementary control
- hybrid PID + NN blending
- hardware deployment
- future multi-axis systems
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from simulation.sensors import SensorReading


class PIDInterface(ABC):
    """
    Abstract controller interface.

    Controllers return normalized torque commands
    approximately within [-1, 1].
    """

    @abstractmethod
    def compute(self, sensors: SensorReading) -> float:
        raise NotImplementedError


class ReactionWheelPIDController(PIDInterface):
    """
    Stabilizing PID controller for a reaction-wheel pendulum.

    Uses:
    - pendulum angle
    - angular velocity
    - integral error accumulation

    The controller computes a normalized torque request
    which is later converted into physical torque by
    HybridController.

    This controller intentionally does NOT simulate:
    - wheel inertia
    - actuator lag
    - motor saturation

    because those are already handled inside the
    environment physics model.
    """

    def __init__(
        self,
        kp: float = 1.8,
        ki: float = 0.0,
        kd: float = 0.45,
        dt: float = 0.01,
        integral_limit: float = 2.0,
    ) -> None:

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.dt = dt

        self.integral_limit = integral_limit

        self.integral = 0.0

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """
        Wrap angle into [-pi, pi].

        Prevents instability when pendulum rotates
        across the branch cut.
        """

        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute(self, sensors: SensorReading) -> float:

        # Recover physical units from normalized sensors.
        theta = sensors.angle * np.pi
        omega = sensors.angular_velocity * 10.0

        theta = self.wrap_angle(theta)

        # Integral accumulation
        self.integral += theta * self.dt

        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        # PID control law
        torque = -(
            self.kp * theta
            + self.kd * omega
            + self.ki * self.integral
        )

        # Normalize output into [-1, 1]
        torque = np.clip(torque, -1.0, 1.0)

        return float(torque)
