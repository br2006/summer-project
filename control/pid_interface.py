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
        kp: float = 2.8,
        ki: float = 0.7,
        kd: float = 0.06,
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

        # Proportional + derivative terms.
        p_term = self.kp * theta
        d_term = self.kd * omega

        # Conditional-integration anti-windup:
        # 1) predict unclipped output with current integral,
        # 2) if saturated and error would push further into saturation,
        #    freeze the integrator for this step,
        # 3) otherwise integrate normally.
        i_term_current = self.ki * self.integral
        unclipped_current = p_term + d_term + i_term_current
        saturated_high = unclipped_current > 1.0
        saturated_low = unclipped_current < -1.0
        pushes_further_high = theta > 0.0
        pushes_further_low = theta < 0.0

        should_freeze_integrator = (
            (saturated_high and pushes_further_high)
            or (saturated_low and pushes_further_low)
        )

        if not should_freeze_integrator:
            self.integral += theta * self.dt
            self.integral = np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit,
            )

        # PID control law.
        # Sign convention note:
        # In the environment dynamics, motor torque acts on the pendulum body
        # as tau_body = -tau_motor. Therefore, a positive theta near upright
        # requires a positive motor command to create restoring negative body torque.
        torque = (
            p_term
            + d_term
            + self.ki * self.integral
        )

        # Normalize output into [-1, 1]
        torque = np.clip(torque, -1.0, 1.0)

        return float(torque)




