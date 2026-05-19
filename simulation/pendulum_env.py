"""
Simple reaction-wheel inverted pendulum simulation (placeholder physics).

This is intentionally low-fidelity: the goal is a debuggable end-to-end NEAT loop,
not accurate multibody dynamics. Replace integrate_step() later with a real model.

Control architecture:
  u_total = u_PID + alpha * u_NN
where u_PID comes from an external placeholder and u_NN from the NEAT network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from control.hybrid_controller import HybridController
from control.pid_interface import PlaceholderPIDController
from neat.genome import Genome
from neat.network import FeedforwardNetwork
from simulation.actuators import SimulatedActuator
from simulation.sensors import SimulatedSensor


@dataclass
class SimulationResult:
    """
    Time-series logs from one rollout used for fitness and FFT analysis.

    All arrays have length = num_steps.
    """

    time: np.ndarray
    angle: np.ndarray
    angular_velocity: np.ndarray
    base_acceleration: np.ndarray
    wheel_velocity: np.ndarray
    nn_output: np.ndarray
    pid_output: np.ndarray
    total_torque: np.ndarray
    seismic_input: np.ndarray


@dataclass
class PendulumEnvConfig:
    """Physical and simulation parameters (tunable via configs/project_config.yaml)."""

    dt: float = 0.01
    duration: float = 10.0
    gravity: float = 9.81
    pendulum_length: float = 0.5
    pendulum_mass: float = 0.2
    pendulum_damping: float = 0.05
    wheel_inertia: float = 0.01
    wheel_damping: float = 0.02
    max_wheel_torque: float = 0.5
    max_wheel_speed: float = 30.0
    nn_torque_scale: float = 0.15
    alpha: float = 0.3
    # Seismic excitation: sum of sinusoids in rad/s
    seismic_frequencies_hz: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    seismic_amplitudes: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.15])
    initial_angle: float = 0.15
    initial_omega: float = 0.0


class PendulumEnv:
    """
    Placeholder environment integrating pendulum + reaction wheel + hybrid control.
    """

    def __init__(self, config: Optional[PendulumEnvConfig] = None) -> None:
        self.config = config or PendulumEnvConfig()
        self.sensor = SimulatedSensor()
        self.actuator = SimulatedActuator()
        self.pid = PlaceholderPIDController()
        self.controller = HybridController(
            pid=self.pid,
            alpha=self.config.alpha,
            nn_torque_scale=self.config.nn_torque_scale,
        )

    def _seismic_accel(self, t: float) -> float:
        """Synthetic base acceleration (ground motion) driving the cart/pivot."""
        cfg = self.config
        a = 0.0
        for f, amp in zip(cfg.seismic_frequencies_hz, cfg.seismic_amplitudes):
            a += amp * np.sin(2.0 * np.pi * f * t)
        return a

    def _integrate_step(
        self,
        theta: float,
        omega: float,
        wheel_omega: float,
        torque: float,
        base_accel: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """
        Euler integration of a lumped inverted-pendulum + reaction wheel model.

        The pendulum feels gravity, damping, seismic base motion, and wheel torque.
        """
        cfg = self.config
        m, g, l = cfg.pendulum_mass, cfg.gravity, cfg.pendulum_length
        # Simplified moment balance (not rigorous — sufficient for NEAT prototyping).
        theta_dd = (m * g * l * np.sin(theta) - cfg.pendulum_damping * omega) / (
            m * l ** 2
        )
        theta_dd += base_accel / l
        theta_dd += torque / (cfg.wheel_inertia + m * l ** 2)

        wheel_dd = torque / cfg.wheel_inertia - cfg.wheel_damping * wheel_omega

        omega_new = omega + theta_dd * dt
        theta_new = theta + omega_new * dt
        wheel_new = wheel_omega + wheel_dd * dt
        wheel_new = np.clip(wheel_new, -cfg.max_wheel_speed, cfg.max_wheel_speed)
        return theta_new, omega_new, wheel_new

    def run_episode(
        self,
        network: Optional[FeedforwardNetwork] = None,
        genome: Optional[Genome] = None,
    ) -> SimulationResult:
        """
        Simulate one episode; provide either a FeedforwardNetwork or Genome.
        """
        if network is None and genome is not None:
            network = FeedforwardNetwork(genome)
        if network is None:
            raise ValueError("run_episode requires network or genome")

        cfg = self.config
        steps = int(cfg.duration / cfg.dt)
        t_arr = np.linspace(0, cfg.duration, steps, endpoint=False)

        theta = cfg.initial_angle
        omega = cfg.initial_omega
        wheel_omega = 0.0

        angles = np.zeros(steps)
        omegas = np.zeros(steps)
        accels = np.zeros(steps)
        wheels = np.zeros(steps)
        nn_out = np.zeros(steps)
        pid_out = np.zeros(steps)
        total_out = np.zeros(steps)
        seismic = np.zeros(steps)

        for i, t in enumerate(t_arr):
            base_a = self._seismic_accel(t)
            seismic[i] = base_a

            self.sensor.update_raw(theta, omega, base_a, wheel_omega)
            reading = self.sensor.read()
            u_nn = float(network.activate(reading.as_array())[0])
            u_pid = self.pid.compute(reading)
            torque = self.controller.compute_total_torque(u_pid, u_nn)
            torque = np.clip(torque, -cfg.max_wheel_torque, cfg.max_wheel_torque)

            nn_out[i] = u_nn
            pid_out[i] = u_pid
            total_out[i] = torque

            theta, omega, wheel_omega = self._integrate_step(
                theta, omega, wheel_omega, torque, base_a, cfg.dt
            )
            angles[i] = theta
            omegas[i] = omega
            accels[i] = base_a
            wheels[i] = wheel_omega

        return SimulationResult(
            time=t_arr,
            angle=angles,
            angular_velocity=omegas,
            base_acceleration=accels,
            wheel_velocity=wheels,
            nn_output=nn_out,
            pid_output=pid_out,
            total_torque=total_out,
            seismic_input=seismic,
        )

    def make_rollout_fn(self, genome: Genome) -> Callable[[], SimulationResult]:
        """Convenience wrapper for fitness evaluators."""
        net = FeedforwardNetwork(genome)

        def rollout() -> SimulationResult:
            return self.run_episode(network=net)

        return rollout
