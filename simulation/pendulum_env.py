"""
Reaction-wheel inverted pendulum simulation with realistic second-order effects.

Still intentionally lightweight and planar for efficient NEAT training,
but includes:
- compound pendulum dynamics
- motor lag
- wheel saturation
- damping/friction
- noisy sensing
- tabletop disturbance approximation
- structural resonance approximation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from control.hybrid_controller import HybridController
from control.pid_interface import ReactionWheelPIDController
from neat.genome import Genome
from neat.network import FeedforwardNetwork
from simulation.actuators import SimulatedActuator
from simulation.compound_pendulum import CompoundPendulumProperties
from simulation.disturbance import DisturbanceGenerator
from simulation.sensors import SimulatedSensor


@dataclass
class SimulationResult:
    time: np.ndarray
    angle: np.ndarray
    angular_velocity: np.ndarray
    base_acceleration: np.ndarray
    wheel_velocity: np.ndarray

    nn_output: np.ndarray
    pid_output: np.ndarray

    commanded_torque: np.ndarray
    actual_torque: np.ndarray

    resonance_signal: np.ndarray

    seismic_input: np.ndarray


@dataclass
class PendulumEnvConfig:
    """Physical and simulation parameters."""

    dt: float = 0.01
    duration: float = 10.0

    gravity: float = 9.81

    # Physical subcomponents: rigid compound pendulum
    arm_mass: float = 0.20
    arm_length: float = 0.28

    motor_mass: float = 0.42
    motor_radius: float = 0.03
    motor_length: float = 0.05

    wheel_mass: float = 0.18
    wheel_radius: float = 0.045
    wheel_thickness: float = 0.012

    # Damping/friction
    pendulum_damping: float = 0.08
    wheel_damping: float = 0.03

    # Reaction wheel
    max_wheel_torque: float = 0.25
    max_wheel_speed: float = 120.0

    # Motor dynamics
    motor_time_constant: float = 0.04

    # NN blending
    nn_torque_scale: float = 0.15
    alpha: float = 0.3

    # Initial conditions
    initial_angle: float = 0.15
    initial_omega: float = 0.0

    # Resonance approximation
    resonance_frequency: float = 11.0
    resonance_gain: float = 0.015

    # Sensor noise
    angle_noise_std: float = 0.002
    gyro_noise_std: float = 0.01
    gyro_bias_drift_rate: float = 0.0001

    # Disturbance model
    broadband_noise_gain: float = 0.04

    vibration_freqs_hz: List[float] = field(
        default_factory=lambda: [3.0, 7.0]
    )

    vibration_amps: List[float] = field(
        default_factory=lambda: [0.03, 0.02]
    )

    impulse_probability: float = 0.001
    impulse_magnitude: float = 0.25


class PendulumEnv:
    """
    Planar reaction-wheel pendulum environment for NEAT training.
    """

    def __init__(
        self,
        config: Optional[PendulumEnvConfig] = None,
    ) -> None:

        self.config = config or PendulumEnvConfig()

        self.compound_properties = CompoundPendulumProperties(
            arm_mass=self.config.arm_mass,
            arm_length=self.config.arm_length,
            motor_mass=self.config.motor_mass,
            motor_radius=self.config.motor_radius,
            motor_length=self.config.motor_length,
            wheel_mass=self.config.wheel_mass,
            wheel_radius=self.config.wheel_radius,
            wheel_thickness=self.config.wheel_thickness,
            gravity=self.config.gravity,
        )

        self.mass_properties = self.compound_properties.debug_summary()
        self.total_mass = self.mass_properties["total_mass"]
        self.r_cm = self.mass_properties["center_of_mass"]
        self.total_inertia = self.mass_properties["total_inertia_about_pivot"]
        self.wheel_inertia = self.mass_properties["wheel_inertia_cm"]

        self._print_mass_property_debug()

        self.sensor = SimulatedSensor(
            angle_noise_std=self.config.angle_noise_std,
            gyro_noise_std=self.config.gyro_noise_std,
            gyro_bias_drift_rate=self.config.gyro_bias_drift_rate,
        )

        self.actuator = SimulatedActuator()

        self.pid = ReactionWheelPIDController(
          kp=1.8,
          ki=0.0,
          kd=0.45,
          dt=self.config.dt,
        )

        self.controller = HybridController(
            pid=self.pid,
            alpha=self.config.alpha,
            nn_torque_scale=self.config.nn_torque_scale,
        )

        self.disturbance = DisturbanceGenerator(
            dt=self.config.dt,
            broadband_gain=self.config.broadband_noise_gain,
            vibration_freqs=self.config.vibration_freqs_hz,
            vibration_amps=self.config.vibration_amps,
            impulse_probability=self.config.impulse_probability,
            impulse_magnitude=self.config.impulse_magnitude,
        )

        # Actual motor torque after lag dynamics.
        self.actual_torque = 0.0

    def _print_mass_property_debug(self) -> None:

        p = self.mass_properties

        print("[CompoundPendulumProperties] arm inertia about pivot:", p["arm_inertia_about_pivot"])
        print("[CompoundPendulumProperties] motor inertia about pivot:", p["motor_inertia_about_pivot"])
        print("[CompoundPendulumProperties] wheel inertia about pivot:", p["wheel_inertia_about_pivot"])
        print("[CompoundPendulumProperties] total mass:", p["total_mass"])
        print("[CompoundPendulumProperties] center of mass:", p["center_of_mass"])
        print("[CompoundPendulumProperties] total inertia about pivot:", p["total_inertia_about_pivot"])
        print("[CompoundPendulumProperties] natural frequency:", p["natural_frequency"])
        print("[CompoundPendulumProperties] oscillation period:", p["oscillation_period"])

    def _resonance_disturbance(
        self,
        t: float,
        torque: float,
    ) -> float:

        cfg = self.config

        amp = cfg.resonance_gain * abs(torque)

        return amp * np.sin(
            2.0 * np.pi * cfg.resonance_frequency * t
        )

    def _integrate_step(
        self,
        theta: float,
        omega: float,
        wheel_omega: float,
        torque_cmd: float,
        disturbance: float,
        resonance: float,
        dt: float,
    ) -> tuple[float, float, float]:

        cfg = self.config

        # Motor lag dynamics
        self.actual_torque += (
            dt / cfg.motor_time_constant
        ) * (
            torque_cmd - self.actual_torque
        )

        self.actual_torque = np.clip(
            self.actual_torque,
            -cfg.max_wheel_torque,
            cfg.max_wheel_torque,
        )

        # Wheel saturation
        if (
            wheel_omega >= cfg.max_wheel_speed
            and self.actual_torque > 0
        ):
            self.actual_torque = 0.0

        if (
            wheel_omega <= -cfg.max_wheel_speed
            and self.actual_torque < 0
        ):
            self.actual_torque = 0.0

        # Gravity torque
        tau_gravity = (
            self.total_mass
            * cfg.gravity
            * self.r_cm
            * np.sin(theta)
        )

        # Damping
        tau_damping = -cfg.pendulum_damping * omega

        # Total torque
        tau_total = (
            tau_gravity
            + tau_damping
            + disturbance
            + resonance
            + self.actual_torque
        )

        # Pendulum angular acceleration
        theta_dd = tau_total / self.total_inertia

        # Wheel dynamics
        wheel_dd = (
            self.actual_torque / self.wheel_inertia
            - cfg.wheel_damping * wheel_omega
        )

        # Euler integration
        omega_new = omega + theta_dd * dt
        theta_new = theta + omega_new * dt

        wheel_new = wheel_omega + wheel_dd * dt

        wheel_new = np.clip(
            wheel_new,
            -cfg.max_wheel_speed,
            cfg.max_wheel_speed,
        )

        return theta_new, omega_new, wheel_new

    def run_episode(
        self,
        network: Optional[FeedforwardNetwork] = None,
        genome: Optional[Genome] = None,
    ) -> SimulationResult:
        """
        Simulate one rollout episode.
        """

        if network is None and genome is not None:
            network = FeedforwardNetwork(genome)

        if network is None:
            raise ValueError(
                "run_episode requires network or genome"
            )

        cfg = self.config

        steps = int(cfg.duration / cfg.dt)

        t_arr = np.linspace(
            0,
            cfg.duration,
            steps,
            endpoint=False,
        )

        theta = cfg.initial_angle
        omega = cfg.initial_omega
        wheel_omega = 0.0

        angles = np.zeros(steps)
        omegas = np.zeros(steps)
        accels = np.zeros(steps)
        wheels = np.zeros(steps)

        nn_out = np.zeros(steps)
        pid_out = np.zeros(steps)

        torque_cmd_log = np.zeros(steps)
        torque_actual_log = np.zeros(steps)

        resonance_log = np.zeros(steps)

        seismic = np.zeros(steps)

        for i, t in enumerate(t_arr):

            base_a = self.disturbance.sample(t)

            seismic[i] = base_a

            self.sensor.update_raw(
                theta,
                omega,
                base_a,
                wheel_omega,
            )

            reading = self.sensor.read()

            u_nn = float(
                network.activate(
                    reading.as_array()
                )[0]
            )

            u_pid = self.pid.compute(reading)

            torque = self.controller.compute_total_torque(
                u_pid,
                u_nn,
            )

            torque = np.clip(
                torque,
                -cfg.max_wheel_torque,
                cfg.max_wheel_torque,
            )

            resonance = self._resonance_disturbance(
                t,
                torque,
            )

            theta, omega, wheel_omega = self._integrate_step(
                theta,
                omega,
                wheel_omega,
                torque,
                base_a,
                resonance,
                cfg.dt,
            )

            nn_out[i] = u_nn
            pid_out[i] = u_pid

            torque_cmd_log[i] = torque
            torque_actual_log[i] = self.actual_torque

            resonance_log[i] = resonance

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
            commanded_torque=torque_cmd_log,
            actual_torque=torque_actual_log,
            resonance_signal=resonance_log,
            seismic_input=seismic,
        )

    def make_rollout_fn(
        self,
        genome: Genome,
    ) -> Callable[[], SimulationResult]:

        net = FeedforwardNetwork(genome)

        def rollout() -> SimulationResult:
            return self.run_episode(network=net)

        return rollout