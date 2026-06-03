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
    wheel_rim_width: float = 0.010
    wheel_hub_radius: float = 0.020
    wheel_spoke_count: int = 2
    wheel_spoke_coverage: float = 0.50

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
    pid_kp: float = 1.8
    pid_ki: float = 0.0
    pid_kd: float = 0.45
    pid_integral_limit: float = 2.0
    pid_torque_scale: float = 0.4

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
    disturbance_model: str = "sinusoidal"
    broadband_noise_gain: float = 0.04

    vibration_freqs_hz: List[float] = field(
        default_factory=lambda: [3.0, 7.0]
    )

    vibration_amps: List[float] = field(
        default_factory=lambda: [0.03, 0.02]
    )

    impulse_probability: float = 0.001
    impulse_magnitude: float = 0.25

    footstep_rate_hz: float = 1.43
    footstep_jitter: float = 0.12
    footstep_accel_mps2: float = 0.25
    footstep_pulse_width_s: float = 0.06
    table_ring_freqs_hz: List[float] = field(default_factory=lambda: [8.0, 14.0])
    table_ring_amps_mps2: List[float] = field(default_factory=lambda: [0.08, 0.04])
    table_ring_decay_s: float = 0.30
    accelerometer_noise_std_mps2: float = 0.015


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
            wheel_rim_width=self.config.wheel_rim_width,
            wheel_hub_radius=self.config.wheel_hub_radius,
            wheel_spoke_count=self.config.wheel_spoke_count,
            wheel_spoke_coverage=self.config.wheel_spoke_coverage,
            gravity=self.config.gravity,
        )

        self.mass_properties = self.compound_properties.debug_summary()
        self.total_mass = self.mass_properties["total_mass"]
        self.r_cm = self.mass_properties["center_of_mass"]
        self.total_inertia = self.mass_properties["total_inertia_about_pivot"]
        self.wheel_inertia = self.mass_properties["wheel_inertia_cm"]
        # Body inertia excluding the rotor spin inertia term.
        # Used with a relative wheel-speed state for consistent coupled dynamics.
        self.body_inertia = max(1e-9, self.total_inertia - self.wheel_inertia)

        self._print_mass_property_debug()

        self.sensor = SimulatedSensor(
            angle_noise_std=self.config.angle_noise_std,
            gyro_noise_std=self.config.gyro_noise_std,
            gyro_bias_drift_rate=self.config.gyro_bias_drift_rate,
        )

        self.actuator = SimulatedActuator()

        self.pid = ReactionWheelPIDController(
            kp=self.config.pid_kp,
            ki=self.config.pid_ki,
            kd=self.config.pid_kd,
            dt=self.config.dt,
            integral_limit=self.config.pid_integral_limit,
        )

        self.controller = HybridController(
            pid=self.pid,
            alpha=self.config.alpha,
            nn_torque_scale=self.config.nn_torque_scale,
            pid_torque_scale=self.config.pid_torque_scale,
        )

        self.disturbance = DisturbanceGenerator(
            dt=self.config.dt,
            broadband_gain=self.config.broadband_noise_gain,
            vibration_freqs=self.config.vibration_freqs_hz,
            vibration_amps=self.config.vibration_amps,
            impulse_probability=self.config.impulse_probability,
            impulse_magnitude=self.config.impulse_magnitude,
            model=self.config.disturbance_model,
            footstep_rate_hz=self.config.footstep_rate_hz,
            footstep_jitter=self.config.footstep_jitter,
            footstep_accel_mps2=self.config.footstep_accel_mps2,
            footstep_pulse_width_s=self.config.footstep_pulse_width_s,
            table_ring_freqs_hz=self.config.table_ring_freqs_hz,
            table_ring_amps_mps2=self.config.table_ring_amps_mps2,
            table_ring_decay_s=self.config.table_ring_decay_s,
            accelerometer_noise_std_mps2=self.config.accelerometer_noise_std_mps2,
        )

        # Actual motor torque after lag dynamics.
        self.actual_torque = 0.0

    def reset(self) -> None:
        """Reset episode-level dynamic states in realism modules."""
        self.actual_torque = 0.0
        self.actuator.apply_torque(0.0)
        self.sensor.reset()
        self.disturbance.reset()
        self.pid.integral = 0.0

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
        base_acceleration: float,
        resonance: float,
        dt: float,
    ) -> tuple[float, float, float]:

        cfg = self.config

        # Motor lag dynamics
        torque_cmd = self.actuator.read_torque()
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

        # Base acceleration torque from a horizontally accelerating support.
        # The disturbance generator outputs m/s^2, matching a base-mounted
        # accelerometer. For an inverted pendulum, horizontal base acceleration
        # creates an inertial force at the centre of mass and therefore a torque
        # about the pivot. Sign convention: positive base acceleration tends to
        # rotate the pendulum in the negative theta direction near upright.
        tau_base = -self.total_mass * self.r_cm * base_acceleration * np.cos(theta)

        # External/body torques (excluding internal motor exchange with the wheel).
        tau_external = (
            tau_gravity
            + tau_damping
            + tau_base
            + resonance
        )

        # Coupled reaction-wheel dynamics using wheel_omega as RELATIVE wheel speed.
        # Motor torque is internal: -tau on body, +tau on wheel.
        # Wheel damping acts on relative speed and reacts on body with opposite sign.
        tau_internal_on_body = -self.actual_torque + cfg.wheel_damping * wheel_omega

        # Pendulum (body) angular acceleration.
        theta_dd = (tau_external + tau_internal_on_body) / self.body_inertia

        # Relative wheel-speed acceleration phi_dd from:
        # Iw * (theta_dd + phi_dd) = tau_motor - b_w * phi_dot
        wheel_dd = (
            (self.actual_torque - cfg.wheel_damping * wheel_omega)
            / max(1e-9, self.wheel_inertia)
            - theta_dd
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
        self.reset()

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

            self.actuator.apply_torque(float(torque))

            resonance = self._resonance_disturbance(
                t,
                torque,
            )

            theta, omega, wheel_omega = self._integrate_step(
                theta,
                omega,
                wheel_omega,
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