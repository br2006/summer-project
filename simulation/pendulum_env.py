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
    control_mode: np.ndarray


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
    max_wheel_torque: float = 0.30
    max_wheel_speed: float = 120.0

    # Motor dynamics
    motor_time_constant: float = 0.04
    # If False, commanded torque is applied immediately (no first-order lag).
    # Useful when matching low-latency motor drivers and for cleaner PID tuning.
    enable_motor_lag: bool = False

    # NN blending
    nn_torque_scale: float = 0.15
    alpha: float = 0.3
    pid_kp: float = 1.8
    pid_ki: float = 0.0
    pid_kd: float = 0.45
    pid_integral_limit: float = 2.0
    pid_torque_scale: float = 0.4

    # Swing-up mode (energy shaping) and mode switching.
    enable_swingup: bool = True
    swingup_gain: float = 0.8
    swingup_max_torque_fraction: float = 0.6
    swingup_soft_zone_deg: float = 25.0
    swingup_upright_zone_deg: float = 16.0
    swingup_velocity_damping: float = 0.04
    swingup_brake_gain: float = 0.10
    switch_threshold_deg: float = 14.0
    fallback_threshold_deg: float = 60.0
    max_switch_velocity: float = 2.0
    max_switch_wheel_speed: float = 45.0
    switch_dwell_time_s: float = 0.05
    swingup_escape_angle_deg: float = 150.0
    swingup_escape_velocity: float = 0.8
    swingup_escape_torque: float = 0.30
    swingup_escape_torque_fraction: float = 0.9
    swingup_escape_half_period_s: float = 0.18
    swingup_low_speed_threshold: float = 0.35

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

        # Enforce a torque budget when NN authority is enabled so PID does not
        # consume all motor headroom. Worst-case blended request magnitude is:
        #   |u_total| <= pid_torque_scale + |alpha| * nn_torque_scale
        # Reserve NN headroom inside max_wheel_torque by capping PID scale.
        nn_headroom = abs(self.config.alpha) * self.config.nn_torque_scale
        pid_scale_cap = max(0.0, self.config.max_wheel_torque - nn_headroom)
        effective_pid_torque_scale = min(self.config.pid_torque_scale, pid_scale_cap)
        if effective_pid_torque_scale < self.config.pid_torque_scale:
            print(
                "[HybridController] Clamping pid_torque_scale "
                f"from {self.config.pid_torque_scale:.4f} to {effective_pid_torque_scale:.4f} "
                f"to preserve NN headroom ({nn_headroom:.4f} Nm) under max torque "
                f"{self.config.max_wheel_torque:.4f} Nm."
            )

        self.controller = HybridController(
            pid=self.pid,
            alpha=self.config.alpha,
            nn_torque_scale=self.config.nn_torque_scale,
            pid_torque_scale=effective_pid_torque_scale,
            max_wheel_torque=self.config.max_wheel_torque,
        )

        # High-level controller mode for swing-up/balance switching.
        self.control_mode = self._initial_control_mode()
        self._swingup_escape_sign = 1.0
        self._swingup_escape_timer = 0.0
        self._switch_dwell_timer = 0.0

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
        self.control_mode = self._initial_control_mode()
        self._swingup_escape_sign = 1.0
        self._swingup_escape_timer = 0.0
        self._switch_dwell_timer = 0.0

    def _initial_control_mode(self) -> str:
        """Single source of truth for startup mode selection."""
        return "SWINGUP" if bool(self.config.enable_swingup) else "BALANCE"

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _swingup_torque(self, theta: float, omega: float, wheel_omega: float) -> float:
        """Energy-shaped swing-up with near-upright damping/braking."""
        cfg = self.config

        # Give swing-up its own torque ceiling so it does not always use full actuator range.
        frac = float(np.clip(cfg.swingup_max_torque_fraction, 0.0, 1.0))
        swingup_torque_limit = max(1e-9, frac * float(cfg.max_wheel_torque))

        theta_err = self._wrap_angle(theta)
        angle_from_upright = abs(theta_err)

        # Dead-zone escape near hanging-down equilibrium.
        if (
            angle_from_upright > np.radians(float(cfg.swingup_escape_angle_deg))
            and abs(omega) < float(cfg.swingup_escape_velocity)
        ):
            escape_mag = float(cfg.swingup_escape_torque)
            if escape_mag <= 0.0:
                escape_mag = float(cfg.swingup_escape_torque_fraction) * float(cfg.max_wheel_torque)
            escape_mag = float(np.clip(abs(escape_mag), 0.0, swingup_torque_limit))
            # Alternate kick direction while trapped near hanging-down to break symmetry.
            theta_sign = np.sign(theta_err)
            if theta_sign != 0.0:
                self._swingup_escape_sign = float(theta_sign)
            half_period = max(1e-3, float(cfg.swingup_escape_half_period_s))
            phase_index = int(self._swingup_escape_timer / half_period)
            escape_sign = self._swingup_escape_sign if (phase_index % 2 == 0) else -self._swingup_escape_sign
            self._swingup_escape_timer += float(cfg.dt)
            return float(escape_sign * escape_mag)
        else:
            self._swingup_escape_timer = 0.0

        # Core pumping action: normalized saturated energy shaping.
        # current_E = 0.5 I omega^2 + m g r_cm cos(theta)
        # target_E  = m g r_cm  (upright)
        # deficit   = target_E - current_E >= 0 when below target.
        current_e = (
            0.5 * float(self.total_inertia) * float(omega) ** 2
            + float(self.total_mass) * float(cfg.gravity) * float(self.r_cm) * np.cos(theta_err)
        )
        target_e = float(self.total_mass) * float(cfg.gravity) * float(self.r_cm)
        energy_deficit = target_e - current_e

        phase_sign = np.sign(omega * np.cos(theta_err))
        if phase_sign == 0.0:
            phase_sign = np.sign(omega)
        if phase_sign == 0.0:
            phase_sign = self._swingup_escape_sign

        # Treat swingup_gain as pump authority fraction in [0, 1].
        gain_frac = float(np.clip(abs(cfg.swingup_gain), 0.0, 1.0))
        deficit_norm = float(np.clip(energy_deficit / max(1e-9, target_e), 0.0, 1.0))
        torque = swingup_torque_limit * gain_frac * deficit_norm * float(phase_sign)

        # Soften approach near upright so residual kinetic energy is reduced.
        soft_zone = np.radians(max(1e-6, float(cfg.swingup_soft_zone_deg)))
        if angle_from_upright < soft_zone:
            torque *= angle_from_upright / soft_zone

        # Progressive damping: stronger near upright to bleed residual kinetic energy.
        switch_threshold = np.radians(float(cfg.switch_threshold_deg))
        brake_zone = max(soft_zone, 2.0 * switch_threshold)
        upright_zone = np.radians(max(1e-6, float(cfg.swingup_upright_zone_deg)))

        proximity = 0.0
        if angle_from_upright < brake_zone:
            proximity = 1.0 - (angle_from_upright / brake_zone)
            proximity = float(np.clip(proximity, 0.0, 1.0))

        damping_gain = float(cfg.swingup_velocity_damping) * (1.0 + 2.5 * proximity)
        wheel_damping_gain = 0.25 * float(cfg.swingup_brake_gain) * (1.0 + 2.0 * proximity)
        torque -= damping_gain * omega
        torque -= wheel_damping_gain * wheel_omega

        # Near-upright braking if speed exceeds handoff target velocity.
        if angle_from_upright < brake_zone:
            speed_excess = abs(omega) - float(cfg.max_switch_velocity)
            if speed_excess > 0.0:
                torque -= (
                    float(cfg.swingup_brake_gain)
                    * (1.0 + 2.0 * proximity)
                    * speed_excess
                    * np.sign(omega)
                )

            # Wheel-speed-aware braking to avoid arriving at handoff with high rotor speed.
            wheel_excess = abs(wheel_omega) - float(cfg.max_switch_wheel_speed)
            if wheel_excess > 0.0:
                torque -= (
                    0.05
                    * float(cfg.swingup_brake_gain)
                    * (1.0 + 2.0 * proximity)
                    * wheel_excess
                    * np.sign(wheel_omega)
                )

        # Extra braking very near upright.
        if angle_from_upright < upright_zone:
            torque -= 0.5 * float(cfg.swingup_brake_gain) * omega

        return float(np.clip(torque, -swingup_torque_limit, swingup_torque_limit))

    def _update_control_mode(self, theta_meas: float, omega_meas: float, wheel_omega_meas: float) -> None:
        """Hysteretic mode switching using measured (noisy) states."""
        if not self.config.enable_swingup:
            self.control_mode = "BALANCE"
            self._switch_dwell_timer = 0.0
            return

        cfg = self.config
        angle_from_upright = abs(self._wrap_angle(theta_meas))
        speed = abs(omega_meas)
        switch_threshold = np.radians(cfg.switch_threshold_deg)
        fallback_threshold = np.radians(cfg.fallback_threshold_deg)

        if self.control_mode == "SWINGUP":
            handoff_ready = (
                angle_from_upright < switch_threshold
                and speed < cfg.max_switch_velocity
                and abs(wheel_omega_meas) < cfg.max_switch_wheel_speed
            )
            if handoff_ready:
                self._switch_dwell_timer += float(cfg.dt)
            else:
                self._switch_dwell_timer = 0.0

            if handoff_ready and self._switch_dwell_timer >= max(0.0, float(cfg.switch_dwell_time_s)):
                self.control_mode = "BALANCE"
                self.pid.integral = 0.0
                self._switch_dwell_timer = 0.0
        else:
            self._switch_dwell_timer = 0.0
            # Keep fallback simple like CodePendulum: angle-only fallback.
            if angle_from_upright > fallback_threshold:
                self.control_mode = "SWINGUP"
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

        # Motor torque path
        torque_cmd = self.actuator.read_torque()
        if cfg.enable_motor_lag and cfg.motor_time_constant > 0.0:
            self.actual_torque += (
                dt / cfg.motor_time_constant
            ) * (
                torque_cmd - self.actual_torque
            )
        else:
            self.actual_torque = torque_cmd

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

        # Numerically robust wheel-speed integration:
        # 1) prevent further acceleration into a saturated wheel-speed rail,
        # 2) limit per-step wheel acceleration to a physically plausible bound.
        at_pos_rail = wheel_omega >= cfg.max_wheel_speed
        at_neg_rail = wheel_omega <= -cfg.max_wheel_speed
        if (at_pos_rail and wheel_dd > 0.0) or (at_neg_rail and wheel_dd < 0.0):
            wheel_dd = 0.0

        max_wheel_accel = cfg.max_wheel_torque / max(1e-9, self.wheel_inertia)
        wheel_dd = float(np.clip(wheel_dd, -max_wheel_accel, max_wheel_accel))

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
        mode_log = np.zeros(steps, dtype=np.int8)

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

            # Use measured state for mode switching, matching CodePendulum update_noisy().
            theta_meas = reading.angle * np.pi
            omega_meas = reading.angular_velocity * 10.0
            wheel_omega_meas = reading.wheel_velocity * self.sensor.wheel_scale
            self._update_control_mode(theta_meas, omega_meas, wheel_omega_meas)

            if self.control_mode == "SWINGUP":
                torque = self._swingup_torque(theta, omega, wheel_omega)
            else:
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
            mode_log[i] = 1 if self.control_mode == "BALANCE" else 0

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
            control_mode=mode_log,
        )

    def make_rollout_fn(
        self,
        genome: Genome,
    ) -> Callable[[], SimulationResult]:

        net = FeedforwardNetwork(genome)

        def rollout() -> SimulationResult:
            return self.run_episode(network=net)

        return rollout