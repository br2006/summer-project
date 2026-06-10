"""NN-only balance environment with existing swing-up procedure."""

from __future__ import annotations

from typing import Optional

import numpy as np

from control.nn_controller import NNOnlyController
from neat.genome import Genome
from neat.network import FeedforwardNetwork
from simulation.pendulum_env import PendulumEnv, SimulationResult


class NNPendulumEnv(PendulumEnv):
    """
    Uses the same plant, sensing, disturbances, and swing-up as PendulumEnv,
    but BALANCE torque is produced solely by the NN.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.nn_controller = NNOnlyController(
            nn_torque_scale=self.config.nn_torque_scale,
            max_wheel_torque=self.config.max_wheel_torque,
        )

    def run_episode(
        self,
        network: Optional[FeedforwardNetwork] = None,
        genome: Optional[Genome] = None,
    ) -> SimulationResult:
        if network is None and genome is not None:
            network = FeedforwardNetwork(genome)

        if network is None:
            raise ValueError("run_episode requires network or genome")

        cfg = self.config
        self.reset()

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

        torque_cmd_log = np.zeros(steps)
        torque_actual_log = np.zeros(steps)
        resonance_log = np.zeros(steps)
        seismic = np.zeros(steps)
        mode_log = np.zeros(steps, dtype=np.int8)

        for i, t in enumerate(t_arr):
            base_a = self.disturbance.sample(t)
            seismic[i] = base_a

            self.sensor.update_raw(theta, omega, base_a, wheel_omega)
            reading = self.sensor.read()

            u_nn = float(network.activate(reading.as_array())[0])

            # Use measured state for hysteretic swing-up/balance mode switching.
            theta_meas = reading.angle * np.pi
            omega_meas = reading.angular_velocity * 10.0
            wheel_omega_meas = reading.wheel_velocity * self.sensor.wheel_scale
            self._update_control_mode(theta_meas, omega_meas, wheel_omega_meas)

            if self.control_mode == "SWINGUP":
                torque = self._swingup_torque(theta, omega, wheel_omega)
            else:
                torque = self.nn_controller.compute_torque(u_nn)

            torque = float(np.clip(torque, -cfg.max_wheel_torque, cfg.max_wheel_torque))
            self.actuator.apply_torque(torque)

            resonance = self._resonance_disturbance(t, torque)
            theta, omega, wheel_omega = self._integrate_step(
                theta,
                omega,
                wheel_omega,
                base_a,
                resonance,
                cfg.dt,
            )

            nn_out[i] = u_nn
            pid_out[i] = 0.0
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
