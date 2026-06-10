"""
Stage-based diagnostics for reaction-wheel pendulum hold-near-upright behaviour.

Stages:
  local       -> no swing-up, no disturbances/noise, near-upright initial angle
  swingup     -> swing-up enabled, no disturbances/noise
  noisy       -> swing-up enabled, sensor noise enabled, no disturbances
  disturbance -> full tabletop disturbance + sensor noise
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from simulation.pendulum_env import PendulumEnv, PendulumEnvConfig


class ZeroNetwork:
    def activate(self, _obs) -> list[float]:
        return [0.0]


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class DiagnosticMetrics:
    rms_angle_deg: float
    max_abs_angle_deg: float
    rms_omega: float
    time_in_band_s: float
    longest_hold_s: float
    band_exit_count: int
    falls_count: int
    torque_saturation_fraction: float
    wheel_saturation_fraction: float
    balance_mode_fraction: float
    handoff_time_s: float | None
    handoff_angle_deg: float | None
    handoff_omega: float | None
    handoff_wheel_omega: float | None
    handoff_torque: float | None


def compute_diagnostics(result, cfg: PendulumEnvConfig, upright_band_deg: float) -> DiagnosticMetrics:
    dt = cfg.dt
    angle = _wrap(result.angle)
    angle_abs = np.abs(angle)
    omega = result.angular_velocity
    wheel = result.wheel_velocity
    mode = result.control_mode.astype(np.int8)
    torque = result.commanded_torque

    upright_band = np.radians(upright_band_deg)
    in_band = angle_abs < upright_band

    # Longest contiguous hold duration in band.
    longest_hold_steps = 0
    current_hold_steps = 0
    for is_hold in in_band:
        if is_hold:
            current_hold_steps += 1
            longest_hold_steps = max(longest_hold_steps, current_hold_steps)
        else:
            current_hold_steps = 0

    band_exit_count = int(np.sum(in_band[:-1] & (~in_band[1:])))

    fallen_thresh = np.radians(90.0)
    fallen_mask = angle_abs > fallen_thresh
    falls_count = int(np.sum((~fallen_mask[:-1]) & fallen_mask[1:]))

    torque_sat_eps = 0.98 * cfg.max_wheel_torque
    wheel_sat_eps = 0.98 * cfg.max_wheel_speed
    torque_sat_fraction = float(np.mean(np.abs(torque) >= torque_sat_eps))
    wheel_sat_fraction = float(np.mean(np.abs(wheel) >= wheel_sat_eps))

    # Detect first SWINGUP -> BALANCE handoff (0 -> 1 transition).
    mode_diff = np.diff(mode)
    handoff_idxs = np.where(mode_diff > 0)[0]
    handoff_i = int(handoff_idxs[0] + 1) if handoff_idxs.size else None

    handoff_time = float(result.time[handoff_i]) if handoff_i is not None else None
    handoff_angle_deg = float(np.degrees(angle[handoff_i])) if handoff_i is not None else None
    handoff_omega = float(omega[handoff_i]) if handoff_i is not None else None
    handoff_wheel_omega = float(wheel[handoff_i]) if handoff_i is not None else None
    handoff_torque = float(torque[handoff_i]) if handoff_i is not None else None

    return DiagnosticMetrics(
        rms_angle_deg=float(np.degrees(np.sqrt(np.mean(angle**2)))),
        max_abs_angle_deg=float(np.degrees(np.max(angle_abs))),
        rms_omega=float(np.sqrt(np.mean(omega**2))),
        time_in_band_s=float(np.sum(in_band) * dt),
        longest_hold_s=float(longest_hold_steps * dt),
        band_exit_count=band_exit_count,
        falls_count=falls_count,
        torque_saturation_fraction=torque_sat_fraction,
        wheel_saturation_fraction=wheel_sat_fraction,
        balance_mode_fraction=float(np.mean(mode.astype(bool))),
        handoff_time_s=handoff_time,
        handoff_angle_deg=handoff_angle_deg,
        handoff_omega=handoff_omega,
        handoff_wheel_omega=handoff_wheel_omega,
        handoff_torque=handoff_torque,
    )


def build_stage_config(base: PendulumEnvConfig, stage: str, duration: float) -> PendulumEnvConfig:
    cfg = deepcopy(base)
    cfg.duration = duration
    cfg.alpha = 0.0
    cfg.nn_torque_scale = 0.0

    if stage == "local":
        cfg.enable_swingup = False
        cfg.initial_angle = 0.05
        cfg.initial_omega = 0.0
        cfg.pid_ki = 0.0

        cfg.disturbance_model = "sinusoidal"
        cfg.broadband_noise_gain = 0.0
        cfg.vibration_amps = [0.0 for _ in cfg.vibration_amps]
        cfg.impulse_probability = 0.0
        cfg.impulse_magnitude = 0.0
        cfg.footstep_accel_mps2 = 0.0
        cfg.table_ring_amps_mps2 = [0.0 for _ in cfg.table_ring_amps_mps2]
        cfg.accelerometer_noise_std_mps2 = 0.0

        cfg.angle_noise_std = 0.0
        cfg.gyro_noise_std = 0.0
        cfg.gyro_bias_drift_rate = 0.0

    elif stage == "swingup":
        cfg.enable_swingup = True
        cfg.initial_angle = np.pi
        cfg.initial_omega = 0.0

        cfg.disturbance_model = "sinusoidal"
        cfg.broadband_noise_gain = 0.0
        cfg.vibration_amps = [0.0 for _ in cfg.vibration_amps]
        cfg.impulse_probability = 0.0
        cfg.impulse_magnitude = 0.0
        cfg.footstep_accel_mps2 = 0.0
        cfg.table_ring_amps_mps2 = [0.0 for _ in cfg.table_ring_amps_mps2]
        cfg.accelerometer_noise_std_mps2 = 0.0

        cfg.angle_noise_std = 0.0
        cfg.gyro_noise_std = 0.0
        cfg.gyro_bias_drift_rate = 0.0

    elif stage == "noisy":
        cfg.enable_swingup = True
        cfg.initial_angle = np.pi
        cfg.initial_omega = 0.0

        cfg.disturbance_model = "sinusoidal"
        cfg.broadband_noise_gain = 0.0
        cfg.vibration_amps = [0.0 for _ in cfg.vibration_amps]
        cfg.impulse_probability = 0.0
        cfg.impulse_magnitude = 0.0
        cfg.footstep_accel_mps2 = 0.0
        cfg.table_ring_amps_mps2 = [0.0 for _ in cfg.table_ring_amps_mps2]
        cfg.accelerometer_noise_std_mps2 = 0.0
        # Keep configured sensor noise for this stage.

    elif stage == "disturbance":
        cfg.enable_swingup = True
        cfg.initial_angle = np.pi
        cfg.initial_omega = 0.0
        cfg.disturbance_model = "footstep_tabletop"
        # Keep configured disturbances + sensor noise.

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return cfg


def run_stage(cfg: PendulumEnvConfig, upright_band_deg: float, seed: int) -> DiagnosticMetrics:
    np.random.seed(seed)
    env = PendulumEnv(cfg)
    result = env.run_episode(network=ZeroNetwork())
    return compute_diagnostics(result, cfg, upright_band_deg)


def print_metrics(stage: str, m: DiagnosticMetrics) -> None:
    print(f"\n=== Stage: {stage} ===")
    print(f"rms_angle_deg:             {m.rms_angle_deg:.3f}")
    print(f"max_abs_angle_deg:         {m.max_abs_angle_deg:.3f}")
    print(f"rms_omega_rad_s:           {m.rms_omega:.3f}")
    print(f"time_in_band_s:            {m.time_in_band_s:.3f}")
    print(f"longest_hold_s:            {m.longest_hold_s:.3f}")
    print(f"band_exit_count:           {m.band_exit_count}")
    print(f"falls_count:               {m.falls_count}")
    print(f"torque_saturation_fraction:{m.torque_saturation_fraction:.3f}")
    print(f"wheel_saturation_fraction: {m.wheel_saturation_fraction:.3f}")
    print(f"balance_mode_fraction:     {m.balance_mode_fraction:.3f}")

    if m.handoff_time_s is None:
        print("handoff:                   none detected")
    else:
        print(
            "handoff:                   "
            f"t={m.handoff_time_s:.3f}s "
            f"angle={m.handoff_angle_deg:+.2f}deg "
            f"omega={m.handoff_omega:+.3f}rad/s "
            f"wheel_omega={m.handoff_wheel_omega:+.3f}rad/s "
            f"torque={m.handoff_torque:+.3f}Nm"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged pendulum balance diagnostics")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project_config.yaml")
    parser.add_argument(
        "--stage",
        choices=["local", "swingup", "noisy", "disturbance", "all"],
        default="all",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--upright-band-deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    project = load_project_config(args.config)
    base_cfg = project.simulation

    stages = [args.stage] if args.stage != "all" else ["local", "swingup", "noisy", "disturbance"]
    for i, stage in enumerate(stages):
        cfg = build_stage_config(base_cfg, stage, args.duration)
        metrics = run_stage(cfg, args.upright_band_deg, seed=args.seed + i)
        print_metrics(stage, metrics)


if __name__ == "__main__":
    main()
