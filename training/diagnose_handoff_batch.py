"""Batch swing-up->handoff diagnostic with explicit pre-training pass/fail checks."""

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
class HandoffEvent:
    time_s: float
    angle_deg: float
    omega: float
    wheel_omega: float


def _first_handoff(result) -> HandoffEvent | None:
    mode = result.control_mode.astype(np.int8)
    transitions = np.where(np.diff(mode) > 0)[0]
    if transitions.size == 0:
        return None

    i = int(transitions[0] + 1)
    angle = _wrap(result.angle)
    return HandoffEvent(
        time_s=float(result.time[i]),
        angle_deg=float(np.degrees(angle[i])),
        omega=float(result.angular_velocity[i]),
        wheel_omega=float(result.wheel_velocity[i]),
    )


def build_eval_cfg(base: PendulumEnvConfig, args: argparse.Namespace) -> PendulumEnvConfig:
    cfg = deepcopy(base)
    cfg.duration = float(args.duration)
    cfg.enable_swingup = bool(args.enable_swingup)
    cfg.initial_angle = np.pi
    cfg.initial_omega = 0.0

    cfg.switch_threshold_deg = float(args.switch_threshold_deg)
    cfg.max_switch_velocity = float(args.max_switch_velocity)
    cfg.max_switch_wheel_speed = float(args.max_switch_wheel_speed)
    cfg.switch_dwell_time_s = float(args.switch_dwell_time_s)

    cfg.swingup_gain = float(args.swingup_gain)
    cfg.swingup_max_torque_fraction = float(args.swingup_max_torque_fraction)
    cfg.swingup_soft_zone_deg = float(args.swingup_soft_zone_deg)
    cfg.swingup_velocity_damping = float(args.swingup_velocity_damping)
    cfg.swingup_brake_gain = float(args.swingup_brake_gain)

    if args.no_disturbance:
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

    return cfg


def run_batch(cfg: PendulumEnvConfig, rollouts: int, seed: int) -> tuple[list[HandoffEvent], int]:
    events: list[HandoffEvent] = []
    handoff_count = 0
    for k in range(rollouts):
        np.random.seed(seed + k)
        env = PendulumEnv(cfg)
        result = env.run_episode(network=ZeroNetwork())
        event = _first_handoff(result)
        if event is not None:
            handoff_count += 1
            events.append(event)
    return events, handoff_count


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch handoff diagnostic for pre-training gating")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "project_config_nn_only.yaml")
    p.add_argument("--rollouts", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--no-disturbance", action="store_true")
    p.add_argument(
        "--enable-swingup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SWINGUP->BALANCE transition logic during this diagnostic",
    )

    p.add_argument("--switch-threshold-deg", type=float, default=5.0)
    p.add_argument("--max-switch-velocity", type=float, default=0.4)
    p.add_argument("--max-switch-wheel-speed", type=float, default=35.0)
    p.add_argument("--switch-dwell-time-s", type=float, default=0.05)

    p.add_argument("--swingup-gain", type=float, default=0.8)
    p.add_argument("--swingup-max-torque-fraction", type=float, default=0.55)
    p.add_argument("--swingup-soft-zone-deg", type=float, default=25.0)
    p.add_argument("--swingup-velocity-damping", type=float, default=0.04)
    p.add_argument("--swingup-brake-gain", type=float, default=0.10)

    p.add_argument("--min-handoff-rate", type=float, default=0.60)
    p.add_argument("--target-handoff-rate", type=float, default=0.80)
    p.add_argument("--require-median-time-fraction", type=float, default=0.70)
    return p


def main() -> None:
    args = make_parser().parse_args()
    project = load_project_config(args.config)
    archived_enable_swingup = bool(project.simulation.enable_swingup)
    cfg = build_eval_cfg(project.simulation, args)

    events, handoff_count = run_batch(cfg, args.rollouts, args.seed)
    handoff_rate = handoff_count / max(1, args.rollouts)

    vel_violations = sum(abs(e.omega) > cfg.max_switch_velocity for e in events)
    wheel_violations = sum(abs(e.wheel_omega) > cfg.max_switch_wheel_speed for e in events)

    handoff_times = [e.time_s for e in events]
    handoff_angles = [e.angle_deg for e in events]
    handoff_omegas = [e.omega for e in events]
    handoff_wheels = [e.wheel_omega for e in events]

    median_handoff_time = _median_or_nan(handoff_times)
    median_handoff_time_fraction = median_handoff_time / max(1e-9, cfg.duration)

    print("=== Batch handoff diagnostic ===")
    print(
        "swingup_mode:                     "
        f"archived={'ON' if archived_enable_swingup else 'OFF'} -> "
        f"effective={'ON' if cfg.enable_swingup else 'OFF'}"
    )
    print(f"rollouts:                          {args.rollouts}")
    print(f"handoff_count:                     {handoff_count}")
    print(f"handoff_rate:                      {handoff_rate:.3f}")
    print(f"velocity_violations_at_handoff:    {vel_violations}")
    print(f"wheel_violations_at_handoff:       {wheel_violations}")
    print(f"median_handoff_time_s:             {median_handoff_time:.3f}")
    print(f"median_handoff_time_fraction:      {median_handoff_time_fraction:.3f}")
    print(f"median_handoff_angle_deg:          {_median_or_nan(handoff_angles):+.3f}")
    print(f"median_handoff_omega_rad_s:        {_median_or_nan(handoff_omegas):+.3f}")
    print(f"median_handoff_wheel_omega_rad_s:  {_median_or_nan(handoff_wheels):+.3f}")

    min_rate_ok = handoff_rate >= args.min_handoff_rate
    rate_target_ok = handoff_rate >= args.target_handoff_rate
    vel_ok = vel_violations == 0
    wheel_ok = wheel_violations == 0
    median_ok = (
        not np.isnan(median_handoff_time_fraction)
        and median_handoff_time_fraction < args.require_median_time_fraction
    )

    print("\n=== Pre-training gates ===")
    print(f"handoff_rate >= {args.min_handoff_rate:.2f}:          {'PASS' if min_rate_ok else 'FAIL'}")
    print(f"handoff_rate >= {args.target_handoff_rate:.2f}:       {'PASS' if rate_target_ok else 'FAIL'}")
    print(f"velocity_violations == 0:             {'PASS' if vel_ok else 'FAIL'}")
    print(f"wheel_violations == 0:                {'PASS' if wheel_ok else 'FAIL'}")
    print(
        f"median_handoff_time < {args.require_median_time_fraction:.2f} * episode: "
        f"{'PASS' if median_ok else 'FAIL'}"
    )

    hard_pass = min_rate_ok and vel_ok and wheel_ok
    print(f"\nOVERALL_HARD_GATE: {'PASS' if hard_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
