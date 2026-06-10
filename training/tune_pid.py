"""
PID tuning and swing-up sweep utilities for the reaction-wheel pendulum.

This script is designed for the low-torque hardware regime (default 0.30 Nm),
and evaluates the full swing-up -> handoff -> balance behaviour instead of only
local near-upright regulation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
from visualisation_code.output import get_output_dir


class ZeroNetwork:
    """Placeholder NN used so PID+swing-up behaviour is evaluated in isolation."""

    def activate(self, _obs) -> list[float]:
        return [0.0]


@dataclass
class RolloutMetrics:
    score: float
    time_to_upright: float
    time_in_band: float
    first_reach_time: float
    band_exit_count: int
    falls_count: int
    settle_time: float
    longest_hold_time: float
    upright_fraction: float
    balance_mode_fraction: float
    fall_rate: float
    saturation_fraction: float
    handoff_count: int


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def evaluate_rollout_metrics(
    result,
    cfg: PendulumEnvConfig,
    upright_band_deg: float,
) -> RolloutMetrics:
    """Compute swing-up + handoff + balance metrics from one rollout."""
    t = result.time
    dt = cfg.dt
    duration = float(cfg.duration)

    angle = np.abs(_wrap(result.angle))
    omega = np.abs(result.angular_velocity)
    mode = result.control_mode
    torque = np.abs(result.commanded_torque)

    upright_band = np.radians(upright_band_deg)
    settle_band = np.radians(min(8.0, max(4.0, cfg.switch_threshold_deg * 0.7)))
    settle_omega = min(1.2, max(0.4, cfg.max_switch_velocity * 0.8))

    upright_mask = angle < upright_band
    settle_mask = (angle < settle_band) & (omega < settle_omega)
    hold_mask = settle_mask

    upright_idx = np.where(upright_mask)[0]
    time_to_upright = float(t[upright_idx[0]]) if upright_idx.size else duration
    first_reach_time = time_to_upright
    time_in_band = float(np.sum(upright_mask) * dt)

    band_exit_count = int(np.sum(upright_mask[:-1] & (~upright_mask[1:])))

    fallen_thresh = np.radians(90.0)
    fallen_mask = angle > fallen_thresh
    falls_count = int(np.sum((~fallen_mask[:-1]) & fallen_mask[1:]))

    settle_time = duration
    if settle_mask.any():
        for i in np.where(settle_mask)[0]:
            if np.all(settle_mask[i:]):
                settle_time = float(t[i])
                break

    # Longest contiguous hold duration within the settle band.
    longest_hold_steps = 0
    current_hold_steps = 0
    for is_hold in hold_mask:
        if is_hold:
            current_hold_steps += 1
            longest_hold_steps = max(longest_hold_steps, current_hold_steps)
        else:
            current_hold_steps = 0
    longest_hold_time = float(longest_hold_steps * dt)

    upright_fraction = float(np.mean(upright_mask))
    balance_mode_fraction = float(np.mean(mode.astype(bool)))

    fall_rate = float(np.mean(angle > fallen_thresh))

    sat_eps = 0.98 * cfg.max_wheel_torque
    saturation_fraction = float(np.mean(torque >= sat_eps))

    handoff_count = int(np.sum(np.abs(np.diff(mode.astype(np.int8))) > 0))

    # Lower is better. Weights prioritize rapid swing-up and robust settling,
    # while penalizing falling, chatter, and excessive saturation.
    score = (
        1.0 * time_to_upright
        + 1.25 * settle_time
        + 3.0 * fall_rate * duration
        + 0.75 * saturation_fraction * duration
        + 0.4 * handoff_count * dt
    )

    return RolloutMetrics(
        score=float(score),
        time_to_upright=time_to_upright,
        time_in_band=time_in_band,
        first_reach_time=first_reach_time,
        band_exit_count=band_exit_count,
        falls_count=falls_count,
        settle_time=settle_time,
        longest_hold_time=longest_hold_time,
        upright_fraction=upright_fraction,
        balance_mode_fraction=balance_mode_fraction,
        fall_rate=fall_rate,
        saturation_fraction=saturation_fraction,
        handoff_count=handoff_count,
    )


def evaluate_candidate(
    cfg: PendulumEnvConfig,
    seeds: list[int],
    upright_band_deg: float,
) -> RolloutMetrics:
    all_metrics: list[RolloutMetrics] = []
    for seed in seeds:
        np.random.seed(seed)
        env = PendulumEnv(cfg)
        result = env.run_episode(network=ZeroNetwork())
        all_metrics.append(evaluate_rollout_metrics(result, cfg, upright_band_deg))

    def _avg(field: str) -> float:
        return float(np.mean([getattr(m, field) for m in all_metrics]))

    return RolloutMetrics(
        score=_avg("score"),
        time_to_upright=_avg("time_to_upright"),
        time_in_band=_avg("time_in_band"),
        first_reach_time=_avg("first_reach_time"),
        band_exit_count=int(round(_avg("band_exit_count"))),
        falls_count=int(round(_avg("falls_count"))),
        settle_time=_avg("settle_time"),
        longest_hold_time=_avg("longest_hold_time"),
        upright_fraction=_avg("upright_fraction"),
        balance_mode_fraction=_avg("balance_mode_fraction"),
        fall_rate=_avg("fall_rate"),
        saturation_fraction=_avg("saturation_fraction"),
        handoff_count=int(round(_avg("handoff_count"))),
    )


def evaluate_band_objective(
    metrics: RolloutMetrics,
    duration: float,
) -> float:
    """Lower-is-better objective using explicit upright-band priorities.

    Priorities:
      1) maximize time in band
      2) maximize longest continuous hold in band
      3) penalize exits and falls
      4) secondary: faster first reach
    """
    time_out_of_band = max(0.0, duration - metrics.time_in_band)
    hold_shortfall = max(0.0, duration - metrics.longest_hold_time)
    return float(
        4.0 * time_out_of_band
        + 3.5 * hold_shortfall
        + 8.0 * metrics.band_exit_count
        + 12.0 * metrics.falls_count
        + 0.8 * metrics.first_reach_time
    )


def select_objective(
    metrics: RolloutMetrics,
    duration: float,
    objective: str,
) -> float:
    if objective == "swingup":
        return float(metrics.score)
    if objective == "band":
        return evaluate_band_objective(metrics, duration)
    raise ValueError(f"Unknown objective: {objective}")


def build_eval_config(base: PendulumEnvConfig, args: argparse.Namespace) -> PendulumEnvConfig:
    cfg = deepcopy(base)
    cfg.max_wheel_torque = args.max_torque
    cfg.pid_torque_scale = min(cfg.pid_torque_scale, args.max_torque)
    cfg.nn_torque_scale = 0.0
    cfg.alpha = 0.0
    cfg.enable_swingup = True
    cfg.initial_angle = np.pi
    cfg.initial_omega = 0.0
    cfg.duration = args.duration

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

    # Conservative handoff defaults for low-torque operation.
    if hasattr(args, "switch_threshold_deg"):
        cfg.switch_threshold_deg = min(cfg.switch_threshold_deg, args.switch_threshold_deg)
    if hasattr(args, "max_switch_velocity"):
        cfg.max_switch_velocity = min(cfg.max_switch_velocity, args.max_switch_velocity)
    if hasattr(args, "max_switch_wheel_speed") and args.max_switch_wheel_speed > 0.0:
        cfg.max_switch_wheel_speed = min(cfg.max_switch_wheel_speed, args.max_switch_wheel_speed)

    return cfg


def run_pid_random_search(args: argparse.Namespace) -> None:
    project = load_project_config(args.config)
    cfg = build_eval_config(project.simulation, args)

    seed_list = [args.seed + i for i in range(args.eval_rollouts)]
    rng = np.random.default_rng(args.seed)

    best = None
    best_params = None
    best_objective_score = None

    for i in range(args.trials):
        trial_cfg = deepcopy(cfg)
        trial_cfg.pid_kp = float(rng.uniform(args.kp_min, args.kp_max))
        trial_cfg.pid_ki = float(rng.uniform(args.ki_min, args.ki_max))
        trial_cfg.pid_kd = float(rng.uniform(args.kd_min, args.kd_max))
        trial_cfg.pid_torque_scale = float(args.max_torque)

        metrics = evaluate_candidate(trial_cfg, seed_list, args.upright_band_deg)
        objective_score = select_objective(metrics, args.duration, args.objective)
        if best is None or best_objective_score is None or objective_score < best_objective_score:
            best = metrics
            best_objective_score = objective_score
            best_params = (
                trial_cfg.pid_kp,
                trial_cfg.pid_ki,
                trial_cfg.pid_kd,
                trial_cfg.pid_torque_scale,
                objective_score,
            )

        if (i + 1) % max(1, args.trials // 10) == 0:
            print(
                f"trial {i+1}/{args.trials} "
                f"best_{args.objective}_score={best_params[4]:.3f}"
            )

    assert best is not None and best_params is not None

    print(f"\n=== Best PID ({args.objective} objective) ===")
    print(f"kp={best_params[0]:.6f} ki={best_params[1]:.6f} kd={best_params[2]:.6f}")
    print(f"pid_torque_scale={best_params[3]:.6f}")
    print(f"objective_score={best_params[4]:.4f}")
    print(f"score={best.score:.4f}")
    print(f"time_to_upright={best.time_to_upright:.3f}s")
    print(f"settle_time={best.settle_time:.3f}s")
    print(f"fall_rate={best.fall_rate:.4f}")
    print(f"saturation_fraction={best.saturation_fraction:.4f}")
    print(f"handoff_count={best.handoff_count}")


def run_pid_successive_halving(args: argparse.Namespace) -> None:
    """Joint successive-halving over PID + handoff/swing-up parameters."""
    project = load_project_config(args.config)
    base_cfg = build_eval_config(project.simulation, args)
    rng = np.random.default_rng(args.seed)

    rollout_schedule = [int(v.strip()) for v in args.sh_rollout_schedule.split(",") if v.strip()]
    duration_schedule = [float(v.strip()) for v in args.sh_duration_schedule.split(",") if v.strip()]
    if len(rollout_schedule) != len(duration_schedule):
        raise ValueError("--sh-rollout-schedule and --sh-duration-schedule must have same length")

    candidates: list[dict[str, float]] = []
    for _ in range(args.sh_initial_candidates):
        candidates.append(
            {
                "kp": float(rng.uniform(args.kp_min, args.kp_max)),
                "ki": float(rng.uniform(args.ki_min, args.ki_max)),
                "kd": float(rng.uniform(args.kd_min, args.kd_max)),
                "switch_threshold_deg": float(
                    rng.uniform(args.switch_threshold_min_deg, args.switch_threshold_max_deg)
                ),
                "max_switch_velocity": float(
                    rng.uniform(args.max_switch_velocity_min, args.max_switch_velocity_max)
                ),
                "fallback_threshold_deg": float(
                    rng.uniform(args.fallback_threshold_min_deg, args.fallback_threshold_max_deg)
                ),
                "max_switch_wheel_speed": float(
                    rng.uniform(args.max_switch_wheel_speed_min, args.max_switch_wheel_speed_max)
                ),
                "swingup_gain": float(
                    rng.uniform(args.swingup_gain_min, args.swingup_gain_max)
                ),
            }
        )

    print(f"Starting successive halving with {len(candidates)} candidates")
    for round_idx, (n_rollouts, eval_duration) in enumerate(
        zip(rollout_schedule, duration_schedule),
        start=1,
    ):
        round_rows: list[tuple[float, RolloutMetrics, dict[str, float]]] = []
        seeds = [args.seed + i for i in range(n_rollouts)]

        for params in candidates:
            cfg = deepcopy(base_cfg)
            cfg.duration = eval_duration
            cfg.pid_kp = params["kp"]
            cfg.pid_ki = params["ki"]
            cfg.pid_kd = params["kd"]
            cfg.pid_torque_scale = float(args.max_torque)
            cfg.switch_threshold_deg = params["switch_threshold_deg"]
            cfg.max_switch_velocity = params["max_switch_velocity"]
            cfg.fallback_threshold_deg = params["fallback_threshold_deg"]
            cfg.max_switch_wheel_speed = params["max_switch_wheel_speed"]
            cfg.swingup_gain = params["swingup_gain"]

            metrics = evaluate_candidate(cfg, seeds, args.upright_band_deg)
            score = evaluate_band_objective(metrics, eval_duration)
            round_rows.append((score, metrics, params))

        round_rows.sort(key=lambda r: r[0])
        keep_n = max(1, int(np.ceil(len(round_rows) * args.sh_keep_ratio)))
        candidates = [r[2] for r in round_rows[:keep_n]]

        best_score, best_metrics, best_params = round_rows[0]
        print(
            f"round {round_idx}/{len(rollout_schedule)} | "
            f"dur={eval_duration:.1f}s rollouts={n_rollouts} | "
            f"candidates={len(round_rows)} -> keep={keep_n} | "
            f"best_band_score={best_score:.3f} "
            f"(in_band={best_metrics.time_in_band:.2f}s hold={best_metrics.longest_hold_time:.2f}s "
            f"exits={best_metrics.band_exit_count} falls={best_metrics.falls_count} "
            f"first={best_metrics.first_reach_time:.2f}s)"
        )

    final_seeds = [args.seed + i for i in range(args.eval_rollouts)]
    best_final = None
    best_final_score = None
    best_final_params = None
    for params in candidates:
        cfg = deepcopy(base_cfg)
        cfg.duration = args.duration
        cfg.pid_kp = params["kp"]
        cfg.pid_ki = params["ki"]
        cfg.pid_kd = params["kd"]
        cfg.pid_torque_scale = float(args.max_torque)
        cfg.switch_threshold_deg = params["switch_threshold_deg"]
        cfg.max_switch_velocity = params["max_switch_velocity"]
        cfg.fallback_threshold_deg = params["fallback_threshold_deg"]
        cfg.max_switch_wheel_speed = params["max_switch_wheel_speed"]
        cfg.swingup_gain = params["swingup_gain"]
        metrics = evaluate_candidate(cfg, final_seeds, args.upright_band_deg)
        score = evaluate_band_objective(metrics, args.duration)
        if best_final is None or score < best_final_score:
            best_final = metrics
            best_final_score = score
            best_final_params = params

    assert best_final is not None and best_final_params is not None and best_final_score is not None

    print("\n=== Best joint params (successive halving, band objective) ===")
    print(
        f"kp={best_final_params['kp']:.6f} "
        f"ki={best_final_params['ki']:.6f} "
        f"kd={best_final_params['kd']:.6f}"
    )
    print(f"switch_threshold_deg={best_final_params['switch_threshold_deg']:.3f}")
    print(f"max_switch_velocity={best_final_params['max_switch_velocity']:.3f}")
    print(f"fallback_threshold_deg={best_final_params['fallback_threshold_deg']:.3f}")
    print(f"max_switch_wheel_speed={best_final_params['max_switch_wheel_speed']:.3f}")
    print(f"swingup_gain={best_final_params['swingup_gain']:.3f}")
    print(f"pid_torque_scale={float(args.max_torque):.6f}")
    print(f"band_score={best_final_score:.4f}")
    print(f"time_in_band={best_final.time_in_band:.3f}s")
    print(f"first_reach_time={best_final.first_reach_time:.3f}s")
    print(f"longest_hold_time={best_final.longest_hold_time:.3f}s")
    print(f"band_exit_count={best_final.band_exit_count}")
    print(f"falls_count={best_final.falls_count}")
    print(f"upright_fraction={best_final.upright_fraction:.4f}")
    print(f"balance_mode_fraction={best_final.balance_mode_fraction:.4f}")
    print(f"fall_rate={best_final.fall_rate:.4f}")
    print(f"saturation_fraction={best_final.saturation_fraction:.4f}")
    print(f"handoff_count={best_final.handoff_count}")


def _parse_range(spec: str) -> list[float]:
    vals = [float(v.strip()) for v in spec.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Invalid range string: {spec}")
    return vals


def run_swingup_sweep(args: argparse.Namespace) -> None:
    project = load_project_config(args.config)
    base_cfg = build_eval_config(project.simulation, args)

    gains = _parse_range(args.swingup_gain_grid)
    escape_fracs = _parse_range(args.escape_fraction_grid)
    switch_degs = _parse_range(args.switch_threshold_grid)
    switch_vels = _parse_range(args.switch_velocity_grid)

    seed_list = [args.seed + i for i in range(args.eval_rollouts)]
    rows = []

    combos = list(itertools.product(gains, escape_fracs, switch_degs, switch_vels))
    print(f"Running sweep over {len(combos)} combinations...")

    for idx, (gain, esc_frac, sw_deg, sw_vel) in enumerate(combos, start=1):
        cfg = deepcopy(base_cfg)
        cfg.swingup_gain = gain
        cfg.swingup_escape_torque_fraction = esc_frac
        cfg.swingup_escape_torque = min(cfg.max_wheel_torque, esc_frac * cfg.max_wheel_torque)
        cfg.switch_threshold_deg = sw_deg
        cfg.max_switch_velocity = sw_vel

        metrics = evaluate_candidate(cfg, seed_list, args.upright_band_deg)
        rows.append(
            {
                "score": metrics.score,
                "time_to_upright": metrics.time_to_upright,
                "settle_time": metrics.settle_time,
                "fall_rate": metrics.fall_rate,
                "saturation_fraction": metrics.saturation_fraction,
                "handoff_count": metrics.handoff_count,
                "swingup_gain": gain,
                "escape_fraction": esc_frac,
                "switch_threshold_deg": sw_deg,
                "max_switch_velocity": sw_vel,
            }
        )

        if idx % max(1, len(combos) // 10) == 0:
            print(f"  progress {idx}/{len(combos)}")

    rows.sort(key=lambda r: r["score"])

    out_dir = args.output or get_output_dir("training", subdir="pid_swingup_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "swingup_sweep_ranked.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Top sweep candidates ===")
    for i, r in enumerate(rows[: args.top_k], start=1):
        print(
            f"#{i} score={r['score']:.3f} | t_up={r['time_to_upright']:.2f}s "
            f"settle={r['settle_time']:.2f}s fall={r['fall_rate']:.3f} "
            f"sat={r['saturation_fraction']:.3f} | "
            f"gain={r['swingup_gain']:.3f} esc_frac={r['escape_fraction']:.3f} "
            f"sw_deg={r['switch_threshold_deg']:.1f} sw_vel={r['max_switch_velocity']:.2f}"
        )

    print(f"\nRanked sweep written to: {out_csv}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PID tuning and swing-up parameter sweep")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project_config.yaml")
    parser.add_argument(
        "--mode",
        choices=["tune-pid", "tune-pid-sh", "sweep-swingup"],
        default="tune-pid",
    )
    parser.add_argument("--output", type=Path, default=None)

    # Core defaults aligned with real hardware target.
    parser.add_argument("--max-torque", type=float, default=0.30)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-disturbance", action="store_true")

    # Low-torque handoff tightening.
    parser.add_argument("--switch-threshold-deg", type=float, default=14.0)
    parser.add_argument("--max-switch-velocity", type=float, default=2.0)
    parser.add_argument("--max-switch-wheel-speed", type=float, default=45.0)

    # PID random search bounds.
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--kp-min", type=float, default=0.5)
    parser.add_argument("--kp-max", type=float, default=8.0)
    parser.add_argument("--ki-min", type=float, default=0.0)
    parser.add_argument("--ki-max", type=float, default=1.0)
    parser.add_argument("--kd-min", type=float, default=0.0)
    parser.add_argument("--kd-max", type=float, default=2.0)
    parser.add_argument("--upright-band-deg", type=float, default=12.0)
    parser.add_argument(
        "--objective",
        choices=["swingup", "band"],
        default="band",
        help="Objective for tune-pid random search.",
    )

    # Joint successive-halving search options.
    parser.add_argument("--sh-initial-candidates", type=int, default=48)
    parser.add_argument("--sh-keep-ratio", type=float, default=0.4)
    parser.add_argument("--sh-rollout-schedule", type=str, default="1,2,4")
    parser.add_argument("--sh-duration-schedule", type=str, default="4,6,8")
    parser.add_argument("--switch-threshold-min-deg", type=float, default=10.0)
    parser.add_argument("--switch-threshold-max-deg", type=float, default=18.0)
    parser.add_argument("--max-switch-velocity-min", type=float, default=1.0)
    parser.add_argument("--max-switch-velocity-max", type=float, default=3.0)
    parser.add_argument("--fallback-threshold-min-deg", type=float, default=50.0)
    parser.add_argument("--fallback-threshold-max-deg", type=float, default=95.0)
    parser.add_argument("--max-switch-wheel-speed-min", type=float, default=30.0)
    parser.add_argument("--max-switch-wheel-speed-max", type=float, default=70.0)
    parser.add_argument("--swingup-gain-min", type=float, default=1.5)
    parser.add_argument("--swingup-gain-max", type=float, default=3.2)

    # Swing-up sweep grids.
    parser.add_argument("--swingup-gain-grid", type=str, default="1.5,2.0,2.5,3.0")
    parser.add_argument("--escape-fraction-grid", type=str, default="0.7,0.8,0.9,1.0")
    parser.add_argument("--switch-threshold-grid", type=str, default="10,12,14")
    parser.add_argument("--switch-velocity-grid", type=str, default="1.2,1.6,2.0")
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.mode == "tune-pid":
        run_pid_random_search(args)
    elif args.mode == "tune-pid-sh":
        run_pid_successive_halving(args)
    else:
        run_swingup_sweep(args)


if __name__ == "__main__":
    main()








