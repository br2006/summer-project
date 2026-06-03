"""
Fast random-search tuner for PID gains and max wheel torque.

Usage examples:
  python training/tune_pid.py
  python training/tune_pid.py --trials 400 --seeds 4 --no-show
  python training/tune_pid.py --pd-only --trials 300
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
import shutil
from typing import List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from simulation.pendulum_env import PendulumEnv
from visualisation_code.output import get_output_dir
from visualisation_code.plots import plot_rollout


class ZeroNetwork:
    """Dummy network so PendulumEnv can run with NN authority disabled."""

    def activate(self, _inputs):
        return [0.0]


@dataclass
class TrialResult:
    kp: float
    ki: float
    kd: float
    max_wheel_torque: float
    score: float
    angle_rms: float
    max_angle: float
    omega_rms: float
    torque_rms: float
    sat_fraction: float
    fall_penalty: float


def write_best_to_project_config(
    best: TrialResult,
    config_path: Path,
) -> tuple[Path, Path]:
    """
    Write best PID/torque values back into project config YAML.

    Returns:
        (backup_path, updated_config_path)
    """
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(config_path.suffix + f".bak.{timestamp}")
    shutil.copy2(config_path, backup_path)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML root type in {config_path}; expected mapping.")

    simulation = data.get("simulation")
    if simulation is None:
        simulation = {}
        data["simulation"] = simulation
    if not isinstance(simulation, dict):
        raise ValueError("'simulation' section exists but is not a mapping.")

    simulation["pid_kp"] = float(best.kp)
    simulation["pid_ki"] = float(best.ki)
    simulation["pid_kd"] = float(best.kd)
    simulation["max_wheel_torque"] = float(best.max_wheel_torque)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return backup_path, config_path


def rollout_metrics(result, max_wheel_speed: float) -> dict:
    angle = np.asarray(result.angle)
    omega = np.asarray(result.angular_velocity)
    torque = np.asarray(result.actual_torque)
    wheel = np.asarray(result.wheel_velocity)

    angle_rms = float(np.sqrt(np.mean(angle**2)))
    max_angle = float(np.max(np.abs(angle)))
    omega_rms = float(np.sqrt(np.mean(omega**2)))
    torque_rms = float(np.sqrt(np.mean(torque**2)))
    sat_fraction = float(np.mean(np.abs(wheel) > 0.95 * max_wheel_speed))

    # Additional hard penalty for large excursions (near-fall behaviour).
    fall_penalty = 0.0
    if max_angle > 0.7:
        fall_penalty += (max_angle - 0.7) * 8.0
    if max_angle > 1.2:
        fall_penalty += (max_angle - 1.2) * 18.0

    score = (
        4.0 * angle_rms
        + 2.0 * max_angle
        + 0.4 * omega_rms
        + 0.25 * torque_rms
        + 6.0 * sat_fraction
        + fall_penalty
    )

    return {
        "score": float(score),
        "angle_rms": angle_rms,
        "max_angle": max_angle,
        "omega_rms": omega_rms,
        "torque_rms": torque_rms,
        "sat_fraction": sat_fraction,
        "fall_penalty": float(fall_penalty),
    }


def evaluate_candidate(project, kp: float, ki: float, kd: float, max_wheel_torque: float, seeds: List[int]) -> TrialResult:
    scores = []
    metric_samples = []

    for seed in seeds:
        np.random.seed(seed)
        sim_cfg = project.simulation
        sim_cfg.pid_kp = kp
        sim_cfg.pid_ki = ki
        sim_cfg.pid_kd = kd
        sim_cfg.max_wheel_torque = max_wheel_torque

        # Force NN fully disabled for clean PID tuning.
        sim_cfg.alpha = 0.0
        sim_cfg.nn_torque_scale = 0.0

        # Silence mass-property debug prints per trial.
        with redirect_stdout(StringIO()):
            env = PendulumEnv(sim_cfg)
        result = env.run_episode(network=ZeroNetwork())

        metrics = rollout_metrics(result, max_wheel_speed=sim_cfg.max_wheel_speed)
        scores.append(metrics["score"])
        metric_samples.append(metrics)

    mean_metrics = {
        k: float(np.mean([m[k] for m in metric_samples]))
        for k in metric_samples[0].keys()
    }

    return TrialResult(
        kp=kp,
        ki=ki,
        kd=kd,
        max_wheel_torque=max_wheel_torque,
        score=float(np.mean(scores)),
        angle_rms=mean_metrics["angle_rms"],
        max_angle=mean_metrics["max_angle"],
        omega_rms=mean_metrics["omega_rms"],
        torque_rms=mean_metrics["torque_rms"],
        sat_fraction=mean_metrics["sat_fraction"],
        fall_penalty=mean_metrics["fall_penalty"],
    )


def run_tuning(
    trials: int,
    seeds: int,
    pd_only: bool,
    show: bool,
    output_subdir: str,
    kp_min: float,
    kp_max: float,
    ki_min: float,
    ki_max: float,
    kd_min: float,
    kd_max: float,
    torque_min: float,
    torque_max: float,
    write_config: bool,
    config_path: Path,
) -> None:
    project = load_project_config()
    rng = np.random.default_rng(project.seed)

    seed_list = [project.seed + i for i in range(max(1, seeds))]

    out_dir = get_output_dir("evaluation", output_subdir)
    trials_csv = out_dir / "pid_tuning_trials.csv"

    all_results: List[TrialResult] = []
    best = None

    for idx in range(trials):
        kp = float(rng.uniform(kp_min, kp_max))
        ki = 0.0 if pd_only else float(rng.uniform(ki_min, ki_max))
        kd = float(rng.uniform(kd_min, kd_max))
        max_torque = float(rng.uniform(torque_min, torque_max))

        trial = evaluate_candidate(project, kp, ki, kd, max_torque, seed_list)
        all_results.append(trial)

        if best is None or trial.score < best.score:
            best = trial

        if (idx + 1) % max(1, trials // 10) == 0:
            print(f"[{idx + 1}/{trials}] current best score={best.score:.4f} | kp={best.kp:.3f}, ki={best.ki:.3f}, kd={best.kd:.3f}, max_wheel_torque={best.max_wheel_torque:.3f}")

    all_results.sort(key=lambda r: r.score)

    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for row in all_results:
            writer.writerow(asdict(row))

    assert best is not None

    # Regenerate best rollout for plotting.
    sim_cfg = project.simulation
    sim_cfg.pid_kp = best.kp
    sim_cfg.pid_ki = best.ki
    sim_cfg.pid_kd = best.kd
    sim_cfg.max_wheel_torque = best.max_wheel_torque
    sim_cfg.alpha = 0.0
    sim_cfg.nn_torque_scale = 0.0
    np.random.seed(project.seed)
    with redirect_stdout(StringIO()):
        env = PendulumEnv(sim_cfg)
    best_rollout = env.run_episode(network=ZeroNetwork())

    plot_rollout(
        best_rollout,
        target_band_hz=project.spectral.target_band_hz,
        save_path=out_dir / "best_pid_rollout.png",
        show=show,
    )

    summary_path = out_dir / "best_pid_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Best PID + max_wheel_torque from random search\n")
        f.write(f"trials={trials}, seeds_per_trial={len(seed_list)}, pd_only={pd_only}\n\n")
        f.write(f"kp={best.kp:.6f}\n")
        f.write(f"ki={best.ki:.6f}\n")
        f.write(f"kd={best.kd:.6f}\n")
        f.write(f"max_wheel_torque={best.max_wheel_torque:.6f}\n")
        f.write(f"score={best.score:.6f}\n\n")
        f.write("Mean metrics:\n")
        f.write(f"angle_rms={best.angle_rms:.6f}\n")
        f.write(f"max_angle={best.max_angle:.6f}\n")
        f.write(f"omega_rms={best.omega_rms:.6f}\n")
        f.write(f"torque_rms={best.torque_rms:.6f}\n")
        f.write(f"sat_fraction={best.sat_fraction:.6f}\n")
        f.write(f"fall_penalty={best.fall_penalty:.6f}\n")

    print("\n=== Best candidate ===")
    print(f"kp={best.kp:.4f}, ki={best.ki:.4f}, kd={best.kd:.4f}, max_wheel_torque={best.max_wheel_torque:.4f}")
    print(f"score={best.score:.6f}")
    print(f"angle_rms={best.angle_rms:.6f}, max_angle={best.max_angle:.6f}, sat_fraction={best.sat_fraction:.6f}")
    print(f"Saved trials CSV: {trials_csv}")
    print(f"Saved best rollout plot: {out_dir / 'best_pid_rollout.png'}")
    print(f"Saved summary: {summary_path}")

    if write_config:
        backup_path, updated_path = write_best_to_project_config(best, config_path)
        print("\nUpdated project config with best gains:")
        print(f"  config: {updated_path}")
        print(f"  backup: {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune PID gains and max_wheel_torque via random search")
    parser.add_argument("--trials", type=int, default=250, help="Number of random candidates to evaluate")
    parser.add_argument("--seeds", type=int, default=3, help="Rollout seeds per candidate for robustness")
    parser.add_argument("--pd-only", action="store_true", help="Force ki=0 for faster PD-only tuning")
    parser.add_argument("--output-subdir", type=str, default="pid_tuning", help="Subdirectory under outputs/figures/evaluation")
    parser.add_argument("--no-show", action="store_true", help="Disable matplotlib window")

    parser.add_argument("--kp-min", type=float, default=0.4)
    parser.add_argument("--kp-max", type=float, default=6.0)
    parser.add_argument("--ki-min", type=float, default=0.0)
    parser.add_argument("--ki-max", type=float, default=0.35)
    parser.add_argument("--kd-min", type=float, default=0.02)
    parser.add_argument("--kd-max", type=float, default=1.2)
    parser.add_argument("--torque-min", type=float, default=0.15)
    parser.add_argument("--torque-max", type=float, default=0.9)
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write best kp/ki/kd/max_wheel_torque back into project config YAML (creates timestamped backup)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=ROOT / "configs" / "project_config.yaml",
        help="Path to project config YAML used with --write-config",
    )

    args = parser.parse_args()
    run_tuning(
        trials=max(1, args.trials),
        seeds=max(1, args.seeds),
        pd_only=args.pd_only,
        show=not args.no_show,
        output_subdir=args.output_subdir,
        kp_min=args.kp_min,
        kp_max=args.kp_max,
        ki_min=args.ki_min,
        ki_max=args.ki_max,
        kd_min=args.kd_min,
        kd_max=args.kd_max,
        torque_min=args.torque_min,
        torque_max=args.torque_max,
        write_config=args.write_config,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
