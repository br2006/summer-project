"""Evaluate a trained NN-only genome and generate standard diagnostic figures."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from neat.fitness_nn_only import NNOnlyFitnessEvaluator
from neat.network import FeedforwardNetwork
from simulation.nn_pendulum_env import NNPendulumEnv
from visualisation_code.animation import run_demo
from visualisation_code.network_graph import draw_network_topology
from visualisation_code.output import get_output_dir
from visualisation_code.plots import (
    plot_fft,
    plot_frequency_diagnostics,
    plot_resonance_comparison,
    plot_rollout,
    print_frequency_diagnostics,
)
from visualisation_code.run_registry import update_run_metadata


def evaluate_nn_only(
    config_path: Path,
    genome_path: Path,
    output_dir: Path | None,
    run_dir: Path | None,
    show: bool,
) -> Dict[str, str]:
    project = load_project_config(config_path)

    if not genome_path.exists():
        raise FileNotFoundError(
            f"Could not find genome file: {genome_path}. "
            "Run training/train_nn_only.py first."
        )

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    network = FeedforwardNetwork(genome)
    env = NNPendulumEnv(project.simulation)
    result = env.run_episode(network=network)

    evaluator = NNOnlyFitnessEvaluator(project, config_path=config_path)
    breakdown = evaluator.evaluate_network(network)

    print("NN-only fitness breakdown:")
    print(f"  Total:                {breakdown.total:.4f}")
    print(f"  Balance:              {breakdown.balance:.4f}")
    print(f"  Upright time:         {breakdown.upright_time:.4f}")
    print(f"  Handover quality:     {breakdown.handover:.4f}")
    print(f"  Stability gate:       {breakdown.stability_gate:.4f}")
    print(f"  Amplification (A):    {breakdown.amplification:.4f}")
    print(f"  Gated amp (G*A):      {breakdown.gated_amplification:.4f}")
    print(f"  Effort penalty:       {breakdown.effort:.4f}")
    print(f"  Wheel penalty:        {breakdown.wheel:.4f}")
    print(f"  Unsafe penalty:       {breakdown.unsafe:.4f}")

    print("\nDominant frequency peaks:")
    print_frequency_diagnostics(result)

    if run_dir is not None:
        figures_dir = run_dir / "figures" / "evaluation"
        figures_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = output_dir
        if figures_dir is None:
            figures_dir = get_output_dir("evaluation", subdir="nn_only")
        else:
            figures_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 1.0 / project.simulation.dt

    plot_rollout(
        result,
        target_band_hz=project.spectral.target_band_hz,
        save_path=figures_dir / "rollout.png",
        show=show,
    )
    plot_fft(
        result.angle,
        sample_rate,
        title="Pendulum angle FFT (NN-only)",
        max_freq_hz=20,
        save_path=figures_dir / "angle_fft.png",
        show=show,
    )
    plot_resonance_comparison(
        result,
        project.spectral.target_band_hz,
        project.spectral.noise_band_hz,
        save_path=figures_dir / "resonance_bands.png",
        show=show,
    )
    plot_frequency_diagnostics(
        result,
        target_band_hz=project.spectral.target_band_hz,
        noise_band_hz=project.spectral.noise_band_hz,
        max_freq_hz=20,
        save_path=figures_dir / "frequency_diagnostics.png",
        show=show,
    )
    draw_network_topology(genome, save_path=figures_dir / "topology.png", show=show)

    animation_path = figures_dir / "rollout_animation.gif"
    run_demo(
        genome_path=genome_path,
        frame_stride=5,
        show=False,
        config_path=config_path,
        nn_only=True,
        save_gif_path=animation_path,
    )

    figure_paths = {
        "rollout": str((figures_dir / "rollout.png").resolve()),
        "angle_fft": str((figures_dir / "angle_fft.png").resolve()),
        "resonance_bands": str((figures_dir / "resonance_bands.png").resolve()),
        "frequency_diagnostics": str((figures_dir / "frequency_diagnostics.png").resolve()),
        "topology": str((figures_dir / "topology.png").resolve()),
        "rollout_animation": str(animation_path.resolve()),
    }

    if run_dir is not None:
        update_run_metadata(
            run_dir,
            {
                "figures": {
                    "evaluation": figure_paths,
                },
                "final_best_fitness": float(breakdown.total),
            },
        )

    print(f"Evaluation plots saved to {figures_dir}")
    return figure_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NN-only NEAT controller")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "project_config_nn_only.yaml",
        help="NN-only project YAML config",
    )
    parser.add_argument(
        "--genome",
        type=Path,
        default=ROOT / "best_genome_nn_only.pkl",
        help="Path to trained NN-only genome pickle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom directory for generated figures",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional archived run directory; writes figures to <run-dir>/figures/evaluation",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    evaluate_nn_only(
        config_path=args.config,
        genome_path=args.genome,
        output_dir=args.output,
        run_dir=args.run_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
