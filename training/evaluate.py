"""
Example evaluation script for a trained NEAT genome.

Runs one simulation rollout, prints fitness breakdown, and generates diagnostic plots.
Use after training/train.py to inspect the best individual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config, weights_for_generation
from neat.evolution import EvolutionConfig, EvolutionEngine
from neat.fitness import FitnessEvaluator, evaluate_rollout
from neat.network import FeedforwardNetwork
from simulation.pendulum_env import PendulumEnv
from visualisation.network_graph import draw_network_topology
from visualisation.plots import plot_fft, plot_resonance_comparison, plot_rollout


def evaluate_best_after_short_train(output_dir: Path, show: bool) -> None:
    """Quick demo: train a few generations, then evaluate the best genome."""
    project = load_project_config()
    evaluator = FitnessEvaluator(project)
    evo_cfg = EvolutionConfig(population_size=project.population_size)
    engine = EvolutionEngine(evo_cfg, fitness_fn=evaluator.make_fitness_fn(), seed=project.seed)

    gens = min(5, project.generations)
    for gen in range(gens):
        evaluator.generation = gen
        engine.run_generation()

    genome = engine.get_best_genome()
    network = FeedforwardNetwork(genome)
    env = PendulumEnv(project.simulation)
    result = env.run_episode(network=network)

    weights = weights_for_generation(project, gens - 1)
    sample_rate = 1.0 / project.simulation.dt
    breakdown = evaluate_rollout(
        result,
        weights,
        sample_rate,
        project.spectral.target_band_hz,
        project.spectral.noise_band_hz,
        project.simulation.max_wheel_speed,
    )

    print("Fitness breakdown:")
    print(f"  Total:         {breakdown.total:.4f}")
    print(f"  Stability (S): {breakdown.stability:.4f}")
    print(f"  Amplify (A):   {breakdown.amplification:.4f}")
    print(f"  Noise (N):     {breakdown.noise:.4f}")
    print(f"  Effort (E):    {breakdown.effort:.4f}")
    print(f"  Unsafe (U):    {breakdown.unsafe:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rollout(
        result,
        target_band_hz=project.spectral.target_band_hz,
        save_path=output_dir / "rollout.png",
        show=show,
    )
    plot_fft(
        result.angle,
        sample_rate,
        title="Pendulum angle FFT",
        save_path=output_dir / "angle_fft.png",
        show=show,
    )
    plot_resonance_comparison(
        result,
        project.spectral.target_band_hz,
        save_path=output_dir / "resonance_bands.png",
        show=show,
    )
    draw_network_topology(genome, save_path=output_dir / "topology.png", show=show)
    print(f"Evaluation plots saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NEAT controller")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    evaluate_best_after_short_train(args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
