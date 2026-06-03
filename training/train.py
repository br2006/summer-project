

"""
Example training loop for NEAT-based supplementary pendulum control.

Supports two backends:
  - custom : educational NEAT implementation in neat/
  - neat-python : Stanley's reference algorithm via neat-python library
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

# Project root on path when run as script.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from neat.evolution import EvolutionConfig, EvolutionEngine
from neat.fitness import FitnessEvaluator
from visualisation_code.plots import (
    plot_training_summary,
)
from visualisation_code.output import get_output_dir



def _save_weight_schedule_csv(
    evaluator: FitnessEvaluator,
    output_dir: Path,
    filename_prefix: str = "",
) -> dict:
    """
    Persist generation-dependent schedule diagnostics to CSV.

    Records sigmoid(g), stability weight, and amplification weight for each
    generation so schedule behaviour can be inspected after training.
    """
    history = evaluator.get_weight_schedule_history()
    generations = history.get("generation", [])

    if not generations:
        return history

    prefix = f"{filename_prefix}_" if filename_prefix else ""

    csv_path = output_dir / f"{prefix}weight_schedule_history.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "sigmoid", "stability_weight", "amplification_weight"])
        for row in zip(
            history["generation"],
            history["sigmoid"],
            history["stability_weight"],
            history["amplification_weight"],
        ):
            writer.writerow(row)
    return history


# ============================================================
# CUSTOM EDUCATIONAL NEAT BACKEND
# ============================================================

def train_custom(
    config_path: Path | None,
    output_dir: Path | None,
    show_plots: bool,
) -> None:

    project = load_project_config(config_path)

    evaluator = FitnessEvaluator(project)

    evo_cfg = EvolutionConfig(
        population_size=project.population_size,
        num_inputs=4,
        num_outputs=1,
    )

    engine = EvolutionEngine(
        evo_cfg,
        fitness_fn=evaluator.make_fitness_fn(),
        seed=project.seed,
    )

    print("Starting custom NEAT training...")

    print(
        f"Population={evo_cfg.population_size}, "
        f"Generations={project.generations}"
    )

    for gen in range(project.generations):

        evaluator.generation = gen

        best, mean, n_species = (
            engine.run_generation()
        )

        print(
            f"Gen {gen + 1:3d}/{project.generations} | "
            f"best={best:.4f} "
            f"mean={mean:.4f} "
            f"species={n_species}"
        )

    # ====================================================
    # SAVE BEST GENOME
    # ====================================================

    best_genome = engine.get_best_genome()

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(best_genome, f)

    print("Saved best genome.")

    # ====================================================
    # OUTPUTS
    # ====================================================

    if output_dir is not None:
        figures_dir = output_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = get_output_dir("training")

    schedule_history = _save_weight_schedule_csv(
        evaluator=evaluator,
        output_dir=figures_dir,
    )


    

    plot_training_summary(
        best=engine.history_best,
        mean=engine.history_mean,
        species_counts=engine.history_species,
        schedule_generations=schedule_history.get("generation"),
        stability_weights=schedule_history.get("stability_weight"),
        amplification_weights=schedule_history.get("amplification_weight"),
        sigmoid_values=schedule_history.get("sigmoid"),
        save_path=figures_dir / "training_summary.png",
        show=show_plots,
    )

    print(
        f"Training complete. "
        f"Best fitness: "
        f"{best_genome.fitness:.4f}"
    )

    print(f"Plots saved to {figures_dir}")


def train_neat_python(
    config_path: Path,
    output_dir: Path | None,
    show_plots: bool,
) -> None:

    import neat

    yaml_path = ROOT / "configs" / "project_config.yaml"

    project = load_project_config(yaml_path)

    evaluator = FitnessEvaluator(project)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    def eval_genomes(genomes, config_obj):
        evaluator.generation = eval_genomes.current_generation
        for _gid, genome in genomes:
            genome.fitness = _evaluate_neat_python_genome(
                genome,
                config_obj,
                evaluator,
            )
        eval_genomes.current_generation += 1

    eval_genomes.current_generation = 0

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    best_genome = population.run(
        eval_genomes,
        project.generations,
    )

    # ========================================================
    # SAVE BEST GENOME
    # ========================================================

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(best_genome, f)

    print("Saved best genome.")

    if output_dir is not None:
        figures_dir = output_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = get_output_dir("training", subdir="neat_python")

    best_history = [
        s.best_genome().fitness
        for s in stats.generation_statistics
    ] if stats.generation_statistics else []

    schedule_history = _save_weight_schedule_csv(
        evaluator=evaluator,
        output_dir=figures_dir,
        filename_prefix="neatpy",
    )

    

    plot_training_summary(
        best=best_history,
        mean=None,
        species_counts=None,
        schedule_generations=schedule_history.get("generation"),
        stability_weights=schedule_history.get("stability_weight"),
        amplification_weights=schedule_history.get("amplification_weight"),
        sigmoid_values=schedule_history.get("sigmoid"),
        save_path=figures_dir / "training_summary_neatpy.png",
        show=show_plots,
    )

    print(
        f"neat-python winner fitness: "
        f"{best_genome.fitness}"
    )

    print(f"Plots saved to {figures_dir}")




# ============================================================
# neat-python FITNESS BRIDGE
# ============================================================

def _evaluate_neat_python_genome(
    genome,
    neat_config,
    evaluator: FitnessEvaluator,
) -> float:

    import neat

    from neat.fitness import evaluate_rollout

    net = neat.nn.FeedForwardNetwork.create(
        genome,
        neat_config,
    )

    cfg = evaluator.config.simulation
    env = evaluator.env
    result = env.run_episode(network=net)

    weights = evaluator.get_generation_weights()

    breakdown = evaluate_rollout(
        result,
        weights,
        1.0 / cfg.dt,
        evaluator.config.spectral.target_band_hz,
        evaluator.config.spectral.noise_band_hz,
        cfg.max_wheel_speed,
    )

    return breakdown.total



# ============================================================
# MAIN ENTRY
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NEAT supplementary controller"
    )
    parser.add_argument(
        "--backend",
        choices=["custom", "neat-python"],
        default="custom",
        help="NEAT implementation (custom=educational, neat-python=library)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "project_config.yaml",
        help="Project YAML config",
    )
    parser.add_argument(
        "--neat-config",
        type=Path,
        default=ROOT / "configs" / "neat_config.ini",
        help="neat-python .ini config (neat-python backend only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom directory for generated figures",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively",
    )

    args = parser.parse_args()
    show = not args.no_show

    if args.backend == "custom":
        train_custom(args.config, args.output, show)
    else:
        train_neat_python(args.neat_config, args.output, show)


if __name__ == "__main__":
    main()


