"""Train NEAT for NN-only balance (swing-up retained, no PID in BALANCE mode)."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# Project root on path when run as script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from neat.evolution import EvolutionConfig, EvolutionEngine
from neat.genome import Genome
from neat.fitness_nn_only import NNOnlyFitnessEvaluator
from visualisation_code.output import get_output_dir
from visualisation_code.plots import (
    plot_fitness_history,
    plot_species_history,
    plot_training_summary,
    plot_weight_schedule_history,
)
from visualisation_code.run_registry import (
    append_nn_only_index,
    create_nn_only_run_dir,
    snapshot_config,
    update_run_metadata,
)

NN_ONLY_RUNS_DIR = ROOT / "outputs" / "runs" / "nn_only"


def _save_placeholder_schedule_csv(output_dir: Path) -> dict:
    """
    Keep output contract similar to existing workflow.

    NN-only fitness does not use the hybrid schedule, so we persist flat traces to
    retain equivalent plotting/export behaviour.
    """
    history = {
        "generation": [0.0],
        "sigmoid": [0.0],
        "stability_weight": [0.0],
        "amplification_weight": [0.0],
    }
    csv_path = output_dir / "nn_only_weight_schedule_history.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "sigmoid", "stability_weight", "amplification_weight"])
        writer.writerow([0, 0.0, 0.0, 0.0])
    return history


def _format_band(band: list[float]) -> str:
    if len(band) < 2:
        return ""
    return f"{float(band[0]):.3f}-{float(band[1]):.3f}"


def _load_custom_neat_overrides(path: Path) -> dict[str, Any]:
    """Load optional `custom_neat` overrides from YAML, returning an empty dict on failure."""
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    custom_neat = data.get("custom_neat")
    if isinstance(custom_neat, dict):
        return custom_neat
    return {}


def _load_warm_start_overrides(path: Path) -> dict[str, Any]:
    """Load optional `warm_start` overrides from the NN-only project YAML."""
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    warm_start = data.get("warm_start")
    if isinstance(warm_start, dict):
        return warm_start
    return {}


def _load_early_stopping_overrides(path: Path) -> dict[str, Any]:
    """Load optional `early_stopping` overrides from the NN-only project YAML."""
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    early_stopping = data.get("early_stopping")
    if isinstance(early_stopping, dict):
        return early_stopping
    return {}


def _run_order_key(run_id: str, timestamp: str | None = None) -> tuple[str, str]:
    """Ordering key for NN-only runs (timestamp first, then run_id)."""
    ts = (timestamp or "").strip()
    return ts, run_id


def _resolve_latest_nn_only_warm_start_genome(runs_dir: Path) -> Path | None:
    """Resolve latest archived NN-only best genome path, if available."""
    candidates: list[tuple[tuple[str, str], Path]] = []

    if runs_dir.exists():
        for child in runs_dir.iterdir():
            if not child.is_dir():
                continue
            artifact_path = child / "artifacts" / "best_genome_nn_only.pkl"
            if artifact_path.exists():
                run_id = child.name
                timestamp = run_id.split("__", maxsplit=1)[0] if "__" in run_id else run_id
                candidates.append((_run_order_key(run_id=run_id, timestamp=timestamp), artifact_path))

    index_path = runs_dir / "index.csv"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    run_id = str(row.get("run_id", "") or "").strip()
                    timestamp = str(row.get("timestamp", "") or "").strip()
                    run_dir_str = str(row.get("run_dir", "") or "").strip()
                    if not run_id and not run_dir_str:
                        continue

                    if run_dir_str:
                        run_dir = Path(run_dir_str)
                    else:
                        run_dir = runs_dir / run_id
                    artifact_path = run_dir / "artifacts" / "best_genome_nn_only.pkl"
                    if artifact_path.exists():
                        if not run_id:
                            run_id = run_dir.name
                        candidates.append((_run_order_key(run_id=run_id, timestamp=timestamp), artifact_path))
        except OSError:
            pass

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def train_nn_only(
    config_path: Path | None,
    output_dir: Path | None,
    show_plots: bool,
    warm_start_genome_path: Path | None = None,
    warm_start_fraction: float | None = None,
    early_stop_patience: int | None = None,
) -> None:
    project = load_project_config(config_path)
    evaluator = NNOnlyFitnessEvaluator(project, config_path=config_path)
    neat_overrides = _load_custom_neat_overrides(ROOT / "configs" / "neat_config.yaml")
    effective_config_path = (
        config_path if config_path is not None else ROOT / "configs" / "project_config_nn_only.yaml"
    )
    warm_start_overrides = _load_warm_start_overrides(effective_config_path)
    early_stopping_overrides = _load_early_stopping_overrides(effective_config_path)

    cfg_warm_path_raw = warm_start_overrides.get("warm_start_genome")
    cfg_warm_fraction_raw = warm_start_overrides.get("warm_start_fraction")
    cfg_warm_mutate_raw = warm_start_overrides.get("warm_start_mutate", True)

    resolved_warm_start_path: Path | None = None
    warm_start_from_latest = False
    if warm_start_genome_path is not None:
        resolved_warm_start_path = warm_start_genome_path
    else:
        cfg_warm_path = str(cfg_warm_path_raw).strip() if cfg_warm_path_raw is not None else ""
        if not cfg_warm_path or cfg_warm_path.lower() == "latest":
            warm_start_from_latest = True
            resolved_warm_start_path = _resolve_latest_nn_only_warm_start_genome(NN_ONLY_RUNS_DIR)
        else:
            candidate = Path(cfg_warm_path)
            if not candidate.is_absolute():
                candidate = effective_config_path.parent / candidate
            resolved_warm_start_path = candidate

    if warm_start_fraction is not None:
        effective_warm_start_fraction = float(warm_start_fraction)
    elif cfg_warm_fraction_raw is not None:
        effective_warm_start_fraction = float(cfg_warm_fraction_raw)
    elif resolved_warm_start_path is not None:
        effective_warm_start_fraction = 0.08
    else:
        effective_warm_start_fraction = 0.0
    effective_warm_start_fraction = max(0.0, min(1.0, effective_warm_start_fraction))

    warm_start_mutate = bool(cfg_warm_mutate_raw)
    warm_start_genome: Genome | None = None
    if resolved_warm_start_path is not None:
        if not resolved_warm_start_path.exists():
            raise FileNotFoundError(
                f"Warm-start genome not found: {resolved_warm_start_path}"
            )
        with open(resolved_warm_start_path, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, Genome):
            raise TypeError(
                f"Warm-start genome at {resolved_warm_start_path} is not a Genome instance"
            )
        warm_start_genome = loaded

    if warm_start_genome is None:
        effective_warm_start_fraction = 0.0
        if warm_start_from_latest:
            print("Warm start disabled: no archived NN-only genome found in outputs/runs/nn_only")

    if resolved_warm_start_path is not None:
        print(f"Warm-start genome resolved: {resolved_warm_start_path.resolve()}")
    else:
        print("Warm-start genome resolved: None")

    cfg_early_patience_raw = early_stopping_overrides.get("patience")
    if early_stop_patience is not None:
        effective_early_stop_patience = int(early_stop_patience)
    elif cfg_early_patience_raw is not None:
        effective_early_stop_patience = int(cfg_early_patience_raw)
    else:
        effective_early_stop_patience = 0
    effective_early_stop_patience = max(0, effective_early_stop_patience)

    run_id, timestamp, run_dir = create_nn_only_run_dir(
        population_size=project.population_size,
        generations=project.generations,
        initial_angle_rad=project.simulation.initial_angle,
    )
    snapshot_path = snapshot_config(config_path, run_dir)

    evo_cfg = EvolutionConfig(
        population_size=project.population_size,
        num_inputs=4,
        num_outputs=1,
        compatibility_threshold=float(
            neat_overrides.get("compatibility_threshold", EvolutionConfig.compatibility_threshold)
        ),
        max_stagnation=int(neat_overrides.get("max_stagnation", EvolutionConfig.max_stagnation)),
        elitism=int(neat_overrides.get("elitism", EvolutionConfig.elitism)),
        survival_threshold=float(
            neat_overrides.get("survival_threshold", EvolutionConfig.survival_threshold)
        ),
        weight_mutate_rate=float(
            neat_overrides.get("weight_mutate_rate", EvolutionConfig.weight_mutate_rate)
        ),
        add_conn_rate=float(neat_overrides.get("add_conn_rate", EvolutionConfig.add_conn_rate)),
        add_node_rate=float(neat_overrides.get("add_node_rate", EvolutionConfig.add_node_rate)),
        toggle_rate=float(neat_overrides.get("toggle_rate", EvolutionConfig.toggle_rate)),
        random_immigrant_rate=float(
            neat_overrides.get("random_immigrant_rate", EvolutionConfig.random_immigrant_rate)
        ),
        warm_start_genome=warm_start_genome,
        warm_start_fraction=effective_warm_start_fraction,
        warm_start_mutate=warm_start_mutate,
    )
    engine = EvolutionEngine(
        evo_cfg,
        fitness_fn=evaluator.make_fitness_fn(),
        seed=project.seed,
    )

    print("Starting custom NEAT training (NN-only BALANCE control)...")
    print(f"Population={evo_cfg.population_size}, Generations={project.generations}")
    if warm_start_genome is not None and evo_cfg.warm_start_fraction > 0.0:
        seeded = int(round(evo_cfg.population_size * evo_cfg.warm_start_fraction))
        print(
            "Warm start enabled: "
            f"seeded={seeded}/{evo_cfg.population_size} "
            f"({evo_cfg.warm_start_fraction:.2%}) "
            f"from {resolved_warm_start_path}"
        )

    interrupted = False
    stopped_early = False
    early_stop_trigger_generation: int | None = None
    early_stop_best_fitness: float | None = None
    no_improve_generations = 0
    best_seen = float("-inf")
    completed_generations = 0
    try:
        for gen in range(project.generations):
            evaluator.generation = gen
            best, mean, n_species = engine.run_generation()
            completed_generations = gen + 1

            if best > best_seen:
                best_seen = best
                no_improve_generations = 0
            else:
                no_improve_generations += 1

            print(
                f"Gen {gen + 1:3d}/{project.generations} | "
                f"best={best:.4f} mean={mean:.4f} species={n_species}"
            )

            if (
                effective_early_stop_patience > 0
                and no_improve_generations >= effective_early_stop_patience
            ):
                stopped_early = True
                early_stop_trigger_generation = gen + 1
                early_stop_best_fitness = float(best_seen)
                print(
                    "Early stopping triggered: "
                    f"no best-fitness improvement for {effective_early_stop_patience} generations."
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user (Ctrl+C). Finalising run artifacts...")

    if completed_generations == 0:
        print("No completed generations yet; skipping finalisation. Re-run and stop after at least one generation.")
        return

    best_genome = engine.get_best_genome()

    artifact_genome_path = run_dir / "artifacts" / "best_genome_nn_only.pkl"
    with open(artifact_genome_path, "wb") as f:
        pickle.dump(best_genome, f)

    with open("best_genome_nn_only.pkl", "wb") as f:
        pickle.dump(best_genome, f)
    print("Saved best NN-only genome.")

    run_training_figures_dir = run_dir / "figures" / "training"
    run_training_figures_dir.mkdir(parents=True, exist_ok=True)
    schedule_history = _save_placeholder_schedule_csv(run_training_figures_dir)

    run_fitness_history_path = run_training_figures_dir / "fitness_history.png"
    plot_fitness_history(
        best=engine.history_best,
        mean=engine.history_mean,
        save_path=run_fitness_history_path,
        show=False,
    )

    run_species_history_path = run_training_figures_dir / "species_history.png"
    plot_species_history(
        species_counts=engine.history_species,
        save_path=run_species_history_path,
        show=False,
    )

    run_schedule_path = run_training_figures_dir / "weight_schedule.png"
    plot_weight_schedule_history(
        generations=schedule_history.get("generation", []),
        stability_weights=schedule_history.get("stability_weight", []),
        amplification_weights=schedule_history.get("amplification_weight", []),
        sigmoid_values=schedule_history.get("sigmoid"),
        save_path=run_schedule_path,
        show=False,
    )

    run_training_summary_path = run_training_figures_dir / "training_summary.png"
    plot_training_summary(
        best=engine.history_best,
        mean=engine.history_mean,
        species_counts=engine.history_species,
        schedule_generations=schedule_history.get("generation"),
        stability_weights=schedule_history.get("stability_weight"),
        amplification_weights=schedule_history.get("amplification_weight"),
        sigmoid_values=schedule_history.get("sigmoid"),
        save_path=run_training_summary_path,
        show=False,
    )

    if output_dir is not None:
        legacy_figures_dir = output_dir
        legacy_figures_dir.mkdir(parents=True, exist_ok=True)
    else:
        legacy_figures_dir = get_output_dir("training", subdir="nn_only")

    _save_placeholder_schedule_csv(legacy_figures_dir)

    plot_fitness_history(
        best=engine.history_best,
        mean=engine.history_mean,
        save_path=legacy_figures_dir / "fitness_history.png",
        show=False,
    )
    plot_species_history(
        species_counts=engine.history_species,
        save_path=legacy_figures_dir / "species_history.png",
        show=False,
    )
    plot_weight_schedule_history(
        generations=schedule_history.get("generation", []),
        stability_weights=schedule_history.get("stability_weight", []),
        amplification_weights=schedule_history.get("amplification_weight", []),
        sigmoid_values=schedule_history.get("sigmoid"),
        save_path=legacy_figures_dir / "weight_schedule.png",
        show=False,
    )

    plot_training_summary(
        best=engine.history_best,
        mean=engine.history_mean,
        species_counts=engine.history_species,
        schedule_generations=schedule_history.get("generation"),
        stability_weights=schedule_history.get("stability_weight"),
        amplification_weights=schedule_history.get("amplification_weight"),
        sigmoid_values=schedule_history.get("sigmoid"),
        save_path=legacy_figures_dir / "training_summary.png",
        show=False,
    )

    config_path_str = str(config_path.resolve()) if config_path is not None else None
    nn_only_weights = asdict(evaluator.weights)
    run_metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": "nn_only",
        "status": "interrupted" if interrupted else "completed",
        "config_source_path": config_path_str,
        "fitness_function": "evaluate_nn_only_rollout",
        "nn_only_fitness": nn_only_weights,
        "evolution": {
            "population_size": int(project.population_size),
            "generations": int(project.generations),
            "completed_generations": int(completed_generations),
            "seed": int(project.seed),
            "neat_effective": {
                "compatibility_threshold": float(evo_cfg.compatibility_threshold),
                "max_stagnation": int(evo_cfg.max_stagnation),
                "elitism": int(evo_cfg.elitism),
                "survival_threshold": float(evo_cfg.survival_threshold),
                "weight_mutate_rate": float(evo_cfg.weight_mutate_rate),
                "add_conn_rate": float(evo_cfg.add_conn_rate),
                "add_node_rate": float(evo_cfg.add_node_rate),
                "toggle_rate": float(evo_cfg.toggle_rate),
                "random_immigrant_rate": float(evo_cfg.random_immigrant_rate),
                "warm_start": {
                    "enabled": bool(
                        warm_start_genome is not None and evo_cfg.warm_start_fraction > 0.0
                    ),
                    "genome_path": (
                        str(resolved_warm_start_path.resolve())
                        if resolved_warm_start_path is not None
                        else None
                    ),
                    "warm_start_fraction": float(evo_cfg.warm_start_fraction),
                    "warm_start_mutate": bool(evo_cfg.warm_start_mutate),
                    "seeded_count": int(
                        round(evo_cfg.population_size * evo_cfg.warm_start_fraction)
                    ),
                },
                "early_stopping": {
                    "patience": int(effective_early_stop_patience),
                    "enabled": bool(effective_early_stop_patience > 0),
                    "stopped_early": bool(stopped_early),
                    "trigger_generation": (
                        int(early_stop_trigger_generation)
                        if early_stop_trigger_generation is not None
                        else None
                    ),
                    "best_fitness_at_stop": early_stop_best_fitness,
                },
            },
        },
        "simulation": {
            "initial_angle_rad": float(project.simulation.initial_angle),
            "initial_angle_deg": float(math.degrees(project.simulation.initial_angle)),
            "dt": float(project.simulation.dt),
            "duration": float(project.simulation.duration),
            "max_wheel_torque": float(project.simulation.max_wheel_torque),
            "max_wheel_speed": float(project.simulation.max_wheel_speed),
        },
        "spectral": {
            "target_band_hz": list(project.spectral.target_band_hz),
            "noise_band_hz": list(project.spectral.noise_band_hz),
        },
        "artifacts": {
            "best_genome_run_copy": str(artifact_genome_path.resolve()),
            "best_genome_legacy": str((ROOT / "best_genome_nn_only.pkl").resolve()),
        },
        "figures": {
            "training": {
                "fitness_history": str(run_fitness_history_path.resolve()),
                "species_history": str(run_species_history_path.resolve()),
                "weight_schedule": str(run_schedule_path.resolve()),
                "training_summary": str(run_training_summary_path.resolve()),
            },
        },
        "metadata": {
            "config_snapshot": str(snapshot_path.resolve()),
        },
        "final_best_fitness": float(best_genome.fitness),
    }
    update_run_metadata(run_dir, run_metadata)

    # Generate evaluation + topology diagnostics directly into the run archive.
    from training.evaluate_nn_only import evaluate_nn_only

    evaluate_nn_only(
        config_path=config_path if config_path is not None else ROOT / "configs" / "project_config_nn_only.yaml",
        genome_path=artifact_genome_path,
        output_dir=None,
        run_dir=run_dir,
        show=False,
    )

    append_nn_only_index(
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "pop": int(project.population_size),
            "generations": int(completed_generations),
            "initial_angle_deg": f"{math.degrees(project.simulation.initial_angle):.3f}",
            "target_band": _format_band(project.spectral.target_band_hz),
            "noise_band": _format_band(project.spectral.noise_band_hz),
            "best_fitness": f"{float(best_genome.fitness):.6f}",
            "run_dir": str(run_dir.resolve()),
        }
    )

    # Display plots only after all run artifacts are safely persisted.
    if show_plots:
        plot_training_summary(
            best=engine.history_best,
            mean=engine.history_mean,
            species_counts=engine.history_species,
            schedule_generations=schedule_history.get("generation"),
            stability_weights=schedule_history.get("stability_weight"),
            amplification_weights=schedule_history.get("amplification_weight"),
            sigmoid_values=schedule_history.get("sigmoid"),
            save_path=None,
            show=True,
        )

    print(f"Training complete. Best fitness: {best_genome.fitness:.4f}")
    print(f"Legacy training plots saved to {legacy_figures_dir}")
    print(f"Run archive saved to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN-only NEAT controller")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "project_config_nn_only.yaml",
        help="NN-only project YAML config",
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
    parser.add_argument(
        "--warm-start-genome",
        type=Path,
        default=None,
        help="Optional path to a pickled Genome used to seed the initial population",
    )
    parser.add_argument(
        "--warm-start-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of initial population to seed from warm-start genome "
            "(CLI overrides config; defaults to 0.08 when warm-start genome is set)"
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help=(
            "Stop training when best fitness does not improve for N generations "
            "(0 disables; CLI overrides config)"
        ),
    )
    args = parser.parse_args()
    train_nn_only(
        args.config,
        args.output,
        show_plots=not args.no_show,
        warm_start_genome_path=args.warm_start_genome,
        warm_start_fraction=args.warm_start_fraction,
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    main()
