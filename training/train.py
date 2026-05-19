"""
Example training loop for NEAT-based supplementary pendulum control.

Supports two backends:
  - custom : educational NEAT implementation in neat/
  - neat-python : Stanley's reference algorithm via neat-python library

Data flow:
  genome -> FeedforwardNetwork -> normalized u_NN
  sensors -> PID placeholder -> u_PID
  HybridController -> torque -> PendulumEnv -> time series -> FFT fitness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root on path when run as script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from neat.evolution import EvolutionConfig, EvolutionEngine
from neat.fitness import FitnessEvaluator
from visualisation.plots import plot_fitness_history, plot_species_history


def train_custom(config_path: Path | None, output_dir: Path, show_plots: bool) -> None:
    """Train using the custom educational NEAT engine."""
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
    print(f"Population={evo_cfg.population_size}, Generations={project.generations}")
    for gen in range(project.generations):
        evaluator.generation = gen
        best, mean, n_species = engine.run_generation()
        print(
            f"Gen {gen + 1:3d}/{project.generations} | "
            f"best={best:.4f} mean={mean:.4f} species={n_species}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fitness_history(
        engine.history_best,
        engine.history_mean,
        save_path=output_dir / "fitness_history.png",
        show=show_plots,
    )
    plot_species_history(
        engine.history_species,
        save_path=output_dir / "species_history.png",
        show=show_plots,
    )
    best_genome = engine.get_best_genome()
    print(f"Training complete. Best fitness: {best_genome.fitness:.4f}")
    print(f"Plots saved to {output_dir}")


def train_neat_python(config_path: Path, output_dir: Path, show_plots: bool) -> None:
    """
    Train using neat-python (primary backend per project spec).

    The fitness function reuses our simulation and spectral evaluator.
    """
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
        for _gid, genome in genomes:
            # Wrap neat-python genome with our fitness (via temporary adapter).
            genome.fitness = _evaluate_neat_python_genome(genome, config_obj, evaluator)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(eval_genomes, project.generations)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_history = [s.best_genome().fitness for s in stats.generation_statistics] if stats.generation_statistics else []
    if best_history:
        plot_fitness_history(
            best_history,
            save_path=output_dir / "fitness_history_neatpy.png",
            show=show_plots,
        )
    print(f"neat-python winner fitness: {winner.fitness}")


def _evaluate_neat_python_genome(genome, neat_config, evaluator: FitnessEvaluator) -> float:
    """Bridge neat-python genome to our pendulum fitness evaluator."""
    import neat

    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    env = evaluator.env

    # Run episode by stepping with neat-python network outputs.
    cfg = evaluator.config.simulation
    steps = int(cfg.duration / cfg.dt)
    theta = cfg.initial_angle
    omega = cfg.initial_omega
    wheel_omega = 0.0
    from simulation.pendulum_env import PendulumEnv
    from control.hybrid_controller import HybridController
    from control.pid_interface import PlaceholderPIDController
    from simulation.sensors import SimulatedSensor

    sensor = SimulatedSensor()
    pid = PlaceholderPIDController()
    hybrid = HybridController(
        pid=pid,
        alpha=cfg.alpha,
        nn_torque_scale=cfg.nn_torque_scale,
    )

    import numpy as np
    import time as _time

    t_arr = np.linspace(0, cfg.duration, steps, endpoint=False)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    accels = np.zeros(steps)
    wheels = np.zeros(steps)
    nn_out = np.zeros(steps)
    pid_out = np.zeros(steps)
    total_out = np.zeros(steps)
    seismic = np.zeros(steps)

    for i, t in enumerate(t_arr):
        base_a = env._seismic_accel(t)
        seismic[i] = base_a
        sensor.update_raw(theta, omega, base_a, wheel_omega)
        reading = sensor.read()
        inp = reading.as_array()
        u_nn = float(net.activate(inp)[0])
        u_pid = pid.compute(reading)
        torque = hybrid.compute_total_torque(u_pid, u_nn)
        torque = np.clip(torque, -cfg.max_wheel_torque, cfg.max_wheel_torque)
        nn_out[i] = u_nn
        pid_out[i] = u_pid
        total_out[i] = torque
        theta, omega, wheel_omega = env._integrate_step(
            theta, omega, wheel_omega, torque, base_a, cfg.dt
        )
        angles[i] = theta
        omegas[i] = omega
        accels[i] = base_a
        wheels[i] = wheel_omega

    from simulation.pendulum_env import SimulationResult
    from neat.fitness import evaluate_rollout
    from config.settings import weights_for_generation

    result = SimulationResult(
        time=t_arr,
        angle=angles,
        angular_velocity=omegas,
        base_acceleration=accels,
        wheel_velocity=wheels,
        nn_output=nn_out,
        pid_output=pid_out,
        total_torque=total_out,
        seismic_input=seismic,
    )
    weights = weights_for_generation(evaluator.config, evaluator.generation)
    breakdown = evaluate_rollout(
        result,
        weights,
        1.0 / cfg.dt,
        evaluator.config.spectral.target_band_hz,
        evaluator.config.spectral.noise_band_hz,
        cfg.max_wheel_speed,
    )
    return breakdown.total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEAT supplementary controller")
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
        default=ROOT / "outputs" / "training",
        help="Directory for plots and logs",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plots interactively")
    args = parser.parse_args()

    show = not args.no_show
    if args.backend == "custom":
        train_custom(args.config, args.output, show)
    else:
        train_neat_python(args.neat_config, args.output, show)


if __name__ == "__main__":
    main()
