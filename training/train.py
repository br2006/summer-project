"""
Example training loop for NEAT-based supplementary pendulum control.

Supports two backends:
  - custom : educational NEAT implementation in neat/
  - neat-python : Stanley's reference algorithm via neat-python library
"""

from __future__ import annotations

import argparse
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
from visualisation.plots import (
    plot_fitness_history,
    plot_species_history,
)


# ============================================================
# CUSTOM EDUCATIONAL NEAT BACKEND
# ============================================================

def train_custom(
    config_path: Path | None,
    output_dir: Path,
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_fitness_history(
        engine.history_best,
        engine.history_mean,
        save_path=(
            output_dir
            / "fitness_history.png"
        ),
        show=show_plots,
    )

    plot_species_history(
        engine.history_species,
        save_path=(
            output_dir
            / "species_history.png"
        ),
        show=show_plots,
    )

    print(
        f"Training complete. "
        f"Best fitness: "
        f"{best_genome.fitness:.4f}"
    )

    print(f"Plots saved to {output_dir}")

def train_neat_python(
    config_path: Path,
    output_dir: Path,
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

        for _gid, genome in genomes:

            genome.fitness = _evaluate_neat_python_genome(
                genome,
                config_obj,
                evaluator,
            )

    population = neat.Population(config)

    population.add_reporter(
        neat.StdOutReporter(True)
    )

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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if stats.generation_statistics:

        best_history = [
            s.best_genome().fitness
            for s in stats.generation_statistics
        ]

        plot_fitness_history(
            best_history,
            save_path=(
                output_dir
                / "fitness_history_neatpy.png"
            ),
            show=show_plots,
        )

    print(
        f"neat-python winner fitness: "
        f"{best_genome.fitness}"
    )


# ============================================================
# neat-python FITNESS BRIDGE
# ============================================================

def _evaluate_neat_python_genome(
    genome,
    neat_config,
    evaluator: FitnessEvaluator,
) -> float:

    import neat

    from config.settings import (
        weights_for_generation,
    )

    from control.hybrid_controller import (
        HybridController,
    )

    from control.pid_interface import (
        PlaceholderPIDController,
    )

    from neat.fitness import (
        evaluate_rollout,
    )

    from simulation.disturbance import (
        DisturbanceGenerator,
    )

    from simulation.pendulum_env import (
        SimulationResult,
    )

    from simulation.sensors import (
        SimulatedSensor,
    )

    net = neat.nn.FeedForwardNetwork.create(
        genome,
        neat_config,
    )

    env = evaluator.env

    cfg = evaluator.config.simulation

    steps = int(cfg.duration / cfg.dt)

    theta = cfg.initial_angle
    omega = cfg.initial_omega
    wheel_omega = 0.0

    sensor = SimulatedSensor(
        angle_noise_std=cfg.angle_noise_std,
        gyro_noise_std=cfg.gyro_noise_std,
        gyro_bias_drift_rate=cfg.gyro_bias_drift_rate,
    )

    pid = PlaceholderPIDController()

    hybrid = HybridController(
        pid=pid,
        alpha=cfg.alpha,
        nn_torque_scale=cfg.nn_torque_scale,
    )

    disturbance = DisturbanceGenerator(
        dt=cfg.dt,
        broadband_gain=cfg.broadband_noise_gain,
        vibration_freqs=cfg.vibration_freqs_hz,
        vibration_amps=cfg.vibration_amps,
        impulse_probability=cfg.impulse_probability,
        impulse_magnitude=cfg.impulse_magnitude,
    )

    t_arr = np.linspace(
        0,
        cfg.duration,
        steps,
        endpoint=False,
    )

    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    accels = np.zeros(steps)
    wheels = np.zeros(steps)

    nn_out = np.zeros(steps)
    pid_out = np.zeros(steps)

    cmd_torque = np.zeros(steps)
    actual_torque = np.zeros(steps)

    resonance_log = np.zeros(steps)

    seismic = np.zeros(steps)

    for i, t in enumerate(t_arr):

        base_a = disturbance.sample(t)

        seismic[i] = base_a

        sensor.update_raw(
            theta,
            omega,
            base_a,
            wheel_omega,
        )

        reading = sensor.read()

        inp = reading.as_array()

        u_nn = float(
            net.activate(inp)[0]
        )

        u_pid = pid.compute(reading)

        torque = hybrid.compute_total_torque(
            u_pid,
            u_nn,
        )

        torque = np.clip(
            torque,
            -cfg.max_wheel_torque,
            cfg.max_wheel_torque,
        )

        resonance = env._resonance_disturbance(
            t,
            torque,
        )

        theta, omega, wheel_omega = env._integrate_step(
            theta,
            omega,
            wheel_omega,
            torque,
            base_a,
            resonance,
            cfg.dt,
        )

        nn_out[i] = u_nn
        pid_out[i] = u_pid

        cmd_torque[i] = torque
        actual_torque[i] = env.actual_torque

        resonance_log[i] = resonance

        angles[i] = theta
        omegas[i] = omega
        accels[i] = base_a
        wheels[i] = wheel_omega

    result = SimulationResult(
        time=t_arr,
        angle=angles,
        angular_velocity=omegas,
        base_acceleration=accels,
        wheel_velocity=wheels,
        nn_output=nn_out,
        pid_output=pid_out,
        commanded_torque=cmd_torque,
        actual_torque=actual_torque,
        resonance_signal=resonance_log,
        seismic_input=seismic,
    )

    weights = weights_for_generation(
        evaluator.config,
        evaluator.generation,
    )

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
        description=(
            "Train NEAT supplementary controller"
        )
    )

    parser.add_argument(
        "--backend",
        choices=[
            "custom",
            "neat-python",
        ],
        default="custom",
        help=(
            "NEAT implementation "
            "(custom=educational, "
            "neat-python=library)"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "project_config.yaml"
        ),
        help="Project YAML config",
    )

    parser.add_argument(
        "--neat-config",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "neat_config.ini"
        ),
        help=(
            "neat-python .ini config "
            "(neat-python backend only)"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "training"
        ),
        help="Directory for plots and logs",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively",
    )

    args = parser.parse_args()

    show = not args.no_show

    if args.backend == "custom":

        train_custom(
            args.config,
            args.output,
            show,
        )

    else:

        train_neat_python(
            args.neat_config,
            args.output,
            show,
        )


if __name__ == "__main__":
    main()
