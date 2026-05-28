"""
Fitness evaluation framework for NEAT genomes on the pendulum task.

Fitness combines stability, spectral amplification, noise rejection, control
effort, and unsafe-behaviour penalties. FFT analysis drives the spectral terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config.settings import FitnessWeights, ProjectConfig, weights_for_generation
from neat.genome import Genome
from neat.network import FeedforwardNetwork
from signal_processing.spectral_metrics import compute_spectral_metrics
from simulation.pendulum_env import PendulumEnv, SimulationResult


@dataclass
class FitnessBreakdown:
    """Per-term fitness components for logging and plots."""

    total: float
    stability: float
    amplification: float
    noise: float
    effort: float
    unsafe: float


def stability_metric(result: SimulationResult) -> float:
    """
    S: reward bounded pendulum motion (upright, moderate velocities).

    Penalizes large angles and runaway angular velocity.
    """
    max_angle = float(np.max(np.abs(result.angle)))
    max_omega = float(np.max(np.abs(result.angular_velocity)))
    angle_score = max(0.0, 1.0 - max_angle / np.pi)
    omega_score = max(0.0, 1.0 - max_omega / 15.0)
    return 0.6 * angle_score + 0.4 * omega_score


def amplification_metric(spectral) -> float:
    """A: reward selective amplification in the target frequency band."""
    # Log-scaled ratio avoids extreme values dominating evolution.
    ratio = spectral.amplification_ratio
    return float(np.clip(np.log1p(ratio), 0.0, 3.0))


def noise_metric(spectral) -> float:
    """N: penalty for broadband / high-frequency energy (noise band)."""
    return float(np.sqrt(spectral.noise_band_power + 1e-9))


def effort_metric(result: SimulationResult) -> float:
    """E: penalty for excessive NN and total control activity."""
    nn_rms = float(np.sqrt(np.mean(result.nn_output ** 2)))
    torque_rms = float(
    np.sqrt(np.mean(result.actual_torque ** 2))
    )
    return 0.5 * nn_rms + 0.5 * torque_rms


def unsafe_metric(result: SimulationResult, max_wheel_speed: float) -> float:
    """
    U: penalty for instability indicators (saturation, divergence).

    Large angles, wheel speed saturation, and angle variance contribute.
    """
    max_angle = float(np.max(np.abs(result.angle)))
    sat_fraction = float(np.mean(np.abs(result.wheel_velocity) > 0.95 * max_wheel_speed))
    variance = float(np.var(result.angle))
    penalty = 0.0
    if max_angle > np.pi * 0.8:
        penalty += (max_angle - np.pi * 0.8) * 2.0
    penalty += sat_fraction * 2.0
    penalty += min(variance, 5.0) * 0.2
    return penalty


def evaluate_rollout(
    result: SimulationResult,
    weights: FitnessWeights,
    sample_rate_hz: float,
    target_band_hz: list,
    noise_band_hz: list,
    max_wheel_speed: float,
) -> FitnessBreakdown:
    """
    Compute F = w_s*S + w_a*A - w_n*N - w_e*E - w_u*U for one simulation.
    """
    spectral = compute_spectral_metrics(
        result.angle,
        result.seismic_input,
        sample_rate_hz,
        target_band_hz,
        noise_band_hz,
    )
    S = stability_metric(result)
    A = amplification_metric(spectral)
    N = noise_metric(spectral)
    E = effort_metric(result)
    U = unsafe_metric(result, max_wheel_speed)

    total = (
        weights.stability * S
        + weights.amplification * A
        - weights.noise * N
        - weights.effort * E
        - weights.unsafe * U
    )
    return FitnessBreakdown(
        total=total,
        stability=S,
        amplification=A,
        noise=N,
        effort=E,
        unsafe=U,
    )


class FitnessEvaluator:
    """
    Evaluates NEAT genomes by running the pendulum environment.

    generation is updated by the training loop for optional weight schedules.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.env = PendulumEnv(config.simulation)
        self.generation = 0

    def evaluate_genome(self, genome: Genome) -> float:
        network = FeedforwardNetwork(genome)
        result = self.env.run_episode(network=network)
        weights = weights_for_generation(self.config, self.generation)
        sample_rate = 1.0 / self.config.simulation.dt
        breakdown = evaluate_rollout(
            result,
            weights,
            sample_rate,
            self.config.spectral.target_band_hz,
            self.config.spectral.noise_band_hz,
            self.config.simulation.max_wheel_speed,
        )
        return breakdown.total

    def make_fitness_fn(self) -> Callable[[Genome], float]:
        return self.evaluate_genome
