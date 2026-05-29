"""
Project configuration: fitness weights, spectral targets, normalization.

YAML files in configs/ override defaults. Generation-dependent fitness schedules
allow early emphasis on stability and later emphasis on spectral shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from simulation.pendulum_env import PendulumEnvConfig


@dataclass
class FitnessWeights:
    """
    F = w_s*S + w_a*A - w_n*N - w_e*E - w_u*U

    Each weight scales its corresponding metric (see neat/fitness.py).
    """

    stability: float = 1.0
    amplification: float = 0.8
    noise: float = 0.5
    effort: float = 0.3
    unsafe: float = 1.5


@dataclass
class SpectralTargets:
    """Frequency band (Hz) where amplification is desired during fitness evaluation."""

    target_band_hz: List[float] = field(default_factory=lambda: [0.4, 1.5])
    noise_band_hz: List[float] = field(default_factory=lambda: [3.0, 8.0])


@dataclass
class FitnessSchedule:
    """
    Smooth generation-dependent scheduling for stability/amplification weights.

    Logistic sigmoid:
        sigmoid(g) = 1 / (1 + exp(-k * (g - midpoint)))

    Amplification schedule:
        w_amp(g) = amp_min + (amp_max - amp_min) * sigmoid(g)

    Stability schedule:
        w_stability(g) = stab_max - (stab_max - stab_min) * sigmoid(g)

    Design intent:
    - Early generations emphasize stability.
    - Mid generations balance stability and amplification.
    - Late generations emphasize amplification while preserving stability reward.
    """

    max_generations: int = 500
    midpoint: float = 250.0
    k: float = 0.025
    stab_max: float = 3.0
    stab_min: float = 0.8
    amp_min: float = 0.5
    amp_max: float = 3.0


@dataclass
class ScheduledFitnessWeights:
    """Container for generation-dependent weight diagnostics."""

    weights: FitnessWeights
    sigmoid_output: float
    generation: int


@dataclass
class ProjectConfig:
    simulation: PendulumEnvConfig = field(default_factory=PendulumEnvConfig)
    fitness: FitnessWeights = field(default_factory=FitnessWeights)
    spectral: SpectralTargets = field(default_factory=SpectralTargets)
    schedule: FitnessSchedule = field(default_factory=FitnessSchedule)
    population_size: int = 40
    generations: int = 30
    seed: int = 42


def logistic_sigmoid(generation: int, midpoint: float, k: float) -> float:
    """Standard logistic sigmoid used for smooth fitness-weight transitions."""
    return 1.0 / (1.0 + exp(-k * (float(generation) - midpoint)))


def scheduled_weights_for_generation(
    config: ProjectConfig,
    generation: int,
) -> ScheduledFitnessWeights:
    """
    Compute generation-adjusted fitness weights and expose scheduling diagnostics.

    Only stability and amplification are scheduled; all other fitness terms keep
    their configured constant weights.
    """
    base = config.fitness
    sched = config.schedule
    sig = logistic_sigmoid(generation, sched.midpoint, sched.k)

    scheduled = FitnessWeights(
        stability=sched.stab_max - (sched.stab_max - sched.stab_min) * sig,
        amplification=sched.amp_min + (sched.amp_max - sched.amp_min) * sig,
        noise=base.noise,
        effort=base.effort,
        unsafe=base.unsafe,
    )
    return ScheduledFitnessWeights(
        weights=scheduled,
        sigmoid_output=sig,
        generation=generation,
    )


def load_project_config(path: Optional[Path] = None) -> ProjectConfig:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "project_config.yaml"
    if not path.exists():
        return ProjectConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = ProjectConfig()
    if "simulation" in data:
        for k, v in data["simulation"].items():
            if hasattr(cfg.simulation, k):
                setattr(cfg.simulation, k, v)
    if "fitness" in data:
        for k, v in data["fitness"].items():
            if hasattr(cfg.fitness, k):
                setattr(cfg.fitness, k, v)
    if "spectral" in data:
        for k, v in data["spectral"].items():
            if hasattr(cfg.spectral, k):
                setattr(cfg.spectral, k, v)
    if "schedule" in data:
        for k, v in data["schedule"].items():
            if hasattr(cfg.schedule, k):
                setattr(cfg.schedule, k, v)
    if "evolution" in data:
        ev = data["evolution"]
        cfg.population_size = ev.get("population_size", cfg.population_size)
        cfg.generations = ev.get("generations", cfg.generations)
        cfg.seed = ev.get("seed", cfg.seed)
    return cfg


def weights_for_generation(config: ProjectConfig, generation: int) -> FitnessWeights:
    return scheduled_weights_for_generation(config, generation).weights
