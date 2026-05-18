"""
Project configuration: fitness weights, spectral targets, normalization.

YAML files in configs/ override defaults. Generation-dependent fitness schedules
allow early emphasis on stability and later emphasis on spectral shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    Optional per-generation weight multipliers.

    Example: increase amplification weight after generation 20.
    """

    generation_thresholds: List[int] = field(default_factory=lambda: [0, 20, 50])
    stability_multipliers: List[float] = field(default_factory=lambda: [1.2, 1.0, 0.9])
    amplification_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.2])


@dataclass
class ProjectConfig:
    simulation: PendulumEnvConfig = field(default_factory=PendulumEnvConfig)
    fitness: FitnessWeights = field(default_factory=FitnessWeights)
    spectral: SpectralTargets = field(default_factory=SpectralTargets)
    schedule: FitnessSchedule = field(default_factory=FitnessSchedule)
    population_size: int = 40
    generations: int = 30
    seed: int = 42


def _apply_schedule(weights: FitnessWeights, schedule: FitnessSchedule, generation: int) -> FitnessWeights:
    """Pick multipliers for the current generation bracket."""
    idx = 0
    for i, thresh in enumerate(schedule.generation_thresholds):
        if generation >= thresh:
            idx = i
    w_s = schedule.stability_multipliers[min(idx, len(schedule.stability_multipliers) - 1)]
    w_a = schedule.amplification_multipliers[min(idx, len(schedule.amplification_multipliers) - 1)]
    return FitnessWeights(
        stability=weights.stability * w_s,
        amplification=weights.amplification * w_a,
        noise=weights.noise,
        effort=weights.effort,
        unsafe=weights.unsafe,
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
    if "evolution" in data:
        ev = data["evolution"]
        cfg.population_size = ev.get("population_size", cfg.population_size)
        cfg.generations = ev.get("generations", cfg.generations)
        cfg.seed = ev.get("seed", cfg.seed)
    return cfg


def weights_for_generation(config: ProjectConfig, generation: int) -> FitnessWeights:
    return _apply_schedule(config.fitness, config.schedule, generation)
