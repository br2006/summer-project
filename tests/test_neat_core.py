"""Smoke tests for custom NEAT core (genome, network, evolution step)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from neat.evolution import EvolutionConfig, EvolutionEngine
from neat.genome import Genome, InnovationTracker
from neat.network import FeedforwardNetwork


def test_minimal_genome_forward():
    tracker = InnovationTracker()
    g = Genome.create_minimal(4, 1, tracker)
    net = FeedforwardNetwork(g)
    out = net.activate(np.array([0.1, 0.0, 0.2, -0.1]))
    assert out.shape == (1,)
    assert -1.0 <= out[0] <= 1.0


def test_one_generation():
    def fitness_fn(genome):
        return 1.0

    engine = EvolutionEngine(
        EvolutionConfig(population_size=10),
        fitness_fn=fitness_fn,
        seed=0,
    )
    best, mean, n_sp = engine.run_generation()
    assert best == 1.0
    assert n_sp >= 1
