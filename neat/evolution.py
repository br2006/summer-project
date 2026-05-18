"""
Evolution engine: population management, reproduction, and generation stepping.

This module orchestrates the NEAT loop:
  evaluate -> speciate -> share fitness -> select parents -> reproduce -> repeat
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from neat.genome import Genome, InnovationTracker
from neat.mutations import crossover, mutate_genome
from neat.species import SpeciesSet


@dataclass
class EvolutionConfig:
    """Hyperparameters for the custom NEAT evolution engine."""

    population_size: int = 50
    num_inputs: int = 4
    num_outputs: int = 1
    elitism: int = 2
    survival_threshold: float = 0.2
    compatibility_threshold: float = 3.0
    max_stagnation: int = 15
    weight_mutate_rate: float = 0.8
    add_conn_rate: float = 0.05
    add_node_rate: float = 0.03
    toggle_rate: float = 0.01


class EvolutionEngine:
    """
    Runs generational NEAT evolution using the custom genome representation.

    fitness_fn: callable(genome) -> float
        Evaluates one genome in the pendulum simulation (or any task).
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn: Callable[[Genome], float],
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.tracker = InnovationTracker()
        self.species_set = SpeciesSet(
            compatibility_threshold=config.compatibility_threshold,
            max_stagnation=config.max_stagnation,
        )
        if seed is not None:
            random.seed(seed)
        self.population: List[Genome] = self._create_initial_population()
        self.generation = 0
        self.history_best: List[float] = []
        self.history_mean: List[float] = []
        self.history_species: List[int] = []
        self._best_genome: Optional[Genome] = None
        self._best_fitness: float = float("-inf")

    def _create_initial_population(self) -> List[Genome]:
        pop = []
        for _ in range(self.config.population_size):
            g = Genome.create_minimal(
                self.config.num_inputs,
                self.config.num_outputs,
                self.tracker,
            )
            pop.append(g)
        return pop

    def _evaluate_population(self) -> None:
        for genome in self.population:
            genome.fitness = self.fitness_fn(genome)

    def _reproduce(self) -> List[Genome]:
        """Select parents proportional to adjusted fitness and create offspring."""
        cfg = self.config
        adjusted = [max(g.adjusted_fitness, 0.0) for g in self.population]
        total = sum(adjusted) or 1.0

        # Elitism: copy best genomes unchanged.
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        new_pop: List[Genome] = [g.copy() for g in sorted_pop[: cfg.elitism]]

        def pick_parent() -> Genome:
            r = random.uniform(0, total)
            cumulative = 0.0
            for g, a in zip(self.population, adjusted):
                cumulative += a
                if cumulative >= r:
                    return g
            return self.population[-1]

        while len(new_pop) < cfg.population_size:
            p1 = pick_parent()
            p2 = pick_parent()
            child = crossover(p1, p2)
            mutate_genome(
                child,
                self.tracker,
                weight_mutate_rate=cfg.weight_mutate_rate,
                add_conn_rate=cfg.add_conn_rate,
                add_node_rate=cfg.add_node_rate,
                toggle_rate=cfg.toggle_rate,
            )
            new_pop.append(child)
        return new_pop

    def run_generation(self) -> Tuple[float, float, int]:
        """
        Execute one generation; return (best_fitness, mean_fitness, num_species).
        """
        self._evaluate_population()
        self.species_set.speciate(self.population)
        self.species_set.adjust_fitness(self.population)

        fitnesses = [g.fitness for g in self.population]
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        num_species = len(self.species_set.species)

        if best > self._best_fitness:
            self._best_fitness = best
            self._best_genome = max(self.population, key=lambda g: g.fitness).copy()

        self.history_best.append(best)
        self.history_mean.append(mean)
        self.history_species.append(num_species)

        self.population = self._reproduce()
        self.generation += 1
        return best, mean, num_species

    def get_best_genome(self) -> Genome:
        if self._best_genome is not None:
            return self._best_genome
        return max(self.population, key=lambda g: g.fitness)
