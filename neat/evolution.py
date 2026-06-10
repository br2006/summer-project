"""
Evolution engine: population management, reproduction, and generation stepping.

This module orchestrates the NEAT loop:
  evaluate -> speciate -> share fitness -> select parents -> reproduce -> repeat
"""

from __future__ import annotations

import random
from math import ceil
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
    elitism: int = 1
    survival_threshold: float = 0.35
    compatibility_threshold: float = 1.2
    max_stagnation: int = 15
    weight_mutate_rate: float = 0.8
    add_conn_rate: float = 0.12
    add_node_rate: float = 0.07
    toggle_rate: float = 0.02
    random_immigrant_rate: float = 0.05
    warm_start_genome: Optional[Genome] = None
    warm_start_fraction: float = 0.0
    warm_start_mutate: bool = True


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
        if config.warm_start_genome is not None:
            self.tracker.register_genome(config.warm_start_genome)
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
        cfg = self.config
        pop: List[Genome] = []

        if cfg.warm_start_genome is not None and cfg.warm_start_fraction > 0.0:
            seeded_fraction = max(0.0, min(1.0, float(cfg.warm_start_fraction)))
            n_seed = int(round(cfg.population_size * seeded_fraction))
            n_seed = max(0, min(cfg.population_size, n_seed))

            if n_seed > 0:
                # 1 exact elite copy from previous best genome.
                pop.append(cfg.warm_start_genome.copy())

            # Remaining seeded slice are mutated clones.
            for _ in range(1, n_seed):
                seeded = cfg.warm_start_genome.copy()
                if cfg.warm_start_mutate:
                    structural_scale = 0.5 if random.random() < 0.5 else 1.0
                    mutate_genome(
                        seeded,
                        self.tracker,
                        weight_mutate_rate=1.0,
                        add_conn_rate=cfg.add_conn_rate * structural_scale,
                        add_node_rate=cfg.add_node_rate * structural_scale,
                        toggle_rate=cfg.toggle_rate,
                    )
                pop.append(seeded)

        for _ in range(len(pop), cfg.population_size):
            g = Genome.create_minimal(
                cfg.num_inputs,
                cfg.num_outputs,
                self.tracker,
            )
            pop.append(g)
        return pop

    def _evaluate_population(self) -> None:
        for genome in self.population:
            genome.fitness = self.fitness_fn(genome)

    def _get_species_parent_pool(self) -> dict[int, List[Genome]]:
        """
        Build per-species parent pools using survival-threshold truncation.

        For each species, only the top ceil(survival_threshold * species_size)
        genomes by fitness are eligible as parents (minimum 1).
        """
        pools: dict[int, List[Genome]] = {}
        st = max(0.0, min(1.0, self.config.survival_threshold))

        for sid, species in self.species_set.species.items():
            if not species.members:
                continue
            ranked = sorted(species.members, key=lambda g: g.fitness, reverse=True)
            n_parents = max(1, ceil(st * len(ranked)))
            pools[sid] = ranked[:n_parents]
        return pools

    @staticmethod
    def _allocate_counts(shares: dict[int, float], total_count: int) -> dict[int, int]:
        """Allocate integer counts proportionally from non-negative shares."""
        if total_count <= 0 or not shares:
            return {sid: 0 for sid in shares}

        clamped = {sid: max(0.0, float(v)) for sid, v in shares.items()}
        total_share = sum(clamped.values())

        if total_share <= 0.0:
            # Uniform fallback across active species.
            base = total_count // len(clamped)
            rem = total_count % len(clamped)
            out = {sid: base for sid in clamped}
            for sid in sorted(clamped.keys())[:rem]:
                out[sid] += 1
            return out

        raw = {sid: (val / total_share) * total_count for sid, val in clamped.items()}
        out = {sid: int(raw[sid]) for sid in clamped}
        assigned = sum(out.values())
        remaining = total_count - assigned

        if remaining > 0:
            remainders = sorted(
                ((sid, raw[sid] - out[sid]) for sid in clamped),
                key=lambda x: x[1],
                reverse=True,
            )
            for sid, _ in remainders[:remaining]:
                out[sid] += 1
        elif remaining < 0:
            # Defensive: trim from smallest remainders first if over-assigned.
            remainders = sorted(
                ((sid, raw[sid] - out[sid]) for sid in clamped),
                key=lambda x: x[1],
            )
            need_trim = -remaining
            idx = 0
            while need_trim > 0 and idx < len(remainders):
                sid = remainders[idx][0]
                if out[sid] > 0:
                    out[sid] -= 1
                    need_trim -= 1
                else:
                    idx += 1

        return out

    @staticmethod
    def _weighted_pick(genomes: List[Genome]) -> Genome:
        """Pick one genome by non-negative adjusted-fitness weight with uniform fallback."""
        if len(genomes) == 1:
            return genomes[0]

        weights = [max(g.adjusted_fitness, 0.0) for g in genomes]
        total = sum(weights)
        if total <= 0.0:
            return random.choice(genomes)

        r = random.uniform(0.0, total)
        cumulative = 0.0
        for g, w in zip(genomes, weights):
            cumulative += w
            if cumulative >= r:
                return g
        return genomes[-1]

    def _reproduce(self) -> List[Genome]:
        """
        Reproduce with global elitism and species-aware offspring allocation.

        Policy:
        - Global elitism: top `elitism` genomes copied unchanged.
        - Species-aware allocation: remaining offspring allocated by per-species
          adjusted-fitness share (sum over members), with robust fallbacks.
        - Parent eligibility is truncated per species by `survival_threshold`.
        """
        cfg = self.config

        # Elitism: copy best genomes unchanged.
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elite_count = max(0, min(cfg.elitism, cfg.population_size))
        new_pop: List[Genome] = [g.copy() for g in sorted_pop[:elite_count]]

        remaining_slots = cfg.population_size - len(new_pop)
        if remaining_slots <= 0:
            return new_pop

        immigrant_rate = max(0.0, min(1.0, float(cfg.random_immigrant_rate)))
        immigrant_count = int(round(immigrant_rate * remaining_slots))
        immigrant_count = max(0, min(immigrant_count, remaining_slots))
        breeding_slots = remaining_slots - immigrant_count

        parent_pools = self._get_species_parent_pool()
        if not parent_pools:
            # Defensive fallback if all species were removed unexpectedly.
            parent_pools = {0: sorted_pop[: max(1, len(sorted_pop))]}

        # Species fitness share from sum of adjusted fitness of members.
        species_shares: dict[int, float] = {}
        for sid, parents in parent_pools.items():
            species = self.species_set.species.get(sid)
            members = species.members if species is not None else parents
            species_shares[sid] = sum(max(g.adjusted_fitness, 0.0) for g in members)

        offspring_by_species = self._allocate_counts(species_shares, breeding_slots)

        for sid in sorted(offspring_by_species.keys()):
            n_children = offspring_by_species.get(sid, 0)
            if n_children <= 0:
                continue
            parents = parent_pools.get(sid, [])
            if not parents:
                continue

            for _ in range(n_children):
                if len(new_pop) >= cfg.population_size:
                    break
                p1 = self._weighted_pick(parents)
                p2 = self._weighted_pick(parents)
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

        # If rounding/extinction left gaps, fill from best available parent pools.
        while len(new_pop) < cfg.population_size:
            if immigrant_count > 0:
                immigrant = Genome.create_minimal(
                    cfg.num_inputs,
                    cfg.num_outputs,
                    self.tracker,
                )
                mutate_genome(
                    immigrant,
                    self.tracker,
                    weight_mutate_rate=1.0,
                    add_conn_rate=cfg.add_conn_rate,
                    add_node_rate=cfg.add_node_rate,
                    toggle_rate=cfg.toggle_rate,
                )
                new_pop.append(immigrant)
                immigrant_count -= 1
                continue

            all_parents = [g for pool in parent_pools.values() for g in pool]
            if not all_parents:
                all_parents = sorted_pop
            p1 = self._weighted_pick(all_parents)
            p2 = self._weighted_pick(all_parents)
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

        if len(new_pop) > cfg.population_size:
            new_pop = new_pop[: cfg.population_size]

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
