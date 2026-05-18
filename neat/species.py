"""
Species management for NEAT speciation.

Speciation protects innovation by allowing structurally different networks to
optimize in separate niches before competing globally. Compatibility distance
measures structural and weight similarity between genomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from neat.genome import ConnectionGene, Genome


def compatibility_distance(
    g1: Genome,
    g2: Genome,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> float:
    """
    δ = (c1 * E / N) + (c2 * D / N) + c3 * W

    E = excess genes, D = disjoint genes, W = average weight difference of matching genes.
    N = size of larger genome (number of connections), minimum 1.
    """
    conns1 = {c.innovation: c for c in g1.connections.values()}
    conns2 = {c.innovation: c for c in g2.connections.values()}
    innovations1 = set(conns1.keys())
    innovations2 = set(conns2.keys())
    matching = innovations1 & innovations2

    max_innov1 = max(innovations1) if innovations1 else 0
    max_innov2 = max(innovations2) if innovations2 else 0
    min_innov1 = min(innovations1) if innovations1 else 0
    min_innov2 = min(innovations2) if innovations2 else 0

    excess = 0
    if max_innov1 > max_innov2:
        excess = sum(1 for i in innovations1 if i > max_innov2)
    elif max_innov2 > max_innov1:
        excess = sum(1 for i in innovations2 if i > max_innov1)

    disjoint = 0
    for i in innovations1 - matching:
        if i > min_innov2:
            disjoint += 1
    for i in innovations2 - matching:
        if i > min_innov1:
            disjoint += 1

    weight_diff = 0.0
    if matching:
        weight_diff = sum(
            abs(conns1[i].weight - conns2[i].weight) for i in matching
        ) / len(matching)

    n = max(len(conns1), len(conns2))
    n = max(n, 1)
    return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * weight_diff)


@dataclass
class Species:
    """
    A species is a cluster of similar genomes sharing a representative genome.

    stagnation: generations without improvement; used for extinction.
    """

    id: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0

    def reset_members(self) -> None:
        self.members = []


class SpeciesSet:
    """
    Assigns genomes to species and tracks stagnation / extinction.
    """

    def __init__(
        self,
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        max_stagnation: int = 15,
    ) -> None:
        self.compatibility_threshold = compatibility_threshold
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.max_stagnation = max_stagnation
        self.species: Dict[int, Species] = {}
        self._next_species_id = 0

    def speciate(self, population: List[Genome]) -> None:
        """Assign each genome to the first compatible species or create a new one."""
        for sp in self.species.values():
            sp.reset_members()

        for genome in population:
            placed = False
            for sp in self.species.values():
                dist = compatibility_distance(
                    genome,
                    sp.representative,
                    self.c1,
                    self.c2,
                    self.c3,
                )
                if dist < self.compatibility_threshold:
                    sp.members.append(genome)
                    genome.key = sp.id
                    placed = True
                    break
            if not placed:
                sid = self._next_species_id
                self._next_species_id += 1
                self.species[sid] = Species(
                    id=sid,
                    representative=genome.copy(),
                    members=[genome],
                )
                genome.key = sid

        # Update representatives and stagnation counters.
        extinct_ids: List[int] = []
        for sid, sp in list(self.species.items()):
            if not sp.members:
                extinct_ids.append(sid)
                continue
            # Representative = median member by fitness (simple robust choice).
            sp.members.sort(key=lambda g: g.fitness, reverse=True)
            sp.representative = sp.members[len(sp.members) // 2].copy()
            best = sp.members[0].fitness
            if best > sp.best_fitness:
                sp.best_fitness = best
                sp.stagnation = 0
            else:
                sp.stagnation += 1
            if sp.stagnation >= self.max_stagnation and len(self.species) > 1:
                extinct_ids.append(sid)

        for sid in extinct_ids:
            del self.species[sid]

    def adjust_fitness(self, population: List[Genome]) -> None:
        """
        Fitness sharing: divide raw fitness by species size.

        Prevents a single large species from dominating selection pressure.
        """
        sizes: Dict[int, int] = {}
        for g in population:
            if g.key is not None:
                sizes[g.key] = sizes.get(g.key, 0) + 1
        for g in population:
            if g.key is not None and sizes.get(g.key, 0) > 0:
                g.adjusted_fitness = g.fitness / sizes[g.key]
            else:
                g.adjusted_fitness = g.fitness

    def get_species_sizes(self) -> Dict[int, int]:
        return {sid: len(sp.members) for sid, sp in self.species.items()}
