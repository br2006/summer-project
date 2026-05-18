"""
NEAT package: educational implementation of NeuroEvolution of Augmenting Topologies.

This package mirrors core concepts from Stanley & Miikkulainen (2002) and can be
used standalone for learning. Production training may also use neat-python via
training/train.py while reusing fitness and simulation modules from this project.
"""

from neat.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeType
from neat.network import FeedforwardNetwork
from neat.evolution import EvolutionConfig, EvolutionEngine

__all__ = [
    "ConnectionGene",
    "Genome",
    "InnovationTracker",
    "NodeGene",
    "NodeType",
    "FeedforwardNetwork",
    "EvolutionConfig",
    "EvolutionEngine",
]
