"""
Genome structures for the custom NEAT implementation.

A genome encodes a neural network as:
  - NodeGene records (inputs, hidden nodes, outputs)
  - ConnectionGene records (directed weighted edges)

Innovation numbers allow historical alignment during crossover, as described in
the original NEAT paper. Without innovation tracking, crossover cannot reliably
match homologous genes between parents.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    """Semantic role of a node in the network."""

    INPUT = auto()
    BIAS = auto()
    HIDDEN = auto()
    OUTPUT = auto()


@dataclass
class NodeGene:
    """
    One node in the phenotype graph.

    Inputs and outputs are created at genome initialization.
    Hidden nodes appear when connections are split during add-node mutation.
    """

    id: int
    node_type: NodeType
    bias: float = 0.0
    # Activation is fixed to tanh in this prototype; kept for future extensions.
    activation: str = "tanh"


@dataclass
class ConnectionGene:
    """
    One directed connection between two nodes.

    enabled=False corresponds to historical marking in NEAT: the gene remains
    in the genome for alignment but does not contribute to the phenotype.
    """

    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int


class InnovationTracker:
    """
    Global innovation counter shared across an entire population.

    When the same structural mutation appears in multiple genomes in the same
    generation, it must receive the same innovation number so crossover can
    treat those genes as homologous.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        # Structural innovations keyed by (in_node, out_node) for new connections.
        self._connection_innovations: Dict[Tuple[int, int], int] = {}
        # Node-split innovations keyed by the connection being split.
        self._node_split_innovations: Dict[int, int] = {}

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._connection_innovations:
            self._counter += 1
            self._connection_innovations[key] = self._counter
        return self._connection_innovations[key]

    def get_node_split_innovation(self, connection_innovation: int) -> int:
        if connection_innovation not in self._node_split_innovations:
            self._counter += 1
            self._node_split_innovations[connection_innovation] = self._counter
        return self._node_split_innovations[connection_innovation]

    def register_genome(self, genome: "Genome") -> None:
        """
        Synchronize tracker state with an externally loaded/evolved genome.

        This prevents innovation collisions when warm-starting a new run from a
        previously trained genome.
        """
        max_innovation = self._counter
        for conn in genome.connections.values():
            max_innovation = max(max_innovation, int(conn.innovation))
            self._connection_innovations[(conn.in_node, conn.out_node)] = int(conn.innovation)
        self._counter = max(self._counter, max_innovation)


@dataclass
class Genome:
    """
    Complete genotype for one individual in the population.

    fitness: raw evaluation score from the environment
    adjusted_fitness: fitness after species sharing (computed by EvolutionEngine)
  """

    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[Tuple[int, int], ConnectionGene] = field(default_factory=dict)
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    key: Optional[int] = None  # species id assigned during speciation

    @classmethod
    def create_minimal(
        cls,
        num_inputs: int,
        num_outputs: int,
        tracker: InnovationTracker,
        input_ids: Optional[List[int]] = None,
        output_ids: Optional[List[int]] = None,
        bias_id: int = -1,
    ) -> "Genome":
        """
        Build the smallest feedforward topology: every input -> every output.

        This is the NEAT "starting genome" with no hidden nodes, matching the
        paper's principle of beginning search in the simplest possible structure.
        """
        genome = cls()
        if input_ids is None:
            input_ids = list(range(num_inputs))
        if output_ids is None:
            output_ids = list(range(num_inputs, num_inputs + num_outputs))

        # Bias node provides a constant 1.0 input (activation handled in network).
        genome.nodes[bias_id] = NodeGene(id=bias_id, node_type=NodeType.BIAS, bias=0.0)

        for nid in input_ids:
            genome.nodes[nid] = NodeGene(id=nid, node_type=NodeType.INPUT)
        for nid in output_ids:
            genome.nodes[nid] = NodeGene(id=nid, node_type=NodeType.OUTPUT)

        for i in input_ids + [bias_id]:
            for o in output_ids:
                innov = tracker.get_connection_innovation(i, o)
                genome.connections[(i, o)] = ConnectionGene(
                    in_node=i,
                    out_node=o,
                    weight=random.uniform(-1.5, 1.5),
                    enabled=True,
                    innovation=innov,
                )
        return genome

    def get_enabled_connections(self) -> List[ConnectionGene]:
        return [c for c in self.connections.values() if c.enabled]

    def copy(self) -> "Genome":
        return copy.deepcopy(self)
