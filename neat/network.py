"""
Feedforward neural network phenotype built from a NEAT genome.

Data flow during control:
  normalized sensor vector -> input nodes -> hidden (if any) -> output nodes -> tanh
The output is in approximately [-1, 1] and must be scaled externally to torque.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from neat.genome import ConnectionGene, Genome, NodeType


def _tanh(x: float) -> float:
    return math.tanh(x)


class FeedforwardNetwork:
    """
    Evaluates a genome as a feedforward network with tanh activation.

    Execution order is determined by a topological sort of enabled connections,
    ensuring each node's inputs are computed before the node itself.
    """

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self._node_order: List[int] = self._build_evaluation_order()

    def _build_evaluation_order(self) -> List[int]:
        """Topological sort: inputs/bias first, then hidden/output by dependency."""
        enabled = self.genome.get_enabled_connections()
        incoming: Dict[int, List[int]] = defaultdict(list)
        for c in enabled:
            incoming[c.out_node].append(c.in_node)

        # Nodes that need computation (non-input, non-bias).
        to_sort = [
            n.id
            for n in self.genome.nodes.values()
            if n.node_type in (NodeType.HIDDEN, NodeType.OUTPUT)
        ]
        order: List[int] = []
        while to_sort:
            progress = False
            for nid in list(to_sort):
                preds = incoming.get(nid, [])
                if all(p not in to_sort for p in preds):
                    order.append(nid)
                    to_sort.remove(nid)
                    progress = True
            if not progress and to_sort:
                # Cycle or invalid graph; fall back to ID order for robustness.
                order.extend(sorted(to_sort))
                break
        return order

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run one forward pass.

        Parameters
        ----------
        inputs : ndarray shape (num_inputs,)
            Normalized values in approximately [-1, 1].

        Returns
        -------
        outputs : ndarray shape (num_outputs,)
            Normalized corrective torque signal in approximately [-1, 1].
        """
        values: Dict[int, float] = {}
        input_ids = sorted(
            n.id for n in self.genome.nodes.values() if n.node_type == NodeType.INPUT
        )
        output_ids = sorted(
            n.id for n in self.genome.nodes.values() if n.node_type == NodeType.OUTPUT
        )

        for i, nid in enumerate(input_ids):
            values[nid] = float(inputs[i]) if i < len(inputs) else 0.0

        for n in self.genome.nodes.values():
            if n.node_type == NodeType.BIAS:
                values[n.id] = 1.0

        enabled = self.genome.get_enabled_connections()
        by_out: Dict[int, List[ConnectionGene]] = defaultdict(list)
        for c in enabled:
            by_out[c.out_node].append(c)

        for nid in self._node_order:
            node = self.genome.nodes[nid]
            total = node.bias
            for conn in by_out.get(nid, []):
                total += values.get(conn.in_node, 0.0) * conn.weight
            values[nid] = _tanh(total)

        return np.array([values[oid] for oid in output_ids], dtype=np.float64)
