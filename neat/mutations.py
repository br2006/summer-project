"""
Mutation and crossover operators for NEAT genomes.

Mutation gradually complexifies topology (add node / add connection) and explores
weight space. Crossover combines two parents while preserving alignment via
innovation numbers.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from neat.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeType


def mutate_weight(
    genome: Genome,
    perturb_rate: float = 0.9,
    perturb_power: float = 0.5,
    replace_rate: float = 0.1,
    max_weight: float = 30.0,
) -> None:
    """Perturb or fully replace connection weights (Stanley & Miikkulainen style)."""
    for conn in genome.connections.values():
        if random.random() < perturb_rate:
            conn.weight += random.gauss(0.0, perturb_power)
        elif random.random() < replace_rate:
            conn.weight = random.uniform(-max_weight, max_weight)
        conn.weight = max(-max_weight, min(max_weight, conn.weight))


def _feedforward_valid(genome: Genome, in_node: int, out_node: int) -> bool:
    """
    Reject connections that would create a cycle in a feedforward network.

    NEAT feedforward networks require that every hidden/output node only receives
    inputs from nodes with lower IDs (when using the standard incremental ID scheme).
    """
    if in_node == out_node:
        return False
    # Quick check: in feedforward NEAT, in_node must be "before" out_node in topology.
    # Hidden nodes inserted by split have IDs between parent in/out.
    if in_node >= out_node:
        return False
    return True


def mutate_add_connection(
    genome: Genome,
    tracker: InnovationTracker,
    max_attempts: int = 20,
) -> bool:
    """Try to add a new enabled connection between unconnected nodes."""
    nodes = list(genome.nodes.keys())
    random.shuffle(nodes)
    for _ in range(max_attempts):
        in_node, out_node = random.sample(nodes, 2)
        if (in_node, out_node) in genome.connections:
            continue
        if not _feedforward_valid(genome, in_node, out_node):
            continue
        innov = tracker.get_connection_innovation(in_node, out_node)
        genome.connections[(in_node, out_node)] = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=random.gauss(0.0, 1.0),
            enabled=True,
            innovation=innov,
        )
        return True
    return False


def mutate_add_node(genome: Genome, tracker: InnovationTracker) -> bool:
    """
    Split an enabled connection: in -> hidden -> out.

    The original connection is disabled but kept for historical alignment.
    """
    enabled = [c for c in genome.connections.values() if c.enabled]
    if not enabled:
        return False
    conn = random.choice(enabled)
    conn.enabled = False

    # New hidden node ID is max existing + 1 (standard NEAT incremental scheme).
    new_id = max(genome.nodes.keys()) + 1
    genome.nodes[new_id] = NodeGene(id=new_id, node_type=NodeType.HIDDEN, bias=0.0)

    innov_split = tracker.get_node_split_innovation(conn.innovation)

    in_to_hidden = tracker.get_connection_innovation(conn.in_node, new_id)
    hidden_to_out = tracker.get_connection_innovation(new_id, conn.out_node)

    genome.connections[(conn.in_node, new_id)] = ConnectionGene(
        in_node=conn.in_node,
        out_node=new_id,
        weight=1.0,
        enabled=True,
        innovation=in_to_hidden,
    )
    genome.connections[(new_id, conn.out_node)] = ConnectionGene(
        in_node=new_id,
        out_node=conn.out_node,
        weight=conn.weight,
        enabled=True,
        innovation=hidden_to_out,
    )
    return True


def mutate_toggle_connection(genome: Genome) -> None:
    """Randomly enable or disable one connection (historical marking)."""
    if not genome.connections:
        return
    conn = random.choice(list(genome.connections.values()))
    conn.enabled = not conn.enabled


def mutate_genome(
    genome: Genome,
    tracker: InnovationTracker,
    weight_mutate_rate: float = 0.8,
    add_conn_rate: float = 0.05,
    add_node_rate: float = 0.03,
    toggle_rate: float = 0.01,
) -> None:
    """Apply all mutation operators with configurable probabilities."""
    if random.random() < weight_mutate_rate:
        mutate_weight(genome)
    if random.random() < add_conn_rate:
        mutate_add_connection(genome, tracker)
    if random.random() < add_node_rate:
        mutate_add_node(genome, tracker)
    if random.random() < toggle_rate:
        mutate_toggle_connection(genome)


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """
    Produce offspring from two parents; fitter parent is parent1 if tie-break by fitness.

    Matching genes (same innovation) are inherited randomly.
    Disjoint and excess genes come only from the fitter parent.
    """
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1
    elif parent2.fitness == parent1.fitness and random.random() > 0.5:
        parent1, parent2 = parent2, parent1

    child = Genome()
    # Union of all nodes from both parents (matching + fitter-only).
    innovations1 = {c.innovation: c for c in parent1.connections.values()}
    innovations2 = {c.innovation: c for c in parent2.connections.values()}
    all_innovations = set(innovations1) | set(innovations2)

    for innov in sorted(all_innovations):
        g1 = innov in innovations1
        g2 = innov in innovations2
        if g1 and g2:
            chosen = random.choice([innovations1[innov], innovations2[innov]])
            child.connections[(chosen.in_node, chosen.out_node)] = ConnectionGene(
                in_node=chosen.in_node,
                out_node=chosen.out_node,
                weight=chosen.weight,
                enabled=chosen.enabled,
                innovation=chosen.innovation,
            )
        elif g1:
            c = innovations1[innov]
            child.connections[(c.in_node, c.out_node)] = ConnectionGene(
                in_node=c.in_node,
                out_node=c.out_node,
                weight=c.weight,
                enabled=c.enabled,
                innovation=c.innovation,
            )

    # Inherit nodes referenced by child connections plus I/O nodes from fitter parent.
    for node in parent1.nodes.values():
        if node.node_type in (NodeType.INPUT, NodeType.BIAS, NodeType.OUTPUT):
            child.nodes[node.id] = NodeGene(
                id=node.id,
                node_type=node.node_type,
                bias=node.bias,
                activation=node.activation,
            )
    for conn in child.connections.values():
        for nid in (conn.in_node, conn.out_node):
            if nid not in child.nodes and nid in parent1.nodes:
                n = parent1.nodes[nid]
                child.nodes[nid] = NodeGene(
                    id=n.id,
                    node_type=n.node_type,
                    bias=n.bias,
                    activation=n.activation,
                )
    return child
