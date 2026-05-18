"""
Network topology visualization for NEAT genomes.

Draws nodes and enabled connections to help interpret evolved structures.
Future: export to Graphviz; recurrent edges would need different layout rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from neat.genome import Genome, NodeType


def draw_network_topology(
    genome: Genome,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Simple layered layout: inputs left, outputs right, hidden in between.
    """
    inputs = sorted(n.id for n in genome.nodes.values() if n.node_type == NodeType.INPUT)
    outputs = sorted(n.id for n in genome.nodes.values() if n.node_type == NodeType.OUTPUT)
    hidden = sorted(n.id for n in genome.nodes.values() if n.node_type == NodeType.HIDDEN)
    bias = [n.id for n in genome.nodes.values() if n.node_type == NodeType.BIAS]

    positions = {}
    x_in, x_out = 0.0, 3.0
    for i, nid in enumerate(inputs):
        positions[nid] = (x_in, i - len(inputs) / 2)
    for i, nid in enumerate(bias):
        positions[nid] = (x_in, len(inputs) + i)
    for i, nid in enumerate(outputs):
        positions[nid] = (x_out, i - len(outputs) / 2)
    x_h = 1.5
    for i, nid in enumerate(hidden):
        positions[nid] = (x_h, i - len(hidden) / 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    for conn in genome.get_enabled_connections():
        if conn.in_node not in positions or conn.out_node not in positions:
            continue
        x1, y1 = positions[conn.in_node]
        x2, y2 = positions[conn.out_node]
        color = "steelblue" if conn.weight >= 0 else "tomato"
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.5 + abs(conn.weight) * 0.1),
        )

    colors = {
        NodeType.INPUT: "lightgreen",
        NodeType.OUTPUT: "salmon",
        NodeType.HIDDEN: "lightgray",
        NodeType.BIAS: "gold",
    }
    for nid, (x, y) in positions.items():
        node = genome.nodes[nid]
        ax.scatter(x, y, s=200, c=colors.get(node.node_type, "white"), edgecolors="black", zorder=5)
        ax.text(x, y, str(nid), ha="center", va="center", fontsize=8)

    ax.set_title("NEAT Network Topology (enabled connections)")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
