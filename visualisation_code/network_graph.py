"""
Network topology visualization for NEAT genomes.

Draws nodes and enabled connections to help interpret evolved structures.
Future: export to Graphviz; recurrent edges would need different layout rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from neat.genome import Genome, NodeType


DEFAULT_INPUT_LABELS = (
    "angle",
    "angular velocity",
    "base acceleration",
    "wheel velocity",
)


NODE_COLORS = {
    NodeType.INPUT: "#B7E4C7",
    NodeType.OUTPUT: "#F4A7A3",
    NodeType.HIDDEN: "#D9D9D9",
    NodeType.BIAS: "#FFD166",
}


def _node_label(
    node_id: int,
    node_type: NodeType,
    input_label_lookup: Dict[int, str],
    presentation: bool,
) -> str:
    """Return a semantic label to place beside a topology node."""
    if node_type == NodeType.INPUT:
        return input_label_lookup.get(node_id, f"input {node_id}")
    elif node_type == NodeType.BIAS:
        return "bias"
    elif node_type == NodeType.OUTPUT:
        return "torque output" if presentation else "output"

    return "hidden"


def _add_topology_legend(ax) -> None:
    """Add a compact legend explaining node and connection encodings."""
    node_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor="black",
            label=label,
        )
        for label, color in (
            ("Input", NODE_COLORS[NodeType.INPUT]),
            ("Bias", NODE_COLORS[NodeType.BIAS]),
            ("Hidden", NODE_COLORS[NodeType.HIDDEN]),
            ("Output", NODE_COLORS[NodeType.OUTPUT]),
        )
    ]
    edge_handles = [
        Line2D([0], [0], color="steelblue", lw=2, label="Positive weight"),
        Line2D([0], [0], color="tomato", lw=2, label="Negative weight"),
    ]
    ax.legend(
        handles=node_handles + edge_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=9,
    )


def _compute_feedforward_depths(
    genome: Genome,
    input_and_bias_nodes: Sequence[int],
    hidden_nodes: Sequence[int],
) -> Dict[int, int]:
    """
    Estimate hidden-node depth from enabled feedforward connections.

    Depth 0 is reserved for inputs/bias. Hidden nodes are assigned to the
    longest upstream path found from an input or bias node. If a hidden node is
    disconnected from inputs, or if an unexpected cycle is encountered, it is
    placed in the first hidden layer so the plot remains robust.
    """
    input_and_bias = set(input_and_bias_nodes)
    hidden = set(hidden_nodes)
    incoming: Dict[int, list[int]] = {node_id: [] for node_id in hidden}

    for conn in genome.get_enabled_connections():
        if conn.out_node in hidden:
            incoming[conn.out_node].append(conn.in_node)

    depths: Dict[int, int] = {node_id: 0 for node_id in input_and_bias}
    visiting: set[int] = set()

    def depth(node_id: int) -> int:
        if node_id in depths:
            return depths[node_id]
        if node_id not in hidden or node_id in visiting:
            return 0

        visiting.add(node_id)
        parents = incoming.get(node_id, [])
        upstream_depths = [depth(parent) for parent in parents]
        visiting.remove(node_id)

        depths[node_id] = max(upstream_depths, default=0) + 1
        return depths[node_id]

    for node_id in hidden_nodes:
        depth(node_id)

    return depths


def draw_network_topology(
    genome: Genome,
    save_path: Optional[Path] = None,
    show: bool = True,
    input_labels: Optional[Sequence[str]] = None,
    presentation: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """
    Draw a readable layered topology plot for an evolved NEAT genome.

    Inputs are placed on the left, outputs on the right, and hidden nodes are
    grouped into intermediate columns based on estimated feedforward depth.
    Enabled connections are shown as lines; colour indicates sign and line
    width indicates relative weight magnitude.
    """
    inputs = sorted(n.id for n in genome.nodes.values() if n.node_type == NodeType.INPUT)
    outputs = sorted(n.id for n in genome.nodes.values() if n.node_type == NodeType.OUTPUT)
    hidden = sorted(n.id for n in genome.nodes.values() if n.node_type == NodeType.HIDDEN)
    bias = [n.id for n in genome.nodes.values() if n.node_type == NodeType.BIAS]

    labels = input_labels or DEFAULT_INPUT_LABELS
    input_label_lookup = {node_id: labels[i] for i, node_id in enumerate(inputs) if i < len(labels)}

    positions = {}
    x_in, x_out = 0.0, 3.0
    input_y_offset = (len(inputs) - 1) / 2 if inputs else 0.0
    for i, nid in enumerate(inputs):
        # Inputs are ordered bottom-to-top so displayed numbers increase upward.
        positions[nid] = (x_in, i - input_y_offset)
    for i, nid in enumerate(bias):
        # Bias is shown underneath the input stack and displayed as node 0.
        positions[nid] = (x_in, -input_y_offset - 1.0 - i)

    depths = _compute_feedforward_depths(genome, inputs + bias, hidden)
    hidden_layers: Dict[int, list[int]] = {}
    for nid in hidden:
        hidden_layers.setdefault(max(1, depths.get(nid, 1)), []).append(nid)

    max_hidden_depth = max(hidden_layers, default=0)
    output_depth = max_hidden_depth + 1
    x_spacing = (x_out - x_in) / output_depth

    for layer_depth, layer_nodes in hidden_layers.items():
        layer_nodes = sorted(layer_nodes)
        y_offset = (len(layer_nodes) - 1) / 2 if layer_nodes else 0.0
        for i, nid in enumerate(layer_nodes):
            positions[nid] = (x_in + layer_depth * x_spacing, i - y_offset)

    for i, nid in enumerate(outputs):
        positions[nid] = (x_out, i - len(outputs) / 2)

    display_ids = {}
    next_display_id = 1
    for nid in sorted(bias, key=lambda n: positions[n][1]):
        display_ids[nid] = 0
    for group in (inputs, hidden, outputs):
        for nid in sorted(group, key=lambda n: positions[n][1]):
            display_ids[nid] = next_display_id
            next_display_id += 1

    enabled_connections = list(genome.get_enabled_connections())
    max_abs_weight = max((abs(conn.weight) for conn in enabled_connections), default=1.0)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for conn in enabled_connections:
        if conn.in_node not in positions or conn.out_node not in positions:
            continue
        x1, y1 = positions[conn.in_node]
        x2, y2 = positions[conn.out_node]
        color = "steelblue" if conn.weight >= 0 else "tomato"
        linewidth = 0.8 + 2.4 * (abs(conn.weight) / max_abs_weight)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            alpha=0.7,
            zorder=1,
        )

    for nid, (x, y) in positions.items():
        node = genome.nodes[nid]
        ax.scatter(
            x,
            y,
            s=900 if presentation else 650,
            c=NODE_COLORS.get(node.node_type, "white"),
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            x,
            y,
            str(display_ids.get(nid, nid)),
            ha="center",
            va="center",
            fontsize=9 if presentation else 8,
            fontweight="bold",
            zorder=6,
        )
        label = _node_label(nid, node.node_type, input_label_lookup, presentation)
        label_x = x - 0.18 if node.node_type == NodeType.OUTPUT else x + 0.18
        label_ha = "right" if node.node_type == NodeType.OUTPUT else "left"
        ax.text(
            label_x,
            y,
            label,
            ha=label_ha,
            va="center",
            fontsize=9,
            color="0.25",
            zorder=6,
        )

    main_title = title or "Evolved NEAT Controller Topology"
    sub_title = subtitle or f"{len(genome.nodes)} nodes, {len(enabled_connections)} enabled connections"
    ax.set_title(
        f"{main_title}\n{sub_title}",
        fontsize=13,
        pad=14,
    )
    _add_topology_legend(ax)
    ax.axis("off")
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
