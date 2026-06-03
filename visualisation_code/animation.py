"""Matplotlib animation for the reaction-wheel pendulum demo."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from config.settings import load_project_config
from neat.network import FeedforwardNetwork
from simulation.pendulum_env import PendulumEnv

from .output import get_output_dir
from .plots import plot_rollout

DEFAULT_FRAME_STRIDE = 2


def load_controller(path: Path) -> FeedforwardNetwork:
    with path.open("rb") as f:
        genome = pickle.load(f)
    return FeedforwardNetwork(genome)


def run_demo(
    genome_path: Path = Path("best_genome.pkl"),
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    show: bool = True,
    config_path: Optional[Path] = None,
) -> FuncAnimation:
    """Run the pendulum simulation and render an animation using project config."""
    project_config = load_project_config(config_path)
    env = PendulumEnv(project_config.simulation)
    network = load_controller(genome_path)
    result = env.run_episode(network=network)

    figures_dir = get_output_dir("demo")
    plot_rollout(result, save_path=figures_dir / "rollout.png", show=False)

    theta = result.angle
    time = result.time
    dt = float(getattr(env.config, "dt", 0.01))

    stride = max(1, frame_stride)
    frame_indices = list(range(0, len(theta), stride))
    if not frame_indices:
        frame_indices = [0]
    if frame_indices[-1] != len(theta) - 1:
        frame_indices.append(len(theta) - 1)

    interval_ms = max(1, int(1000.0 * dt * stride))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Reaction-wheel pendulum simulation")

    arm_length = float(getattr(env.config, "arm_length", 1.0))
    motor_radius = float(getattr(env.config, "motor_radius", 0.02))
    wheel_radius = float(getattr(env.config, "wheel_radius", 0.03))
    end_radius = max(motor_radius, wheel_radius)

    pendulum_length = max(1e-6, arm_length)
    viz_extent = 1.1 * (pendulum_length + end_radius)

    ax.set_xlim(-viz_extent, viz_extent)
    ax.set_ylim(-viz_extent, viz_extent)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    (line,) = ax.plot([], [], lw=3, color="steelblue")

    motor_patch = Circle(
        (0.0, 0.0),
        radius=motor_radius,
        facecolor="darkorange",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.75,
    )
    wheel_patch = Circle(
        (0.0, 0.0),
        radius=wheel_radius,
        facecolor="none",
        edgecolor="black",
        linewidth=2.0,
    )
    ax.add_patch(motor_patch)
    ax.add_patch(wheel_patch)

    time_text = ax.text(-1.12, 1.02, "", fontsize=12)

    def init():
        line.set_data([], [])
        motor_patch.center = (0.0, 0.0)
        wheel_patch.center = (0.0, 0.0)
        time_text.set_text("")
        return line, motor_patch, wheel_patch, time_text

    def update(frame_idx: int):
        th = theta[frame_idx]
        x = pendulum_length * np.sin(th)
        y = -pendulum_length * np.cos(th)
        line.set_data([0.0, x], [0.0, y])
        motor_patch.center = (x, y)
        wheel_patch.center = (x, y)
        time_text.set_text(f"t = {time[frame_idx]:.2f} s")
        return line, motor_patch, wheel_patch, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=interval_ms,
        blit=True,
    )

    fig.animation = ani  # type: ignore[attr-defined]

    if show:
        plt.show()

    return ani


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
