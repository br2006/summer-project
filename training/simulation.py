"""
Convenience entry point to run a single pendulum rollout without evolution.

Example:
  python training/simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from neat.genome import Genome, InnovationTracker
from neat.network import FeedforwardNetwork
from simulation.pendulum_env import PendulumEnv
from visualisation_code.plots import plot_rollout



def main() -> None:
    project = load_project_config()
    tracker = InnovationTracker()
    genome = Genome.create_minimal(4, 1, tracker)
    net = FeedforwardNetwork(genome)
    env = PendulumEnv(project.simulation)
    result = env.run_episode(network=net)
    plot_rollout(result, target_band_hz=project.spectral.target_band_hz)


if __name__ == "__main__":
    main()
