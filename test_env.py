from simulation.pendulum_env import (
    PendulumEnv,
    PendulumEnvConfig,
)

import numpy as np

from visualisation_code.plots import plot_rollout



# Simple dummy NN for testing.
# Always outputs zero torque contribution.
class DummyNetwork:
    def activate(self, x):
        return [0.0]


def run_test_rollout(show: bool = True) -> None:
    """Execute a single rollout and display diagnostics."""
    env = PendulumEnv(PendulumEnvConfig())
    result = env.run_episode(network=DummyNetwork())

    print("Simulation completed.\n")

    print("Max angle:", np.max(result.angle))
    print("Min angle:", np.min(result.angle))
    print("Max wheel speed:", np.max(result.wheel_velocity))
    print("Max commanded torque:", np.max(result.commanded_torque))
    print("Max actual torque:", np.max(result.actual_torque))
    print("\nNo NaNs in angle:", not np.isnan(result.angle).any())
    print("No NaNs in wheel speed:", not np.isnan(result.wheel_velocity).any())

    plot_rollout(result, show=show)


if __name__ == "__main__":
    run_test_rollout(show=True)