from simulation.pendulum_env import (
    PendulumEnv,
    PendulumEnvConfig,
)

import matplotlib.pyplot as plt
import numpy as np


# Simple dummy NN for testing.
# Always outputs zero torque contribution.
class DummyNetwork:
    def activate(self, x):
        return [0.0]


# Create environment
env = PendulumEnv(
    PendulumEnvConfig()
)

# Run one rollout
result = env.run_episode(
    network=DummyNetwork()
)

print("Simulation completed.\n")

print("Max angle:", np.max(result.angle))
print("Min angle:", np.min(result.angle))

print("Max wheel speed:", np.max(result.wheel_velocity))

print("Max commanded torque:",
      np.max(result.commanded_torque))

print("Max actual torque:",
      np.max(result.actual_torque))

print("\nNo NaNs in angle:",
      not np.isnan(result.angle).any())

print("No NaNs in wheel speed:",
      not np.isnan(result.wheel_velocity).any())


# -----------------------------
# Plot pendulum angle
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    result.time,
    result.angle,
)

plt.title("Pendulum Angle")

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")

plt.grid(True)


# -----------------------------
# Plot wheel speed
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    result.time,
    result.wheel_velocity,
)

plt.title("Reaction Wheel Velocity")

plt.xlabel("Time (s)")
plt.ylabel("Wheel Velocity")

plt.grid(True)


# -----------------------------
# Plot commanded vs actual torque
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    result.time,
    result.commanded_torque,
    label="Commanded Torque",
)

plt.plot(
    result.time,
    result.actual_torque,
    label="Actual Torque",
)

plt.title("Motor Lag Behaviour")

plt.xlabel("Time (s)")
plt.ylabel("Torque")

plt.legend()
plt.grid(True)

plt.show()