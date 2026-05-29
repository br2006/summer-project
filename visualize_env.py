from simulation.pendulum_env import (
    PendulumEnv,
    PendulumEnvConfig,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import pickle
from neat.network import FeedforwardNetwork


# Load trained genome
with open("best_genome.pkl", "rb") as f:
    genome = pickle.load(f)

network = FeedforwardNetwork(genome)


# -----------------------------------------
# Run simulation
# -----------------------------------------
env = PendulumEnv(
    PendulumEnvConfig()
)

result = env.run_episode(
    network=network
)

theta = result.angle
wheel_vel = result.wheel_velocity
t = result.time


# ==========================================
# FAST LIGHTWEIGHT RENDERER
# ==========================================

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

ax.set_aspect("equal")
ax.grid(True)

pivot = (0, 0)

line, = ax.plot([], [], lw=3)
mass, = ax.plot([], [], "o", markersize=20)

time_text = ax.text(
    -1.1,
    1.05,
    "",
    fontsize=12,
)

L = 1.0


def init():

    line.set_data([], [])
    mass.set_data([], [])

    time_text.set_text("")

    return line, mass, time_text


def update(frame):

    th = theta[frame]

    x = L * np.sin(th)
    y = -L * np.cos(th)

    line.set_data(
        [0, x],
        [0, y],
    )

    mass.set_data([x], [y])

    time_text.set_text(
        f"t = {t[frame]:.2f}s"
    )

    return line, mass, time_text


ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(theta), 2),
    init_func=init,
    interval=20,
    blit=True,
)

plt.show()