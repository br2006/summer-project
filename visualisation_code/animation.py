"""Pygame animation for the reaction-wheel pendulum demo."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:  # pragma: no cover - runtime dependency guard
    pygame = None  # type: ignore[assignment]

from config.settings import load_project_config
from neat.network import FeedforwardNetwork
from simulation.pendulum_env import PendulumEnv

from .output import get_output_dir
from .plots import plot_rollout

DEFAULT_FRAME_STRIDE = 2
WINDOW_SIZE = (1000, 760)
BASE_CANVAS_SIZE = (2000, 1520)
BACKGROUND = (245, 247, 250)
ARM_COLOR = (44, 95, 145)
PIVOT_COLOR = (45, 45, 45)
MOTOR_COLOR = (226, 124, 44)
WHEEL_RIM_COLOR = (35, 35, 35)
WHEEL_HUB_COLOR = (85, 85, 85)
SPOKE_COLOR = (70, 70, 70)
HUD_BG = (255, 255, 255, 220)
HUD_TEXT = (20, 20, 20)


def load_controller(path: Path) -> FeedforwardNetwork:
    with path.open("rb") as f:
        genome = pickle.load(f)
    return FeedforwardNetwork(genome)


def run_demo(
    genome_path: Path = Path("best_genome.pkl"),
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    show: bool = True,
    config_path: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """Run the pendulum simulation and render an animation using project config."""

    project_config = load_project_config(config_path)
    env = PendulumEnv(project_config.simulation)
    network = load_controller(genome_path)
    result = env.run_episode(network=network)

    figures_dir = get_output_dir("demo")
    plot_rollout(result, save_path=figures_dir / "rollout.png", show=False)

    theta = result.angle
    omega = result.angular_velocity
    time = result.time
    wheel_omega = result.wheel_velocity
    torque_cmd = result.commanded_torque
    torque_actual = result.actual_torque
    dt = float(getattr(env.config, "dt", 0.01))

    stride = max(1, frame_stride)
    frame_indices = list(range(0, len(theta), stride))
    if not frame_indices:
        frame_indices = [0]
    if frame_indices[-1] != len(theta) - 1:
        frame_indices.append(len(theta) - 1)

    arm_length = float(getattr(env.config, "arm_length", 1.0))
    motor_radius = float(getattr(env.config, "motor_radius", 0.02))
    wheel_radius = float(getattr(env.config, "wheel_radius", 0.03))
    scale = 0.78 * (BASE_CANVAS_SIZE[1] / max(arm_length + wheel_radius, 1e-6))

    def world_to_screen(x_m: float, y_m: float) -> tuple[int, int]:
        cx = BASE_CANVAS_SIZE[0] // 2
        cy = int(BASE_CANVAS_SIZE[1] * 0.26)
        x_px = int(round(cx + x_m * scale))
        y_px = int(round(cy - y_m * scale))
        return x_px, y_px

    def radius_to_px(radius_m: float) -> int:
        return max(2, int(round(radius_m * scale)))

    def draw_wheel_design(
        surf: pygame.Surface,
        center: tuple[int, int],
        radius_px: int,
        spin_angle: float,
    ) -> None:
        rim_thickness = max(4, radius_px // 5)
        hub_radius = max(4, radius_px // 5)
        spoke_count = 6

        pygame.gfxdraw.filled_circle(surf, center[0], center[1], radius_px, (235, 238, 243))
        pygame.gfxdraw.aacircle(surf, center[0], center[1], radius_px, WHEEL_RIM_COLOR)
        for t in range(rim_thickness):
            pygame.gfxdraw.aacircle(surf, center[0], center[1], max(1, radius_px - t), WHEEL_RIM_COLOR)

        for k in range(spoke_count):
            angle = spin_angle + (2.0 * np.pi * k / spoke_count)
            x2 = int(round(center[0] + (radius_px - rim_thickness - 2) * np.cos(angle)))
            y2 = int(round(center[1] + (radius_px - rim_thickness - 2) * np.sin(angle)))
            pygame.draw.line(surf, SPOKE_COLOR, center, (x2, y2), max(2, radius_px // 14))

        pygame.gfxdraw.filled_circle(surf, center[0], center[1], hub_radius, WHEEL_HUB_COLOR)
        pygame.gfxdraw.aacircle(surf, center[0], center[1], hub_radius, WHEEL_RIM_COLOR)

    def render_frame(
        surf: pygame.Surface,
        font: pygame.font.Font,
        frame_idx: int,
    ) -> None:
        surf.fill(BACKGROUND)

        pivot_px = world_to_screen(0.0, 0.0)
        th = float(theta[frame_idx])
        x_m = arm_length * np.sin(th)
        y_m = -arm_length * np.cos(th)
        bob_px = world_to_screen(x_m, y_m)

        pygame.draw.line(
            surf,
            ARM_COLOR,
            pivot_px,
            bob_px,
            width=max(5, radius_to_px(0.008)),
        )

        pivot_r = radius_to_px(0.011)
        pygame.gfxdraw.filled_circle(surf, pivot_px[0], pivot_px[1], pivot_r, PIVOT_COLOR)
        pygame.gfxdraw.aacircle(surf, pivot_px[0], pivot_px[1], pivot_r, (0, 0, 0))

        motor_r = radius_to_px(motor_radius)
        pygame.gfxdraw.filled_circle(surf, bob_px[0], bob_px[1], motor_r, MOTOR_COLOR)
        pygame.gfxdraw.aacircle(surf, bob_px[0], bob_px[1], motor_r, (10, 10, 10))

        wheel_r = radius_to_px(wheel_radius)
        draw_wheel_design(surf, bob_px, wheel_r, spin_angle=float(wheel_omega[frame_idx]) * float(time[frame_idx]))

        hud = pygame.Surface((520, 180), pygame.SRCALPHA)
        pygame.draw.rect(hud, HUD_BG, hud.get_rect(), border_radius=18)
        lines = [
            f"t = {time[frame_idx]:6.2f} s",
            f"theta = {np.degrees(theta[frame_idx]):7.2f} deg   omega = {omega[frame_idx]:7.3f} rad/s",
            f"wheel = {wheel_omega[frame_idx]:7.2f} rad/s",
            f"torque cmd = {torque_cmd[frame_idx]:7.3f} N m   actual = {torque_actual[frame_idx]:7.3f} N m",
        ]
        y = 14
        for text in lines:
            surf_text = font.render(text, True, HUD_TEXT)
            hud.blit(surf_text, (16, y))
            y += 40
        surf.blit(hud, (28, BASE_CANVAS_SIZE[1] - 220))

    if not show:
        return {
            "time": result.time,
            "angle": result.angle,
            "angular_velocity": result.angular_velocity,
            "wheel_velocity": result.wheel_velocity,
        }

    if pygame is None:
        raise ImportError(
            "Pygame is required for visualisation. Install with `pip install pygame`."
        )

    pygame.init()
    pygame.display.set_caption("Reaction-wheel pendulum simulation")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    base_canvas = pygame.Surface(BASE_CANVAS_SIZE, pygame.SRCALPHA)
    font = pygame.font.SysFont("consolas", 34)

    frame_pointer = 0
    running = True
    interval_ms = max(1, int(1000.0 * dt * stride))
    accumulator = 0.0

    while running:
        elapsed = clock.tick(120)
        accumulator += elapsed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        while accumulator >= interval_ms and frame_pointer < len(frame_indices) - 1:
            accumulator -= interval_ms
            frame_pointer += 1

        render_frame(base_canvas, font, frame_indices[frame_pointer])

        window_w, window_h = screen.get_size()
        scaled = pygame.transform.smoothscale(base_canvas, (window_w, window_h))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        if frame_pointer >= len(frame_indices) - 1:
            # Keep last frame visible until user closes window.
            continue

    pygame.quit()

    return {
        "time": result.time,
        "angle": result.angle,
        "angular_velocity": result.angular_velocity,
        "wheel_velocity": result.wheel_velocity,
    }


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
