"""Pygame animation for the reaction-wheel pendulum demo."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - runtime dependency guard
    pygame = None  # type: ignore[assignment]

from config.settings import load_project_config
from neat.network import FeedforwardNetwork
from simulation.nn_pendulum_env import NNPendulumEnv
from simulation.pendulum_env import PendulumEnv

from .output import get_output_dir
from .plots import plot_rollout

DEFAULT_FRAME_STRIDE = 5  # display every N simulation ticks
DEFAULT_WINDOW_SIZE = (800, 600)
SUPERSAMPLE_SCALE = 3
DEMO_DURATION_SECONDS = 20.0
PENDULUM_PIVOT_X_RATIO = 0.42

# Warm light palette
BG = (250, 247, 242)
PANEL_BG = (255, 252, 247)
PANEL_BORDER = (232, 220, 205)
ROD_COL = (28, 18, 12)
PIVOT_COL = (22, 14, 8)
BOB_RIM = (35, 22, 14)
SPOKE_COL = (40, 26, 16)
HUB_COL = (30, 18, 10)
MOTOR_COL = (55, 42, 34)
MOTOR_RING = (90, 72, 58)
TIP_COL = (220, 155, 60)
TARGET_COL = (180, 210, 160)
LABEL_COL = (160, 130, 105)
VALUE_COL = (70, 55, 45)
HINT_COL = (195, 180, 165)
MODE_BAL = (80, 170, 110)
MODE_BAL_BG = (235, 248, 238)
MODE_SWG = (210, 120, 55)
MODE_SWG_BG = (253, 242, 228)
DIVIDER = (225, 212, 198)


def load_controller(path: Path) -> FeedforwardNetwork:
    with path.open("rb") as f:
        genome = pickle.load(f)
    return FeedforwardNetwork(genome)


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rounded_rect(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    rect: tuple[int, int, int, int],
    radius: int,
    border_color: Optional[tuple[int, int, int]] = None,
    border_width: int = 1,
) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color is not None:
        pygame.draw.rect(
            surface,
            border_color,
            rect,
            width=border_width,
            border_radius=radius,
        )


def _draw_dashed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    p1: tuple[int, int],
    p2: tuple[int, int],
    dash: int,
    gap: int,
    width: int,
) -> None:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = float(np.hypot(dx, dy))
    if length <= 1e-9:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    on = True
    while pos < length:
        seg = min(pos + (dash if on else gap), length)
        if on:
            a = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
            b = (int(p1[0] + ux * seg), int(p1[1] + uy * seg))
            pygame.draw.line(surface, color, a, b, width)
        pos += dash if on else gap
        on = not on


def _draw_flywheel(
    canvas: pygame.Surface,
    cx: int,
    cy: int,
    phi: float,
    wheel_radius_px: int,
) -> None:
    outer_r = max(8, wheel_radius_px)
    inner_r = max(3, outer_r - max(4, int(round(outer_r * 0.13))))
    steps = 48

    diam = outer_r * 2 + 4
    wsurf = pygame.Surface((diam, diam), pygame.SRCALPHA)
    wsurf.fill((0, 0, 0, 0))
    ox, oy = diam // 2, diam // 2

    # Annular ring
    pygame.draw.circle(wsurf, (*SPOKE_COL, 255), (ox, oy), outer_r)
    pygame.draw.circle(wsurf, (0, 0, 0, 0), (ox, oy), inner_r)

    # Outer rim
    pygame.draw.circle(wsurf, (*BOB_RIM, 255), (ox, oy), outer_r, max(1, outer_r // 18))

    # Two thick spoke sectors, 90° each
    spoke_half_angle = np.pi / 4.0
    for offset in (0.0, np.pi):
        points = [(ox, oy)]
        for step in range(steps + 1):
            a = phi + offset - spoke_half_angle + step * (2.0 * spoke_half_angle / steps)
            points.append((int(ox + outer_r * np.cos(a)), int(oy + outer_r * np.sin(a))))
        pygame.draw.polygon(wsurf, (*BOB_RIM, 255), points)

    # Hub stack
    pygame.draw.circle(wsurf, (*PANEL_BG, 255), (ox, oy), max(3, outer_r // 8))
    pygame.draw.circle(wsurf, (*HUB_COL, 255), (ox, oy), max(2, outer_r // 12))
    pygame.draw.circle(wsurf, (*PANEL_BG, 255), (ox, oy), max(1, outer_r // 24))

    canvas.blit(wsurf, (cx - ox, cy - oy))

    # Tip marker
    tip_r = max(2, outer_r // 18)
    tip = (
        int(cx + (outer_r - max(2, outer_r // 14)) * np.cos(phi)),
        int(cy + (outer_r - max(2, outer_r // 14)) * np.sin(phi)),
    )
    pygame.draw.circle(canvas, PANEL_BG, tip, tip_r + 2)
    pygame.draw.circle(canvas, TIP_COL, tip, tip_r)


def run_demo(
    genome_path: Path = Path("best_genome.pkl"),
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    show: bool = True,
    config_path: Optional[Path] = None,
    nn_only: bool = False,
    save_gif_path: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """Run the pendulum simulation and render an animation using project config."""

    project_config = load_project_config(config_path)
    # Force a fixed 20-second rollout for each visualisation run.
    project_config.simulation.duration = DEMO_DURATION_SECONDS

    # For NN-only visualisation, disable swing-up so animation reflects pure NN balance behaviour.
    if nn_only:
        project_config.simulation.enable_swingup = False

    env = NNPendulumEnv(project_config.simulation) if nn_only else PendulumEnv(project_config.simulation)
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
    control_mode = getattr(result, "control_mode", np.ones_like(theta))
    dt = float(getattr(env.config, "dt", 0.01))

    stride = max(1, int(frame_stride))
    frame_indices = list(range(0, len(theta), stride))
    if not frame_indices:
        frame_indices = [0]
    if frame_indices[-1] != len(theta) - 1:
        frame_indices.append(len(theta) - 1)

    arm_length = float(getattr(env.config, "arm_length", 1.0))
    wheel_radius = float(getattr(env.config, "wheel_radius", 0.03))
    motor_radius = float(getattr(env.config, "motor_radius", 0.015))
    total_inertia = float(getattr(env, "total_inertia", 1.0))
    total_mass = float(getattr(env, "total_mass", 1.0))
    r_cm = float(getattr(env, "r_cm", arm_length))
    gravity = float(getattr(env.config, "gravity", 9.81))

    # Display wheel angle integrated from backend wheel velocity for faithful spin progression.
    wheel_phi = np.cumsum(wheel_omega * dt)

    def render_frame(
        surf: pygame.Surface,
        font_ui: pygame.font.Font,
        font_bold: pygame.font.Font,
        font_hint: pygame.font.Font,
        frame_idx: int,
        render_w: int,
        render_h: int,
    ) -> None:
        def s(n: float) -> int:
            return int(round(n * SUPERSAMPLE_SCALE))

        surf.fill(BG)

        # Keep pendulum clearly separated from the telemetry panel on the right.
        pivot = (int(round(PENDULUM_PIVOT_X_RATIO * render_w)), int(round(0.56 * render_h)))
        px, py = pivot

        # Physically consistent scaling: map actual mechanical size to viewport.
        physical_extent = max(1e-6, arm_length + max(wheel_radius, motor_radius))
        visual_extent_px = 0.32 * min(render_w, render_h)
        meters_to_px = visual_extent_px / physical_extent
        arm_len_px = int(round(arm_length * meters_to_px))

        th = float(theta[frame_idx])
        bob_x = int(round(px + (arm_length * np.sin(th)) * meters_to_px))
        bob_y = int(round(py - (arm_length * np.cos(th)) * meters_to_px))

        target_len = int(round(1.136 * arm_len_px))
        _draw_dashed_line(
            surf,
            TARGET_COL,
            (px, py),
            (px, py - target_len),
            dash=s(7),
            gap=s(5),
            width=max(1, s(1)),
        )

        pygame.draw.line(surf, ROD_COL, pivot, (bob_x, bob_y), s(8))

        pygame.draw.circle(surf, PANEL_BG, pivot, s(13))
        pygame.draw.circle(surf, PIVOT_COL, pivot, s(10))
        pygame.draw.circle(surf, PANEL_BG, pivot, s(3))

        # Draw motor housing at the arm tip (same axis as wheel in dynamics model).
        motor_r_px = max(s(6), int(round(motor_radius * meters_to_px)))
        pygame.draw.circle(surf, MOTOR_RING, (bob_x, bob_y), motor_r_px + s(2))
        pygame.draw.circle(surf, MOTOR_COL, (bob_x, bob_y), motor_r_px)
        pygame.draw.circle(surf, PANEL_BG, (bob_x, bob_y), max(s(2), motor_r_px // 3))

        wheel_r_px = max(s(10), int(round(wheel_radius * meters_to_px)))
        _draw_flywheel(
            surf,
            bob_x,
            bob_y,
            phi=float(wheel_phi[frame_idx]),
            wheel_radius_px=wheel_r_px,
        )

        # Mode badge from backend control-mode telemetry.
        is_balance_mode = int(control_mode[frame_idx]) == 1
        if is_balance_mode:
            balance_label = "NN-only" if nn_only else "Balance · PID/Hybrid"
            m_col, m_bg, m_txt = MODE_BAL, MODE_BAL_BG, balance_label
        else:
            swingup_label = "NN-only" if nn_only else "Swing-up · Energy"
            m_col, m_bg, m_txt = MODE_SWG, MODE_SWG_BG, swingup_label

        badge_surf = font_bold.render(m_txt, True, m_col)
        badge_w = badge_surf.get_width() + s(24)
        badge_h = badge_surf.get_height() + s(10)
        _rounded_rect(
            surf,
            m_bg,
            (s(12), s(12), badge_w, badge_h),
            s(6),
            border_color=m_col,
            border_width=max(1, s(1)),
        )
        surf.blit(badge_surf, (s(24), s(17)))

        # Telemetry card + physically-consistent energy rows from PendulumEnv mass properties.
        theta_i = float(theta[frame_idx])
        omega_i = float(omega[frame_idx])
        wheel_i = float(wheel_omega[frame_idx])
        cmd_i = float(torque_cmd[frame_idx])
        act_i = float(torque_actual[frame_idx])

        kinetic_e = 0.5 * total_inertia * omega_i**2
        potential_e = -total_mass * gravity * r_cm * np.cos(theta_i)
        target_e = -total_mass * gravity * r_cm

        # Compact telemetry rows to avoid occluding pendulum swing envelope.
        rows = [
            ("θ", f"{theta_i:+.4f} rad {np.degrees(theta_i):+.1f}°"),
            ("dθ/dt", f"{omega_i:+.4f} rad/s"),
            ("Torque cmd", f"{cmd_i:+.5f} N·m"),
            ("Torque act", f"{act_i:+.5f} N·m"),
            ("Wheel ω", f"{wheel_i:+.4f} rad/s"),
            ("ΔE", f"{(kinetic_e + potential_e - target_e):+.5f} J"),
        ]

        card_w = s(250)
        card_x = render_w - card_w - s(12)
        card_y = s(12) + badge_h + s(8)
        row_h = s(22)
        label_w = s(74)
        val_x = card_x + label_w + s(8)
        card_h = len(rows) * row_h + s(12)
        pad_top = s(5)

        _rounded_rect(
            surf,
            PANEL_BG,
            (card_x, card_y, card_w, card_h),
            s(8),
            border_color=PANEL_BORDER,
            border_width=max(1, s(1)),
        )

        for i, (label, value) in enumerate(rows):
            y = card_y + pad_top + i * row_h
            if i > 0:
                pygame.draw.line(
                    surf,
                    DIVIDER,
                    (card_x + s(8), y - s(1)),
                    (card_x + card_w - s(8), y - s(1)),
                    max(1, s(1)),
                )
            surf.blit(font_ui.render(label, True, LABEL_COL), (card_x + s(12), y + s(5)))
            surf.blit(font_ui.render(value, True, VALUE_COL), (val_x, y + s(5)))

        hint = font_hint.render("R Restart · ESC/Q Quit", True, HINT_COL)
        surf.blit(hint, (s(12), render_h - s(22)))

    if not show and save_gif_path is None:
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
    display_flags = pygame.RESIZABLE if show else pygame.HIDDEN
    pygame.display.set_caption("Reaction-wheel pendulum simulation")
    screen = pygame.display.set_mode(DEFAULT_WINDOW_SIZE, display_flags)
    clock = pygame.time.Clock()

    base_w, base_h = DEFAULT_WINDOW_SIZE
    render_w = base_w * SUPERSAMPLE_SCALE
    render_h = base_h * SUPERSAMPLE_SCALE
    canvas = pygame.Surface((render_w, render_h), pygame.SRCALPHA)

    font_ui = pygame.font.SysFont("segoeui", int(round(14 * SUPERSAMPLE_SCALE)))
    font_bold = pygame.font.SysFont("segoeui", int(round(14 * SUPERSAMPLE_SCALE)), bold=True)
    font_hint = pygame.font.SysFont("segoeui", int(round(13 * SUPERSAMPLE_SCALE)))

    frame_pointer = 0
    running = True
    # Match reference cadence intent: frame every TPF ticks with dt simulation tick.
    display_fps = max(1.0, 1.0 / (dt * stride))

    gif_frames = []
    if save_gif_path is not None:
        from PIL import Image

    if show:
        while running:
            clock.tick(int(round(display_fps)))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        # Restart from first frame on demand.
                        frame_pointer = -1

            # Loop continuously through frames for a repeating animation.
            if frame_pointer < len(frame_indices) - 1:
                frame_pointer += 1
            else:
                frame_pointer = 0

            render_frame(
                canvas,
                font_ui,
                font_bold,
                font_hint,
                frame_indices[frame_pointer],
                render_w,
                render_h,
            )

            window_w, window_h = screen.get_size()
            scaled = pygame.transform.smoothscale(canvas, (window_w, window_h))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()

            if save_gif_path is not None:
                raw = pygame.image.tostring(scaled, "RGB")
                gif_frames.append(Image.frombytes("RGB", (window_w, window_h), raw))
    else:
        for idx in frame_indices:
            render_frame(
                canvas,
                font_ui,
                font_bold,
                font_hint,
                int(idx),
                render_w,
                render_h,
            )
            window_w, window_h = DEFAULT_WINDOW_SIZE
            scaled = pygame.transform.smoothscale(canvas, (window_w, window_h))
            screen.blit(scaled, (0, 0))
            pygame.display.flip()

            if save_gif_path is not None:
                raw = pygame.image.tostring(scaled, "RGB")
                gif_frames.append(Image.frombytes("RGB", (window_w, window_h), raw))

    if save_gif_path is not None and gif_frames:
        save_gif_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(20, int(round(1000 / max(1.0, display_fps))))
        gif_frames[0].save(
            save_gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )

    pygame.quit()

    return {
        "time": result.time,
        "angle": result.angle,
        "angular_velocity": result.angular_velocity,
        "wheel_velocity": result.wheel_velocity,
    }


def save_rollout_animation_gif(
    result,
    save_path: Path,
    fps: int = 20,
    max_frames: int = 240,
    frame_size: tuple[int, int] = (560, 320),
) -> Path:
    """Backward-compatible alias retained for older callers."""
    raise RuntimeError(
        "save_rollout_animation_gif(result, ...) is deprecated. "
        "Use run_demo(..., save_gif_path=...) so exported visuals match visualize_env.py exactly."
    )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
