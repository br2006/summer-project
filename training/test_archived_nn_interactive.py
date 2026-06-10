"""Interactive tester for an archived NN-only run with live animation controls.

Features:
- Uses archived run config + best genome
- Same visual style as visualisation_code.animation
- Supports multiple initial angles (manual selection via --initial-angles-deg)
- Keyboard controls:
    * R: restart current angle
    * Left/Right arrows: inject small disturbance torque while held
    * Esc/Q: quit
- Auto-restarts every 30 seconds (configurable)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_project_config
from simulation.nn_pendulum_env import NNPendulumEnv
from visualisation_code.animation import (
    BG,
    BOB_RIM,
    DEFAULT_WINDOW_SIZE,
    DIVIDER,
    HINT_COL,
    HUB_COL,
    LABEL_COL,
    MODE_BAL,
    MODE_BAL_BG,
    MODE_SWG,
    MODE_SWG_BG,
    MOTOR_COL,
    MOTOR_RING,
    PANEL_BG,
    PANEL_BORDER,
    PENDULUM_PIVOT_X_RATIO,
    PIVOT_COL,
    ROD_COL,
    SPOKE_COL,
    SUPERSAMPLE_SCALE,
    TARGET_COL,
    TIP_COL,
    VALUE_COL,
    _draw_dashed_line,
    _draw_flywheel,
    _rounded_rect,
    _wrap_angle,
    load_controller,
    pygame,
)


@dataclass
class LiveState:
    theta: float
    omega: float
    wheel_omega: float
    sim_t: float
    wheel_phi: float


def _parse_angle_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("At least one initial angle must be provided.")
    return values


def _make_live_state(env: NNPendulumEnv, initial_angle_rad: float) -> LiveState:
    env.reset()
    return LiveState(
        theta=float(initial_angle_rad),
        omega=float(env.config.initial_omega),
        wheel_omega=0.0,
        sim_t=0.0,
        wheel_phi=0.0,
    )


def _simulate_step(
    env: NNPendulumEnv,
    network,
    state: LiveState,
    manual_torque: float,
) -> tuple[LiveState, dict[str, float | int]]:
    cfg = env.config
    dt = float(cfg.dt)

    base_a = float(env.disturbance.sample(state.sim_t))
    env.sensor.update_raw(state.theta, state.omega, base_a, state.wheel_omega)
    reading = env.sensor.read()

    u_nn = float(network.activate(reading.as_array())[0])

    theta_meas = reading.angle * np.pi
    omega_meas = reading.angular_velocity * 10.0
    wheel_omega_meas = reading.wheel_velocity * env.sensor.wheel_scale
    env._update_control_mode(theta_meas, omega_meas, wheel_omega_meas)

    if env.control_mode == "SWINGUP":
        torque_ctrl = float(env._swingup_torque(state.theta, state.omega, state.wheel_omega))
    else:
        torque_ctrl = float(env.nn_controller.compute_torque(u_nn))

    torque_cmd = float(
        np.clip(
            torque_ctrl + manual_torque,
            -cfg.max_wheel_torque,
            cfg.max_wheel_torque,
        )
    )
    env.actuator.apply_torque(torque_cmd)

    resonance = float(env._resonance_disturbance(state.sim_t, torque_cmd))

    theta, omega, wheel_omega = env._integrate_step(
        state.theta,
        state.omega,
        state.wheel_omega,
        base_a,
        resonance,
        dt,
    )

    next_state = LiveState(
        theta=float(theta),
        omega=float(omega),
        wheel_omega=float(wheel_omega),
        sim_t=float(state.sim_t + dt),
        wheel_phi=float(state.wheel_phi + wheel_omega * dt),
    )

    telemetry: dict[str, float | int] = {
        "u_nn": u_nn,
        "torque_ctrl": torque_ctrl,
        "torque_cmd": torque_cmd,
        "torque_actual": float(env.actual_torque),
        "base_a": base_a,
        "resonance": resonance,
        "mode": 1 if env.control_mode == "BALANCE" else 0,
    }
    return next_state, telemetry


def run_interactive(
    run_dir: Path,
    initial_angles_deg: list[float],
    manual_torque: float,
    auto_restart_seconds: float,
    max_wheel_torque: float | None,
    nn_torque_scale: float | None,
    seed: int | None,
    deterministic: bool,
    vary_disturbance_per_restart: bool,
    enable_swingup: bool | None,
) -> None:
    if pygame is None:
        raise ImportError("Pygame is required. Install with `pip install pygame`.")

    genome_path = run_dir / "artifacts" / "best_genome_nn_only.pkl"
    config_path = run_dir / "metadata" / "config_snapshot.yaml"

    if not genome_path.exists():
        raise FileNotFoundError(f"Genome not found: {genome_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config snapshot not found: {config_path}")

    project = load_project_config(config_path)
    archived_enable_swingup = bool(project.simulation.enable_swingup)
    if enable_swingup is not None:
        project.simulation.enable_swingup = bool(enable_swingup)
    if max_wheel_torque is not None:
        project.simulation.max_wheel_torque = float(max_wheel_torque)
    if nn_torque_scale is not None:
        project.simulation.nn_torque_scale = float(nn_torque_scale)

    if deterministic:
        sim = project.simulation
        sim.angle_noise_std = 0.0
        sim.gyro_noise_std = 0.0
        sim.gyro_bias_drift_rate = 0.0
        sim.broadband_noise_gain = 0.0
        sim.vibration_amps = [0.0 for _ in sim.vibration_amps]
        sim.table_ring_amps_mps2 = [0.0 for _ in sim.table_ring_amps_mps2]
        sim.impulse_probability = 0.0
        sim.impulse_magnitude = 0.0
        sim.footstep_accel_mps2 = 0.0
        sim.accelerometer_noise_std_mps2 = 0.0

    if seed is not None:
        np.random.seed(int(seed))

    env = NNPendulumEnv(project.simulation)
    network = load_controller(genome_path)

    arm_length = float(getattr(env.config, "arm_length", 1.0))
    wheel_radius = float(getattr(env.config, "wheel_radius", 0.03))
    motor_radius = float(getattr(env.config, "motor_radius", 0.015))
    total_inertia = float(getattr(env, "total_inertia", 1.0))
    total_mass = float(getattr(env, "total_mass", 1.0))
    r_cm = float(getattr(env, "r_cm", arm_length))
    gravity = float(getattr(env.config, "gravity", 9.81))

    pygame.init()
    pygame.display.set_caption("Archived NN-only interactive test")
    screen = pygame.display.set_mode(DEFAULT_WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()

    base_w, base_h = DEFAULT_WINDOW_SIZE
    render_w = base_w * SUPERSAMPLE_SCALE
    render_h = base_h * SUPERSAMPLE_SCALE
    canvas = pygame.Surface((render_w, render_h), pygame.SRCALPHA)

    font_ui = pygame.font.SysFont("segoeui", int(round(14 * SUPERSAMPLE_SCALE)))
    font_bold = pygame.font.SysFont("segoeui", int(round(14 * SUPERSAMPLE_SCALE)), bold=True)
    font_hint = pygame.font.SysFont("segoeui", int(round(13 * SUPERSAMPLE_SCALE)))

    angle_index = 0
    episode_index = 0
    state = _make_live_state(env, np.radians(initial_angles_deg[angle_index]))
    telemetry: dict[str, float | int] = {
        "u_nn": 0.0,
        "torque_ctrl": 0.0,
        "torque_cmd": 0.0,
        "torque_actual": 0.0,
        "manual_input": 0.0,
        "base_a": 0.0,
        "resonance": 0.0,
        "mode": 1 if env.control_mode == "BALANCE" else 0,
    }

    def s(n: float) -> int:
        return int(round(n * SUPERSAMPLE_SCALE))

    def restart(cycle_angle: bool) -> None:
        nonlocal angle_index, episode_index, state, telemetry
        episode_index += 1
        if cycle_angle and len(initial_angles_deg) > 1:
            angle_index = (angle_index + 1) % len(initial_angles_deg)
        if seed is not None:
            if vary_disturbance_per_restart:
                np.random.seed(int(seed) + episode_index)
            else:
                np.random.seed(int(seed))
        state = _make_live_state(env, np.radians(initial_angles_deg[angle_index]))
        telemetry = {
            "u_nn": 0.0,
            "torque_ctrl": 0.0,
            "torque_cmd": 0.0,
            "torque_actual": 0.0,
            "manual_input": 0.0,
            "base_a": 0.0,
            "resonance": 0.0,
            "mode": 1 if env.control_mode == "BALANCE" else 0,
        }
        print(
            f"[restart] initial angle = {initial_angles_deg[angle_index]:+.1f} deg "
            f"(index {angle_index + 1}/{len(initial_angles_deg)})"
        )

    def render_frame() -> None:
        nonlocal state, telemetry
        canvas.fill(BG)

        pivot = (int(round(PENDULUM_PIVOT_X_RATIO * render_w)), int(round(0.56 * render_h)))
        px, py = pivot

        physical_extent = max(1e-6, arm_length + max(wheel_radius, motor_radius))
        visual_extent_px = 0.32 * min(render_w, render_h)
        meters_to_px = visual_extent_px / physical_extent
        arm_len_px = int(round(arm_length * meters_to_px))

        th = float(state.theta)
        bob_x = int(round(px + (arm_length * np.sin(th)) * meters_to_px))
        bob_y = int(round(py - (arm_length * np.cos(th)) * meters_to_px))

        target_len = int(round(1.136 * arm_len_px))
        _draw_dashed_line(
            canvas,
            TARGET_COL,
            (px, py),
            (px, py - target_len),
            dash=s(7),
            gap=s(5),
            width=max(1, s(1)),
        )

        pygame.draw.line(canvas, ROD_COL, pivot, (bob_x, bob_y), s(8))

        pygame.draw.circle(canvas, PANEL_BG, pivot, s(13))
        pygame.draw.circle(canvas, PIVOT_COL, pivot, s(10))
        pygame.draw.circle(canvas, PANEL_BG, pivot, s(3))

        motor_r_px = max(s(6), int(round(motor_radius * meters_to_px)))
        pygame.draw.circle(canvas, MOTOR_RING, (bob_x, bob_y), motor_r_px + s(2))
        pygame.draw.circle(canvas, MOTOR_COL, (bob_x, bob_y), motor_r_px)
        pygame.draw.circle(canvas, PANEL_BG, (bob_x, bob_y), max(s(2), motor_r_px // 3))

        wheel_r_px = max(s(10), int(round(wheel_radius * meters_to_px)))
        _draw_flywheel(
            canvas,
            bob_x,
            bob_y,
            phi=float(state.wheel_phi),
            wheel_radius_px=wheel_r_px,
        )

        is_balance_mode = int(telemetry["mode"]) == 1
        if is_balance_mode:
            m_col, m_bg, m_txt = MODE_BAL, MODE_BAL_BG, "Balance · NN-only"
        else:
            m_col, m_bg, m_txt = MODE_SWG, MODE_SWG_BG, "Swing-up · Energy"

        badge_surf = font_bold.render(m_txt, True, m_col)
        badge_w = badge_surf.get_width() + s(24)
        badge_h = badge_surf.get_height() + s(10)
        _rounded_rect(
            canvas,
            m_bg,
            (s(12), s(12), badge_w, badge_h),
            s(6),
            border_color=m_col,
            border_width=max(1, s(1)),
        )
        canvas.blit(badge_surf, (s(24), s(17)))

        kinetic_e = 0.5 * total_inertia * state.omega**2
        potential_e = -total_mass * gravity * r_cm * np.cos(state.theta)
        target_e = -total_mass * gravity * r_cm

        rows = [
            ("θ", f"{state.theta:+.4f} rad {np.degrees(state.theta):+.1f}°"),
            ("dθ/dt", f"{state.omega:+.4f} rad/s"),
            ("Wheel ω", f"{state.wheel_omega:+.4f} rad/s"),
            ("Torque cmd", f"{float(telemetry['torque_cmd']):+.5f} N·m"),
            ("Torque act", f"{float(telemetry['torque_actual']):+.5f} N·m"),
            ("Manual τ", f"{float(telemetry['manual_input']):+.4f} N·m"),
            ("Init angle", f"{initial_angles_deg[angle_index]:+.1f}° ({angle_index + 1}/{len(initial_angles_deg)})"),
            ("Time", f"{state.sim_t:5.2f} s / {auto_restart_seconds:.0f} s"),
            ("ΔE", f"{(kinetic_e + potential_e - target_e):+.5f} J"),
        ]

        card_w = s(300)
        card_x = render_w - card_w - s(12)
        card_y = s(12) + badge_h + s(8)
        row_h = s(22)
        label_w = s(90)
        val_x = card_x + label_w + s(8)
        card_h = len(rows) * row_h + s(12)
        pad_top = s(5)

        _rounded_rect(
            canvas,
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
                    canvas,
                    DIVIDER,
                    (card_x + s(8), y - s(1)),
                    (card_x + card_w - s(8), y - s(1)),
                    max(1, s(1)),
                )
            canvas.blit(font_ui.render(label, True, LABEL_COL), (card_x + s(12), y + s(5)))
            canvas.blit(font_ui.render(value, True, VALUE_COL), (val_x, y + s(5)))

        hint = font_hint.render(
            "R Restart · ←/→ Disturb · Auto-restart 30s · ESC/Q Quit",
            True,
            HINT_COL,
        )
        canvas.blit(hint, (s(12), render_h - s(22)))

        window_w, window_h = screen.get_size()
        scaled = pygame.transform.smoothscale(canvas, (window_w, window_h))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

    dt = float(env.config.dt)
    display_fps = int(np.clip(round(1.0 / max(1e-6, dt)), 30, 120))
    running = True

    print(f"Loaded run: {run_dir}")
    print(f"Angles (deg): {initial_angles_deg}")
    print(f"Manual disturbance torque magnitude: {manual_torque:.4f} N·m")
    print(
        "Swing-up mode: "
        f"archived={'ON' if archived_enable_swingup else 'OFF'} -> "
        f"effective={'ON' if project.simulation.enable_swingup else 'OFF'}"
    )
    print(
        "Max wheel torque: "
        f"{project.simulation.max_wheel_torque:.4f} N·m"
        + (" (archived)" if max_wheel_torque is None else " (override)")
    )
    print(
        f"NN torque scale: {project.simulation.nn_torque_scale:.4f}"
        + (" (archived)" if nn_torque_scale is None else " (override)")
    )
    print(f"Auto-restart every {auto_restart_seconds:.1f} seconds")
    print(f"Deterministic mode: {'ON' if deterministic else 'OFF'}")
    if seed is not None:
        print(f"Random seed base: {int(seed)}")
        print(
            "Disturbance realization per restart: "
            + ("varied" if vary_disturbance_per_restart else "fixed (archived-like)")
        )
    print("Controls: R restart, Left/Right arrows disturb, Esc/Q quit")

    r_debounce_ms = 250
    last_r_ms = -10_000
    left_pressed = False
    right_pressed = False

    while running:
        clock.tick(display_fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    now_ms = pygame.time.get_ticks()
                    if now_ms - last_r_ms >= r_debounce_ms:
                        restart(cycle_angle=False)
                        last_r_ms = now_ms
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    left_pressed = True
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    right_pressed = True
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    left_pressed = False
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    right_pressed = False

        manual = 0.0
        if left_pressed:
            manual += abs(manual_torque)
        if right_pressed:
            manual -= abs(manual_torque)

        state, telemetry = _simulate_step(env, network, state, manual)
        telemetry["manual_input"] = float(manual)
        if state.sim_t >= auto_restart_seconds:
            restart(cycle_angle=False)

        render_frame()

    pygame.quit()


def main() -> None:
    default_run = (
        ROOT
        / "outputs"
        / "runs"
        / "nn_only"
        / "20260606_160106__pop-80__gen-200__angle-40p0"
    )

    parser = argparse.ArgumentParser(
        description="Interactive tester for archived NN-only runs"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run,
        help="Archived run directory containing artifacts/ and metadata/",
    )
    parser.add_argument(
        "--initial-angles-deg",
        type=str,
        default="40",
        help="Comma-separated list of initial angles in degrees (uses first value unless manually restarted with a different run argument)",
    )
    parser.add_argument(
        "--manual-torque",
        type=float,
        default=0.3,
        help="Manual disturbance torque magnitude (N·m) applied by arrow keys",
    )
    parser.add_argument(
        "--auto-restart-seconds",
        type=float,
        default=30.0,
        help="Automatically restart after this many simulated seconds",
    )
    parser.add_argument(
        "--max-wheel-torque",
        type=float,
        default=0.8,
        help="Max actuator torque override (N·m). Default: 0.8",
    )
    parser.add_argument(
        "--nn-torque-scale",
        type=float,
        default=0.8,
        help="NN output-to-torque scale override. Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for reproducible behaviour (use -1 to disable seeding)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable sensor/disturbance randomness for clean reproducible balancing tests",
    )
    parser.add_argument(
        "--vary-disturbance-per-restart",
        action="store_true",
        help="Use a different seeded disturbance/noise realization on each restart",
    )
    parser.add_argument(
        "--enable-swingup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional SWINGUP->BALANCE mode override (omit to keep archived setting)",
    )
    args = parser.parse_args()

    angles = _parse_angle_list(args.initial_angles_deg)
    run_interactive(
        run_dir=args.run_dir,
        initial_angles_deg=angles,
        manual_torque=float(args.manual_torque),
        auto_restart_seconds=float(args.auto_restart_seconds),
        max_wheel_torque=None if args.max_wheel_torque is None else float(args.max_wheel_torque),
        nn_torque_scale=None if args.nn_torque_scale is None else float(args.nn_torque_scale),
        seed=None if int(args.seed) < 0 else int(args.seed),
        deterministic=bool(args.deterministic),
        vary_disturbance_per_restart=bool(args.vary_disturbance_per_restart),
        enable_swingup=args.enable_swingup,
    )


if __name__ == "__main__":
    main()
