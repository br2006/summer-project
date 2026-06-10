"""NN-only fitness evaluation with balance-dominant, gated amplification reward."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

from config.settings import ProjectConfig
from neat.genome import Genome
from neat.network import FeedforwardNetwork
from signal_processing.spectral_metrics import compute_spectral_metrics
from simulation.nn_pendulum_env import NNPendulumEnv
from simulation.pendulum_env import SimulationResult


@dataclass
class NNOnlyFitnessWeights:
    balance: float = 6.0
    upright_time: float = 3.0
    handover: float = 0.8
    amplification: float = 0.8
    effort: float = 0.4
    wheel: float = 1.2
    unsafe: float = 5.0


@dataclass
class NNOnlyFitnessBreakdown:
    total: float
    balance: float
    upright_time: float
    handover: float
    stability_gate: float
    amplification: float
    gated_amplification: float
    effort: float
    wheel: float
    unsafe: float


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _first_handover_index(control_mode: np.ndarray) -> Optional[int]:
    mode = control_mode.astype(np.int8)
    transitions = np.where(np.diff(mode) > 0)[0]
    if transitions.size == 0:
        return None
    return int(transitions[0] + 1)


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def load_nn_only_fitness_weights(path: Optional[Path]) -> NNOnlyFitnessWeights:
    """Load nn_only_fitness weights from YAML; fallback to defaults if missing."""
    weights = NNOnlyFitnessWeights()
    if path is None or not path.exists():
        return weights

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    section = data.get("nn_only_fitness") or {}
    for key, value in section.items():
        if hasattr(weights, key):
            setattr(weights, key, float(value))
    return weights


def evaluate_nn_only_rollout(
    result: SimulationResult,
    weights: NNOnlyFitnessWeights,
    sample_rate_hz: float,
    target_band_hz: list,
    noise_band_hz: list,
    max_wheel_speed: float,
    max_wheel_torque: float,
) -> NNOnlyFitnessBreakdown:
    """
    NN-only objective:
      F = w_b*B + w_t*T + w_h*H + w_a*(G*A) - w_e*E - w_w*W - w_u*U
    """
    angle = _wrap_angle(result.angle)
    omega = result.angular_velocity
    wheel = result.wheel_velocity
    mode_balance = result.control_mode.astype(bool)

    has_balance_phase = bool(np.any(mode_balance))
    reward_mask = mode_balance if has_balance_phase else np.zeros_like(mode_balance, dtype=bool)
    penalty_mask = mode_balance if has_balance_phase else np.ones_like(mode_balance, dtype=bool)

    # Balance quality after handover.
    if np.any(reward_mask):
        angle_bal = angle[reward_mask]
        omega_bal = omega[reward_mask]
        angle_rms = _safe_rms(angle_bal)
        omega_rms = _safe_rms(omega_bal)
        angle_score = float(np.exp(-angle_rms / np.radians(10.0)))
        omega_score = float(np.exp(-omega_rms / 2.0))
        B = 0.7 * angle_score + 0.3 * omega_score
    else:
        angle_rms = float(np.pi)
        omega_rms = 10.0
        B = 0.0

    # Time spent upright while in BALANCE mode.
    upright_mask = np.abs(angle) < np.radians(15.0)
    T = float(np.mean(upright_mask & mode_balance))

    # Handover quality at first swing-up -> balance transition.
    handover_idx = _first_handover_index(result.control_mode)
    if handover_idx is None:
        H = 0.0
    else:
        handover_angle = abs(float(angle[handover_idx]))
        handover_omega = abs(float(omega[handover_idx]))
        handover_wheel = abs(float(wheel[handover_idx])) / max(1e-9, float(max_wheel_speed))
        H = (
            0.5 * float(np.exp(-handover_angle / np.radians(5.0)))
            + 0.3 * float(np.exp(-handover_omega / 0.5))
            + 0.2 * float(np.exp(-handover_wheel / 0.15))
        )

    # Amplification score from frequency-domain metric.
    spectral = compute_spectral_metrics(
        result.angle,
        result.seismic_input,
        sample_rate_hz,
        target_band_hz,
        noise_band_hz,
    )
    A = float(np.clip(np.log1p(spectral.amplification_ratio), 0.0, 3.0))

    # Stability gate G in [0, 1]: amplification reward only when truly stable.
    if np.any(reward_mask):
        angle_bal_abs = np.abs(angle[reward_mask])
        omega_bal_abs = np.abs(omega[reward_mask])
        wheel_bal_abs = np.abs(wheel[reward_mask])

        max_angle = float(np.max(angle_bal_abs))
        omega_bal_rms = _safe_rms(omega_bal_abs)
        wheel_sat_fraction_bal = float(
            np.mean(wheel_bal_abs > (0.9 * float(max_wheel_speed)))
        )

        g_angle = float(np.clip(1.0 - max_angle / np.radians(30.0), 0.0, 1.0))
        g_rms = float(np.clip(1.0 - angle_rms / np.radians(12.0), 0.0, 1.0))
        g_omega = float(np.clip(1.0 - omega_bal_rms / 4.0, 0.0, 1.0))
        g_wheel = float(np.clip(1.0 - wheel_sat_fraction_bal / 0.2, 0.0, 1.0))
        G = g_angle * g_rms * g_omega * g_wheel

        # Require minimum time in BALANCE mode before rewarding amplification.
        if float(np.mean(mode_balance)) < 0.35:
            G = 0.0
    else:
        G = 0.0

    gated_amp = G * A

    # Effort penalty (NN activity + torque demand).
    nn_rms = _safe_rms(result.nn_output[penalty_mask])
    torque_rms = _safe_rms(result.actual_torque[penalty_mask]) / max(1e-9, float(max_wheel_torque))
    E = 0.5 * nn_rms + 0.5 * torque_rms

    # Wheel usage/saturation penalty.
    wheel_usage_rms = _safe_rms(result.wheel_velocity[penalty_mask] / max(1e-9, float(max_wheel_speed)))
    wheel_sat_fraction = float(
        np.mean(np.abs(result.wheel_velocity[penalty_mask]) > (0.9 * float(max_wheel_speed)))
    )
    W = 0.5 * wheel_usage_rms + 2.0 * wheel_sat_fraction

    # Unsafe/fall behaviour penalty over full rollout.
    fall_threshold = np.radians(45.0)
    angle_abs_full = np.abs(angle)
    fall_fraction = float(np.mean(angle_abs_full > fall_threshold))
    max_angle_excess = max(0.0, float(np.max(angle_abs_full) - fall_threshold))
    U = 3.0 * fall_fraction + (max_angle_excess / float(fall_threshold))
    if not has_balance_phase:
        U += 1.0

    total = (
        weights.balance * B
        + weights.upright_time * T
        + weights.handover * H
        + weights.amplification * gated_amp
        - weights.effort * E
        - weights.wheel * W
        - weights.unsafe * U
    )

    return NNOnlyFitnessBreakdown(
        total=float(total),
        balance=float(B),
        upright_time=float(T),
        handover=float(H),
        stability_gate=float(G),
        amplification=float(A),
        gated_amplification=float(gated_amp),
        effort=float(E),
        wheel=float(W),
        unsafe=float(U),
    )


class NNOnlyFitnessEvaluator:
    """Evaluate genomes in the NN-only post-swing-up balance workflow."""

    def __init__(self, config: ProjectConfig, config_path: Optional[Path] = None) -> None:
        self.config = config
        self.config_path = config_path
        self.weights = load_nn_only_fitness_weights(config_path)
        self.env = NNPendulumEnv(config.simulation)
        self.generation = 0
        self._base_initial_angle = float(config.simulation.initial_angle)

    def evaluate_network(self, network) -> NNOnlyFitnessBreakdown:
        sample_rate = 1.0 / self.config.simulation.dt

        base = abs(self._base_initial_angle)
        angle_candidates = [base] if base <= 1e-12 else [base, -base]
        breakdowns: list[NNOnlyFitnessBreakdown] = []

        old_angle = float(self.env.config.initial_angle)
        try:
            for initial_angle in angle_candidates:
                self.env.config.initial_angle = float(initial_angle)
                result = self.env.run_episode(network=network)
                breakdowns.append(
                    evaluate_nn_only_rollout(
                        result=result,
                        weights=self.weights,
                        sample_rate_hz=sample_rate,
                        target_band_hz=self.config.spectral.target_band_hz,
                        noise_band_hz=self.config.spectral.noise_band_hz,
                        max_wheel_speed=self.config.simulation.max_wheel_speed,
                        max_wheel_torque=self.config.simulation.max_wheel_torque,
                    )
                )
        finally:
            self.env.config.initial_angle = old_angle

        if len(breakdowns) == 1:
            return breakdowns[0]

        return NNOnlyFitnessBreakdown(
            total=float(np.mean([b.total for b in breakdowns])),
            balance=float(np.mean([b.balance for b in breakdowns])),
            upright_time=float(np.mean([b.upright_time for b in breakdowns])),
            handover=float(np.mean([b.handover for b in breakdowns])),
            stability_gate=float(np.mean([b.stability_gate for b in breakdowns])),
            amplification=float(np.mean([b.amplification for b in breakdowns])),
            gated_amplification=float(np.mean([b.gated_amplification for b in breakdowns])),
            effort=float(np.mean([b.effort for b in breakdowns])),
            wheel=float(np.mean([b.wheel for b in breakdowns])),
            unsafe=float(np.mean([b.unsafe for b in breakdowns])),
        )

    def evaluate_genome(self, genome: Genome) -> float:
        network = FeedforwardNetwork(genome)
        breakdown = self.evaluate_network(network)
        return breakdown.total

    def make_fitness_fn(self) -> Callable[[Genome], float]:
        return self.evaluate_genome
