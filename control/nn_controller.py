"""NN-only torque mapping for balance control."""

from __future__ import annotations


class NNOnlyController:
    """Maps normalized NN output directly to physical wheel torque."""

    def __init__(self, nn_torque_scale: float = 0.3, max_wheel_torque: float = 0.30) -> None:
        self.nn_torque_scale = float(nn_torque_scale)
        self.max_wheel_torque = float(abs(max_wheel_torque))

    def compute_torque(self, u_nn_norm: float) -> float:
        torque = float(u_nn_norm) * self.nn_torque_scale
        max_t = self.max_wheel_torque
        if max_t > 0.0:
            torque = max(-max_t, min(max_t, torque))
        return float(torque)
