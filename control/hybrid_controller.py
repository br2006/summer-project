"""
Hybrid control law: u_total = u_PID + alpha * u_NN.

u_PID : external stabilizing controller (placeholder in this project)
u_NN  : NEAT supplementary torque (normalized)
alpha : NN authority scaling (0 = NN disabled, 1 = full supplementary authority)
"""

from __future__ import annotations

from control.pid_interface import PIDInterface


class HybridController:
    """
    Blends external PID with NEAT output and scales to physical torque.

    nn_torque_scale converts normalized NN output to Newton-metres.
    """

    def __init__(
        self,
        pid: PIDInterface,
        alpha: float = 0.3,
        nn_torque_scale: float = 0.15,
        pid_torque_scale: float = 0.4,
    ) -> None:
        self.pid = pid
        self.alpha = alpha
        self.nn_torque_scale = nn_torque_scale
        self.pid_torque_scale = pid_torque_scale

    def compute_total_torque(self, u_pid_norm: float, u_nn_norm: float) -> float:
        """
        Combine normalized control signals into physical torque.

        Future hardware: send result to SimulatedActuator / motor driver API.
        """
        u_pid_phys = u_pid_norm * self.pid_torque_scale
        u_nn_phys = u_nn_norm * self.nn_torque_scale
        return u_pid_phys + self.alpha * u_nn_phys
