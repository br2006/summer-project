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
        max_wheel_torque: float = 0.30,
    ) -> None:
        self.pid = pid
        self.alpha = alpha
        self.nn_torque_scale = nn_torque_scale
        self.pid_torque_scale = pid_torque_scale
        self.max_wheel_torque = max_wheel_torque

    def compute_total_torque(self, u_pid_norm: float, u_nn_norm: float) -> float:
        """
        Combine normalized control signals into physical torque.

        Future hardware: send result to SimulatedActuator / motor driver API.
        """
        u_pid_phys = float(u_pid_norm * self.pid_torque_scale)
        u_nn_phys = float(u_nn_norm * self.nn_torque_scale)
        nn_contrib = float(self.alpha * u_nn_phys)

        # Default blend: PID + NN.
        torque = u_pid_phys + nn_contrib
        max_t = float(abs(self.max_wheel_torque))

        # If blend exceeds actuator limits, reduce PID first while preserving NN authority.
        # This avoids NN+PID summation commanding unreachable torque.
        if max_t > 0.0:
            if torque > max_t:
                # Need to remove positive excess from PID (without inverting PID sign).
                excess = torque - max_t
                if u_pid_phys > 0.0:
                    u_pid_phys = max(0.0, u_pid_phys - excess)
                torque = u_pid_phys + nn_contrib
            elif torque < -max_t:
                # Need to remove negative excess from PID (without inverting PID sign).
                excess = -max_t - torque
                if u_pid_phys < 0.0:
                    u_pid_phys = min(0.0, u_pid_phys + excess)
                torque = u_pid_phys + nn_contrib

            # Final hard safety clamp for edge cases where NN alone exceeds limits.
            torque = max(-max_t, min(max_t, torque))

        return float(torque)
