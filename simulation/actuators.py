"""
Actuator interface abstraction for reaction-wheel torque commands.

The NEAT network outputs a normalized signal in [-1, 1]. Physical torque is
applied only after external scaling (max_torque, alpha blending with PID).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ActuatorInterface(ABC):
    """Abstract actuator; implement for motor drivers or simulation."""

    @abstractmethod
    def apply_torque(self, torque_nm: float) -> None:
        raise NotImplementedError


class SimulatedActuator(ActuatorInterface):
    """Stores the latest commanded torque for the physics integrator."""

    def __init__(self) -> None:
        self.torque_nm: float = 0.0

    def apply_torque(self, torque_nm: float) -> None:
        self.torque_nm = torque_nm

    def read_torque(self) -> float:
        return self.torque_nm
