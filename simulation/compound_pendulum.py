"""Compound pendulum mass-property calculations for rigid assemblies."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True)
class ComponentMassProperties:
    name: str
    mass: float
    com_distance: float
    inertia_cm: float

    @property
    def inertia_about_pivot(self) -> float:
        return self.inertia_cm + self.mass * (self.com_distance ** 2)


@dataclass
class CompoundPendulumProperties:
    """Computes rigid-body pendulum properties from component geometry."""

    arm_mass: float
    arm_length: float

    motor_mass: float
    motor_radius: float
    motor_length: float

    wheel_mass: float
    wheel_radius: float
    wheel_thickness: float

    gravity: float = 9.81

    def _arm_component(self) -> ComponentMassProperties:
        inertia_cm = (1.0 / 12.0) * self.arm_mass * (self.arm_length ** 2)
        return ComponentMassProperties(
            name="arm",
            mass=self.arm_mass,
            com_distance=self.arm_length / 2.0,
            inertia_cm=inertia_cm,
        )

    def _motor_component(self) -> ComponentMassProperties:
        inertia_cm = (1.0 / 12.0) * self.motor_mass * (
            3.0 * (self.motor_radius ** 2) + (self.motor_length ** 2)
        )
        return ComponentMassProperties(
            name="motor",
            mass=self.motor_mass,
            com_distance=self.arm_length,
            inertia_cm=inertia_cm,
        )

    def _wheel_component(self) -> ComponentMassProperties:
        inertia_cm = 0.5 * self.wheel_mass * (self.wheel_radius ** 2)
        return ComponentMassProperties(
            name="wheel",
            mass=self.wheel_mass,
            com_distance=self.arm_length,
            inertia_cm=inertia_cm,
        )

    def components(self) -> tuple[ComponentMassProperties, ...]:
        return (
            self._arm_component(),
            self._motor_component(),
            self._wheel_component(),
        )

    def compute_total_mass(self) -> float:
        return sum(c.mass for c in self.components())

    def compute_center_of_mass(self) -> float:
        total_mass = self.compute_total_mass()
        if total_mass <= 0.0:
            return 0.0
        return sum(c.mass * c.com_distance for c in self.components()) / total_mass

    def compute_total_inertia(self) -> float:
        return sum(c.inertia_about_pivot for c in self.components())

    def compute_natural_frequency(self) -> float:
        total_mass = self.compute_total_mass()
        r_com = self.compute_center_of_mass()
        inertia = self.compute_total_inertia()
        if inertia <= 0.0 or total_mass <= 0.0 or r_com <= 0.0:
            return 0.0
        return sqrt((total_mass * self.gravity * r_com) / inertia)

    def compute_period(self) -> float:
        total_mass = self.compute_total_mass()
        r_com = self.compute_center_of_mass()
        inertia = self.compute_total_inertia()
        denom = total_mass * self.gravity * r_com
        if denom <= 0.0 or inertia <= 0.0:
            return float("inf")
        return 2.0 * pi * sqrt(inertia / denom)

    def debug_summary(self) -> dict[str, float]:
        arm, motor, wheel = self.components()
        total_mass = self.compute_total_mass()
        com = self.compute_center_of_mass()
        total_inertia = self.compute_total_inertia()
        natural_frequency = self.compute_natural_frequency()
        period = self.compute_period()

        return {
            "arm_inertia_about_pivot": arm.inertia_about_pivot,
            "motor_inertia_about_pivot": motor.inertia_about_pivot,
            "wheel_inertia_about_pivot": wheel.inertia_about_pivot,
            "total_mass": total_mass,
            "center_of_mass": com,
            "total_inertia_about_pivot": total_inertia,
            "natural_frequency": natural_frequency,
            "oscillation_period": period,
            "wheel_inertia_cm": wheel.inertia_cm,
        }
