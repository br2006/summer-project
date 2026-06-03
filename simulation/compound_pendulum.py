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

    # Design-aware wheel architecture defaults (rim + spokes + hub).
    rim_mass_fraction: float = 0.72
    spoke_mass_fraction: float = 0.18
    hub_mass_fraction: float = 0.10

    def _wheel_inertia_from_design(self) -> float:
        """
        Compute wheel MOI around wheel spin axis using a rim/spokes/hub model.

        This intentionally preserves configured total wheel mass/radius/thickness,
        while replacing the old solid-disk assumption.
        """
        m = max(0.0, self.wheel_mass)
        r_outer = max(0.0, self.wheel_radius)
        t = max(0.0, self.wheel_thickness)

        if m <= 0.0 or r_outer <= 0.0:
            return 0.0

        # Geometry heuristics from available dimensions.
        rim_band = min(0.22 * r_outer, max(0.08 * r_outer, 0.35 * t))
        r_inner_rim = max(0.0, r_outer - rim_band)

        r_hub = min(0.28 * r_outer, max(0.08 * r_outer, 0.45 * t))
        if r_hub >= r_inner_rim:
            r_hub = 0.5 * r_inner_rim

        # Mass split (normalized defensively).
        frac_sum = self.rim_mass_fraction + self.spoke_mass_fraction + self.hub_mass_fraction
        if frac_sum <= 0.0:
            rim_frac, spoke_frac, hub_frac = 0.72, 0.18, 0.10
        else:
            rim_frac = self.rim_mass_fraction / frac_sum
            spoke_frac = self.spoke_mass_fraction / frac_sum
            hub_frac = self.hub_mass_fraction / frac_sum

        m_rim = m * rim_frac
        m_spokes = m * spoke_frac
        m_hub = m * hub_frac

        # Rim as annular disk.
        i_rim = 0.5 * m_rim * (r_outer**2 + r_inner_rim**2)

        # Hub as solid disk.
        i_hub = 0.5 * m_hub * (r_hub**2)

        # Spokes as radial slender rods distributed between hub and rim inner edge.
        # For one rod spanning [r1, r2]: I = m*(r1^2 + r1*r2 + r2^2)/3.
        r1 = r_hub
        r2 = max(r_hub, r_inner_rim)
        i_spokes = m_spokes * (r1**2 + r1 * r2 + r2**2) / 3.0

        return i_rim + i_hub + i_spokes

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
        inertia_cm = self._wheel_inertia_from_design()
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
