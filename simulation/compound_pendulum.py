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

    # Wheel geometry model (matching printed design):
    # - outer annular rim
    # - central hub disk
    # - N annular-sector spokes between hub and rim
    wheel_rim_width: float = 0.010
    wheel_hub_radius: float = 0.020
    wheel_spoke_count: int = 2
    wheel_spoke_coverage: float = 0.50

    gravity: float = 9.81

    def _wheel_inertia_from_design(self) -> float:
        """
        Compute wheel MOI about the wheel COM spin axis from explicit wheel geometry.

        The model assumes a uniform-density printed wheel made of:
        1) outer rim annulus [r_rim_inner, r_outer]
        2) hub disk [0, r_hub]
        3) identical annular-sector spokes on [r_hub, r_rim_inner]
        """
        m = max(0.0, self.wheel_mass)
        r_outer = max(1e-9, self.wheel_radius)
        t = max(0.0, self.wheel_thickness)
        rim_width = max(0.0, self.wheel_rim_width)
        r_hub = max(0.0, self.wheel_hub_radius)
        n_spokes = max(1, int(self.wheel_spoke_count))
        coverage = min(1.0, max(0.0, float(self.wheel_spoke_coverage)))

        if m <= 0.0 or t <= 0.0:
            return 0.0

        r_rim_inner = max(0.0, r_outer - rim_width)
        if r_hub >= r_rim_inner:
            # Defensive fallback for invalid geometry.
            r_hub = 0.6 * r_rim_inner

        # Areas in wheel plane.
        area_rim = pi * max(0.0, r_outer**2 - r_rim_inner**2)
        area_hub = pi * max(0.0, r_hub**2)
        area_spoke_region = pi * max(0.0, r_rim_inner**2 - r_hub**2)
        area_spokes_total = coverage * area_spoke_region

        # Convert to volume and infer a single density from total wheel mass.
        vol_rim = area_rim * t
        vol_hub = area_hub * t
        vol_spokes = area_spokes_total * t
        total_vol = vol_rim + vol_hub + vol_spokes
        if total_vol <= 1e-12:
            return 0.0

        density = m / total_vol
        m_rim = density * vol_rim
        m_hub = density * vol_hub
        m_spokes_total = density * vol_spokes

        # Rim as annular disk.
        i_rim = 0.5 * m_rim * (r_outer**2 + r_rim_inner**2)

        # Hub as solid disk.
        i_hub = 0.5 * m_hub * (r_hub**2)

        # Spokes as annular sectors; each spoke gets equal mass.
        # For any annular sector with uniform density around center:
        #   I_z = (1/4) * m * (r_inner^2 + r_outer^2)
        # independent of sector angle.
        m_per_spoke = m_spokes_total / n_spokes
        i_per_spoke = 0.25 * m_per_spoke * (r_hub**2 + r_rim_inner**2)
        i_spokes = n_spokes * i_per_spoke

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
            "wheel_model_rim_width": self.wheel_rim_width,
            "wheel_model_hub_radius": self.wheel_hub_radius,
            "wheel_model_spoke_count": float(self.wheel_spoke_count),
            "wheel_model_spoke_coverage": self.wheel_spoke_coverage,
        }
