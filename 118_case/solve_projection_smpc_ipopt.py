"""Project a deterministic TCN schedule onto the shared SMPC feasible set."""

from pathlib import Path

import numpy as np
import pyomo.environ as pyo

from solve_shared_smpc_ipopt import IPOPT_PATH, SharedTrajectorySMPCAcopf


class SharedTrajectorySMPCProjection(SharedTrajectorySMPCAcopf):
    """SMPC feasibility recovery with minimum squared Euclidean distance.

    The inherited model retains every constraint of the shared-trajectory
    SMPC-ACOPF. Its economic objective is deactivated and replaced by the
    distance from the deterministic TCN schedule in the dimensionless control
    coordinates: generator active powers and voltage setpoints are both p.u.
    """

    def __init__(
        self,
        horizon: int = 16,
        n_scenarios: int = 20,
        ramp_fraction: float = 0.25,
        ipopt_path: Path = IPOPT_PATH,
        ipopt_tolerance: float = 1e-8,
        ipopt_constraint_tolerance: float = 1e-8,
    ):
        super().__init__(
            horizon=horizon,
            n_scenarios=n_scenarios,
            ramp_fraction=ramp_fraction,
            ipopt_path=ipopt_path,
            ipopt_tolerance=ipopt_tolerance,
            ipopt_constraint_tolerance=ipopt_constraint_tolerance,
        )
        m = self.model
        m.objective.deactivate()
        m.u_target = pyo.Param(m.T, m.U, mutable=True, initialize=0.0)
        m.projection_objective = pyo.Objective(expr=sum(
            (m.u[t, d] - m.u_target[t, d]) ** 2
            for t in m.T for d in m.U
        ))
        self._target_pu: np.ndarray | None = None

    @property
    def free_dim(self) -> int:
        """Number of shared active-power and voltage control variables."""
        return len(self.free_pg) + len(self.voltage_buses)

    def set_projection_target(self, schedule: np.ndarray) -> None:
        """Assign a physical TCN schedule as the projection target."""
        schedule = np.asarray(schedule, dtype=float)
        expected = (self.horizon, self.free_dim)
        if schedule.shape != expected:
            raise ValueError(f"projection target must have shape {expected}")
        if not np.isfinite(schedule).all():
            raise ValueError("projection target must contain only finite values")

        target_pu = schedule.copy()
        target_pu[:, :len(self.free_pg)] /= self.case.base_mva
        for time in range(self.horizon):
            for component in range(self.free_dim):
                self.model.u_target[time, component] = float(
                    target_pu[time, component]
                )
        self._target_pu = target_pu

    def initialize(self) -> None:
        """Initialize dependent states normally and controls at the TCN target."""
        super().initialize()
        if self._target_pu is None:
            return
        for time in range(self.horizon):
            for component in range(self.free_dim):
                variable = self.model.u[time, component]
                lower, upper = variable.bounds
                value = float(np.clip(
                    self._target_pu[time, component], lower, upper,
                ))
                variable.set_value(value)

    def solve(
        self,
        current_load: np.ndarray,
        future_load: np.ndarray,
        target_schedule: np.ndarray,
        tee: bool = False,
    ) -> dict:
        """Project one TCN schedule and return its feasible SMPC trajectory."""
        target_schedule = np.asarray(target_schedule, dtype=float)
        self.set_projection_target(target_schedule)
        result = super().solve(current_load, future_load, tee=tee)
        if not result["solved"]:
            return result

        economic_objective = result.pop("ipopt_objective")
        squared_distance = max(
            float(pyo.value(self.model.projection_objective)), 0.0,
        )
        delta = np.asarray(result["schedule"], dtype=float) - target_schedule
        result.update({
            "economic_objective": economic_objective,
            "projection_objective_squared": squared_distance,
            "projection_distance_l2_pu": float(np.sqrt(squared_distance)),
            "projection_pg_l2_mw": float(np.linalg.norm(
                delta[:, :len(self.free_pg)],
            )),
            "projection_voltage_l2_pu": float(np.linalg.norm(
                delta[:, len(self.free_pg):],
            )),
        })
        return result

