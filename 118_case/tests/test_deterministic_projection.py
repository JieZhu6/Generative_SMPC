"""Focused structural tests for deterministic SMPC feasibility projection."""

import unittest

import numpy as np
import pyomo.environ as pyo

from solve_projection_smpc_ipopt import (
    IPOPT_PATH,
    SharedTrajectorySMPCProjection,
)


@unittest.skipUnless(IPOPT_PATH.is_file(), "configured IPOPT executable is unavailable")
class DeterministicProjectionTests(unittest.TestCase):
    """Check objective replacement and physical-to-p.u. target mapping."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.projector = SharedTrajectorySMPCProjection(
            horizon=3, n_scenarios=2, ipopt_path=IPOPT_PATH,
        )

    def test_projection_is_the_only_active_objective(self) -> None:
        objectives = list(self.projector.model.component_objects(
            pyo.Objective, active=True,
        ))
        self.assertEqual([objective.name for objective in objectives], [
            "projection_objective",
        ])
        self.assertFalse(self.projector.model.objective.active)

    def test_target_mapping_and_initialization(self) -> None:
        projector = self.projector
        schedule = np.zeros((projector.horizon, projector.free_dim))
        for component in range(projector.free_dim):
            lower, upper = projector.model.u[0, component].bounds
            midpoint = 0.5 * (lower + upper)
            if component < len(projector.free_pg):
                midpoint *= projector.case.base_mva
            schedule[:, component] = midpoint

        projector.set_projection_target(schedule)
        projector.initialize()
        target = np.array([
            [pyo.value(projector.model.u_target[t, d])
             for d in projector.model.U]
            for t in projector.model.T
        ])
        expected = schedule.copy()
        expected[:, :len(projector.free_pg)] /= projector.case.base_mva
        self.assertTrue(np.allclose(target, expected))
        initialized = np.array([
            [pyo.value(projector.model.u[t, d]) for d in projector.model.U]
            for t in projector.model.T
        ])
        self.assertTrue(np.allclose(initialized, expected))


if __name__ == "__main__":
    unittest.main()
