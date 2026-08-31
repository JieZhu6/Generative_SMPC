"""Structural and dimensional checks for the local IEEE-118 experiment."""

import unittest

import numpy as np

from Data_generation.case118_pglib import load_case118
from Neural_network.decs import ACReconstruction


class Case118StructureTests(unittest.TestCase):
    """Guard the network partitions that determine every learned dimension."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the case and differentiable reconstruction once for all tests."""
        cls.case = load_case118()
        cls.physics = ACReconstruction()

    def test_pglib_structure(self) -> None:
        """Match the online IEEE-118 buses, generators, branches, and partitions."""
        self.assertEqual(self.case.name, "pglib_opf_case118_ieee")
        self.assertEqual(self.case.n_bus, 118)
        self.assertEqual(self.case.n_gen, 54)
        self.assertEqual(len(self.case.branch), 186)
        self.assertEqual(len(self.case.load_buses), 99)
        self.assertEqual(len(self.case.pv_buses), 53)
        self.assertEqual(len(self.case.pq_buses), 64)
        self.assertEqual(self.case.reference_bus, 68)
        self.assertEqual(len(self.case.active_generators), 19)
        self.assertEqual(len(self.case.nonreference_active_generators), 18)
        self.assertEqual(len(np.unique(self.case.generator_buses)), 54)

    def test_learning_dimensions(self) -> None:
        """Match the paper-aligned free, DECS input, and DECS output widths."""
        self.assertEqual(self.physics.n_free_pg, 18)
        self.assertEqual(self.physics.free_dim, 72)
        self.assertEqual(self.physics.rho_dim, 235)
        self.assertEqual(self.physics.chi_dim, 181)
        self.assertEqual(3 * len(self.case.load_buses) * 2, 594)

    def test_admittance_dimensions_and_finiteness(self) -> None:
        """Ensure all parsed network admittances are finite and dimensionally valid."""
        self.assertEqual(self.case.ybus.shape, (118, 118))
        for values in (
            self.case.ybus,
            self.case.yff,
            self.case.yft,
            self.case.ytf,
            self.case.ytt,
        ):
            self.assertTrue(np.isfinite(values.real).all())
            self.assertTrue(np.isfinite(values.imag).all())


if __name__ == "__main__":
    unittest.main()
