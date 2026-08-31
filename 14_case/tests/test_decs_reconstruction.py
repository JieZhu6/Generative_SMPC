"""Numerical equivalence tests for sparse AC reconstruction and CVaR selection."""

import unittest

import torch

from Neural_network.decs import ACReconstruction
from Neural_network.train_generator import mean_cvar_violation


def dense_nodal_injection_reference(
    physics: ACReconstruction,
    vm: torch.Tensor,
    va: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the original dense Ybus nodal-injection equations.

    Parameters
    ----------
    physics : ACReconstruction
        IEEE-14 network tensors and base-MVA scaling.
    vm, va : torch.Tensor, shape (batch, n_bus)
        Voltage magnitudes in p.u. and phase angles in rad.
    """
    angle = va[:, :, None] - va[:, None, :]
    vivj = vm[:, :, None] * vm[:, None, :]
    p = (
        vivj * (physics.g * torch.cos(angle) + physics.b * torch.sin(angle))
    ).sum(dim=2)
    q = (
        vivj * (physics.g * torch.sin(angle) - physics.b * torch.cos(angle))
    ).sum(dim=2)
    return p * physics.base_mva, q * physics.base_mva


class ACReconstructionEquivalenceTests(unittest.TestCase):
    """Check sparse branch accumulation against the original dense equations."""

    def setUp(self) -> None:
        """Create deterministic random IEEE-14 voltage states."""
        torch.manual_seed(2026)
        self.physics = ACReconstruction()
        self.vm = (0.94 + 0.12 * torch.rand(64, self.physics.n_bus)).requires_grad_()
        self.va = (
            (torch.rand(64, self.physics.n_bus) - 0.5) * 0.4
        ).requires_grad_()

    def test_sparse_nodal_values_match_dense_ybus(self) -> None:
        """Sparse branch and shunt accumulation preserves nodal powers."""
        dense = dense_nodal_injection_reference(self.physics, self.vm, self.va)
        sparse = self.physics.nodal_injection(self.vm, self.va)
        for sparse_value, dense_value in zip(sparse, dense):
            self.assertTrue(torch.allclose(
                sparse_value, dense_value, rtol=2e-6, atol=1e-3,
            ))

    def test_sparse_nodal_gradients_match_dense_ybus(self) -> None:
        """Sparse accumulation preserves voltage-state gradients numerically."""
        weight_p = torch.randn(64, self.physics.n_bus)
        weight_q = torch.randn(64, self.physics.n_bus)
        dense_p, dense_q = dense_nodal_injection_reference(
            self.physics, self.vm, self.va,
        )
        dense_grad = torch.autograd.grad(
            (dense_p * weight_p + dense_q * weight_q).sum(),
            (self.vm, self.va), retain_graph=True,
        )
        sparse_p, sparse_q = self.physics.nodal_injection(self.vm, self.va)
        sparse_grad = torch.autograd.grad(
            (sparse_p * weight_p + sparse_q * weight_q).sum(),
            (self.vm, self.va),
        )
        dense_vector = torch.cat([value.flatten() for value in dense_grad])
        difference = torch.cat([
            (sparse_value - dense_value).flatten()
            for sparse_value, dense_value in zip(sparse_grad, dense_grad)
        ])
        relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            dense_vector,
        )
        self.assertLess(float(relative_l2), 5e-6)

    def test_cached_completion_inputs_preserve_reconstruction(self) -> None:
        """Passing precomputed loads and rho returns identical reconstructed states."""
        batch = 8
        lower, upper = self.physics.free_variable_bounds(batch)
        u = lower + torch.rand_like(lower) * (upper - lower)
        load = 20.0 + 50.0 * torch.rand(batch, len(self.physics.load_buses), 2)
        chi = 0.02 * torch.randn(batch, self.physics.chi_dim)
        n_pv = len(self.physics.pv)
        n_pq = len(self.physics.pq)
        chi[:, n_pv + n_pq:] += 1.0
        direct = self.physics.reconstruct(u, load, chi)
        bus_load = self.physics.full_load(load)
        rho = self.physics._specification_from_full_load(u, *bus_load)
        cached = self.physics.reconstruct(
            u, load, chi, rho=rho, bus_load=bus_load,
        )
        self.assertEqual(set(direct), set(cached))
        for name in direct:
            self.assertTrue(torch.equal(direct[name], cached[name]), name)


class MeanCVAREquivalenceTests(unittest.TestCase):
    """Check optimized max/unsorted-top-k paths against sorted top-k."""

    def test_values_and_gradients_match_sorted_topk(self) -> None:
        """CVaR values and gradients are unchanged for representative residuals."""
        torch.manual_seed(2027)
        for width in (2, 18, 40):
            residual = torch.rand(2, 3, 4, 5, width, requires_grad=True)
            tail_count = max(1, int(torch.ceil(torch.tensor(0.1 * width))))
            reference = (
                0.1 * residual.mean(dim=-1)
                + 0.9 * residual.topk(tail_count, dim=-1).values.mean(dim=-1)
            )
            optimized = mean_cvar_violation(residual, 0.1, 0.1)
            self.assertTrue(torch.allclose(optimized, reference, rtol=1e-7, atol=1e-7))
            reference_grad = torch.autograd.grad(
                reference.sum(), residual, retain_graph=True,
            )[0]
            optimized_grad = torch.autograd.grad(optimized.sum(), residual)[0]
            self.assertTrue(torch.equal(optimized_grad, reference_grad))


if __name__ == "__main__":
    unittest.main()
