"""Focused tests for the IEEE-118 training defaults and scale corrections."""

import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from Neural_network.train_deterministic_tcn import (
    build_parser as build_deterministic_parser,
    resolve_objective_cost_scale,
)
from Neural_network.train_generator import (
    build_parser as build_generator_parser,
    diversity_loss,
    run_epoch,
)


class IEEE118TrainingHyperparameterTests(unittest.TestCase):
    """Verify the enlarged-system defaults and dimension-stable operations."""

    def test_generator_defaults_match_ieee118_initial_configuration(self) -> None:
        """The stochastic generator exposes the selected IEEE-118 defaults."""
        args = build_generator_parser("csng").parse_args([])
        self.assertEqual(args.data.name, "e2e118_N10000_S20_T16")
        self.assertEqual(args.stage1_epochs, 40)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.hidden_channels, 256)
        self.assertEqual(args.latent_dim, 32)
        self.assertEqual(args.latent_embedding_dim, 64)
        self.assertEqual(args.stage2_learning_rate, 5.0e-5)
        self.assertEqual(args.lambda_fea, 5.0e-2)
        self.assertEqual(args.economic_warmup_epochs, 20)
        self.assertEqual(args.patience, 50)
        self.assertEqual(args.economic_min_delta, 100.0)

    def test_diversity_kernel_is_invariant_to_repeated_dimensions(self) -> None:
        """Repeating identical controls does not collapse the diversity kernel."""
        trajectory = torch.tensor([[[[0.0]], [[1.0]]]])
        lower = torch.tensor([0.0])
        upper = torch.tensor([2.0])
        score = torch.ones(1, 2)
        base = diversity_loss(trajectory, lower, upper, score, sigma=0.5)

        repeated = trajectory.repeat(1, 1, 1, 12)
        repeated_loss = diversity_loss(
            repeated,
            lower.repeat(12),
            upper.repeat(12),
            score,
            sigma=0.5,
        )
        self.assertTrue(torch.allclose(base, repeated_loss, atol=1.0e-7))

    def test_gradient_accumulation_averages_each_update_group(self) -> None:
        """Two micro-batches produce one averaged optimizer update."""
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(
            TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)),
            batch_size=1,
            shuffle=False,
        )
        args = Namespace(gradient_accumulation_steps=2, max_grad_norm=0.0)

        metric_names = (
            "loss", "fea", "div", "eco", "candidate_violation", "soft_score",
            "feasible", "hit", "max_violation", "violation_pg", "violation_qg",
            "violation_voltage", "violation_angle", "violation_thermal",
            "violation_ramp", "cost_mean", "cost_best", "pf_residual",
        )

        def fake_compute_batch(model, completion, condition, scenario_load, *unused):
            """Return a scalar quadratic loss and zero-valued reporting metrics."""
            loss = model(condition).square().mean()
            metrics = {name: torch.zeros(()) for name in metric_names}
            metrics["loss"] = loss.detach()
            metrics["worst_violation"] = torch.zeros(())
            metrics["best_cost_sum"] = torch.zeros(())
            metrics["best_cost_count"] = torch.zeros((), dtype=torch.int64)
            return loss, metrics

        with patch("Neural_network.train_generator.compute_batch", fake_compute_batch):
            run_epoch(
                model,
                torch.nn.Identity(),
                loader,
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                args,
                torch.device("cpu"),
                optimizer,
                economic_weight=0.0,
                latent_seed=2026,
            )
        self.assertAlmostEqual(float(model.weight), 0.64, places=6)

    def test_deterministic_cost_scale_defaults_to_auto(self) -> None:
        """The deterministic benchmark freezes a calibrated IEEE-118 scale."""
        args = build_deterministic_parser().parse_args([])
        self.assertEqual(args.data.name, "e2e118_N10000_S20_T16")
        self.assertEqual(args.hidden_channels, 256)
        self.assertEqual(args.lambda_fea, 5.0e-2)
        self.assertEqual(args.objective_cost_scale, 0.0)
        self.assertEqual(resolve_objective_cost_scale(0.0, 123456.0), 123456.0)
        self.assertEqual(resolve_objective_cost_scale(8.0e5, 123456.0), 8.0e5)


if __name__ == "__main__":
    unittest.main()
