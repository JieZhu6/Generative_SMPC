"""Focused tests for the S-CSNG and WD-CSNG benchmark definitions."""

import tempfile
import unittest
from pathlib import Path

import torch

from Neural_network.generator_benchmarks import (
    build_generator_model,
    get_benchmark_spec,
)
from Neural_network.noncausal_tcn_s_csng import SIGMOID_PROJECTION_METHOD
from Neural_network.train_generator import build_parser, combine_generator_losses
from evaluate_generator_tcn import load_generator


class GeneratorBenchmarkTests(unittest.TestCase):
    """Verify the two ablations differ from CSNG only as intended."""

    @staticmethod
    def model_config(benchmark: str) -> dict[str, object]:
        """Return a small valid model configuration for ``benchmark``."""
        spec = get_benchmark_spec(benchmark)
        return {
            "condition_dim": 6,
            "free_dim": 4,
            "n_free_pg": 2,
            "projection_method": spec["projection_method"],
            "hidden_channels": 8,
            "latent_dim": 3,
            "latent_embedding_dim": 4,
            "kernel_size": 3,
            "dilations": [1, 2],
        }

    def test_sigmoid_projection_and_initialization(self) -> None:
        """S-CSNG uses sigmoid coordinates centered at one half."""
        model = build_generator_model("s_csng", **self.model_config("s_csng"))
        self.assertEqual(model.projection_method, SIGMOID_PROJECTION_METHOD)
        self.assertTrue(torch.equal(model.output_mapping.bias, torch.zeros(4)))

        raw = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        projected = model.project_unit_interval(raw)
        self.assertTrue(torch.all((projected > 0.0) & (projected < 1.0)))
        self.assertAlmostEqual(float(projected[1]), 0.5, places=7)
        projected.sum().backward()
        expected_gradient = projected.detach() * (1.0 - projected.detach())
        self.assertTrue(torch.allclose(raw.grad, expected_gradient))

    def test_sigmoid_model_preserves_bounds_and_ramps(self) -> None:
        """Recursive physical scaling remains admissible under sigmoid output."""
        model = build_generator_model("s_csng", **self.model_config("s_csng"))
        raw = torch.randn(2, 3, 5, 4)
        lower = torch.tensor([0.0, 10.0, 0.90, 0.95])
        upper = torch.tensor([100.0, 80.0, 1.10, 1.05])
        ramp = torch.tensor([8.0, 6.0])
        trajectory, dynamic_lower, dynamic_upper = model.enforce_admissible_output(
            raw, lower, upper, ramp, ramp,
        )
        self.assertEqual(tuple(trajectory.shape), (2, 3, 5, 4))
        self.assertTrue(torch.all(trajectory >= dynamic_lower))
        self.assertTrue(torch.all(trajectory <= dynamic_upper))
        movement = trajectory[:, :, 1:, :2] - trajectory[:, :, :-1, :2]
        self.assertTrue(torch.all(movement.abs() <= ramp + 1.0e-6))

    def test_wd_objective_excludes_diversity_graph(self) -> None:
        """WD-CSNG omits Ldiv rather than multiplying it by zero."""
        feasibility = torch.tensor(2.0, requires_grad=True)
        diversity = torch.tensor(3.0, requires_grad=True)
        economic = torch.tensor(5.0, requires_grad=True)
        total = combine_generator_losses(
            feasibility, diversity, economic,
            feasibility_weight=0.2,
            diversity_weight=99.0,
            economic_weight=0.4,
            use_diversity=False,
        )
        self.assertAlmostEqual(float(total), 2.4, places=6)
        total.backward()
        self.assertAlmostEqual(float(feasibility.grad), 0.2, places=7)
        self.assertIsNone(diversity.grad)
        self.assertAlmostEqual(float(economic.grad), 0.4, places=7)

    def test_wd_cli_has_no_diversity_hyperparameters(self) -> None:
        """WD-CSNG does not expose inactive diversity settings."""
        parser = build_parser("wd_csng")
        help_text = parser.format_help()
        self.assertNotIn("--lambda-div", help_text)
        self.assertNotIn("--sigma-div", help_text)
        self.assertNotIn("--loss-epsilon", help_text)
        args = parser.parse_args([])
        self.assertEqual(args.lambda_div, 0.0)

    def test_checkpoint_loader_rejects_cross_benchmark_use(self) -> None:
        """Inference accepts the matching model and rejects a mixed method."""
        benchmark = "s_csng"
        spec = get_benchmark_spec(benchmark)
        config = self.model_config(benchmark)
        model = build_generator_model(benchmark, **config)
        checkpoint = {
            "benchmark": benchmark,
            "method": spec["method"],
            "model_class": spec["model_class"],
            "model_config": model.configuration(),
            "model_state": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s_csng.pt"
            torch.save(checkpoint, path)
            loaded, _ = load_generator(path, torch.device("cpu"), benchmark)
            self.assertEqual(loaded.projection_method, SIGMOID_PROJECTION_METHOD)
            with self.assertRaisesRegex(ValueError, "checkpoint benchmark"):
                load_generator(path, torch.device("cpu"), "wd_csng")


if __name__ == "__main__":
    unittest.main()
