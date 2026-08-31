"""Benchmark a deterministic WD-CSNG training batch and compare saved results."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import load_decs_checkpoint
from Neural_network.generator_benchmarks import build_generator_model
from Neural_network.train_generator import (
    GeneratorDataset,
    build_parser,
    compute_batch,
    compute_training_feature_bounds,
)


def flattened_gradients(model: torch.nn.Module) -> torch.Tensor:
    """Return all model gradients as one detached CPU vector.

    Parameters
    ----------
    model : torch.nn.Module
        Generator whose scalar loss has already been backpropagated.
    """
    return torch.cat([
        parameter.grad.detach().flatten().cpu()
        for parameter in model.parameters()
        if parameter.grad is not None
    ])


def compare_results(current: dict, reference: dict) -> None:
    """Print numerical differences between current and saved benchmark results.

    Parameters
    ----------
    current, reference : dict
        Benchmark payloads containing loss, metrics, gradients, timing, and memory.
    """
    metric_names = sorted(current["metrics"])
    metric_abs = {
        name: abs(current["metrics"][name] - reference["metrics"][name])
        for name in metric_names
    }
    gradient_difference = current["gradients"] - reference["gradients"]
    gradient_relative_l2 = float(
        torch.linalg.vector_norm(gradient_difference)
        / torch.linalg.vector_norm(reference["gradients"]).clamp_min(1e-30)
    )
    print(f"loss_abs_difference={abs(current['loss'] - reference['loss']):.9e}")
    print(f"metric_max_abs_difference={max(metric_abs.values()):.9e}")
    print(f"gradient_max_abs_difference={gradient_difference.abs().max():.9e}")
    print(f"gradient_relative_l2_difference={gradient_relative_l2:.9e}")
    print(f"speedup={reference['mean_step_ms'] / current['mean_step_ms']:.3f}x")
    print(
        f"allocated_memory_reduction="
        f"{reference['peak_allocated_gib'] / current['peak_allocated_gib']:.3f}x"
    )


def main() -> None:
    """Run one fixed-latent training graph repeatedly and optionally compare it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / "e2e118_N10000_S20_T16",
        help="SMPC dataset containing the saved train split",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "Neural_network" / "generator_tcn_wd_csng.pt",
        help="trained WD-CSNG checkpoint used to fix model parameters",
    )
    parser.add_argument(
        "--decs", type=Path,
        default=ROOT / "Neural_network" / "decs_pgm_fixedpv.pt",
        help="frozen DECS checkpoint used by the training graph",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="optional output path for the benchmark payload",
    )
    parser.add_argument(
        "--compare", type=Path, default=None,
        help="optional saved benchmark payload used for numerical and timing comparison",
    )
    parser.add_argument(
        "--repeats", type=int, default=20,
        help="positive number of synchronized training steps used for mean timing",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this efficiency benchmark requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = build_parser("wd_csng").parse_args([])
    training_args.use_diversity = False
    training_args.economic_cost_scale = checkpoint["loss_hyperparameters"][
        "economic_cost_scale"
    ]

    train_indices = np.load(args.data / "split" / "train_indices.npy")
    feature_min, feature_max = compute_training_feature_bounds(args.data, train_indices)
    dataset = GeneratorDataset(args.data, train_indices, feature_min, feature_max)
    loader = DataLoader(dataset, batch_size=training_args.batch_size, shuffle=False)
    condition, scenario_load = next(iter(loader))
    condition = condition.to(device)
    scenario_load = scenario_load.to(device)

    model = build_generator_model("wd_csng", **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    completion = load_decs_checkpoint(args.decs, device)
    for parameter in completion.parameters():
        parameter.requires_grad_(False)
    lower = checkpoint["free_variable_lower"].to(device)
    upper = checkpoint["free_variable_upper"].to(device)
    free_ramp = checkpoint["free_ramp_mw_per_period"].to(device)
    reference_ramp = torch.as_tensor(
        checkpoint["reference_ramp_mw_per_period"], device=device,
    )
    latent = torch.randn(
        len(condition), training_args.candidates, model.latent_dim,
        dtype=condition.dtype, device=device,
        generator=torch.Generator(device=device).manual_seed(2026),
    )

    latest_loss = None
    latest_metrics = None

    def training_step() -> None:
        """Execute the fixed-input Stage-2 forward and backward graph once."""
        nonlocal latest_loss, latest_metrics
        model.zero_grad(set_to_none=True)
        latest_loss, latest_metrics = compute_batch(
            model, completion, condition, scenario_load,
            lower, upper, free_ramp, reference_ramp,
            training_args, training_args.lambda_eco, latent,
        )
        latest_loss.backward()

    for _ in range(3):
        training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for _ in range(args.repeats):
        training_step()
    torch.cuda.synchronize()
    mean_step_ms = 1e3 * (perf_counter() - start) / args.repeats
    if latest_loss is None or latest_metrics is None:
        raise RuntimeError("benchmark did not execute a training step")

    result = {
        "device": torch.cuda.get_device_name(0),
        "batch_size": len(condition),
        "candidates": training_args.candidates,
        "scenarios": dataset.n_scenarios,
        "horizon": dataset.horizon,
        "loss": float(latest_loss.detach()),
        "metrics": {
            name: float(value) for name, value in latest_metrics.items()
        },
        "gradients": flattened_gradients(model),
        "mean_step_ms": mean_step_ms,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    print(
        f"device={result['device']} | B={result['batch_size']} "
        f"K={result['candidates']} S={result['scenarios']} T={result['horizon']}"
    )
    print(f"loss={result['loss']:.9e}")
    print(f"mean_step={result['mean_step_ms']:.3f} ms")
    print(
        f"peak_allocated={result['peak_allocated_gib']:.3f} GiB | "
        f"peak_reserved={result['peak_reserved_gib']:.3f} GiB"
    )
    if args.compare is not None:
        compare_results(result, torch.load(args.compare, weights_only=False))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.save)
        print(f"saved={args.save}")


if __name__ == "__main__":
    main()
