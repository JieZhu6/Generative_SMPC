"""Run one real-data forward/backward batch for S-CSNG and WD-CSNG."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import load_decs_checkpoint
from Neural_network.generator_benchmarks import (
    build_generator_model,
    get_benchmark_spec,
)
from Neural_network.train_generator import (
    GeneratorDataset,
    build_parser,
    compute_batch,
    compute_training_feature_bounds,
)


DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e14_N5000_S20_T16"
DEFAULT_DECS = ROOT / "Neural_network" / "decs_pgm_fixedpv.pt"


def run_benchmark_batch(
    benchmark: str,
    completion: torch.nn.Module,
    condition: torch.Tensor,
    scenario_load: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate both training stages for one benchmark and backpropagate.

    Parameters
    ----------
    benchmark : str
        ``s_csng`` or ``wd_csng``.
    completion : torch.nn.Module
        Frozen differentiable equality-completion surrogate.
    condition : torch.Tensor, shape (1,T,condition_dim)
        One normalized pooled uncertainty sequence.
    scenario_load : torch.Tensor, shape (1,S,T,n_load,2)
        Physical loads for the smoke scenarios.
    device : torch.device
        CPU or CUDA device used by all tensors and models.

    Returns
    -------
    dict[str, float]
        Stage-1 loss, Stage-2 loss, and Stage-1 diversity metric.
    """
    spec = get_benchmark_spec(benchmark)
    physics = completion.physics
    free_lower, free_upper = physics.free_variable_bounds(1)
    free_lower = free_lower[0].to(device)
    free_upper = free_upper[0].to(device)
    free_ramp = 0.25 * physics.pg_max[physics.free_pg]
    reference_ramp = 0.25 * physics.pg_max[physics.reference_generator]
    model = build_generator_model(
        benchmark,
        condition_dim=condition.shape[-1],
        free_dim=physics.free_dim,
        n_free_pg=physics.n_free_pg,
        projection_method=spec["projection_method"],
        hidden_channels=16,
        latent_dim=4,
        latent_embedding_dim=8,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
    ).to(device)

    args = build_parser(benchmark).parse_args([])
    args.use_diversity = bool(spec["diversity_loss_enabled"])
    args.candidates = 2
    args.economic_top_k = 2
    args.economic_cost_scale = 4.0e4
    latent = torch.randn(
        1, args.candidates, model.latent_dim,
        dtype=condition.dtype, device=device,
        generator=torch.Generator(device=device).manual_seed(2026),
    )

    stage1_loss, stage1_metrics = compute_batch(
        model, completion, condition, scenario_load,
        free_lower, free_upper, free_ramp, reference_ramp,
        args, economic_weight=0.0, latent=latent,
    )
    stage1_loss.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError(f"{benchmark} produced non-finite Stage-1 gradients")
    model.zero_grad(set_to_none=True)

    stage2_loss, _ = compute_batch(
        model, completion, condition, scenario_load,
        free_lower, free_upper, free_ramp, reference_ramp,
        args, economic_weight=args.lambda_eco, latent=latent,
    )
    stage2_loss.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError(f"{benchmark} produced non-finite Stage-2 gradients")
    if benchmark == "wd_csng" and float(stage1_metrics["div"]) != 0.0:
        raise AssertionError("WD-CSNG smoke batch unexpectedly computed diversity loss")
    return {
        "stage1_loss": float(stage1_loss.detach()),
        "stage2_loss": float(stage2_loss.detach()),
        "stage1_diversity": float(stage1_metrics["div"]),
    }


def main() -> None:
    """Load one held-in instance and smoke-test both benchmark training graphs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="base SMPC dataset containing the saved training split",
    )
    parser.add_argument(
        "--decs", type=Path, default=DEFAULT_DECS,
        help="frozen differentiable equality-completion checkpoint",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="device used for the smoke forward/backward passes",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    train_indices = np.load(args.data / "split" / "train_indices.npy")
    feature_min, feature_max = compute_training_feature_bounds(args.data, train_indices)
    dataset = GeneratorDataset(
        args.data, train_indices[:1], feature_min, feature_max,
    )
    condition, scenario_load = dataset[0]
    condition = condition[None].to(device)
    # Two scenarios exercise scenario broadcasting while keeping the smoke run fast.
    scenario_load = scenario_load[None, :2].to(device)
    completion = load_decs_checkpoint(args.decs, device)
    for parameter in completion.parameters():
        parameter.requires_grad_(False)

    for benchmark in ("s_csng", "wd_csng"):
        metrics = run_benchmark_batch(
            benchmark, completion, condition, scenario_load, device,
        )
        print(
            f"{benchmark}: stage1={metrics['stage1_loss']:.6e}, "
            f"stage2={metrics['stage2_loss']:.6e}, "
            f"Ldiv={metrics['stage1_diversity']:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
