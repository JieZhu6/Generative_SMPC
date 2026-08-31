"""Smoke-test the DECS and deterministic-TCN training graphs on real data."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import (  # noqa: E402
    ACReconstruction,
    EqualityCompletionSurrogate,
    load_decs_checkpoint,
)
from Neural_network.deterministic_tcn import DeterministicTCN  # noqa: E402
from Neural_network.train_decs import DecsDataset, compute_loss as compute_decs_loss  # noqa: E402
from Neural_network.train_deterministic_tcn import (  # noqa: E402
    build_parser as build_deterministic_parser,
    compute_batch as compute_deterministic_batch,
)
from Neural_network.train_generator import (  # noqa: E402
    GeneratorDataset,
    compute_training_feature_bounds,
)


DEFAULT_DECS_DATA = (
    ROOT / "Data_generation" / "data" / "e2e14_decs_pgm_fixedpv_N10000"
)
DEFAULT_GENERATOR_DATA = ROOT / "Data_generation" / "data" / "e2e14_N5000_S20_T16"
DEFAULT_DECS_CHECKPOINT = ROOT / "Neural_network" / "decs_pgm_fixedpv.pt"
DEFAULT_DETERMINISTIC_CHECKPOINT = ROOT / "Neural_network" / "deterministic_tcn.pt"


def require_finite_gradients(model: torch.nn.Module, name: str) -> None:
    """Raise when a trainable parameter has a missing or non-finite gradient."""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(value is None or not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError(f"{name} has missing or non-finite gradients")


def smoke_decs(data: Path, checkpoint_path: Path, device: torch.device) -> float:
    """Load the old DECS checkpoint and run one real-data training batch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = EqualityCompletionSurrogate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        output_dim=int(checkpoint["output_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    physics = ACReconstruction().to(device)
    normalization = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in checkpoint["normalization"].items()
    }
    batch = next(iter(DataLoader(DecsDataset(data, "train"), batch_size=4)))
    loss, _ = compute_decs_loss(
        model, physics, batch, normalization,
        lambda_physics=float(checkpoint["lambda_physics"]), device=device,
    )
    loss.backward()
    require_finite_gradients(model, "DECS")
    return float(loss.detach())


def smoke_deterministic_tcn(
    data: Path,
    decs_checkpoint: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> float:
    """Load the old deterministic checkpoint and run one real-data batch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DeterministicTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    completion = load_decs_checkpoint(decs_checkpoint, device)
    for parameter in completion.parameters():
        parameter.requires_grad_(False)

    train_indices = np.load(data / "split" / "train_indices.npy")
    feature_min, feature_max = compute_training_feature_bounds(data, train_indices)
    dataset = GeneratorDataset(data, train_indices[:1], feature_min, feature_max)
    condition, scenario_load = dataset[0]
    condition = condition[None].to(device)
    scenario_load = scenario_load[None, :2].to(device)

    physics = completion.physics
    free_lower, free_upper = physics.free_variable_bounds(1)
    free_lower = free_lower[0].to(device)
    free_upper = free_upper[0].to(device)
    ramp_fraction = float(checkpoint["ramp_fraction_of_pmax"])
    free_ramp = ramp_fraction * physics.pg_max[physics.free_pg]
    reference_ramp = ramp_fraction * physics.pg_max[physics.reference_generator]
    args = build_deterministic_parser().parse_args([])
    for name, value in checkpoint["loss_hyperparameters"].items():
        setattr(args, name, value)

    loss, _ = compute_deterministic_batch(
        model, completion, condition, scenario_load,
        free_lower, free_upper, free_ramp, reference_ramp, args,
    )
    loss.backward()
    require_finite_gradients(model, "deterministic TCN")
    return float(loss.detach())


def main() -> None:
    """Run both compatibility checks on CPU or CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--decs-data", type=Path, default=DEFAULT_DECS_DATA)
    parser.add_argument("--generator-data", type=Path, default=DEFAULT_GENERATOR_DATA)
    parser.add_argument("--decs", type=Path, default=DEFAULT_DECS_CHECKPOINT)
    parser.add_argument(
        "--deterministic", type=Path, default=DEFAULT_DETERMINISTIC_CHECKPOINT,
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    decs_loss = smoke_decs(args.decs_data, args.decs, device)
    deterministic_loss = smoke_deterministic_tcn(
        args.generator_data, args.decs, args.deterministic, device,
    )
    print(f"DECS: loss={decs_loss:.6e}, backward=finite")
    print(f"deterministic TCN: loss={deterministic_loss:.6e}, backward=finite")


if __name__ == "__main__":
    main()
