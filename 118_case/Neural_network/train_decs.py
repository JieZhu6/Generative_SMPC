"""Train the fixed-PV differentiable equality completion surrogate (DECS).

The IEEE-118 mapping is ``rho[235] -> chi[181]``, where
``chi=[theta_PV,theta_PQ,V_PQ]``. PV/reference voltage magnitudes are supplied
by projected controls. Training combines normalized supervision and the AC
power-balance residual; there is no PV/PQ mode target or switching loss.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import ACReconstruction, EqualityCompletionSurrogate  # noqa: E402


def format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class DecsDataset(Dataset):
    """Memory-map one split of the fixed-PV DECS dataset."""

    def __init__(self, data_dir: Path, split: str):
        self.rho = np.load(data_dir / "rho.npy", mmap_mode="r")
        self.chi = np.load(data_dir / "chi.npy", mmap_mode="r")
        self.u = np.load(data_dir / "u.npy", mmap_mode="r")
        self.load = np.load(data_dir / "load.npy", mmap_mode="r")
        self.indices = np.load(data_dir / "split" / f"{split}_indices.npy")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = int(self.indices[index])
        return (
            torch.tensor(self.rho[sample], dtype=torch.float32),
            torch.tensor(self.chi[sample], dtype=torch.float32),
            torch.tensor(self.u[sample], dtype=torch.float32),
            torch.tensor(self.load[sample], dtype=torch.float32),
        )


def compute_loss(
    model: EqualityCompletionSurrogate,
    physics: ACReconstruction,
    batch: tuple[torch.Tensor, ...],
    normalization: dict[str, torch.Tensor],
    lambda_physics: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate normalized supervision plus fixed-PV AC-balance loss."""
    rho, chi, u, load = [value.to(device) for value in batch]
    rho_normalized = (rho - normalization["rho_mean"]) / normalization["rho_std"]
    chi_normalized = (chi - normalization["chi_mean"]) / normalization["chi_std"]
    prediction_normalized = model(rho_normalized)
    prediction = (
        prediction_normalized * normalization["chi_std"] + normalization["chi_mean"]
    )

    supervised = nn.functional.mse_loss(prediction_normalized, chi_normalized)
    state = physics.reconstruct(u, load, prediction)
    physics_loss = state["balance_residual"].square().mean()
    loss = supervised + lambda_physics * physics_loss

    n_angle = len(physics.pv) + len(physics.pq)
    return loss, {
        "loss": loss.detach(),
        "supervised": supervised.detach(),
        "physics": physics_loss.detach(),
        "angle_mae": (
            prediction[:, :n_angle] - chi[:, :n_angle]
        ).abs().mean().detach(),
        "voltage_mae": (
            prediction[:, n_angle:] - chi[:, n_angle:]
        ).abs().mean().detach(),
        "pf_residual": state["pf_residual"].mean().detach(),
    }


def run_epoch(
    model: EqualityCompletionSurrogate,
    physics: ACReconstruction,
    loader: DataLoader,
    normalization: dict[str, torch.Tensor],
    lambda_physics: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Run one train/evaluation epoch and return sample-weighted metrics."""
    training = optimizer is not None
    model.train(training)
    names = ("loss", "supervised", "physics", "angle_mae", "voltage_mae", "pf_residual")
    totals = {name: 0.0 for name in names}
    seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            loss, metrics = compute_loss(
                model, physics, batch, normalization, lambda_physics, device,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            count = len(batch[0])
            seen += count
            for name in names:
                totals[name] += float(metrics[name]) * count
    return {name: value / seen for name, value in totals.items()}


def detailed_metrics(
    model: EqualityCompletionSurrogate,
    physics: ACReconstruction,
    loader: DataLoader,
    normalization: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float | int]:
    """Compute final error distribution for one held-out split."""
    n_angle = len(physics.pv) + len(physics.pq)
    angle_sum = voltage_sum = 0.0
    residuals = []
    count = 0
    model.eval()
    with torch.no_grad():
        for rho, chi, u, load in loader:
            rho, chi, u, load = (
                rho.to(device), chi.to(device), u.to(device), load.to(device),
            )
            prediction = model(
                (rho - normalization["rho_mean"]) / normalization["rho_std"]
            ) * normalization["chi_std"] + normalization["chi_mean"]
            state = physics.reconstruct(u, load, prediction)
            count += len(rho)
            angle_sum += float((prediction[:, :n_angle] - chi[:, :n_angle]).abs().mean(dim=1).sum())
            voltage_sum += float((prediction[:, n_angle:] - chi[:, n_angle:]).abs().mean(dim=1).sum())
            residuals.append(state["pf_residual"].cpu())
    residual = torch.cat(residuals)
    return {
        "count": count,
        "angle_mae_rad": angle_sum / count,
        "voltage_mae_pu": voltage_sum / count,
        "pf_residual_mean_pu": float(residual.mean()),
        "pf_residual_p95_pu": float(torch.quantile(residual, 0.95)),
        "pf_residual_max_pu": float(residual.max()),
    }


def main() -> None:
    """Train on train, select on validation, then report test once."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / "e2e118_decs_pgm_fixedpv_N50000",
        help="IEEE-118 fixed-PV DECS dataset generated by native PGM",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "decs_pgm_fixedpv.pt",
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs=2, default=(256, 256),
        metavar=("H1", "H2"), help="two IEEE-118 DECS hidden-layer widths",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-physics", type=float, default=1.0,
        help="weight of the normalized AC power-balance MSE",
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.patience, args.print_every) < 1:
        raise ValueError("epochs, batch size, patience, and print interval must be positive")
    if args.learning_rate <= 0.0 or min(args.lambda_physics, args.min_delta) < 0.0:
        raise ValueError("learning rate must be positive; loss weight and min_delta nonnegative")
    if any(width < 1 for width in args.hidden_dims) or args.num_workers < 0:
        raise ValueError("hidden widths must be positive and num_workers nonnegative")
    if not args.data.is_dir():
        raise FileNotFoundError(f"DECS dataset not found: {args.data}")
    data_metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    physics = ACReconstruction()
    if not (
        data_metadata.get("case_name") == "pglib_opf_case118_ieee"
        and int(data_metadata.get("rho_dim", -1)) == physics.rho_dim
        and data_metadata.get("bus_type_model") == "fixed_PV_PQ"
        and data_metadata.get("enforce_q_limits") is False
        and data_metadata.get("pv_to_pq_switching") is False
        and int(data_metadata.get("chi_dim", -1)) == physics.chi_dim
    ):
        raise ValueError(
            "DECS data must contain IEEE-118 fixed-PV labels without PV/PQ switching"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    datasets = {split: DecsDataset(args.data, split) for split in ("train", "validation", "test")}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for split, dataset in datasets.items()
    }
    parameters = np.load(args.data / "normalization_parameters.npz")
    normalization = {
        name: torch.tensor(parameters[name], dtype=torch.float32, device=device)
        for name in ("rho_mean", "rho_std", "chi_mean", "chi_std")
    }
    physics = physics.to(device)
    if datasets["train"].rho.shape[1] != physics.rho_dim:
        raise ValueError("DECS rho width does not match ACReconstruction")
    if datasets["train"].chi.shape[1] != physics.chi_dim:
        raise ValueError(f"fixed-PV DECS chi width must be {physics.chi_dim}")
    if (
        normalization["rho_mean"].numel() != physics.rho_dim
        or normalization["chi_mean"].numel() != physics.chi_dim
    ):
        raise ValueError("normalization dimensions do not match fixed-PV DECS data")

    model = EqualityCompletionSurrogate(
        input_dim=physics.rho_dim,
        hidden_dims=tuple(args.hidden_dims),
        output_dim=physics.chi_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Fixed-PV DECS training configuration")
    print(f"  device: {device}")
    print(f"  samples: train={len(datasets['train'])}, validation={len(datasets['validation'])}, test={len(datasets['test'])}")
    print(f"  mapping: rho_dim={physics.rho_dim} -> chi_dim={physics.chi_dim}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, learning_rate={args.learning_rate:.3e}, lambda_physics={args.lambda_physics:g}")
    print("  checkpoint selection and early stopping use validation only; test runs once after selection.")

    best_loss = np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start_time = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, physics, loaders["train"], normalization,
            args.lambda_physics, device, optimizer,
        )
        validation_metrics = run_epoch(
            model, physics, loaders["validation"], normalization,
            args.lambda_physics, device, optimizer=None,
        )
        improved = validation_metrics["loss"] < best_loss - args.min_delta
        if improved:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        })
        if epoch == 1 or epoch % args.print_every == 0 or stale_epochs == args.patience:
            elapsed = perf_counter() - start_time
            eta = elapsed / epoch * (args.epochs - epoch)
            status = "new best" if improved else f"stale {stale_epochs}/{args.patience}"
            print(
                f"[epoch {epoch:4d}/{args.epochs}] elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)} | {status}", flush=True,
            )
            print(
                f"  train: total={train_metrics['loss']:.6e} "
                f"supervised={train_metrics['supervised']:.3e} "
                f"AC={train_metrics['physics']:.3e}", flush=True,
            )
            print(
                f"  validation: total={validation_metrics['loss']:.6e} "
                f"supervised={validation_metrics['supervised']:.3e} "
                f"AC={validation_metrics['physics']:.3e} "
                f"PF={validation_metrics['pf_residual']:.3e} p.u.", flush=True,
            )
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    test_loss = run_epoch(
        model, physics, loaders["test"], normalization,
        args.lambda_physics, device, optimizer=None,
    )
    test_report = detailed_metrics(
        model, physics, loaders["test"], normalization, device,
    )
    print(
        f"best-checkpoint test: total={test_loss['loss']:.6e}, "
        f"theta_MAE={test_report['angle_mae_rad']:.3e} rad, "
        f"V_MAE={test_report['voltage_mae_pu']:.3e} p.u., "
        f"PF_p95={test_report['pf_residual_p95_pu']:.3e} p.u."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "input_dim": physics.rho_dim,
        "hidden_dims": list(args.hidden_dims),
        "output_dim": physics.chi_dim,
        "label_definition": ["theta_PV", "theta_PQ", "V_PQ"],
        "bus_type_model": "fixed_PV_PQ",
        "pv_to_pq_supported": False,
        "normalization": {
            name: value.detach().cpu().numpy()
            for name, value in normalization.items()
        },
        "lambda_physics": args.lambda_physics,
        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta,
            "best_epoch": best_epoch,
            "stopped_epoch": history[-1]["epoch"],
        },
        "best_validation_loss": best_loss,
        "test_loss": test_loss,
        "training_history": history,
        "test_metrics": test_report,
        "data_metadata": data_metadata,
        "seed": args.seed,
    }, args.output)
    print(f"saved fixed-PV DECS checkpoint to {args.output}")


if __name__ == "__main__":
    main()
