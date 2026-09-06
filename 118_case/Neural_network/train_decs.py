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


RECONSTRUCTION_GROUPS = (
    "qg_pv",
    "slack_pg",
    "slack_qg",
    "branch_p",
    "branch_q",
)


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


def reconstruction_outputs(
    state: dict[str, torch.Tensor],
    physics: ACReconstruction,
) -> dict[str, torch.Tensor]:
    """Select the dependent PF variables supervised by reconstruction loss.

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        Output of :meth:`ACReconstruction.reconstruct` for one batch.
    physics : ACReconstruction
        Physics module supplying nonreference and reference generator indices.

    Returns
    -------
    dict[str, torch.Tensor]
        Nonreference PV-generator reactive powers, reference-generator active
        and reactive powers, and both-end branch active/reactive flows. Power
        quantities are in MW or Mvar.
    """
    reference = physics.reference_generator
    return {
        "qg_pv": state["qg"][:, physics.free_pg],
        "slack_pg": state["pg"][:, reference:reference + 1],
        "slack_qg": state["qg"][:, reference:reference + 1],
        "branch_p": torch.cat([state["pf"], state["pt"]], dim=1),
        "branch_q": torch.cat([state["qf"], state["qt"]], dim=1),
    }


def estimate_reconstruction_scales(
    physics: ACReconstruction,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Estimate one training-set RMS normalization scale per output group.

    Parameters
    ----------
    physics : ACReconstruction
        Shared differentiable AC reconstruction module.
    loader : DataLoader
        Non-shuffled training-split loader yielding ``rho, chi, u, load``.
    device : torch.device
        Device used for reconstruction and accumulation.

    Returns
    -------
    dict[str, torch.Tensor]
        Positive scalar RMS scales for the five reconstruction groups.
    """
    square_sums = {
        name: torch.zeros((), dtype=torch.float64, device=device)
        for name in RECONSTRUCTION_GROUPS
    }
    counts = {name: 0 for name in RECONSTRUCTION_GROUPS}
    with torch.no_grad():
        for _, chi, u, load in loader:
            chi, u, load = chi.to(device), u.to(device), load.to(device)
            outputs = reconstruction_outputs(
                physics.reconstruct(u, load, chi), physics,
            )
            for name, value in outputs.items():
                square_sums[name] += value.double().square().sum()
                counts[name] += value.numel()
    if min(counts.values()) < 1:
        raise ValueError("training loader must contain reconstruction targets")
    scales = {
        name: torch.sqrt(square_sums[name] / counts[name])
        .clamp_min(1e-6)
        .float()
        for name in RECONSTRUCTION_GROUPS
    }
    if not all(bool(torch.isfinite(value)) for value in scales.values()):
        raise FloatingPointError("non-finite reconstruction normalization scale")
    return scales


def reconstruction_supervised_loss(
    predicted_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    physics: ACReconstruction,
    scales: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the equal-weight mean of five normalized reconstruction MSEs.

    Parameters
    ----------
    predicted_state, target_state : dict[str, torch.Tensor]
        AC reconstruction outputs produced from predicted and true ``chi``.
        The predicted tensors retain autograd; target tensors are detached.
    physics : ACReconstruction
        Physics module supplying generator indices.
    scales : dict[str, torch.Tensor]
        Positive scalar training-set RMS scale for each output group.

    Returns
    -------
    loss, parts : tuple[torch.Tensor, dict[str, torch.Tensor]]
        Equal-weight reconstruction loss and its five normalized MSE terms.
    """
    predicted = reconstruction_outputs(predicted_state, physics)
    target = reconstruction_outputs(target_state, physics)
    if set(scales) != set(RECONSTRUCTION_GROUPS):
        raise ValueError("reconstruction scales do not match output groups")
    parts = {
        name: ((predicted[name] - target[name]) / scales[name]).square().mean()
        for name in RECONSTRUCTION_GROUPS
    }
    return torch.stack([parts[name] for name in RECONSTRUCTION_GROUPS]).mean(), parts


def compute_loss(
    model: EqualityCompletionSurrogate,
    physics: ACReconstruction,
    batch: tuple[torch.Tensor, ...],
    normalization: dict[str, torch.Tensor],
    reconstruction_scales: dict[str, torch.Tensor],
    lambda_physics: float,
    lambda_reconstruction: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate state supervision, AC balance, and reconstruction supervision.

    Parameters
    ----------
    model : EqualityCompletionSurrogate
        Trainable mapping from normalized ``rho`` to normalized ``chi``.
    physics : ACReconstruction
        Shared AC reconstruction used by both prediction and target branches.
    batch : tuple[torch.Tensor, ...]
        ``rho, chi, u, load`` tensors from :class:`DecsDataset`.
    normalization : dict[str, torch.Tensor]
        Training-set means and standard deviations for ``rho`` and ``chi``.
    reconstruction_scales : dict[str, torch.Tensor]
        Training-set RMS scales returned by
        :func:`estimate_reconstruction_scales`.
    lambda_physics : float
        Nonnegative multiplier of the original AC-balance loss.
    lambda_reconstruction : float
        Nonnegative multiplier of the new supervised reconstruction loss.
    device : torch.device
        Device used for the model, physics, and batch tensors.

    Returns
    -------
    loss_total, metrics : tuple[torch.Tensor, dict[str, torch.Tensor]]
        Differentiable total loss and detached component metrics.
    """
    rho, chi, u, load = [value.to(device) for value in batch]
    rho_normalized = (rho - normalization["rho_mean"]) / normalization["rho_std"]
    chi_normalized = (chi - normalization["chi_mean"]) / normalization["chi_std"]
    prediction_normalized = model(rho_normalized)
    prediction = (
        prediction_normalized * normalization["chi_std"] + normalization["chi_mean"]
    )

    loss_sup = nn.functional.mse_loss(prediction_normalized, chi_normalized)
    predicted_state = physics.reconstruct(u, load, prediction)
    loss_phy = predicted_state["balance_residual"].square().mean()
    with torch.no_grad():
        target_state = physics.reconstruct(u, load, chi)
    loss_rec, reconstruction_parts = reconstruction_supervised_loss(
        predicted_state, target_state, physics, reconstruction_scales,
    )
    loss_total = (
        loss_sup
        + lambda_physics * loss_phy
        + lambda_reconstruction * loss_rec
    )

    n_angle = len(physics.pv) + len(physics.pq)
    return loss_total, {
        "loss_total": loss_total.detach(),
        "loss_sup": loss_sup.detach(),
        "loss_phy": loss_phy.detach(),
        "loss_rec": loss_rec.detach(),
        **{
            f"loss_rec_{name}": value.detach()
            for name, value in reconstruction_parts.items()
        },
        "angle_mae": (
            prediction[:, :n_angle] - chi[:, :n_angle]
        ).abs().mean().detach(),
        "voltage_mae": (
            prediction[:, n_angle:] - chi[:, n_angle:]
        ).abs().mean().detach(),
        "pf_residual": predicted_state["pf_residual"].mean().detach(),
    }


def run_epoch(
    model: EqualityCompletionSurrogate,
    physics: ACReconstruction,
    loader: DataLoader,
    normalization: dict[str, torch.Tensor],
    reconstruction_scales: dict[str, torch.Tensor],
    lambda_physics: float,
    lambda_reconstruction: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Run one train/evaluation epoch and return sample-weighted metrics.

    Parameters
    ----------
    model : EqualityCompletionSurrogate
        DECS model to train or evaluate.
    physics : ACReconstruction
        Shared fixed-PV reconstruction module.
    loader : DataLoader
        Split loader yielding ``rho, chi, u, load`` batches.
    normalization : dict[str, torch.Tensor]
        Training-set normalization for ``rho`` and ``chi``.
    reconstruction_scales : dict[str, torch.Tensor]
        Fixed training-set RMS scales for reconstruction outputs.
    lambda_physics, lambda_reconstruction : float
        Nonnegative physical and reconstruction loss multipliers.
    device : torch.device
        Training or evaluation device.
    optimizer : torch.optim.Optimizer or None
        Optimizer for training; ``None`` performs evaluation without gradients.

    Returns
    -------
    dict[str, float]
        Sample-weighted total, component, error, and residual metrics.
    """
    training = optimizer is not None
    model.train(training)
    names = (
        "loss_total", "loss_sup", "loss_phy", "loss_rec",
        *(f"loss_rec_{name}" for name in RECONSTRUCTION_GROUPS),
        "angle_mae", "voltage_mae", "pf_residual",
    )
    totals = {name: 0.0 for name in names}
    seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            loss, metrics = compute_loss(
                model, physics, batch, normalization, reconstruction_scales,
                lambda_physics, lambda_reconstruction, device,
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
        default=ROOT / "Data_generation" / "data" / "e2e118_decs_pandapower_N50000",
        help="IEEE-118 fixed-PV DECS dataset generated by pandapower or PGM",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "decs_pgm_fixedpv.pt",
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs=2, default=(512,512),
        metavar=("H1", "H2"), help="two IEEE-118 DECS hidden-layer widths",
    )
    parser.add_argument(
        "--epochs", type=int, default=1500,
        help="maximum number of DECS training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="training and evaluation samples per batch",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="initial Adam learning rate before scheduler reductions",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--lr-scheduler", choices=("none", "plateau"), default="plateau",
        help="learning-rate schedule; plateau monitors validation total loss",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.5,
        help="multiplicative LR reduction factor used by the plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-patience", type=int, default=50,
        help="validation epochs without material improvement before LR reduction",
    )
    parser.add_argument(
        "--lr-min-delta", type=float, default=5e-7,
        help="absolute validation-loss improvement required by the LR scheduler",
    )
    parser.add_argument(
        "--min-learning-rate", type=float, default=2e-6,
        help="lower learning-rate bound used by the plateau scheduler",
    )
    parser.add_argument(
        "--lambda-physics", type=float, default=0.03,
        help="weight of the normalized AC power-balance MSE",
    )
    parser.add_argument(
        "--lambda-reconstruction", type=float, default=0.10,
        help=(
            "weight of the equal-group normalized supervised loss on PV Qg, "
            "reference Pg/Qg, and both-end branch P/Q"
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=180,
        help="stale validation epochs before early stopping",
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-7,
        help="absolute validation-loss improvement required for checkpoint selection",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="data-loader worker processes; zero loads in the main process",
    )
    parser.add_argument(
        "--print-every", type=int, default=5,
        help="epoch interval for concise progress output",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="NumPy, PyTorch, and data-shuffle random seed",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="training device; auto selects CUDA when available",
    )
    args = parser.parse_args()

    if min(
        args.epochs, args.batch_size, args.patience, args.print_every,
        args.lr_decay_patience,
    ) < 1:
        raise ValueError("epochs, batch size, patience, and print interval must be positive")
    if (
        args.learning_rate <= 0.0
        or args.min_learning_rate <= 0.0
        or args.min_learning_rate > args.learning_rate
        or min(
            args.lambda_physics, args.lambda_reconstruction,
            args.min_delta, args.lr_min_delta,
        ) < 0.0
    ):
        raise ValueError("learning rate must be positive; loss weight and min_delta nonnegative")
    if not 0.0 < args.lr_decay_factor < 1.0:
        raise ValueError("lr_decay_factor must lie in (0,1)")
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
    scale_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    reconstruction_scales = estimate_reconstruction_scales(
        physics, scale_loader, device,
    )

    model = EqualityCompletionSurrogate(
        input_dim=physics.rho_dim,
        hidden_dims=tuple(args.hidden_dims),
        output_dim=physics.chi_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            threshold=args.lr_min_delta,
            threshold_mode="abs",
            min_lr=args.min_learning_rate,
        )
        if args.lr_scheduler == "plateau" else None
    )
    print("Fixed-PV DECS training configuration")
    print(f"  device: {device}")
    print(f"  samples: train={len(datasets['train'])}, validation={len(datasets['validation'])}, test={len(datasets['test'])}")
    print(f"  mapping: rho_dim={physics.rho_dim} -> chi_dim={physics.chi_dim}")
    print(
        f"  epochs={args.epochs}, batch_size={args.batch_size}, "
        f"learning_rate={args.learning_rate:.3e}, "
        f"lambda_physics={args.lambda_physics:g}, "
        f"lambda_reconstruction={args.lambda_reconstruction:g}"
    )
    print(
        "  reconstruction RMS scales: "
        + ", ".join(
            f"{name}={float(reconstruction_scales[name]):.3e}"
            for name in RECONSTRUCTION_GROUPS
        )
    )
    print(
        f"  lr_scheduler={args.lr_scheduler}, factor={args.lr_decay_factor:g}, "
        f"scheduler_patience={args.lr_decay_patience}, min_lr={args.min_learning_rate:.3e}"
    )
    print("  checkpoint selection and early stopping use validation only; test runs once after selection.")

    best_loss = np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start_time = perf_counter()
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model, physics, loaders["train"], normalization,
            reconstruction_scales, args.lambda_physics,
            args.lambda_reconstruction, device, optimizer,
        )
        validation_metrics = run_epoch(
            model, physics, loaders["validation"], normalization,
            reconstruction_scales, args.lambda_physics,
            args.lambda_reconstruction, device, optimizer=None,
        )
        improved = validation_metrics["loss_total"] < best_loss - args.min_delta
        if improved:
            best_loss = validation_metrics["loss_total"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step(validation_metrics["loss_total"])
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        })
        if epoch == 1 or epoch % args.print_every == 0 or stale_epochs == args.patience:
            elapsed = perf_counter() - start_time
            eta = elapsed / epoch * (args.epochs - epoch)
            status = "new best" if improved else f"stale {stale_epochs}/{args.patience}"
            print(
                f"[epoch {epoch:4d}/{args.epochs}] elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)} | lr {next_learning_rate:.2e} | {status}",
                flush=True,
            )
            print(
                f"  train: loss_total={train_metrics['loss_total']:.6e} "
                f"loss_sup={train_metrics['loss_sup']:.3e} "
                f"loss_phy={train_metrics['loss_phy']:.3e} "
                f"loss_rec={train_metrics['loss_rec']:.3e}", flush=True,
            )
            print(
                f"  validation: loss_total={validation_metrics['loss_total']:.6e} "
                f"loss_sup={validation_metrics['loss_sup']:.3e} "
                f"loss_phy={validation_metrics['loss_phy']:.3e} "
                f"loss_rec={validation_metrics['loss_rec']:.3e} "
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
        reconstruction_scales, args.lambda_physics,
        args.lambda_reconstruction, device, optimizer=None,
    )
    test_report = detailed_metrics(
        model, physics, loaders["test"], normalization, device,
    )
    print(
        f"best-checkpoint test: loss_total={test_loss['loss_total']:.6e}, "
        f"loss_sup={test_loss['loss_sup']:.3e}, "
        f"loss_phy={test_loss['loss_phy']:.3e}, "
        f"loss_rec={test_loss['loss_rec']:.3e}, "
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
        "lambda_reconstruction": args.lambda_reconstruction,
        "reconstruction_loss": {
            "groups": list(RECONSTRUCTION_GROUPS),
            "normalization": "training_split_group_rms",
            "group_weighting": "equal_mean",
            "scales": {
                name: float(value.detach().cpu())
                for name, value in reconstruction_scales.items()
            },
        },
        "training_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lambda_physics": args.lambda_physics,
            "lambda_reconstruction": args.lambda_reconstruction,
            "lr_scheduler": args.lr_scheduler,
            "lr_decay_factor": args.lr_decay_factor,
            "lr_decay_patience": args.lr_decay_patience,
            "lr_min_delta": args.lr_min_delta,
            "min_learning_rate": args.min_learning_rate,
        },
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
