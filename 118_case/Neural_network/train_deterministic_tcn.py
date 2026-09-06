"""Train a deterministic non-causal TCN benchmark for SMPC dispatch.

The benchmark uses the same pooled uncertainty data, frozen differentiable
equality-completion surrogate, admissible output map, physical constraint
families, and hierarchical mean-CVaR penalty as ``train_generator.py``. It
removes latent variables, candidate sampling, diversity regularization, and
feasibility-dependent economic shaping. Every epoch directly minimizes the
normalized SMPC objective plus the constraint-violation penalty.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import (  # noqa: E402
    DifferentiableEqualityCompletion,
    load_decs_checkpoint,
)
from Neural_network.deterministic_tcn import DeterministicTCN  # noqa: E402
from Neural_network.noncausal_tcn import PROJECTION_METHOD  # noqa: E402
from Neural_network.train_generator import (  # noqa: E402
    GeneratorDataset,
    compute_training_feature_bounds,
    expected_generation_objective,
    file_sha256,
    format_duration,
    hierarchical_feasibility_loss,
)


def direct_objective_penalty_loss(
    objective: torch.Tensor,
    feasibility_loss: torch.Tensor,
    objective_cost_scale: float,
    lambda_objective: float,
    lambda_feasibility: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine the direct SMPC objective and constraint penalty.

    Parameters
    ----------
    objective : torch.Tensor, shape (batch, 1) or (batch,)
        Expected horizon generation cost in monetary units.
    feasibility_loss : scalar torch.Tensor
        Hierarchical mean-CVaR violation summed over scenarios, time, and
        constraint families and averaged over batch instances.
    objective_cost_scale : float
        Positive frozen cost divisor in the same units as ``objective``.
    lambda_objective : float
        Nonnegative coefficient of the normalized direct objective.
    lambda_feasibility : float
        Nonnegative coefficient of the constraint-violation penalty.

    Returns
    -------
    total_loss : torch.Tensor
        Scalar weighted training loss.
    normalized_objective : torch.Tensor
        Unweighted batch-mean objective divided by ``objective_cost_scale``.
    """
    if objective.ndim not in (1, 2) or objective.numel() < 1:
        raise ValueError("objective must be a nonempty batch tensor")
    if feasibility_loss.ndim != 0:
        raise ValueError("feasibility_loss must be scalar")
    if objective_cost_scale <= 0:
        raise ValueError("objective_cost_scale must be positive")
    if min(lambda_objective, lambda_feasibility) < 0:
        raise ValueError("loss weights must be nonnegative")
    normalized_objective = objective.mean() / objective_cost_scale
    total_loss = (
        lambda_objective * normalized_objective
        + lambda_feasibility * feasibility_loss
    )
    return total_loss, normalized_objective


def resolve_objective_cost_scale(
    requested_scale: float,
    calibration_cost: float,
) -> float:
    """Return a positive fixed cost divisor for deterministic training.

    Parameters
    ----------
    requested_scale : float
        User-specified monetary cost divisor. Zero requests automatic scaling.
    calibration_cost : float
        Mean untrained-model horizon cost over a held-in training subset, used
        only when ``requested_scale`` is zero.

    Returns
    -------
    float
        Positive fixed divisor used for every training epoch.
    """
    if requested_scale < 0:
        raise ValueError("requested objective cost scale must be nonnegative")
    if requested_scale > 0:
        return float(requested_scale)
    if not np.isfinite(calibration_cost) or calibration_cost <= 0:
        raise ValueError("automatic objective cost calibration must be finite and positive")
    return float(calibration_cost)


def compute_batch(
    model: DeterministicTCN,
    completion: DifferentiableEqualityCompletion,
    condition: torch.Tensor,
    scenario_load: torch.Tensor,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    reference_ramp: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Generate one schedule and evaluate its direct penalized objective.

    Parameters
    ----------
    model : DeterministicTCN
        Trainable deterministic non-causal temporal network.
    completion : DifferentiableEqualityCompletion
        Frozen DECS model followed by exact AC state reconstruction.
    condition : torch.Tensor, shape (B, T, condition_dim)
        Normalized pooled temporal input features.
    scenario_load : torch.Tensor, shape (B, S, T, n_load_buses, 2)
        Full physical load scenarios in MW/Mvar.
    free_lower, free_upper : torch.Tensor, shape (free_dim,)
        Static free-variable bounds in MW and p.u.
    free_ramp : torch.Tensor, shape (n_free_pg,)
        Free-generator ramp limit in MW per period.
    reference_ramp : scalar torch.Tensor
        Reference-generator ramp limit in MW per period.
    args : argparse.Namespace
        Validated loss and reporting hyperparameters from :func:`build_parser`.

    Returns
    -------
    loss : torch.Tensor
        Scalar differentiable direct-objective-plus-penalty loss.
    metrics : dict[str, torch.Tensor]
        Detached cost, violation, feasibility, and DECS diagnostics.
    """
    batch, scenarios, horizon = scenario_load.shape[:3]
    trajectory, _, _ = model(
        condition, free_lower, free_upper, free_ramp, free_ramp,
    )
    free_dim = completion.physics.free_dim
    if tuple(trajectory.shape) != (batch, horizon, free_dim):
        raise RuntimeError("deterministic TCN trajectory dimensions do not match the batch")

    # Retain a singleton candidate dimension so the exact reference penalty and
    # objective helpers are used without changing their reduction semantics.
    candidate_trajectory = trajectory[:, None]
    candidate_scenarios = candidate_trajectory[:, :, None].expand(
        -1, -1, scenarios, -1, -1,
    )
    repeated_load = scenario_load[:, None].expand(
        -1, 1, -1, -1, -1, -1,
    )
    state = completion(
        candidate_scenarios.reshape(batch * scenarios * horizon, free_dim),
        repeated_load.reshape(
            batch * scenarios * horizon, *scenario_load.shape[3:],
        ),
    )

    physics = completion.physics
    physical_violation = physics.constraint_violation(state).reshape(
        batch, 1, scenarios, horizon, -1,
    )
    category_sizes = physics.remaining_constraint_sizes()
    if sum(category_sizes.values()) != physical_violation.shape[-1]:
        raise RuntimeError("physical constraint categories do not match violation width")
    physical_categories = dict(zip(
        category_sizes,
        torch.split(physical_violation, tuple(category_sizes.values()), dim=-1),
    ))

    reference_pg = state["pg"].reshape(
        batch, 1, scenarios, horizon, physics.n_gen,
    )[..., physics.reference_generator]
    delta_pg = reference_pg[:, :, :, 1:] - reference_pg[:, :, :, :-1]
    ramp_scale = reference_ramp.clamp_min(1e-8)
    ramp_violation = torch.stack([
        torch.relu((delta_pg - ramp_scale) / ramp_scale),
        torch.relu((-delta_pg - ramp_scale) / ramp_scale),
    ], dim=-1)
    violation_families = {**physical_categories, "ramp": ramp_violation}
    schedule_violation, feasibility_loss, shaped_families = (
        hierarchical_feasibility_loss(
            violation_families,
            args.mean_cvar_alpha,
            args.cvar_tail_fraction,
            check_nonnegative=args.detect_anomaly,
        )
    )
    maximum_violation = torch.stack([
        values.flatten(start_dim=2).amax(dim=2)
        for values in violation_families.values()
    ], dim=2).amax(dim=2)

    scenario_pg = state["pg"].reshape(
        batch, 1, scenarios, horizon, physics.n_gen,
    )
    objective = expected_generation_objective(scenario_pg, completion)
    strict_feasibility_loss = maximum_violation.mean()
    optimization_feasibility_loss = (
        feasibility_loss
        + args.lambda_max_violation * strict_feasibility_loss
    )
    loss, normalized_objective = direct_objective_penalty_loss(
        objective,
        optimization_feasibility_loss,
        args.objective_cost_scale,
        args.lambda_objective,
        args.lambda_fea,
    )

    feasible = (maximum_violation[:, 0] <= args.feasibility_tolerance).detach()
    objective_per_instance = objective[:, 0]
    metrics = {
        "loss": loss.detach(),
        "objective_normalized": normalized_objective.detach(),
        "fea": optimization_feasibility_loss.detach(),
        "hierarchical_fea": feasibility_loss.detach(),
        "strict_fea": strict_feasibility_loss.detach(),
        "schedule_violation": schedule_violation.mean().detach(),
        "feasible": feasible.float().mean().detach(),
        "max_violation": maximum_violation.mean().detach(),
        "worst_violation": maximum_violation.amax().detach(),
        "violation_pg": shaped_families["pg"].mean().detach(),
        "violation_qg": shaped_families["qg"].mean().detach(),
        "violation_voltage": shaped_families["voltage"].mean().detach(),
        "violation_angle": shaped_families["angle"].mean().detach(),
        "violation_thermal": shaped_families["thermal"].mean().detach(),
        "violation_ramp": shaped_families["ramp"].mean().detach(),
        "cost_mean": objective_per_instance.mean().detach(),
        "pf_residual": state["pf_residual"].mean().detach(),
        "feasible_cost_sum": torch.where(
            feasible, objective_per_instance, torch.zeros_like(objective_per_instance),
        ).sum().detach(),
        "feasible_cost_count": feasible.sum().detach(),
    }
    return loss, metrics


def run_epoch(
    model: DeterministicTCN,
    completion: DifferentiableEqualityCompletion,
    loader: DataLoader,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    reference_ramp: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Run one training or validation epoch and aggregate instance metrics.

    Parameters
    ----------
    model, completion, loader
        Deterministic TCN, frozen equality completion, and split mini-batches.
    free_lower, free_upper, free_ramp, reference_ramp
        Physical tensors documented by :func:`compute_batch`.
    args : argparse.Namespace
        Validated command-line hyperparameters.
    device : torch.device
        CPU or CUDA computation device.
    optimizer : torch.optim.Optimizer or None
        Adam during training; ``None`` selects validation.

    Returns
    -------
    dict[str, float]
        Sample-weighted loss, cost, feasibility, and violation metrics.
    """
    training = optimizer is not None
    model.train(training)
    completion.eval()
    names = (
        "loss", "objective_normalized", "fea", "hierarchical_fea", "strict_fea",
        "schedule_violation", "feasible",
        "max_violation", "violation_pg", "violation_qg", "violation_voltage",
        "violation_angle", "violation_thermal", "violation_ramp", "cost_mean",
        "pf_residual",
    )
    totals = {name: 0.0 for name in names}
    worst_violation = 0.0
    feasible_cost_sum = 0.0
    feasible_cost_count = 0
    seen = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_number, (condition, scenario_load) in enumerate(loader, start=1):
            condition = condition.to(device)
            scenario_load = scenario_load.to(device)
            loss, metrics = compute_batch(
                model, completion, condition, scenario_load,
                free_lower, free_upper, free_ramp, reference_ramp, args,
            )
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss before backward in training batch {batch_number}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm, error_if_nonfinite=True,
                    )
                else:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError(
                                f"non-finite TCN gradient in {name} at batch {batch_number}"
                            )
                optimizer.step()

            count = len(condition)
            seen += count
            for name in names:
                totals[name] += float(metrics[name]) * count
            worst_violation = max(worst_violation, float(metrics["worst_violation"]))
            feasible_cost_sum += float(metrics["feasible_cost_sum"])
            feasible_cost_count += int(metrics["feasible_cost_count"])

    if seen == 0:
        raise ValueError("data loader contains no instances")
    result = {name: value / seen for name, value in totals.items()}
    result["worst_violation"] = worst_violation
    result["feasible_cost_mean"] = (
        feasible_cost_sum / feasible_cost_count
        if feasible_cost_count else float("nan")
    )
    result["feasible_cost_count"] = feasible_cost_count
    return result


def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    elapsed: float,
    eta: float,
    status: str,
    learning_rate: float,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    """Print compact loss, cost, feasibility, and constraint diagnostics.

    Parameters
    ----------
    epoch, total_epochs : int
        Current and maximum epoch numbers.
    elapsed, eta : float
        Elapsed and estimated remaining wall time in seconds.
    status : str
        Validation checkpoint status for this epoch.
    learning_rate : float
        Current Adam learning rate.
    args : argparse.Namespace
        Loss coefficients and objective scale used in this epoch.
    train_metrics, validation_metrics : dict[str, float]
        Aggregated split metrics returned by :func:`run_epoch`.
    """
    print(
        f"[epoch {epoch:4d}/{total_epochs} | lr {learning_rate:.3e} | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | {status}]",
        flush=True,
    )
    for split, values in (
        ("train", train_metrics), ("validation", validation_metrics),
    ):
        weighted_objective = args.lambda_objective * values["objective_normalized"]
        weighted_penalty = args.lambda_fea * values["fea"]
        feasible_cost = values["feasible_cost_mean"]
        feasible_cost_text = f"{feasible_cost:.3f}" if np.isfinite(feasible_cost) else "n/a"
        print(
            f"  {split:10s} loss={values['loss']:.6e} | "
            f"objective(norm)={values['objective_normalized']:.3e} "
            f"weighted={weighted_objective:.3e} | "
            f"penalty={values['fea']:.3e} weighted={weighted_penalty:.3e}",
            flush=True,
        )
        if args.lambda_max_violation > 0:
            print(
                f"  {split:10s} penalty parts: hierarchical="
                f"{values['hierarchical_fea']:.3e} + "
                f"{args.lambda_max_violation:.3g}*trajectory_max="
                f"{values['strict_fea']:.3e}",
                flush=True,
            )
        print(
            f"  {split:10s} cost mean={values['cost_mean']:.3f} "
            f"feasible_mean={feasible_cost_text} | feasible={values['feasible']:.3f} "
            f"max_mean={values['max_violation']:.3e} worst={values['worst_violation']:.3e} "
            f"DECS_PF={values['pf_residual']:.3e} p.u.",
            flush=True,
        )
        print(
            f"  {split:10s} mean-CVaR: Pg={values['violation_pg']:.2e} "
            f"Qg={values['violation_qg']:.2e} V={values['violation_voltage']:.2e} "
            f"angle={values['violation_angle']:.2e} "
            f"thermal={values['violation_thermal']:.2e} "
            f"ramp={values['violation_ramp']:.2e}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    """Declare command-line parameters for reproducible benchmark training."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / "e2e118_N5000_S20_T16",
        help="SMPC dataset containing pooled conditions and full load scenarios",
    )
    parser.add_argument(
        "--decs", type=Path,
        default=Path(__file__).resolve().parent / "decs_pgm_fixedpv.pt",
        help="frozen DECS checkpoint used for differentiable equality completion",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "deterministic_tcn.pt",
        help="checkpoint path for the best validation-loss model",
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="maximum number of direct objective-plus-penalty training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="IEEE-118 SMPC scenario-tree instances per gradient update",
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=128,
        help="input-embedding and temporal-block feature width",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3,
        help="positive odd non-causal convolution width in time periods",
    )
    parser.add_argument(
        "--dilations", type=int, nargs="+", default=(1, 2, 4, 8), metavar="D",
        help="positive TCN dilations whose receptive field must cover the horizon",
    )
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="symmetric one-period ramp limit as a fraction of generator Pmax",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="Adam learning rate used for all epochs",
    )
    parser.add_argument(
        "--lambda-objective", type=float, default=4e-3,
        help="weight of the normalized direct SMPC generation objective",
    )
    parser.add_argument(
        "--lambda-fea", type=float, default=0.5,
        help=(
            "weight of the single-trajectory hierarchical mean-CVaR violation "
            "penalty; the default allows a controlled feasibility/economy tradeoff"
        ),
    )
    parser.add_argument(
        "--lambda-max-violation", type=float, default=0.0,
        help=(
            "additional weight on each trajectory's maximum normalized violation; "
            "zero exactly preserves the reference hierarchical penalty"
        ),
    )
    parser.add_argument(
        "--objective-cost-scale", type=float, default=0.0, metavar="COST",
        help=(
            "horizon-cost divisor in monetary units; zero freezes the mean "
            "untrained-model cost over up to 128 training instances"
        ),
    )
    parser.add_argument(
        "--mean-cvar-alpha", type=float, default=0.05,
        help="mean weight alpha_c in (0,1); the remainder weights upper-tail CVaR",
    )
    parser.add_argument(
        "--cvar-tail-fraction", type=float, default=0.05,
        help="largest residual fraction rho_c averaged in every constraint family",
    )
    parser.add_argument(
        "--feasibility-tolerance", type=float, default=1e-4,
        help="reporting-only maximum normalized violation for feasibility",
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="stale validation epochs before early stopping",
    )
    parser.add_argument(
        "--min-delta", type=float, default=0.0,
        help="minimum validation-loss reduction counted as improvement",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="global TCN gradient clipping norm; zero disables clipping",
    )
    parser.add_argument(
        "--detect-anomaly", action="store_true",
        help="enable autograd anomaly tracing and full residual checks",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker count; zero is reliable on Windows",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="seed for TCN initialization and training-set shuffling",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="training device; auto selects CUDA when available",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid hyperparameters and missing inputs before data loading.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed values declared by :func:`build_parser`.
    """
    if min(args.epochs, args.batch_size, args.hidden_channels, args.patience) < 1:
        raise ValueError("epochs, batch size, hidden channels, and patience must be positive")
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not args.dilations or min(args.dilations) < 1:
        raise ValueError("dilations must be positive")
    if not 0 < args.ramp_fraction <= 1:
        raise ValueError("ramp_fraction must lie in (0,1]")
    if args.learning_rate <= 0 or args.objective_cost_scale < 0:
        raise ValueError("learning rate must be positive and objective cost scale nonnegative")
    if not 0 < args.mean_cvar_alpha < 1:
        raise ValueError("mean_cvar_alpha must lie in (0,1)")
    if not 0 < args.cvar_tail_fraction <= 1:
        raise ValueError("cvar_tail_fraction must lie in (0,1]")
    if min(
        args.lambda_objective, args.lambda_fea, args.lambda_max_violation,
        args.feasibility_tolerance,
        args.min_delta, args.max_grad_norm, args.num_workers,
    ) < 0:
        raise ValueError("loss weights, tolerances, and worker count must be nonnegative")
    if not args.data.is_dir() or not args.decs.is_file():
        raise FileNotFoundError("SMPC dataset or DECS checkpoint does not exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def main() -> None:
    """Train, validate, and save the best deterministic TCN checkpoint."""
    args = build_parser().parse_args()
    validate_arguments(args)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    n_instances = int(np.load(args.data / "current_load.npy", mmap_mode="r").shape[0])
    split_dir = args.data / "split"
    split_indices = {
        name: np.load(split_dir / f"{name}_indices.npy")
        for name in ("train", "validation", "test")
    }
    combined_indices = np.concatenate(list(split_indices.values()))
    if (
        len(combined_indices) != n_instances
        or not np.array_equal(np.sort(combined_indices), np.arange(n_instances))
    ):
        raise ValueError("saved train/validation/test indices are not a partition")
    metadata_counts = metadata.get("split", {}).get("counts", {})
    if any(
        int(metadata_counts.get(name, -1)) != len(values)
        for name, values in split_indices.items()
    ):
        raise ValueError("saved split counts differ from metadata.json")

    feature_min, feature_max = compute_training_feature_bounds(
        args.data, split_indices["train"],
    )
    datasets = {
        split: GeneratorDataset(args.data, indices, feature_min, feature_max)
        for split, indices in split_indices.items()
        if split in ("train", "validation")
    }
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

    completion = load_decs_checkpoint(args.decs, device)
    decs_metadata = torch.load(
        args.decs, map_location="cpu", weights_only=False,
    ).get("data_metadata", {})
    if decs_metadata.get("case_name") != metadata.get("case_name"):
        raise ValueError("DECS and deterministic TCN data use different network cases")
    if not (
        decs_metadata.get("bus_type_model") == "fixed_PV_PQ"
        and decs_metadata.get("enforce_q_limits") is False
        and decs_metadata.get("pv_to_pq_switching") is False
        and int(decs_metadata.get("chi_dim", -1)) == completion.physics.chi_dim
    ):
        raise ValueError(
            "Deterministic TCN training requires the IEEE-118 fixed-PV DECS "
            "without PV-to-PQ switching"
        )
    if decs_metadata.get("source_base_usage") != "all_time_load_range_only":
        raise ValueError(
            "DECS must use only the base dataset's configured all-time load "
            "range, not empirical current/future samples"
        )
    source_metadata = decs_metadata.get("source_base_metadata", {})
    identity_fields = (
        "dataset_type", "case_name", "n_instances", "n_scenarios", "horizon", "seed",
    )
    for field in identity_fields:
        if source_metadata.get(field) != metadata.get(field):
            raise ValueError(f"DECS and TCN data differ in metadata field '{field}'")
    decs_sha256 = file_sha256(args.decs)
    for parameter in completion.parameters():
        parameter.requires_grad_(False)

    physics = completion.physics
    free_lower, free_upper = physics.free_variable_bounds(1)
    free_lower = free_lower[0].to(device)
    free_upper = free_upper[0].to(device)
    free_ramp = args.ramp_fraction * physics.pg_max[physics.free_pg]
    reference_ramp = args.ramp_fraction * physics.pg_max[physics.reference_generator]
    active_ramp = args.ramp_fraction * physics.pg_max[physics.active]
    model = DeterministicTCN(
        condition_dim=datasets["train"].condition_dim,
        free_dim=physics.free_dim,
        n_free_pg=physics.n_free_pg,
        projection_method=PROJECTION_METHOD,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
    ).to(device)
    if model.receptive_field < datasets["train"].horizon:
        raise ValueError(
            f"TCN receptive field {model.receptive_field} is shorter than "
            f"horizon {datasets['train'].horizon}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    requested_cost_scale = args.objective_cost_scale
    cost_scale_source = "command_line"
    calibration_count = 0
    if requested_cost_scale == 0.0:
        calibration_count = min(128, len(datasets["train"]))
        calibration_loader = DataLoader(
            Subset(datasets["train"], range(calibration_count)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        args.objective_cost_scale = 1.0
        calibration_metrics = run_epoch(
            model, completion, calibration_loader, free_lower, free_upper,
            free_ramp, reference_ramp, args, device, optimizer=None,
        )
        args.objective_cost_scale = resolve_objective_cost_scale(
            requested_cost_scale, calibration_metrics["cost_mean"],
        )
        cost_scale_source = f"initial_training_mean_{calibration_count}"

    print("Deterministic non-causal TCN benchmark training configuration")
    print(f"  device: {device}")
    print(
        f"  instances: train={len(datasets['train'])}, "
        f"validation={len(datasets['validation'])}, "
        f"test={len(split_indices['test'])} (reserved; not evaluated)"
    )
    print(
        f"  condition=(B,{datasets['train'].horizon},{datasets['train'].condition_dim}), "
        f"loads=(B,{datasets['train'].n_scenarios},{datasets['train'].horizon},"
        f"{datasets['train'].current.shape[1]},2)"
    )
    print(
        f"  TCN: channels={args.hidden_channels}, kernel={args.kernel_size}, "
        f"dilations={list(args.dilations)}, receptive_field={model.receptive_field}"
    )
    print(
        f"  projection={model.projection_method}, output bias=0.5, "
        "saturated gradients retained only toward the admissible interval"
    )
    print("  deterministic outputs: one trajectory per SMPC instance; no latent variable")
    print(
        f"  loss={args.lambda_objective:.3e}*(C/{args.objective_cost_scale:.3f}) "
        f"+ {args.lambda_fea:.3e}*L_fea | alpha={args.mean_cvar_alpha:.2f}, "
        f"rho={args.cvar_tail_fraction:.2f}"
    )
    if args.lambda_max_violation > 0:
        print(
            "  L_fea=L_hierarchical + "
            f"{args.lambda_max_violation:g}*mean(trajectory maximum violation)"
        )
    print(f"  objective cost scale source: {cost_scale_source}")
    print(
        f"  ramp={100 * args.ramp_fraction:.1f}% Pmax/period, "
        f"free={free_ramp.detach().cpu().tolist()} MW, "
        f"reference={float(reference_ramp):.3f} MW"
    )

    constraint_names = [*physics.remaining_constraint_sizes(), "ramp"]
    checkpoint_common = {
        "model_class": "DeterministicTCN",
        "benchmark_definition": {
            "deterministic": True,
            "latent_variable": False,
            "candidates_per_instance": 1,
            "diversity_loss": False,
            "soft_feasibility_economic_shaping": False,
            "projection_method": PROJECTION_METHOD,
        },
        "model_config": model.configuration(),
        "normalization": {
            "feature_min": datasets["train"].feature_min.numpy(),
            "feature_max": datasets["train"].feature_max.numpy(),
        },
        "free_variable_lower": free_lower.detach().cpu(),
        "free_variable_upper": free_upper.detach().cpu(),
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "free_ramp_mw_per_period": free_ramp.detach().cpu(),
        "reference_ramp_mw_per_period": reference_ramp.detach().cpu(),
        "active_ramp_mw_per_period": active_ramp.detach().cpu(),
        "constraint_scaling": {
            "two_sided_limits": "inverse physical operating range",
            "thermal": "inverse branch MVA rating",
            "thermal_directions_per_branch": 2,
            "per_scenario_time_sizes": physics.remaining_constraint_sizes(),
            "dg_diagonal_per_scenario_time": {
                name: value.detach().cpu()
                for name, value in physics.remaining_constraint_scales().items()
            },
            "reference_ramp_inverse_mw": reference_ramp.reciprocal().detach().cpu(),
            "hierarchical_aggregation": {
                "families": constraint_names,
                "alpha_c": {name: args.mean_cvar_alpha for name in constraint_names},
                "rho_c": {name: args.cvar_tail_fraction for name in constraint_names},
                "reduction": "sum scenarios, time, and families; mean batch",
            },
        },
        "objective_definition": (
            "sum all-generator cost over time per scenario, including reconstructed "
            "slack Pg, then average over scenarios"
        ),
        "loss_definition": (
            "direct_objective_plus_hierarchical_mean_cvar_penalty"
            if args.lambda_max_violation == 0
            else "direct_objective_plus_hierarchical_and_trajectory_max_penalties"
        ),
        "loss_hyperparameters": {
            "lambda_objective": args.lambda_objective,
            "lambda_fea": args.lambda_fea,
            "lambda_max_violation": args.lambda_max_violation,
            "objective_cost_scale": args.objective_cost_scale,
            "objective_cost_scale_source": cost_scale_source,
            "mean_cvar_alpha": args.mean_cvar_alpha,
            "cvar_tail_fraction": args.cvar_tail_fraction,
            "feasibility_tolerance": args.feasibility_tolerance,
        },
        "training_hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_grad_norm": args.max_grad_norm,
            "detect_anomaly": args.detect_anomaly,
            "optimizer": "Adam",
            "training_stages": 1,
        },
        "decs_checkpoint": str(args.decs.resolve()),
        "decs_checkpoint_sha256": decs_sha256,
        "split": {
            "ratio": metadata["split"]["ratio"],
            "seed": metadata["split"]["seed"],
            "counts": {name: len(values) for name, values in split_indices.items()},
            "indices": {
                name: values.astype(np.int64).tolist()
                for name, values in split_indices.items()
            },
        },
        "data_metadata": metadata,
        "seed": args.seed,
    }

    best_loss = np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float | int | bool]] = []
    start_time = perf_counter()
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, completion, loaders["train"], free_lower, free_upper,
            free_ramp, reference_ramp, args, device, optimizer,
        )
        validation_metrics = run_epoch(
            model, completion, loaders["validation"], free_lower, free_upper,
            free_ramp, reference_ramp, args, device, optimizer=None,
        )
        if not np.isfinite(validation_metrics["loss"]):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
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
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        })
        elapsed = perf_counter() - start_time
        eta = elapsed / epoch * (args.epochs - epoch)
        status = "new best" if improved else f"stale {stale_epochs}/{args.patience}"
        print_epoch_metrics(
            epoch, args.epochs, elapsed, eta, status,
            optimizer.param_groups[0]["lr"], args,
            train_metrics, validation_metrics,
        )
        if stale_epochs >= args.patience:
            stopped_early = True
            print(
                f"early stopping at epoch {epoch}; best epoch {best_epoch}, "
                f"validation loss {best_loss:.6e}"
            )
            break

    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **checkpoint_common,
        "model_state": best_state,
        "training_phase": "single_stage_direct_objective_penalty",
        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta,
            "best_epoch": best_epoch,
            "stopped_epoch": history[-1]["epoch"],
            "stopped_early": stopped_early,
        },
        "best_validation_loss": best_loss,
        "test_evaluated_during_training": False,
        "training_history": history,
    }, args.output)
    print(
        f"saved deterministic TCN checkpoint from epoch {best_epoch} "
        f"with validation loss {best_loss:.6e} to {args.output}"
    )


if __name__ == "__main__":
    main()
