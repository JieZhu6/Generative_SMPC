"""Train an experimental Stage-1 Generator with hit-aligned candidate losses.

This program leaves ``train_generator.py`` unchanged.  It isolates the current
feasibility/diversity bottleneck before economic Stage 2 is attempted.  The
default improved objective combines the lowest-violation candidates with a
small all-candidate coverage term and evaluates diversity among the current
top-M candidates using a batch-adaptive Gaussian width.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Neural_network.decs import load_decs_checkpoint  # noqa: E402
from Neural_network.generator_benchmarks import (  # noqa: E402
    build_generator_model,
    get_benchmark_spec,
)
from Neural_network.train_generator import (  # noqa: E402
    GeneratorDataset,
    atomic_torch_save,
    build_plateau_scheduler,
    compute_training_feature_bounds,
    diversity_loss,
    expected_generation_objective,
    file_sha256,
    format_duration,
    hierarchical_feasibility_loss,
    soft_feasibility_score,
)


def hit_aligned_feasibility_loss(
    candidate_violation: torch.Tensor,
    top_k: int,
    coverage_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine lowest-candidate violation with weak population coverage.

    Parameters
    ----------
    candidate_violation : torch.Tensor, shape (batch, candidates)
        Nonnegative scenario/time aggregated violation of every candidate.
    top_k : int
        Number of lowest-violation candidates included in the hit-oriented
        component.  ``1`` most directly approximates best-of-K recovery.
    coverage_weight : float
        Nonnegative multiplier on the all-candidate mean violation.

    Returns
    -------
    total, hit_component, coverage_component : tuple of torch.Tensor
        Scalar loss and its two unweighted components.
    """
    if candidate_violation.ndim != 2:
        raise ValueError("candidate_violation must have shape (batch,candidates)")
    if not 1 <= top_k <= candidate_violation.shape[1]:
        raise ValueError("top_k must lie in [1,candidates]")
    if coverage_weight < 0:
        raise ValueError("coverage_weight must be nonnegative")
    hit_component = candidate_violation.topk(
        top_k, dim=1, largest=False, sorted=False,
    ).values.mean()
    coverage_component = candidate_violation.mean()
    total = hit_component + coverage_weight * coverage_component
    return total, hit_component, coverage_component


def normalized_pairwise_distance(
    trajectory: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Return off-diagonal normalized RMS distances between trajectories.

    Parameters
    ----------
    trajectory : torch.Tensor, shape (batch, candidates, horizon, free_dim)
        Generated free-variable trajectories in physical units.
    lower, upper : torch.Tensor, shape (free_dim,)
        Static physical limits used to normalize active power and voltage.

    Returns
    -------
    torch.Tensor, shape (batch, candidates*(candidates-1)/2)
        Upper-triangular candidate distances for each instance.
    """
    center = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower).clamp_min(1e-8)
    normalized = ((trajectory - center) / half_range).flatten(start_dim=2)
    distance2 = (
        normalized[:, :, None, :] - normalized[:, None, :, :]
    ).square().mean(dim=-1)
    mask = torch.triu(
        torch.ones_like(distance2, dtype=torch.bool), diagonal=1,
    )
    return distance2[mask].reshape(len(trajectory), -1).clamp_min(0).sqrt()


def adaptive_topm_diversity_loss(
    trajectory: torch.Tensor,
    candidate_violation: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    top_m: int,
    sigma_scale: float,
    sigma_min: float,
    sigma_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repel the top-M candidates without absolute-feasibility gating.

    Parameters
    ----------
    trajectory : torch.Tensor, shape (batch, candidates, horizon, free_dim)
        Generated free-variable trajectories.
    candidate_violation : torch.Tensor, shape (batch, candidates)
        Candidate violations used only for detached top-M selection.
    lower, upper : torch.Tensor, shape (free_dim,)
        Static free-variable bounds used for dimensionless distances.
    top_m : int
        Number of currently lowest-violation candidates to diversify.
    sigma_scale : float
        Positive multiplier on each instance's detached median distance.
    sigma_min, sigma_max : float
        Positive lower and upper bounds for the adaptive Gaussian width.

    Returns
    -------
    loss, sigma_mean, distance_mean : tuple of torch.Tensor
        Gaussian similarity to minimize and detached diagnostics.
    """
    batch, candidates, horizon, free_dim = trajectory.shape
    if candidate_violation.shape != (batch, candidates):
        raise ValueError("candidate_violation shape does not match trajectory")
    if not 2 <= top_m <= candidates:
        raise ValueError("top_m must lie in [2,candidates]")
    if sigma_scale <= 0 or not 0 < sigma_min <= sigma_max:
        raise ValueError("adaptive sigma settings must be positive and ordered")

    selected_indices = candidate_violation.detach().topk(
        top_m, dim=1, largest=False, sorted=False,
    ).indices
    gather_indices = selected_indices[:, :, None, None].expand(
        -1, -1, horizon, free_dim,
    )
    selected = trajectory.gather(1, gather_indices)
    distances = normalized_pairwise_distance(selected, lower, upper)
    distance2 = distances.square()
    sigma = (
        sigma_scale * distances.detach().quantile(0.5, dim=1)
    ).clamp(min=sigma_min, max=sigma_max)
    kernel = torch.exp(-distance2 / (2.0 * sigma[:, None].square()))
    return kernel.mean(), sigma.mean().detach(), distances.mean().detach()


def compute_batch(
    model: torch.nn.Module,
    completion: torch.nn.Module,
    condition: torch.Tensor,
    scenario_load: torch.Tensor,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    reference_ramp: torch.Tensor,
    args: argparse.Namespace,
    latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate baseline or improved Stage-1 loss on one SMPC batch.

    Parameters
    ----------
    model, completion : torch.nn.Module
        Trainable Generator and frozen differentiable equality completion.
    condition : torch.Tensor, shape (B,T,condition_dim)
        Normalized min/mean/max condition sequence.
    scenario_load : torch.Tensor, shape (B,S,T,n_load,2)
        Physical scenario loads in MW/Mvar.
    free_lower, free_upper : torch.Tensor, shape (free_dim,)
        Static Generator decision limits.
    free_ramp : torch.Tensor, shape (n_free_pg,)
        Projected free-generator ramp limits in MW/period.
    reference_ramp : scalar torch.Tensor
        Reference-generator ramp limit in MW/period.
    args : argparse.Namespace
        Validated experiment and loss settings.
    latent : torch.Tensor or None, shape (B,K,latent_dim)
        Optional fixed latent samples for deterministic validation.

    Returns
    -------
    loss, metrics : tuple
        Differentiable scalar objective and detached diagnostics.
    """
    batch, scenarios, horizon = scenario_load.shape[:3]
    candidates = args.candidates
    trajectory, _, _ = model(
        condition, candidates, free_lower, free_upper,
        free_ramp, free_ramp, latent,
    )
    free_dim = completion.physics.free_dim
    if tuple(trajectory.shape) != (batch, candidates, horizon, free_dim):
        raise RuntimeError("generator trajectory dimensions do not match the batch")

    repeated_u = trajectory[:, :, None].expand(-1, -1, scenarios, -1, -1)
    repeated_load = scenario_load[:, None].expand(
        -1, candidates, -1, -1, -1, -1,
    )
    state = completion(
        repeated_u.reshape(batch * candidates * scenarios * horizon, free_dim),
        repeated_load.reshape(
            batch * candidates * scenarios * horizon, *scenario_load.shape[3:],
        ),
    )

    physics = completion.physics
    physical_violation = physics.constraint_violation(state).reshape(
        batch, candidates, scenarios, horizon, -1,
    )
    category_sizes = physics.remaining_constraint_sizes()
    physical_categories = dict(zip(
        category_sizes,
        torch.split(physical_violation, tuple(category_sizes.values()), dim=-1),
    ))
    reference_pg = state["pg"].reshape(
        batch, candidates, scenarios, horizon, physics.n_gen,
    )[..., physics.reference_generator]
    delta_pg = reference_pg[:, :, :, 1:] - reference_pg[:, :, :, :-1]
    ramp_scale = reference_ramp.clamp_min(1e-8)
    ramp_violation = torch.stack([
        torch.relu((delta_pg - ramp_scale) / ramp_scale),
        torch.relu((-delta_pg - ramp_scale) / ramp_scale),
    ], dim=-1)
    violation_families = {**physical_categories, "ramp": ramp_violation}
    candidate_violation, baseline_feasibility, shaped = (
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

    hit_component = candidate_violation.amin(dim=1).mean()
    coverage_component = candidate_violation.mean()
    if args.objective_mode == "improved":
        # The summed mean-CVaR term supplies dense gradients, while this exact
        # worst residual aligns candidate ranking with strict feasibility.
        ranking_violation = (
            candidate_violation
            + args.worst_violation_weight * maximum_violation
        )
        loss_feasibility, hit_component, coverage_component = (
            hit_aligned_feasibility_loss(
                ranking_violation,
                args.feasibility_top_k,
                args.coverage_weight,
            )
        )
        loss_diversity, sigma_effective, selected_distance = (
            adaptive_topm_diversity_loss(
                trajectory,
                ranking_violation,
                free_lower,
                free_upper,
                args.diversity_top_k,
                args.adaptive_sigma_scale,
                args.adaptive_sigma_min,
                args.sigma_div,
            )
        )
    else:
        loss_feasibility = baseline_feasibility
        score = soft_feasibility_score(candidate_violation, args.tau_feas)
        loss_diversity = diversity_loss(
            trajectory,
            free_lower,
            free_upper,
            score,
            args.sigma_div,
            args.loss_epsilon,
        )
        sigma_effective = torch.as_tensor(
            args.sigma_div, dtype=trajectory.dtype, device=trajectory.device,
        )
        selected_distance = normalized_pairwise_distance(
            trajectory.detach(), free_lower, free_upper,
        ).mean()
    feasible = (maximum_violation <= args.feasibility_tolerance).detach()
    score = soft_feasibility_score(candidate_violation, args.tau_feas)
    loss = args.lambda_fea * loss_feasibility + args.lambda_div * loss_diversity

    scenario_pg = state["pg"].reshape(
        batch, candidates, scenarios, horizon, physics.n_gen,
    )
    objective = expected_generation_objective(scenario_pg, completion)
    metrics = {
        "loss": loss.detach(),
        "fea": loss_feasibility.detach(),
        "div": loss_diversity.detach(),
        "hit_component": hit_component.detach(),
        "coverage_component": coverage_component.detach(),
        "candidate_violation": candidate_violation.mean().detach(),
        "ranking_violation": (
            candidate_violation
            + args.worst_violation_weight * maximum_violation
        ).mean().detach(),
        "soft_score": score.mean().detach(),
        "feasible": feasible.float().mean().detach(),
        "hit": feasible.any(dim=1).float().mean().detach(),
        "max_violation": maximum_violation.mean().detach(),
        "best_max_violation": maximum_violation.amin(dim=1).mean().detach(),
        "worst_violation": maximum_violation.amax().detach(),
        "candidate_distance": normalized_pairwise_distance(
            trajectory.detach(), free_lower, free_upper,
        ).mean(),
        "selected_distance": selected_distance.detach(),
        "sigma_effective": sigma_effective.detach(),
        "violation_pg": shaped["pg"].mean().detach(),
        "violation_qg": shaped["qg"].mean().detach(),
        "violation_voltage": shaped["voltage"].mean().detach(),
        "violation_angle": shaped["angle"].mean().detach(),
        "violation_thermal": shaped["thermal"].mean().detach(),
        "violation_ramp": shaped["ramp"].mean().detach(),
        "cost_mean": objective.mean().detach(),
        "pf_residual": state["pf_residual"].mean().detach(),
    }
    return loss, metrics


def run_epoch(
    model: torch.nn.Module,
    completion: torch.nn.Module,
    loader: DataLoader,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    reference_ramp: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    latent_seed: int,
) -> dict[str, float]:
    """Run one Stage-1 epoch and aggregate losses and diagnostics.

    Parameters
    ----------
    model, completion, loader
        Generator, frozen DECS, and split-specific data loader.
    free_lower, free_upper, free_ramp, reference_ramp
        Physical decision and ramp tensors documented by :func:`compute_batch`.
    args : argparse.Namespace
        Validated experiment configuration.
    device : torch.device
        CPU or CUDA device.
    optimizer : torch.optim.Optimizer or None
        Adam during training; ``None`` selects validation.
    latent_seed : int
        Fixed validation latent seed; ignored during training.

    Returns
    -------
    dict[str, float]
        Instance-weighted metrics plus gradient clipping diagnostics.
    """
    training = optimizer is not None
    model.train(training)
    completion.eval()
    names = (
        "loss", "fea", "div", "hit_component", "coverage_component",
        "candidate_violation", "soft_score", "feasible", "hit",
        "ranking_violation",
        "max_violation", "best_max_violation",
        "candidate_distance", "selected_distance",
        "sigma_effective", "violation_pg", "violation_qg",
        "violation_voltage", "violation_angle", "violation_thermal",
        "violation_ramp", "cost_mean", "pf_residual",
    )
    totals = {name: 0.0 for name in names}
    worst_violation = 0.0
    seen = 0
    gradient_norm_sum = 0.0
    optimizer_steps = 0
    clipped_steps = 0
    latent_generator = None
    if not training:
        latent_generator = torch.Generator(device=device).manual_seed(latent_seed)

    if training:
        optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_number, (condition, scenario_load) in enumerate(loader, start=1):
            condition = condition.to(device)
            scenario_load = scenario_load.to(device)
            latent = None
            if latent_generator is not None:
                latent = torch.randn(
                    len(condition), args.candidates, model.latent_dim,
                    dtype=condition.dtype, device=device,
                    generator=latent_generator,
                )
            loss, metrics = compute_batch(
                model, completion, condition, scenario_load,
                free_lower, free_upper, free_ramp, reference_ramp,
                args, latent,
            )
            if training:
                group_start = (
                    (batch_number - 1) // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                group_size = min(
                    args.gradient_accumulation_steps, n_batches - group_start,
                )
                (loss / group_size).backward()
                update_due = (
                    batch_number % args.gradient_accumulation_steps == 0
                    or batch_number == n_batches
                )
                if update_due:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    gradient_norm_sum += float(grad_norm)
                    optimizer_steps += 1
                    clipped_steps += int(float(grad_norm) > args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            count = len(condition)
            seen += count
            for name in names:
                totals[name] += float(metrics[name]) * count
            worst_violation = max(worst_violation, float(metrics["worst_violation"]))

    result = {name: value / seen for name, value in totals.items()}
    result["worst_violation"] = worst_violation
    result["gradient_norm"] = (
        gradient_norm_sum / optimizer_steps if optimizer_steps else 0.0
    )
    result["clip_fraction"] = (
        clipped_steps / optimizer_steps if optimizer_steps else 0.0
    )
    return result


def stage1_selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    """Return a lexicographic key prioritizing hit-rate over proxy loss.

    Parameters
    ----------
    metrics : dict[str, float]
        Validation metrics returned by :func:`run_epoch`.

    Returns
    -------
    tuple[float, ...]
        Larger is better: hit-rate, candidate-feasible rate, negative maximum
        violation, negative candidate violation, then negative training loss.
    """
    return (
        metrics["hit"],
        metrics["feasible"],
        -metrics["best_max_violation"],
        -metrics["candidate_violation"],
        -metrics["loss"],
    )


def build_parser() -> argparse.ArgumentParser:
    """Declare the reproducible improved Stage-1 experiment interface."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / "e2e118_N5000_S20_T16",
        help="SMPC dataset with saved train/validation/test split",
    )
    parser.add_argument(
        "--decs", type=Path,
        default=Path(__file__).resolve().parent / "decs_pgm_fixedpv.pt",
        help="frozen DECS checkpoint supplying differentiable AC states",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "generator_tcn_improved_stage1.pt",
        help="best hit-aligned Stage-1 checkpoint",
    )
    parser.add_argument(
        "--objective-mode", choices=("improved", "baseline"), default="improved",
        help="improved top-K/adaptive objective or the original Stage-1 objective",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Stage-1 epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="SMPC instances per micro-batch")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=8,
        help="micro-batches averaged before each Adam update",
    )
    parser.add_argument("--candidates", type=int, default=50, help="trajectory candidates K")
    parser.add_argument("--hidden-channels", type=int, default=64, help="TCN feature width")
    parser.add_argument("--latent-dim", type=int, default=16, help="trajectory latent dimension")
    parser.add_argument("--latent-embedding-dim", type=int, default=32, help="latent embedding width")
    parser.add_argument("--kernel-size", type=int, default=3, help="odd temporal convolution width")
    parser.add_argument("--dilations", type=int, nargs="+", default=(1, 2, 4, 8), help="TCN dilations")
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="symmetric generator ramp as a fraction of Pmax per period",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--lambda-fea", type=float, default=1.0, help="weight of the selected feasibility loss")
    parser.add_argument("--lambda-div", type=float, default=0.1, help="weight of diversity similarity")
    parser.add_argument(
        "--feasibility-top-k", type=int, default=5,
        help="lowest-violation candidates averaged by the improved hit component",
    )
    parser.add_argument(
        "--coverage-weight", type=float, default=0.1,
        help="all-candidate mean-violation weight inside improved feasibility",
    )
    parser.add_argument(
        "--worst-violation-weight", type=float, default=10.0,
        help="weight of each candidate's exact worst normalized inequality residual",
    )
    parser.add_argument(
        "--diversity-top-k", type=int, default=10,
        help="lowest-violation candidates used by improved diversity",
    )
    parser.add_argument(
        "--adaptive-sigma-scale", type=float, default=1.0,
        help="multiplier on detached median top-M candidate distance",
    )
    parser.add_argument(
        "--adaptive-sigma-min", type=float, default=1e-4,
        help="minimum adaptive Gaussian width in normalized trajectory units",
    )
    parser.add_argument(
        "--sigma-div", type=float, default=0.1,
        help="baseline fixed sigma or improved adaptive-sigma upper bound",
    )
    parser.add_argument("--tau-feas", type=float, default=2.0, help="reporting and baseline feasibility-score temperature")
    parser.add_argument("--mean-cvar-alpha", type=float, default=0.05, help="mean weight within each constraint family")
    parser.add_argument("--cvar-tail-fraction", type=float, default=0.05, help="upper-tail fraction within each constraint family")
    parser.add_argument("--loss-epsilon", type=float, default=1e-8, help="baseline diversity denominator epsilon")
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-4, help="reporting-only normalized feasibility threshold")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="positive global gradient clipping norm")
    parser.add_argument(
        "--detect-anomaly", action="store_true",
        help="enable expensive autograd anomaly tracing and residual checks",
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.5, help="plateau learning-rate multiplier")
    parser.add_argument("--lr-decay-patience", type=int, default=5, help="validation-loss plateau epochs")
    parser.add_argument("--min-learning-rate", type=float, default=5e-5, help="learning-rate floor")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=2026, help="model, shuffle, and latent seed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="training device")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate improved Stage-1 dimensions, weights, paths, and device."""
    positive_ints = (
        args.epochs, args.batch_size, args.gradient_accumulation_steps,
        args.candidates, args.hidden_channels, args.latent_dim,
        args.latent_embedding_dim, args.lr_decay_patience,
    )
    if min(positive_ints) < 1:
        raise ValueError("epochs, batch and model dimensions must be positive")
    if not 1 <= args.feasibility_top_k <= args.candidates:
        raise ValueError("feasibility_top_k must lie in [1,candidates]")
    if not 2 <= args.diversity_top_k <= args.candidates:
        raise ValueError("diversity_top_k must lie in [2,candidates]")
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not args.dilations or min(args.dilations) < 1:
        raise ValueError("dilations must be positive")
    if min(
        args.learning_rate, args.lambda_fea, args.sigma_div,
        args.adaptive_sigma_scale, args.adaptive_sigma_min,
        args.tau_feas, args.max_grad_norm,
    ) <= 0:
        raise ValueError("learning rate, active weights, temperatures and clipping must be positive")
    if min(
        args.lambda_div, args.coverage_weight,
        args.worst_violation_weight, args.feasibility_tolerance,
    ) < 0:
        raise ValueError("loss weights and tolerance must be nonnegative")
    if args.adaptive_sigma_min > args.sigma_div:
        raise ValueError("adaptive_sigma_min must not exceed sigma_div")
    if not 0 < args.mean_cvar_alpha < 1 or not 0 < args.cvar_tail_fraction <= 1:
        raise ValueError("mean-CVaR parameters must lie in their open/closed unit ranges")
    if not 0 < args.lr_decay_factor < 1:
        raise ValueError("lr_decay_factor must lie in (0,1)")
    if not args.data.is_dir() or not args.decs.is_file():
        raise FileNotFoundError("SMPC dataset or DECS checkpoint does not exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def main() -> None:
    """Train and save the best Stage-1 model under hit-aligned validation."""
    args = build_parser().parse_args()
    validate_arguments(args)
    torch.autograd.set_detect_anomaly(False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    split_indices = {
        name: np.load(args.data / "split" / f"{name}_indices.npy")
        for name in ("train", "validation", "test")
    }
    feature_min, feature_max = compute_training_feature_bounds(
        args.data, split_indices["train"],
    )
    datasets = {
        name: GeneratorDataset(args.data, indices, feature_min, feature_max)
        for name, indices in split_indices.items() if name != "test"
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for name, dataset in datasets.items()
    }

    completion = load_decs_checkpoint(args.decs, device)
    for parameter in completion.parameters():
        parameter.requires_grad_(False)
    physics = completion.physics
    if metadata.get("case_name") != "pglib_opf_case118_ieee":
        raise ValueError("improved trainer currently expects PGLib IEEE-118")
    free_lower, free_upper = physics.free_variable_bounds(1)
    free_lower, free_upper = free_lower[0].to(device), free_upper[0].to(device)
    free_ramp = args.ramp_fraction * physics.pg_max[physics.free_pg]
    reference_ramp = args.ramp_fraction * physics.pg_max[physics.reference_generator]

    spec = get_benchmark_spec("csng")
    model = build_generator_model(
        "csng",
        condition_dim=datasets["train"].condition_dim,
        free_dim=physics.free_dim,
        n_free_pg=physics.n_free_pg,
        projection_method=str(spec["projection_method"]),
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        latent_embedding_dim=args.latent_embedding_dim,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
    ).to(device)
    if model.receptive_field < datasets["train"].horizon:
        raise ValueError("TCN receptive field is shorter than the horizon")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = build_plateau_scheduler(
        optimizer,
        args.lr_decay_factor,
        args.lr_decay_patience,
        0.0,
        args.min_learning_rate,
    )
    print(
        f"Improved Generator Stage-1 | mode={args.objective_mode} | device={device} | "
        f"N={len(datasets['train'])}/{len(datasets['validation'])} | "
        f"K={args.candidates}, S={datasets['train'].n_scenarios}, "
        f"T={datasets['train'].horizon}",
        flush=True,
    )
    print(
        f"  lambda_fea={args.lambda_fea:g}, lambda_div={args.lambda_div:g}, "
        f"fea_top_k={args.feasibility_top_k}, coverage={args.coverage_weight:g}, "
        f"worst_weight={args.worst_violation_weight:g}, "
        f"div_top_m={args.diversity_top_k}, adaptive_sigma="
        f"[{args.adaptive_sigma_min:g},{args.sigma_div:g}]",
        flush=True,
    )

    history: list[dict[str, float | int]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model, completion, loaders["train"],
            free_lower, free_upper, free_ramp, reference_ramp,
            args, device, optimizer, args.seed + 11,
        )
        validation_metrics = run_epoch(
            model, completion, loaders["validation"],
            free_lower, free_upper, free_ramp, reference_ramp,
            args, device, None, args.seed + 11,
        )
        scheduler.step(validation_metrics["loss"])
        key = stage1_selection_key(validation_metrics)
        improved = best_key is None or key > best_key
        record = {
            "epoch": epoch,
            "learning_rate": lr,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        }
        history.append(record)
        if improved:
            best_key = key
            best_epoch = epoch
            checkpoint = {
                "benchmark": "csng",
                "method": "CSNG-Improved-Stage1",
                "model_class": str(spec["model_class"]),
                "model_config": model.configuration(),
                "model_state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "training_phase": "stage1_hit_aligned_feasibility_diversity",
                "objective_mode": args.objective_mode,
                "normalization": {
                    "feature_min": feature_min,
                    "feature_max": feature_max,
                },
                "free_variable_lower": free_lower.detach().cpu(),
                "free_variable_upper": free_upper.detach().cpu(),
                "free_ramp_mw_per_period": free_ramp.detach().cpu(),
                "reference_ramp_mw_per_period": reference_ramp.detach().cpu(),
                "ramp_fraction_of_pmax": args.ramp_fraction,
                "decs_checkpoint": str(args.decs.resolve()),
                "decs_checkpoint_sha256": file_sha256(args.decs),
                "loss_definition": (
                    "topk_hit_plus_coverage_adaptive_topm_diversity_v1"
                    if args.objective_mode == "improved"
                    else "original_hierarchical_feasibility_diversity"
                ),
                "loss_hyperparameters": {
                    "lambda_fea": args.lambda_fea,
                    "lambda_div": args.lambda_div,
                    "feasibility_top_k": args.feasibility_top_k,
                    "coverage_weight": args.coverage_weight,
                    "worst_violation_weight": args.worst_violation_weight,
                    "diversity_top_k": args.diversity_top_k,
                    "adaptive_sigma_scale": args.adaptive_sigma_scale,
                    "adaptive_sigma_min": args.adaptive_sigma_min,
                    "sigma_div": args.sigma_div,
                    "tau_feas": args.tau_feas,
                    "mean_cvar_alpha": args.mean_cvar_alpha,
                    "cvar_tail_fraction": args.cvar_tail_fraction,
                    "feasibility_tolerance": args.feasibility_tolerance,
                },
                "training_hyperparameters": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "candidates": args.candidates,
                    "learning_rate": args.learning_rate,
                },
                "checkpoint_selection": (
                    "lexicographic validation hit, candidate feasibility, "
                    "negative max violation, negative mean violation, negative loss"
                ),
                "best_epoch": best_epoch,
                "best_validation_metrics": dict(validation_metrics),
                "training_history": list(history),
                "split": {
                    "ratio": metadata["split"]["ratio"],
                    "seed": metadata["split"]["seed"],
                    "counts": {name: len(value) for name, value in split_indices.items()},
                    "indices": {
                        name: value.astype(np.int64).tolist()
                        for name, value in split_indices.items()
                    },
                },
                "data_metadata": metadata,
                "seed": args.seed,
                "test_evaluated_during_training": False,
            }
            atomic_torch_save(checkpoint, args.output)

        elapsed = perf_counter() - start
        eta = elapsed / epoch * (args.epochs - epoch)
        print(
            f"[epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
            f"{' | best' if improved else ''}]",
            flush=True,
        )
        for split, values in (("train", train_metrics), ("validation", validation_metrics)):
            print(
                f"  {split:10s} total={values['loss']:.4e} "
                f"fea={values['fea']:.4e} div={values['div']:.4e} | "
                f"hit={values['hit']:.3f} cand={values['feasible']:.3f} "
                f"Vmean={values['candidate_violation']:.3e} "
                f"Vmax={values['max_violation']:.3e} "
                f"bestVmax={values['best_max_violation']:.3e} | "
                f"distance={values['candidate_distance']:.3e} "
                f"sigma={values['sigma_effective']:.3e} "
                f"score={values['soft_score']:.3e}",
                flush=True,
            )
        print(
            f"  constraint: Pg={validation_metrics['violation_pg']:.2e} "
            f"Qg={validation_metrics['violation_qg']:.2e} "
            f"V={validation_metrics['violation_voltage']:.2e} "
            f"angle={validation_metrics['violation_angle']:.2e} "
            f"thermal={validation_metrics['violation_thermal']:.2e} "
            f"ramp={validation_metrics['violation_ramp']:.2e} | "
            f"train_grad={train_metrics['gradient_norm']:.2e} "
            f"clip={train_metrics['clip_fraction']:.2f}",
            flush=True,
        )

    print(
        f"saved best epoch {best_epoch} to {args.output}; test set was not evaluated",
        flush=True,
    )


if __name__ == "__main__":
    main()
