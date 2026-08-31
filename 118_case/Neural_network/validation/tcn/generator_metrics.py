"""Compute Generator-TCN economic, feasibility, and diversity distributions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from Neural_network.decs import DifferentiableEqualityCompletion
from Neural_network.noncausal_tcn import ConditionalStochasticTCN


def _to_numpy(parts: Iterable[torch.Tensor]) -> np.ndarray:
    """Concatenate detached CPU tensors from successive validation batches.

    Parameters
    ----------
    parts : iterable of torch.Tensor
        Nonempty tensors sharing every dimension except the first.

    Returns
    -------
    np.ndarray
        Concatenated NumPy values in loader order.
    """
    return torch.cat(list(parts), dim=0).numpy()


def evaluate_generator_distributions(
    model: ConditionalStochasticTCN,
    completion: DifferentiableEqualityCompletion,
    loader: DataLoader,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    active_ramp: torch.Tensor,
    candidates: int,
    feasibility_tolerance: float,
    latent_seed: int,
    device: torch.device,
    progress_every: int = 10,
) -> dict[str, np.ndarray]:
    """Evaluate all test candidates with the frozen DECS completion model.

    Parameters
    ----------
    model : ConditionalStochasticTCN
        Trained stochastic non-causal temporal generator.
    completion : DifferentiableEqualityCompletion
        Frozen DECS network and AC state reconstruction used during training.
    loader : torch.utils.data.DataLoader
        Non-shuffled test loader yielding conditions and scenario loads.
    free_lower, free_upper : torch.Tensor, shape (free_dim,)
        Static dispatch bounds in MW for active power and p.u. for voltages.
    free_ramp : torch.Tensor, shape (n_free_pg,)
        Generator output-map ramp limit in MW per period.
    active_ramp : torch.Tensor, shape (n_active_generators,)
        Validation ramp limits in MW per period for all active generators.
    candidates : int
        Number of latent trajectories generated per scenario tree; at least two.
    feasibility_tolerance : float
        Largest accepted normalized positive inequality violation.
    latent_seed : int
        Seed for reproducible trajectory-level Gaussian latent variables.
    device : torch.device
        CPU or CUDA inference device.
    progress_every : int, default=10
        Print progress after this many validation batches.

    Returns
    -------
    dict[str, np.ndarray]
        Raw candidate, instance, constraint-category, cost, and pairwise-distance
        distributions. Candidate feasibility is DECS-based, not an exact
        pandapower certificate.
    """
    if candidates < 2 or progress_every < 1:
        raise ValueError("candidates must be at least two and progress_every positive")
    if feasibility_tolerance < 0:
        raise ValueError("feasibility_tolerance must be nonnegative")

    model.eval()
    completion.eval()
    latent_generator = torch.Generator(device=device).manual_seed(latent_seed)
    pair_indices = torch.triu_indices(candidates, candidates, offset=1, device=device)
    free_span = (free_upper - free_lower).clamp_min(1e-8)
    physics = completion.physics
    category_sizes = physics.remaining_constraint_sizes()
    category_names = tuple(category_sizes)
    collected: dict[str, list[torch.Tensor]] = {
        "candidate_cost": [],
        "best_candidate_cost": [],
        "best_feasible_cost": [],
        "candidate_feasible": [],
        "candidate_mean_violation": [],
        "candidate_max_violation": [],
        "pairwise_first_stage_distance": [],
        "pairwise_trajectory_distance": [],
        "instance_first_stage_diversity": [],
        "instance_trajectory_diversity": [],
        "first_stage_std_by_variable": [],
        **{f"violation_{name}": [] for name in (*category_names, "ramp")},
    }

    total_batches = len(loader)
    with torch.inference_mode():
        for batch_number, (condition, scenario_load) in enumerate(loader, start=1):
            condition = condition.to(device)
            scenario_load = scenario_load.to(device)
            batch, scenarios, horizon = scenario_load.shape[:3]
            latent = torch.randn(
                batch, candidates, model.latent_dim,
                dtype=condition.dtype, device=device, generator=latent_generator,
            )
            trajectory, admissible_lower, admissible_upper = model(
                condition, candidates, free_lower, free_upper,
                free_ramp, free_ramp, latent,
            )
            if not torch.isfinite(trajectory).all():
                raise FloatingPointError(f"non-finite TCN output in batch {batch_number}")

            candidate_scenarios = trajectory[:, :, None].expand(
                -1, -1, scenarios, -1, -1,
            )
            repeated_load = scenario_load[:, None].expand(
                -1, candidates, -1, -1, -1, -1,
            )
            state = completion(
                candidate_scenarios.reshape(
                    batch * candidates * scenarios * horizon, physics.free_dim,
                ),
                repeated_load.reshape(
                    batch * candidates * scenarios * horizon,
                    *scenario_load.shape[3:],
                ),
            )
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise FloatingPointError(f"non-finite DECS state in batch {batch_number}")

            physical_violation = physics.constraint_violation(state).reshape(
                batch, candidates, scenarios, horizon, -1,
            )
            if sum(category_sizes.values()) != physical_violation.shape[-1]:
                raise RuntimeError("constraint-category widths do not match DECS output")
            category_tensors = dict(zip(
                category_names,
                torch.split(physical_violation, tuple(category_sizes.values()), dim=-1),
            ))

            active_pg = state["pg"].reshape(
                batch, candidates, scenarios, horizon, physics.n_gen,
            )[..., physics.active]
            delta_pg = active_pg[:, :, :, 1:] - active_pg[:, :, :, :-1]
            ramp_scale = active_ramp.view(1, 1, 1, 1, -1).clamp_min(1e-8)
            ramp_violation = torch.relu(torch.cat([
                (delta_pg - ramp_scale) / ramp_scale,
                (-delta_pg - ramp_scale) / ramp_scale,
            ], dim=4))
            all_violation = torch.cat([
                physical_violation.flatten(start_dim=2),
                ramp_violation.flatten(start_dim=2),
            ], dim=2)
            candidate_mean_violation = all_violation.mean(dim=2)
            candidate_max_violation = all_violation.amax(dim=2)
            candidate_feasible = candidate_max_violation <= feasibility_tolerance

            period_cost = physics.generation_cost(state["pg"]).reshape(
                batch, candidates, scenarios, horizon,
            )
            candidate_cost = period_cost.mean(dim=2).sum(dim=2)
            feasible_cost = torch.where(
                candidate_feasible, candidate_cost,
                torch.full_like(candidate_cost, torch.inf),
            ).amin(dim=1)
            feasible_cost = torch.where(
                torch.isfinite(feasible_cost), feasible_cost,
                torch.full_like(feasible_cost, torch.nan),
            )

            first_center = 0.5 * (admissible_lower[:, :, 0] + admissible_upper[:, :, 0])
            first_half_range = 0.5 * (
                admissible_upper[:, :, 0] - admissible_lower[:, :, 0]
            ).clamp_min(1e-8)
            first_normalized = (trajectory[:, :, 0] - first_center) / first_half_range
            first_distances = torch.cdist(first_normalized, first_normalized)[
                :, pair_indices[0], pair_indices[1]
            ]

            global_normalized = (trajectory - free_lower) / free_span
            flattened_trajectory = global_normalized.flatten(start_dim=2)
            trajectory_distances = torch.cdist(
                flattened_trajectory, flattened_trajectory,
            )[:, pair_indices[0], pair_indices[1]] / np.sqrt(horizon * physics.free_dim)

            collected["candidate_cost"].append(candidate_cost.cpu())
            collected["best_candidate_cost"].append(candidate_cost.amin(dim=1).cpu())
            collected["best_feasible_cost"].append(feasible_cost.cpu())
            collected["candidate_feasible"].append(candidate_feasible.cpu())
            collected["candidate_mean_violation"].append(candidate_mean_violation.cpu())
            collected["candidate_max_violation"].append(candidate_max_violation.cpu())
            collected["pairwise_first_stage_distance"].append(first_distances.cpu())
            collected["pairwise_trajectory_distance"].append(trajectory_distances.cpu())
            collected["instance_first_stage_diversity"].append(
                first_distances.mean(dim=1).cpu(),
            )
            collected["instance_trajectory_diversity"].append(
                trajectory_distances.mean(dim=1).cpu(),
            )
            collected["first_stage_std_by_variable"].append(
                global_normalized[:, :, 0].std(dim=1, unbiased=False).cpu(),
            )
            for name, values in category_tensors.items():
                collected[f"violation_{name}"].append(values.flatten(start_dim=2).amax(dim=2).cpu())
            collected["violation_ramp"].append(
                ramp_violation.flatten(start_dim=2).amax(dim=2).cpu(),
            )

            if batch_number % progress_every == 0 or batch_number == total_batches:
                completed = min(batch_number * loader.batch_size, len(loader.dataset))
                print(
                    f"\rTCN validation: {completed}/{len(loader.dataset)} instances",
                    end="", flush=True,
                )
    print()

    result = {name: _to_numpy(parts) for name, parts in collected.items()}
    result["candidate_feasible"] = result["candidate_feasible"].astype(bool)
    return result
