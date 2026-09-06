"""Evaluate Generator-TCN candidates with PGM/fixed-PV batches.

The data split, Generator checkpoint, latent seeds, operating constraints, and
cost definition match :mod:`evaluate_generator_tcn`.  For each instance all
candidate/scenario/time operating points are flattened into one native PGM
batch. Native failures or different solution branches use the matching
fixed-PV high-voltage states. A deterministic sample is independently
recomputed by pandapower and saved as a numerical equivalence report.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from Data_generation.case118_pglib import Case118, load_case118
from Data_generation.generate_decs_dataset import build_pandapower_network
from Neural_network.generator_benchmarks import get_benchmark_spec
from evaluate_generator_tcn import (
    DEFAULT_DATA,
    build_condition,
    checkpoint_split_indices,
    finite_statistics,
    format_reported_cost,
    load_generator,
    solve_exact_power_flow,
    summarize_solution_costs,
    write_csv,
    write_json,
)
from pgm_batch_power_flow import build_pgm_case, solve_pv_batch


ROOT = Path(__file__).resolve().parent


def flatten_candidate_points(
    current_load: np.ndarray,
    future_load: np.ndarray,
    schedules: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``K`` shared trajectories into PGM batch operating points.

    The point order inside each candidate is ``current`` followed by
    ``scenario-major future time``.  The current point is solved only once.
    """
    candidates, horizon, free_dim = schedules.shape
    scenarios, future_steps = future_load.shape[:2]
    if horizon != future_steps + 1:
        raise ValueError("schedule and future-load horizons differ")
    point_loads = np.concatenate(
        [current_load[None], future_load.reshape(-1, *current_load.shape)], axis=0,
    )
    loads = np.broadcast_to(
        point_loads[None], (candidates, len(point_loads), *current_load.shape),
    ).reshape(-1, *current_load.shape)
    future_controls = np.broadcast_to(
        schedules[:, None, 1:, :],
        (candidates, scenarios, future_steps, free_dim),
    ).reshape(candidates, scenarios * future_steps, free_dim)
    controls = np.concatenate([schedules[:, :1], future_controls], axis=1)
    return loads, controls.reshape(-1, free_dim)


def _generation_costs(case: Case118, pg: np.ndarray) -> np.ndarray:
    """Vectorized one-period quadratic generation cost."""
    return np.sum(
        case.gencost[:, 4] * pg**2
        + case.gencost[:, 5] * pg
        + case.gencost[:, 6],
        axis=-1,
    )


def _balance_residuals(
    case: Case118,
    loads: np.ndarray,
    pg: np.ndarray,
    qg: np.ndarray,
    vm: np.ndarray,
    va: np.ndarray,
) -> np.ndarray:
    """Independently recompute the maximum nodal AC residual in MVA."""
    voltage = vm * np.exp(1j * va)
    injection = voltage * np.conj(np.einsum("ij,bj->bi", case.ybus, voltage))
    injection *= case.base_mva
    pd = np.zeros((len(loads), case.n_bus))
    qd = np.zeros_like(pd)
    pd[:, case.load_buses] = loads[..., 0]
    qd[:, case.load_buses] = loads[..., 1]
    pgen = np.zeros_like(pd)
    qgen = np.zeros_like(qd)
    for generator, bus in enumerate(case.generator_buses):
        pgen[:, bus] += pg[:, generator]
        qgen[:, bus] += qg[:, generator]
    return np.maximum(
        np.max(np.abs(pgen - pd - injection.real), axis=1),
        np.max(np.abs(qgen - qd - injection.imag), axis=1),
    )


def assess_candidate_batch(
    case: Case118,
    loads: np.ndarray,
    states: dict[str, np.ndarray | int | float],
    candidates: int,
    scenarios: int,
    horizon: int,
    ramp_fraction: float,
    feasibility_tolerance: float,
    balance_tolerance_mva: float,
) -> dict[str, np.ndarray]:
    """Compute cost and exact constraints for all candidates without loops."""
    points = 1 + scenarios * (horizon - 1)
    pg_flat = np.asarray(states["pg"])
    qg_flat = np.asarray(states["qg"])
    vm_flat = np.asarray(states["vm"])
    va_flat = np.asarray(states["va"])
    pf_flat = np.asarray(states["pf"])
    qf_flat = np.asarray(states["qf"])
    pt_flat = np.asarray(states["pt"])
    qt_flat = np.asarray(states["qt"])
    if len(pg_flat) != candidates * points:
        raise ValueError("PGM state count does not match candidate point count")

    pg = pg_flat.reshape(candidates, points, case.n_gen)
    qg = qg_flat.reshape(candidates, points, case.n_gen)
    vm = vm_flat.reshape(candidates, points, case.n_bus)
    va = va_flat.reshape(candidates, points, case.n_bus)

    active = case.active_generators
    pg_min, pg_max = case.gen[active, 9], case.gen[active, 8]
    pg_scale = np.maximum(pg_max - pg_min, 1.0e-12)
    pg_violation = np.maximum.reduce([
        np.zeros_like(pg[..., active]),
        (pg[..., active] - pg_max) / pg_scale,
        (pg_min - pg[..., active]) / pg_scale,
    ]).max(axis=(1, 2))

    qg_min, qg_max = case.gen[:, 4], case.gen[:, 3]
    qg_scale = np.maximum(qg_max - qg_min, 1.0e-12)
    qg_violation = np.maximum.reduce([
        np.zeros_like(qg),
        (qg - qg_max) / qg_scale,
        (qg_min - qg) / qg_scale,
    ]).max(axis=(1, 2))

    vm_min, vm_max = case.bus[:, 12], case.bus[:, 11]
    vm_scale = np.maximum(vm_max - vm_min, 1.0e-12)
    voltage_violation = np.maximum.reduce([
        np.zeros_like(vm),
        (vm - vm_max) / vm_scale,
        (vm_min - vm) / vm_scale,
    ]).max(axis=(1, 2))

    fbus = case.branch[:, 0].astype(int) - 1
    tbus = case.branch[:, 1].astype(int) - 1
    angle = va[..., fbus] - va[..., tbus]
    angle_min = np.deg2rad(case.branch[:, 11])
    angle_max = np.deg2rad(case.branch[:, 12])
    angle_scale = np.maximum(angle_max - angle_min, 1.0e-12)
    angle_violation = np.maximum.reduce([
        np.zeros_like(angle),
        (angle - angle_max) / angle_scale,
        (angle_min - angle) / angle_scale,
    ]).max(axis=(1, 2))

    apparent_from = np.hypot(pf_flat, qf_flat).reshape(candidates, points, -1)
    apparent_to = np.hypot(pt_flat, qt_flat).reshape(candidates, points, -1)
    thermal_violation = np.maximum(
        np.maximum(apparent_from, apparent_to) / case.branch[:, 5] - 1.0, 0.0,
    ).max(axis=(1, 2))

    pg_future = pg[:, 1:].reshape(candidates, scenarios, horizon - 1, case.n_gen)
    pg_trajectory = np.concatenate([
        np.broadcast_to(
            pg[:, 0, None, None, :],
            (candidates, scenarios, 1, case.n_gen),
        ),
        pg_future,
    ], axis=2)
    ramp_limit = ramp_fraction * case.gen[active, 8]
    ramp_violation = np.maximum(
        np.abs(np.diff(pg_trajectory[..., active], axis=2))
        / np.maximum(ramp_limit, 1.0e-12) - 1.0,
        0.0,
    ).max(axis=(1, 2, 3))

    point_cost = _generation_costs(case, pg)
    objective = point_cost[:, 0] + point_cost[:, 1:].reshape(
        candidates, scenarios, horizon - 1,
    ).sum(axis=2).mean(axis=1)
    balance = _balance_residuals(case, loads, pg_flat, qg_flat, vm_flat, va_flat)
    balance = balance.reshape(candidates, points).max(axis=1)
    pv_converged = np.asarray(states["pv_converged"]).reshape(candidates, points).all(axis=1)

    violations = np.column_stack([
        pg_violation, qg_violation, voltage_violation,
        angle_violation, thermal_violation, ramp_violation,
    ])
    maximum_violation = violations.max(axis=1)
    # A candidate containing any failed PGM point has no valid exact objective.
    # Keep the remaining candidates usable instead of aborting the whole batch.
    objective = np.where(pv_converged & np.isfinite(objective), objective, np.inf)
    maximum_violation = np.where(
        pv_converged & np.isfinite(maximum_violation), maximum_violation, np.inf,
    )
    balance = np.where(pv_converged & np.isfinite(balance), balance, np.inf)
    feasible = (
        pv_converged
        & (maximum_violation <= feasibility_tolerance)
        & (balance <= balance_tolerance_mva)
    )
    return {
        "feasible": feasible,
        "objective": objective,
        "maximum_violation": maximum_violation,
        "max_pf_residual_mva": balance,
        "violation_pg": pg_violation,
        "violation_qg": qg_violation,
        "violation_voltage": voltage_violation,
        "violation_angle": angle_violation,
        "violation_thermal": thermal_violation,
        "violation_ramp": ramp_violation,
        "pg_trajectory": pg_trajectory,
        "qg_trajectory": np.concatenate([
            np.broadcast_to(
                qg[:, 0, None, None, :],
                (candidates, scenarios, 1, case.n_gen),
            ),
            qg[:, 1:].reshape(candidates, scenarios, horizon - 1, case.n_gen),
        ], axis=2),
        "vm_trajectory": np.concatenate([
            np.broadcast_to(
                vm[:, 0, None, None, :],
                (candidates, scenarios, 1, case.n_bus),
            ),
            vm[:, 1:].reshape(candidates, scenarios, horizon - 1, case.n_bus),
        ], axis=2),
        "va_trajectory": np.concatenate([
            np.broadcast_to(
                va[:, 0, None, None, :],
                (candidates, scenarios, 1, case.n_bus),
            ),
            va[:, 1:].reshape(candidates, scenarios, horizon - 1, case.n_bus),
        ], axis=2),
    }


def compare_with_pandapower(
    case: Case118,
    loads: np.ndarray,
    controls: np.ndarray,
    states: dict[str, np.ndarray | int | float],
    point_count: int,
    pf_tolerance_mva: float,
    max_pf_iterations: int,
) -> dict:
    """Compare a deterministic spread of PGM batch points with pandapower."""
    if point_count <= 0:
        return {"points_requested": 0, "points_converged": 0, "passed": None}
    indices = np.unique(np.linspace(0, len(loads) - 1, point_count, dtype=int))
    net = build_pandapower_network(case)
    metric_names = ("vm", "va", "pg", "qg", "pf", "qf", "pt", "qt")
    absolute_errors = {name: [] for name in metric_names}
    point_rows = []
    for point in indices:
        pp_state, attempts = solve_exact_power_flow(
            net, case, loads[point], controls[point], False,
            pf_tolerance_mva, max_pf_iterations,
        )
        if pp_state is None:
            point_rows.append({"batch_point": int(point), "converged": False})
            continue
        row = {"batch_point": int(point), "converged": True, "pp_attempts": attempts}
        for name in metric_names:
            pgm_value = np.asarray(states[name])[point]
            pp_value = np.asarray(pp_state[name])
            error = float(np.max(np.abs(pgm_value - pp_value)))
            absolute_errors[name].append(error)
            row[f"max_abs_{name}"] = error
        point_rows.append(row)

    maxima = {
        name: (float(max(values)) if values else None)
        for name, values in absolute_errors.items()
    }
    power_metrics = ("pg", "qg", "pf", "qf", "pt", "qt")
    all_converged = bool(point_rows) and all(row["converged"] for row in point_rows)
    passed = bool(
        all_converged
        and maxima["vm"] is not None
        and maxima["vm"] <= 1.0e-6
        and maxima["va"] <= 1.0e-6
        and max(maxima[name] for name in power_metrics) <= 1.0e-4
    )
    return {
        "points_requested": int(len(indices)),
        "points_converged": int(sum(row["converged"] for row in point_rows)),
        "acceptance": {
            "voltage_magnitude_pu": 1.0e-6,
            "voltage_angle_rad": 1.0e-6,
            "generator_and_branch_power_mva": 1.0e-4,
        },
        "maximum_absolute_error": maxima,
        "passed": passed,
        "points": point_rows,
    }


def main(benchmark: str = "csng") -> None:
    """Generate, batch-screen, and verify one benchmark on a held-out split.

    Parameters
    ----------
    benchmark : str, default="csng"
        Experiment key selecting CSNG, S-CSNG, or WD-CSNG.
    """
    spec = get_benchmark_spec(benchmark)
    parser = argparse.ArgumentParser(
        description=f"Evaluate {spec['method']} with PGM/fixed-PV batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "Neural_network" / str(spec["checkpoint"]),
        help="trained checkpoint for the selected benchmark",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max-instances", type=int, default=200)
    parser.add_argument("--ramp-fraction", type=float, default=0.25)
    parser.add_argument("--feasibility-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--balance-tolerance-mva", type=float, default=1.0e-5)
    parser.add_argument("--pgm-error-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-pf-iterations", type=int, default=50)
    parser.add_argument(
        "--fixed-pv-chunk-size", type=int, default=1024,
        help="maximum fixed-PV Newton points per float64 dense batch",
    )
    parser.add_argument(
        "--threading", type=int, default=0,
        help="PGM workers: -1 sequential, 0 all hardware threads, >0 exact count",
    )
    parser.add_argument(
        "--verify-instances", type=int, default=1,
        help="number of initial instances sampled for pandapower equivalence",
    )
    parser.add_argument(
        "--verify-points", type=int, default=500,
        help="spread of PGM batch points compared per verification instance; 0 disables",
    )
    parser.add_argument("--pandapower-tolerance-mva", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (
        args.candidates < 1
        or args.max_pf_iterations < 1
        or args.fixed_pv_chunk_size < 1
    ):
        raise ValueError("candidate and iteration counts must be positive")
    if args.verify_instances < 0 or args.verify_points < 0:
        raise ValueError("verification counts cannot be negative")
    if not args.data.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("base dataset or Generator-TCN checkpoint is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir is None:
        args.output_dir = (
            ROOT / "output" / f"{spec['pgm_output_prefix']}_{args.split}"
        )

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model, checkpoint = load_generator(args.checkpoint, device, benchmark)
    normalization = checkpoint["normalization"]
    feature_min = np.asarray(normalization["feature_min"], dtype=float)
    feature_max = np.asarray(normalization["feature_max"], dtype=float)
    free_lower = torch.as_tensor(checkpoint["free_variable_lower"], device=device)
    free_upper = torch.as_tensor(checkpoint["free_variable_upper"], device=device)
    free_ramp = torch.as_tensor(checkpoint["free_ramp_mw_per_period"], device=device)

    current = np.load(args.data / "current_load.npy", mmap_mode="r")
    pool = np.load(args.data / "future_pool.npy", mmap_mode="r")
    future = np.load(args.data / "future_load.npy", mmap_mode="r")
    instance_indices = checkpoint_split_indices(
        checkpoint, args.data, len(current), args.split, args.max_instances,
    )
    case = load_case118()
    pgm_case = build_pgm_case(case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"{args.split}_indices.npy", instance_indices)

    selected_count = len(instance_indices)
    scenarios = future.shape[1]
    horizon = future.shape[2] + 1
    free_dim = checkpoint["model_config"]["free_dim"]
    selected_u = np.full((selected_count, horizon, free_dim), np.nan, dtype="float32")
    trajectory_shape = (selected_count, scenarios, horizon)
    selected_pg = np.full(trajectory_shape + (case.n_gen,), np.nan, dtype="float32")
    selected_qg = np.full_like(selected_pg, np.nan)
    selected_vm = np.full(trajectory_shape + (case.n_bus,), np.nan, dtype="float32")
    selected_va = np.full_like(selected_vm, np.nan)
    rows = []
    verification_reports = []

    warmup_condition_np = build_condition(
        np.asarray(current[instance_indices[0]]),
        np.asarray(pool[instance_indices[0]]),
        feature_min,
        feature_max,
    )
    warmup_condition = torch.from_numpy(warmup_condition_np)[None].to(device)
    warmup_generator = torch.Generator(device=device).manual_seed(
        args.seed + int(instance_indices[0]),
    )
    warmup_latent = torch.randn(
        1, args.candidates, model.latent_dim,
        dtype=warmup_condition.dtype, device=device, generator=warmup_generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_start = perf_counter()
    with torch.inference_mode():
        model(
            warmup_condition, args.candidates, free_lower, free_upper,
            free_ramp, free_ramp, warmup_latent,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_warmup_seconds = perf_counter() - warmup_start

    print(
        f"Generator-TCN PGM/fixed-PV batch: split={args.split}, "
        f"instances={selected_count}, candidates={args.candidates}, device={device}",
        flush=True,
    )
    for local_index, instance in enumerate(instance_indices):
        condition_np = build_condition(
            np.asarray(current[instance]), np.asarray(pool[instance]),
            feature_min, feature_max,
        )
        condition = torch.from_numpy(condition_np)[None].to(device)
        latent_generator = torch.Generator(device=device).manual_seed(args.seed + int(instance))
        latent = torch.randn(
            1, args.candidates, model.latent_dim,
            dtype=condition.dtype, device=device, generator=latent_generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generation_start = perf_counter()
        with torch.inference_mode():
            schedules, _, _ = model(
                condition, args.candidates, free_lower, free_upper,
                free_ramp, free_ramp, latent,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generation_seconds = perf_counter() - generation_start
        schedules_np = schedules[0].cpu().numpy()
        loads, controls = flatten_candidate_points(
            np.asarray(current[instance]), np.asarray(future[instance]), schedules_np,
        )

        evaluation_start = perf_counter()
        states = solve_pv_batch(
            pgm_case, case, loads, controls,
            pgm_error_tolerance=args.pgm_error_tolerance,
            pgm_max_iterations=args.max_pf_iterations,
            threading=args.threading,
            continue_on_batch_error=True,
            fixed_pv_chunk_size=args.fixed_pv_chunk_size,
            fixed_pv_device=str(device),
        )
        assessed = assess_candidate_batch(
            case, loads, states, args.candidates, scenarios, horizon,
            args.ramp_fraction, args.feasibility_tolerance,
            args.balance_tolerance_mva,
        )
        evaluation_seconds = perf_counter() - evaluation_start
        constraint_seconds = evaluation_seconds - float(states["t_total"])

        feasible_indices = np.flatnonzero(assessed["feasible"])
        best_feasible_candidate = (
            int(feasible_indices[np.argmin(assessed["objective"][feasible_indices])])
            if len(feasible_indices) else -1
        )
        best_all_candidate = int(np.nanargmin(assessed["objective"]))
        feasible = best_feasible_candidate >= 0
        selected_candidate = (
            best_feasible_candidate if feasible else best_all_candidate
        )
        point_success = np.asarray(states["batch_success"]).reshape(args.candidates, -1)
        native_success = np.asarray(states["native_pgm_success"]).reshape(
            args.candidates, -1,
        )
        wrong_branch = np.asarray(states["native_pgm_wrong_branch"]).reshape(
            args.candidates, -1,
        )
        fallback_used = np.asarray(states["fixed_pv_fallback_used"]).reshape(
            args.candidates, -1,
        )
        failed_batch_points = int(np.count_nonzero(~native_success))
        failed_candidates = int(np.count_nonzero(~native_success.all(axis=1)))
        wrong_branch_points = int(np.count_nonzero(wrong_branch))
        wrong_branch_candidates = int(np.count_nonzero(wrong_branch.any(axis=1)))
        fallback_points = int(np.count_nonzero(fallback_used))
        fallback_candidates = int(np.count_nonzero(fallback_used.any(axis=1)))
        final_failed_points = int(np.count_nonzero(~point_success))
        final_failed_candidates = int(np.count_nonzero(~point_success.all(axis=1)))
        selected_cost = float(assessed["objective"][selected_candidate])
        selected_u[local_index] = schedules_np[selected_candidate]
        selected_pg[local_index] = assessed["pg_trajectory"][selected_candidate]
        selected_qg[local_index] = assessed["qg_trajectory"][selected_candidate]
        selected_vm[local_index] = assessed["vm_trajectory"][selected_candidate]
        selected_va[local_index] = assessed["va_trajectory"][selected_candidate]

        if local_index < args.verify_instances and args.verify_points:
            report = compare_with_pandapower(
                case, loads, controls, states, args.verify_points,
                args.pandapower_tolerance_mva, args.max_pf_iterations,
            )
            report["instance_index"] = int(instance)
            verification_reports.append(report)
            write_json(
                args.output_dir / "pgm_pandapower_verification.json",
                {"reports": verification_reports},
            )

        row = {
            "instance_index": int(instance),
            "feasible": int(feasible),
            "candidates": args.candidates,
            "feasible_candidates": int(len(feasible_indices)),
            "pgm_failed_batch_points": failed_batch_points,
            "pgm_failed_candidates": failed_candidates,
            "native_pgm_wrong_branch_points": wrong_branch_points,
            "native_pgm_wrong_branch_candidates": wrong_branch_candidates,
            "fixed_pv_fallback_points": fallback_points,
            "fixed_pv_fallback_candidates": fallback_candidates,
            "final_failed_batch_points": final_failed_points,
            "final_failed_candidates": final_failed_candidates,
            "best_candidate_index": selected_candidate,
            "best_feasible_candidate_index": best_feasible_candidate,
            "best_all_candidate_index": best_all_candidate,
            "cost_available": int(np.isfinite(selected_cost)),
            "feasible_cost": selected_cost if feasible else np.nan,
            "all_cost": selected_cost,
            "best_cost": selected_cost,
            "generation_seconds": generation_seconds,
            "pgm_batch_seconds": float(states["t_total"]),
            "pgm_prepare_seconds": float(states["t_prepare"]),
            "pgm_power_flow_seconds": float(states["t_power_flow"]),
            "pgm_extract_seconds": float(states["t_extract"]),
            "fixed_pv_newton_seconds": float(states["t_fixed_pv"]),
            "constraint_evaluation_seconds": constraint_seconds,
            "total_seconds": generation_seconds + evaluation_seconds,
            "batch_points": int(len(loads)),
            "pv_iterations": int(states["pv_iterations"]),
            "max_batch_pv_voltage_error_pu": float(states["max_pv_voltage_error_pu"]),
        }
        for name in (
            "maximum_violation", "max_pf_residual_mva", "violation_pg",
            "violation_qg", "violation_voltage", "violation_angle",
            "violation_thermal", "violation_ramp",
        ):
            row[name] = float(assessed[name][selected_candidate])
        rows.append(row)
        write_csv(args.output_dir / "generator_pgm_results.csv", rows)
        print(
            f"[{local_index + 1:4d}/{selected_count}] instance={instance} "
            f"feasible={feasible} feasible_candidates={len(feasible_indices)}/{args.candidates} "
            f"native_pgm_failed={failed_batch_points}/{len(loads)} "
            f"wrong_branch={wrong_branch_points} fallback={fallback_points} "
            f"final_failed={final_failed_points} "
            f"cost={format_reported_cost(row['all_cost'])} "
            f"total={row['total_seconds']:.3f}s",
            flush=True,
        )

    np.savez_compressed(
        args.output_dir / "generator_pgm_solutions.npz",
        data_split=np.asarray(args.split), instance_indices=instance_indices,
        u=selected_u, pg=selected_pg, qg=selected_qg, vm=selected_vm, va=selected_va,
    )
    feasible_rows = [row for row in rows if row["feasible"]]
    cost_summary = summarize_solution_costs(rows)
    verification_passed = bool(
        verification_reports and all(report["passed"] for report in verification_reports)
    ) if args.verify_points and args.verify_instances else None
    summary = {
        "method": f"{spec['method']} + PGM batch with fixed-PV high-voltage fallback",
        "benchmark": benchmark,
        "projection_method": spec["projection_method"],
        "diversity_loss_enabled": spec["diversity_loss_enabled"],
        "power_grid_model_version": version("power-grid-model"),
        "checkpoint": str(args.checkpoint.resolve()),
        "data_split": args.split,
        "n_instances": selected_count,
        "candidates_per_instance": args.candidates,
        "native_batch_points_per_instance": args.candidates * (1 + scenarios * (horizon - 1)),
        "fixed_pv_formulation": (
            "float64 full Newton high-voltage state + native PGM PQ batch; "
            "native failures or different solution branches use the fixed-PV state"
        ),
        "fixed_pv_device": str(device),
        "fixed_pv_chunk_size": args.fixed_pv_chunk_size,
        "feasible_instances": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / selected_count,
        "inference_warmup_seconds": inference_warmup_seconds,
        "time_seconds": finite_statistics([row["total_seconds"] for row in rows]),
        "generation_seconds": finite_statistics([row["generation_seconds"] for row in rows]),
        "pgm_batch_seconds": finite_statistics([row["pgm_batch_seconds"] for row in rows]),
        "pgm_prepare_seconds": finite_statistics(
            [row["pgm_prepare_seconds"] for row in rows]
        ),
        "pgm_power_flow_seconds": finite_statistics(
            [row["pgm_power_flow_seconds"] for row in rows]
        ),
        "pgm_extract_seconds": finite_statistics(
            [row["pgm_extract_seconds"] for row in rows]
        ),
        "pgm_failed_batch_points": finite_statistics(
            [row["pgm_failed_batch_points"] for row in rows]
        ),
        "pgm_failed_candidates": finite_statistics(
            [row["pgm_failed_candidates"] for row in rows]
        ),
        "native_pgm_wrong_branch_points": finite_statistics(
            [row["native_pgm_wrong_branch_points"] for row in rows]
        ),
        "fixed_pv_fallback_points": finite_statistics(
            [row["fixed_pv_fallback_points"] for row in rows]
        ),
        "final_failed_batch_points": finite_statistics(
            [row["final_failed_batch_points"] for row in rows]
        ),
        "fixed_pv_newton_seconds": finite_statistics(
            [row["fixed_pv_newton_seconds"] for row in rows]
        ),
        "constraint_evaluation_seconds": finite_statistics(
            [row["constraint_evaluation_seconds"] for row in rows]
        ),
        # ``best_cost`` remains as a compatibility alias for all output costs.
        "best_cost": cost_summary["all_solution_cost"],
        **cost_summary,
        "pandapower_equivalence_passed": verification_passed,
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "feasibility_tolerance": args.feasibility_tolerance,
        "balance_tolerance_mva": args.balance_tolerance_mva,
        "pgm_error_tolerance": args.pgm_error_tolerance,
        "seed": args.seed,
    }
    write_json(args.output_dir / "generator_pgm_summary.json", summary)
    print(f"saved PGM/fixed-PV benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
