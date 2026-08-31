"""Evaluate Deterministic-TCN with IPOPT feasibility projection.

The original deterministic schedule is screened once by native batched Power
Grid Model. A schedule rejected by that screen is projected onto the complete
shared-trajectory SMPC feasible set. IPOPT optimal termination under the given
solver tolerances is the final feasibility criterion for a projected result;
no second PGM validation is performed.
"""

import argparse
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from Data_generation.case118_pglib import load_case118
from evaluate_deterministic_tcn import load_deterministic_tcn
from evaluate_generator_tcn import (
    build_condition,
    checkpoint_split_indices,
    finite_statistics,
    format_reported_cost,
    summarize_solution_costs,
    write_csv,
    write_json,
)
from evaluate_generator_tcn_pgm_batch import (
    assess_candidate_batch,
    compare_with_pandapower,
    flatten_candidate_points,
)
from pgm_batch_power_flow import build_pgm_case, solve_pv_batch
from solve_projection_smpc_ipopt import (
    IPOPT_PATH,
    SharedTrajectorySMPCProjection,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e118_N10000_S20_T16"
DEFAULT_CHECKPOINT = ROOT / "Neural_network" / "deterministic_tcn.pt"
DEFAULT_OUTPUT = ROOT / "output" / "deterministic_tcn_projection_test"


def build_parser() -> argparse.ArgumentParser:
    """Declare deterministic inference, PGM screen, and IPOPT arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--ramp-fraction", type=float, default=0.25)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-5)
    parser.add_argument("--balance-tolerance-mva", type=float, default=1e-5)
    parser.add_argument("--pgm-error-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-pf-iterations", type=int, default=30)
    parser.add_argument("--threading", type=int, default=0)
    parser.add_argument("--verify-instances", type=int, default=1)
    parser.add_argument("--verify-points", type=int, default=16)
    parser.add_argument("--pandapower-tolerance-mva", type=float, default=1e-8)
    parser.add_argument("--ipopt-path", type=Path, default=IPOPT_PATH)
    parser.add_argument("--ipopt-tolerance", type=float, default=1e-8)
    parser.add_argument("--ipopt-constraint-tolerance", type=float, default=1e-8)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    """Reject inconsistent benchmark settings before loading large arrays."""
    positive = (
        args.ramp_fraction,
        args.pgm_error_tolerance,
        args.pandapower_tolerance_mva,
        args.ipopt_tolerance,
        args.ipopt_constraint_tolerance,
    )
    if any(value <= 0 for value in positive) or args.max_pf_iterations < 1:
        raise ValueError("ramp, solver tolerances, and iteration count must be positive")
    if (
        args.feasibility_tolerance < 0
        or args.balance_tolerance_mva < 0
        or args.verify_instances < 0
        or args.verify_points < 0
    ):
        raise ValueError("PGM feasibility and verification settings are invalid")
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    if not args.data.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("base dataset or deterministic TCN checkpoint is missing")
    if not args.ipopt_path.is_file():
        raise FileNotFoundError(f"IPOPT executable not found: {args.ipopt_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def main() -> None:
    """Screen deterministic schedules and project only the infeasible ones."""
    args = build_parser().parse_args()
    _validate_arguments(args)
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OUTPUT if args.split == "test"
            else DEFAULT_OUTPUT.parent / "deterministic_tcn_projection_validation"
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    model_load_start = perf_counter()
    model, checkpoint = load_deterministic_tcn(args.checkpoint, device)
    model_load_seconds = perf_counter() - model_load_start
    normalization = checkpoint["normalization"]
    feature_min = np.asarray(normalization["feature_min"], dtype=float)
    feature_max = np.asarray(normalization["feature_max"], dtype=float)
    free_lower = torch.as_tensor(
        checkpoint["free_variable_lower"], dtype=torch.float32, device=device,
    )
    free_upper = torch.as_tensor(
        checkpoint["free_variable_upper"], dtype=torch.float32, device=device,
    )
    free_ramp = torch.as_tensor(
        checkpoint["free_ramp_mw_per_period"], dtype=torch.float32, device=device,
    )

    current = np.load(args.data / "current_load.npy", mmap_mode="r")
    pool = np.load(args.data / "future_pool.npy", mmap_mode="r")
    future = np.load(args.data / "future_load.npy", mmap_mode="r")
    instance_indices = checkpoint_split_indices(
        checkpoint, args.data, len(current), args.split, args.max_instances,
    )
    scenarios = future.shape[1]
    horizon = 1 + future.shape[2]
    free_dim = int(checkpoint["model_config"]["free_dim"])

    case = load_case118()
    setup_start = perf_counter()
    pgm_case = build_pgm_case(case)
    pgm_setup_seconds = perf_counter() - setup_start
    projection_build_start = perf_counter()
    projector = SharedTrajectorySMPCProjection(
        horizon=horizon,
        n_scenarios=scenarios,
        ramp_fraction=args.ramp_fraction,
        ipopt_path=args.ipopt_path,
        ipopt_tolerance=args.ipopt_tolerance,
        ipopt_constraint_tolerance=args.ipopt_constraint_tolerance,
    )
    projection_model_build_seconds = perf_counter() - projection_build_start
    if projector.free_dim != free_dim:
        raise ValueError("checkpoint and projection model free dimensions differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / f"{args.split}_indices.npy"
    if index_path.is_file():
        previous = np.load(index_path)
        if not np.array_equal(previous[:len(instance_indices)], instance_indices):
            raise ValueError(f"existing and current {args.split} indices differ")
    np.save(index_path, instance_indices)

    n_selected = len(instance_indices)
    raw_u = np.full((n_selected, horizon, free_dim), np.nan, dtype="float32")
    final_u = np.full_like(raw_u, np.nan)
    trajectory_shape = (n_selected, scenarios, horizon)
    raw_pg = np.full(trajectory_shape + (case.n_gen,), np.nan, dtype="float32")
    raw_qg = np.full_like(raw_pg, np.nan)
    raw_vm = np.full(trajectory_shape + (case.n_bus,), np.nan, dtype="float32")
    raw_va = np.full_like(raw_vm, np.nan)
    final_pg = np.full_like(raw_pg, np.nan)
    final_qg = np.full_like(raw_qg, np.nan)
    final_vm = np.full_like(raw_vm, np.nan)
    final_va = np.full_like(raw_va, np.nan)
    projection_applied = np.zeros(n_selected, dtype=bool)
    rows: list[dict] = []
    verification_reports: list[dict] = []

    warmup_np = build_condition(
        np.asarray(current[instance_indices[0]]),
        np.asarray(pool[instance_indices[0]]),
        feature_min,
        feature_max,
    )
    warmup = torch.from_numpy(warmup_np)[None].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_start = perf_counter()
    with torch.inference_mode():
        model(warmup, free_lower, free_upper, free_ramp, free_ramp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_warmup_seconds = perf_counter() - warmup_start

    print(
        f"Deterministic-TCN + IPOPT projection: split={args.split}, "
        f"instances={n_selected}, device={device}",
        flush=True,
    )
    for local_index, instance in enumerate(instance_indices):
        condition_np = build_condition(
            np.asarray(current[instance]), np.asarray(pool[instance]),
            feature_min, feature_max,
        )
        condition = torch.from_numpy(condition_np)[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generation_start = perf_counter()
        with torch.inference_mode():
            schedule, _, _ = model(
                condition, free_lower, free_upper, free_ramp, free_ramp,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generation_seconds = perf_counter() - generation_start
        schedule_np = schedule[0].cpu().numpy()
        raw_u[local_index] = schedule_np

        loads, controls = flatten_candidate_points(
            np.asarray(current[instance]), np.asarray(future[instance]),
            schedule_np[None],
        )
        validation_start = perf_counter()
        raw_states = solve_pv_batch(
            pgm_case, case, loads, controls,
            pgm_error_tolerance=args.pgm_error_tolerance,
            pgm_max_iterations=args.max_pf_iterations,
            threading=args.threading,
        )
        assessed = assess_candidate_batch(
            case, loads, raw_states, 1, scenarios, horizon,
            args.ramp_fraction, args.feasibility_tolerance,
            args.balance_tolerance_mva,
        )
        raw_validation_seconds = perf_counter() - validation_start
        raw_constraint_seconds = raw_validation_seconds - float(raw_states["t_total"])
        raw_feasible = bool(assessed["feasible"][0])
        raw_cost = float(assessed["objective"][0])
        for name, target in (
            ("pg_trajectory", raw_pg), ("qg_trajectory", raw_qg),
            ("vm_trajectory", raw_vm), ("va_trajectory", raw_va),
        ):
            target[local_index] = assessed[name][0]

        if local_index < args.verify_instances and args.verify_points:
            report = compare_with_pandapower(
                case, loads, controls, raw_states, args.verify_points,
                args.pandapower_tolerance_mva, args.max_pf_iterations,
            )
            report["instance_index"] = int(instance)
            verification_reports.append(report)
            write_json(
                args.output_dir / "pgm_pandapower_verification.json",
                {"reports": verification_reports},
            )

        projection_required = not raw_feasible
        projection_applied[local_index] = projection_required
        projected = None
        if projection_required:
            projected = projector.solve(
                np.asarray(current[instance]), np.asarray(future[instance]),
                schedule_np, tee=args.tee,
            )
        projection_solved = bool(projected and projected["solved"])
        final_feasible = raw_feasible or projection_solved

        if raw_feasible:
            final_u[local_index] = schedule_np
            final_cost = raw_cost
            for source, target in (
                (raw_pg, final_pg), (raw_qg, final_qg),
                (raw_vm, final_vm), (raw_va, final_va),
            ):
                target[local_index] = source[local_index]
        elif projection_solved:
            final_u[local_index] = projected["schedule"]
            final_cost = float(projected["economic_objective"])
            for name, target in (
                ("pg", final_pg), ("qg", final_qg),
                ("vm", final_vm), ("va", final_va),
            ):
                target[local_index] = projected["states"][name]
        else:
            final_cost = np.nan

        projection_seconds = (
            float(projected["solve_seconds"]) if projected is not None else 0.0
        )
        row = {
            "instance_index": int(instance),
            "raw_feasible": int(raw_feasible),
            "projection_required": int(projection_required),
            "projection_solved": int(projection_solved),
            "feasible": int(final_feasible),
            "cost_available": int(np.isfinite(final_cost)),
            "raw_cost": raw_cost,
            "feasible_cost": final_cost if final_feasible else np.nan,
            "all_cost": final_cost,
            "best_cost": final_cost,
            "generation_seconds": generation_seconds,
            "raw_pgm_validation_seconds": raw_validation_seconds,
            "raw_pgm_batch_seconds": float(raw_states["t_total"]),
            "raw_pgm_power_flow_seconds": float(raw_states["t_power_flow"]),
            "raw_constraint_evaluation_seconds": raw_constraint_seconds,
            "projection_seconds": projection_seconds,
            "total_seconds": generation_seconds + projection_seconds,
            "projection_termination": (
                projected["termination"] if projected is not None else "not_required"
            ),
            "projection_retries": (
                int(projected["retries"]) if projected is not None else 0
            ),
            "projection_objective_squared": (
                projected.get("projection_objective_squared", np.nan)
                if projected is not None else 0.0
            ),
            "projection_distance_l2_pu": (
                projected.get("projection_distance_l2_pu", np.nan)
                if projected is not None else 0.0
            ),
            "projection_pg_l2_mw": (
                projected.get("projection_pg_l2_mw", np.nan)
                if projected is not None else 0.0
            ),
            "projection_voltage_l2_pu": (
                projected.get("projection_voltage_l2_pu", np.nan)
                if projected is not None else 0.0
            ),
            "ipopt_max_constraint_violation": (
                projected.get("ipopt_max_constraint_violation", np.nan)
                if projected is not None else 0.0
            ),
            "raw_maximum_violation": float(assessed["maximum_violation"][0]),
            "raw_max_pf_residual_mva": float(assessed["max_pf_residual_mva"][0]),
        }
        for name in (
            "violation_pg", "violation_qg", "violation_voltage",
            "violation_angle", "violation_thermal", "violation_ramp",
        ):
            row[f"raw_{name}"] = float(assessed[name][0])
        rows.append(row)
        write_csv(args.output_dir / "deterministic_projection_results.csv", rows)
        print(
            f"[{local_index + 1:4d}/{n_selected}] instance={instance} "
            f"raw_feasible={raw_feasible} projected={projection_solved} "
            f"final_cost={format_reported_cost(final_cost)} "
            f"total={row['total_seconds']:.3f}s",
            flush=True,
        )

    np.savez_compressed(
        args.output_dir / "deterministic_projection_solutions.npz",
        data_split=np.asarray(args.split),
        instance_indices=instance_indices,
        **{f"{args.split}_indices": instance_indices},
        projection_applied=projection_applied,
        raw_feasible=np.asarray([row["raw_feasible"] for row in rows], dtype=bool),
        projection_solved=np.asarray([
            row["projection_solved"] for row in rows
        ], dtype=bool),
        final_feasible=np.asarray([row["feasible"] for row in rows], dtype=bool),
        raw_cost=np.asarray([row["raw_cost"] for row in rows], dtype=float),
        final_cost=np.asarray([row["all_cost"] for row in rows], dtype=float),
        raw_u=raw_u, raw_pg=raw_pg, raw_qg=raw_qg, raw_vm=raw_vm, raw_va=raw_va,
        u=final_u, pg=final_pg, qg=final_qg, vm=final_vm, va=final_va,
    )

    raw_feasible_rows = [row for row in rows if row["raw_feasible"]]
    projection_rows = [row for row in rows if row["projection_required"]]
    recovered_rows = [row for row in projection_rows if row["projection_solved"]]
    feasible_rows = [row for row in rows if row["feasible"]]
    cost_summary = summarize_solution_costs(rows)
    verification_passed = (
        bool(verification_reports)
        and all(report["passed"] for report in verification_reports)
    ) if args.verify_points and args.verify_instances else None
    summary = {
        "method": "Deterministic-TCN + shared-SMPC IPOPT feasibility projection",
        "power_grid_model_version": version("power-grid-model"),
        "checkpoint": str(args.checkpoint.resolve()),
        "data_split": args.split,
        "n_instances": n_selected,
        "raw_feasible_instances": len(raw_feasible_rows),
        "raw_feasibility_rate": len(raw_feasible_rows) / n_selected,
        "projection_required_instances": len(projection_rows),
        "projection_solved_instances": len(recovered_rows),
        "projection_recovery_rate": (
            len(recovered_rows) / len(projection_rows) if projection_rows else None
        ),
        "feasible_instances": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / n_selected,
        "deterministic_model_load_seconds": model_load_seconds,
        "pgm_setup_seconds": pgm_setup_seconds,
        "projection_model_build_seconds": projection_model_build_seconds,
        "inference_warmup_seconds": inference_warmup_seconds,
        "time_excludes_raw_pgm_screen": True,
        "time_seconds": finite_statistics([row["total_seconds"] for row in rows]),
        "generation_seconds": finite_statistics([
            row["generation_seconds"] for row in rows
        ]),
        "raw_pgm_validation_seconds": finite_statistics([
            row["raw_pgm_validation_seconds"] for row in rows
        ]),
        "projection_seconds_attempted": finite_statistics([
            row["projection_seconds"] for row in projection_rows
        ]),
        "projection_distance_l2_pu": finite_statistics([
            row["projection_distance_l2_pu"] for row in recovered_rows
        ]),
        "projection_pg_l2_mw": finite_statistics([
            row["projection_pg_l2_mw"] for row in recovered_rows
        ]),
        "projection_voltage_l2_pu": finite_statistics([
            row["projection_voltage_l2_pu"] for row in recovered_rows
        ]),
        "raw_feasible_solution_cost": finite_statistics([
            row["all_cost"] for row in raw_feasible_rows
        ]),
        "recovered_solution_cost": finite_statistics([
            row["all_cost"] for row in recovered_rows
        ]),
        "best_cost": cost_summary["all_solution_cost"],
        **cost_summary,
        "projected_solution_feasibility_rule": (
            "IPOPT optimal termination under configured solver tolerances; "
            "no post-projection PGM validation"
        ),
        "pandapower_equivalence_passed_for_raw_pgm": verification_passed,
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "raw_pgm_feasibility_tolerance": args.feasibility_tolerance,
        "raw_pgm_balance_tolerance_mva": args.balance_tolerance_mva,
        "ipopt_tolerance": args.ipopt_tolerance,
        "ipopt_constraint_tolerance": args.ipopt_constraint_tolerance,
        "ipopt_path": str(args.ipopt_path.resolve()),
    }
    write_json(args.output_dir / "deterministic_projection_summary.json", summary)
    print(f"saved projected Deterministic-TCN benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
