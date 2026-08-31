"""Evaluate Deterministic-TCN with native batched PGM AC power flow.

The saved validation or test split is reconstructed from the checkpoint. One
shared deterministic schedule is generated per instance, and all scenario/time
operating points are solved together by Power Grid Model. Pandapower is used
only for an optional numerical-equivalence sample and never decides feasibility.
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data_generation.case118_pglib import load_case118  # noqa: E402
from evaluate_generator_tcn import (  # noqa: E402
    build_condition,
    checkpoint_split_indices,
    finite_statistics,
    format_reported_cost,
    summarize_solution_costs,
    write_csv,
    write_json,
)
from evaluate_generator_tcn_pgm_batch import (  # noqa: E402
    assess_candidate_batch,
    compare_with_pandapower,
    flatten_candidate_points,
)
from Neural_network.deterministic_tcn import DeterministicTCN  # noqa: E402
from Neural_network.noncausal_tcn import PROJECTION_METHOD  # noqa: E402
from pgm_batch_power_flow import build_pgm_case, solve_pv_batch  # noqa: E402


DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e118_N10000_S20_T16"
DEFAULT_CHECKPOINT = ROOT / "Neural_network" / "deterministic_tcn.pt"
DEFAULT_OUTPUT = ROOT / "output" / "deterministic_tcn_test"


def load_deterministic_tcn(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DeterministicTCN, dict]:
    """Load one deterministic checkpoint on ``device`` for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_class") != "DeterministicTCN":
        raise ValueError("checkpoint is not a DeterministicTCN")
    benchmark = checkpoint.get("benchmark_definition", {})
    if benchmark.get("latent_variable") is not False:
        raise ValueError("checkpoint does not declare a latent-free benchmark")
    if checkpoint.get("model_config", {}).get("projection_method") != PROJECTION_METHOD:
        raise ValueError(
            "checkpoint uses the obsolete sigmoid projection; retrain "
            f"DeterministicTCN with projection_method={PROJECTION_METHOD!r}"
        )
    model = DeterministicTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def build_parser() -> argparse.ArgumentParser:
    """Declare reproducible native-PGM evaluation arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="SMPC dataset directory containing current, pooled, and future loads",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="trained Deterministic-TCN checkpoint",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="saved held-out split; use test only for the frozen final benchmark",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="result directory; selected automatically from the split when omitted",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="optional positive prefix length for a smoke evaluation",
    )
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="one-period generator ramp limit as a positive fraction of Pmax",
    )
    parser.add_argument(
        "--feasibility-tolerance", type=float, default=1.0e-5,
        help="maximum normalized Pg/Qg/V/angle/thermal/ramp excess",
    )
    parser.add_argument(
        "--balance-tolerance-mva", type=float, default=1.0e-5,
        help="diagnostic maximum nodal AC-balance residual in MVA",
    )
    parser.add_argument(
        "--pgm-error-tolerance", type=float, default=1.0e-10,
        help="positive native PGM Newton-Raphson error tolerance",
    )
    parser.add_argument(
        "--max-pf-iterations", type=int, default=30,
        help="positive maximum native PGM Newton iterations",
    )
    parser.add_argument(
        "--threading", type=int, default=0,
        help="PGM workers: negative is sequential, zero uses all hardware threads",
    )
    parser.add_argument(
        "--verify-instances", type=int, default=1,
        help="number of initial instances sampled for pandapower equivalence",
    )
    parser.add_argument(
        "--verify-points", type=int, default=16,
        help="spread of batch points compared per instance; zero disables",
    )
    parser.add_argument(
        "--pandapower-tolerance-mva", type=float, default=1.0e-8,
        help="pandapower Newton tolerance used only by equivalence checks",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="TCN inference device; auto selects CUDA when available",
    )
    return parser


def main() -> None:
    """Generate, validate with PGM, and save one schedule per held-out instance."""
    args = build_parser().parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    if args.max_pf_iterations < 1 or args.ramp_fraction <= 0:
        raise ValueError("iteration count and ramp_fraction must be positive")
    if (
        args.feasibility_tolerance < 0
        or args.balance_tolerance_mva < 0
        or args.pgm_error_tolerance <= 0
        or args.pandapower_tolerance_mva <= 0
        or args.verify_instances < 0
        or args.verify_points < 0
    ):
        raise ValueError("PGM, feasibility, and verification settings are invalid")
    if not args.data.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("base dataset or deterministic TCN checkpoint is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OUTPUT if args.split == "test"
            else DEFAULT_OUTPUT.parent / "deterministic_tcn_validation"
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
    case = load_case118()
    setup_start = perf_counter()
    pgm_case = build_pgm_case(case)
    setup_seconds = perf_counter() - setup_start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / f"{args.split}_indices.npy"
    if index_path.is_file():
        previous = np.load(index_path)
        if not np.array_equal(previous[:len(instance_indices)], instance_indices):
            raise ValueError(f"existing and current {args.split} indices differ")
    np.save(index_path, instance_indices)

    n_selected = len(instance_indices)
    scenarios = future.shape[1]
    horizon = 1 + future.shape[2]
    free_dim = int(checkpoint["model_config"]["free_dim"])
    selected_u = np.full((n_selected, horizon, free_dim), np.nan, dtype="float32")
    trajectory_shape = (n_selected, scenarios, horizon)
    selected_pg = np.full(trajectory_shape + (case.n_gen,), np.nan, dtype="float32")
    selected_qg = np.full_like(selected_pg, np.nan)
    selected_vm = np.full(trajectory_shape + (case.n_bus,), np.nan, dtype="float32")
    selected_va = np.full_like(selected_vm, np.nan)
    rows: list[dict] = []
    verification_reports: list[dict] = []

    warmup_condition_np = build_condition(
        np.asarray(current[instance_indices[0]]),
        np.asarray(pool[instance_indices[0]]),
        feature_min,
        feature_max,
    )
    warmup_condition = torch.from_numpy(warmup_condition_np)[None].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_start = perf_counter()
    with torch.inference_mode():
        model(warmup_condition, free_lower, free_upper, free_ramp, free_ramp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_warmup_seconds = perf_counter() - warmup_start

    print(
        f"Deterministic-TCN native PGM evaluation: split={args.split}, "
        f"instances={n_selected}, candidates=1, device={device}",
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

        loads, controls = flatten_candidate_points(
            np.asarray(current[instance]),
            np.asarray(future[instance]),
            schedule_np[None],
        )
        validation_start = perf_counter()
        states = solve_pv_batch(
            pgm_case, case, loads, controls,
            pgm_error_tolerance=args.pgm_error_tolerance,
            pgm_max_iterations=args.max_pf_iterations,
            threading=args.threading,
        )
        assessed = assess_candidate_batch(
            case, loads, states, 1, scenarios, horizon,
            args.ramp_fraction, args.feasibility_tolerance,
            args.balance_tolerance_mva,
        )
        validation_seconds = perf_counter() - validation_start
        constraint_seconds = validation_seconds - float(states["t_total"])

        feasible = bool(assessed["feasible"][0])
        objective = float(assessed["objective"][0])
        selected_u[local_index] = schedule_np
        selected_pg[local_index] = assessed["pg_trajectory"][0]
        selected_qg[local_index] = assessed["qg_trajectory"][0]
        selected_vm[local_index] = assessed["vm_trajectory"][0]
        selected_va[local_index] = assessed["va_trajectory"][0]

        # Pandapower comparison is outside the reported neural time.
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
            "candidates": 1,
            "pgm_candidates_tested": 1,
            "pgm_rejected_candidates": int(not feasible),
            "best_candidate_index": 0,
            "cost_available": int(np.isfinite(objective)),
            "feasible_cost": objective if feasible else np.nan,
            "all_cost": objective,
            "best_cost": objective,
            "generation_seconds": generation_seconds,
            "pgm_validation_seconds": validation_seconds,
            "pgm_batch_seconds": float(states["t_total"]),
            "pgm_power_flow_seconds": float(states["t_power_flow"]),
            "constraint_evaluation_seconds": constraint_seconds,
            # Deterministic timing measures the neural solver only.
            "total_seconds": generation_seconds,
            "batch_points": int(len(loads)),
            "maximum_violation": float(assessed["maximum_violation"][0]),
            "max_pf_residual_mva": float(assessed["max_pf_residual_mva"][0]),
        }
        for name in (
            "violation_pg", "violation_qg", "violation_voltage",
            "violation_angle", "violation_thermal", "violation_ramp",
        ):
            row[name] = float(assessed[name][0])
        rows.append(row)
        write_csv(args.output_dir / "deterministic_results.csv", rows)
        print(
            f"[{local_index + 1:4d}/{n_selected}] instance={instance} "
            f"feasible={feasible} cost={format_reported_cost(objective)} "
            f"total={row['total_seconds']:.3f}s",
            flush=True,
        )

    np.savez_compressed(
        args.output_dir / "deterministic_solutions.npz",
        data_split=np.asarray(args.split), instance_indices=instance_indices,
        **{f"{args.split}_indices": instance_indices},
        u=selected_u, pg=selected_pg, qg=selected_qg,
        vm=selected_vm, va=selected_va,
    )
    feasible_rows = [row for row in rows if row["feasible"]]
    cost_summary = summarize_solution_costs(rows)
    verification_passed = (
        bool(verification_reports)
        and all(report["passed"] for report in verification_reports)
    ) if args.verify_points and args.verify_instances else None
    summary = {
        "method": "Deterministic-TCN + native Power Grid Model validation",
        "power_grid_model_version": version("power-grid-model"),
        "checkpoint": str(args.checkpoint.resolve()),
        "data_split": args.split,
        "n_instances": n_selected,
        "candidates_per_instance": 1,
        "feasible_instances": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / n_selected,
        "deterministic_model_load_seconds": model_load_seconds,
        "pgm_setup_seconds": setup_seconds,
        "inference_warmup_seconds": inference_warmup_seconds,
        "neural_time_excludes_pgm_validation": True,
        "time_seconds": finite_statistics([row["total_seconds"] for row in rows]),
        "generation_seconds": finite_statistics([
            row["generation_seconds"] for row in rows
        ]),
        "pgm_validation_seconds": finite_statistics([
            row["pgm_validation_seconds"] for row in rows
        ]),
        "pgm_power_flow_seconds": finite_statistics([
            row["pgm_power_flow_seconds"] for row in rows
        ]),
        "best_cost": cost_summary["all_solution_cost"],
        **cost_summary,
        "pandapower_equivalence_passed": verification_passed,
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "feasibility_tolerance": args.feasibility_tolerance,
        "balance_tolerance_mva": args.balance_tolerance_mva,
        "pgm_error_tolerance": args.pgm_error_tolerance,
        "pgm_feasibility_rule": (
            "converged native AC power flow and Pg/Qg/V/angle/thermal/ramp bounds"
        ),
    }
    write_json(args.output_dir / "deterministic_summary.json", summary)
    print(f"saved Deterministic-TCN benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
