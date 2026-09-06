"""Compare DECS and exact-PF inequalities on Generator trajectories."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data_generation.case118_pglib import load_case118  # noqa: E402
from Data_generation.generate_decs_dataset import build_pandapower_network  # noqa: E402
from evaluate_generator_tcn import (  # noqa: E402
    build_condition,
    checkpoint_split_indices,
    load_generator,
    solve_exact_power_flow,
)
from evaluate_generator_tcn_pgm_batch import flatten_candidate_points  # noqa: E402
from Neural_network.decs import load_decs_checkpoint  # noqa: E402
from Neural_network.validation.decs.visualize_decs_validation import (  # noqa: E402
    signed_constraint_residuals,
)
from pgm_batch_power_flow import build_pgm_case, solve_pv_batch  # noqa: E402


CATEGORIES = ("pg", "qg", "voltage", "angle", "thermal", "ramp")
UNITS = {
    "pg": "MW",
    "qg": "Mvar",
    "voltage": "p.u.",
    "angle": "degree",
    "thermal": "MVA",
    "ramp": "MW",
}


def new_accumulator() -> dict[str, dict]:
    """Create empty per-family arrays and decision counters.

    Returns
    -------
    dict[str, dict]
        Mutable accumulator keyed by the six inequality families.
    """
    return {
        name: {
            "residual_error": [], "positive_violation_error": [], "physical_error": [],
            "target_violated_magnitude_error": [],
            "target_violated_magnitude_bias": [],
            "target_violated_magnitude": [],
            "predicted_on_target_violated_magnitude": [],
            "point_max_violation_error": [],
            "target_violated_point_max_error": [],
            "target_violated_physical_magnitude_error": [],
            "target_violated_physical_magnitude_bias": [],
            "target_violated_physical_magnitude": [],
            "predicted_on_target_violated_physical_magnitude": [],
            "target_violated_point_physical_max_error": [],
            "element_count": 0, "element_agree": 0,
            "element_false_feasible": 0, "element_false_infeasible": 0,
            "element_target_violated": 0, "element_predicted_violated": 0,
            "point_count": 0, "point_agree": 0,
            "point_false_feasible": 0, "point_false_infeasible": 0,
            "point_target_violated": 0, "point_predicted_violated": 0,
        }
        for name in CATEGORIES
    }


def summarize(values: list[np.ndarray]) -> dict[str, float | int | None]:
    """Return size, mean, P95, and maximum for collected absolute errors.

    Parameters
    ----------
    values : list[np.ndarray]
        One-dimensional finite absolute-error arrays from evaluated instances.

    Returns
    -------
    dict
        Exact sample count and finite distribution statistics.
    """
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    array = np.concatenate(values).astype(float, copy=False)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def summarize_distribution(values: list[np.ndarray]) -> dict[str, float | int | None]:
    """Return basic quantiles for collected scalar values."""
    if not values:
        return {
            "count": 0, "min": None, "p05": None, "median": None,
            "mean": None, "p95": None, "max": None,
        }
    array = np.concatenate(values).astype(float, copy=False)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0, "min": None, "p05": None, "median": None,
            "mean": None, "p95": None, "max": None,
        }
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def update_comparison(
    accumulator: dict,
    category: str,
    predicted: np.ndarray,
    target: np.ndarray,
    physical_error: np.ndarray,
    tolerance: float,
    predicted_physical_residual: np.ndarray | None = None,
    target_physical_residual: np.ndarray | None = None,
) -> None:
    """Accumulate elementwise and operating-point constraint comparisons.

    Parameters
    ----------
    accumulator : dict
        Mutable category accumulator created in :func:`main`.
    category : str
        One of the six entries in :data:`CATEGORIES`.
    predicted, target : np.ndarray, shape (n_points, n_constraints)
        DECS and exact-PGM signed normalized residuals; positive is violated.
    physical_error : np.ndarray
        Absolute state-variable error in the unit declared by :data:`UNITS`.
    tolerance : float
        Normalized residual threshold used for feasible/infeasible decisions.
    predicted_physical_residual, target_physical_residual : np.ndarray or None
        Signed residuals in the family's physical unit. Positive entries are
        violations. Supplying both enables physical violation-magnitude metrics.
    """
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError(f"{category} residual arrays must share a two-dimensional shape")
    finite = np.isfinite(predicted) & np.isfinite(target)
    predicted_flat = predicted[finite]
    target_flat = target[finite]
    error = np.abs(predicted_flat - target_flat)
    positive_error = np.abs(
        np.maximum(predicted_flat, 0.0) - np.maximum(target_flat, 0.0)
    )
    predicted_violated = predicted_flat > tolerance
    target_violated = target_flat > tolerance
    predicted_positive = np.maximum(predicted_flat, 0.0)
    target_positive = np.maximum(target_flat, 0.0)

    record = accumulator[category]
    record["residual_error"].append(error.astype("float32"))
    record["positive_violation_error"].append(positive_error.astype("float32"))
    if np.any(target_violated):
        magnitude_delta = predicted_positive[target_violated] - target_positive[target_violated]
        record["target_violated_magnitude_error"].append(
            np.abs(magnitude_delta).astype("float32")
        )
        record["target_violated_magnitude_bias"].append(
            magnitude_delta.astype("float32")
        )
        record["target_violated_magnitude"].append(
            target_positive[target_violated].astype("float32")
        )
        record["predicted_on_target_violated_magnitude"].append(
            predicted_positive[target_violated].astype("float32")
        )
    if (predicted_physical_residual is None) != (target_physical_residual is None):
        raise ValueError("both physical residual arrays must be supplied together")
    if predicted_physical_residual is not None:
        predicted_physical = np.asarray(predicted_physical_residual, dtype=float)
        target_physical = np.asarray(target_physical_residual, dtype=float)
        if predicted_physical.shape != predicted.shape or target_physical.shape != target.shape:
            raise ValueError(f"{category} physical residual arrays must match normalized shapes")
        predicted_physical_positive = np.maximum(predicted_physical[finite], 0.0)
        target_physical_positive = np.maximum(target_physical[finite], 0.0)
        if np.any(target_violated):
            physical_delta = (
                predicted_physical_positive[target_violated]
                - target_physical_positive[target_violated]
            )
            record["target_violated_physical_magnitude_error"].append(
                np.abs(physical_delta).astype("float32")
            )
            record["target_violated_physical_magnitude_bias"].append(
                physical_delta.astype("float32")
            )
            record["target_violated_physical_magnitude"].append(
                target_physical_positive[target_violated].astype("float32")
            )
            record["predicted_on_target_violated_physical_magnitude"].append(
                predicted_physical_positive[target_violated].astype("float32")
            )
    record["physical_error"].append(
        np.asarray(physical_error, dtype="float32").reshape(-1)
    )
    record["element_count"] += int(len(error))
    record["element_agree"] += int(np.sum(predicted_violated == target_violated))
    record["element_false_feasible"] += int(np.sum(~predicted_violated & target_violated))
    record["element_false_infeasible"] += int(np.sum(predicted_violated & ~target_violated))
    record["element_target_violated"] += int(np.sum(target_violated))
    record["element_predicted_violated"] += int(np.sum(predicted_violated))

    row_finite = finite.all(axis=1)
    predicted_point = predicted[row_finite].max(axis=1) > tolerance
    target_point = target[row_finite].max(axis=1) > tolerance
    predicted_point_magnitude = np.maximum(predicted[row_finite], 0.0).max(axis=1)
    target_point_magnitude = np.maximum(target[row_finite], 0.0).max(axis=1)
    if predicted_physical_residual is not None:
        predicted_point_physical = np.maximum(
            predicted_physical[row_finite], 0.0,
        ).max(axis=1)
        target_point_physical = np.maximum(
            target_physical[row_finite], 0.0,
        ).max(axis=1)
        if np.any(target_point):
            record["target_violated_point_physical_max_error"].append(
                np.abs(
                    predicted_point_physical[target_point]
                    - target_point_physical[target_point]
                ).astype("float32")
            )
    point_magnitude_error = np.abs(predicted_point_magnitude - target_point_magnitude)
    record["point_max_violation_error"].append(point_magnitude_error.astype("float32"))
    if np.any(target_point):
        record["target_violated_point_max_error"].append(
            point_magnitude_error[target_point].astype("float32")
        )
    record["point_count"] += int(len(target_point))
    record["point_agree"] += int(np.sum(predicted_point == target_point))
    record["point_false_feasible"] += int(np.sum(~predicted_point & target_point))
    record["point_false_infeasible"] += int(np.sum(predicted_point & ~target_point))
    record["point_target_violated"] += int(np.sum(target_point))
    record["point_predicted_violated"] += int(np.sum(predicted_point))


def finalize_category(record: dict, unit: str) -> dict:
    """Convert one raw accumulator into JSON-ready error and decision metrics.

    Parameters
    ----------
    record : dict
        Raw arrays and integer counts accumulated across instances.
    unit : str
        Physical unit associated with the category state error.

    Returns
    -------
    dict
        Residual, physical-error, and feasibility-decision summaries.
    """
    elements = max(record["element_count"], 1)
    points = max(record["point_count"], 1)
    return {
        "normalized_signed_residual_abs_error": summarize(record["residual_error"]),
        "normalized_positive_violation_abs_error": summarize(
            record["positive_violation_error"]
        ),
        "violation_magnitude_accuracy": {
            "target_violated_element_abs_error": summarize(
                record["target_violated_magnitude_error"]
            ),
            "target_violated_element_signed_bias": summarize_distribution(
                record["target_violated_magnitude_bias"]
            ),
            "target_violation_magnitude": summarize_distribution(
                record["target_violated_magnitude"]
            ),
            "decs_magnitude_on_target_violated": summarize_distribution(
                record["predicted_on_target_violated_magnitude"]
            ),
            "operating_point_max_abs_error": summarize(
                record["point_max_violation_error"]
            ),
            "target_violated_operating_point_max_abs_error": summarize(
                record["target_violated_point_max_error"]
            ),
        },
        "physical_violation_magnitude_accuracy": {
            "unit": unit,
            "target_violated_element_abs_error": summarize(
                record["target_violated_physical_magnitude_error"]
            ),
            "target_violated_element_signed_bias": summarize_distribution(
                record["target_violated_physical_magnitude_bias"]
            ),
            "target_violation_magnitude": summarize_distribution(
                record["target_violated_physical_magnitude"]
            ),
            "decs_magnitude_on_target_violated": summarize_distribution(
                record["predicted_on_target_violated_physical_magnitude"]
            ),
            "target_violated_operating_point_max_abs_error": summarize(
                record["target_violated_point_physical_max_error"]
            ),
        },
        "physical_state_abs_error": {
            "unit": unit,
            **summarize(record["physical_error"]),
        },
        "elementwise_decision": {
            "count": record["element_count"],
            "agreement_percent": 100.0 * record["element_agree"] / elements,
            "false_feasible_percent": 100.0 * record["element_false_feasible"] / elements,
            "false_infeasible_percent": 100.0 * record["element_false_infeasible"] / elements,
            "target_violation_percent": 100.0 * record["element_target_violated"] / elements,
            "decs_violation_percent": 100.0 * record["element_predicted_violated"] / elements,
        },
        "operating_point_decision": {
            "count": record["point_count"],
            "agreement_percent": 100.0 * record["point_agree"] / points,
            "false_feasible_percent": 100.0 * record["point_false_feasible"] / points,
            "false_infeasible_percent": 100.0 * record["point_false_infeasible"] / points,
            "target_violation_percent": 100.0 * record["point_target_violated"] / points,
            "decs_violation_percent": 100.0 * record["point_predicted_violated"] / points,
        },
    }


def physical_errors(
    predicted: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    completion,
) -> dict[str, np.ndarray]:
    """Return absolute DECS-versus-PGM state errors by inequality family.

    Parameters
    ----------
    predicted, target : dict[str, torch.Tensor]
        Aligned DECS and exact-PGM states for converged operating points.
    completion : DifferentiableEqualityCompletion
        Frozen completion whose physics buffers define bus and branch indices.

    Returns
    -------
    dict[str, np.ndarray]
        Flattened absolute errors in MW, Mvar, p.u., degree, and MVA.
    """
    physics = completion.physics
    reference = physics.reference_generator
    pred_angle = predicted["va"][:, physics.fbus] - predicted["va"][:, physics.tbus]
    true_angle = target["va"][:, physics.fbus] - target["va"][:, physics.tbus]
    pred_thermal = torch.cat([
        torch.hypot(predicted["pf"], predicted["qf"]),
        torch.hypot(predicted["pt"], predicted["qt"]),
    ], dim=1)
    true_thermal = torch.cat([
        torch.hypot(target["pf"], target["qf"]),
        torch.hypot(target["pt"], target["qt"]),
    ], dim=1)
    return {
        "pg": (predicted["pg"][:, reference] - target["pg"][:, reference])
        .abs().cpu().numpy(),
        "qg": (predicted["qg"] - target["qg"]).abs().cpu().numpy(),
        "voltage": (
            predicted["vm"][:, physics.pq] - target["vm"][:, physics.pq]
        ).abs().cpu().numpy(),
        "angle": torch.rad2deg((pred_angle - true_angle).abs()).cpu().numpy(),
        "thermal": (pred_thermal - true_thermal).abs().cpu().numpy(),
    }


def physical_constraint_residuals(
    state: dict[str, torch.Tensor],
    completion,
) -> dict[str, torch.Tensor]:
    """Return signed constraint residuals in physical engineering units.

    Parameters
    ----------
    state : dict[str,torch.Tensor]
        DECS-reconstructed or exact power-flow state for aligned points.
    completion : DifferentiableEqualityCompletion
        Frozen DECS object supplying IEEE-118 limits and topology indices.

    Returns
    -------
    dict[str,torch.Tensor]
        Upper/lower residual blocks in MW, Mvar, p.u., degree, and MVA.
        Positive entries are physical constraint violations.
    """
    physics = completion.physics
    reference = physics.reference_generator
    pg = state["pg"][:, reference:reference + 1]
    qg = state["qg"]
    vm = state["vm"][:, physics.pq]
    angle = state["va"][:, physics.fbus] - state["va"][:, physics.tbus]
    apparent_from = torch.hypot(state["pf"], state["qf"])
    apparent_to = torch.hypot(state["pt"], state["qt"])
    return {
        "pg": torch.cat([
            pg - physics.pg_max[reference],
            physics.pg_min[reference] - pg,
        ], dim=1),
        "qg": torch.cat([
            qg - physics.qg_max,
            physics.qg_min - qg,
        ], dim=1),
        "voltage": torch.cat([
            vm - physics.vm_max[physics.pq],
            physics.vm_min[physics.pq] - vm,
        ], dim=1),
        "angle": torch.rad2deg(torch.cat([
            angle - physics.angle_max,
            physics.angle_min - angle,
        ], dim=1)),
        "thermal": torch.cat([
            apparent_from - physics.rate,
            apparent_to - physics.rate,
        ], dim=1),
    }


def solve_pv_in_chunks(
    pgm_case,
    case,
    loads: np.ndarray,
    controls: np.ndarray,
    *,
    chunk_size: int,
    pgm_error_tolerance: float,
    pgm_max_iterations: int,
    threading: int,
) -> dict[str, np.ndarray]:
    """Run native PGM batches, recursively isolating native failures."""
    state_names = (
        "pg", "qg", "vm", "va", "pf", "qf", "pt", "qt",
        "pv_converged", "batch_success",
    )

    def failed_point() -> dict[str, np.ndarray]:
        """Represent one native-solver failure as a non-converged NaN state."""
        return {
            "pg": np.full((1, case.n_gen), np.nan),
            "qg": np.full((1, case.n_gen), np.nan),
            "vm": np.full((1, case.n_bus), np.nan),
            "va": np.full((1, case.n_bus), np.nan),
            "pf": np.full((1, len(case.branch)), np.nan),
            "qf": np.full((1, len(case.branch)), np.nan),
            "pt": np.full((1, len(case.branch)), np.nan),
            "qt": np.full((1, len(case.branch)), np.nan),
            "pv_converged": np.zeros(1, dtype=bool),
            "batch_success": np.zeros(1, dtype=bool),
        }

    def solve_range(start: int, stop: int, local_pgm_case) -> list[dict]:
        """Solve one slice or bisect it when the native library rejects it."""
        try:
            return [solve_pv_batch(
                local_pgm_case, case, loads[start:stop], controls[start:stop],
                pgm_error_tolerance=pgm_error_tolerance,
                pgm_max_iterations=pgm_max_iterations,
                threading=threading,
                continue_on_batch_error=True,
            )]
        except RuntimeError:
            if stop - start == 1:
                return [failed_point()]
            middle = (start + stop) // 2
            return (
                solve_range(start, middle, build_pgm_case(case))
                + solve_range(middle, stop, build_pgm_case(case))
            )

    chunks = []
    for start in range(0, len(loads), chunk_size):
        stop = min(start + chunk_size, len(loads))
        chunks.extend(solve_range(start, stop, build_pgm_case(case)))
    return {
        name: np.concatenate([np.asarray(chunk[name]) for chunk in chunks], axis=0)
        for name in state_names
    }


def solve_pandapower_points(
    case,
    loads: np.ndarray,
    controls: np.ndarray,
    *,
    tolerance_mva: float,
    max_iterations: int,
    progress_every: int,
) -> dict[str, np.ndarray]:
    """Solve operating points independently with pandapower Newton-Raphson.

    Parameters
    ----------
    case : Case118
        IEEE-118 network data and fixed PV/PQ partition.
    loads : np.ndarray, shape (n_points,n_load_buses,2)
        Active/reactive net loads in MW/Mvar.
    controls : np.ndarray, shape (n_points,free_dim)
        Generator active-power and regulated-voltage controls.
    tolerance_mva : float
        Newton nodal mismatch tolerance in MVA.
    max_iterations : int
        Maximum Newton iterations for warm and flat initialization.
    progress_every : int
        Print progress after this many attempted operating points.
    """
    count = len(loads)
    branch_count = len(case.branch)
    states = {
        "pg": np.full((count, case.n_gen), np.nan),
        "qg": np.full((count, case.n_gen), np.nan),
        "vm": np.full((count, case.n_bus), np.nan),
        "va": np.full((count, case.n_bus), np.nan),
        "pf": np.full((count, branch_count), np.nan),
        "qf": np.full((count, branch_count), np.nan),
        "pt": np.full((count, branch_count), np.nan),
        "qt": np.full((count, branch_count), np.nan),
    }
    converged = np.zeros(count, dtype=bool)
    net = build_pandapower_network(case)
    for index in range(count):
        state, _ = solve_exact_power_flow(
            net, case, loads[index], controls[index],
            warm_start=index > 0,
            tolerance_mva=tolerance_mva,
            max_iterations=max_iterations,
        )
        if state is not None:
            converged[index] = True
            for name in states:
                states[name][index] = np.asarray(state[name])
        completed = index + 1
        if completed % progress_every == 0 or completed == count:
            print(
                f"  pandapower {completed}/{count}, "
                f"converged={converged[:completed].sum()}",
                flush=True,
            )
    states["pv_converged"] = converged
    states["batch_success"] = converged.copy()
    return states


def main() -> None:
    """Generate held-out candidates and save DECS-versus-exact-PF metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / "e2e118_N5000_S20_T16",
        help="Generator dataset containing the exact saved train/validation/test split",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "Neural_network" / "generator_tcn_clip_stage1.pt",
        help="Generator checkpoint whose trajectories are compared",
    )
    parser.add_argument(
        "--decs", type=Path,
        default=ROOT / "Neural_network" / "decs_pgm_fixedpv.pt",
        help="frozen DECS checkpoint used for approximate inequality evaluation",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--ramp-fraction", type=float, default=0.25)
    parser.add_argument("--decision-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--minimum-pq-voltage", type=float, default=None,
        help=(
            "optionally compare constraints only where the exact PGM solution has "
            "minimum PQ voltage at or above this value; 0.7 matches DECS data filtering"
        ),
    )
    parser.add_argument("--pgm-error-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-pf-iterations", type=int, default=30)
    parser.add_argument(
        "--exact-solver", choices=("pgm", "pandapower"), default="pandapower",
        help="independent fixed-PV exact power-flow backend",
    )
    parser.add_argument(
        "--exact-progress-every", type=int, default=1000,
        help="pandapower progress interval in attempted operating points",
    )
    parser.add_argument("--threading", type=int, default=0)
    parser.add_argument(
        "--pgm-chunk-size", type=int, default=2048,
        help="native PGM batch size; chunking avoids very-large-batch failures on Windows",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / "generator_decs_pgm_constraint_comparison.json",
        help="JSON output containing six-family residual and decision metrics",
    )
    args = parser.parse_args()
    if (
        args.candidates < 2 or args.max_pf_iterations < 1
        or args.pgm_chunk_size < 1 or args.exact_progress_every < 1
    ):
        raise ValueError("candidates must be at least two and iterations positive")
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    if args.decision_tolerance < 0 or args.pgm_error_tolerance <= 0:
        raise ValueError("decision tolerance must be nonnegative and PGM tolerance positive")
    if not (args.data.is_dir() and args.checkpoint.is_file() and args.decs.is_file()):
        raise FileNotFoundError("Generator data, Generator checkpoint, or DECS is missing")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model, checkpoint = load_generator(args.checkpoint, device)
    completion = load_decs_checkpoint(args.decs, device)
    case = load_case118()
    pgm_case = build_pgm_case(case) if args.exact_solver == "pgm" else None
    current = np.load(args.data / "current_load.npy", mmap_mode="r")
    pool = np.load(args.data / "future_pool.npy", mmap_mode="r")
    future = np.load(args.data / "future_load.npy", mmap_mode="r")
    indices = checkpoint_split_indices(
        checkpoint, args.data, len(current), args.split, args.max_instances,
    )
    normalization = checkpoint["normalization"]
    feature_min = np.asarray(normalization["feature_min"], dtype=float)
    feature_max = np.asarray(normalization["feature_max"], dtype=float)
    free_lower = torch.as_tensor(checkpoint["free_variable_lower"], device=device)
    free_upper = torch.as_tensor(checkpoint["free_variable_upper"], device=device)
    free_ramp = torch.as_tensor(checkpoint["free_ramp_mw_per_period"], device=device)
    reference = completion.physics.reference_generator
    reference_ramp = args.ramp_fraction * float(completion.physics.pg_max[reference])

    accumulator = new_accumulator()
    instance_rows = []
    exact_min_pq_values = []
    decs_min_pq_values = []
    exact_pv_target_error_values = []
    exact_low_decs_high = 0
    exact_high_decs_low = 0
    branch_threshold = 0.7
    scenarios = int(future.shape[1])
    horizon = int(future.shape[2] + 1)
    state_names = ("pg", "qg", "vm", "va", "pf", "qf", "pt", "qt")

    for position, instance in enumerate(indices, start=1):
        condition_np = build_condition(
            np.asarray(current[instance]), np.asarray(pool[instance]),
            feature_min, feature_max,
        )
        condition = torch.from_numpy(condition_np)[None].to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + int(instance))
        latent = torch.randn(
            1, args.candidates, model.latent_dim,
            dtype=condition.dtype, device=device, generator=generator,
        )
        with torch.inference_mode():
            schedules, _, _ = model(
                condition, args.candidates, free_lower, free_upper,
                free_ramp, free_ramp, latent,
            )
        loads, controls = flatten_candidate_points(
            np.asarray(current[instance]), np.asarray(future[instance]),
            schedules[0].cpu().numpy(),
        )
        if args.exact_solver == "pgm":
            exact = solve_pv_in_chunks(
                pgm_case, case, loads, controls,
                chunk_size=args.pgm_chunk_size,
                pgm_error_tolerance=args.pgm_error_tolerance,
                pgm_max_iterations=args.max_pf_iterations,
                threading=args.threading,
            )
        else:
            exact = solve_pandapower_points(
                case, loads, controls,
                tolerance_mva=args.pgm_error_tolerance,
                max_iterations=args.max_pf_iterations,
                progress_every=args.exact_progress_every,
            )
        solver_success = np.asarray(exact["batch_success"], dtype=bool)
        converged = np.asarray(exact["pv_converged"], dtype=bool)
        voltage_control_buses = np.r_[case.pv_buses, case.reference_bus]
        voltage_start = len(case.nonreference_active_generators)
        exact_pv_target_error = np.max(
            np.abs(
                np.asarray(exact["vm"])[:, voltage_control_buses]
                - controls[:, voltage_start:]
            ),
            axis=1,
        )
        exact_pv_target_error_values.append(
            exact_pv_target_error[solver_success].astype("float32")
        )
        if not np.any(converged):
            instance_rows.append({
                "instance_index": int(instance), "points": int(len(loads)),
                "solver_success_points": int(solver_success.sum()),
                "converged_points": 0, "convergence_rate": 0.0,
            })
            print(f"[{position}/{len(indices)}] instance={instance}: no converged points")
            continue

        controls_tensor = torch.from_numpy(np.asarray(controls, dtype="float32")).to(device)
        loads_tensor = torch.from_numpy(np.asarray(loads, dtype="float32")).to(device)
        with torch.inference_mode():
            predicted_all = completion(controls_tensor, loads_tensor)
        exact_min_pq = np.min(np.asarray(exact["vm"])[:, case.pq_buses], axis=1)
        decs_min_pq = (
            predicted_all["vm"][:, completion.physics.pq].amin(dim=1).cpu().numpy()
        )
        exact_min_pq_values.append(exact_min_pq[converged].astype("float32"))
        decs_min_pq_values.append(decs_min_pq[converged].astype("float32"))
        exact_low_decs_high += int(np.sum(
            converged & (exact_min_pq < branch_threshold)
            & (decs_min_pq >= branch_threshold)
        ))
        exact_high_decs_low += int(np.sum(
            converged & (exact_min_pq >= branch_threshold)
            & (decs_min_pq < branch_threshold)
        ))

        comparison = converged.copy()
        if args.minimum_pq_voltage is not None:
            comparison &= exact_min_pq >= args.minimum_pq_voltage
        valid_tensor = torch.from_numpy(comparison).to(device)
        predicted = {name: predicted_all[name][valid_tensor] for name in state_names}
        target = {
            name: torch.as_tensor(np.asarray(exact[name])[comparison], device=device)
            for name in state_names
        }
        if np.any(comparison):
            predicted_residual = signed_constraint_residuals(predicted, completion)
            target_residual = signed_constraint_residuals(target, completion)
            predicted_physical_residual = physical_constraint_residuals(
                predicted, completion,
            )
            target_physical_residual = physical_constraint_residuals(
                target, completion,
            )
            state_error = physical_errors(predicted, target, completion)
            for name in CATEGORIES[:-1]:
                update_comparison(
                    accumulator, name,
                    predicted_residual[f"constraint_{name}"].cpu().numpy(),
                    target_residual[f"constraint_{name}"].cpu().numpy(),
                    state_error[name], args.decision_tolerance,
                    predicted_physical_residual[name].cpu().numpy(),
                    target_physical_residual[name].cpu().numpy(),
                )

        points = 1 + scenarios * (horizon - 1)
        predicted_reference = predicted_all["pg"][:, reference].reshape(
            args.candidates, points,
        ).cpu().numpy()
        target_reference = np.asarray(exact["pg"])[:, reference].reshape(
            args.candidates, points,
        )
        point_valid = comparison.reshape(args.candidates, points)
        predicted_trajectory = np.concatenate([
            np.broadcast_to(
                predicted_reference[:, 0, None, None],
                (args.candidates, scenarios, 1),
            ),
            predicted_reference[:, 1:].reshape(args.candidates, scenarios, horizon - 1),
        ], axis=2)
        target_trajectory = np.concatenate([
            np.broadcast_to(
                target_reference[:, 0, None, None],
                (args.candidates, scenarios, 1),
            ),
            target_reference[:, 1:].reshape(args.candidates, scenarios, horizon - 1),
        ], axis=2)
        valid_trajectory = np.concatenate([
            np.broadcast_to(
                point_valid[:, 0, None, None],
                (args.candidates, scenarios, 1),
            ),
            point_valid[:, 1:].reshape(args.candidates, scenarios, horizon - 1),
        ], axis=2)
        valid_delta = valid_trajectory[:, :, 1:] & valid_trajectory[:, :, :-1]
        predicted_delta = np.diff(predicted_trajectory, axis=2)[valid_delta]
        target_delta = np.diff(target_trajectory, axis=2)[valid_delta]
        predicted_ramp = np.column_stack([
            (predicted_delta - reference_ramp) / reference_ramp,
            (-predicted_delta - reference_ramp) / reference_ramp,
        ])
        target_ramp = np.column_stack([
            (target_delta - reference_ramp) / reference_ramp,
            (-target_delta - reference_ramp) / reference_ramp,
        ])
        predicted_ramp_physical = np.column_stack([
            predicted_delta - reference_ramp,
            -predicted_delta - reference_ramp,
        ])
        target_ramp_physical = np.column_stack([
            target_delta - reference_ramp,
            -target_delta - reference_ramp,
        ])
        if len(predicted_delta):
            update_comparison(
                accumulator, "ramp", predicted_ramp, target_ramp,
                np.abs(predicted_delta - target_delta), args.decision_tolerance,
                predicted_ramp_physical, target_ramp_physical,
            )

        raw_point_valid = converged.reshape(args.candidates, points)
        complete_candidates = int(np.sum(raw_point_valid.all(axis=1)))
        high_voltage_points = converged & (exact_min_pq >= branch_threshold)
        instance_rows.append({
            "instance_index": int(instance),
            "points": int(len(loads)),
            "solver_success_points": int(solver_success.sum()),
            "solver_success_rate": float(solver_success.mean()),
            "converged_points": int(converged.sum()),
            "convergence_rate": float(converged.mean()),
            "complete_candidates": complete_candidates,
            "high_voltage_converged_points": int(high_voltage_points.sum()),
            "low_voltage_converged_points": int((converged & ~high_voltage_points).sum()),
            "comparison_points": int(comparison.sum()),
        })
        print(
            f"[{position}/{len(indices)}] instance={instance}: "
            f"converged={converged.sum()}/{len(converged)}, "
            f"complete_candidates={complete_candidates}/{args.candidates}",
            flush=True,
        )

    total_points = sum(row["points"] for row in instance_rows)
    solver_success_points = sum(
        row.get("solver_success_points", 0) for row in instance_rows
    )
    converged_points = sum(row["converged_points"] for row in instance_rows)
    comparison_points = sum(row.get("comparison_points", 0) for row in instance_rows)
    exact_min_pq_array = (
        np.concatenate(exact_min_pq_values) if exact_min_pq_values else np.empty(0)
    )
    decs_min_pq_array = (
        np.concatenate(decs_min_pq_values) if decs_min_pq_values else np.empty(0)
    )
    report = {
        "protocol": {
            "data": str(args.data.resolve()),
            "generator_checkpoint": str(args.checkpoint.resolve()),
            "decs_checkpoint": str(args.decs.resolve()),
            "split": args.split,
            "instances": int(len(indices)),
            "candidates": args.candidates,
            "scenarios": scenarios,
            "horizon": horizon,
            "decision_tolerance": args.decision_tolerance,
            "pgm_error_tolerance": args.pgm_error_tolerance,
            "pgm_chunk_size": args.pgm_chunk_size,
            "exact_solver": args.exact_solver,
            "seed": args.seed,
            "minimum_pq_voltage_for_comparison": args.minimum_pq_voltage,
            "comparison_scope": (
                f"static inequalities use selected converged {args.exact_solver} "
                "operating points; reference-Pg ramp uses adjacent selected pairs"
            ),
        },
        "pgm_convergence": {
            "total_points": total_points,
            "solver_success_points": solver_success_points,
            "solver_success_rate": solver_success_points / max(total_points, 1),
            "converged_points": converged_points,
            "rate": converged_points / max(total_points, 1),
            "maximum_controlled_voltage_error_pu_on_solver_success": (
                summarize_distribution(exact_pv_target_error_values)
            ),
            "instances": instance_rows,
        },
        "solution_branch_diagnostics": {
            "training_branch_threshold_pu": branch_threshold,
            "comparison_points": comparison_points,
            "exact_pgm_minimum_pq_voltage_pu": summarize_distribution(
                exact_min_pq_values
            ),
            "decs_minimum_pq_voltage_pu": summarize_distribution(
                decs_min_pq_values
            ),
            "exact_pgm_low_voltage_percent": (
                100.0 * np.sum(exact_min_pq_array < branch_threshold)
                / max(len(exact_min_pq_array), 1)
            ),
            "decs_low_voltage_percent": (
                100.0 * np.sum(decs_min_pq_array < branch_threshold)
                / max(len(decs_min_pq_array), 1)
            ),
            "exact_low_but_decs_high_percent": (
                100.0 * exact_low_decs_high / max(converged_points, 1)
            ),
            "exact_high_but_decs_low_percent": (
                100.0 * exact_high_decs_low / max(converged_points, 1)
            ),
        },
        "constraint_metrics": {
            name: finalize_category(accumulator[name], UNITS[name])
            for name in CATEGORIES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved comparison to {args.output}")


if __name__ == "__main__":
    main()
