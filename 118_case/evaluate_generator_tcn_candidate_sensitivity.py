"""Evaluate Generator-TCN sensitivity to the number of sampled candidates.

The script runs :mod:`evaluate_generator_tcn_pgm_batch` for each requested
candidate count, aggregates economic quality, wall-clock solution time, and
feasibility, then exports a compact two-panel IEEE TSG figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_EVALUATOR = ROOT / "evaluate_generator_tcn_pgm_batch.py"
DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e118_N5000_S20_T16"
DEFAULT_CHECKPOINT = ROOT / "Neural_network" / "generator_tcn_clip.pt"
DEFAULT_OUTPUT = ROOT / "output" / "generator_tcn_candidate_sensitivity"
DEFAULT_CANDIDATES = (1, 5, 10, 20, 50, 100, 200)


def configure_ieee_tsg_style() -> None:
    """Configure the required Times New Roman IEEE TSG plotting style."""
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(
            "Times New Roman is required for the IEEE TSG figure but is not installed."
        ) from error
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.fallback": None,
        "font.size": 8,
        "font.weight": "normal",
        "axes.labelsize": 8.5,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def finite_mean_ci(values: list[float]) -> tuple[float, float]:
    """Return the finite mean and normal 95% confidence half-width.

    Parameters
    ----------
    values : list[float]
        Repeated scalar observations for one candidate count.

    Returns
    -------
    mean, ci95 : tuple[float, float]
        Arithmetic mean and ``1.96 * standard error``; the interval is zero
        for a single finite observation and NaN when no value is finite.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    mean = float(np.mean(array))
    ci95 = (
        float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))
        if len(array) > 1 else 0.0
    )
    return mean, ci95


def read_evaluation_rows(path: Path) -> list[dict[str, str]]:
    """Read one evaluator CSV result file.

    Parameters
    ----------
    path : pathlib.Path
        ``generator_pgm_results.csv`` produced for one candidate count.
    """
    if not path.is_file():
        raise FileNotFoundError(f"evaluation result is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def reusable_run_matches(
    summary_path: Path,
    candidates: int,
    split: str,
    max_instances: int | None,
) -> bool:
    """Check whether an existing evaluator run matches the sensitivity point.

    Parameters
    ----------
    summary_path : pathlib.Path
        Existing ``generator_pgm_summary.json`` path.
    candidates : int
        Required candidates per instance.
    split : str
        Required validation or test split.
    max_instances : int or None
        Required instance count; ``None`` accepts the complete saved split.
    """
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return bool(
        summary.get("candidates_per_instance") == candidates
        and summary.get("data_split") == split
        and (
            max_instances is None
            or summary.get("n_instances") == max_instances
        )
    )


def run_candidate_evaluation(
    args: argparse.Namespace,
    candidates: int,
    run_dir: Path,
) -> None:
    """Run the native-PGM evaluator for one candidate count.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed common experiment and solver settings.
    candidates : int
        Number of Generator-TCN trajectories sampled per data instance.
    run_dir : pathlib.Path
        Dedicated evaluator output directory for this candidate count.
    """
    summary_path = run_dir / "generator_pgm_summary.json"
    result_path = run_dir / "generator_pgm_results.csv"
    if (
        args.reuse_existing
        and result_path.is_file()
        and reusable_run_matches(
            summary_path, candidates, args.split, args.max_instances,
        )
    ):
        print(f"K={candidates}: reusing {run_dir}", flush=True)
        return

    command = [
        sys.executable,
        str(args.evaluator),
        "--data", str(args.data),
        "--checkpoint", str(args.checkpoint),
        "--split", args.split,
        "--output-dir", str(run_dir),
        "--candidates", str(candidates),
        "--ramp-fraction", str(args.ramp_fraction),
        "--feasibility-tolerance", str(args.feasibility_tolerance),
        "--balance-tolerance-mva", str(args.balance_tolerance_mva),
        "--pgm-error-tolerance", str(args.pgm_error_tolerance),
        "--max-pf-iterations", str(args.max_pf_iterations),
        "--fixed-pv-chunk-size", str(args.fixed_pv_chunk_size),
        "--threading", str(args.threading),
        "--verify-instances", "0",
        "--verify-points", "0",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.max_instances is not None:
        command.extend(["--max-instances", str(args.max_instances)])
    print(f"K={candidates}: running {args.split} sensitivity point", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_candidate_run(
    candidates: int,
    rows: list[dict[str, str]],
) -> tuple[dict[str, float | int], dict[int, float]]:
    """Aggregate one candidate-count run into economic, time, and feasibility metrics.

    Parameters
    ----------
    candidates : int
        Number of candidates sampled per instance.
    rows : list[dict[str, str]]
        Per-instance rows produced by the native-PGM evaluator.

    Returns
    -------
    summary : dict
        Aggregate metrics for CSV, JSON, and plotting.
    feasible_cost_by_instance : dict[int, float]
        Finite best-feasible objective indexed by base-data instance.
    """
    if not rows:
        raise ValueError(f"K={candidates} evaluation contains no instances")
    feasible = np.asarray([int(row["feasible"]) for row in rows], dtype=bool)
    feasible_counts = np.asarray(
        [int(row["feasible_candidates"]) for row in rows], dtype=int,
    )
    costs = np.asarray([float(row["feasible_cost"]) for row in rows], dtype=float)
    total_seconds = [float(row["total_seconds"]) for row in rows]
    pgm_seconds = [float(row["pgm_batch_seconds"]) for row in rows]
    generation_seconds = [float(row["generation_seconds"]) for row in rows]
    failed_points = [float(row.get("pgm_failed_batch_points", 0)) for row in rows]
    failed_candidates = [float(row.get("pgm_failed_candidates", 0)) for row in rows]

    feasible_costs = costs[feasible & np.isfinite(costs)]
    mean_cost, cost_ci95 = finite_mean_ci(feasible_costs.tolist())
    mean_time, time_ci95 = finite_mean_ci(total_seconds)
    mean_pgm_time, _ = finite_mean_ci(pgm_seconds)
    mean_generation_time, _ = finite_mean_ci(generation_seconds)
    mean_failed_points, _ = finite_mean_ci(failed_points)
    mean_failed_candidates, _ = finite_mean_ci(failed_candidates)
    n_instances = len(rows)
    summary: dict[str, float | int] = {
        "candidates": candidates,
        "instances": n_instances,
        "feasible_instances": int(np.count_nonzero(feasible)),
        "instance_feasibility_rate": float(np.mean(feasible)),
        "candidate_feasibility_rate": float(
            np.sum(feasible_counts) / (n_instances * candidates)
        ),
        "mean_feasible_cost": mean_cost,
        "feasible_cost_ci95": cost_ci95,
        "mean_total_seconds": mean_time,
        "total_seconds_ci95": time_ci95,
        "mean_pgm_seconds": mean_pgm_time,
        "mean_generation_seconds": mean_generation_time,
        "mean_pgm_failed_points": mean_failed_points,
        "mean_pgm_failed_candidates": mean_failed_candidates,
    }
    feasible_cost_by_instance = {
        int(row["instance_index"]): float(row["feasible_cost"])
        for row in rows
        if int(row["feasible"]) and np.isfinite(float(row["feasible_cost"]))
    }
    return summary, feasible_cost_by_instance


def add_common_instance_costs(
    summaries: list[dict[str, float | int]],
    costs_by_candidates: list[dict[int, float]],
) -> list[int]:
    """Add fair economic statistics over instances feasible for every K.

    Parameters
    ----------
    summaries : list[dict]
        Mutable aggregate records ordered by candidate count.
    costs_by_candidates : list[dict[int, float]]
        Per-K feasible objective maps keyed by base-data instance.

    Returns
    -------
    list[int]
        Sorted base-data indices feasible for every candidate count.
    """
    common = set(costs_by_candidates[0])
    for costs in costs_by_candidates[1:]:
        common.intersection_update(costs)
    common_indices = sorted(common)
    for summary, costs in zip(summaries, costs_by_candidates):
        values = [costs[index] for index in common_indices]
        mean, ci95 = finite_mean_ci(values)
        summary["common_feasible_instances"] = len(common_indices)
        summary["mean_common_feasible_cost"] = mean
        summary["common_feasible_cost_ci95"] = ci95
    return common_indices


def write_summary_csv(path: Path, summaries: list[dict[str, float | int]]) -> None:
    """Write aggregate sensitivity records to CSV.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV path.
    summaries : list[dict]
        Candidate-count aggregate records with identical fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def plot_sensitivity(
    summaries: list[dict[str, float | int]],
    output_stem: Path,
) -> str:
    """Render the two-panel IEEE TSG sensitivity figure.

    Parameters
    ----------
    summaries : list[dict]
        Ordered aggregate metrics for all candidate counts.
    output_stem : pathlib.Path
        Figure path without an extension.
    Returns
    -------
    str
        Economic aggregation key used in panel (a).
    """
    configure_ieee_tsg_style()
    positions = np.arange(len(summaries))
    candidates = np.asarray([row["candidates"] for row in summaries], dtype=int)
    common_cost = np.asarray(
        [row["mean_common_feasible_cost"] for row in summaries], dtype=float,
    )
    use_common = np.isfinite(common_cost).all()
    cost_key = "mean_common_feasible_cost" if use_common else "mean_feasible_cost"
    costs = np.asarray([row[cost_key] for row in summaries], dtype=float) / 1e6
    times = np.asarray([row["mean_total_seconds"] for row in summaries], dtype=float)
    time_ci = np.asarray([row["total_seconds_ci95"] for row in summaries], dtype=float)
    feasibility = 100.0 * np.asarray(
        [row["instance_feasibility_rate"] for row in summaries], dtype=float,
    )

    blue, orange = "#0072B2", "#D55E00"
    figure, (cost_axis, time_axis) = plt.subplots(1, 2, figsize=(3.5, 2))
    cost_axis.plot(
        positions, costs, color=blue, marker="o",
        markersize=3.8, linewidth=1.1,
    )
    cost_axis.set_ylabel(r"Mean best cost ($10^6$)")
    cost_axis.set_xlabel(r"Candidates $K$")
    cost_axis.set_xticks(positions, candidates)

    time_axis.errorbar(
        positions, times, yerr=time_ci, color=orange, marker="s",
        markersize=3.6, linewidth=1.1, linestyle="-", capsize=1.8,
        capthick=0.7,
    )
    time_axis.set_ylabel("Mean time (s/instance)", color=orange)
    time_axis.set_xlabel(r"Candidates $K$")
    time_axis.set_xticks(positions, candidates)
    time_axis.tick_params(axis="y", labelcolor=orange)
    feasibility_axis = time_axis.twinx()
    feasibility_axis.plot(
        positions, feasibility, color=blue, marker="D", markersize=3.4,
        linewidth=1.1, linestyle="--",
    )
    feasibility_axis.set_ylabel("Feasible instances (%)", color=blue)
    feasibility_axis.tick_params(axis="y", labelcolor=blue)
    feasibility_bottom = max(0.0, float(np.min(feasibility)) - 10.0)
    feasibility_axis.set_ylim(feasibility_bottom, 102.0)

    for axis in (cost_axis, time_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=2.5, width=0.65)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
        plt.setp(
            axis.get_xticklabels(), rotation=40, ha="right",
            rotation_mode="anchor",
        )
    feasibility_axis.spines["top"].set_visible(False)
    feasibility_axis.spines["left"].set_visible(False)
    feasibility_axis.tick_params(direction="out", length=2.5, width=0.65)

    figure.subplots_adjust(
        left=0.15, right=0.86, top=0.97, bottom=0.32, wspace=0.42,
    )
    figure.canvas.draw()
    for axis, descriptor in zip(
        (cost_axis, time_axis),
        ("(a) Best feasible cost", "(b) Time and feasibility"),
    ):
        box = axis.get_position()
        figure.text(
            0.5 * (box.x0 + box.x1), 0.045, descriptor,
            ha="center", va="bottom", fontsize=8.5, fontweight="normal",
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=400)
    plt.close(figure)
    return cost_key


def build_parser() -> argparse.ArgumentParser:
    """Declare reproducible candidate-sensitivity experiment arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate-values", type=int, nargs="+", default=list(DEFAULT_CANDIDATES),
        help="positive candidate counts evaluated in the listed order",
    )
    parser.add_argument(
        "--max-instances", type=int, default=200,
        help="number of leading held-out instances per candidate count; 0 uses all",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="held-out data split shared by every sensitivity point",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="base SMPC dataset directory",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Generator-TCN checkpoint evaluated at every candidate count",
    )
    parser.add_argument(
        "--evaluator", type=Path, default=DEFAULT_EVALUATOR,
        help="native-PGM Generator-TCN evaluation program",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="sensitivity table, figure, metadata, and per-K run directory",
    )
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="reuse complete matching per-K outputs instead of rerunning them",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="neural inference and fixed-PV Newton device",
    )
    parser.add_argument(
        "--threading", type=int, default=0,
        help="PGM workers: negative is sequential, zero uses all hardware threads",
    )
    parser.add_argument(
        "--fixed-pv-chunk-size", type=int, default=1024,
        help="maximum float64 fixed-PV Newton points per dense batch",
    )
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="symmetric ramp limit as a fraction of generator Pmax",
    )
    parser.add_argument(
        "--feasibility-tolerance", type=float, default=1e-4,
        help="maximum normalized constraint violation for feasibility",
    )
    parser.add_argument(
        "--balance-tolerance-mva", type=float, default=1e-5,
        help="maximum nodal AC-balance residual in MVA",
    )
    parser.add_argument(
        "--pgm-error-tolerance", type=float, default=1e-8,
        help="native PGM Newton-Raphson error tolerance",
    )
    parser.add_argument(
        "--max-pf-iterations", type=int, default=30,
        help="maximum native PGM Newton iterations per operating point",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="common latent seed base used by every candidate count",
    )
    return parser


def main() -> None:
    """Run all sensitivity points, aggregate metrics, and export the figure."""
    args = build_parser().parse_args()
    if args.max_instances == 0:
        args.max_instances = None
    if (
        not args.candidate_values
        or any(value < 1 for value in args.candidate_values)
        or len(set(args.candidate_values)) != len(args.candidate_values)
    ):
        raise ValueError("candidate values must be unique positive integers")
    if (
        (args.max_instances is not None and args.max_instances < 1)
        or args.fixed_pv_chunk_size < 1
        or args.max_pf_iterations < 1
        or args.feasibility_tolerance < 0
        or args.balance_tolerance_mva < 0
        or args.pgm_error_tolerance <= 0
    ):
        raise ValueError("instance, solver, tolerance, or DPI setting is invalid")
    missing = [
        str(path) for path in (args.data, args.checkpoint, args.evaluator)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("missing sensitivity input paths: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float | int]] = []
    costs_by_candidates: list[dict[int, float]] = []
    for candidates in args.candidate_values:
        run_dir = args.output_dir / f"K_{candidates:03d}"
        run_candidate_evaluation(args, candidates, run_dir)
        rows = read_evaluation_rows(run_dir / "generator_pgm_results.csv")
        summary, costs = summarize_candidate_run(candidates, rows)
        summaries.append(summary)
        costs_by_candidates.append(costs)

    common_indices = add_common_instance_costs(summaries, costs_by_candidates)
    csv_path = args.output_dir / "candidate_sensitivity_results.csv"
    json_path = args.output_dir / "candidate_sensitivity_summary.json"
    figure_stem = args.output_dir / "candidate_sensitivity_ieee_tsg"
    write_summary_csv(csv_path, summaries)
    cost_key = plot_sensitivity(summaries, figure_stem)
    metadata = {
        "method": "Generator-TCN + fixed-PV/native-PGM batch screening",
        "data_split": args.split,
        "max_instances": args.max_instances,
        "candidate_values": args.candidate_values,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluator": str(args.evaluator.resolve()),
        "economic_figure_metric": cost_key,
        "common_feasible_instances": common_indices,
        "timing_definition": (
            "mean per-instance neural generation + fixed-PV Newton + native PGM "
            "batch + constraint assessment; evaluator warm-up is excluded"
        ),
        "pandapower_verification_in_timing": False,
        "results": summaries,
        "outputs": {
            "csv": str(csv_path.resolve()),
            "png": str(figure_stem.with_suffix('.png').resolve()),
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved sensitivity table: {csv_path}")
    print(f"saved sensitivity metadata: {json_path}")
    print(f"saved IEEE TSG figure: {figure_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
