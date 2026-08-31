"""Visualize current Generator-TCN economics, feasibility, and diversity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import load_decs_checkpoint  # noqa: E402
from Neural_network.noncausal_tcn import (  # noqa: E402
    PROJECTION_METHOD,
    ConditionalStochasticTCN,
)
from evaluate_generator_tcn import checkpoint_test_indices  # noqa: E402
from Neural_network.train_generator import GeneratorDataset, file_sha256  # noqa: E402
from Neural_network.validation.tcn.generator_metrics import (  # noqa: E402
    evaluate_generator_distributions,
)


DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e14_N5000_S20_T16"
DEFAULT_CHECKPOINT = ROOT / "Neural_network" / "generator_tcn_clip.pt"
DEFAULT_DECS = ROOT / "Neural_network" / "decs_pgm_fixedpv.pt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def finite_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize one numerical distribution after removing NaN and infinity.

    Parameters
    ----------
    values : np.ndarray
        Distribution with arbitrary shape and optional non-finite entries.

    Returns
    -------
    dict
        Count, mean, standard deviation, median, and 5/95 percentiles.
    """
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0, "mean": None, "std": None,
            "median": None, "p05": None, "p95": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def summarize_results(
    result: dict[str, np.ndarray],
    checkpoint: dict,
    checkpoint_path: Path,
    candidates: int,
    tolerance: float,
    latent_seed: int,
    test_indices: np.ndarray,
) -> dict:
    """Build a JSON-safe scientific summary of the three metric families.

    Parameters
    ----------
    result : dict[str, np.ndarray]
        Raw distributions returned by ``evaluate_generator_distributions``.
    checkpoint : dict
        Loaded Generator-TCN checkpoint metadata.
    checkpoint_path : pathlib.Path
        Actual checkpoint path used for this validation run.
    candidates : int
        Number of candidates generated per test scenario tree.
    tolerance : float
        Normalized feasibility threshold applied to candidate maxima.
    latent_seed : int
        Reproducible Gaussian latent seed used in this evaluation.
    test_indices : np.ndarray
        Original base-data indices of evaluated test instances.

    Returns
    -------
    dict
        JSON-serializable protocol, economic, feasibility, and diversity metrics.
    """
    feasible = result["candidate_feasible"]
    hit = feasible.any(axis=1)
    max_violation = result["candidate_max_violation"]
    category_names = ("pg", "qg", "voltage", "angle", "thermal", "ramp")
    return {
        "protocol": {
            "checkpoint": str(checkpoint_path.resolve()),
            "best_checkpoint_epoch": checkpoint.get("early_stopping", {}).get("best_epoch"),
            "loss_definition": checkpoint.get("loss_definition", "legacy_generator_loss"),
            "backend": "frozen DECS completion with reconstructed AC inequalities",
            "backend_note": (
                "This broad distribution evaluation matches generator training. "
                "Use evaluate_generator_tcn.py for exact pandapower certification."
            ),
            "test_instances": int(len(test_indices)),
            "test_index_min": int(test_indices.min()),
            "test_index_max": int(test_indices.max()),
            "candidates_per_instance": int(candidates),
            "feasibility_tolerance": float(tolerance),
            "latent_seed": int(latent_seed),
            "ramp_fraction_of_pmax": float(checkpoint["ramp_fraction_of_pmax"]),
        },
        "economics": {
            "all_candidate_horizon_cost": finite_statistics(result["candidate_cost"]),
            "best_candidate_cost_per_instance": finite_statistics(result["best_candidate_cost"]),
            "best_feasible_cost_per_hit_instance": finite_statistics(result["best_feasible_cost"]),
        },
        "feasibility": {
            "candidate_feasible_rate": float(feasible.mean()),
            "instance_hit_rate": float(hit.mean()),
            "hit_instances": int(hit.sum()),
            "candidate_max_normalized_violation": finite_statistics(max_violation),
            "candidate_mean_normalized_violation": finite_statistics(
                result["candidate_mean_violation"],
            ),
            "near_feasible_rate_at_1e-3": float((max_violation <= 1e-3).mean()),
            "near_feasible_rate_at_1e-2": float((max_violation <= 1e-2).mean()),
            "category_max_normalized_violation": {
                name: finite_statistics(result[f"violation_{name}"])
                for name in category_names
            },
        },
        "diversity": {
            "definition": {
                "first_stage": "Euclidean pair distance in paper-style centered [-1,1] coordinates",
                "trajectory": "RMS pair distance over globally normalized time-variable coordinates",
            },
            "first_stage_pairwise_distance": finite_statistics(
                result["pairwise_first_stage_distance"],
            ),
            "trajectory_pairwise_rms_distance": finite_statistics(
                result["pairwise_trajectory_distance"],
            ),
            "instance_mean_first_stage_distance": finite_statistics(
                result["instance_first_stage_diversity"],
            ),
            "instance_mean_trajectory_distance": finite_statistics(
                result["instance_trajectory_diversity"],
            ),
        },
    }


def add_histogram(
    axis: plt.Axes,
    values: np.ndarray,
    label: str,
    color: str,
    bins: int = 45,
    density: bool = True,
) -> None:
    """Draw one finite distribution as a histogram or singleton marker.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target subplot.
    values : np.ndarray
        Values to flatten and filter before plotting.
    label : str
        Legend label.
    color : str
        Matplotlib-compatible color.
    bins : int, default=45
        Number of histogram intervals.
    density : bool, default=True
        Normalize histogram area when true.
    """
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if len(finite) > 1:
        axis.hist(
            finite, bins=bins, density=density, alpha=0.48,
            color=color, label=f"{label} (n={len(finite)})",
        )
    elif len(finite) == 1:
        axis.axvline(
            finite[0], color=color, linewidth=2.2,
            label=f"{label} (n=1)",
        )


def plot_results(
    result: dict[str, np.ndarray],
    summary: dict,
    output: Path,
    dpi: int,
) -> None:
    """Save a six-panel distribution figure for the three validation metrics.

    Parameters
    ----------
    result : dict[str, np.ndarray]
        Raw validation distributions.
    summary : dict
        Scalar summary returned by ``summarize_results``.
    output : pathlib.Path
        Destination PNG path.
    dpi : int
        Positive raster resolution in dots per inch.
    """
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))
    colors = {"all": "#4C78A8", "best": "#F58518", "feasible": "#54A24B"}

    add_histogram(axes[0, 0], result["candidate_cost"], "all candidates", colors["all"])
    add_histogram(axes[0, 0], result["best_candidate_cost"], "best candidate", colors["best"])
    add_histogram(axes[0, 0], result["best_feasible_cost"], "best feasible", colors["feasible"])
    axes[0, 0].set(title="A  Economic-cost distributions", xlabel="Expected horizon cost", ylabel="Density")
    axes[0, 0].legend(frameon=False, fontsize=8)

    log_violation = np.log10(np.maximum(result["candidate_max_violation"].ravel(), 1e-12))
    add_histogram(axes[0, 1], log_violation, "candidate maximum", "#E45756", density=False)
    axes[0, 1].axvline(
        np.log10(summary["protocol"]["feasibility_tolerance"]),
        color="black", linestyle="--", linewidth=1.4, label="feasibility threshold",
    )
    axes[0, 1].set(
        title="B  Feasibility distribution",
        xlabel=r"$\log_{10}(\max\ normalized\ violation)$", ylabel="Candidate count",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    category_names = ("Pg", "Qg", "V", "angle", "thermal", "ramp")
    category_values = [
        np.log10(np.maximum(result[f"violation_{name.lower() if name != 'V' else 'voltage'}"].ravel(), 1e-12))
        for name in category_names
    ]
    box = axes[0, 2].boxplot(
        category_values, tick_labels=category_names, showfliers=False,
        patch_artist=True, whis=(5, 95),
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#72B7B2")
        patch.set_alpha(0.75)
    axes[0, 2].axhline(-6.0, color="black", linestyle="--", linewidth=1.2)
    axes[0, 2].set(
        title="C  Violation by constraint family",
        ylabel=r"$\log_{10}(\max\ normalized\ violation)$",
    )

    add_histogram(
        axes[1, 0], result["pairwise_first_stage_distance"],
        "first-stage pairs", "#B279A2",
    )
    add_histogram(
        axes[1, 0], result["pairwise_trajectory_distance"],
        "full-trajectory pairs", "#FF9DA6",
    )
    axes[1, 0].set(
        title="D  Pairwise diversity distributions",
        xlabel="Normalized pair distance", ylabel="Density",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    add_histogram(
        axes[1, 1], result["instance_first_stage_diversity"],
        "instance mean: first stage", "#B279A2",
    )
    add_histogram(
        axes[1, 1], result["instance_trajectory_diversity"],
        "instance mean: trajectory", "#FF9DA6",
    )
    axes[1, 1].set(
        title="E  Diversity across scenario trees",
        xlabel="Mean pair distance per instance", ylabel="Density",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    variable_labels = (r"$P_{g,2}$", r"$V_1$", r"$V_2$", r"$V_3$", r"$V_6$", r"$V_8$")
    std_values = result["first_stage_std_by_variable"]
    box = axes[1, 2].boxplot(
        [std_values[:, index] for index in range(std_values.shape[1])],
        tick_labels=variable_labels[:std_values.shape[1]], showfliers=False,
        patch_artist=True, whis=(5, 95),
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#9D755D")
        patch.set_alpha(0.75)
    axes[1, 2].set(
        title="F  First-stage spread by decision",
        ylabel="Candidate standard deviation (global range units)",
    )

    feasibility = summary["feasibility"]
    figure.suptitle(
        "Generator-TCN held-out distributions (DECS-based screening)\n"
        f"candidate feasible={100 * feasibility['candidate_feasible_rate']:.4f}% | "
        f"instance hit={100 * feasibility['instance_hit_rate']:.2f}% | "
        f"K={summary['protocol']['candidates_per_instance']}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Declare reproducible Generator-TCN validation command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="SMPC scenario-tree dataset")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="trained Generator-TCN checkpoint")
    parser.add_argument("--decs", type=Path, default=DEFAULT_DECS, help="frozen DECS checkpoint used for broad screening")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for PNG, JSON, and compressed raw distributions")
    parser.add_argument("--candidates", type=int, default=50, help="latent trajectories generated per test instance; at least two")
    parser.add_argument("--batch-size", type=int, default=8, help="test scenario trees evaluated per GPU/CPU batch")
    parser.add_argument("--max-instances", type=int, default=None, help="optional positive test prefix for smoke validation; default uses all 500")
    parser.add_argument("--feasibility-tolerance", type=float, default=None, help="normalized inequality threshold; default uses checkpoint value")
    parser.add_argument("--latent-seed", type=int, default=None, help="trajectory latent seed; default uses checkpoint seed plus 29")
    parser.add_argument("--progress-every", type=int, default=10, help="print progress after this many batches")
    parser.add_argument("--dpi", type=int, default=200, help="output PNG resolution in dots per inch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="inference device; auto selects CUDA when available")
    return parser


def main() -> None:
    """Run held-out inference and save TCN validation distributions and plots."""
    args = build_parser().parse_args()
    if min(args.candidates, args.batch_size, args.progress_every, args.dpi) < 1:
        raise ValueError("candidate, batch, progress, and dpi values must be positive")
    if args.candidates < 2:
        raise ValueError("candidates must be at least two for diversity")
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive when provided")
    if not args.data.is_dir() or not args.checkpoint.is_file() or not args.decs.is_file():
        raise FileNotFoundError("SMPC data, Generator-TCN checkpoint, or DECS checkpoint is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("decs_checkpoint_sha256") != file_sha256(args.decs):
        raise ValueError("visualization DECS differs from the Generator training DECS")
    if checkpoint.get("model_config", {}).get("projection_method") != PROJECTION_METHOD:
        raise ValueError(
            "visualization requires a newly trained inward-clip Generator checkpoint"
        )
    model = ConditionalStochasticTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    completion = load_decs_checkpoint(args.decs, device)
    normalization = checkpoint["normalization"]
    n_instances = int(np.load(args.data / "current_load.npy", mmap_mode="r").shape[0])
    test_indices = checkpoint_test_indices(
        checkpoint, args.data, n_instances, args.max_instances,
    )
    dataset = GeneratorDataset(
        args.data, test_indices,
        np.asarray(normalization["feature_min"]),
        np.asarray(normalization["feature_max"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    tolerance = (
        float(checkpoint["loss_hyperparameters"]["feasibility_tolerance"])
        if args.feasibility_tolerance is None else args.feasibility_tolerance
    )
    latent_seed = int(checkpoint["seed"] + 29 if args.latent_seed is None else args.latent_seed)
    if tolerance < 0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    active_ramp = checkpoint.get("active_ramp_mw_per_period")
    if active_ramp is None:
        ramp_fraction = float(checkpoint["ramp_fraction_of_pmax"])
        active_ramp = ramp_fraction * completion.physics.pg_max[completion.physics.active]

    print(
        f"Generator-TCN validation: device={device}, instances={len(dataset)}, "
        f"candidates={args.candidates}, tolerance={tolerance:.1e}",
    )
    result = evaluate_generator_distributions(
        model=model,
        completion=completion,
        loader=loader,
        free_lower=torch.as_tensor(checkpoint["free_variable_lower"], device=device),
        free_upper=torch.as_tensor(checkpoint["free_variable_upper"], device=device),
        free_ramp=torch.as_tensor(checkpoint["free_ramp_mw_per_period"], device=device),
        active_ramp=torch.as_tensor(active_ramp, device=device),
        candidates=args.candidates,
        feasibility_tolerance=tolerance,
        latent_seed=latent_seed,
        device=device,
        progress_every=args.progress_every,
    )
    result["test_indices"] = test_indices
    summary = summarize_results(
        result, checkpoint, args.checkpoint, args.candidates,
        tolerance, latent_seed, test_indices,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "generator_tcn_validation.png"
    metrics_path = args.output_dir / "generator_tcn_validation.json"
    raw_path = args.output_dir / "generator_tcn_validation_data.npz"
    plot_results(result, summary, figure_path, args.dpi)
    np.savez_compressed(raw_path, **result)
    metrics_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    feasibility = summary["feasibility"]
    economics = summary["economics"]
    diversity = summary["diversity"]
    print(
        f"candidate feasible={100 * feasibility['candidate_feasible_rate']:.4f}% | "
        f"instance hit={100 * feasibility['instance_hit_rate']:.2f}% "
        f"({feasibility['hit_instances']}/{len(test_indices)})",
    )
    print(
        "best feasible cost mean="
        f"{economics['best_feasible_cost_per_hit_instance']['mean']} | "
        "trajectory pair distance mean="
        f"{diversity['trajectory_pairwise_rms_distance']['mean']:.4f}",
    )
    print(f"saved figure: {figure_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved raw distributions: {raw_path}")


if __name__ == "__main__":
    main()
