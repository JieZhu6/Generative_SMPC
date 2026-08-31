"""Create a two-panel IEEE single-column dispatch-quality figure.

Panel (a) quantifies schedule diversity by pairwise normalized trajectory
distance. Panel (b) uses the benchmark's native PGM feasibility rule to show
every candidate's economic gap to the common IPOPT optimum. The same Gaussian
latent samples are used by S-CSNG, WD-CSNG, and CSNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data_generation.case14_pglib import load_case14  # noqa: E402
from evaluate_deterministic_tcn import load_deterministic_tcn  # noqa: E402
from evaluate_generator_tcn import (  # noqa: E402
    build_condition,
    checkpoint_split_indices,
    file_sha256,
    load_generator,
    screen_candidates_with_decs,
)
from evaluate_generator_tcn_pgm_batch import (  # noqa: E402
    assess_candidate_batch,
    flatten_candidate_points,
)
from pgm_batch_power_flow import build_pgm_case, solve_pv_batch  # noqa: E402
from Neural_network.decs import load_decs_checkpoint  # noqa: E402
from solve_shared_smpc_ipopt import (  # noqa: E402
    IPOPT_PATH,
    SharedTrajectorySMPCAcopf,
)


DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e14_N5000_S20_T16"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dispatch_distributions"
DEFAULT_DECS = ROOT / "Neural_network" / "decs_pgm_fixedpv.pt"
DEFAULT_INSTANCE_INDEX = 1966
DEFAULT_CHECKPOINTS = {
    "d_nn": ROOT / "Neural_network" / "deterministic_tcn.pt",
    "s_csng": ROOT / "Neural_network" / "generator_tcn_s_csng.pt",
    "wd_csng": ROOT / "Neural_network" / "generator_tcn_wd_csng.pt",
    "csng": ROOT / "Neural_network" / "generator_tcn_clip.pt",
}
DEFAULT_IPOPT_RESULTS = (
    ROOT / "output" / "five_benchmark_test" / "ipopt" / "ipopt_solutions.npz",
    ROOT / "output" / "ipopt_test" / "ipopt_solutions.npz",
    ROOT / "output" / "smpc_benchmark" / "ipopt_solutions.npz",
)
METHOD_LABELS = {
    "d_nn": "D-NN",
    "s_csng": "S-CSNG",
    "wd_csng": "WD-CSNG",
    "csng": "CSNG",
}
METHOD_COLORS = {
    "d_nn": "#0072B2",
    "s_csng": "#E69F00",
    "wd_csng": "#CC79A7",
    "csng": "#009E73",
}


def configure_plot_style() -> None:
    """Apply an IEEE single-column, colorblind-safe plotting style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def validate_common_protocol(
    checkpoints: dict[str, dict],
    data_dir: Path,
    split: str,
    n_instances: int,
) -> np.ndarray:
    """Verify all neural checkpoints use one split and normalization.

    Parameters
    ----------
    checkpoints : dict[str, dict]
        Loaded D-NN and generator checkpoint dictionaries.
    data_dir : pathlib.Path
        Base SMPC dataset containing saved split arrays.
    split : str
        ``validation`` or ``test``.
    n_instances : int
        Total number of base-data instances.

    Returns
    -------
    np.ndarray
        Common base-data indices in the requested split.
    """
    indices = {
        name: checkpoint_split_indices(checkpoint, data_dir, n_instances, split)
        for name, checkpoint in checkpoints.items()
    }
    reference_name = "csng"
    reference_indices = indices[reference_name]
    for name, values in indices.items():
        if not np.array_equal(values, reference_indices):
            raise ValueError(
                f"{METHOD_LABELS[name]} and CSNG use different {split} indices"
            )

    reference_norm = checkpoints[reference_name]["normalization"]
    for name, checkpoint in checkpoints.items():
        normalization = checkpoint["normalization"]
        if not (
            np.allclose(normalization["feature_min"], reference_norm["feature_min"])
            and np.allclose(normalization["feature_max"], reference_norm["feature_max"])
        ):
            raise ValueError(f"{METHOD_LABELS[name]} uses different input normalization")
    return reference_indices


def resolve_instance(
    split_indices: np.ndarray,
    test_position: int | None,
    instance_index: int | None,
) -> tuple[int, int]:
    """Resolve a split position and base-data instance index.

    Parameters
    ----------
    split_indices : np.ndarray
        Ordered base-data indices in the selected split.
    test_position : int or None
        Optional zero-based split position used when ``instance_index`` is omitted.
    instance_index : int or None
        Optional explicit base-data index that must belong to the split.

    Returns
    -------
    position, instance : tuple[int, int]
        Zero-based split position and corresponding base-data index.
    """
    if instance_index is not None:
        matches = np.flatnonzero(split_indices == instance_index)
        if not len(matches):
            raise ValueError(f"instance {instance_index} does not belong to this split")
        return int(matches[0]), int(instance_index)
    if test_position is None:
        raise ValueError("test_position and instance_index cannot both be omitted")
    if not 0 <= test_position < len(split_indices):
        raise ValueError(f"test_position must lie in [0,{len(split_indices) - 1}]")
    return test_position, int(split_indices[test_position])


def generate_neural_schedules(
    models: dict[str, torch.nn.Module],
    checkpoints: dict[str, dict],
    condition: torch.Tensor,
    candidates: int,
    latent_seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate one deterministic and three stochastic schedule sets.

    Parameters
    ----------
    models : dict[str, torch.nn.Module]
        Evaluation-mode D-NN, S-CSNG, WD-CSNG, and CSNG models.
    checkpoints : dict[str, dict]
        Matching checkpoint metadata with physical output bounds.
    condition : torch.Tensor, shape (1,T,condition_dim)
        Common normalized condition sequence.
    candidates : int
        Number of latent trajectories generated by every stochastic model.
    latent_seed : int
        Seed for the common Gaussian latent matrix.
    device : torch.device
        CPU or CUDA inference device.

    Returns
    -------
    schedules : dict[str, np.ndarray]
        D-NN array with shape ``(1,T,F)`` and generator arrays ``(K,T,F)``.
    latent : np.ndarray, shape (K,latent_dim)
        Shared latent samples saved for reproducibility.
    """
    latent_dims = {
        int(checkpoints[name]["model_config"]["latent_dim"])
        for name in ("s_csng", "wd_csng", "csng")
    }
    if len(latent_dims) != 1:
        raise ValueError("generative checkpoints use different latent dimensions")
    latent_dim = latent_dims.pop()
    latent_cpu = torch.randn(
        1, candidates, latent_dim,
        dtype=condition.dtype,
        generator=torch.Generator(device="cpu").manual_seed(latent_seed),
    )
    latent = latent_cpu.to(device)
    schedules: dict[str, np.ndarray] = {}

    d_checkpoint = checkpoints["d_nn"]
    d_lower = torch.as_tensor(d_checkpoint["free_variable_lower"], device=device)
    d_upper = torch.as_tensor(d_checkpoint["free_variable_upper"], device=device)
    d_ramp = torch.as_tensor(d_checkpoint["free_ramp_mw_per_period"], device=device)
    with torch.inference_mode():
        deterministic, _, _ = models["d_nn"](
            condition, d_lower, d_upper, d_ramp, d_ramp,
        )
    schedules["d_nn"] = deterministic.cpu().numpy()

    for name in ("s_csng", "wd_csng", "csng"):
        checkpoint = checkpoints[name]
        lower = torch.as_tensor(checkpoint["free_variable_lower"], device=device)
        upper = torch.as_tensor(checkpoint["free_variable_upper"], device=device)
        ramp = torch.as_tensor(checkpoint["free_ramp_mw_per_period"], device=device)
        with torch.inference_mode():
            generated, _, _ = models[name](
                condition, candidates, lower, upper, ramp, ramp, latent,
            )
        schedules[name] = generated[0].cpu().numpy()
    return schedules, latent_cpu[0].numpy()


def load_saved_ipopt_schedule(
    path: Path,
    instance_index: int,
) -> np.ndarray | None:
    """Load one finite IPOPT free-variable trajectory when present.

    Parameters
    ----------
    path : pathlib.Path
        Compressed result written by ``solve_shared_smpc_ipopt.py``.
    instance_index : int
        Base-data index to locate in ``instance_indices``.

    Returns
    -------
    np.ndarray or None
        Schedule with shape ``(T,F)``, or ``None`` if the instance is absent or
        its stored solve failed.
    """
    with np.load(path) as archive:
        if "instance_indices" in archive and "u" in archive:
            indices = np.asarray(archive["instance_indices"], dtype=int)
            matches = np.flatnonzero(indices == instance_index)
            if not len(matches):
                return None
            schedule = np.asarray(archive["u"][matches[0]], dtype=float)
        elif "instance_index" in archive and "ipopt" in archive:
            stored_instance = int(np.asarray(archive["instance_index"]))
            if stored_instance != instance_index:
                return None
            schedule = np.asarray(archive["ipopt"], dtype=float)
        else:
            raise ValueError(
                f"{path} is neither an IPOPT benchmark nor visualization archive"
            )
    return schedule if np.isfinite(schedule).all() else None


def resolve_ipopt_schedule(
    requested_path: Path | None,
    instance_index: int,
    current_load: np.ndarray,
    future_load: np.ndarray,
    ramp_fraction: float,
    ipopt_path: Path,
) -> tuple[np.ndarray, str]:
    """Load a matching IPOPT result or solve this single instance.

    Parameters
    ----------
    requested_path : pathlib.Path or None
        Explicit result archive. A missing explicit path is rejected.
    instance_index : int
        Base-data instance visualized by all methods.
    current_load : np.ndarray
        Current-period compressed load in MW/Mvar.
    future_load : np.ndarray
        Scenario-dependent future loads in MW/Mvar.
    ramp_fraction : float
        Symmetric ramp limit as a fraction of generator Pmax.
    ipopt_path : pathlib.Path
        IPOPT executable used only when no saved finite solution is available.

    Returns
    -------
    schedule, source : tuple[np.ndarray, str]
        Optimal free-variable trajectory and its file/solver provenance.
    """
    if requested_path is not None:
        if not requested_path.is_file():
            raise FileNotFoundError(f"IPOPT result archive not found: {requested_path}")
        schedule = load_saved_ipopt_schedule(requested_path, instance_index)
        if schedule is not None:
            return schedule, str(requested_path.resolve())
        print(
            f"instance {instance_index} is absent/non-finite in {requested_path}; "
            "solving it with IPOPT",
            flush=True,
        )
    else:
        for path in DEFAULT_IPOPT_RESULTS:
            if not path.is_file():
                continue
            schedule = load_saved_ipopt_schedule(path, instance_index)
            if schedule is not None:
                return schedule, str(path.resolve())

    optimizer = SharedTrajectorySMPCAcopf(
        horizon=1 + future_load.shape[1],
        n_scenarios=future_load.shape[0],
        ramp_fraction=ramp_fraction,
        ipopt_path=ipopt_path,
    )
    solved = optimizer.solve(current_load, future_load, tee=False)
    if not solved["solved"] or "schedule" not in solved:
        raise RuntimeError(
            f"IPOPT failed for instance {instance_index}: {solved['termination']}"
        )
    return np.asarray(solved["schedule"], dtype=float), "single-instance IPOPT solve"


def free_variable_labels() -> list[str]:
    """Return paper-order control labels with physical units."""
    case = load_case14()
    pg_labels = [
        f"$P^g_{{{case.generator_buses[index] + 1}}}$ (MW)"
        for index in case.nonreference_active_generators
    ]
    voltage_labels = [
        f"$V_{{{bus + 1}}}$ (p.u.)" for bus in case.voltage_control_buses
    ]
    return pg_labels + voltage_labels


def normalized_pairwise_distances(
    schedules: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return pairwise RMS distances in normalized full-trajectory space."""
    case = load_case14()
    lower = np.concatenate([
        case.gen[case.nonreference_active_generators, 9],
        case.bus[case.voltage_control_buses, 12],
    ])
    upper = np.concatenate([
        case.gen[case.nonreference_active_generators, 8],
        case.bus[case.voltage_control_buses, 11],
    ])
    scale = np.maximum(upper - lower, 1e-12)
    distances: dict[str, np.ndarray] = {}
    for name, values in schedules.items():
        normalized = ((values - lower) / scale).reshape(len(values), -1)
        if len(values) == 1:
            distances[name] = np.zeros(1)
            continue
        row, column = np.triu_indices(len(values), k=1)
        delta = normalized[row] - normalized[column]
        distances[name] = np.sqrt(np.mean(delta**2, axis=1))
    return distances


def assess_schedules_with_pgm(
    schedules: dict[str, np.ndarray],
    ipopt_schedule: np.ndarray,
    current_load: np.ndarray,
    future_load: np.ndarray,
    ramp_fraction: float,
    feasibility_tolerance: float,
    balance_tolerance_mva: float,
    pgm_error_tolerance: float,
    max_pf_iterations: int,
    threading: int,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Evaluate all neural candidates and the IPOPT schedule by native PGM."""
    case = load_case14()
    scenarios, future_steps = future_load.shape[:2]
    horizon = future_steps + 1
    schedule_sets = {**schedules, "ipopt": ipopt_schedule[None]}
    assessments: dict[str, dict[str, np.ndarray | float]] = {}
    for name, values in schedule_sets.items():
        # Native PGM 1.13 can retain batch-size-specific workspace state.
        # D-NN/IPOPT and generator batches differ in size, so each method gets
        # an independent model while retaining the identical network data.
        pgm_case = build_pgm_case(case)
        loads, controls = flatten_candidate_points(
            current_load, future_load, values,
        )
        states = solve_pv_batch(
            pgm_case, case, loads, controls,
            pgm_error_tolerance=pgm_error_tolerance,
            pgm_max_iterations=max_pf_iterations,
            threading=threading,
        )
        assessed = assess_candidate_batch(
            case, loads, states, len(values), scenarios, horizon,
            ramp_fraction, feasibility_tolerance, balance_tolerance_mva,
        )
        assessments[name] = {
            "feasible": np.asarray(assessed["feasible"], dtype=bool),
            "objective": np.asarray(assessed["objective"], dtype=float),
            "maximum_violation": np.asarray(
                assessed["maximum_violation"], dtype=float,
            ),
            "pgm_seconds": float(states["t_total"]),
        }
    return assessments


def assess_schedules_with_decs(
    schedules: dict[str, np.ndarray],
    ipopt_schedule: np.ndarray,
    current_load: np.ndarray,
    future_load: np.ndarray,
    ramp_fraction: float,
    feasibility_tolerance: float,
    decs_path: Path,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Evaluate candidate landscapes with the frozen differentiable completion."""
    completion = load_decs_checkpoint(decs_path, device)
    case = load_case14()
    reference_ramp = torch.tensor(
        ramp_fraction * case.gen[case.reference_generator, 8],
        dtype=torch.float32, device=device,
    )
    schedule_sets = {**schedules, "ipopt": ipopt_schedule[None]}
    assessments: dict[str, dict[str, np.ndarray | float]] = {}
    for name, values in schedule_sets.items():
        screened = screen_candidates_with_decs(
            completion,
            torch.as_tensor(values, dtype=torch.float32, device=device),
            current_load,
            future_load,
            reference_ramp,
            feasibility_tolerance,
        )
        assessments[name] = {
            "feasible": np.asarray(screened["feasible"], dtype=bool),
            "objective": np.asarray(screened["objective"], dtype=float),
            "maximum_violation": np.asarray(
                screened["maximum_violation"], dtype=float,
            ),
            "pgm_seconds": np.nan,
        }
    return assessments


def plot_dispatch_distributions(
    schedules: dict[str, np.ndarray],
    assessments: dict[str, dict[str, np.ndarray | float]],
    png_path: Path,
    pdf_path: Path,
    tiff_path: Path,
    dpi: int,
    validation_backend: str,
) -> dict[str, dict[str, float | int | None]]:
    """Render the two-panel diversity, feasibility, and economy figure."""
    configure_plot_style()
    methods = ("d_nn", "s_csng", "wd_csng", "csng")
    distances = normalized_pairwise_distances(schedules)
    ipopt_cost = float(np.asarray(assessments["ipopt"]["objective"])[0])
    if not np.isfinite(ipopt_cost):
        raise RuntimeError("PGM did not produce a finite IPOPT reference cost")

    figure, (diversity_axis, quality_axis) = plt.subplots(
        1, 2, figsize=(3.5, 2.15),
    )
    positions = np.arange(len(methods))
    labels = [METHOD_LABELS[name] for name in methods]

    for position, name in enumerate(methods):
        values = distances[name]
        color = METHOD_COLORS[name]
        if len(schedules[name]) > 1:
            violin = diversity_axis.violinplot(
                values, positions=[position], widths=0.72,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.30)
            q25, median, q75 = np.quantile(values, (0.25, 0.50, 0.75))
            diversity_axis.vlines(position, q25, q75, color=color, linewidth=3.0)
            diversity_axis.scatter(
                position, median, s=17, marker="o", facecolor="white",
                edgecolor=color, linewidth=0.9, zorder=4,
            )
        else:
            diversity_axis.scatter(
                position, 0.0, s=22, marker="D", facecolor=color,
                edgecolor="white", linewidth=0.5, zorder=4,
            )

    diversity_axis.set_ylabel("Normalized RMS distance")
    diversity_axis.set_xticks(positions, labels)
    diversity_axis.set_xlim(-0.55, len(methods) - 0.45)
    diversity_axis.set_ylim(bottom=-0.008)
    diversity_axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    diversity_axis.text(
        0.5, -0.35, "(a) Diversity",
        transform=diversity_axis.transAxes,
        ha="center", va="top", fontsize=8.5,
    )

    for position, name in enumerate(methods):
        feasible = np.asarray(assessments[name]["feasible"], dtype=bool)
        objective = np.asarray(assessments[name]["objective"], dtype=float)
        gap = 100.0 * (objective - ipopt_cost) / max(abs(ipopt_cost), 1e-12)
        finite = np.isfinite(gap)
        rng = np.random.default_rng(710 + position)
        jitter = rng.uniform(-0.16, 0.16, len(gap))
        color = METHOD_COLORS[name]
        good = finite & feasible
        bad = finite & ~feasible
        quality_axis.scatter(
            position + jitter[bad], gap[bad], s=13, marker="o",
            facecolor="white", edgecolor=color, linewidth=0.75,
            alpha=0.90, zorder=2,
        )
        quality_axis.scatter(
            position + jitter[good], gap[good], s=13, marker="o",
            facecolor=color, edgecolor="white", linewidth=0.35,
            alpha=0.72, zorder=3,
        )
        if np.any(good):
            best = np.flatnonzero(good)[np.argmin(gap[good])]
            quality_axis.scatter(
                position, gap[best], s=65, marker="*", facecolor=color,
                edgecolor="#111111", linewidth=0.55, zorder=5,
            )
    quality_axis.axhline(
        0.0, color="#111111", linestyle="--", linewidth=0.9, zorder=1,
    )
    quality_axis.set_ylabel("Cost gap to IPOPT (%)")
    quality_axis.set_xticks(positions, labels)
    quality_axis.set_xlim(-0.55, len(methods) - 0.45)
    quality_axis.margins(y=0.14)
    quality_axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    quality_axis.text(
        0.5, -0.35, "(b) Solution quality",
        transform=quality_axis.transAxes,
        ha="center", va="top", fontsize=8.5,
    )
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#777777",
               markeredgecolor="white", markersize=4.5,
               label=f"Feasible"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#C91313", markersize=4.5, label="Infeasible"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#777777",
               markeredgecolor="#111111", markersize=7, label="Best"),
        Line2D([0], [0], color="#111111", linestyle="--", linewidth=0.9,
               label="IPOPT"),
    ]
    # figure.legend(
    #     handles=[
    #         *legend_handles,
    #     ],
    #     loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
    #     frameon=False, columnspacing=0.72, handletextpad=0.35,
    # )
    for axis in (diversity_axis, quality_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=2.5, width=0.65)
        plt.setp(
            axis.get_xticklabels(), rotation=35, ha="right",
            rotation_mode="anchor",
        )

    figure.subplots_adjust(
        left=0.145, right=0.99, top=0.88, bottom=0.29, wspace=0.50,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(pdf_path)
    figure.savefig(
        tiff_path, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in methods:
        feasible = np.asarray(assessments[name]["feasible"], dtype=bool)
        objective = np.asarray(assessments[name]["objective"], dtype=float)
        feasible_cost = objective[feasible & np.isfinite(objective)]
        metrics[name] = {
            "candidates": int(len(objective)),
            "feasible_candidates": int(feasible.sum()),
            "pairwise_distance_median": float(np.median(distances[name])),
            "pairwise_distance_p95": float(np.quantile(distances[name], 0.95)),
            "best_feasible_cost": (
                float(feasible_cost.min()) if len(feasible_cost) else None
            ),
            "best_feasible_cost_gap_percent": (
                float(100.0 * (feasible_cost.min() - ipopt_cost) / abs(ipopt_cost))
                if len(feasible_cost) else None
            ),
        }
    metrics["ipopt"] = {"objective": ipopt_cost}
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Declare reproducible inference, IPOPT, and figure arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="SMPC dataset containing current, pooled, future, and split arrays",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="held-out split containing the visualized instance",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--test-position", type=int, default=None,
        help="zero-based position inside the selected split; overrides the fixed example",
    )
    selection.add_argument(
        "--instance-index", type=int, default=None,
        help=(
            "explicit base-data instance index belonging to the selected split; "
            f"if neither selector is given, test instance {DEFAULT_INSTANCE_INDEX} is used"
        ),
    )
    parser.add_argument(
        "--candidates", type=int, default=50,
        help="common latent samples generated by each stochastic network",
    )
    parser.add_argument(
        "--latent-seed", type=int, default=2026,
        help="base seed; the selected base-data index is added to it",
    )
    parser.add_argument(
        "--d-nn-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["d_nn"],
        help="trained latent-free D-NN checkpoint",
    )
    parser.add_argument(
        "--s-csng-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["s_csng"],
        help="trained sigmoid-output S-CSNG checkpoint",
    )
    parser.add_argument(
        "--wd-csng-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["wd_csng"],
        help="trained no-diversity WD-CSNG checkpoint",
    )
    parser.add_argument(
        "--csng-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["csng"],
        help="trained proposed CSNG checkpoint",
    )
    parser.add_argument(
        "--ipopt-solutions", type=Path, default=None,
        help="optional saved ipopt_solutions.npz; otherwise common paths are searched",
    )
    parser.add_argument(
        "--ipopt-path", type=Path, default=IPOPT_PATH,
        help="IPOPT executable used if no saved finite solution contains the instance",
    )
    parser.add_argument(
        "--validation-backend", choices=("pgm", "decs"), default="pgm",
        help="candidate feasibility/cost backend; PGM is the strict benchmark backend",
    )
    parser.add_argument(
        "--decs", type=Path, default=DEFAULT_DECS,
        help="frozen completion checkpoint used only by the DECS figure backend",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="directory for PNG, PDF, TIFF, NPZ, and JSON outputs",
    )
    parser.add_argument(
        "--feasibility-tolerance", type=float, default=1e-4,
        help="maximum normalized PGM constraint excess",
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
        help="maximum native PGM Newton iterations",
    )
    parser.add_argument(
        "--threading", type=int, default=0,
        help="PGM workers: negative is sequential, zero uses all hardware threads",
    )
    parser.add_argument(
        "--dpi", type=int, default=600,
        help="PNG and TIFF resolution in dots per inch",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="neural inference device; auto selects CUDA when available",
    )
    return parser


def main() -> None:
    """Generate comparable schedules, resolve IPOPT, and save the figure."""
    args = build_parser().parse_args()
    if args.candidates < 2 or args.dpi < 300 or args.max_pf_iterations < 1:
        raise ValueError(
            "candidates must be at least two, dpi at least 300, and iterations positive"
        )
    if (
        args.feasibility_tolerance < 0
        or args.balance_tolerance_mva < 0
        or args.pgm_error_tolerance <= 0
    ):
        raise ValueError("PGM feasibility and solver tolerances are invalid")
    paths = {
        "d_nn": args.d_nn_checkpoint,
        "s_csng": args.s_csng_checkpoint,
        "wd_csng": args.wd_csng_checkpoint,
        "csng": args.csng_checkpoint,
    }
    required_paths = [args.data, *paths.values()]
    if args.validation_backend == "decs":
        required_paths.append(args.decs)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing input paths: " + ", ".join(missing))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    d_model, d_checkpoint = load_deterministic_tcn(paths["d_nn"], device)
    models: dict[str, torch.nn.Module] = {"d_nn": d_model}
    checkpoints: dict[str, dict] = {"d_nn": d_checkpoint}
    for name in ("s_csng", "wd_csng", "csng"):
        model, checkpoint = load_generator(paths[name], device, name)
        models[name] = model
        checkpoints[name] = checkpoint
    decs_matches_checkpoint = None
    if args.validation_backend == "decs":
        decs_digest = file_sha256(args.decs)
        decs_matches_checkpoint = {
            name: checkpoint.get("decs_checkpoint_sha256") == decs_digest
            for name, checkpoint in checkpoints.items()
        }
        mismatched = [
            METHOD_LABELS[name] for name, matches in decs_matches_checkpoint.items()
            if not matches
        ]
        if mismatched:
            print(
                "warning: figure DECS differs from the training DECS for "
                + ", ".join(mismatched),
                flush=True,
            )

    current = np.load(args.data / "current_load.npy", mmap_mode="r")
    pool = np.load(args.data / "future_pool.npy", mmap_mode="r")
    future = np.load(args.data / "future_load.npy", mmap_mode="r")
    split_indices = validate_common_protocol(
        checkpoints, args.data, args.split, len(current),
    )
    if args.test_position is None and args.instance_index is None:
        if args.split == "test":
            args.instance_index = DEFAULT_INSTANCE_INDEX
        else:
            args.test_position = 0
    split_position, instance_index = resolve_instance(
        split_indices, args.test_position, args.instance_index,
    )
    normalization = checkpoints["csng"]["normalization"]
    condition_np = build_condition(
        np.asarray(current[instance_index]), np.asarray(pool[instance_index]),
        np.asarray(normalization["feature_min"]),
        np.asarray(normalization["feature_max"]),
    )
    condition = torch.from_numpy(condition_np)[None].to(device)
    effective_seed = args.latent_seed + instance_index
    schedules, latent = generate_neural_schedules(
        models, checkpoints, condition, args.candidates, effective_seed, device,
    )

    ramp_fractions = {
        float(checkpoint["ramp_fraction_of_pmax"])
        for checkpoint in checkpoints.values()
    }
    if len(ramp_fractions) != 1:
        raise ValueError("neural checkpoints use different ramp fractions")
    ipopt_schedule, ipopt_source = resolve_ipopt_schedule(
        args.ipopt_solutions, instance_index,
        np.asarray(current[instance_index]), np.asarray(future[instance_index]),
        ramp_fractions.pop(), args.ipopt_path,
    )
    ramp_fraction = float(checkpoints["csng"]["ramp_fraction_of_pmax"])
    if args.validation_backend == "pgm":
        assessments = assess_schedules_with_pgm(
            schedules, ipopt_schedule,
            np.asarray(current[instance_index]), np.asarray(future[instance_index]),
            ramp_fraction,
            args.feasibility_tolerance,
            args.balance_tolerance_mva,
            args.pgm_error_tolerance,
            args.max_pf_iterations,
            args.threading,
        )
    else:
        assessments = assess_schedules_with_decs(
            schedules, ipopt_schedule,
            np.asarray(current[instance_index]), np.asarray(future[instance_index]),
            ramp_fraction,
            args.feasibility_tolerance,
            args.decs,
            device,
        )

    labels = free_variable_labels()
    stem = (
        f"dispatch_quality_ieee_{args.validation_backend}_{args.split}"
        f"_instance_{instance_index}"
    )
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    tiff_path = args.output_dir / f"{stem}.tiff"
    npz_path = args.output_dir / f"{stem}.npz"
    json_path = args.output_dir / f"{stem}.json"
    metrics = plot_dispatch_distributions(
        schedules, assessments, png_path, pdf_path, tiff_path, args.dpi,
        args.validation_backend,
    )
    assessment_payload = {}
    for name, values in assessments.items():
        for field in ("feasible", "objective", "maximum_violation"):
            assessment_payload[f"{name}_{field}"] = values[field]
    np.savez_compressed(
        npz_path,
        instance_index=np.asarray(instance_index),
        split_position=np.asarray(split_position),
        latent_seed=np.asarray(effective_seed),
        latent=latent,
        d_nn=schedules["d_nn"],
        s_csng=schedules["s_csng"],
        wd_csng=schedules["wd_csng"],
        csng=schedules["csng"],
        ipopt=ipopt_schedule,
        **assessment_payload,
    )
    metadata = {
        "data_split": args.split,
        "split_position": split_position,
        "instance_index": instance_index,
        "generative_candidates": args.candidates,
        "latent_seed": effective_seed,
        "ipopt_source": ipopt_source,
        "validation_backend": args.validation_backend,
        "decs_checkpoint": (
            str(args.decs.resolve()) if args.validation_backend == "decs" else None
        ),
        "decs_matches_training_checkpoint": decs_matches_checkpoint,
        "free_variables": labels,
        "pgm_feasibility_tolerance": args.feasibility_tolerance,
        "pgm_balance_tolerance_mva": args.balance_tolerance_mva,
        "pgm_error_tolerance": args.pgm_error_tolerance,
        "figure_width_inches": 3.5,
        "metrics": metrics,
        "checkpoints": {name: str(path.resolve()) for name, path in paths.items()},
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "tiff": str(tiff_path.resolve()),
            "npz": str(npz_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"visualized {args.split} position {split_position}, base instance "
        f"{instance_index}, K={args.candidates}, device={device}",
        flush=True,
    )
    print(f"IPOPT reference: {ipopt_source}")
    print(f"saved figure: {png_path}")
    print(f"saved vector figure: {pdf_path}")
    print(f"saved TIFF figure: {tiff_path}")
    print(f"saved raw schedules: {npz_path}")
    print(f"saved metadata: {json_path}")


if __name__ == "__main__":
    main()
