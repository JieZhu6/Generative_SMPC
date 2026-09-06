"""Visualize DECS accuracy for system inequality-constraint decisions.

The paper figure contains two horizontally arranged panels: full-range
normalized signed-residual parity and false feasible/false infeasible rates for
the five dependent inequality blocks. The parity panel uses symmetric-log axes
to retain both boundary detail and distant feasible/violating values. Optional
OOD experiments are written to the metrics JSON without adding figure panels.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case14_pglib import Case14, load_case14  # noqa: E402
from Data_generation.generate_decs_dataset import (  # noqa: E402
    build_pandapower_network,
    solve_power_flow,
)
from Neural_network.decs import (  # noqa: E402
    DifferentiableEqualityCompletion,
    load_decs_checkpoint,
)


def generator_state_variables(
    state: dict[str, torch.Tensor],
    completion: DifferentiableEqualityCompletion,
) -> dict[str, torch.Tensor]:
    """Extract voltage, branch-angle, and apparent-power state variables.

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        AC state returned by ``completion.physics.reconstruct``.
    completion : DifferentiableEqualityCompletion
        DECS completion object supplying branch endpoint indices.

    Returns
    -------
    dict[str, torch.Tensor]
        Bus voltages in p.u., branch angle differences in rad, and apparent
        powers at both branch ends in MVA.
    """
    physics = completion.physics
    angle_difference = state["va"][:, physics.fbus] - state["va"][:, physics.tbus]
    apparent_power = torch.cat([
        torch.hypot(state["pf"], state["qf"]),
        torch.hypot(state["pt"], state["qt"]),
    ], dim=1)
    return {
        "voltage": state["vm"],
        "angle_difference": angle_difference,
        "apparent_power": apparent_power,
    }


def signed_constraint_residuals(
    state: dict[str, torch.Tensor],
    completion: DifferentiableEqualityCompletion,
) -> dict[str, torch.Tensor]:
    """Return signed normalized residuals for every fitted inequality block.

    The definitions match ``ACReconstruction.constraint_violation`` before its
    final ReLU. A nonpositive value is feasible and a positive value violates
    the corresponding upper/lower or thermal limit.
    """
    physics = completion.physics
    reference = physics.reference_generator
    pg = state["pg"][:, reference:reference + 1]
    pg_scale = (physics.pg_max[reference] - physics.pg_min[reference]).clamp_min(1e-12)
    qg = state["qg"]
    qg_scale = (physics.qg_max - physics.qg_min).clamp_min(1e-12)
    vm = state["vm"][:, physics.pq]
    vm_scale = (
        physics.vm_max[physics.pq] - physics.vm_min[physics.pq]
    ).clamp_min(1e-12)
    angle = state["va"][:, physics.fbus] - state["va"][:, physics.tbus]
    angle_scale = (physics.angle_max - physics.angle_min).clamp_min(1e-12)
    apparent_from = torch.sqrt(state["pf"].square() + state["qf"].square() + 1e-12)
    apparent_to = torch.sqrt(state["pt"].square() + state["qt"].square() + 1e-12)
    return {
        "constraint_pg": torch.cat([
            (pg - physics.pg_max[reference]) / pg_scale,
            (physics.pg_min[reference] - pg) / pg_scale,
        ], dim=1),
        "constraint_qg": torch.cat([
            (qg - physics.qg_max) / qg_scale,
            (physics.qg_min - qg) / qg_scale,
        ], dim=1),
        "constraint_voltage": torch.cat([
            (vm - physics.vm_max[physics.pq]) / vm_scale,
            (physics.vm_min[physics.pq] - vm) / vm_scale,
        ], dim=1),
        "constraint_angle": torch.cat([
            (angle - physics.angle_max) / angle_scale,
            (physics.angle_min - angle) / angle_scale,
        ], dim=1),
        "constraint_thermal": torch.cat([
            apparent_from / physics.rate - 1.0,
            apparent_to / physics.rate - 1.0,
        ], dim=1),
    }


def validation_predictions(
    data_dir: Path,
    completion: DifferentiableEqualityCompletion,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Predict the complete DECS validation split in bounded memory.

    Parameters
    ----------
    data_dir : pathlib.Path
        DECS dataset directory containing ``rho/chi/u/load`` arrays.
    completion : DifferentiableEqualityCompletion
        Trained DECS network and differentiable AC reconstruction.
    batch_size : int
        Validation samples evaluated per forward pass.
    device : torch.device
        CPU or CUDA inference device.

    Returns
    -------
    dict[str, np.ndarray]
        Target/predicted constraint variables in validation-index order.
    """
    rho = np.load(data_dir / "rho.npy", mmap_mode="r")
    chi = np.load(data_dir / "chi.npy", mmap_mode="r")
    u = np.load(data_dir / "u.npy", mmap_mode="r")
    load = np.load(data_dir / "load.npy", mmap_mode="r")
    indices = np.load(data_dir / "split" / "validation_indices.npy")

    target_variables = {name: [] for name in (
        "voltage", "angle_difference", "apparent_power",
        "constraint_pg", "constraint_qg", "constraint_voltage",
        "constraint_angle", "constraint_thermal",
    )}
    predicted_variables = {name: [] for name in target_variables}
    print(f"validation inference: {len(indices)} samples on {device}", flush=True)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            rho_batch = torch.tensor(np.asarray(rho[batch_indices]), device=device)
            chi_batch = torch.tensor(np.asarray(chi[batch_indices]), device=device)
            u_batch = torch.tensor(np.asarray(u[batch_indices]), device=device)
            load_batch = torch.tensor(np.asarray(load[batch_indices]), device=device)
            prediction = completion.predict_chi(rho_batch)
            target_state = completion.physics.reconstruct(u_batch, load_batch, chi_batch)
            predicted_state = completion.physics.reconstruct(u_batch, load_batch, prediction)
            target_batch = generator_state_variables(target_state, completion)
            predicted_batch = generator_state_variables(predicted_state, completion)
            target_batch.update(signed_constraint_residuals(target_state, completion))
            predicted_batch.update(signed_constraint_residuals(
                predicted_state, completion,
            ))
            for name in target_variables:
                target_variables[name].append(target_batch[name].cpu().numpy())
                predicted_variables[name].append(predicted_batch[name].cpu().numpy())
            completed = min(start + batch_size, len(indices))
            print(
                f"\rvalidation inference: {completed}/{len(indices)} "
                f"({100.0 * completed / len(indices):6.2f}%)",
                end="", flush=True,
            )
    print()
    result = {}
    for name in target_variables:
        result[f"{name}_target"] = np.concatenate(target_variables[name])
        result[f"{name}_prediction"] = np.concatenate(predicted_variables[name])
    return result


def resolve_base_data(data_dir: Path, requested: Path | None) -> Path:
    """Locate the SMPC base dataset that defined the DECS load range.

    Parameters
    ----------
    data_dir : pathlib.Path
        DECS dataset directory containing ``metadata.json``.
    requested : pathlib.Path or None
        Explicit base-data path.  If omitted, use ``source_base_data`` from
        DECS metadata, with a repository-local fallback after project moves.

    Returns
    -------
    pathlib.Path
        Existing base-data directory containing normalization parameters.
    """
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    source = Path(metadata["source_base_data"])
    candidates = [requested] if requested is not None else [
        source,
        ROOT / "Data_generation" / "data" / source.name,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"SMPC base dataset not found; checked: {candidates}")


def training_domain(
    data_dir: Path,
    base_data: Path,
    case: Case14,
) -> dict[str, np.ndarray | float]:
    """Recover the coordinate-wise sampling bounds used for DECS training.

    Parameters
    ----------
    data_dir : pathlib.Path
        DECS dataset directory containing generation metadata.
    base_data : pathlib.Path
        SMPC base dataset defining minimum and maximum load factors.
    case : Case14
        IEEE-14 network data defining nominal loads and free-variable bounds.

    Returns
    -------
    dict[str, np.ndarray | float]
        Load-factor bounds, physical free-variable bounds, and the extended
        free-variable bounds used for DECS training.
    """
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    factor_lower = np.full(len(case.load_buses), metadata["load_factor_lower"])
    factor_upper = np.full(len(case.load_buses), metadata["load_factor_upper"])
    free_lower = np.asarray(metadata["free_variable_lower"], dtype=float)
    free_upper = np.asarray(metadata["free_variable_upper"], dtype=float)
    train_free_lower = np.asarray(metadata["free_sampling_lower"], dtype=float)
    train_free_upper = np.asarray(metadata["free_sampling_upper"], dtype=float)
    return {
        "factor_lower": factor_lower,
        "factor_upper": factor_upper,
        "free_lower": free_lower,
        "free_upper": free_upper,
        "train_free_lower": train_free_lower,
        "train_free_upper": train_free_upper,
        "padding_fraction": float(metadata["free_range_extension"]),
    }


def draw_ood_operating_point(
    rng: np.random.Generator,
    case: Case14,
    domain: dict[str, np.ndarray | float],
    category: str,
    load_extension: float,
    free_extension: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one operating point outside specified DECS training bounds.

    Parameters
    ----------
    rng : numpy.random.Generator
        Reproducible random-number generator.
    case : Case14
        IEEE-14 nominal load data.
    domain : dict[str, np.ndarray | float]
        Bounds returned by :func:`training_domain`.
    category : {"load_ood", "free_ood", "joint_ood"}
        Coordinates forced outside their training ranges.
    load_extension : float
        OOD shell width divided by each load factor's training span.
    free_extension : float
        OOD shell width divided by each free variable's physical span.

    Returns
    -------
    load, u : tuple[np.ndarray, np.ndarray]
        Candidate net load, shape ``(11,2)``, and free variables, shape ``(6,)``.
    """
    factor_lower = np.asarray(domain["factor_lower"])
    factor_upper = np.asarray(domain["factor_upper"])
    free_lower = np.asarray(domain["free_lower"])
    free_upper = np.asarray(domain["free_upper"])
    train_free_lower = np.asarray(domain["train_free_lower"])
    train_free_upper = np.asarray(domain["train_free_upper"])

    factor = rng.uniform(factor_lower, factor_upper)
    u = rng.uniform(free_lower, free_upper)
    if category in {"load_ood", "joint_ood"}:
        dimension = rng.integers(len(factor))
        distance = rng.uniform(0.05, 1.0) * load_extension * (
            factor_upper[dimension] - factor_lower[dimension]
        )
        factor[dimension] = (
            factor_upper[dimension] + distance
            if rng.random() < 0.5
            else factor_lower[dimension] - distance
        )
    if category in {"free_ood", "joint_ood"}:
        dimension = rng.integers(len(u))
        distance = rng.uniform(0.05, 1.0) * free_extension * (
            free_upper[dimension] - free_lower[dimension]
        )
        u[dimension] = (
            train_free_upper[dimension] + distance
            if rng.random() < 0.5
            else train_free_lower[dimension] - distance
        )
    base_load = case.bus[case.load_buses, 2:4]
    return (base_load * factor[:, None]).astype(np.float32), u.astype(np.float32)


def pandapower_ood_predictions(
    completion: DifferentiableEqualityCompletion,
    device: torch.device,
    case: Case14,
    domain: dict[str, np.ndarray | float],
    n_samples: int,
    load_extension: float,
    free_extension: float,
    tolerance_mva: float,
    max_iterations: int,
    max_attempt_factor: float,
    progress_every: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate pandapower-labelled OOD samples and evaluate the DECS model.

    Parameters
    ----------
    completion : DifferentiableEqualityCompletion
        Trained DECS network and differentiable AC reconstruction.
    device : torch.device
        CPU or CUDA inference device.
    case : Case14
        IEEE-14 network data.
    domain : dict[str, np.ndarray | float]
        Training bounds returned by :func:`training_domain`.
    n_samples : int
        Number of converged samples retained in each OOD category.
    load_extension : float
        Load-OOD shell width relative to the training load-factor span.
    free_extension : float
        Free-OOD shell width relative to the physical free-variable span.
    tolerance_mva : float
        pandapower Newton-Raphson mismatch tolerance in MVA.
    max_iterations : int
        Maximum Newton-Raphson iterations per initialization.
    max_attempt_factor : float
        Maximum PF attempts divided by requested converged samples.
    progress_every : int
        Print progress after this many accepted samples.
    seed : int
        Reproducible OOD sampling seed.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Target/predicted constraint variables, sampled operating points, and
        strict OOD indicators for each category.
    """
    categories = ("load_ood", "free_ood", "joint_ood")
    rng = np.random.default_rng(seed)
    net = build_pandapower_network(case)
    results: dict[str, dict[str, np.ndarray]] = {}
    maximum_attempts = math.ceil(max_attempt_factor * n_samples)

    for category in categories:
        loads: list[np.ndarray] = []
        free_variables: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        attempts = failures = 0
        print(f"pandapower {category}: target {n_samples} converged OOD labels", flush=True)
        while len(labels) < n_samples and attempts < maximum_attempts:
            load, u = draw_ood_operating_point(
                rng, case, domain, category, load_extension, free_extension,
            )
            attempts += 1
            chi = solve_power_flow(
                net, case, load, u, tolerance_mva, max_iterations,
            )
            if chi is None or not np.isfinite(chi).all():
                failures += 1
                continue
            loads.append(load)
            free_variables.append(u)
            labels.append(chi.astype(np.float32))
            accepted = len(labels)
            if accepted % progress_every == 0 or accepted == n_samples:
                print(
                    f"  {accepted}/{n_samples} ({100.0 * accepted / n_samples:6.2f}%) | "
                    f"attempts {attempts} | failures {failures}",
                    flush=True,
                )
        if len(labels) < n_samples:
            raise RuntimeError(
                f"{category}: only {len(labels)} converged OOD samples after "
                f"{attempts} pandapower attempts"
            )

        load_array = np.stack(loads)
        u_array = np.stack(free_variables)
        target = np.stack(labels)
        factor = load_array[:, :, 0] / case.bus[case.load_buses, 2]
        outside_load = np.any(
            (factor < np.asarray(domain["factor_lower"]))
            | (factor > np.asarray(domain["factor_upper"])),
            axis=1,
        )
        outside_free = np.any(
            (u_array < np.asarray(domain["train_free_lower"]))
            | (u_array > np.asarray(domain["train_free_upper"])),
            axis=1,
        )
        expected_load = category in {"load_ood", "joint_ood"}
        expected_free = category in {"free_ood", "joint_ood"}
        if not np.all(outside_load == expected_load) or not np.all(outside_free == expected_free):
            raise RuntimeError(f"{category}: sampled points do not satisfy the OOD definition")

        u_tensor = torch.from_numpy(u_array).to(device)
        load_tensor = torch.from_numpy(load_array).to(device)
        target_tensor = torch.from_numpy(target).to(device)
        with torch.no_grad():
            rho = completion.physics.specification(u_tensor, load_tensor)
            prediction = completion.predict_chi(rho)
            prediction_state = completion.physics.reconstruct(
                u_tensor, load_tensor, prediction,
            )
            label_state = completion.physics.reconstruct(
                u_tensor, load_tensor, target_tensor,
            )
            target_variables = generator_state_variables(label_state, completion)
            predicted_variables = generator_state_variables(prediction_state, completion)
            target_variables.update(signed_constraint_residuals(
                label_state, completion,
            ))
            predicted_variables.update(signed_constraint_residuals(
                prediction_state, completion,
            ))
        results[category] = {
            "load": load_array,
            "u": u_array,
            "outside_load": outside_load,
            "outside_free": outside_free,
            **{
                f"{name}_target": value.cpu().numpy()
                for name, value in target_variables.items()
            },
            **{
                f"{name}_prediction": value.cpu().numpy()
                for name, value in predicted_variables.items()
            },
        }
    return results


def sample_pairs(
    target: np.ndarray,
    prediction: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample pairs while retaining target and prediction extrema.

    Parameters
    ----------
    target, prediction : np.ndarray
        Arrays with identical shape.
    max_points : int
        Maximum displayed scalar pairs.
    seed : int
        Reproducible subsampling seed.

    Returns
    -------
    target_sample, prediction_sample : tuple[np.ndarray, np.ndarray]
        Flattened arrays containing at most ``max_points`` entries. The four
        target/prediction minimum/maximum pairs are retained when subsampling.
    """
    target_flat = target.reshape(-1)
    prediction_flat = prediction.reshape(-1)
    if len(target_flat) <= max_points:
        return target_flat, prediction_flat
    extrema = np.unique([
        np.argmin(target_flat), np.argmax(target_flat),
        np.argmin(prediction_flat), np.argmax(prediction_flat),
    ])
    if max_points <= len(extrema):
        return target_flat[extrema[:max_points]], prediction_flat[extrema[:max_points]]
    candidates = np.setdiff1d(
        np.arange(len(target_flat)), extrema, assume_unique=True,
    )
    random_indices = np.random.default_rng(seed).choice(
        candidates, size=max_points - len(extrema), replace=False,
    )
    indices = np.concatenate([extrema, random_indices])
    return target_flat[indices], prediction_flat[indices]


def parity_plot(
    axis: plt.Axes,
    target: np.ndarray,
    prediction: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Draw a target-versus-prediction scatter plot with the identity line.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Destination subplot.
    target, prediction : np.ndarray
        One-dimensional values in identical physical units.
    xlabel, ylabel : str
        Axis labels including units.
    title : str
        Subplot title.
    """
    lower = min(float(target.min()), float(prediction.min()))
    upper = max(float(target.max()), float(prediction.max()))
    padding = max(0.04 * (upper - lower), 1e-6)
    limits = (lower - padding, upper + padding)
    axis.scatter(target, prediction, s=7, alpha=0.18, color="#0072B2", edgecolors="none")
    axis.plot(limits, limits, color="#D55E00", linewidth=1.5, linestyle="--")
    axis.set(xlim=limits, ylim=limits, xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)


def variable_metrics(
    result: dict[str, np.ndarray],
    completion: DifferentiableEqualityCompletion,
) -> dict:
    """Summarize generator-relevant constraint-variable prediction quality.

    Parameters
    ----------
    result : dict[str, np.ndarray]
        Target and predicted voltage, angle-difference, and apparent-power arrays.
    completion : DifferentiableEqualityCompletion
        Completion object supplying network partitions and inequality limits.

    Returns
    -------
    dict
        Physical-unit MAE, 95th-percentile error, and maximum error for each variable.
    """
    physics = completion.physics
    voltage_target = result["voltage_target"]
    voltage_prediction = result["voltage_prediction"]
    angle_target = result["angle_difference_target"]
    angle_prediction = result["angle_difference_prediction"]
    apparent_target = result["apparent_power_target"]
    apparent_prediction = result["apparent_power_prediction"]
    pq = physics.pq.detach().cpu().numpy()

    voltage_error = np.abs(voltage_prediction - voltage_target)
    angle_error_degree = np.rad2deg(np.abs(angle_prediction - angle_target))
    apparent_error = np.abs(apparent_prediction - apparent_target)
    metrics = {
        "n_samples": len(voltage_target),
        "voltage": {
            "mae_all_bus_pu": float(voltage_error.mean()),
            "mae_pq_bus_pu": float(voltage_error[:, pq].mean()),
            "p95_abs_error_pu": float(np.quantile(voltage_error, 0.95)),
            "max_abs_error_pu": float(voltage_error.max()),
        },
        "angle_difference": {
            "mae_degree": float(angle_error_degree.mean()),
            "p95_abs_error_degree": float(np.quantile(angle_error_degree, 0.95)),
            "max_abs_error_degree": float(angle_error_degree.max()),
        },
        "apparent_power": {
            "mae_mva": float(apparent_error.mean()),
            "p95_abs_error_mva": float(np.quantile(apparent_error, 0.95)),
            "max_abs_error_mva": float(apparent_error.max()),
        },
    }
    if "boundary" in result:
        metrics["n_padded_boundary"] = int(result["boundary"].sum())
    if "outside_load" in result:
        metrics["n_outside_load_range"] = int(result["outside_load"].sum())
        metrics["n_outside_free_range"] = int(result["outside_free"].sum())
    return metrics


def metric_bars(
    axis: plt.Axes,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
) -> None:
    """Draw one compact metric comparison bar chart.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Destination subplot.
    labels : list[str]
        IID/OOD category labels.
    values : list[float]
        Scalar metric values in label order.
    ylabel : str
        Vertical-axis label including physical units.
    title : str
        Subplot title.
    """
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    bars = axis.bar(labels, values, color=colors, alpha=0.85)
    axis.set(ylabel=ylabel, title=title)
    axis.tick_params(axis="x", labelrotation=18)
    axis.grid(alpha=0.25, axis="y", which="both")
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.2e}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7,
        )


CONSTRAINT_BLOCKS = (
    ("pg", r"$P_{\mathrm{ref}}$"),
    ("qg", r"$Q_g$"),
    ("voltage", r"$V$"),
    ("angle", r"$\Delta\theta$"),
    ("thermal", r"$|S|$"),
)

CONSTRAINT_COLORS = {
    "pg": "#D55E00",
    "qg": "#0072B2",
    "voltage": "#009E73",
    "angle": "#CC79A7",
    "thermal": "#E69F00",
}


def symmetric_log_ticks(
    maximum_absolute: float,
    linear_half_width: float,
) -> tuple[np.ndarray, list[str]]:
    """Return uncluttered signed decade ticks for a symmetric-log axis.

    Parameters
    ----------
    maximum_absolute : float
        Positive absolute plotting limit.
    linear_half_width : float
        Positive half-width of the central linear region.

    Returns
    -------
    ticks, labels : tuple[np.ndarray, list[str]]
        Symmetric ticks containing zero and decades no smaller than the linear
        region, with compact base-10 labels.
    """
    minimum_exponent = math.ceil(math.log10(linear_half_width))
    maximum_exponent = math.floor(math.log10(maximum_absolute))
    exponents = np.arange(
        minimum_exponent, max(minimum_exponent, maximum_exponent) + 1, 2,
    )
    if maximum_exponent > minimum_exponent and exponents[-1] != maximum_exponent:
        exponents = np.append(exponents, maximum_exponent)
    positive = np.power(10.0, exponents)
    ticks = np.concatenate([-positive[::-1], [0.0], positive])

    def label(exponent: int, sign: str = "") -> str:
        """Format one signed decade tick without verbose decimal zeros."""
        if exponent == 0:
            return f"{sign}1"
        return rf"${sign}10^{{{exponent}}}$"

    labels = (
        [label(int(exponent), "-") for exponent in exponents[::-1]]
        + ["0"]
        + [label(int(exponent)) for exponent in exponents]
    )
    return ticks, labels


def constraint_decision_metrics(
    result: dict[str, np.ndarray],
    decision_tolerance: float,
) -> dict[str, dict[str, float | int]]:
    """Summarize normalized residual error and inequality-decision mismatch."""
    metrics: dict[str, dict[str, float | int]] = {}
    for name, _ in CONSTRAINT_BLOCKS:
        target = np.asarray(result[f"constraint_{name}_target"], dtype=float)
        prediction = np.asarray(
            result[f"constraint_{name}_prediction"], dtype=float,
        )
        error = np.abs(prediction - target)
        target_violated = target > decision_tolerance
        predicted_violated = prediction > decision_tolerance
        metrics[name] = {
            "n_constraint_values": int(target.size),
            "mae_normalized_residual": float(error.mean()),
            "p95_abs_normalized_residual_error": float(np.quantile(error, 0.95)),
            "decision_agreement_percent": float(
                100.0 * np.mean(target_violated == predicted_violated)
            ),
            "false_feasible_percent": float(
                100.0 * np.mean(target_violated & ~predicted_violated)
            ),
            "false_infeasible_percent": float(
                100.0 * np.mean(~target_violated & predicted_violated)
            ),
            "target_violation_percent": float(100.0 * target_violated.mean()),
        }
    return metrics


def configure_publication_style() -> None:
    """Apply the compact IEEE single-column style used by the case study."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
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


def save_publication_figure(
    figure: plt.Figure,
    output: Path,
    dpi: int,
) -> None:
    """Export an IEEE figure as PNG, vector PDF, and LZW-compressed TIFF.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Fully composed figure to export.
    output : pathlib.Path
        Output path whose stem is shared by all three formats.
    dpi : int
        Raster resolution in dots per inch; 600 is the publication default.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=dpi)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(
        output.with_suffix(".tiff"), dpi=dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_constraint_validation(
    output: Path,
    result: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float | int]],
    residual_window: float,
    max_points: int,
    seed: int,
    dpi: int,
) -> None:
    """Save full-range residual parity and inequality-decision errors.

    Parameters
    ----------
    output : pathlib.Path
        Destination PNG path.
    result : dict[str, np.ndarray]
        Target and predicted signed residual arrays for all constraint blocks.
    metrics : dict[str, dict[str, float | int]]
        Constraint-decision metrics in the same block order as the plot.
    residual_window : float
        Half-width of the linear region around zero; values outside it remain
        visible on symmetric-log axes.
    max_points : int
        Maximum total scatter points, divided approximately equally among the
        five constraint blocks.
    seed : int
        Reproducible point-subsampling seed.
    dpi : int
        Saved PNG resolution in dots per inch.
    """
    configure_publication_style()
    n_blocks = len(CONSTRAINT_BLOCKS)
    base_budget, extra_points = divmod(max_points, n_blocks)
    sampled_parts = []
    lower = math.inf
    upper = -math.inf
    for index, (name, label) in enumerate(CONSTRAINT_BLOCKS):
        target = np.asarray(result[f"constraint_{name}_target"], dtype=float).ravel()
        prediction = np.asarray(
            result[f"constraint_{name}_prediction"], dtype=float,
        ).ravel()
        lower = min(lower, float(target.min()), float(prediction.min()))
        upper = max(upper, float(target.max()), float(prediction.max()))
        budget = base_budget + int(index < extra_points)
        target_sample, prediction_sample = sample_pairs(
            target, prediction, budget, seed + index,
        )
        sampled_parts.append((name, label, target_sample, prediction_sample))

    figure, (parity_axis, decision_axis) = plt.subplots(
        1, 2, figsize=(3.5, 2.15),
    )
    for name, label, target, prediction in sampled_parts:
        parity_axis.scatter(
            target, prediction, s=3, alpha=0.20,
            color=CONSTRAINT_COLORS[name], label=label, edgecolors="none",
            rasterized=True,
        )
    maximum_absolute = max(abs(lower), abs(upper), residual_window) * 1.08
    limits = (-maximum_absolute, maximum_absolute)
    parity_axis.plot(limits, limits, color="#333333", linestyle="--", linewidth=1.0)
    parity_axis.axhline(0.0, color="#777777", linestyle=":", linewidth=0.7)
    parity_axis.axvline(0.0, color="#777777", linestyle=":", linewidth=0.7)
    parity_axis.set_xscale(
        "symlog", linthresh=residual_window, linscale=1.0, base=10,
    )
    parity_axis.set_yscale(
        "symlog", linthresh=residual_window, linscale=1.0, base=10,
    )
    parity_axis.set(
        xlim=limits, ylim=limits,
        xlabel="Target residual", ylabel="DECS residual",
    )
    ticks, tick_labels = symmetric_log_ticks(maximum_absolute, residual_window)
    parity_axis.set_xticks(ticks, tick_labels)
    parity_axis.set_yticks(ticks, tick_labels)
    parity_axis.set_box_aspect(1.0)
    parity_axis.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.98),
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
        ncol=1, handlelength=1.0, handletextpad=0.25,
        labelspacing=0.05, borderpad=0.08,
        borderaxespad=0.0, markerscale=1.5,
    )
    parity_axis.text(
        0.5, -0.33, "(a) Residual parity",
        transform=parity_axis.transAxes,
        ha="center", va="top", fontsize=8.5,
    )

    positions = np.arange(len(CONSTRAINT_BLOCKS))
    false_feasible = np.asarray([
        metrics[name]["false_feasible_percent"] for name, _ in CONSTRAINT_BLOCKS
    ], dtype=float)
    false_infeasible = np.asarray([
        metrics[name]["false_infeasible_percent"] for name, _ in CONSTRAINT_BLOCKS
    ], dtype=float)
    decision_axis.plot(
        positions, false_feasible, color="#D55E00", marker="o", markersize=4.5,
        linewidth=1.1, label="False feasible",
    )
    decision_axis.plot(
        positions, false_infeasible, color="#0072B2", marker="s", markersize=4.2,
        linewidth=1.1, linestyle="--", label="False infeasible",
    )
    maximum = max(float(false_feasible.max()), float(false_infeasible.max()), 1e-3)
    decision_axis.set_ylim(-0.06 * maximum, 1.55 * maximum)
    decision_axis.set_xticks(
        positions, [label for _, label in CONSTRAINT_BLOCKS],
    )
    decision_axis.set_ylabel("Decision error (%)")
    decision_axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    decision_axis.legend(
        loc="upper center", frameon=False, ncol=1,
        handlelength=1.7, handletextpad=0.45,
    )
    decision_axis.set_box_aspect(1.0)
    decision_axis.text(
        0.5, -0.33, "(b) Decision errors", transform=decision_axis.transAxes,
        ha="center", va="top", fontsize=8.5,
    )

    for axis in (parity_axis, decision_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=2.5, width=0.65)
    figure.subplots_adjust(
        left=0.15, right=0.99, top=0.98, bottom=0.31, wspace=0.52,
    )
    save_publication_figure(figure, output, dpi)
    plt.close(figure)


def plot_validation_variables(
    output: Path,
    result: dict[str, np.ndarray],
    metrics: dict,
    completion: DifferentiableEqualityCompletion,
    max_points: int,
    seed: int,
    dpi: int,
) -> None:
    """Save validation parity plots for generator constraint variables.

    Parameters
    ----------
    output : pathlib.Path
        Destination PNG path.
    result : dict[str, np.ndarray]
        Validation constraint-variable predictions.
    metrics : dict
        Validation summary returned by :func:`variable_metrics`.
    completion : DifferentiableEqualityCompletion
        Completion object defining PQ-bus indices.
    max_points : int
        Maximum flattened values drawn in each parity plot.
    seed : int
        Reproducible scatter subsampling seed.
    dpi : int
        Saved PNG resolution in dots per inch.
    """
    pq = completion.physics.pq.detach().cpu().numpy()
    voltage = sample_pairs(
        result["voltage_target"][:, pq], result["voltage_prediction"][:, pq],
        max_points, seed,
    )
    angle = sample_pairs(
        np.rad2deg(result["angle_difference_target"]),
        np.rad2deg(result["angle_difference_prediction"]),
        max_points, seed + 1,
    )
    apparent = sample_pairs(
        result["apparent_power_target"], result["apparent_power_prediction"],
        max_points, seed + 2,
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    parity_plot(
        axes[0], *voltage,
        "pandapower PQ-bus voltage (p.u.)", "DECS PQ-bus voltage (p.u.)",
        f"Voltage (MAE={metrics['voltage']['mae_pq_bus_pu']:.2e} p.u.)",
    )
    parity_plot(
        axes[1], *angle,
        "pandapower branch angle difference (degree)",
        "DECS branch angle difference (degree)",
        f"Angle difference (MAE={metrics['angle_difference']['mae_degree']:.2e} deg)",
    )
    parity_plot(
        axes[2], *apparent,
        "pandapower branch apparent power (MVA)",
        "DECS branch apparent power (MVA)",
        f"Both-end apparent power (MAE={metrics['apparent_power']['mae_mva']:.2e} MVA)",
    )
    figure.suptitle(
        f"DECS validation of generator state variables (N={metrics['n_samples']})",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_ood_comparison(
    output: Path,
    metrics: dict[str, dict],
    dpi: int,
) -> None:
    """Save IID/OOD comparisons of generator constraint-variable accuracy.

    Parameters
    ----------
    output : pathlib.Path
        Destination PNG path.
    metrics : dict[str, dict]
        IID and OOD summaries returned by :func:`variable_metrics`.
    dpi : int
        Saved raster resolution in dots per inch.
    """
    keys = ["validation", "load_ood", "free_ood", "joint_ood"]
    labels = ["IID", "load OOD", "free OOD", "joint OOD"]
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    metric_bars(
        axes[0], labels,
        [float(metrics[key]["voltage"]["mae_pq_bus_pu"]) for key in keys],
        "PQ-bus voltage MAE (p.u.)", "Voltage magnitude",
    )
    metric_bars(
        axes[1], labels,
        [float(metrics[key]["angle_difference"]["mae_degree"]) for key in keys],
        "Branch angle-difference MAE (degree)", "Branch angle difference",
    )
    metric_bars(
        axes[2], labels,
        [float(metrics[key]["apparent_power"]["mae_mva"]) for key in keys],
        "Both-end apparent-power MAE (MVA)", "Branch apparent power",
    )
    figure.suptitle(
        "DECS generalization of generator state variables",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Evaluate DECS inequality decisions and save a compact paper figure."""
    n_sample = 10000
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / f"e2e14_decs_pgm_fixedpv_N{n_sample}",
        help="DECS dataset directory containing the validation split",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "Neural_network" / "decs_pgm_fixedpv.pt",
        help="trained DECS checkpoint containing weights and training history",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "decs_validation.png",
        help="output path; matching PDF and TIFF figures are also saved",
    )
    parser.add_argument(
        "--base-data", type=Path, default=None,
        help="SMPC base dataset; omitted means source_base_data in DECS metadata",
    )
    parser.add_argument(
        "--n-ood", type=int, default=0,
        help="optional pandapower samples per OOD category; zero disables OOD diagnostics",
    )
    parser.add_argument(
        "--ood-load-extension", type=float, default=0.1,
        help="OOD load-shell width divided by each training load-factor span",
    )
    parser.add_argument(
        "--ood-free-extension", type=float, default=0.05,
        help="OOD free-variable shell width divided by each physical span",
    )
    parser.add_argument(
        "--ood-tolerance-mva", type=float, default=1e-7,
        help="pandapower mismatch tolerance used to generate OOD labels",
    )
    parser.add_argument(
        "--ood-max-iterations", type=int, default=30,
        help="maximum pandapower Newton-Raphson iterations per OOD attempt",
    )
    parser.add_argument(
        "--ood-max-attempt-factor", type=float, default=3.0,
        help="maximum OOD PF attempts divided by requested converged samples",
    )
    parser.add_argument(
        "--ood-progress-every", type=int, default=50,
        help="print OOD generation progress after this many converged labels",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="validation samples evaluated per inference batch",
    )
    parser.add_argument(
        "--max-points", type=int, default=20_000,
        help="maximum flattened points in each parity scatter plot",
    )
    parser.add_argument(
        "--residual-window", type=float, default=0.3,
        help=(
            "half-width of the linear region around zero on the full-range "
            "symmetric-log residual axes"
        ),
    )
    parser.add_argument(
        "--decision-tolerance", type=float, default=1e-4,
        help="normalized residual tolerance used for feasible/infeasible decisions",
    )
    parser.add_argument(
        "--dpi", type=int, default=600,
        help="saved PNG/TIFF resolution in dots per inch",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="seed for OOD sampling and parity-plot point subsampling",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="inference device; auto selects CUDA when available",
    )
    args = parser.parse_args()

    if min(
        args.batch_size, args.max_points, args.dpi,
        args.ood_max_iterations, args.ood_progress_every,
    ) < 1:
        raise ValueError(
            "batch, plotting, iteration, and progress values must be positive"
        )
    if args.n_ood < 0:
        raise ValueError("n_ood must be nonnegative")
    if args.residual_window <= 0.0 or args.decision_tolerance < 0.0:
        raise ValueError("residual_window must be positive and decision_tolerance nonnegative")
    if min(args.ood_load_extension, args.ood_free_extension, args.ood_tolerance_mva) <= 0.0:
        raise ValueError("OOD extensions and pandapower tolerance must be positive")
    if args.ood_max_attempt_factor < 1.0:
        raise ValueError("ood_max_attempt_factor must be at least one")
    if not args.data.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("DECS dataset directory or checkpoint does not exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    completion = load_decs_checkpoint(args.checkpoint, device)
    result = validation_predictions(args.data, completion, args.batch_size, device)
    validation_summary = constraint_decision_metrics(
        result, args.decision_tolerance,
    )
    plot_constraint_validation(
        args.output, result, validation_summary, args.residual_window,
        args.max_points, args.seed, args.dpi,
    )

    all_metrics: dict[str, dict] = {"validation": validation_summary}
    protocol = {
        "residual_definition": "normalized signed residual; <= tolerance is feasible",
        "decision_tolerance": args.decision_tolerance,
        "residual_plot": {
            "scale": "symmetric_log",
            "linear_half_width": args.residual_window,
            "sampling": "equal point budget per constraint block with extrema retained",
            "maximum_points": args.max_points,
        },
        "evaluated_inequality_blocks": {
            "pg": "reference-generator active-power limits",
            "qg": "generator reactive-power limits",
            "voltage": "PQ-bus voltage-magnitude limits",
            "angle": "branch angle-difference limits",
            "thermal": "both-end branch apparent-power limits",
        },
        "seed": args.seed,
    }

    if args.n_ood > 0:
        print("generating optional OOD labels with pandapower...", flush=True)
        base_data = resolve_base_data(args.data, args.base_data)
        case = load_case14()
        domain = training_domain(args.data, base_data, case)
        ood_results = pandapower_ood_predictions(
            completion=completion,
            device=device,
            case=case,
            domain=domain,
            n_samples=args.n_ood,
            load_extension=args.ood_load_extension,
            free_extension=args.ood_free_extension,
            tolerance_mva=args.ood_tolerance_mva,
            max_iterations=args.ood_max_iterations,
            max_attempt_factor=args.ood_max_attempt_factor,
            progress_every=args.ood_progress_every,
            seed=args.seed + 2,
        )
        all_metrics.update({
            category: constraint_decision_metrics(
                category_result, args.decision_tolerance,
            )
            for category, category_result in ood_results.items()
        })
        protocol["ood"] = {
            "label_solver": "pandapower.runpp",
            "base_data": str(base_data.resolve()),
            "n_samples_per_category": args.n_ood,
            "load_extension_fraction": args.ood_load_extension,
            "free_extension_fraction": args.ood_free_extension,
            "pandapower_tolerance_mva": args.ood_tolerance_mva,
            "pandapower_max_iterations": args.ood_max_iterations,
            "seed": args.seed + 2,
            "training_padding_fraction": float(domain["padding_fraction"]),
            "definitions": {
                "load_ood": "one load factor outside its DECS training range",
                "free_ood": "one free variable outside the padded training range",
                "joint_ood": "both OOD conditions hold",
            },
        }

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps({"constraint_metrics": all_metrics, "protocol": protocol}, indent=2),
        encoding="utf-8",
    )
    print(f"saved validation figures to {args.output.with_suffix('.*')}")
    print(f"saved inequality-decision metrics to {metrics_path}")


if __name__ == "__main__":
    main()
