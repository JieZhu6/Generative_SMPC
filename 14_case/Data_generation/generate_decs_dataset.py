"""Generate fixed-PV IEEE-14 DECS labels with native PGM batches.

Every operating point is synthesized directly. Loads are sampled over the
base SMPC generator's complete all-time scale range, enlarged slightly on both
sides by ``--load-range-extension``. Free neural controls are sampled over
their physical projection bounds with a small symmetric extension controlled
by ``--free-range-extension``. PGM solves fixed-PV AC power flow in batches
and a small deterministic subset is cross-checked with pandapower.
"""

import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case14_pglib import Case14, load_case14  # noqa: E402
from Neural_network.decs import ACReconstruction  # noqa: E402
from pgm_batch_power_flow import build_pgm_case, solve_pv_batch  # noqa: E402


def build_pandapower_network(case: Case14):
    """Convert the local MATPOWER case to a reusable pandapower network."""
    from pandapower.converter.pypower.from_ppc import from_ppc

    return from_ppc({
        "version": "2",
        "baseMVA": case.base_mva,
        "bus": case.bus.copy(),
        "gen": case.gen.copy(),
        "branch": case.branch.copy(),
        "gencost": case.gencost.copy(),
    }, validate_conversion=False)


def assign_operating_point(net, case: Case14, load: np.ndarray, u: np.ndarray) -> None:
    """Assign ``load`` and ``[Pg_nonref,V_PV,V_ref]`` to pandapower."""
    pd = np.zeros(case.n_bus)
    qd = np.zeros(case.n_bus)
    pd[case.load_buses] = load[:, 0]
    qd[case.load_buses] = load[:, 1]
    load_rows = net.load.bus.to_numpy(dtype=int) - 1
    net.load.loc[:, "p_mw"] = pd[load_rows]
    net.load.loc[:, "q_mvar"] = qd[load_rows]

    pg = case.gen[:, 9].copy()
    pg[case.nonreference_active_generators] = u[:len(case.nonreference_active_generators)]
    pg_by_bus = {int(bus + 1): pg[g] for g, bus in enumerate(case.generator_buses)}
    vm_by_bus = {
        int(bus + 1): value
        for bus, value in zip(
            case.voltage_control_buses,
            u[len(case.nonreference_active_generators):],
        )
    }
    net.gen.loc[:, "p_mw"] = [pg_by_bus[int(bus)] for bus in net.gen.bus]
    net.gen.loc[:, "vm_pu"] = [vm_by_bus[int(bus)] for bus in net.gen.bus]
    net.ext_grid.loc[:, "vm_pu"] = vm_by_bus[case.reference_bus + 1]
    net.ext_grid.loc[:, "va_degree"] = 0.0


def solve_power_flow(
    net,
    case: Case14,
    load: np.ndarray,
    u: np.ndarray,
    tolerance_mva: float,
    max_iterations: int,
) -> np.ndarray | None:
    """Solve one fixed-PV pandapower PF and return the 22-state label."""
    import pandapower as pp

    assign_operating_point(net, case, load, u)
    try:
        pp.runpp(
            net,
            algorithm="nr",
            calculate_voltage_angles=True,
            init="flat",
            max_iteration=max_iterations,
            tolerance_mva=tolerance_mva,
            enforce_q_lims=False,
            voltage_depend_loads=False,
            check_connectivity=True,
            numba=True,
        )
    except Exception:
        return None
    if not net.converged:
        return None
    bus_ids = case.bus[:, 0].astype(int)
    vm = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float)
    va = np.deg2rad(net.res_bus.loc[bus_ids, "va_degree"].to_numpy(dtype=float))
    return np.r_[va[case.pv_buses], va[case.pq_buses], vm[case.pq_buses]]


def solve_power_flow_batch(
    pgm_case,
    case: Case14,
    load: np.ndarray,
    u: np.ndarray,
    error_tolerance: float,
    max_iterations: int,
    threading: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one native fixed-PV PGM batch and return 22-state labels."""
    states = solve_pv_batch(
        pgm_case,
        case,
        load,
        u,
        pgm_error_tolerance=error_tolerance,
        pgm_max_iterations=max_iterations,
        threading=threading,
        continue_on_batch_error=True,
    )
    va = np.asarray(states["va"])
    vm = np.asarray(states["vm"])
    chi = np.concatenate([
        va[:, case.pv_buses],
        va[:, case.pq_buses],
        vm[:, case.pq_buses],
    ], axis=1)
    converged = (
        np.asarray(states["batch_success"], dtype=bool)
        & np.asarray(states["pv_converged"], dtype=bool)
        & np.isfinite(chi).all(axis=1)
    )
    return chi, converged


def compare_labels_with_pandapower(
    case: Case14,
    load: np.ndarray,
    u: np.ndarray,
    chi: np.ndarray,
    point_count: int,
    tolerance_mva: float,
    max_iterations: int,
) -> dict:
    """Cross-check fixed-PV PGM labels against fixed-PV pandapower."""
    if point_count <= 0:
        return {"points_requested": 0, "points_converged": 0, "passed": None}
    indices = np.unique(np.linspace(0, len(load) - 1, point_count, dtype=int))
    net = build_pandapower_network(case)
    rows = []
    errors = []
    for index in indices:
        reference = solve_power_flow(
            net, case, load[index], u[index], tolerance_mva, max_iterations,
        )
        error = (
            float(np.max(np.abs(chi[index] - reference)))
            if reference is not None else None
        )
        if error is not None:
            errors.append(error)
        rows.append({
            "sample_index": int(index),
            "converged": reference is not None,
            "maximum_absolute_chi_error": error,
        })
    maximum = float(max(errors)) if errors else None
    acceptance = 1.0e-6
    return {
        "points_requested": int(len(indices)),
        "points_converged": len(errors),
        "maximum_absolute_chi_error": maximum,
        "acceptance_max_abs_chi": acceptance,
        "passed": bool(
            len(errors) == len(indices)
            and maximum is not None
            and maximum <= acceptance
        ),
        "points": rows,
    }


def format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_generation_progress(
    accepted: int,
    target: int,
    attempts: int,
    failures: int,
    start_time: float,
) -> None:
    """Print fixed-PV batch generation progress."""
    elapsed = perf_counter() - start_time
    rate = accepted / max(elapsed, 1e-12)
    eta = (target - accepted) / max(rate, 1e-12)
    print(
        f"[PF] {accepted:>{len(str(target))}}/{target} "
        f"({100.0 * accepted / target:6.2f}%) | attempts {attempts} | "
        f"failures {failures} | rate {rate:7.2f} samples/s | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
        flush=True,
    )


class OperatingPointSampler:
    """Generate loads and controls over slightly extended training ranges."""

    def __init__(
        self,
        base_metadata: dict,
        case: Case14,
        seed: int,
        load_range_extension: float,
        free_range_extension: float,
    ):
        """Initialize a scrambled Halton sampler from theoretical load limits."""
        scale_min = float(base_metadata["load_scale_min"])
        scale_max = float(base_metadata["load_scale_max"])
        span = scale_max - scale_min
        self.load_factor_lower = max(0.0, scale_min - load_range_extension * span)
        self.load_factor_upper = scale_max + load_range_extension * span
        self.case = case
        self.sequence = qmc.Halton(
            d=len(case.load_buses) + 6,
            scramble=True,
            seed=seed,
        )
        self.free_lower = np.r_[
            case.gen[case.nonreference_active_generators, 9],
            case.bus[case.voltage_control_buses, 12],
        ]
        self.free_upper = np.r_[
            case.gen[case.nonreference_active_generators, 8],
            case.bus[case.voltage_control_buses, 11],
        ]
        free_span = self.free_upper - self.free_lower
        self.free_sampling_lower = self.free_lower - free_range_extension * free_span
        self.free_sampling_upper = self.free_upper + free_range_extension * free_span

    def draw(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Return synthetic loads and free controls near physical boundaries."""
        unit = self.sequence.random(n_samples)
        n_load = len(self.case.load_buses)
        load_factor = (
            self.load_factor_lower
            + unit[:, :n_load] * (self.load_factor_upper - self.load_factor_lower)
        )
        base_load = self.case.bus[self.case.load_buses, 2:4]
        load = base_load[None, :, :] * load_factor[:, :, None]
        u = self.free_sampling_lower + unit[:, n_load:] * (
            self.free_sampling_upper - self.free_sampling_lower
        )
        return load.astype(np.float32), u.astype(np.float32)


def validate_labels(
    u: np.ndarray,
    load: np.ndarray,
    chi: np.ndarray,
    batch_size: int = 4096,
) -> float:
    """Return the largest normalized AC equality residual in the labels."""
    physics = ACReconstruction()
    maximum = 0.0
    with torch.no_grad():
        for start in range(0, len(u), batch_size):
            stop = min(start + batch_size, len(u))
            state = physics.reconstruct(
                torch.from_numpy(u[start:stop]),
                torch.from_numpy(load[start:stop]),
                torch.from_numpy(chi[start:stop]),
            )
            maximum = max(maximum, float(state["pf_residual"].max()))
    return maximum


def main() -> None:
    """Generate, validate, split, and save a fixed-PV DECS dataset."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    n_samples = 10000
    parser.add_argument(
        "--base-data", type=Path,
        default=ROOT / "Data_generation" / "data" / "e2e14_N5000_S20_T16",
        help="SMPC dataset whose metadata defines the complete load scale range",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "Data_generation" / "data" / f"e2e14_decs_pgm_fixedpv_N{n_samples}",
        help="new directory for fixed-PV DECS arrays and metadata",
    )
    parser.add_argument("--n-samples", type=int, default=n_samples)
    parser.add_argument(
        "--load-range-extension", type=float, default=0.02,
        help="range extension on each side divided by load_scale_max-load_scale_min",
    )
    parser.add_argument(
        "--free-range-extension", type=float, default=0.05,
        help=(
            "free-control extension on each side divided by its physical range; "
            "applies to nonreference Pg, PV voltage, and reference voltage"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="native PGM operating points per batch; directly adjustable",
    )
    parser.add_argument("--pgm-error-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--threading", type=int, default=0,
        help="PGM workers: negative is sequential, zero uses all hardware threads",
    )
    parser.add_argument(
        "--verify-points", type=int, default=30,
        help="saved labels cross-checked with fixed-PV pandapower; zero disables",
    )
    parser.add_argument("--pandapower-tolerance-mva", type=float, default=1e-8)
    parser.add_argument("--max-attempt-factor", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if min(args.n_samples, args.batch_size, args.max_iterations, args.progress_every) < 1:
        raise ValueError("sample count, batch size, iterations, and progress interval must be positive")
    if args.n_samples < 3 or args.verify_points < 0:
        raise ValueError("at least three samples are required and verify_points is nonnegative")
    if min(args.load_range_extension, args.free_range_extension) < 0.0:
        raise ValueError("load and free range extensions must be nonnegative")
    if (
        args.pgm_error_tolerance <= 0.0
        or args.pandapower_tolerance_mva <= 0.0
        or args.max_attempt_factor < 1.0
    ):
        raise ValueError("tolerances must be positive and max_attempt_factor at least one")
    if not args.base_data.is_dir():
        raise FileNotFoundError(f"base dataset not found: {args.base_data}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    case = load_case14()
    metadata_base = json.loads(
        (args.base_data / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata_base.get("case_name") != case.name:
        raise ValueError("base dataset and local IEEE-14 case do not match")
    sampler = OperatingPointSampler(
        metadata_base, case, args.seed,
        args.load_range_extension, args.free_range_extension,
    )

    print("Fixed-PV DECS dataset generation", flush=True)
    print(f"  base metadata: {args.base_data.resolve()}", flush=True)
    print(f"  output: {args.output.resolve()}", flush=True)
    print(f"  target samples: {args.n_samples}", flush=True)
    print(
        f"  all-time synthetic load factors: "
        f"[{sampler.load_factor_lower:.4f},{sampler.load_factor_upper:.4f}] "
        f"(extension={args.load_range_extension:.1%} of base span per side)",
        flush=True,
    )
    print(
        f"  free controls: physical bounds extended by "
        f"{args.free_range_extension:.1%} of each span per side",
        flush=True,
    )
    print("  bus model: fixed PV; Q limits are not enforced by the PF solver", flush=True)
    pgm_case = build_pgm_case(case)
    print(
        f"  power-grid-model {version('power-grid-model')}; "
        f"batch_size={args.batch_size}, threading={args.threading}",
        flush=True,
    )

    saved_load: list[np.ndarray] = []
    saved_u: list[np.ndarray] = []
    saved_chi: list[np.ndarray] = []
    accepted = attempts = failures = 0
    maximum_attempts = math.ceil(args.max_attempt_factor * args.n_samples)
    start_time = perf_counter()
    while accepted < args.n_samples and attempts < maximum_attempts:
        draw_count = min(
            args.batch_size,
            args.n_samples - accepted,
            maximum_attempts - attempts,
        )
        load_batch, u_batch = sampler.draw(draw_count)
        chi_batch, converged = solve_power_flow_batch(
            pgm_case, case, load_batch, u_batch,
            args.pgm_error_tolerance, args.max_iterations, args.threading,
        )
        attempts += draw_count
        failures += int((~converged).sum())
        if converged.any():
            saved_load.append(load_batch[converged])
            saved_u.append(u_batch[converged])
            saved_chi.append(chi_batch[converged].astype(np.float32))
            accepted += int(converged.sum())
        if accepted >= args.n_samples or accepted % args.progress_every < draw_count:
            print_generation_progress(
                accepted, args.n_samples, attempts, failures, start_time,
            )

    if accepted < args.n_samples:
        raise RuntimeError(
            f"only {accepted} converged samples after {attempts} attempts; "
            "increase --max-attempt-factor or reduce a range-extension parameter"
        )

    load = np.concatenate(saved_load)[:args.n_samples]
    u = np.concatenate(saved_u)[:args.n_samples]
    chi = np.concatenate(saved_chi)[:args.n_samples]
    physics = ACReconstruction()
    with torch.no_grad():
        rho = physics.specification(torch.from_numpy(u), torch.from_numpy(load)).numpy()
    maximum_residual = validate_labels(u, load, chi)
    if maximum_residual > 1e-5:
        raise RuntimeError(
            f"PGM labels disagree with local AC equations: {maximum_residual:.3e} p.u."
        )
    pandapower_report = compare_labels_with_pandapower(
        case, load, u, chi, args.verify_points,
        args.pandapower_tolerance_mva, args.max_iterations,
    )
    if args.verify_points and not pandapower_report["passed"]:
        raise RuntimeError(
            "fixed-PV PGM labels failed pandapower equivalence: "
            f"max error={pandapower_report['maximum_absolute_chi_error']}"
        )

    order = np.random.default_rng(args.seed + 1).permutation(args.n_samples)
    n_validation = max(1, int(0.1 * args.n_samples))
    n_test = max(1, int(0.1 * args.n_samples))
    n_train = args.n_samples - n_validation - n_test
    split = {
        "train": order[:n_train],
        "validation": order[n_train:n_train + n_validation],
        "test": order[n_train + n_validation:n_train + n_validation + n_test],
    }
    train = split["train"]
    rho_mean, rho_std = rho[train].mean(axis=0), rho[train].std(axis=0)
    chi_mean, chi_std = chi[train].mean(axis=0), chi[train].std(axis=0)
    rho_std = np.maximum(rho_std, 1e-8)
    chi_std = np.maximum(chi_std, 1e-8)

    args.output.mkdir(parents=True)
    np.save(args.output / "rho.npy", rho.astype(np.float32))
    np.save(args.output / "chi.npy", chi.astype(np.float32))
    np.save(args.output / "u.npy", u.astype(np.float32))
    np.save(args.output / "load.npy", load.astype(np.float32))
    split_dir = args.output / "split"
    split_dir.mkdir()
    for name, indices in split.items():
        np.save(split_dir / f"{name}_indices.npy", indices)
    np.savez(
        args.output / "normalization_parameters.npz",
        rho_mean=rho_mean.astype(np.float32),
        rho_std=rho_std.astype(np.float32),
        chi_mean=chi_mean.astype(np.float32),
        chi_std=chi_std.astype(np.float32),
    )
    metadata = {
        "dataset_type": "decs_power_flow",
        "case_name": case.name,
        "solver": "power_grid_model.native_batch_newton_raphson",
        "power_grid_model_version": version("power-grid-model"),
        "bus_type_model": "fixed_PV_PQ",
        "enforce_q_limits": False,
        "pv_to_pq_switching": False,
        "reactive_limits_role": "feasibility_check_only",
        "pandapower_equivalence_version": version("pandapower"),
        "pandapower_equivalence": pandapower_report,
        "n_samples": args.n_samples,
        "n_attempts": attempts,
        "n_failures": failures,
        "load_sampling": "synthetic_low_discrepancy_all_time_scale_range",
        "load_range_extension": args.load_range_extension,
        "load_factor_lower": sampler.load_factor_lower,
        "load_factor_upper": sampler.load_factor_upper,
        "free_variable_sampling": "symmetric_extension_of_physical_projection_bounds",
        "free_range_extension": args.free_range_extension,
        "pgm_error_tolerance": args.pgm_error_tolerance,
        "max_iterations": args.max_iterations,
        "threading": args.threading,
        "batch_size": args.batch_size,
        "batch_size_selection": "explicit_default_1000",
        "maximum_label_residual_pu": maximum_residual,
        "label_residual_acceptance_pu": 1.0e-5,
        "seed": args.seed,
        "rho_order": ["p_PV", "V_PV_target", "V_ref", "p_PQ", "q_PQ"],
        "chi_order": ["theta_PV", "theta_PQ", "V_PQ"],
        "chi_dim": int(chi.shape[1]),
        "pv_buses": (case.pv_buses + 1).tolist(),
        "pq_buses": (case.pq_buses + 1).tolist(),
        "reference_bus": case.reference_bus + 1,
        "free_variable_lower": sampler.free_lower.tolist(),
        "free_variable_upper": sampler.free_upper.tolist(),
        "free_sampling_lower": sampler.free_sampling_lower.tolist(),
        "free_sampling_upper": sampler.free_sampling_upper.tolist(),
        "source_base_data": str(args.base_data.resolve()),
        "source_base_usage": "all_time_load_range_only",
        "source_base_metadata": metadata_base,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(f"maximum label PF residual: {maximum_residual:.3e} p.u.")
    print(
        "pandapower max label difference: "
        f"{pandapower_report['maximum_absolute_chi_error']}"
    )
    print(f"total generation time: {format_duration(perf_counter() - start_time)}")
    print(f"saved fixed-PV DECS dataset to {args.output}")


if __name__ == "__main__":
    main()
