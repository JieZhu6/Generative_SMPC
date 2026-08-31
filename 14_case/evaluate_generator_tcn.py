"""Evaluate Generator-TCN with native batched PGM AC power flow.

Direct execution uses the native PGM batch evaluator so every generated
candidate is assessed under all scenario/time points. The pandapower helpers in
this module remain available only for independent numerical-equivalence checks
and legacy result reproduction.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


# The py3.10 environment cannot write Numba's package-local cache reliably.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case14_pglib import Case14, load_case14  # noqa: E402
from Data_generation.generate_decs_dataset import (  # noqa: E402
    assign_operating_point,
    build_pandapower_network,
)
from Data_generation.smpc_acopf_model import (  # noqa: E402
    branch_states,
    nodal_injections,
)
from Neural_network.noncausal_tcn import ConditionalStochasticTCN  # noqa: E402
from Neural_network.generator_benchmarks import (  # noqa: E402
    build_generator_model,
    get_benchmark_spec,
)
from Neural_network.decs import (  # noqa: E402
    DifferentiableEqualityCompletion,
    load_decs_checkpoint,
)


DEFAULT_DATA = ROOT / "Data_generation" / "data" / "e2e14_N5000_S20_T16"
DEFAULT_CHECKPOINT = ROOT / "Neural_network" / "generator_tcn_clip.pt"
DEFAULT_DECS = ROOT / "Neural_network" / "decs_pgm_fixedpv.pt"
DEFAULT_OUTPUT = ROOT / "output" / "ipopt_test"


def normalize(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map physical load values to dimensionless ``[-1, 1]`` coordinates.

    Parameters
    ----------
    value : np.ndarray
        Load values whose final dimensions are ``(n_load_buses, 2)``.
    lower, upper : np.ndarray, shape (n_load_buses, 2)
        Training-only extrema stored in the generator checkpoint.

    Returns
    -------
    np.ndarray
        Normalized values with the same shape as ``value``.
    """
    return 2.0 * (value - lower) / np.maximum(upper - lower, 1e-8) - 1.0


def build_condition(
    current_load: np.ndarray,
    future_pool: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
) -> np.ndarray:
    """Construct the paper's time-major min/mean/max NCTCN condition.

    Parameters
    ----------
    current_load : np.ndarray, shape (n_load_buses, 2)
        Deterministic first-period load in MW/Mvar.
    future_pool : np.ndarray, shape (3, T-1, n_load_buses, 2)
        Minimum, mean, and maximum future scenario loads.
    feature_min, feature_max : np.ndarray, shape (n_load_buses, 2)
        Checkpoint input-normalization extrema.

    Returns
    -------
    np.ndarray, shape (T, 3*n_load_buses*2)
        Temporal generator condition.
    """
    current = normalize(current_load, feature_min, feature_max)
    future = normalize(future_pool, feature_min, feature_max)
    current_pool = np.broadcast_to(current[None, None], (3, 1, *current.shape))
    pooled = np.concatenate([current_pool, future], axis=1)
    return pooled.transpose(1, 0, 2, 3).reshape(pooled.shape[1], -1).astype("float32")


def checkpoint_split_indices(
    checkpoint: dict,
    data_dir: Path,
    n_instances: int,
    split_name: str,
    max_instances: int | None = None,
) -> np.ndarray:
    """Load and verify one exact held-out partition used during training.

    Parameters
    ----------
    checkpoint : dict
        Loaded Generator-TCN checkpoint containing split metadata.
    data_dir : pathlib.Path
        Base dataset containing ``split/*_indices.npy`` and ``metadata.json``.
    n_instances : int
        Total number of base SMPC instances.
    split_name : {"validation", "test"}
        Held-out partition selected for model comparison.
    max_instances : int or None
        Optional positive prefix length for smoke tests.

    Returns
    -------
    np.ndarray
        Saved selected-partition indices, optionally truncated for a smoke test.
    """
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be 'validation' or 'test'")
    split = checkpoint.get("split", {})
    counts = split.get("counts", {})
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("checkpoint does not contain train/validation/test counts")
    if sum(int(value) for value in counts.values()) != n_instances:
        raise ValueError("checkpoint split counts do not match the base dataset")

    names = ("train", "validation", "test")
    saved = {
        name: np.load(data_dir / "split" / f"{name}_indices.npy")
        for name in names
    }
    for name, indices in saved.items():
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"saved {name} indices must be a one-dimensional integer array")
        if len(indices) != int(counts[name]):
            raise ValueError(f"saved {name} count differs from the checkpoint")
    combined = np.concatenate([saved[name] for name in names])
    if not np.array_equal(np.sort(combined), np.arange(n_instances)):
        raise ValueError("saved train/validation/test indices are not a disjoint partition")

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    checkpoint_metadata = checkpoint.get("data_metadata", {})
    identity_fields = (
        "dataset_type", "case_name", "n_instances", "n_scenarios", "horizon", "seed",
    )
    for field in identity_fields:
        if checkpoint_metadata.get(field) != metadata.get(field):
            raise ValueError(f"checkpoint and dataset differ in metadata field '{field}'")

    checkpoint_indices = split.get("indices")
    if checkpoint_indices is not None:
        for name in names:
            expected = np.asarray(checkpoint_indices[name], dtype=np.int64)
            if not np.array_equal(saved[name], expected):
                raise ValueError(f"saved {name} indices differ from the training checkpoint")
    else:
        # Compatibility path for old checkpoints: still require the saved files
        # to match the split seed instead of silently evaluating another subset.
        order = np.random.default_rng(int(split["seed"])).permutation(n_instances)
        expected = {
            "train": order[:int(counts["train"])],
            "validation": order[
                int(counts["train"]):int(counts["train"]) + int(counts["validation"])
            ],
            "test": order[int(counts["train"]) + int(counts["validation"]):],
        }
        if any(not np.array_equal(saved[name], expected[name]) for name in names):
            raise ValueError("saved split files differ from the checkpoint split seed")

    indices = saved[split_name]
    if max_instances is not None:
        if max_instances < 1:
            raise ValueError("max_instances must be positive")
        indices = indices[:max_instances]
    return indices


def checkpoint_test_indices(
    checkpoint: dict,
    data_dir: Path,
    n_instances: int,
    max_instances: int | None = None,
) -> np.ndarray:
    """Return the verified test indices for existing test-only consumers.

    Parameters
    ----------
    checkpoint : dict
        Loaded Generator-TCN checkpoint containing split metadata.
    data_dir : pathlib.Path
        Base dataset containing saved split files and metadata.
    n_instances : int
        Total number of base SMPC instances.
    max_instances : int or None
        Optional positive test-prefix length.

    Returns
    -------
    np.ndarray
        Verified test indices, optionally prefix-truncated.
    """
    return checkpoint_split_indices(
        checkpoint, data_dir, n_instances, "test", max_instances,
    )


def generation_cost(case: Case14, pg: np.ndarray) -> float:
    """Return one-period quadratic generation cost for ``pg`` in MW.

    Parameters
    ----------
    case : Case14
        PGLib generator cost coefficients.
    pg : np.ndarray, shape (n_gen,)
        Generator active powers in MW.

    Returns
    -------
    float
        Sum of all generator costs for one time period.
    """
    return float(np.sum(
        case.gencost[:, 4] * pg**2
        + case.gencost[:, 5] * pg
        + case.gencost[:, 6]
    ))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of checkpoint file ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_decs_candidates(
    feasible: np.ndarray,
    objective: np.ndarray,
    maximum_violation: np.ndarray,
) -> np.ndarray:
    """Order DECS candidates for sequential pandapower certification.

    Parameters
    ----------
    feasible : np.ndarray, shape (K,)
        DECS-predicted feasibility flags.
    objective : np.ndarray, shape (K,)
        DECS expected horizon generation costs.
    maximum_violation : np.ndarray, shape (K,)
        Maximum normalized DECS inequality violations.

    Returns
    -------
    np.ndarray, shape (K,)
        Candidate indices: predicted-feasible candidates by increasing cost,
        followed by false-negative fallbacks by violation then cost.
    """
    feasible = np.asarray(feasible, dtype=bool)
    objective = np.asarray(objective, dtype=float)
    maximum_violation = np.asarray(maximum_violation, dtype=float)
    if not (feasible.ndim == objective.ndim == maximum_violation.ndim == 1):
        raise ValueError("DECS ranking inputs must be one-dimensional")
    if not (len(feasible) == len(objective) == len(maximum_violation)):
        raise ValueError("DECS ranking inputs must have equal lengths")
    safe_objective = np.where(np.isfinite(objective), objective, np.inf)
    safe_violation = np.where(np.isfinite(maximum_violation), maximum_violation, np.inf)
    feasible_indices = np.flatnonzero(feasible)
    infeasible_indices = np.flatnonzero(~feasible)
    feasible_order = feasible_indices[np.argsort(
        safe_objective[feasible_indices], kind="stable",
    )]
    infeasible_order = infeasible_indices[np.lexsort((
        safe_objective[infeasible_indices], safe_violation[infeasible_indices],
    ))]
    return np.concatenate([feasible_order, infeasible_order])


def screen_candidates_with_decs(
    completion: DifferentiableEqualityCompletion,
    schedules: torch.Tensor,
    current_load: np.ndarray,
    future_load: np.ndarray,
    reference_ramp_mw: torch.Tensor,
    feasibility_tolerance: float,
) -> dict[str, np.ndarray | float]:
    """Batch-screen and rank one instance's candidates with frozen DECS.

    Parameters
    ----------
    completion : DifferentiableEqualityCompletion
        Frozen DECS network and exact algebraic AC reconstruction.
    schedules : torch.Tensor, shape (K,T,free_dim)
        Generator trajectories in MW and p.u. on the DECS device.
    current_load : np.ndarray, shape (n_load_buses,2)
        Deterministic first-period load in MW/Mvar.
    future_load : np.ndarray, shape (S,T-1,n_load_buses,2)
        Future scenario loads in MW/Mvar.
    reference_ramp_mw : scalar torch.Tensor
        Symmetric reference-generator ramp limit in MW per period.
    feasibility_tolerance : float
        Maximum normalized DECS inequality violation used only for prescreening.

    Returns
    -------
    dict
        DECS feasibility, expected cost, maximum violation, PF residual, and a
        candidate order with feasible candidates first in increasing cost.
    """
    candidates, horizon, free_dim = schedules.shape
    scenarios, future_steps = future_load.shape[:2]
    if horizon != future_steps + 1 or free_dim != completion.physics.free_dim:
        raise ValueError("DECS schedule and scenario dimensions do not match")
    device, dtype = schedules.device, schedules.dtype
    current = torch.as_tensor(
        np.array(current_load, copy=True), dtype=dtype, device=device,
    )
    future = torch.as_tensor(
        np.array(future_load, copy=True), dtype=dtype, device=device,
    )
    scenario_load = torch.cat([
        current[None, None].expand(scenarios, 1, -1, -1), future,
    ], dim=1)
    candidate_scenarios = schedules[:, None].expand(-1, scenarios, -1, -1)
    repeated_load = scenario_load[None].expand(candidates, -1, -1, -1, -1)
    state = completion(
        candidate_scenarios.reshape(-1, free_dim),
        repeated_load.reshape(-1, *scenario_load.shape[2:]),
    )

    physical = completion.physics.constraint_violation(state).reshape(
        candidates, scenarios, horizon, -1,
    )
    reference = completion.physics.reference_generator
    pg = state["pg"].reshape(
        candidates, scenarios, horizon, completion.physics.n_gen,
    )
    delta_reference = pg[:, :, 1:, reference] - pg[:, :, :-1, reference]
    ramp = reference_ramp_mw.clamp_min(1.0e-8)
    ramp_violation = torch.maximum(
        torch.relu((delta_reference - ramp) / ramp),
        torch.relu((-delta_reference - ramp) / ramp),
    ).flatten(start_dim=1).amax(dim=1)
    maximum_violation = torch.maximum(
        physical.flatten(start_dim=1).amax(dim=1), ramp_violation,
    )
    objective = completion.physics.generation_cost(pg).sum(dim=2).mean(dim=1)
    pf_residual = state["pf_residual"].reshape(
        candidates, scenarios, horizon,
    ).amax(dim=(1, 2))
    finite = (
        torch.isfinite(objective)
        & torch.isfinite(maximum_violation)
        & torch.isfinite(pf_residual)
    )
    feasible = finite & (maximum_violation <= feasibility_tolerance)

    objective_np = objective.detach().cpu().numpy()
    violation_np = maximum_violation.detach().cpu().numpy()
    feasible_np = feasible.detach().cpu().numpy()
    return {
        "feasible": feasible_np,
        "objective": objective_np,
        "maximum_violation": violation_np,
        "pf_residual": pf_residual.detach().cpu().numpy(),
        "order": rank_decs_candidates(feasible_np, objective_np, violation_np),
    }


def solve_exact_power_flow(
    net,
    case: Case14,
    load: np.ndarray,
    u: np.ndarray,
    warm_start: bool,
    tolerance_mva: float,
    max_iterations: int,
) -> tuple[dict[str, np.ndarray | float] | None, int]:
    """Solve one operating point with pandapower Newton-Raphson.

    Parameters
    ----------
    net : pandapowerNet
        Reusable IEEE-14 pandapower network.
    case : Case14
        Network topology, limits, and index partitions.
    load : np.ndarray, shape (n_load_buses, 2)
        Net P/Q loads in MW/Mvar.
    u : np.ndarray, shape (free_dim,)
        Shared ``[Pg_nonref, V_PV, V_ref]`` control for this time period.
    warm_start : bool
        Try the previous converged voltage before a flat start.
    tolerance_mva : float
        Positive pandapower nodal mismatch tolerance in MVA.
    max_iterations : int
        Positive Newton iteration limit per initialization.

    Returns
    -------
    state : dict or None
        Exact PF state, generator powers, branch flows, and balance residual;
        ``None`` if all initializations fail.
    attempts : int
        Number of pandapower calls made.
    """
    import pandapower as pp

    assign_operating_point(net, case, load, u)
    attempts = 0
    for initialization in (("results", "flat") if warm_start else ("flat",)):
        attempts += 1
        try:
            pp.runpp(
                net,
                algorithm="nr",
                calculate_voltage_angles=True,
                init=initialization,
                max_iteration=max_iterations,
                tolerance_mva=tolerance_mva,
                enforce_q_lims=False,
                voltage_depend_loads=False,
                check_connectivity=True,
                numba=True,
            )
        except Exception:
            continue
        if not net.converged:
            continue

        bus_ids = case.bus[:, 0].astype(int)
        vm = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float)
        va = np.deg2rad(net.res_bus.loc[bus_ids, "va_degree"].to_numpy(dtype=float))
        pd = np.zeros(case.n_bus)
        qd = np.zeros(case.n_bus)
        pd[case.load_buses] = load[:, 0]
        qd[case.load_buses] = load[:, 1]
        pnet, qnet = nodal_injections(case, vm, va)

        pg = case.gen[:, 9].copy()
        pg[case.nonreference_active_generators] = u[:len(case.nonreference_active_generators)]
        pg[case.reference_generator] = pd[case.reference_bus] + pnet[case.reference_bus]
        qg = qd[case.generator_buses] + qnet[case.generator_buses]
        state = branch_states(case, vm, va)
        state.update({
            "pg": pg,
            "qg": qg,
            "vm": vm,
            "va": va,
        })
        return state, attempts
    return None, attempts


def physical_point_excesses(
    case: Case14,
    state: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Compute direct physical-limit excesses for one pandapower state.

    Parameters
    ----------
    case : Case14
        PGLib equipment and network limits.
    state : dict
        Converged state returned by :func:`solve_exact_power_flow`.

    Returns
    -------
    dict[str, float]
        Maximum positive excesses in MW, Mvar, p.u., rad, and MVA.  A zero
        value means the corresponding physical inequality is satisfied.
    """
    pg = np.asarray(state["pg"])
    pg_min = case.gen[:, 9]
    pg_max = case.gen[:, 8]
    pg_excess = np.maximum.reduce([
        np.zeros_like(pg), pg - pg_max, pg_min - pg,
    ])

    qg = np.asarray(state["qg"])
    qg_min, qg_max = case.gen[:, 4], case.gen[:, 3]
    qg_excess = np.maximum.reduce([
        np.zeros_like(qg), qg - qg_max, qg_min - qg,
    ])

    vm = np.asarray(state["vm"])
    vm_min, vm_max = case.bus[:, 12], case.bus[:, 11]
    voltage_excess = np.maximum.reduce([
        np.zeros_like(vm), vm - vm_max, vm_min - vm,
    ])

    fbus = case.branch[:, 0].astype(int) - 1
    tbus = case.branch[:, 1].astype(int) - 1
    angle = np.asarray(state["va"])[fbus] - np.asarray(state["va"])[tbus]
    angle_min = np.deg2rad(case.branch[:, 11])
    angle_max = np.deg2rad(case.branch[:, 12])
    angle_excess = np.maximum.reduce([
        np.zeros_like(angle),
        angle - angle_max,
        angle_min - angle,
    ])

    apparent_from = np.hypot(state["pf"], state["qf"])
    apparent_to = np.hypot(state["pt"], state["qt"])
    thermal_excess = np.maximum(
        np.maximum(apparent_from, apparent_to) - case.branch[:, 5],
        0.0,
    )
    return {
        "pg": float(np.max(pg_excess)),
        "qg": float(np.max(qg_excess)),
        "voltage": float(np.max(voltage_excess)),
        "angle": float(np.max(angle_excess)),
        "thermal": float(np.max(thermal_excess)),
    }


def physical_limits_satisfied(
    excesses: dict[str, float],
    power_tolerance_mva: float,
    voltage_tolerance_pu: float,
    angle_tolerance_rad: float,
) -> bool:
    """Return whether direct physical excesses are within numerical tolerances.

    Parameters
    ----------
    excesses : dict[str,float]
        Output of :func:`physical_point_excesses`.
    power_tolerance_mva : float
        Numerical tolerance for Pg, Qg, and thermal limits in MW/Mvar/MVA.
    voltage_tolerance_pu : float
        Numerical tolerance for voltage bounds in p.u.
    angle_tolerance_rad : float
        Numerical tolerance for branch-angle bounds in rad.
    """
    return bool(
        excesses["pg"] <= power_tolerance_mva
        and excesses["qg"] <= power_tolerance_mva
        and excesses["thermal"] <= power_tolerance_mva
        and excesses["voltage"] <= voltage_tolerance_pu
        and excesses["angle"] <= angle_tolerance_rad
    )


def validate_shared_schedule(
    net,
    case: Case14,
    current_load: np.ndarray,
    future_load: np.ndarray,
    schedule: np.ndarray,
    ramp_fraction: float = 0.25,
    power_limit_tolerance_mva: float = 1e-5,
    voltage_limit_tolerance_pu: float = 1e-7,
    angle_limit_tolerance_rad: float = 1e-7,
    pf_tolerance_mva: float = 1e-6,
    max_pf_iterations: int = 30,
    early_stop: bool = True,
    save_states: bool = False,
) -> dict:
    """Validate one scenario-shared schedule using only pandapower PF results.

    Parameters
    ----------
    net : pandapowerNet
        Reusable network modified in place for each operating point.
    case : Case14
        IEEE-14 topology, costs, and operating limits.
    current_load : np.ndarray, shape (n_load_buses, 2)
        Deterministic t=1 load in MW/Mvar.
    future_load : np.ndarray, shape (S, T-1, n_load_buses, 2)
        Future scenario loads in MW/Mvar.
    schedule : np.ndarray, shape (T, free_dim)
        Shared Generator/IPOPT free-variable trajectory.
    ramp_fraction : float, default=0.25
        Symmetric one-period ramp limit as a fraction of Pmax.
    power_limit_tolerance_mva : float, default=1e-5
        Numerical tolerance for Pg/Qg/ramp/thermal physical limits.
    voltage_limit_tolerance_pu : float, default=1e-7
        Numerical tolerance for voltage-magnitude physical limits.
    angle_limit_tolerance_rad : float, default=1e-7
        Numerical tolerance for branch-angle physical limits.
    pf_tolerance_mva : float, default=1e-6
        Pandapower Newton mismatch tolerance in MVA.
    max_pf_iterations : int, default=30
        Newton iterations allowed per warm/flat attempt.
    early_stop : bool, default=True
        Stop at the first failed operating point or violated constraint.
    save_states : bool, default=False
        Return full Pg/Qg/V/angle arrays when every power flow converges,
        including physically infeasible schedules.

    Returns
    -------
    dict
        Exact feasibility, expected horizon cost whenever all power flows
        converge, violations, PF counts, and optional scenario-state arrays.
    """
    scenarios, future_steps = future_load.shape[:2]
    horizon = future_steps + 1
    if schedule.shape != (horizon, 1 + len(case.voltage_control_buses)):
        raise ValueError("schedule must have shape (T, free_dim)")
    if (
        ramp_fraction <= 0 or power_limit_tolerance_mva < 0
        or voltage_limit_tolerance_pu < 0 or angle_limit_tolerance_rad < 0
        or pf_tolerance_mva <= 0
    ):
        raise ValueError("invalid ramp fraction or physical-limit tolerance")

    maxima = {
        "pg": 0.0, "qg": 0.0, "voltage": 0.0,
        "angle": 0.0, "thermal": 0.0, "ramp": 0.0,
    }
    pf_attempts = 0
    pf_points = 0
    failed = False
    power_flow_complete = True
    state_shape = (scenarios, horizon)
    states = {
        "pg": np.full(state_shape + (case.n_gen,), np.nan),
        "qg": np.full(state_shape + (case.n_gen,), np.nan),
        "vm": np.full(state_shape + (case.n_bus,), np.nan),
        "va": np.full(state_shape + (case.n_bus,), np.nan),
    }

    first_state, attempts = solve_exact_power_flow(
        net, case, current_load, schedule[0], False,
        pf_tolerance_mva, max_pf_iterations,
    )
    pf_attempts += attempts
    if first_state is None:
        failed = True
        power_flow_complete = False
    else:
        pf_points += 1
        first_violations = physical_point_excesses(case, first_state)
        for name, value in first_violations.items():
            maxima[name] = max(maxima[name], value)
        for name in states:
            states[name][:, 0] = first_state[name]
        if not physical_limits_satisfied(
            first_violations,
            power_limit_tolerance_mva,
            voltage_limit_tolerance_pu,
            angle_limit_tolerance_rad,
        ):
            failed = True

    first_cost = generation_cost(case, np.asarray(first_state["pg"])) if first_state else np.nan
    future_costs = np.zeros(scenarios)
    ramp_limit = ramp_fraction * case.gen[case.active_generators, 8]
    if not failed or not early_stop:
        for scenario in range(scenarios):
            previous_pg = None if first_state is None else np.asarray(first_state["pg"])
            for time in range(1, horizon):
                state, attempts = solve_exact_power_flow(
                    net, case, future_load[scenario, time - 1], schedule[time],
                    warm_start=True, tolerance_mva=pf_tolerance_mva,
                    max_iterations=max_pf_iterations,
                )
                pf_attempts += attempts
                if state is None:
                    failed = True
                    power_flow_complete = False
                    break
                pf_points += 1
                for name in states:
                    states[name][scenario, time] = state[name]
                violations = physical_point_excesses(case, state)
                for name, value in violations.items():
                    maxima[name] = max(maxima[name], value)
                if previous_pg is not None:
                    delta = np.abs(
                        np.asarray(state["pg"])[case.active_generators]
                        - previous_pg[case.active_generators]
                    )
                    ramp_violation = np.maximum(delta - ramp_limit, 0.0)
                    maxima["ramp"] = max(maxima["ramp"], float(np.max(ramp_violation)))
                previous_pg = np.asarray(state["pg"])
                future_costs[scenario] += generation_cost(case, previous_pg)
                point_failed = (
                    not physical_limits_satisfied(
                        violations,
                        power_limit_tolerance_mva,
                        voltage_limit_tolerance_pu,
                        angle_limit_tolerance_rad,
                    )
                    or maxima["ramp"] > power_limit_tolerance_mva
                )
                if point_failed:
                    failed = True
                    if early_stop:
                        break
            if failed and early_stop:
                break

    feasible = bool(
        not failed
        and maxima["pg"] <= power_limit_tolerance_mva
        and maxima["qg"] <= power_limit_tolerance_mva
        and maxima["thermal"] <= power_limit_tolerance_mva
        and maxima["ramp"] <= power_limit_tolerance_mva
        and maxima["voltage"] <= voltage_limit_tolerance_pu
        and maxima["angle"] <= angle_limit_tolerance_rad
    )
    expected_pf_points = 1 + scenarios * (horizon - 1)
    power_flow_complete = bool(
        power_flow_complete and pf_points == expected_pf_points
    )
    objective = (
        float(first_cost + future_costs.mean())
        if power_flow_complete else np.nan
    )
    result = {
        "feasible": feasible,
        "objective": objective,
        "cost_available": bool(np.isfinite(objective)),
        "power_flow_complete": power_flow_complete,
        "pf_attempts": pf_attempts,
        "pf_points": pf_points,
        "max_pg_excess_mw": maxima["pg"],
        "max_qg_excess_mvar": maxima["qg"],
        "max_voltage_excess_pu": maxima["voltage"],
        "max_angle_excess_rad": maxima["angle"],
        "max_thermal_excess_mva": maxima["thermal"],
        "max_ramp_excess_mw": maxima["ramp"],
    }
    if save_states and power_flow_complete:
        result["states"] = states
    return result


def load_generator(
    checkpoint_path: Path,
    device: torch.device,
    benchmark: str = "csng",
) -> tuple[ConditionalStochasticTCN, dict]:
    """Load a Generator-TCN checkpoint on ``device`` for inference.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Checkpoint written by ``Neural_network/train_generator.py``.
    device : torch.device
        CPU or CUDA inference device.
    benchmark : str, default="csng"
        Expected benchmark identity used to reject mixed checkpoints.

    Returns
    -------
    model, checkpoint : tuple
        Evaluation-mode model and full checkpoint dictionary.
    """
    spec = get_benchmark_spec(benchmark)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_benchmark = checkpoint.get("benchmark", "csng")
    if checkpoint_benchmark != benchmark:
        raise ValueError(
            f"checkpoint benchmark is {checkpoint_benchmark!r}, expected {benchmark!r}"
        )
    if checkpoint.get("model_class") != spec["model_class"]:
        raise ValueError(
            f"checkpoint model_class is {checkpoint.get('model_class')!r}, "
            f"expected {spec['model_class']!r}"
        )
    if (
        checkpoint.get("model_config", {}).get("projection_method")
        != spec["projection_method"]
    ):
        raise ValueError(
            "checkpoint projection does not match the selected benchmark: "
            f"expected {spec['projection_method']!r}"
        )
    model = build_generator_model(benchmark, **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def finite_statistics(values: list[float]) -> dict[str, float | None]:
    """Return mean, median, and 95th percentile of finite ``values``."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def summarize_solution_costs(rows: list[dict]) -> dict[str, object]:
    """Summarize feasible-only and feasibility-agnostic output costs.

    Parameters
    ----------
    rows : list[dict]
        Per-instance results containing ``feasible`` and ``all_cost``.

    Returns
    -------
    dict[str, object]
        Feasible-only statistics, all finite-cost statistics, and the number
        of instances whose complete pandapower trajectory produced a cost.
    """
    all_costs = [float(row["all_cost"]) for row in rows]
    feasible_costs = [
        float(row["all_cost"]) for row in rows if bool(row["feasible"])
    ]
    available = int(np.isfinite(all_costs).sum())
    return {
        "feasible_solution_cost": finite_statistics(feasible_costs),
        "all_solution_cost": finite_statistics(all_costs),
        "cost_available_instances": available,
        "cost_unavailable_instances": len(rows) - available,
    }


def format_reported_cost(value: float) -> str:
    """Format a finite cost and never emit the literal text ``nan``."""
    return f"{float(value):.3f}" if np.isfinite(value) else "unavailable"


def write_json(path: Path, value: dict) -> None:
    """Write ``value`` as readable UTF-8 JSON to ``path``."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Rewrite accumulated ``rows`` to CSV so long runs preserve progress.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    rows : list[dict]
        Nonempty homogeneous row dictionaries.
    """
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_pandapower_reference() -> None:
    """Run the retained DECS-plus-pandapower reference evaluation."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="base SMPC dataset")
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help=(
            "saved held-out data partition evaluated by Generator-TCN; "
            "test is the final frozen-model benchmark"
        ),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="trained Generator-TCN checkpoint",
    )
    parser.add_argument(
        "--decs", type=Path, default=DEFAULT_DECS,
        help="frozen DECS checkpoint used to train and batch-screen the Generator",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "result directory; default is output/smpc_benchmark for test and "
            "output/smpc_validation for validation"
        ),
    )
    parser.add_argument(
        "--candidates", type=int, default=100,
        help="trajectory-level candidates generated for every selected scenario tree",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="optional positive selected-split prefix; default evaluates all 500",
    )
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="one-period ramp limit as a fraction of generator Pmax",
    )
    parser.add_argument(
        "--decs-feasibility-tolerance", "--feasibility-tolerance",
        dest="decs_feasibility_tolerance", type=float, default=1e-4,
        help=(
            "maximum normalized DECS inequality violation for prescreening only; "
            "pandapower feasibility uses direct physical limits"
        ),
    )
    parser.add_argument(
        "--power-limit-tolerance-mva", type=float, default=1e-5,
        help="numerical tolerance for pandapower Pg/Qg/ramp/thermal limits",
    )
    parser.add_argument(
        "--voltage-limit-tolerance-pu", type=float, default=1e-5,
        help="numerical tolerance for pandapower voltage-magnitude limits",
    )
    parser.add_argument(
        "--angle-limit-tolerance-rad", type=float, default=1e-5,
        help="numerical tolerance for pandapower branch-angle limits",
    )
    parser.add_argument(
        "--pf-tolerance-mva", type=float, default=1e-5,
        help="pandapower Newton mismatch tolerance in MVA",
    )
    parser.add_argument(
        "--max-pf-iterations", type=int, default=50,
        help="Newton iterations per warm or flat pandapower attempt",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="base latent seed; each instance uses seed plus its base-data index",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="TCN device; auto selects CUDA when available",
    )
    args = parser.parse_args()
    if args.candidates < 1 or args.max_pf_iterations < 1:
        raise ValueError("candidates and max_pf_iterations must be positive")
    if (
        args.ramp_fraction <= 0 or args.decs_feasibility_tolerance < 0
        or args.power_limit_tolerance_mva < 0
        or args.voltage_limit_tolerance_pu < 0
        or args.angle_limit_tolerance_rad < 0
        or args.pf_tolerance_mva <= 0
    ):
        raise ValueError("invalid ramp fraction or feasibility tolerance")
    if not args.data.is_dir() or not args.checkpoint.is_file() or not args.decs.is_file():
        raise FileNotFoundError("base dataset, Generator checkpoint, or DECS checkpoint is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OUTPUT if args.split == "test"
            else DEFAULT_OUTPUT.parent / "ipopt_validation"
        )

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model_load_start = perf_counter()
    model, checkpoint = load_generator(args.checkpoint, device)
    generator_model_load_seconds = perf_counter() - model_load_start
    expected_decs_sha256 = checkpoint.get("decs_checkpoint_sha256")
    if expected_decs_sha256 and file_sha256(args.decs) != expected_decs_sha256:
        raise ValueError("DECS checkpoint differs from the model used for Generator training")
    decs_load_start = perf_counter()
    completion = load_decs_checkpoint(args.decs, device)
    decs_model_load_seconds = perf_counter() - decs_load_start
    normalization = checkpoint["normalization"]
    feature_min = np.asarray(normalization["feature_min"], dtype=float)
    feature_max = np.asarray(normalization["feature_max"], dtype=float)
    free_lower = torch.as_tensor(checkpoint["free_variable_lower"], device=device)
    free_upper = torch.as_tensor(checkpoint["free_variable_upper"], device=device)
    free_ramp = torch.as_tensor(checkpoint["free_ramp_mw_per_period"], device=device)
    reference_ramp = (
        args.ramp_fraction
        * completion.physics.pg_max[completion.physics.reference_generator]
    )

    current = np.load(args.data / "current_load.npy", mmap_mode="r")
    pool = np.load(args.data / "future_pool.npy", mmap_mode="r")
    future = np.load(args.data / "future_load.npy", mmap_mode="r")
    instance_indices = checkpoint_split_indices(
        checkpoint, args.data, len(current), args.split, args.max_instances,
    )
    case = load_case14()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / f"{args.split}_indices.npy"
    if index_path.is_file():
        previous = np.load(index_path)
        if not np.array_equal(previous[:len(instance_indices)], instance_indices):
            raise ValueError(f"existing and current {args.split} indices differ")
    np.save(index_path, instance_indices)

    build_start = perf_counter()
    net = build_pandapower_network(case)
    setup_seconds = perf_counter() - build_start
    n_selected = len(instance_indices)
    horizon, free_dim = 1 + future.shape[2], checkpoint["model_config"]["free_dim"]
    selected_u = np.full((n_selected, horizon, free_dim), np.nan, dtype="float32")
    selected_pg = np.full((n_selected, future.shape[1], horizon, case.n_gen), np.nan, dtype="float32")
    selected_qg = np.full_like(selected_pg, np.nan)
    selected_vm = np.full((n_selected, future.shape[1], horizon, case.n_bus), np.nan, dtype="float32")
    selected_va = np.full_like(selected_vm, np.nan)
    rows: list[dict] = []
    certification_rows: list[dict] = []

    warmup_condition_np = build_condition(
        np.asarray(current[instance_indices[0]]), np.asarray(pool[instance_indices[0]]),
        feature_min, feature_max,
    )
    warmup_condition = torch.from_numpy(warmup_condition_np)[None].to(device)
    warmup_latent = torch.zeros(
        1, args.candidates, model.latent_dim,
        dtype=warmup_condition.dtype, device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_start = perf_counter()
    with torch.inference_mode():
        warmup_schedules, _, _ = model(
            warmup_condition, args.candidates, free_lower, free_upper,
            free_ramp, free_ramp, warmup_latent,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_warmup_seconds = perf_counter() - warmup_start
    decs_warmup_start = perf_counter()
    with torch.inference_mode():
        screen_candidates_with_decs(
            completion,
            warmup_schedules[0, :1],
            np.asarray(current[instance_indices[0]]),
            np.asarray(future[instance_indices[0]]),
            reference_ramp,
            args.decs_feasibility_tolerance,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decs_warmup_seconds = perf_counter() - decs_warmup_start

    print(
        f"Generator-TCN DECS-screened benchmark: split={args.split}, instances={n_selected}, "
        f"candidates={args.candidates}, "
        f"device={device}, pandapower setup={setup_seconds:.3f}s, "
        f"Generator/DECS warmup={inference_warmup_seconds:.3f}/"
        f"{decs_warmup_seconds:.3f}s",
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

        decs_start = perf_counter()
        with torch.inference_mode():
            decs = screen_candidates_with_decs(
                completion,
                schedules[0],
                np.asarray(current[instance]),
                np.asarray(future[instance]),
                reference_ramp,
                args.decs_feasibility_tolerance,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decs_seconds = perf_counter() - decs_start
        candidate_order = np.asarray(decs["order"], dtype=int)
        decs_feasible_count = int(np.count_nonzero(decs["feasible"]))

        validation_start = perf_counter()
        best_candidate = -1
        best_result = None
        total_pf_attempts = 0
        total_pf_points = 0
        candidates_tested = 0
        selected_rank = -1
        tested_results: dict[int, dict] = {}
        certification_positions: dict[int, int] = {}
        for rank, candidate in enumerate(candidate_order):
            candidate = int(candidate)
            schedule = schedules_np[candidate]
            result = validate_shared_schedule(
                net, case, np.asarray(current[instance]), np.asarray(future[instance]),
                schedule, ramp_fraction=args.ramp_fraction,
                power_limit_tolerance_mva=args.power_limit_tolerance_mva,
                voltage_limit_tolerance_pu=args.voltage_limit_tolerance_pu,
                angle_limit_tolerance_rad=args.angle_limit_tolerance_rad,
                pf_tolerance_mva=args.pf_tolerance_mva,
                max_pf_iterations=args.max_pf_iterations,
                early_stop=True, save_states=True,
            )
            tested_results[candidate] = result
            candidates_tested += 1
            total_pf_attempts += result["pf_attempts"]
            total_pf_points += result["pf_points"]
            certification_row = {
                "instance_index": int(instance),
                "decs_rank": rank + 1,
                "candidate_index": int(candidate),
                "decs_feasible": int(np.asarray(decs["feasible"])[candidate]),
                "decs_cost": float(np.asarray(decs["objective"])[candidate]),
                "decs_max_violation": float(
                    np.asarray(decs["maximum_violation"])[candidate]
                ),
                "decs_pf_residual_pu": float(np.asarray(decs["pf_residual"])[candidate]),
                "evaluation_mode": "feasibility_search",
                "pandapower_feasible": int(result["feasible"]),
                "pandapower_cost": float(result["objective"]),
                "cost_available": int(result["cost_available"]),
                "pf_attempts": result["pf_attempts"],
                "pf_points": result["pf_points"],
                **{
                    name: float(result[name])
                    for name in (
                        "max_pg_excess_mw", "max_qg_excess_mvar",
                        "max_voltage_excess_pu",
                        "max_angle_excess_rad", "max_thermal_excess_mva",
                        "max_ramp_excess_mw",
                    )
                },
            }
            certification_positions[candidate] = len(certification_rows)
            certification_rows.append(certification_row)
            write_csv(
                args.output_dir / "pandapower_certification_attempts.csv",
                certification_rows,
            )
            if not result["feasible"]:
                continue
            # The DECS-feasible portion is already sorted by economic cost, so
            # the first pandapower-certified candidate is the requested answer.
            best_candidate = candidate
            selected_rank = rank
            best_result = result
            break

        if best_result is None:
            # Every candidate failed the physical limits. Preserve the highest
            # ranked generated output and fully evaluate its pandapower cost.
            best_candidate = int(candidate_order[0])
            selected_rank = 0
            best_result = tested_results[best_candidate]
            if not best_result["cost_available"]:
                best_result = validate_shared_schedule(
                    net, case,
                    np.asarray(current[instance]), np.asarray(future[instance]),
                    schedules_np[best_candidate], ramp_fraction=args.ramp_fraction,
                    power_limit_tolerance_mva=args.power_limit_tolerance_mva,
                    voltage_limit_tolerance_pu=args.voltage_limit_tolerance_pu,
                    angle_limit_tolerance_rad=args.angle_limit_tolerance_rad,
                    pf_tolerance_mva=args.pf_tolerance_mva,
                    max_pf_iterations=args.max_pf_iterations,
                    early_stop=False, save_states=True,
                )
                total_pf_attempts += best_result["pf_attempts"]
                total_pf_points += best_result["pf_points"]
                position = certification_positions[best_candidate]
                certification_rows[position].update({
                    "evaluation_mode": "full_cost_after_infeasible_search",
                    "pandapower_feasible": int(best_result["feasible"]),
                    "pandapower_cost": float(best_result["objective"]),
                    "cost_available": int(best_result["cost_available"]),
                    "pf_attempts": best_result["pf_attempts"],
                    "pf_points": best_result["pf_points"],
                    **{
                        name: float(best_result[name])
                        for name in (
                            "max_pg_excess_mw", "max_qg_excess_mvar",
                            "max_voltage_excess_pu", "max_angle_excess_rad",
                            "max_thermal_excess_mva", "max_ramp_excess_mw",
                        )
                    },
                })
                write_csv(
                    args.output_dir / "pandapower_certification_attempts.csv",
                    certification_rows,
                )
        validation_seconds = perf_counter() - validation_start
        feasible = bool(best_result["feasible"])
        selected_u[local_index] = schedules_np[best_candidate]
        if "states" in best_result:
            for name, target in (
                ("pg", selected_pg), ("qg", selected_qg),
                ("vm", selected_vm), ("va", selected_va),
            ):
                target[local_index] = best_result["states"][name]

        row = {
            "instance_index": int(instance),
            "feasible": int(feasible),
            "candidates": args.candidates,
            "decs_feasible_candidates": decs_feasible_count,
            "pandapower_candidates_tested": candidates_tested,
            "pandapower_rejected_candidates": candidates_tested - int(feasible),
            "best_candidate_index": best_candidate,
            "selected_decs_rank": selected_rank + 1,
            "selected_decs_feasible": (
                int(np.asarray(decs["feasible"])[best_candidate])
            ),
            "selected_decs_cost": (
                float(np.asarray(decs["objective"])[best_candidate])
            ),
            "selected_decs_max_violation": (
                float(np.asarray(decs["maximum_violation"])[best_candidate])
            ),
            "selected_decs_pf_residual_pu": (
                float(np.asarray(decs["pf_residual"])[best_candidate])
            ),
            "cost_available": int(best_result["cost_available"]),
            "feasible_cost": (
                float(best_result["objective"]) if feasible else np.nan
            ),
            "all_cost": float(best_result["objective"]),
            "best_cost": float(best_result["objective"]),
            "generation_seconds": generation_seconds,
            "decs_screening_seconds": decs_seconds,
            "pf_validation_seconds": validation_seconds,
            "total_seconds": generation_seconds + decs_seconds + validation_seconds,
            "pf_attempts": total_pf_attempts,
            "pf_points": total_pf_points,
        }
        for name in (
            "max_pg_excess_mw", "max_qg_excess_mvar",
            "max_voltage_excess_pu",
            "max_angle_excess_rad", "max_thermal_excess_mva",
            "max_ramp_excess_mw",
        ):
            row[name] = float(best_result[name])
        rows.append(row)
        write_csv(args.output_dir / "generator_results.csv", rows)
        print(
            f"[{local_index + 1:4d}/{n_selected}] instance={instance} "
            f"feasible={feasible} DECS_feasible={decs_feasible_count}/{args.candidates} "
            f"PP_tested={candidates_tested} cost="
            f"{format_reported_cost(row['all_cost'])} "
            f"total={row['total_seconds']:.3f}s",
            flush=True,
        )

    solution_payload = {
        "data_split": np.asarray(args.split),
        "instance_indices": instance_indices,
        f"{args.split}_indices": instance_indices,
        "u": selected_u, "pg": selected_pg, "qg": selected_qg,
        "vm": selected_vm, "va": selected_va,
    }
    np.savez_compressed(
        args.output_dir / "generator_solutions.npz", **solution_payload,
    )
    feasible_rows = [row for row in rows if row["feasible"]]
    cost_summary = summarize_solution_costs(rows)
    summary = {
        "method": "Generator-TCN + DECS batch ranking + pandapower certification",
        "checkpoint": str(args.checkpoint.resolve()),
        "decs_checkpoint": str(args.decs.resolve()),
        "data_split": args.split,
        "n_instances": n_selected,
        "candidates_per_instance": args.candidates,
        "feasible_instances": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / n_selected,
        "generator_model_load_seconds": generator_model_load_seconds,
        "decs_model_load_seconds": decs_model_load_seconds,
        "setup_seconds": setup_seconds,
        "inference_warmup_seconds": inference_warmup_seconds,
        "decs_warmup_seconds": decs_warmup_seconds,
        "time_seconds": finite_statistics([row["total_seconds"] for row in rows]),
        "generation_seconds": finite_statistics([row["generation_seconds"] for row in rows]),
        "decs_screening_seconds": finite_statistics([
            row["decs_screening_seconds"] for row in rows
        ]),
        "pf_validation_seconds": finite_statistics([row["pf_validation_seconds"] for row in rows]),
        "pandapower_candidates_tested": finite_statistics([
            row["pandapower_candidates_tested"] for row in rows
        ]),
        # ``best_cost`` remains as a compatibility alias for all output costs.
        "best_cost": cost_summary["all_solution_cost"],
        **cost_summary,
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "decs_feasibility_tolerance": args.decs_feasibility_tolerance,
        "pandapower_feasibility_rule": (
            "converged AC power flow and direct Pg/Qg/V/angle/thermal/ramp bounds"
        ),
        "power_limit_tolerance_mva": args.power_limit_tolerance_mva,
        "voltage_limit_tolerance_pu": args.voltage_limit_tolerance_pu,
        "angle_limit_tolerance_rad": args.angle_limit_tolerance_rad,
        "pf_tolerance_mva": args.pf_tolerance_mva,
        "seed": args.seed,
    }
    write_json(args.output_dir / "generator_summary.json", summary)
    print(f"saved Generator benchmark to {args.output_dir}")


def main() -> None:
    """Run the native PGM batch evaluator used for Generator results."""
    from evaluate_generator_tcn_pgm_batch import main as pgm_main

    pgm_main()


if __name__ == "__main__":
    main()
