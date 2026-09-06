"""Batched fixed-PV AC power flow for the local PGLib IEEE-118 case.

A vectorized full-Newton solve recovers the fixed-PV high-voltage states and
unconstrained PV-generator reactive powers. PGM then evaluates all points in
one native PQ batch. Native results are retained when they match that branch;
native failures or different branches use the fixed-PV states.
"""

from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter

import numpy as np
import torch
from power_grid_model import (
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    LoadGenType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.validation import assert_valid_input_data

from Data_generation.case118_pglib import Case118


W_PER_MW = 1.0e6


@dataclass(frozen=True)
class PGMCase:
    """PGM model and deterministic component mappings for IEEE-118."""

    model: PowerGridModel
    node_ids: np.ndarray
    branch_ids: np.ndarray
    load_ids: np.ndarray
    generator_ids: np.ndarray
    source_id: int
    nonreference_generators: np.ndarray
    nonreference_generator_buses: np.ndarray


def build_pgm_case(
    case: Case118,
    source_short_circuit_mva: float = 1.0e14,
) -> PGMCase:
    """Build a native PGM representation of the MATPOWER case.

    ``generic_branch`` is used for every MATPOWER branch so that line charging,
    off-nominal tap ratios, and phase shifts follow the same PI model.

    Generators are represented as PQ injections.  :func:`solve_pv_batch`
    computes their unconstrained reactive powers so that the requested PV
    voltage magnitudes hold.  Reactive limits remain post-solve feasibility
    inequalities and never trigger PV-to-PQ switching.
    """
    if source_short_circuit_mva <= 0:
        raise ValueError("source_short_circuit_mva must be positive")
    node_ids = case.bus[:, 0].astype(np.int32)
    nodes = initialize_array(DatasetType.input, ComponentType.node, case.n_bus)
    nodes["id"] = node_ids
    nodes["u_rated"] = case.bus[:, 9] * 1.0e3

    branch_ids = np.arange(10_000, 10_000 + len(case.branch), dtype=np.int32)
    branches = initialize_array(
        DatasetType.input, ComponentType.generic_branch, len(case.branch),
    )
    branches["id"] = branch_ids
    branches["from_node"] = case.branch[:, 0].astype(np.int32)
    branches["to_node"] = case.branch[:, 1].astype(np.int32)
    branches["from_status"] = 1
    branches["to_status"] = 1
    to_bus = case.branch[:, 1].astype(int) - 1
    z_base = (case.bus[to_bus, 9] * 1.0e3) ** 2 / (case.base_mva * W_PER_MW)
    y_base = 1.0 / z_base
    branches["r1"] = case.branch[:, 2] * z_base
    branches["x1"] = case.branch[:, 3] * z_base
    branches["g1"] = 0.0
    branches["b1"] = case.branch[:, 4] * y_base
    branches["k"] = np.where(case.branch[:, 8] == 0.0, 1.0, case.branch[:, 8])
    branches["theta"] = np.deg2rad(case.branch[:, 9])
    branches["sn"] = case.branch[:, 5] * W_PER_MW

    load_ids = np.arange(20_000, 20_000 + len(case.load_buses), dtype=np.int32)
    loads = initialize_array(
        DatasetType.input, ComponentType.sym_load, len(case.load_buses),
    )
    loads["id"] = load_ids
    loads["node"] = node_ids[case.load_buses]
    loads["status"] = 1
    loads["type"] = LoadGenType.const_power
    loads["p_specified"] = case.bus[case.load_buses, 2] * W_PER_MW
    loads["q_specified"] = case.bus[case.load_buses, 3] * W_PER_MW

    nonreference = np.flatnonzero(np.arange(case.n_gen) != case.reference_generator)
    nonreference_buses = case.generator_buses[nonreference]
    if len(nonreference) != len(case.pv_buses) or not np.array_equal(
        nonreference_buses, case.pv_buses,
    ):
        raise ValueError("native PV mapping requires one nonreference generator per PV bus")
    generator_ids = np.arange(30_000, 30_000 + len(nonreference), dtype=np.int32)
    generators = initialize_array(
        DatasetType.input, ComponentType.sym_gen, len(nonreference),
    )
    generators["id"] = generator_ids
    generators["node"] = node_ids[nonreference_buses]
    generators["status"] = 1
    generators["type"] = LoadGenType.const_power
    generators["p_specified"] = case.gen[nonreference, 1] * W_PER_MW
    # This value is replaced by the fixed-PV batch solve before evaluation.
    generators["q_specified"] = case.gen[nonreference, 2] * W_PER_MW

    source_id = 40_000
    source = initialize_array(DatasetType.input, ComponentType.source, 1)
    source["id"] = source_id
    source["node"] = node_ids[case.reference_bus]
    source["status"] = 1
    source["u_ref"] = case.bus[case.reference_bus, 7]
    source["u_ref_angle"] = 0.0
    source["sk"] = source_short_circuit_mva * W_PER_MW
    source["rx_ratio"] = 0.1
    source["z01_ratio"] = 1.0

    input_data = {
        ComponentType.node: nodes,
        ComponentType.generic_branch: branches,
        ComponentType.sym_load: loads,
        ComponentType.sym_gen: generators,
        ComponentType.source: source,
    }
    shunt_buses = np.flatnonzero((case.bus[:, 4] != 0.0) | (case.bus[:, 5] != 0.0))
    if len(shunt_buses):
        shunts = initialize_array(DatasetType.input, ComponentType.shunt, len(shunt_buses))
        shunts["id"] = np.arange(50_000, 50_000 + len(shunt_buses), dtype=np.int32)
        shunts["node"] = node_ids[shunt_buses]
        shunts["status"] = 1
        voltage_squared = (case.bus[shunt_buses, 9] * 1.0e3) ** 2
        shunts["g1"] = case.bus[shunt_buses, 4] * W_PER_MW / voltage_squared
        shunts["b1"] = case.bus[shunt_buses, 5] * W_PER_MW / voltage_squared
        shunts["g0"] = shunts["g1"]
        shunts["b0"] = shunts["b1"]
        input_data[ComponentType.shunt] = shunts

    assert_valid_input_data(
        input_data, calculation_type=CalculationType.power_flow, symmetric=True,
    )

    return PGMCase(
        model=PowerGridModel(input_data),
        node_ids=node_ids,
        branch_ids=branch_ids,
        load_ids=load_ids,
        generator_ids=generator_ids,
        source_id=source_id,
        nonreference_generators=nonreference,
        nonreference_generator_buses=nonreference_buses,
    )


def _batch_updates(
    pgm_case: PGMCase,
    case: Case118,
    loads_mva: np.ndarray,
    controls: np.ndarray,
    q_nonreference_mvar: np.ndarray,
) -> dict:
    """Create dense PGM PQ updates using fixed-PV reactive-power solutions."""
    batch = len(loads_mva)
    load_update = initialize_array(
        DatasetType.update, ComponentType.sym_load, (batch, len(pgm_case.load_ids)),
    )
    load_update["id"] = pgm_case.load_ids
    load_update["p_specified"] = loads_mva[..., 0] * W_PER_MW
    load_update["q_specified"] = loads_mva[..., 1] * W_PER_MW

    gen_update = initialize_array(
        DatasetType.update,
        ComponentType.sym_gen,
        (batch, len(pgm_case.generator_ids)),
    )
    gen_update["id"] = pgm_case.generator_ids
    p_nonreference = np.broadcast_to(
        case.gen[pgm_case.nonreference_generators, 1],
        (batch, len(pgm_case.generator_ids)),
    ).copy()
    active_position = {
        int(gen): position
        for position, gen in enumerate(pgm_case.nonreference_generators)
    }
    for control_index, generator in enumerate(case.nonreference_active_generators):
        p_nonreference[:, active_position[int(generator)]] = controls[:, control_index]
    gen_update["p_specified"] = p_nonreference * W_PER_MW
    if q_nonreference_mvar.shape != (batch, len(pgm_case.generator_ids)):
        raise ValueError("q_nonreference_mvar has an invalid shape")
    gen_update["q_specified"] = q_nonreference_mvar * W_PER_MW

    source_update = initialize_array(
        DatasetType.update, ComponentType.source, (batch, 1),
    )
    source_update["id"] = pgm_case.source_id
    source_update["u_ref"] = controls[:, -1, None]
    # The neural control has no slack angle; use the existing zero-angle convention.
    source_update["u_ref_angle"] = 0.0
    return {
        ComponentType.sym_load: load_update,
        ComponentType.sym_gen: gen_update,
        ComponentType.source: source_update,
    }


def _fixed_pv_newton_chunk(
    case: Case118,
    loads_mva: np.ndarray,
    controls: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve one chunk of fixed-PV equations with a full float64 AC Jacobian.

    Parameters
    ----------
    case : Case118
        MATPOWER network and bus partitions.
    loads_mva : ndarray, shape (B,n_load,2)
        Active/reactive loads in MW/Mvar.
    controls : ndarray, shape (B,free_dim)
        ``[Pg_nonref,V_PV,V_ref]`` controls.
    tolerance : float
        Maximum P/Q mismatch in per unit.
    max_iterations : int
        Maximum full-Newton iterations.
    device : torch.device
        CPU or CUDA device used for batched dense linear solves.

    Returns
    -------
    vm, va, qg_nonreference, errors, iterations
        Solved voltages, unconstrained PV-generator Q in Mvar, final per-point
        mismatches, and the number of iterations used.
    """
    real = torch.float64
    loads = torch.tensor(loads_mva, dtype=real, device=device)
    control = torch.tensor(controls, dtype=real, device=device)
    ybus = torch.as_tensor(case.ybus, dtype=torch.complex128, device=device)
    gmat, bmat = ybus.real, ybus.imag
    pv = torch.as_tensor(case.pv_buses, dtype=torch.long, device=device)
    pq = torch.as_tensor(case.pq_buses, dtype=torch.long, device=device)
    nonreference_buses = torch.cat((pv, pq))
    load_buses = torch.as_tensor(case.load_buses, dtype=torch.long, device=device)
    generator_buses = torch.as_tensor(
        case.generator_buses, dtype=torch.long, device=device,
    )
    active_generators = torch.as_tensor(
        case.nonreference_active_generators, dtype=torch.long, device=device,
    )
    voltage_buses = torch.as_tensor(
        case.voltage_control_buses, dtype=torch.long, device=device,
    )
    nonreference_generators = torch.as_tensor(
        np.flatnonzero(np.arange(case.n_gen) != case.reference_generator),
        dtype=torch.long,
        device=device,
    )

    batch = len(loads_mva)
    pd = torch.zeros((batch, case.n_bus), dtype=real, device=device)
    qd = torch.zeros_like(pd)
    pd[:, load_buses] = loads[..., 0]
    qd[:, load_buses] = loads[..., 1]
    pg = torch.as_tensor(case.gen[:, 9], dtype=real, device=device)
    pg = pg.expand(batch, -1).clone()
    active_count = len(case.nonreference_active_generators)
    pg[:, active_generators] = control[:, :active_count]
    p_generation = torch.zeros_like(pd)
    p_generation.index_add_(1, generator_buses, pg)
    p_specified = (p_generation - pd) / case.base_mva
    q_specified = -qd / case.base_mva

    vm = torch.ones_like(pd)
    va = torch.zeros_like(pd)
    vm[:, voltage_buses] = control[:, active_count:]
    diagonal = torch.arange(case.n_bus, device=device)
    n_angle, n_pq = len(nonreference_buses), len(pq)
    errors = torch.full((batch,), torch.inf, dtype=real, device=device)

    for iteration in range(1, max_iterations + 1):
        voltage = torch.polar(vm, va)
        injection = voltage * torch.conj(voltage @ ybus.T)
        active_mismatch = (
            p_specified[:, nonreference_buses]
            - injection.real[:, nonreference_buses]
        )
        reactive_mismatch = (
            q_specified[:, pq] - injection.imag[:, pq]
        )
        errors = torch.maximum(
            active_mismatch.abs().amax(dim=1),
            reactive_mismatch.abs().amax(dim=1),
        )
        unconverged = torch.where(errors > tolerance)[0]
        if len(unconverged) == 0:
            break

        vm_active, va_active = vm[unconverged], va[unconverged]
        angle = va_active[:, :, None] - va_active[:, None, :]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        voltage_product = vm_active[:, :, None] * vm_active[:, None, :]
        h = voltage_product * (gmat * sine - bmat * cosine)
        n = vm_active[:, :, None] * (gmat * cosine + bmat * sine)
        m = -voltage_product * (gmat * cosine + bmat * sine)
        ell = vm_active[:, :, None] * (gmat * sine - bmat * cosine)
        h[:, diagonal, diagonal] = (
            -injection.imag[unconverged] - torch.diag(bmat) * vm_active**2
        )
        n[:, diagonal, diagonal] = (
            injection.real[unconverged] / vm_active
            + torch.diag(gmat) * vm_active
        )
        m[:, diagonal, diagonal] = (
            injection.real[unconverged] - torch.diag(gmat) * vm_active**2
        )
        ell[:, diagonal, diagonal] = (
            injection.imag[unconverged] / vm_active
            - torch.diag(bmat) * vm_active
        )
        top = torch.cat((
            h[:, nonreference_buses][:, :, nonreference_buses],
            n[:, nonreference_buses][:, :, pq],
        ), dim=2)
        bottom = torch.cat((
            m[:, pq][:, :, nonreference_buses],
            ell[:, pq][:, :, pq],
        ), dim=2)
        jacobian = torch.cat((top, bottom), dim=1)
        mismatch = torch.cat((
            active_mismatch[unconverged], reactive_mismatch[unconverged],
        ), dim=1)
        step = torch.linalg.solve(jacobian, mismatch[..., None])[..., 0]
        va[unconverged[:, None], nonreference_buses] += step[:, :n_angle]
        vm[unconverged[:, None], pq] += step[:, n_angle:n_angle + n_pq]

    voltage = torch.polar(vm, va)
    injection = voltage * torch.conj(voltage @ ybus.T) * case.base_mva
    qg = qd[:, generator_buses] + injection.imag[:, generator_buses]
    result = (
        vm.detach().cpu().numpy(),
        va.detach().cpu().numpy(),
        qg[:, nonreference_generators].detach().cpu().numpy(),
        errors.detach().cpu().numpy(),
        iteration,
    )
    return result


def _fixed_pv_reactive_power_batch(
    case: Case118,
    loads_mva: np.ndarray,
    controls: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Solve fixed-PV states and Q for an arbitrarily large batch.

    Parameters are identical to :func:`_fixed_pv_newton_chunk`; ``chunk_size``
    limits the number of dense Jacobians resident in memory.
    """
    batch = len(loads_mva)
    nonreference_count = case.n_gen - 1
    vm = np.full((batch, case.n_bus), np.nan)
    va = np.full_like(vm, np.nan)
    qg = np.full((batch, nonreference_count), np.nan)
    errors = np.full(batch, np.inf)
    maximum_iterations = 0
    start_time = perf_counter()
    for start in range(0, batch, chunk_size):
        stop = min(start + chunk_size, batch)
        vm_chunk, va_chunk, q_chunk, error_chunk, iterations = _fixed_pv_newton_chunk(
            case,
            loads_mva[start:stop],
            controls[start:stop],
            tolerance=tolerance,
            max_iterations=max_iterations,
            device=device,
        )
        vm[start:stop] = vm_chunk
        va[start:stop] = va_chunk
        qg[start:stop] = q_chunk
        errors[start:stop] = error_chunk
        maximum_iterations = max(maximum_iterations, iterations)
    return vm, va, qg, errors, maximum_iterations, perf_counter() - start_time


def _states_from_fixed_pv_voltage(
    case: Case118,
    loads_mva: np.ndarray,
    controls: np.ndarray,
    vm: np.ndarray,
    va: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recover generator and branch states with pandapower conventions."""
    batch = len(loads_mva)
    voltage = vm * np.exp(1j * va)
    injection = voltage * np.conj(voltage @ case.ybus.T) * case.base_mva
    pd = np.zeros((batch, case.n_bus))
    qd = np.zeros_like(pd)
    pd[:, case.load_buses] = loads_mva[..., 0]
    qd[:, case.load_buses] = loads_mva[..., 1]

    pg = np.broadcast_to(case.gen[:, 9], (batch, case.n_gen)).copy()
    active_count = len(case.nonreference_active_generators)
    pg[:, case.nonreference_active_generators] = controls[:, :active_count]
    pg[:, case.reference_generator] = (
        pd[:, case.reference_bus] + injection.real[:, case.reference_bus]
    )
    qg = qd[:, case.generator_buses] + injection.imag[:, case.generator_buses]

    fbus = case.branch[:, 0].astype(int) - 1
    tbus = case.branch[:, 1].astype(int) - 1
    current_from = case.yff[None] * voltage[:, fbus] + case.yft[None] * voltage[:, tbus]
    current_to = case.ytf[None] * voltage[:, fbus] + case.ytt[None] * voltage[:, tbus]
    power_from = voltage[:, fbus] * np.conj(current_from) * case.base_mva
    power_to = voltage[:, tbus] * np.conj(current_to) * case.base_mva
    return {
        "pg": pg,
        "qg": qg,
        "pf": power_from.real,
        "qf": power_from.imag,
        "pt": power_to.real,
        "qt": power_to.imag,
    }


def solve_pv_batch(
    pgm_case: PGMCase,
    case: Case118,
    loads_mva: np.ndarray,
    controls: np.ndarray,
    *,
    pgm_error_tolerance: float = 1.0e-10,
    pgm_max_iterations: int = 30,
    threading: int = -1,
    continue_on_batch_error: bool = False,
    fixed_pv_chunk_size: int = 1024,
    fixed_pv_device: str = "auto",
) -> dict[str, np.ndarray | int | float]:
    """Solve fixed-PV points and cross-check them in one native PGM batch.

    Parameters
    ----------
    loads_mva : ndarray, shape (B, n_load, 2)
        Active/reactive loads in MW/Mvar.
    controls : ndarray, shape (B, free_dim)
        ``[Pg_nonref, V_PV..., V_ref]`` controls used by the neural model.
    pgm_error_tolerance : float, default=1e-10
        Positive per-unit mismatch tolerance for both Newton solves.
    pgm_max_iterations : int, default=30
        Positive iteration limit for both Newton solves.
    threading : int, default=-1
        Native PGM batch worker count; negative means sequential.
    continue_on_batch_error : bool, default=False
        Whether points that fail the fixed-PV Newton stage may remain in the
        returned batch. Native PGM partial errors are always collected.
    fixed_pv_chunk_size : int, default=1024
        Maximum number of dense fixed-PV Jacobians solved simultaneously.
    fixed_pv_device : {"auto","cpu","cuda"}, default="auto"
        Torch device for the vectorized fixed-PV Newton stage.
    """
    total_start = perf_counter()
    loads_mva = np.asarray(loads_mva, dtype=float)
    controls = np.asarray(controls, dtype=float)
    batch = len(loads_mva)
    if loads_mva.shape != (batch, len(case.load_buses), 2):
        raise ValueError("loads_mva must have shape (B, n_load_buses, 2)")
    expected_free = len(case.nonreference_active_generators) + len(case.voltage_control_buses)
    if controls.shape != (batch, expected_free):
        raise ValueError("controls must have shape (B, free_dim)")
    if pgm_error_tolerance <= 0 or pgm_max_iterations < 1:
        raise ValueError("invalid PGM Newton-Raphson settings")
    if fixed_pv_chunk_size < 1:
        raise ValueError("fixed_pv_chunk_size must be positive")
    if fixed_pv_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("fixed_pv_device must be auto, cpu, or cuda")
    if fixed_pv_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for fixed-PV Newton but is unavailable")
    device = torch.device(
        "cuda" if fixed_pv_device == "auto" and torch.cuda.is_available()
        else "cpu" if fixed_pv_device == "auto" else fixed_pv_device
    )
    fixed_pv_tolerance = min(pgm_error_tolerance, 1.0e-10)
    fixed_vm, fixed_va, q_nonreference, fixed_pv_errors, fixed_pv_iterations, fixed_pv_seconds = (
        _fixed_pv_reactive_power_batch(
            case,
            loads_mva,
            controls,
            tolerance=fixed_pv_tolerance,
            max_iterations=pgm_max_iterations,
            chunk_size=fixed_pv_chunk_size,
            device=device,
        )
    )
    fixed_pv_success = np.isfinite(fixed_pv_errors) & (
        fixed_pv_errors <= fixed_pv_tolerance
    )
    if not np.all(fixed_pv_success) and not continue_on_batch_error:
        failed = np.flatnonzero(~fixed_pv_success)
        raise RuntimeError(
            "fixed-PV Newton failed for "
            f"{len(failed)}/{batch} points; first failed indices={failed[:10].tolist()}"
        )
    update_data = _batch_updates(
        pgm_case, case, loads_mva, controls, q_nonreference,
    )
    prepare_seconds = perf_counter() - total_start

    power_flow_start = perf_counter()
    try:
        output = pgm_case.model.calculate_power_flow(
            symmetric=True,
            error_tolerance=pgm_error_tolerance,
            max_iterations=pgm_max_iterations,
            calculation_method=CalculationMethod.newton_raphson,
            update_data=update_data,
            threading=threading,
            continue_on_batch_error=True,
            output_component_types=[
                ComponentType.node,
                ComponentType.generic_branch,
                ComponentType.sym_gen,
                ComponentType.source,
            ],
        )
    except OSError as error:
        raise RuntimeError(
            "Native PGM batch calculation failed in power-grid-model "
            f"{version('power-grid-model')} for batch_size={batch}. "
            "The fixed-PV Q recovery succeeded before this native PQ solve."
        ) from error
    power_flow_seconds = perf_counter() - power_flow_start

    native_pgm_success = np.ones(batch, dtype=bool)
    batch_error = pgm_case.model.batch_error
    if batch_error is not None:
        native_pgm_success[np.asarray(batch_error.failed_scenarios, dtype=int)] = False

    extract_start = perf_counter()
    node_output = output[ComponentType.node]
    branch_output = output[ComponentType.generic_branch]
    gen_output = output[ComponentType.sym_gen]
    source_output = output[ComponentType.source]

    native_pg = np.empty((batch, case.n_gen))
    native_qg = np.empty_like(native_pg)
    native_pg[:, case.reference_generator] = source_output["p"][:, 0] / W_PER_MW
    native_qg[:, case.reference_generator] = source_output["q"][:, 0] / W_PER_MW
    native_pg[:, pgm_case.nonreference_generators] = gen_output["p"] / W_PER_MW
    native_qg[:, pgm_case.nonreference_generators] = q_nonreference
    native_states = {
        "pg": native_pg,
        "qg": native_qg,
        "vm": np.asarray(node_output["u_pu"], dtype=float),
        "va": np.asarray(node_output["u_angle"], dtype=float),
        "pf": np.asarray(branch_output["p_from"], dtype=float) / W_PER_MW,
        "qf": np.asarray(branch_output["q_from"], dtype=float) / W_PER_MW,
        "pt": np.asarray(branch_output["p_to"], dtype=float) / W_PER_MW,
        "qt": np.asarray(branch_output["q_to"], dtype=float) / W_PER_MW,
    }
    target_vm = controls[:, len(case.nonreference_active_generators):-1]
    diagnostic_tolerance = max(pgm_error_tolerance, 10.0 * np.finfo(float).eps)
    native_pv_error = np.max(
        np.abs(native_states["vm"][:, case.pv_buses] - target_vm), axis=1,
    )
    native_voltage = native_states["vm"] * np.exp(1j * native_states["va"])
    fixed_voltage = fixed_vm * np.exp(1j * fixed_va)
    native_voltage_error = np.max(np.abs(native_voltage - fixed_voltage), axis=1)
    native_pgm_match = (
        fixed_pv_success
        & native_pgm_success
        & np.isfinite(native_voltage_error)
        & (native_voltage_error <= 1.0e-7)
        & np.isfinite(native_pv_error)
        & (native_pv_error <= diagnostic_tolerance)
    )
    native_pgm_wrong_branch = fixed_pv_success & native_pgm_success & ~native_pgm_match
    fallback_used = fixed_pv_success & ~native_pgm_match

    fixed_states = _states_from_fixed_pv_voltage(
        case, loads_mva, controls, fixed_vm, fixed_va,
    )
    final_states = {"vm": fixed_vm.copy(), "va": fixed_va.copy(), **fixed_states}
    for name in ("pg", "qg", "vm", "va", "pf", "qf", "pt", "qt"):
        final_states[name][native_pgm_match] = native_states[name][native_pgm_match]

    batch_success = fixed_pv_success
    raw_pv_error = np.abs(final_states["vm"][:, case.pv_buses] - target_vm)
    maximum_error_by_point = np.where(
        batch_success, np.max(raw_pv_error, axis=1), np.inf,
    )
    pv_converged = batch_success & (maximum_error_by_point <= diagnostic_tolerance)
    maximum_error = (
        float(np.max(maximum_error_by_point[batch_success]))
        if np.any(batch_success) else np.inf
    )
    extract_seconds = perf_counter() - extract_start
    total_seconds = perf_counter() - total_start

    return {
        **final_states,
        "pv_converged": pv_converged,
        "batch_success": batch_success,
        "native_pgm_success": native_pgm_success,
        "native_pgm_matches_fixed_pv": native_pgm_match,
        "native_pgm_wrong_branch": native_pgm_wrong_branch,
        "native_pgm_voltage_error_pu": native_voltage_error,
        "fixed_pv_fallback_used": fallback_used,
        "pv_iterations": fixed_pv_iterations,
        "pv_voltage_error_pu": raw_pv_error,
        "max_pv_voltage_error_pu": maximum_error,
        "fixed_pv_equation_error_pu": fixed_pv_errors,
        "fixed_pv_device": str(device),
        "t_fixed_pv": fixed_pv_seconds,
        "t_prepare": prepare_seconds,
        "t_power_flow": power_flow_seconds,
        "t_extract": extract_seconds,
        "t_total": total_seconds,
    }
