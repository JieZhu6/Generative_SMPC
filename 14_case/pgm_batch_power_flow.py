"""Native batched AC power flow for the local PGLib IEEE-14 case.

PGM ``voltage_regulator`` components enforce PV-voltage setpoints and generator
reactive-power limits inside one native Newton-Raphson batch solve.  No
Python-side reactive-power correction is used here.
"""

from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter

import numpy as np
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

from Data_generation.case14_pglib import Case14


W_PER_MW = 1.0e6
NATIVE_VOLTAGE_REGULATOR_MIN_VERSION = "1.13.154"


def _native_voltage_regulator_component():
    """Return the native regulator component or raise an actionable error."""
    installed = version("power-grid-model")
    component = getattr(ComponentType, "voltage_regulator", None)
    message = (
        "Native PGM voltage_regulator support is "
        f"required. Installed power-grid-model={installed}; version "
        f">={NATIVE_VOLTAGE_REGULATOR_MIN_VERSION} is required. "
        "Please upgrade power-grid-model."
    )
    installed_release = tuple(int(part) for part in installed.split(".")[:3])
    required_release = tuple(
        int(part) for part in NATIVE_VOLTAGE_REGULATOR_MIN_VERSION.split(".")
    )
    if installed_release < required_release or component is None:
        raise RuntimeError(message)

    required_fields = {
        DatasetType.input: {"id", "regulated_object", "status", "u_ref", "q_min", "q_max"},
        DatasetType.update: {"id", "u_ref"},
        DatasetType.sym_output: {"id", "energized", "limit_violated"},
    }
    try:
        for dataset_type, fields in required_fields.items():
            names = set(initialize_array(dataset_type, component, 0).dtype.names or ())
            if not fields.issubset(names):
                raise RuntimeError(message)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(message) from error
    return component


@dataclass(frozen=True)
class PGMCase:
    """PGM model and deterministic component mappings for IEEE-14."""

    model: PowerGridModel
    node_ids: np.ndarray
    branch_ids: np.ndarray
    load_ids: np.ndarray
    generator_ids: np.ndarray
    voltage_regulator_ids: np.ndarray
    voltage_regulator_component: object
    source_id: int
    nonreference_generators: np.ndarray
    nonreference_generator_buses: np.ndarray


def build_pgm_case(
    case: Case14,
    source_short_circuit_mva: float = 1.0e14,
) -> PGMCase:
    """Build a native PGM representation of the MATPOWER case.

    ``generic_branch`` is used for every MATPOWER branch so that line charging,
    off-nominal tap ratios, and phase shifts follow the same PI model.

    This builder always uses the fixed-PV formulation shared by data
    generation, neural training, IPOPT comparison, and final evaluation.
    Reactive limits remain feasibility inequalities; they do not change a PV
    bus into a PQ bus during the power-flow solve.
    """
    if source_short_circuit_mva <= 0:
        raise ValueError("source_short_circuit_mva must be positive")
    regulator_component = _native_voltage_regulator_component()

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
    # Required by sym_gen input schema, but native regulators determine Q.
    generators["q_specified"] = case.gen[nonreference, 2] * W_PER_MW

    voltage_regulator_ids = np.arange(
        60_000, 60_000 + len(nonreference), dtype=np.int32,
    )
    voltage_regulators = initialize_array(
        DatasetType.input, regulator_component, len(nonreference),
    )
    voltage_regulators["id"] = voltage_regulator_ids
    voltage_regulators["regulated_object"] = generator_ids
    voltage_regulators["status"] = 1
    voltage_regulators["u_ref"] = case.bus[nonreference_buses, 7]
    # PGM requires finite regulator limits. A deliberately inactive numerical
    # range preserves PV voltage control; the physical Q limits are checked on
    # the solved qg values by the common feasibility evaluator.
    voltage_regulators["q_min"] = -1.0e6 * W_PER_MW
    voltage_regulators["q_max"] = 1.0e6 * W_PER_MW

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
        regulator_component: voltage_regulators,
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
        voltage_regulator_ids=voltage_regulator_ids,
        voltage_regulator_component=regulator_component,
        source_id=source_id,
        nonreference_generators=nonreference,
        nonreference_generator_buses=nonreference_buses,
    )


def _batch_updates(
    pgm_case: PGMCase,
    case: Case14,
    loads_mva: np.ndarray,
    controls: np.ndarray,
) -> dict:
    """Create dense PGM update arrays for one batch of operating points."""
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

    regulator_update = initialize_array(
        DatasetType.update,
        pgm_case.voltage_regulator_component,
        (batch, len(pgm_case.voltage_regulator_ids)),
    )
    regulator_update["id"] = pgm_case.voltage_regulator_ids
    voltage_start = len(case.nonreference_active_generators)
    regulator_update["u_ref"] = controls[:, voltage_start:-1]

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
        pgm_case.voltage_regulator_component: regulator_update,
        ComponentType.source: source_update,
    }


def solve_pv_batch(
    pgm_case: PGMCase,
    case: Case14,
    loads_mva: np.ndarray,
    controls: np.ndarray,
    *,
    pgm_error_tolerance: float = 1.0e-10,
    pgm_max_iterations: int = 30,
    threading: int = -1,
    continue_on_batch_error: bool = False,
) -> dict[str, np.ndarray | int | float]:
    """Solve many MATPOWER PV operating points in one native PGM batch.

    Parameters
    ----------
    loads_mva : ndarray, shape (B, n_load, 2)
        Active/reactive loads in MW/Mvar.
    controls : ndarray, shape (B, free_dim)
        ``[Pg_nonref, V_PV..., V_ref]`` controls used by the neural model.
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
    _native_voltage_regulator_component()
    update_data = _batch_updates(pgm_case, case, loads_mva, controls)
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
            continue_on_batch_error=continue_on_batch_error,
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
            "No Python-side reactive-power correction fallback is used."
        ) from error
    power_flow_seconds = perf_counter() - power_flow_start

    batch_success = np.ones(batch, dtype=bool)
    batch_error = pgm_case.model.batch_error
    if batch_error is not None:
        batch_success[np.asarray(batch_error.failed_scenarios, dtype=int)] = False

    extract_start = perf_counter()
    node_output = output[ComponentType.node]
    branch_output = output[ComponentType.generic_branch]
    gen_output = output[ComponentType.sym_gen]
    source_output = output[ComponentType.source]

    pg = np.empty((batch, case.n_gen))
    qg = np.empty_like(pg)
    pg[:, case.reference_generator] = source_output["p"][:, 0] / W_PER_MW
    qg[:, case.reference_generator] = source_output["q"][:, 0] / W_PER_MW
    pg[:, pgm_case.nonreference_generators] = gen_output["p"] / W_PER_MW
    qg[:, pgm_case.nonreference_generators] = gen_output["q"] / W_PER_MW

    target_vm = controls[:, len(case.nonreference_active_generators):-1]
    raw_pv_error = np.abs(node_output["u_pu"][:, case.pv_buses] - target_vm)
    maximum_error_by_point = np.where(
        batch_success, np.max(raw_pv_error, axis=1), np.inf,
    )
    diagnostic_tolerance = max(pgm_error_tolerance, 10.0 * np.finfo(float).eps)
    pv_converged = batch_success & (maximum_error_by_point <= diagnostic_tolerance)
    maximum_error = (
        float(np.max(maximum_error_by_point[batch_success]))
        if np.any(batch_success) else np.inf
    )
    extract_seconds = perf_counter() - extract_start
    total_seconds = perf_counter() - total_start

    return {
        "pg": pg,
        "qg": qg,
        "vm": node_output["u_pu"],
        "va": node_output["u_angle"],
        "pf": branch_output["p_from"] / W_PER_MW,
        "qf": branch_output["q_from"] / W_PER_MW,
        "pt": branch_output["p_to"] / W_PER_MW,
        "qt": branch_output["q_to"] / W_PER_MW,
        "pv_converged": pv_converged,
        "batch_success": batch_success,
        "pv_iterations": 1,
        "pv_voltage_error_pu": raw_pv_error,
        "max_pv_voltage_error_pu": maximum_error,
        "t_prepare": prepare_seconds,
        "t_power_flow": power_flow_seconds,
        "t_extract": extract_seconds,
        "t_total": total_seconds,
    }
