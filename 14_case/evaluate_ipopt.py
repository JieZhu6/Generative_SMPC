"""Solve the scenario-shared trajectory SMPC-ACOPF with Pyomo and IPOPT.

Unlike the existing two-stage extensive model, all uncertainty scenarios use
one shared free-control trajectory ``u[t,d]``. Scenario indices appear only on
dependent AC states. IPOPT directly enforces the complete nonlinear AC power
flow and operating constraints; its termination status is the feasibility
criterion used by this baseline.
"""

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pyomo.environ as pyo


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case14_pglib import Case14, load_case14  # noqa: E402
from evaluate_generator_tcn import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    finite_statistics,
    write_csv,
    write_json,
)


IPOPT_PATH = Path("D:/anaconda/envs/py3.10/Library/bin/ipopt.exe")


def load_dataset_split_indices(
    data_dir: Path,
    split_name: str,
    n_instances: int,
    max_instances: int | None = None,
) -> np.ndarray:
    """Load one verified validation/test partition directly from the dataset.

    Parameters
    ----------
    data_dir : pathlib.Path
        Base SMPC dataset containing ``split/*_indices.npy``.
    split_name : {"validation", "test"}
        Held-out partition to solve with IPOPT.
    n_instances : int
        Number of instances in ``current_load.npy``.
    max_instances : int or None, default=None
        Optional positive prefix length for a smoke run.

    Returns
    -------
    np.ndarray
        Verified instance indices in their saved deterministic order.
    """
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be 'validation' or 'test'")
    names = ("train", "validation", "test")
    indices = {
        name: np.load(data_dir / "split" / f"{name}_indices.npy")
        for name in names
    }
    for name, values in indices.items():
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"saved {name} indices must be a one-dimensional integer array")
    combined = np.concatenate([indices[name] for name in names])
    if not np.array_equal(np.sort(combined), np.arange(n_instances)):
        raise ValueError("saved train/validation/test indices are not a disjoint partition")
    selected = indices[split_name]
    if max_instances is not None:
        if max_instances < 1:
            raise ValueError("max_instances must be positive")
        selected = selected[:max_instances]
    return selected


def tighten_case_limits(
    case: Case14,
    pg_qg_margin_fraction: float = 0.0,
    voltage_margin_pu: float = 0.0,
    thermal_margin_fraction: float = 0.0,
    angle_margin_fraction: float = 0.0,
) -> Case14:
    """Return a case whose inequality limits reserve explicit operating margin.

    Parameters
    ----------
    case : Case14
        Original PGLib network case.
    pg_qg_margin_fraction : float, default=0
        Fraction of every nonzero Pg/Qg range removed from each boundary.
    voltage_margin_pu : float, default=0
        Absolute p.u. voltage margin removed from both voltage boundaries.
    thermal_margin_fraction : float, default=0
        Fraction of every branch MVA rating reserved as headroom.
    angle_margin_fraction : float, default=0
        Fraction by which both signed angle-difference limits contract toward zero.

    Returns
    -------
    Case14
        Independent arrays with unchanged admittances and tightened limits.
    """
    fractions = (
        pg_qg_margin_fraction, thermal_margin_fraction, angle_margin_fraction,
    )
    if any(value < 0.0 or value >= 0.5 for value in fractions):
        raise ValueError("fractional operating margins must lie in [0,0.5)")
    voltage_span = case.bus[:, 11] - case.bus[:, 12]
    if voltage_margin_pu < 0.0 or np.any(2.0 * voltage_margin_pu >= voltage_span):
        raise ValueError("voltage margin must be nonnegative and smaller than half the range")

    bus = case.bus.copy()
    gen = case.gen.copy()
    branch = case.branch.copy()
    for lower_column, upper_column in ((9, 8), (4, 3)):
        span = gen[:, upper_column] - gen[:, lower_column]
        margin = pg_qg_margin_fraction * span
        gen[:, lower_column] += margin
        gen[:, upper_column] -= margin
    bus[:, 12] += voltage_margin_pu
    bus[:, 11] -= voltage_margin_pu
    branch[:, 5] *= 1.0 - thermal_margin_fraction
    branch[:, 11:13] *= 1.0 - angle_margin_fraction
    return replace(case, bus=bus, gen=gen, branch=branch)


class SharedTrajectorySMPCAcopf:
    """Reusable Pyomo SMPC model with one scenario-shared control trajectory.

    Parameters
    ----------
    horizon : int, default=16
        Number of dispatch periods; at least two.
    n_scenarios : int, default=20
        Number of equally weighted future load scenarios.
    ramp_fraction : float, default=0.25
        Symmetric one-period active-power ramp limit as a fraction of Pmax.
    ipopt_path : pathlib.Path, default=IPOPT_PATH
        IPOPT executable in the Anaconda py3.10 environment.
    pg_qg_margin_fraction : float, default=0
        Fraction of Pg/Qg range reserved at both boundaries.
    voltage_margin_pu : float, default=0
        Voltage headroom reserved at both boundaries in p.u.
    thermal_margin_fraction, angle_margin_fraction : float, defaults=0
        Fractions reserved from branch MVA and angle-difference limits.
    ramp_margin_fraction : float, default=0
        Fraction of the nominal ramp capability held in reserve.
    ipopt_tolerance : float, default=1e-8
        IPOPT scaled optimality tolerance.
    ipopt_constraint_tolerance : float, default=1e-8
        IPOPT absolute constraint-violation tolerance.
    """

    def __init__(
        self,
        horizon: int = 16,
        n_scenarios: int = 20,
        ramp_fraction: float = 0.25,
        ipopt_path: Path = IPOPT_PATH,
        pg_qg_margin_fraction: float = 0.0,
        voltage_margin_pu: float = 0.0,
        thermal_margin_fraction: float = 0.0,
        angle_margin_fraction: float = 0.0,
        ramp_margin_fraction: float = 0.0,
        ipopt_tolerance: float = 1e-8,
        ipopt_constraint_tolerance: float = 1e-8,
    ):
        """Build the network, generalized Pyomo model, and IPOPT solver.

        Parameters
        ----------
        horizon, n_scenarios, ramp_fraction, ipopt_path
            Values documented by :class:`SharedTrajectorySMPCAcopf`.
        pg_qg_margin_fraction : float, default=0
            Fraction removed from both Pg and Qg operating boundaries.
        voltage_margin_pu : float, default=0
            Absolute voltage headroom at each boundary in p.u.
        thermal_margin_fraction, angle_margin_fraction : float, defaults=0
            Reserved fractions of branch rating and angle-difference limits.
        ramp_margin_fraction : float, default=0
            Fraction of the nominal ramp limit held in reserve.
        ipopt_tolerance, ipopt_constraint_tolerance : float, defaults=1e-8
            IPOPT optimality and constraint-violation tolerances.
        """
        if (
            horizon < 2 or n_scenarios < 1 or ramp_fraction <= 0
            or ipopt_tolerance <= 0 or ipopt_constraint_tolerance <= 0
        ):
            raise ValueError("invalid horizon, scenario count, or ramp fraction")
        if ramp_margin_fraction < 0.0 or ramp_margin_fraction >= 1.0:
            raise ValueError("ramp_margin_fraction must lie in [0,1)")
        if not ipopt_path.is_file():
            raise FileNotFoundError(f"IPOPT executable not found: {ipopt_path}")
        self.original_case = load_case14()
        self.case = tighten_case_limits(
            self.original_case,
            pg_qg_margin_fraction,
            voltage_margin_pu,
            thermal_margin_fraction,
            angle_margin_fraction,
        )
        self.horizon = int(horizon)
        self.n_scenarios = int(n_scenarios)
        self.nominal_ramp_fraction = float(ramp_fraction)
        self.ramp_margin_fraction = float(ramp_margin_fraction)
        self.ramp_fraction = ramp_fraction * (1.0 - ramp_margin_fraction)
        self.free_pg = self.case.nonreference_active_generators.tolist()
        self.voltage_buses = self.case.voltage_control_buses.tolist()
        self.free_pg_position = {g: j for j, g in enumerate(self.free_pg)}
        self.voltage_position = {bus: j for j, bus in enumerate(self.voltage_buses)}
        self.points = [(0, 0)] + [
            (scenario, time)
            for scenario in range(self.n_scenarios)
            for time in range(1, self.horizon)
        ]
        self.model = self._build_model()
        self.solver = pyo.SolverFactory("ipopt", executable=str(ipopt_path))
        self.solver.options.update({
            "print_level": 0,
            "tol": ipopt_tolerance,
            "constr_viol_tol": ipopt_constraint_tolerance,
            "max_iter": 2500,
            "bound_push": 1e-8,
            "bound_frac": 1e-8,
            "honor_original_bounds": "yes",
        })

    def _build_model(self) -> pyo.ConcreteModel:
        """Build the shared-control ACOPF generalized-form model.

        Returns
        -------
        pyomo.ConcreteModel
            Shared ``u[t,d]``, scenario states, AC equations, operating limits,
            ramp constraints, and expected generation-cost objective.
        """
        case = self.case
        m = pyo.ConcreteModel()
        m.N = pyo.RangeSet(0, case.n_bus - 1)
        m.G = pyo.RangeSet(0, case.n_gen - 1)
        m.L = pyo.RangeSet(0, len(case.branch) - 1)
        m.T = pyo.RangeSet(0, self.horizon - 1)
        m.U = pyo.RangeSet(0, len(self.free_pg) + len(self.voltage_buses) - 1)
        m.P = pyo.Set(dimen=2, initialize=self.points, ordered=True)
        m.PQ = pyo.Set(initialize=case.pq_buses.tolist(), ordered=True)
        nonreference = [i for i in range(case.n_bus) if i != case.reference_bus]
        m.NR = pyo.Set(initialize=nonreference, ordered=True)
        m.RP = pyo.Set(
            dimen=2,
            initialize=[
                (scenario, time)
                for scenario in range(self.n_scenarios)
                for time in range(1, self.horizon)
            ],
            ordered=True,
        )
        m.pd = pyo.Param(m.P, m.N, mutable=True, initialize=0.0)
        m.qd = pyo.Param(m.P, m.N, mutable=True, initialize=0.0)

        def u_bounds(_, time, component):
            """Return physical bounds for one shared free-control component."""
            if component < len(self.free_pg):
                g = self.free_pg[component]
                return case.gen[g, 9] / case.base_mva, case.gen[g, 8] / case.base_mva
            bus = self.voltage_buses[component - len(self.free_pg)]
            return case.bus[bus, 12], case.bus[bus, 11]

        ref = case.reference_generator
        m.u = pyo.Var(m.T, m.U, bounds=u_bounds)
        m.pg_ref = pyo.Var(
            m.P,
            bounds=(case.gen[ref, 9] / case.base_mva, case.gen[ref, 8] / case.base_mva),
        )
        m.qg = pyo.Var(
            m.P, m.G,
            bounds=lambda _, s, t, g: (
                case.gen[g, 4] / case.base_mva,
                case.gen[g, 3] / case.base_mva,
            ),
        )
        m.vm_pq = pyo.Var(
            m.P, m.PQ,
            bounds=lambda _, s, t, i: (case.bus[i, 12], case.bus[i, 11]),
        )
        m.va = pyo.Var(m.P, m.NR, bounds=(-np.pi, np.pi))

        gens_at_bus = {
            i: [g for g in range(case.n_gen) if case.generator_buses[g] == i]
            for i in range(case.n_bus)
        }
        neighbors = {
            i: np.flatnonzero(np.abs(case.ybus[i]) > 1e-14).tolist()
            for i in range(case.n_bus)
        }
        gmat, bmat = case.ybus.real, case.ybus.imag

        def pg(model, s, t, g):
            """Return scenario state or shared-control active power in p.u."""
            if g == case.reference_generator:
                return model.pg_ref[s, t]
            if g in self.free_pg_position:
                return model.u[t, self.free_pg_position[g]]
            return 0.0

        def vm(model, s, t, i):
            """Return shared controlled or scenario-dependent voltage magnitude."""
            if i in self.voltage_position:
                component = len(self.free_pg) + self.voltage_position[i]
                return model.u[t, component]
            return model.vm_pq[s, t, i]

        def va(model, s, t, i):
            """Return zero reference angle or a scenario-dependent bus angle."""
            return 0.0 if i == case.reference_bus else model.va[s, t, i]

        def balance(model, s, t, i, reactive):
            """Return one nodal active or reactive AC power-balance equality."""
            generation = sum(
                model.qg[s, t, g] if reactive else pg(model, s, t, g)
                for g in gens_at_bus[i]
            )
            if reactive:
                injection = vm(model, s, t, i) * sum(
                    vm(model, s, t, j) * (
                        gmat[i, j] * pyo.sin(va(model, s, t, i) - va(model, s, t, j))
                        - bmat[i, j] * pyo.cos(va(model, s, t, i) - va(model, s, t, j))
                    )
                    for j in neighbors[i]
                )
                return generation - model.qd[s, t, i] == injection
            injection = vm(model, s, t, i) * sum(
                vm(model, s, t, j) * (
                    gmat[i, j] * pyo.cos(va(model, s, t, i) - va(model, s, t, j))
                    + bmat[i, j] * pyo.sin(va(model, s, t, i) - va(model, s, t, j))
                )
                for j in neighbors[i]
            )
            return generation - model.pd[s, t, i] == injection

        m.p_balance = pyo.Constraint(
            m.P, m.N, rule=lambda model, s, t, i: balance(model, s, t, i, False),
        )
        m.q_balance = pyo.Constraint(
            m.P, m.N, rule=lambda model, s, t, i: balance(model, s, t, i, True),
        )

        def flow(model, s, t, ell, from_end, reactive):
            """Return one branch-end active or reactive flow in p.u."""
            row = case.branch[ell]
            i, j = int(row[0]) - 1, int(row[1]) - 1
            if from_end:
                yii, yij = case.yff[ell], case.yft[ell]
            else:
                i, j = j, i
                yii, yij = case.ytt[ell], case.ytf[ell]
            delta = va(model, s, t, i) - va(model, s, t, j)
            if reactive:
                return -yii.imag * vm(model, s, t, i) ** 2 + vm(model, s, t, i) * vm(model, s, t, j) * (
                    yij.real * pyo.sin(delta) - yij.imag * pyo.cos(delta)
                )
            return yii.real * vm(model, s, t, i) ** 2 + vm(model, s, t, i) * vm(model, s, t, j) * (
                yij.real * pyo.cos(delta) + yij.imag * pyo.sin(delta)
            )

        m.pf = pyo.Expression(
            m.P, m.L, rule=lambda model, s, t, ell: flow(model, s, t, ell, True, False),
        )
        m.qf = pyo.Expression(
            m.P, m.L, rule=lambda model, s, t, ell: flow(model, s, t, ell, True, True),
        )
        m.pt = pyo.Expression(
            m.P, m.L, rule=lambda model, s, t, ell: flow(model, s, t, ell, False, False),
        )
        m.qt = pyo.Expression(
            m.P, m.L, rule=lambda model, s, t, ell: flow(model, s, t, ell, False, True),
        )
        m.thermal_f = pyo.Constraint(
            m.P, m.L,
            rule=lambda model, s, t, ell:
            model.pf[s, t, ell] ** 2 + model.qf[s, t, ell] ** 2
            <= (case.branch[ell, 5] / case.base_mva) ** 2,
        )
        m.thermal_t = pyo.Constraint(
            m.P, m.L,
            rule=lambda model, s, t, ell:
            model.pt[s, t, ell] ** 2 + model.qt[s, t, ell] ** 2
            <= (case.branch[ell, 5] / case.base_mva) ** 2,
        )

        def angle_limit(model, s, t, ell, upper):
            """Return one branch angle-difference upper or lower constraint."""
            row = case.branch[ell]
            i, j = int(row[0]) - 1, int(row[1]) - 1
            delta = va(model, s, t, i) - va(model, s, t, j)
            limit = np.deg2rad(row[12] if upper else row[11])
            return delta <= limit if upper else delta >= limit

        m.angle_low = pyo.Constraint(
            m.P, m.L,
            rule=lambda model, s, t, ell: angle_limit(model, s, t, ell, False),
        )
        m.angle_high = pyo.Constraint(
            m.P, m.L,
            rule=lambda model, s, t, ell: angle_limit(model, s, t, ell, True),
        )

        m.free_ramp_up = pyo.Constraint(
            pyo.RangeSet(1, self.horizon - 1), range(len(self.free_pg)),
            rule=lambda model, t, j:
            model.u[t, j] - model.u[t - 1, j]
            <= self.ramp_fraction * case.gen[self.free_pg[j], 8] / case.base_mva,
        )
        m.free_ramp_down = pyo.Constraint(
            pyo.RangeSet(1, self.horizon - 1), range(len(self.free_pg)),
            rule=lambda model, t, j:
            model.u[t - 1, j] - model.u[t, j]
            <= self.ramp_fraction * case.gen[self.free_pg[j], 8] / case.base_mva,
        )
        ref_ramp = self.ramp_fraction * case.gen[ref, 8] / case.base_mva

        def previous_point(s, t):
            """Map a future scenario-time point to its preceding PF state."""
            return (0, 0) if t == 1 else (s, t - 1)

        m.ref_ramp_up = pyo.Constraint(
            m.RP,
            rule=lambda model, s, t:
            model.pg_ref[s, t] - model.pg_ref[previous_point(s, t)] <= ref_ramp,
        )
        m.ref_ramp_down = pyo.Constraint(
            m.RP,
            rule=lambda model, s, t:
            model.pg_ref[previous_point(s, t)] - model.pg_ref[s, t] <= ref_ramp,
        )

        def cost(model, s, t, g):
            """Return one generator's physical single-period quadratic cost."""
            power_mw = case.base_mva * pg(model, s, t, g)
            return (
                case.gencost[g, 4] * power_mw**2
                + case.gencost[g, 5] * power_mw
                + case.gencost[g, 6]
            )

        m.objective = pyo.Objective(expr=sum(
            (1.0 if t == 0 else 1.0 / self.n_scenarios)
            * sum(cost(m, s, t, g) for g in m.G)
            for s, t in m.P
        ))
        return m

    def _physical_u(self, time: int) -> np.ndarray:
        """Extract the current shared control at ``time`` in MW and p.u."""
        values = np.array([pyo.value(self.model.u[time, d]) for d in self.model.U])
        values[:len(self.free_pg)] *= self.case.base_mva
        return values

    def set_loads(
        self,
        current_load: np.ndarray,
        future_load: np.ndarray,
        blend: float = 1.0,
    ) -> np.ndarray:
        """Assign one scenario tree, optionally contracted toward its mean.

        Parameters
        ----------
        current_load : np.ndarray, shape (n_load_buses, 2)
            Deterministic first-period net load in MW/Mvar.
        future_load : np.ndarray, shape (S, T-1, n_load_buses, 2)
            Future scenario loads in MW/Mvar.
        blend : float, default=1.0
            Scenario-spread continuation factor in ``[0,1]``.

        Returns
        -------
        np.ndarray
            Blended future loads actually assigned to the model.
        """
        expected = (
            self.n_scenarios, self.horizon - 1,
            len(self.case.load_buses), 2,
        )
        if current_load.shape != (len(self.case.load_buses), 2) or future_load.shape != expected:
            raise ValueError("scenario tree dimensions do not match the Pyomo model")
        if not 0 <= blend <= 1:
            raise ValueError("blend must lie in [0,1]")
        mean = future_load.mean(axis=0, keepdims=True)
        blended = mean + blend * (future_load - mean)

        def full_bus(load):
            """Expand compressed P/Q loads to all buses in p.u."""
            values = np.zeros((self.case.n_bus, 2))
            values[self.case.load_buses] = load
            return values / self.case.base_mva

        first = full_bus(current_load)
        for i in range(self.case.n_bus):
            self.model.pd[0, 0, i] = float(first[i, 0])
            self.model.qd[0, 0, i] = float(first[i, 1])
        for scenario in range(self.n_scenarios):
            for time in range(1, self.horizon):
                load = full_bus(blended[scenario, time - 1])
                for i in range(self.case.n_bus):
                    self.model.pd[scenario, time, i] = float(load[i, 0])
                    self.model.qd[scenario, time, i] = float(load[i, 1])
        return blended

    def initialize(self) -> None:
        """Initialize Pyomo variables from the PGLib case without a PF solver."""
        case, m = self.case, self.model
        for time in range(self.horizon):
            for j, g in enumerate(self.free_pg):
                value = np.clip(case.gen[g, 1], case.gen[g, 9], case.gen[g, 8])
                m.u[time, j].set_value(value / case.base_mva)
            for j, bus in enumerate(self.voltage_buses):
                voltage = np.clip(case.bus[bus, 7], case.bus[bus, 12], case.bus[bus, 11])
                m.u[time, len(self.free_pg) + j].set_value(float(voltage))

        ref = case.reference_generator
        for s, t in self.points:
            pg_ref = np.clip(case.gen[ref, 1], case.gen[ref, 9], case.gen[ref, 8])
            m.pg_ref[s, t].set_value(float(pg_ref / case.base_mva))
            for g in range(case.n_gen):
                qg = np.clip(case.gen[g, 2], case.gen[g, 4], case.gen[g, 3])
                m.qg[s, t, g].set_value(float(qg / case.base_mva))
            for bus in case.pq_buses:
                voltage = np.clip(
                    case.bus[bus, 7], case.bus[bus, 12], case.bus[bus, 11],
                )
                m.vm_pq[s, t, int(bus)].set_value(float(voltage))
            for bus in range(case.n_bus):
                if bus != case.reference_bus:
                    m.va[s, t, bus].set_value(float(np.deg2rad(case.bus[bus, 8])))

    def _extract_states(self) -> dict[str, np.ndarray]:
        """Extract IPOPT Pg/Qg/V/angle states in physical units.

        Returns
        -------
        dict[str, np.ndarray]
            Arrays with shape ``(S,T,n_gen)`` for Pg/Qg and
            ``(S,T,n_bus)`` for voltage magnitude/angle. The common first
            period is repeated across scenarios.
        """
        case, m = self.case, self.model
        pg = np.zeros((self.n_scenarios, self.horizon, case.n_gen))
        qg = np.zeros_like(pg)
        vm = np.zeros((self.n_scenarios, self.horizon, case.n_bus))
        va = np.zeros_like(vm)
        for scenario in range(self.n_scenarios):
            for time in range(self.horizon):
                point = (0, 0) if time == 0 else (scenario, time)
                s, t = point
                pg[scenario, time, case.reference_generator] = (
                    pyo.value(m.pg_ref[s, t]) * case.base_mva
                )
                for j, generator in enumerate(self.free_pg):
                    pg[scenario, time, generator] = (
                        pyo.value(m.u[time, j]) * case.base_mva
                    )
                for generator in range(case.n_gen):
                    qg[scenario, time, generator] = (
                        pyo.value(m.qg[s, t, generator]) * case.base_mva
                    )
                for bus in range(case.n_bus):
                    if bus in self.voltage_position:
                        component = len(self.free_pg) + self.voltage_position[bus]
                        vm[scenario, time, bus] = pyo.value(m.u[time, component])
                    else:
                        vm[scenario, time, bus] = pyo.value(m.vm_pq[s, t, bus])
                    if bus != case.reference_bus:
                        va[scenario, time, bus] = pyo.value(m.va[s, t, bus])
        return {"pg": pg, "qg": qg, "vm": vm, "va": va}

    def _maximum_constraint_violation(self) -> float:
        """Return the largest absolute Pyomo constraint violation after IPOPT."""
        maximum = 0.0
        for constraint in self.model.component_data_objects(
            pyo.Constraint, active=True,
        ):
            body = float(pyo.value(constraint.body))
            if constraint.lower is not None:
                maximum = max(maximum, float(pyo.value(constraint.lower)) - body)
            if constraint.upper is not None:
                maximum = max(maximum, body - float(pyo.value(constraint.upper)))
        return max(maximum, 0.0)

    def _attempt(self, tee: bool) -> tuple[bool, str]:
        """Run one IPOPT attempt and return success plus termination text.

        Parameters
        ----------
        tee : bool
            Stream IPOPT iteration output when true.
        """
        try:
            result = self.solver.solve(self.model, tee=tee, load_solutions=False)
        except Exception as error:
            return False, f"exception: {error}"
        termination = str(result.solver.termination_condition)
        if "optimal" not in termination.lower():
            return False, termination
        self.model.solutions.load_from(result)
        return True, termination

    def solve(
        self,
        current_load: np.ndarray,
        future_load: np.ndarray,
        tee: bool = False,
    ) -> dict:
        """Solve one scenario tree with direct and continuation IPOPT attempts.

        Parameters
        ----------
        current_load : np.ndarray, shape (n_load_buses, 2)
            Deterministic first-period load in MW/Mvar.
        future_load : np.ndarray, shape (S,T-1,n_load_buses,2)
            Future load scenarios in MW/Mvar.
        tee : bool, default=False
            Stream IPOPT output.

        Returns
        -------
        dict
            Solver status, shared schedule, Pyomo objective, retry count, and
            elapsed solve time including initialization and continuation.
        """
        start = perf_counter()
        self.set_loads(current_load, future_load, blend=1.0)
        self.initialize()
        solved, termination = self._attempt(tee)
        retries = 0
        if not solved:
            for blend in np.linspace(0.0, 1.0, 5):
                retries += 1
                self.set_loads(current_load, future_load, float(blend))
                if blend == 0.0:
                    self.initialize()
                solved, termination = self._attempt(tee)
                if not solved:
                    break
        elapsed = perf_counter() - start
        self.set_loads(current_load, future_load, blend=1.0)
        if not solved:
            return {
                "solved": False, "termination": termination,
                "retries": retries, "solve_seconds": elapsed,
            }
        schedule = np.stack([self._physical_u(time) for time in range(self.horizon)])
        return {
            "solved": True,
            "termination": termination,
            "retries": retries,
            "solve_seconds": elapsed,
            "schedule": schedule,
            "states": self._extract_states(),
            "ipopt_objective": float(pyo.value(self.model.objective)),
            "ipopt_max_constraint_violation": self._maximum_constraint_violation(),
        }

def compare_methods(output_dir: Path, ipopt_rows: list[dict]) -> dict | None:
    """Merge matching Generator and IPOPT rows and write comparison artifacts.

    Parameters
    ----------
    output_dir : pathlib.Path
        Benchmark directory containing ``generator_results.csv``.
    ipopt_rows : list[dict]
        Current IPOPT per-instance results.

    Returns
    -------
    dict or None
        Aggregate comparison summary, or ``None`` when Generator results are absent.
    """
    generator_path = output_dir / "generator_results.csv"
    if not generator_path.is_file():
        return None
    with generator_path.open("r", newline="", encoding="utf-8") as stream:
        generator_rows = {
            int(row["instance_index"]): row for row in csv.DictReader(stream)
        }
    comparison = []
    for ipopt in ipopt_rows:
        instance = int(ipopt["instance_index"])
        if instance not in generator_rows:
            continue
        generator = generator_rows[instance]
        generator_feasible = bool(int(generator["feasible"]))
        ipopt_feasible = bool(int(ipopt["feasible"]))
        generator_cost = float(generator["best_cost"]) if generator_feasible else np.nan
        ipopt_cost = float(ipopt["best_cost"]) if ipopt_feasible else np.nan
        both = generator_feasible and ipopt_feasible
        gap = (
            100.0 * (generator_cost - ipopt_cost) / max(abs(ipopt_cost), 1e-12)
            if both else np.nan
        )
        comparison.append({
            "instance_index": instance,
            "generator_feasible": int(generator_feasible),
            "ipopt_feasible": int(ipopt_feasible),
            "generator_cost": generator_cost,
            "ipopt_cost": ipopt_cost,
            "generator_minus_ipopt_percent": gap,
            "generator_seconds": float(generator["total_seconds"]),
            "ipopt_seconds": float(ipopt["total_seconds"]),
            "generator_feasible_candidates": int(generator["feasible_candidates"]),
        })
    if not comparison:
        return None
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    both_rows = [
        row for row in comparison
        if row["generator_feasible"] and row["ipopt_feasible"]
    ]
    gaps = [row["generator_minus_ipopt_percent"] for row in both_rows]
    summary = {
        "n_matched_instances": len(comparison),
        "generator_feasible": sum(row["generator_feasible"] for row in comparison),
        "ipopt_feasible": sum(row["ipopt_feasible"] for row in comparison),
        "both_feasible": len(both_rows),
        "generator_only_feasible": sum(
            row["generator_feasible"] and not row["ipopt_feasible"] for row in comparison
        ),
        "ipopt_only_feasible": sum(
            row["ipopt_feasible"] and not row["generator_feasible"] for row in comparison
        ),
        "generator_time_seconds": finite_statistics([
            row["generator_seconds"] for row in comparison
        ]),
        "ipopt_time_seconds": finite_statistics([
            row["ipopt_seconds"] for row in comparison
        ]),
        "generator_minus_ipopt_cost_percent": finite_statistics(gaps),
        "generator_lower_cost_count": sum(value < -1e-6 for value in gaps),
        "ipopt_lower_cost_count": sum(value > 1e-6 for value in gaps),
        "cost_tie_count": sum(abs(value) <= 1e-6 for value in gaps),
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def main() -> None:
    """Solve one saved data split with IPOPT and save its native solution."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="base SMPC dataset")
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="saved held-out data partition solved by IPOPT; test is the final benchmark",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "result directory; default is output/smpc_benchmark for test and "
            "output/smpc_validation for validation"
        ),
    )
    parser.add_argument(
        "--ipopt-path", type=Path, default=IPOPT_PATH,
        help="IPOPT executable in Anaconda py3.10",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="optional positive prefix of the selected split; default solves all 500",
    )
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="one-period ramp limit as a fraction of generator Pmax",
    )
    parser.add_argument(
        "--ipopt-tolerance", type=float, default=1e-8,
        help="positive IPOPT scaled optimality tolerance",
    )
    parser.add_argument(
        "--ipopt-constraint-tolerance", type=float, default=1e-8,
        help="positive IPOPT absolute constraint-violation tolerance",
    )
    parser.add_argument("--tee", action="store_true", help="stream IPOPT iterations")
    args = parser.parse_args()
    if (
        args.ramp_fraction <= 0 or args.ipopt_tolerance <= 0
        or args.ipopt_constraint_tolerance <= 0
    ):
        raise ValueError("invalid ramp fraction or tolerance")
    if not args.data.is_dir():
        raise FileNotFoundError(f"base dataset not found: {args.data}")
    if not args.ipopt_path.is_file():
        raise FileNotFoundError(f"IPOPT executable not found: {args.ipopt_path}")
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OUTPUT if args.split == "test"
            else DEFAULT_OUTPUT.parent / "smpc_validation"
        )

    current = np.load(args.data / "current_load.npy", mmap_mode="r")
    future = np.load(args.data / "future_load.npy", mmap_mode="r")
    instance_indices = load_dataset_split_indices(
        args.data, args.split, len(current), args.max_instances,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_indices = args.output_dir / f"{args.split}_indices.npy"
    if existing_indices.is_file():
        previous = np.load(existing_indices)
        if not np.array_equal(previous[:len(instance_indices)], instance_indices):
            raise ValueError(f"existing and current {args.split} indices differ")
    np.save(existing_indices, instance_indices)

    build_start = perf_counter()
    optimizer = SharedTrajectorySMPCAcopf(
        horizon=1 + future.shape[2],
        n_scenarios=future.shape[1],
        ramp_fraction=args.ramp_fraction,
        ipopt_path=args.ipopt_path,
        ipopt_tolerance=args.ipopt_tolerance,
        ipopt_constraint_tolerance=args.ipopt_constraint_tolerance,
    )
    model_build_seconds = perf_counter() - build_start
    n_selected = len(instance_indices)
    horizon = optimizer.horizon
    free_dim = len(optimizer.free_pg) + len(optimizer.voltage_buses)
    selected_u = np.full((n_selected, horizon, free_dim), np.nan, dtype="float32")
    selected_pg = np.full(
        (n_selected, optimizer.n_scenarios, horizon, optimizer.case.n_gen),
        np.nan, dtype="float32",
    )
    selected_qg = np.full_like(selected_pg, np.nan)
    selected_vm = np.full(
        (n_selected, optimizer.n_scenarios, horizon, optimizer.case.n_bus),
        np.nan, dtype="float32",
    )
    selected_va = np.full_like(selected_vm, np.nan)
    rows: list[dict] = []
    print(
        f"Shared-trajectory IPOPT benchmark: split={args.split}, n={n_selected}, "
        f"build={model_build_seconds:.3f}s, ipopt={args.ipopt_path}",
        flush=True,
    )

    for local_index, instance in enumerate(instance_indices):
        solved = optimizer.solve(
            np.asarray(current[instance]), np.asarray(future[instance]), tee=args.tee,
        )
        feasible = bool(solved["solved"])
        if feasible:
            selected_u[local_index] = solved["schedule"]
            for name, target in (
                ("pg", selected_pg), ("qg", selected_qg),
                ("vm", selected_vm), ("va", selected_va),
            ):
                target[local_index] = solved["states"][name]
        row = {
            "instance_index": int(instance),
            "solver_solved": int(solved["solved"]),
            "feasible": int(feasible),
            "termination": solved["termination"],
            "retries": solved["retries"],
            "ipopt_objective": solved.get("ipopt_objective", np.nan),
            "best_cost": solved.get("ipopt_objective", np.nan),
            "ipopt_max_constraint_violation": solved.get(
                "ipopt_max_constraint_violation", np.nan,
            ),
            "solve_seconds": solved["solve_seconds"],
            "total_seconds": solved["solve_seconds"],
        }
        rows.append(row)
        write_csv(args.output_dir / "ipopt_results.csv", rows)
        print(
            f"[{local_index + 1:4d}/{n_selected}] instance={instance} "
            f"solved={solved['solved']} feasible={feasible} "
            f"cost={row['best_cost']:.3f} total={row['total_seconds']:.3f}s "
            f"max_constr={row['ipopt_max_constraint_violation']:.2e} "
            f"term={solved['termination']}",
            flush=True,
        )

    np.savez_compressed(
        args.output_dir / "ipopt_solutions.npz",
        instance_indices=instance_indices,
        split=np.asarray(args.split),
        u=selected_u, pg=selected_pg, qg=selected_qg, vm=selected_vm, va=selected_va,
    )
    feasible_rows = [row for row in rows if row["feasible"]]
    summary = {
        "method": "shared-trajectory Pyomo/IPOPT",
        "data_split": args.split,
        "n_instances": n_selected,
        "solver_solved_instances": sum(row["solver_solved"] for row in rows),
        "feasible_instances": len(feasible_rows),
        "feasibility_rate": len(feasible_rows) / n_selected,
        "model_build_seconds": model_build_seconds,
        "solve_seconds": finite_statistics([row["solve_seconds"] for row in rows]),
        "total_seconds": finite_statistics([row["total_seconds"] for row in rows]),
        "best_cost": finite_statistics([row["best_cost"] for row in feasible_rows]),
        "ipopt_max_constraint_violation": finite_statistics([
            row["ipopt_max_constraint_violation"] for row in feasible_rows
        ]),
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "ipopt_tolerance": args.ipopt_tolerance,
        "ipopt_constraint_tolerance": args.ipopt_constraint_tolerance,
        "ipopt_path": str(args.ipopt_path.resolve()),
    }
    write_json(args.output_dir / "ipopt_summary.json", summary)
    comparison = compare_methods(args.output_dir, rows)
    if comparison is None:
        print("Generator results not found; comparison files were not created.")
    else:
        print(
            f"comparison: matched={comparison['n_matched_instances']} "
            f"both_feasible={comparison['both_feasible']}"
        )
    print(f"saved IPOPT benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
